{{ config(materialized='view', tags=['staging']) }}

{#
    Regional demand and supply per dispatch interval.

    `available_generation` minus `total_demand` is the headroom that decides
    whether a spike is possible at all, and the semi-scheduled columns split
    that between wind and solar, which behave differently: solar falls away
    predictably at sunset, wind does not.
#}

with source as (

    select *
    from {{ source('raw', 'dispatch_regionsum_v9') }}
    where "INTERVENTION" = '0'

),

typed as (

    select
        {{ nem_to_utc('"SETTLEMENTDATE"') }}     as settlement_end,
        {{ interval_start('"SETTLEMENTDATE"') }} as settlement_start,
        "REGIONID"                               as region_id,
        "RUNNO"                                  as run_no,
        {{ to_number('"TOTALDEMAND"') }}            as total_demand,
        {{ to_number('"AVAILABLEGENERATION"') }}    as available_generation,
        {{ to_number('"AVAILABLELOAD"') }}          as available_load,
        {{ to_number('"DEMANDFORECAST"') }}         as demand_forecast,
        {{ to_number('"DISPATCHABLEGENERATION"') }} as dispatchable_generation,
        {{ to_number('"NETINTERCHANGE"') }}         as net_interchange,
        {{ to_number('"EXCESSGENERATION"') }}       as excess_generation,
        {{ to_number('"UIGF"') }}                   as semi_scheduled_forecast,
        {{ to_number('"SS_SOLAR_UIGF"') }}          as solar_forecast,
        {{ to_number('"SS_WIND_UIGF"') }}           as wind_forecast,
        {{ to_number('"SS_SOLAR_CLEAREDMW"') }}     as solar_cleared,
        {{ to_number('"SS_WIND_CLEAREDMW"') }}      as wind_cleared,
        {{ to_number('"SS_SOLAR_AVAILABILITY"') }}  as solar_available,
        {{ to_number('"SS_WIND_AVAILABILITY"') }}   as wind_available,
        {{ to_number('"TOTALINTERMITTENTGENERATION"') }} as intermittent_generation,
        {{ to_number('"DEMAND_AND_NONSCHEDGEN"') }} as demand_and_nonsched,
        -- Bidirectional units: grid-scale batteries. Their available range is
        -- the market's fastest response to a price spike, so how much of it is
        -- left matters.
        {{ to_number('"BDU_MIN_AVAIL"') }}          as battery_min_available,
        {{ to_number('"BDU_MAX_AVAIL"') }}          as battery_max_available,
        _known_at                                as known_at,
        _vintage_source                          as vintage_source,
        _ingested_at                             as ingested_at
    from source

),

deduplicated as (

    select
        *,
        row_number() over (
            partition by settlement_end, region_id
            order by known_at desc, ingested_at desc
        ) as vintage_rank
    from typed

)

select
    settlement_end,
    settlement_start,
    region_id,
    run_no,
    total_demand,
    available_generation,
    available_load,
    demand_forecast,
    dispatchable_generation,
    net_interchange,
    excess_generation,
    semi_scheduled_forecast,
    solar_forecast,
    wind_forecast,
    solar_cleared,
    wind_cleared,
    solar_available,
    wind_available,
    intermittent_generation,
    demand_and_nonsched,
    battery_min_available,
    battery_max_available,
    -- Local generation minus local demand. Deliberately not called "headroom":
    -- AVAILABLEGENERATION counts only generation inside the region, and a
    -- region can meet demand with imports, so this goes negative whenever a
    -- region leans on its neighbours. 114 intervals do exactly that.
    --
    -- Real supply headroom has to add import capacity, which means joining the
    -- interconnector model. That belongs in the feature layer on Day 4, not
    -- here, because it is a derived quantity rather than a renamed column.
    available_generation - total_demand as local_generation_balance,
    known_at,
    vintage_source,
    ingested_at
from deduplicated
where vintage_rank = 1
