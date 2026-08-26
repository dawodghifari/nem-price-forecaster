{{ config(materialized='view', tags=['staging', 'benchmark']) }}

{#
    AEMO's own five-minute pre-dispatch. Both a model feature and the incumbent
    the project is measured against, which makes it the single most dangerous
    table in the warehouse to get wrong.

    **Three timestamps, and choosing the wrong one is the whole leak.**

    - `run_at`      RUN_DATETIME: when AEMO made this forecast.
    - `interval_end` INTERVAL_DATETIME: which future interval it forecasts.
    - `known_at`    when the file carrying it actually appeared on NEMWeb.

    `run_at` is not when the forecast became readable. Measured 18 Aug 2026
    across 576 files: publication ran 7.5 minutes behind the run at the median
    and 23.4 minutes at the worst. A point-in-time join keyed on `run_at` would
    hand the model AEMO's forecast up to twenty-three minutes before anyone
    could have read it -- on the one comparison the project exists to make.

    So the feature layer joins on `known_at`, never `run_at`. `run_at` is kept
    because the difference between them is itself informative: a long
    publication lag is a signal that AEMO's systems were struggling, which
    tends to coincide with the conditions that produce spikes.

    `forecast_horizon` is measured from `known_at` for the same reason.
#}

with source as (

    select *
    from {{ source('raw', 'p5min_regionsolution_v10') }}
    where "INTERVENTION" = '0'

),

typed as (

    select
        {{ nem_to_utc('"RUN_DATETIME"') }}      as run_at,
        {{ nem_to_utc('"INTERVAL_DATETIME"') }} as interval_end,
        "REGIONID"                              as region_id,
        {{ to_number('"RRP"') }}                as forecast_rrp,
        {{ to_number('"ROP"') }}                as forecast_rop,
        {{ to_number('"TOTALDEMAND"') }}        as forecast_demand,
        {{ to_number('"DEMANDFORECAST"') }}     as demand_forecast_adjustment,
        {{ to_number('"AVAILABLEGENERATION"') }} as forecast_available_generation,
        {{ to_number('"AVAILABLELOAD"') }}      as forecast_available_load,
        {{ to_number('"NETINTERCHANGE"') }}     as forecast_net_interchange,
        {{ to_number('"UIGF"') }}               as forecast_semi_scheduled,
        {{ to_number('"SS_SOLAR_UIGF"') }}      as forecast_solar,
        {{ to_number('"SS_WIND_UIGF"') }}       as forecast_wind,
        {{ to_number('"BDU_ENERGY_STORAGE"') }} as forecast_battery_storage,
        _known_at                               as known_at,
        _vintage_source                         as vintage_source,
        _ingested_at                            as ingested_at
    from source

),

deduplicated as (

    select
        *,
        row_number() over (
            partition by run_at, interval_end, region_id
            order by known_at desc, ingested_at desc
        ) as vintage_rank
    from typed

)

select
    run_at,
    interval_end,
    region_id,
    forecast_rrp,
    forecast_rop,
    forecast_demand,
    demand_forecast_adjustment,
    forecast_available_generation,
    forecast_available_load,
    forecast_net_interchange,
    forecast_semi_scheduled,
    forecast_solar,
    forecast_wind,
    forecast_battery_storage,
    known_at,
    -- How far ahead this forecast reaches from the moment it became readable.
    -- Negative means the interval had already settled by publication, which
    -- happens on a slow run and makes that row useless as a forecast.
    interval_end - known_at as forecast_horizon,
    -- How long after the run's *nominal* time it became readable.
    --
    -- Routinely negative, which is not an error. RUN_DATETIME labels the
    -- interval the run targets, and AEMO generates the file roughly four and a
    -- half minutes before that label: a run labelled 00:50 was generated at
    -- 00:45:33 and could appear on NEMWeb at 00:48, two minutes before its own
    -- name. Measured across the archive, 7,260 rows are negative.
    --
    -- The invariant that does hold -- a file is never readable before it was
    -- generated -- is enforced upstream, where known_at is the later of the
    -- generation and listing times. It needs no test here because it cannot be
    -- violated by construction.
    known_at - run_at as publish_offset,
    vintage_source,
    ingested_at
from deduplicated
where vintage_rank = 1
