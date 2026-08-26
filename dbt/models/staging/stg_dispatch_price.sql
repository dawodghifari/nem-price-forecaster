{{
    config(
        materialized='view',
        tags=['staging', 'label'],
    )
}}

{#
    The label: the price each dispatch interval settled at.

    Two filters here, and only one of them does anything today.

    **Intervention.** When AEMO intervenes in the market it solves dispatch
    twice -- once for the physical outcome and once for pricing -- and both
    solutions appear for the same interval, distinguished by INTERVENTION. The
    pricing run is what settles, so that is the one kept.

    Across all 375 days loaded on 26 Aug 2026, INTERVENTION was '0' for every
    one of 545,620 rows. So this filter is currently a no-op and, more to the
    point, **is untested against real data**. It is written anyway, because the
    first day AEMO does intervene a model without it silently doubles its rows
    for that interval. `_models.yml` carries a test that fails the moment a
    value other than '0' appears, which is when the choice above needs
    confirming rather than assuming.

    **Vintage.** AEMO can republish an interval. Raw is append-only, so a
    revision arrives as extra rows rather than an update, and the latest
    `_known_at` wins. No duplicate keys existed as at 26 Aug 2026, so this is
    also currently a no-op -- but it is the difference between a revision being
    handled and a revision doubling a row.
#}

with source as (

    select *
    from {{ source('raw', 'dispatch_price_v5') }}
    where "INTERVENTION" = '0'

),

typed as (

    select
        {{ nem_to_utc('"SETTLEMENTDATE"') }}      as settlement_end,
        {{ interval_start('"SETTLEMENTDATE"') }}  as settlement_start,
        "REGIONID"                                as region_id,
        "RUNNO"                                   as run_no,
        "INTERVENTION"                            as intervention,
        {{ to_number('"RRP"') }}                  as rrp,
        {{ to_number('"ROP"') }}                  as rop,
        {{ to_number('"EEP"') }}                  as eep,
        "APCFLAG"                                 as apc_flag,
        "MARKETSUSPENDEDFLAG"                     as market_suspended_flag,
        "PRICE_STATUS"                            as price_status,
        -- Cumulative price drives the administered price cap: once it breaches
        -- a threshold AEMO caps prices, which is a regime change in the label
        -- rather than an ordinary market outcome.
        {{ to_number('"CUMUL_PRE_AP_ENERGY_PRICE"') }} as cumulative_price,
        _known_at                                 as known_at,
        _vintage_source                           as vintage_source,
        _ingested_at                              as ingested_at
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
    intervention,
    rrp,
    rop,
    eep,
    apc_flag,
    market_suspended_flag,
    price_status,
    cumulative_price,
    known_at,
    vintage_source,
    ingested_at
from deduplicated
where vintage_rank = 1
