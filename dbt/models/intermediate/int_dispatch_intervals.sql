{{
    config(
        materialized='table',
        tags=['intermediate'],
        indexes=[
            {'columns': ['settlement_end'], 'unique': True},
            {'columns': ['known_at']},
        ],
    )
}}

{#
    One row per dispatch interval for the region under study: price, demand,
    supply, congestion, and when it all became readable.

    Materialised and indexed because everything downstream joins against it by
    timestamp, and the feature table does that roughly a hundred thousand
    times.

    **Incomplete days are dropped here**, not later. Any interval from a day
    AEMO did not fully publish is removed before a single lag or rolling
    statistic is computed, because a window spanning an outage returns a number
    rather than a gap and nothing downstream can tell the difference.

    Interconnector data is aggregated market-wide rather than attributed to
    this region. Direction conventions differ per interconnector and getting
    one backwards is a silent error; the region's own net position is already
    in NETINTERCHANGE, so what the interconnector table adds here is congestion
    -- how many limits are binding at all.
#}

{%- set region = var('region') -%}

with complete_days as (

    select utc_date
    from {{ ref('stg_interval_completeness') }}
    where region_id = '{{ region }}'
      and is_complete

),

congestion as (

    select
        settlement_end,
        count(*) filter (where is_binding)        as binding_interconnectors,
        max(abs(marginal_value))                  as max_abs_marginal_value,
        sum(abs(mw_flow))                         as total_absolute_flow
    from {{ ref('stg_dispatch_interconnector') }}
    group by 1

)

select
    p.settlement_end,
    p.settlement_start,
    p.region_id,

    -- Label source and price context
    p.rrp,
    p.apc_flag,
    p.market_suspended_flag,
    p.cumulative_price,

    -- Demand and supply
    r.total_demand,
    r.available_generation,
    r.demand_forecast,
    r.net_interchange,
    r.local_generation_balance,
    r.solar_forecast,
    r.wind_forecast,
    r.solar_available,
    r.wind_available,
    r.battery_min_available,
    r.battery_max_available,

    -- Congestion, market-wide
    coalesce(c.binding_interconnectors, 0) as binding_interconnectors,
    c.max_abs_marginal_value,
    c.total_absolute_flow,

    -- The vintage. Both feeds must be readable, so the later of the two.
    greatest(p.known_at, r.known_at) as known_at,
    p.vintage_source

from {{ ref('stg_dispatch_price') }} p
join {{ ref('stg_dispatch_regionsum') }} r
    on r.settlement_end = p.settlement_end
   and r.region_id = p.region_id
left join congestion c
    on c.settlement_end = p.settlement_end
join complete_days d
    on d.utc_date = p.settlement_end::date
where p.region_id = '{{ region }}'
