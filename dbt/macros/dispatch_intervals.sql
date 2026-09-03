{#
    One dispatch-interval definition, used by both the training path and the
    serving path.

    They differ in exactly one place. Training drops days AEMO did not fully
    publish, so no rolling window can span an outage. Serving cannot: the
    current day is always incomplete, so applying that filter would leave the
    freshest available prediction more than a day stale -- which is not a
    prediction.

    Writing the SQL twice would let the two drift, and a serving path whose
    features quietly diverge from the training path is the classic way a model
    that backtested well fails in production. So it is written once, and the
    one difference is an argument.
#}

{% macro dispatch_intervals(require_complete_days=true, since_days=none) %}

{%- set region = var('region') -%}

with
{% if require_complete_days -%}
complete_days as (
    select utc_date
    from {{ ref('stg_interval_completeness') }}
    where region_id = '{{ region }}'
      and is_complete
),
{%- endif %}
congestion as (
    select
        settlement_end,
        count(*) filter (where is_binding)        as binding_interconnectors,
        max(abs(marginal_value))                  as max_abs_marginal_value,
        sum(abs(mw_flow))                         as total_absolute_flow
    from {{ ref('stg_dispatch_interconnector') }}
    {%- if since_days %}
    where settlement_end >= now() - interval '{{ since_days }} days'
    {%- endif %}
    group by 1
)

select
    p.settlement_end,
    p.settlement_start,
    p.region_id,
    p.rrp,
    p.apc_flag,
    p.market_suspended_flag,
    p.cumulative_price,
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
    coalesce(c.binding_interconnectors, 0) as binding_interconnectors,
    c.max_abs_marginal_value,
    c.total_absolute_flow,
    greatest(p.known_at, r.known_at) as known_at,
    p.vintage_source
from {{ ref('stg_dispatch_price') }} p
join {{ ref('stg_dispatch_regionsum') }} r
    on r.settlement_end = p.settlement_end
   and r.region_id = p.region_id
left join congestion c
    on c.settlement_end = p.settlement_end
{% if require_complete_days -%}
join complete_days d
    on d.utc_date = p.settlement_end::date
{%- endif %}
where p.region_id = '{{ region }}'
{%- if since_days %}
  and p.settlement_end >= now() - interval '{{ since_days }} days'
{%- endif %}

{% endmacro %}
