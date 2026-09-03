{#
    Every feature, computed once, for both the training mart and the live mart.

    What is deliberately absent is the target. `mart_spike_features` joins it on
    and filters to rows whose thirty-minute outcome is complete and was unknown
    at prediction time; `mart_live_features` does neither, because the rows
    worth predicting on are precisely the ones whose outcome has not happened.

    That split was forced by Day 7. The training mart filters on
    `tgt.n_intervals = 6`, which is right for training and makes the table
    useless for serving: the freshest row it can offer is more than half an hour
    old. Splitting here rather than writing the features a second time is what
    keeps train and serve identical, which was the whole point of anchoring
    predictions to publication events on Day 4.

    ## When a prediction happens

    Not on a clock. Each row is anchored to the moment a dispatch interval's
    price became readable, and `prediction_at` is that moment. AEMO solves
    dispatch at the start of a five-minute interval and publishes about two
    minutes before the interval's own timestamp, so at any given clock time the
    price of the interval ending shortly ahead is already public. A target of
    "any spike in (t, t+30min]" therefore always contains an interval whose
    answer is already known -- `tests/test_no_leakage.py` demonstrates that the
    naive window leaks on every row.

    ## Gaps

    Rolling statistics carry their own contributing-row count and are nulled
    when the window is short, because a mean over four intervals reported as a
    mean over seven is exactly the silent error this project keeps finding. That
    guard is what makes the live path safe without the completeness filter: an
    incomplete current day produces nulls, not wrong numbers.

    Lags are exact self-joins on the timestamp rather than `lag()` over row
    order. Positional lags bridge a missing interval silently; a timestamp join
    returns null, which is what absence should look like.

    ## Why the forecast relation is an argument

    The first live build returned a null `benchmark_max_forecast_rrp` on every
    row, and the reason was subtle: the live models were views, but they joined
    to `int_p5min_forecast`, which is a table. The as-of lookup could only see
    pre-dispatch runs that existed at the last `dbt build`, and none of those
    forecast a window two hours later -- so AEMO's forecast, the model's
    second-largest source of gain, silently vanished at serving time.

    A materialised model anywhere in a serving path pins that path's freshness.
    Both relations are arguments now so the live chain can be views end to end.
#}

{% macro feature_rows(dispatch_relation, forecast_relation) %}

{%- set threshold = var('spike_threshold') -%}
{%- set horizon = 30 -%}

with base as (

    select
        d.*,

        -- Rolling statistics over time, not over row position. A RANGE frame
        -- measured in minutes shrinks when intervals are missing instead of
        -- silently reaching further back.
        max(d.rrp) over w30  as rolling_max_rrp_30m,
        avg(d.rrp) over w30  as rolling_mean_rrp_30m,
        count(*)   over w30  as rolling_n_30m,

        max(d.rrp) over w60  as rolling_max_rrp_60m,
        avg(d.rrp) over w60  as rolling_mean_rrp_60m,
        stddev_samp(d.rrp) over w60 as rolling_std_rrp_60m,
        count(*)   over w60  as rolling_n_60m,

        max(d.rrp) over w240 as rolling_max_rrp_240m,
        avg(d.rrp) over w240 as rolling_mean_rrp_240m,
        count(*)   over w240 as rolling_n_240m,

        sum(case when d.rrp > {{ threshold }} then 1 else 0 end)
            over w24h as spikes_trailing_24h,
        count(*) over w24h as rolling_n_24h

    from {{ dispatch_relation }} d
    window
        w30  as (order by d.settlement_end
                 range between interval '30 minutes'  preceding and current row),
        w60  as (order by d.settlement_end
                 range between interval '60 minutes'  preceding and current row),
        w240 as (order by d.settlement_end
                 range between interval '240 minutes' preceding and current row),
        w24h as (order by d.settlement_end
                 range between interval '24 hours'    preceding and current row)

)

select
    -- Identity ---------------------------------------------------------------
    b.known_at                     as prediction_at,
    b.settlement_end               as anchor_settlement_end,
    b.region_id,

    -- Price features ---------------------------------------------------------
    b.rrp                          as rrp_now,
    l1.rrp                         as rrp_lag_5m,
    l2.rrp                         as rrp_lag_10m,
    l3.rrp                         as rrp_lag_15m,
    l6.rrp                         as rrp_lag_30m,
    l12.rrp                        as rrp_lag_60m,

    case when b.rolling_n_30m  >= 7  then b.rolling_max_rrp_30m   end as rolling_max_rrp_30m,
    case when b.rolling_n_30m  >= 7  then b.rolling_mean_rrp_30m  end as rolling_mean_rrp_30m,
    case when b.rolling_n_60m  >= 13 then b.rolling_max_rrp_60m   end as rolling_max_rrp_60m,
    case when b.rolling_n_60m  >= 13 then b.rolling_mean_rrp_60m  end as rolling_mean_rrp_60m,
    case when b.rolling_n_60m  >= 13 then b.rolling_std_rrp_60m   end as rolling_std_rrp_60m,
    case when b.rolling_n_240m >= 49 then b.rolling_max_rrp_240m  end as rolling_max_rrp_240m,
    case when b.rolling_n_240m >= 49 then b.rolling_mean_rrp_240m end as rolling_mean_rrp_240m,
    case when b.rolling_n_24h  >= 288 then b.spikes_trailing_24h  end as spikes_trailing_24h,

    -- Demand and supply ------------------------------------------------------
    b.total_demand,
    b.total_demand - l6.total_demand   as demand_ramp_30m,
    b.total_demand - lw.total_demand   as demand_vs_same_time_last_week,
    b.available_generation,
    b.local_generation_balance,
    b.net_interchange,
    b.solar_forecast,
    b.wind_forecast,
    b.battery_max_available,

    -- Congestion -------------------------------------------------------------
    b.binding_interconnectors,
    b.max_abs_marginal_value,

    -- Price-cap regime -------------------------------------------------------
    b.apc_flag,
    b.market_suspended_flag,
    b.cumulative_price,

    -- AEMO's own forecast: feature and benchmark -----------------------------
    bench.benchmark_max_forecast_rrp,
    (bench.benchmark_max_forecast_rrp > {{ threshold }}) as benchmark_predicts_spike,
    bench.benchmark_run_at,
    bench.benchmark_intervals,

    -- Weather ----------------------------------------------------------------
    wx.temperature_c,
    wx.apparent_temperature_c,
    wx.wind_speed_100m,
    wx.cloud_cover_pct,

    -- Calendar ---------------------------------------------------------------
    extract(hour   from b.settlement_end at time zone 'Etc/GMT-10') as nem_hour,
    extract(dow    from b.settlement_end at time zone 'Etc/GMT-10') as nem_day_of_week,
    extract(month  from b.settlement_end at time zone 'Etc/GMT-10') as nem_month,
    -- The NEM trading day runs 04:00 to 04:00, so an overnight interval
    -- belongs to the previous trading day.
    ((b.settlement_end at time zone 'Etc/GMT-10') - interval '4 hours')::date as nem_trading_date,

    -- Provenance: the evidence for point-in-time correctness -----------------
    greatest(
        b.known_at,
        coalesce(bench.benchmark_known_at, b.known_at),
        coalesce(wx.known_at, b.known_at)
    ) as max_feature_known_at,
    true as day_is_complete,
    b.vintage_source

from base b

-- Exact-timestamp lags. Null when the interval is absent, which is correct.
left join {{ dispatch_relation }} l1
    on l1.settlement_end = b.settlement_end - interval '5 minutes'
left join {{ dispatch_relation }} l2
    on l2.settlement_end = b.settlement_end - interval '10 minutes'
left join {{ dispatch_relation }} l3
    on l3.settlement_end = b.settlement_end - interval '15 minutes'
left join {{ dispatch_relation }} l6
    on l6.settlement_end = b.settlement_end - interval '30 minutes'
left join {{ dispatch_relation }} l12
    on l12.settlement_end = b.settlement_end - interval '60 minutes'
left join {{ dispatch_relation }} lw
    on lw.settlement_end = b.settlement_end - interval '7 days'

-- AEMO's pre-dispatch, as of the prediction moment. Two steps rather than one
-- correlated subquery: find the latest run readable at prediction time, then
-- read that run's forecast for the target window.
left join lateral (
    select f.run_at
    from {{ forecast_relation }} f
    where f.known_at <= b.known_at
    order by f.known_at desc, f.run_at desc
    limit 1
) latest_run on true

left join lateral (
    select
        max(f.forecast_rrp) as benchmark_max_forecast_rrp,
        max(f.known_at)     as benchmark_known_at,
        max(f.run_at)       as benchmark_run_at,
        count(*)            as benchmark_intervals
    from {{ forecast_relation }} f
    where f.run_at = latest_run.run_at
      and f.interval_end >  b.settlement_end
      and f.interval_end <= b.settlement_end + interval '{{ horizon }} minutes'
) bench on true

-- Weather, as of the prediction moment.
left join lateral (
    select w.temperature_c, w.apparent_temperature_c,
           w.wind_speed_100m, w.cloud_cover_pct, w.known_at
    from {{ ref('stg_weather') }} w
    where w.region_id = b.region_id
      and w.known_at <= b.known_at
    order by w.known_at desc
    limit 1
) wx on true

{% endmacro %}
