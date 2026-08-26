{{ config(materialized='table', tags=['staging', 'quality']) }}

{#
    Which days are complete, derived rather than declared.

    `utc_date` is the UTC calendar date, deliberately not called a trading
    date: the NEM trading day runs 04:00 to 04:00, so the two are different
    things. Counting completeness by calendar date is fine because both feeds
    are bucketed the same way, but naming it `trading_date` would invite a
    wrong join the moment calendar features arrive.

    A hardcoded gap list is wrong the moment AEMO drops another interval, and
    nothing tells you. This computes it: for each region and day, how many
    dispatch intervals and pre-dispatch runs actually arrived.

    Both feeds are counted separately because **they do not fail together**.
    On 10 March 2026 DISPATCHIS lost 21 intervals while P5MIN lost 24. A day
    whose label is intact but whose benchmark is missing is still unusable, so
    `is_complete` requires both.

    The first and last calendar day of coverage are excluded from the check and
    are not gaps. An interval stamped 00:00 covers 23:55 to midnight of the
    evening before, so it ships in the previous day's rollup: the oldest day
    always shows 287 and the newest holds only that one. Flagging those two
    would mean the check fires on every run forever, and a check that always
    fires is one nobody reads.

    Downstream, the feature layer excludes whole incomplete days rather than
    the individual missing intervals. The intervals that did publish inside an
    outage have no usable history either side of them, so any lag or rolling
    feature computed on them reaches back into absent data.
#}

with dispatch as (

    select
        region_id,
        settlement_end::date            as utc_date,
        count(distinct settlement_end)  as dispatch_intervals,
        min(settlement_end)             as first_interval,
        max(settlement_end)             as last_interval
    from {{ ref('stg_dispatch_price') }}
    group by 1, 2

),

forecast as (

    select
        region_id,
        run_at::date          as utc_date,
        count(distinct run_at) as forecast_runs
    from {{ ref('stg_p5min_forecast') }}
    group by 1, 2

),

coverage as (

    {#
        Joint coverage: the window where BOTH feeds exist, not either one.

        The two archives do not start together. DISPATCHIS reaches back to
        12 Aug 2025 and P5MIN only to the 13th, so 12 August has a full 288
        dispatch intervals and 119 forecast runs. Excluding that day is right --
        a day without a benchmark is unusable -- but reporting it as a gap
        implies AEMO lost 169 runs, which it did not.

        Taking the later start and the earlier end makes an edge an edge.
    #}

    select
        d.region_id,
        greatest(d.first_day, f.first_day) as first_day,
        least(d.last_day, f.last_day)      as last_day
    from (
        select region_id, min(utc_date) as first_day, max(utc_date) as last_day
        from dispatch group by 1
    ) d
    join (
        select region_id, min(utc_date) as first_day, max(utc_date) as last_day
        from forecast group by 1
    ) f using (region_id)

)

select
    d.region_id,
    d.utc_date,
    d.dispatch_intervals,
    coalesce(f.forecast_runs, 0)                          as forecast_runs,
    {{ var('intervals_per_day') }} - d.dispatch_intervals as missing_intervals,
    {{ var('intervals_per_day') }} - coalesce(f.forecast_runs, 0) as missing_runs,
    d.first_interval,
    d.last_interval,
    -- At or outside the joint window. Not gaps, and excluded from the
    -- completeness judgement rather than reported as problems.
    --
    -- `<=` and `>=` rather than equality: the two archives start on different
    -- days, so there are dates that fall entirely OUTSIDE the joint window,
    -- not just on its boundary. An equality test left those flagged as if AEMO
    -- had lost them.
    (d.utc_date <= c.first_day or d.utc_date >= c.last_day) as is_coverage_edge,
    (
        d.utc_date > c.first_day
        and d.utc_date < c.last_day
        and d.dispatch_intervals = {{ var('intervals_per_day') }}
        and coalesce(f.forecast_runs, 0) = {{ var('intervals_per_day') }}
    ) as is_complete
from dispatch d
left join forecast f
    on f.region_id = d.region_id
   and f.utc_date = d.utc_date
join coverage c
    on c.region_id = d.region_id
