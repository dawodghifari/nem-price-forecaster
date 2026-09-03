{#
    AEMO's pre-dispatch, typed and vintage-deduplicated. One definition, two
    callers, and one argument that decides whether it costs milliseconds or
    minutes.

    ## Why `since` exists

    The dedupe uses `row_number() over (partition by run_at, interval_end,
    region_id)`, and a window function is an optimisation fence. Postgres has to
    rank every row before it can apply `where vintage_rank = 1`, and it cannot
    push a caller's predicate below that. So filtering the *result* -- `where
    known_at >= now() - interval '6 hours'` -- still ranks all 6.59 million raw
    rows first. The filter looks cheap and is free only in the sense that it
    changes nothing.

    That cost a serving rebuild three hours and forty-five minutes, then hung a
    second attempt after the model was changed from a view to a table, because
    building the table pays the same fence. `_known_at` is indexed on the raw
    table; the fix is to filter there, before the window, so the dedupe ranks
    hundreds of rows instead of millions.

    `since` is therefore not a convenience. It is the difference between a
    serving refresh that takes a second and one that does not finish.
#}

{% macro p5min_forecast(since=none) %}

with source as (
    select *
    from {{ source('raw', 'p5min_regionsolution_v10') }}
    where "INTERVENTION" = '0'
    {%- if since %}
      -- Before the window function, on the indexed raw column. Filtering the
      -- result instead would rank the whole table first.
      and _known_at >= now() - interval '{{ since }}'
    {%- endif %}
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
    interval_end - known_at as forecast_horizon,
    known_at - run_at as publish_offset,
    vintage_source,
    ingested_at
from deduplicated
where vintage_rank = 1

{% endmacro %}
