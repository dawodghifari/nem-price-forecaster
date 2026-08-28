{{
    config(
        materialized='table',
        tags=['intermediate', 'benchmark'],
        indexes=[
            {'columns': ['known_at']},
            {'columns': ['run_at', 'interval_end']},
            {'columns': ['interval_end', 'known_at']},
        ],
    )
}}

{#
    AEMO's pre-dispatch for the region under study, materialised so the as-of
    join has something indexed to work against.

    Three indexes, one per lookup the feature table performs:

    - `known_at` answers "the latest run readable at time t", as a descending
      index scan with LIMIT 1.
    - `(run_at, interval_end)` answers "that run's forecast for the target
      window". The feature table asks this about 107,000 times, and without
      run_at leading the index each one degrades into a scan of the whole
      table. It was missing on the first pass, which would have made the build
      look like it had hung rather than fail.
    - `(interval_end, known_at)` serves the reverse lookup, used by the
      independent leakage check.

    Filtered to the region here rather than downstream: 6.5 million rows across
    five regions becomes about 1.3 million, and the whole point of this model
    is to make the join cheap.
#}

{%- set region = var('region') -%}

select
    run_at,
    interval_end,
    region_id,
    known_at,
    forecast_rrp,
    forecast_demand,
    forecast_available_generation,
    forecast_net_interchange,
    forecast_solar,
    forecast_wind,
    forecast_battery_storage,
    publish_offset
from {{ ref('stg_p5min_forecast') }}
where region_id = '{{ region }}'
