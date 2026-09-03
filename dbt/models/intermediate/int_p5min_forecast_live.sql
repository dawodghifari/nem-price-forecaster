{{
    config(
        materialized='table',
        tags=['intermediate', 'serving'],
        indexes=[
            {'columns': ['run_at', 'interval_end']},
            {'columns': ['known_at']},
        ],
    )
}}

{#
    AEMO's pre-dispatch for the serving path: a few hours of runs, typed and
    deduplicated by the same macro the staging model uses, but with the window
    restricted at the raw level.

    ## Why not simply select from `stg_p5min_forecast`

    Because that is what took three hours and forty-five minutes. The staging
    model deduplicates vintages with `row_number() over (...)`, which is an
    optimisation fence: a predicate applied to its output cannot be pushed below
    the window, so every query through it ranks all 6.59 million raw rows first.
    `where known_at >= now() - interval '6 hours'` reads like a cheap filter and
    removes no work at all.

    Sharing the macro rather than the model means the typing and the vintage
    rule stay in one place -- a serving path whose parsing drifts from the
    training path is the same class of bug as one whose features drift -- while
    the filter lands on the indexed `_known_at` column before the dedupe runs.

    ## Why a table

    The benchmark as-of lookup is a lateral join, evaluated once per anchor row.
    A view carries no index, so each evaluation would rescan. This is the
    conclusion Day 4 reached for the training model, and dropping that table
    dropped its indexes with it.

    The cost is that the benchmark is only as fresh as the last
    `make refresh-serving`. That belongs on the poller's cadence, and `/health`
    reports `benchmark_present` so a missed refresh is visible rather than a
    quietly worse prediction.
#}

{{ p5min_forecast(since=var('live_forecast_hours', 6) ~ ' hours') }}
