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

{{ p5min_forecast() }}
