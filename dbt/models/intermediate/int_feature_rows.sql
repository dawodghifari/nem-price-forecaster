{{
    config(
        materialized='table',
        tags=['intermediate'],
        indexes=[
            {'columns': ['prediction_at'], 'unique': True},
            {'columns': ['anchor_settlement_end']},
        ],
    )
}}

{#
    Every feature, no target. `mart_spike_features` adds the target and its
    filters; `mart_live_features` adds neither. Definition in
    `macros/feature_rows.sql`, written once so the two cannot drift.
#}

{{ feature_rows(ref('int_dispatch_intervals'), ref('int_p5min_forecast')) }}
