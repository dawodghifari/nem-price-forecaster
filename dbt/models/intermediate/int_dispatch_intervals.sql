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
    Training-path dispatch intervals: days AEMO did not fully publish are
    removed here, before any window function runs, so no rolling statistic can
    span an outage.

    The SQL lives in `macros/dispatch_intervals.sql` because the serving path
    needs the same joins without that filter. See the macro for why.
#}

{{ dispatch_intervals(require_complete_days=true) }}
