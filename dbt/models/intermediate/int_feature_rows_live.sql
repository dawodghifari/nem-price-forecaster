{{
    config(
        materialized='table',
        tags=['intermediate', 'serving'],
        indexes=[{'columns': ['prediction_at'], 'unique': True}],
    )
}}

{#
    The same features as `int_feature_rows`, computed over the unfiltered recent
    window. Identical SQL -- the only difference is which dispatch-interval
    relation it reads.

    A view for the same reason as its source: materialising it would pin the
    API's freshness to the last dbt build rather than to the poller.
#}

{{ feature_rows(ref('int_dispatch_intervals_live'), ref('int_p5min_forecast_live')) }}
