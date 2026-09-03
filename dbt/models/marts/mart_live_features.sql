{{
    config(
        materialized='view',
        tags=['marts', 'serving'],
    )
}}

{#
    What the API scores. The same features the model was trained on, over rows
    whose outcome has not happened yet.

    A view rather than a table, deliberately. The underlying
    `int_feature_rows_live` covers ten days and rebuilds in seconds; a
    materialised copy would add a staleness window between the poller writing a
    new interval and the API being able to see it, which is the opposite of what
    a serving path wants.

    Every row here is a prediction nobody could grade yet -- that is the point.
    `target_max_rrp` and `is_spike` are absent by construction rather than null,
    so nothing downstream can mistake an ungraded row for a negative one.

    The freshest row is roughly two minutes old: AEMO publishes a dispatch price
    shortly before the interval it belongs to ends, and `prediction_at` is that
    publication moment.
#}

select
    f.*,
    f.anchor_settlement_end + interval '5 minutes'  as target_window_start,
    f.anchor_settlement_end + interval '30 minutes' as target_window_end,

    -- How stale the row is, computed here so the API reports it rather than
    -- deriving it and getting the timezone wrong.
    now() - f.prediction_at as age

from {{ ref('int_feature_rows_live') }} f
order by f.prediction_at desc
