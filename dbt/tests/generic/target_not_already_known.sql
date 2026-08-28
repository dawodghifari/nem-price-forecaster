{#
    Every interval being predicted must still have been unpublished at the
    moment of prediction.

    Strict inequality: an interval readable at exactly the prediction moment is
    already known, and predicting a known value is not forecasting.

    This is the half PLAN §4.1 does not describe. A leaky target passes every
    feature-side check -- nothing arrived from the future, the label simply was
    not a prediction. AEMO publishes a dispatch price about two minutes before
    the interval it belongs to ends, so a target window defined by the clock
    rather than by publication contains an already-known interval on every row.
#}

{% test target_not_already_known(model) %}

select
    prediction_at,
    min_target_known_at,
    prediction_at - min_target_known_at as known_in_advance_by
from {{ model }}
where min_target_known_at <= prediction_at

{% endtest %}
