{#
    No feature may come from a source published after the prediction was made.

    Equality passes: the anchor interval's own publication defines the
    prediction moment, so a source readable at exactly that instant was
    readable.

    This mirrors nemfc.leakage.check_features_not_from_the_future. Having it in
    both places is deliberate -- the SQL fails the build, the Python test proves
    the check fires on a planted leak, and neither alone is sufficient.
#}

{% test features_not_from_the_future(model) %}

select
    prediction_at,
    max_feature_known_at,
    max_feature_known_at - prediction_at as leaked_by
from {{ model }}
where max_feature_known_at > prediction_at

{% endtest %}
