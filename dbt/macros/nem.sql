{#
    Time handling for the NEM, in one place.

    Every timestamp AEMO publishes is naive local time in Australian Eastern
    Standard Time -- UTC+10 all year, no daylight saving. Postgres expresses
    that as 'Etc/GMT-10', where the POSIX sign convention is inverted: GMT-10
    means ten hours *ahead* of UTC. Writing 'Etc/GMT+10' would silently shift
    every row twenty hours, and nothing downstream would look wrong.
#}

{% macro nem_to_utc(column) %}
    (({{ column }})::timestamp AT TIME ZONE 'Etc/GMT-10')
{% endmacro %}


{#
    A dispatch interval is stamped with its END. Confirmed 18 Aug 2026 against
    a live file: a row stamped 00:50:00 came from a solve whose header was
    written at 00:45:13. So the interval it describes began five minutes before
    its label.
#}
{% macro interval_start(column) %}
    ({{ nem_to_utc(column) }} - interval '5 minutes')
{% endmacro %}


{#
    AEMO writes numbers as text and occasionally leaves a field empty. A plain
    cast raises on the empty string, which would fail a build for a blank
    optional column rather than for anything meaningful.
#}
{% macro to_number(column) %}
    nullif(trim({{ column }}), '')::double precision
{% endmacro %}
