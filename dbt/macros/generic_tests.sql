{#
    Generic tests, written here rather than pulled from dbt_utils.

    Three of them, all short. Taking the package would add a dependency, a
    `dbt deps` step on every clean clone, and a network call, in exchange for
    about forty lines of SQL. It also means these tests can be read in the repo
    rather than looked up.

    A dbt test passes when it returns no rows, so each of these selects the
    rows that violate the condition.
#}


{% test unique_combination(model, combination_of_columns) %}

    {%- set columns = combination_of_columns | join(', ') -%}

    select
        {{ columns }},
        count(*) as duplicate_rows
    from {{ model }}
    group by {{ columns }}
    having count(*) > 1

{% endtest %}


{% test accepted_range(model, column_name, min_value=none, max_value=none) %}

    {#
        Nulls pass. A missing value is a not_null test's business, and folding
        the two together makes a failure ambiguous: you would not know whether
        the column was absent or out of bounds.
    #}

    {%- set conditions = [] -%}
    {%- if min_value is not none -%}
        {%- do conditions.append(column_name ~ ' < ' ~ min_value) -%}
    {%- endif -%}
    {%- if max_value is not none -%}
        {%- do conditions.append(column_name ~ ' > ' ~ max_value) -%}
    {%- endif -%}

    {%- if conditions | length == 0 -%}
        {{ exceptions.raise_compiler_error(
            "accepted_range on " ~ column_name ~ " needs min_value or max_value"
        ) }}
    {%- endif -%}

    select *
    from {{ model }}
    where {{ column_name }} is not null
      and ({{ conditions | join(' or ') }})

{% endtest %}


{% test expression_is_true(model, column_name, expression) %}

    {#
        `expression` is appended to the column, so `">= interval '0 seconds'"`
        reads as `column >= interval '0 seconds'`.
    #}

    select *
    from {{ model }}
    where {{ column_name }} is not null
      and not ({{ column_name }} {{ expression }})

{% endtest %}
