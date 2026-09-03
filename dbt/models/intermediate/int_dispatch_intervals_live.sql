{{
    config(
        materialized='table',
        tags=['intermediate', 'serving'],
        indexes=[{'columns': ['settlement_end'], 'unique': True}],
    )
}}

{#
    Serving-path dispatch intervals: no completeness filter, a few days deep,
    materialised and indexed.

    ## No completeness filter

    The current day is always incomplete -- it is still happening -- so the
    training filter would leave the freshest available prediction more than a
    day stale. What makes dropping it safe is that the rolling statistics
    downstream carry their contributing-row counts and null themselves when a
    window is short, so a genuine gap produces a null rather than a wrong
    number.

    ## A table, after trying a view and measuring it

    The first version was a view, on the reasoning that a serving path must be
    view-fresh or it is not live. `EXPLAIN` put the live mart at a total cost of
    220 billion and showed why that reasoning was wrong.

    The feature macro references this relation seven times -- once as the base
    and once for each lag (5m, 10m, 15m, 30m, 60m, 7d). As a view, every
    reference re-derives the entire staging pipeline: a Hash Join across two raw
    tables, each carrying its own `row_number()` vintage dedupe. Seven full
    recomputations to produce one row. The plan showed all seven, side by side.

    Materialising collapses that to one computation and gives the lag joins an
    index to hit. Freshness comes from running `make refresh-serving` often --
    seconds, because the window is days rather than thirteen months -- rather
    than from forcing the query to rebuild everything on each request.

    That is the same conclusion Day 4 reached for the training path. Reaching
    for views here was an over-correction after discovering that a stale table
    had quietly served a day-old prediction; the fix for staleness is refreshing
    often, not refusing to materialise.
#}

{{ dispatch_intervals(require_complete_days=false, since_days=var('live_window_days', 10)) }}
