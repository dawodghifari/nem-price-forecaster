{{
    config(
        materialized='table',
        tags=['marts', 'features'],
        indexes=[
            {'columns': ['prediction_at'], 'unique': True},
            {'columns': ['anchor_settlement_end']},
        ],
    )
}}

{#
    The training table: every feature from `int_feature_rows`, plus the answer.

    ## What is being predicted

    The six intervals after the anchor: a genuine thirty minutes ahead of the
    last price anyone can see. All six must exist, so a row is dropped rather
    than shortened when the market data does not cover the whole window.

    That filter is also why this table cannot serve predictions, and why the
    features moved into `int_feature_rows` on Day 7. Requiring a complete
    outcome excludes exactly the rows a live forecast is about. The serving
    table is `mart_live_features`, over the same feature definition.
#}

{%- set threshold = var('spike_threshold') -%}
{%- set horizon = 30 -%}
{%- set target_intervals = (horizon / 5) | int -%}

select
    f.*,

    tgt.target_max_rrp,
    (tgt.target_max_rrp > {{ threshold }}) as is_spike,
    f.anchor_settlement_end + interval '5 minutes'             as target_window_start,
    f.anchor_settlement_end + interval '{{ horizon }} minutes' as target_window_end,
    tgt.min_target_known_at

from {{ ref('int_feature_rows') }} f

-- The target: the six intervals after the anchor.
left join lateral (
    select
        max(t.rrp)      as target_max_rrp,
        min(t.known_at) as min_target_known_at,
        count(*)        as n_intervals
    from {{ ref('int_dispatch_intervals') }} t
    where t.settlement_end >  f.anchor_settlement_end
      and t.settlement_end <= f.anchor_settlement_end + interval '{{ horizon }} minutes'
) tgt on true

-- A partial target is not a shortened label, it is no label.
where tgt.n_intervals = {{ target_intervals }}

  -- And the target must genuinely have been unknown at prediction time.
  --
  -- This looks redundant given how the window is built, and it is not. Four
  -- rows in 108,276 violated it on the first build, all at the two seams where
  -- backfilled data meets polled data: 16 Aug 14:02 UTC and 23 Aug 10:47 UTC.
  --
  -- The two routes measure vintage differently. A live row carries its
  -- observed NEMWeb listing time, roughly two minutes BEFORE its interval
  -- ends. An archive row has no listing time -- daily rollups do not preserve
  -- one -- so it gets generation time plus the observed publication lag, about
  -- seven minutes AFTER. That conservatism is right for features, but it
  -- pushes an archive-sourced anchor's prediction time nine minutes later than
  -- a live one, and the live intervals that follow it then appear already
  -- published.
  --
  -- The giveaway is precision: the offending targets are stamped on whole
  -- minutes, which is directory-listing granularity, while their anchors carry
  -- seconds from a file header.
  --
  -- Neither timestamp is wrong; they are simply not comparable at the seam. So
  -- these rows are dropped rather than adjusted. Inventing a reconciliation
  -- would mean asserting a publication order nobody observed, and four rows in
  -- a hundred thousand is not worth a fiction.
  and tgt.min_target_known_at > f.prediction_at
