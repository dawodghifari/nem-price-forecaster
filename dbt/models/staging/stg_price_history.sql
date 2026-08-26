{{ config(materialized='view', tags=['staging', 'baseline']) }}

{#
    AEMO's separately published price and demand series, 2015 onward.

    Feeds the climatology baseline, the spike-threshold sensitivity check and
    the regime comparison in PLAN 2.2. **Never the feature table** -- it carries
    only price and demand, and it is a settled series with no vintage, so
    joining it to features would be joining data that was not knowable at
    prediction time.

    `spacing_minutes` is recorded per row because it changes: half-hourly
    before five-minute settlement in October 2021, five-minute after. A
    climatology estimated across the change without accounting for it would
    weight the recent years six times too heavily.
#}

select
    region              as region_id,
    settlement_end,
    settlement_end - make_interval(mins => spacing_minutes) as settlement_start,
    spacing_minutes,
    total_demand,
    rrp,
    period_type,
    (rrp > {{ var('spike_threshold') }}) as is_spike,
    source_url,
    ingested_at
from {{ source('raw', 'aggregated_price_demand') }}
