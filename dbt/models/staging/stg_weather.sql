{{ config(materialized='view', tags=['staging']) }}

{#
    Archived weather forecasts from a named model.

    Two vintages again, and the strict one is the default. `valid_at` is the
    hour the forecast describes; `known_at` is when it may be used. With the
    zero lead currently configured they are equal, which means a prediction at
    time t may use weather valid at or before t, and may not use weather valid
    at the target interval.

    That is deliberately conservative -- see src/nemfc/weather.py. It costs
    little at a thirty-minute horizon because temperature and wind move slowly,
    and it means no weather feature can be attacked as forward-looking.
#}

select
    region              as region_id,
    valid_at,
    known_at,
    source,
    lead_seconds,
    temperature_2m      as temperature_c,
    apparent_temperature as apparent_temperature_c,
    wind_speed_10m      as wind_speed_10m,
    wind_speed_100m     as wind_speed_100m,
    cloud_cover         as cloud_cover_pct,
    shortwave_radiation as shortwave_radiation,
    ingested_at
from {{ source('raw', 'weather_forecast') }}
