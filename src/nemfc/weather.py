"""Open-Meteo historical forecast archive.

Temperature drives demand and wind drives supply, so weather belongs in the
feature set. The trap, and the reason PLAN §4.1 names it, is that there are two
Open-Meteo products and only one of them is legitimate here:

- **Historical Forecast API** (used here) archives what past forecast runs
  actually said.
- **Historical Weather API** is ERA5 reanalysis — observed conditions
  reconstructed afterwards. Observed temperature at 14:30 was not knowable at
  14:00, so joining it is a direct leak.

**Choosing the right endpoint is not sufficient, which the plan did not say.**
Measured 18 Aug 2026: with no model named, the *forecast* endpoint returns ERA5
reanalysis, matching the reanalysis endpoint to the last decimal place over a
settled window. Nothing in the response indicates this. Name a model and it
returns real forecast output — see `DEFAULT_MODEL` for the figures.

So the leak §4.1 warned about was real and sat one level deeper than described.
A reader who followed the plan exactly, used the historical forecast API as
instructed, and never passed `models=`, would have trained on observations and
had no way of knowing.

## The vintage question, and why the default is strict

The archive returns, for each valid hour, a value from a model run. Which run
is not stated in the response, and model runs are issued every few hours, so a
value valid at 14:00 may have come from a run initialised at 12:00 — after a
prediction made at 11:00 would have been served.

Rather than guess, the default rule treats a value valid at time `V` as
knowable only from `V` itself. A prediction at `t` may then use weather valid
at or before `t`, and may not use weather valid at the target interval.

**This costs almost nothing at a thirty-minute horizon.** Temperature and wind
speed barely move in thirty minutes, so the last known conditions carry nearly
all the signal the target-interval forecast would have. Paying a small amount
of predictive power for a claim that cannot be attacked is the right trade on
the one project whose headline is a comparison against AEMO.

`scripts/probe_weather.py` establishes empirically what lead the archive
represents. If it turns out runs are old enough to be safe, `lead` can be
loosened, and the leakage test will keep the change honest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd
from sqlalchemy import Engine, text

log = logging.getLogger(__name__)

BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Naming the model is not optional. Measured 18 Aug 2026 over 26-28 July, mean
# absolute difference against ERA5 reanalysis for the same 72 hours:
#
#     unspecified     0.000 °C     <- reanalysis, not a forecast
#     ecmwf_ifs025    0.586 °C
#     icon_seamless   0.954 °C
#     gfs_seamless    1.018 °C
#
# The named models disagree with each other by up to 1.519 °C, which is what
# distinct numerical weather models do. Left unspecified, the endpoint returns
# reanalysis exactly, and says nothing about it in the response.
#
# ECMWF IFS is chosen because it is the strongest global model. That it also
# sits closest to ERA5 is shared lineage rather than leakage: ERA5 is ECMWF's
# reanalysis and IFS is ECMWF's forecast model. The evidence that it is a
# forecast is that its figure is not zero while the unspecified one is.
DEFAULT_MODEL = "ecmwf_ifs025"

DEFAULT_HOURLY = (
    "temperature_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "wind_speed_100m",
    "cloud_cover",
    "shortwave_radiation",
)

# Region load centres. NSW1 is the Sydney basin, which is where most of the
# state's demand sits; the others are their capitals for the same reason. A
# single point per region is a simplification — a demand-weighted average over
# several sites would be better and is a Phase 2 refinement, not a Day 2 one.
LOAD_CENTRES: dict[str, tuple[float, float]] = {
    "NSW1": (-33.87, 151.21),
    "VIC1": (-37.81, 144.96),
    "QLD1": (-27.47, 153.03),
    "SA1": (-34.93, 138.60),
    "TAS1": (-42.88, 147.33),
}


@dataclass(frozen=True)
class WeatherRequest:
    region: str
    start: date
    end: date
    hourly: tuple[str, ...] = DEFAULT_HOURLY
    lead: timedelta = field(default=timedelta(0))
    model: str | None = DEFAULT_MODEL

    @property
    def coordinates(self) -> tuple[float, float]:
        try:
            return LOAD_CENTRES[self.region]
        except KeyError as exc:
            raise KeyError(f"no load centre for {self.region}") from exc

    def params(self) -> dict[str, object]:
        lat, lon = self.coordinates
        params: dict[str, object] = {
            "latitude": lat,
            "longitude": lon,
            "start_date": self.start.isoformat(),
            "end_date": self.end.isoformat(),
            "hourly": ",".join(self.hourly),
            "timezone": "UTC",
        }
        if self.model:
            params["models"] = self.model
        return params


def parse_response(payload: dict, request: WeatherRequest) -> pd.DataFrame:
    """Turn an Open-Meteo response into rows carrying an explicit vintage.

    `known_at` is `valid_at` minus `lead`, which with the default zero lead
    means a value becomes usable exactly when it becomes valid.
    """
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError(f"no hourly block in response for {request.region}")

    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.rename(columns={"time": "valid_at"})
    frame.insert(0, "region", request.region)
    frame["known_at"] = frame["valid_at"] - request.lead
    frame["lead_seconds"] = int(request.lead.total_seconds())
    frame["source"] = "open-meteo historical-forecast-api"
    return frame


def fetch(client: httpx.Client, request: WeatherRequest, timeout: float = 60.0) -> pd.DataFrame:
    response = client.get(BASE, params=request.params(), timeout=timeout)
    response.raise_for_status()
    return parse_response(response.json(), request)


def ensure_table(engine: Engine, columns: list[str]) -> None:
    numeric = [c for c in columns if c not in {"region", "valid_at", "known_at", "source"}]
    cols = ",\n            ".join(f'"{c}" double precision' for c in numeric)
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS raw.weather_forecast (
                    region      text        NOT NULL,
                    valid_at    timestamptz NOT NULL,
                    known_at    timestamptz NOT NULL,
                    source      text        NOT NULL,
                    {cols},
                    ingested_at timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (region, valid_at)
                );
                """
            )
        )


def load(engine: Engine, frame: pd.DataFrame) -> int:
    ensure_table(engine, list(frame.columns))
    region = frame["region"].iloc[0]
    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM raw.weather_forecast WHERE region = :r AND valid_at BETWEEN :a AND :b"
            ),
            {"r": region, "a": frame["valid_at"].min(), "b": frame["valid_at"].max()},
        )
    frame.to_sql(
        "weather_forecast",
        engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=2000,
    )
    return len(frame)


def chunk_dates(start: date, end: date, days: int = 90):
    """Split a range into windows. Open-Meteo caps how much one call returns."""
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=days - 1), end)
        yield cursor, stop
        cursor = stop + timedelta(days=1)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
