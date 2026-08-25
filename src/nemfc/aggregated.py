"""AEMO's aggregated price and demand series — the long history.

One monthly CSV per region, back to December 1998. It feeds the climatology
baseline, the spike-threshold sensitivity check and the regime comparison in
PLAN §2.2. It never feeds the feature table: it carries only price and demand,
and the model's features need the full dispatch tables.

**Resolution changes partway through, and the change is detected rather than
assumed.** Since five-minute settlement began in October 2021 these files are
five-minute; before that they are half-hourly. The August 2026 file holds 4,896
rows for seventeen days, which is 288 a day. A backfill spanning the change
would silently mix the two if it assumed either one, so `fetch_month` records
the spacing it measured and the loader stores it per row.

That the recent history is five-minute is worth more than a footnote: it means
the climatology baseline can be estimated at the resolution the model actually
runs at, over roughly five years, rather than being built at thirty minutes and
compared against a five-minute model.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd
from sqlalchemy import Engine, text

from .nemweb import NEM_TZ

log = logging.getLogger(__name__)

URL = "https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{ym}_{region}.csv"

FIVE_MINUTE_SETTLEMENT_START = date(2021, 10, 1)


@dataclass
class MonthlySeries:
    region: str
    month: date
    frame: pd.DataFrame
    spacing_minutes: int
    url: str

    @property
    def matches_expected_spacing(self) -> bool:
        """Whether the measured spacing agrees with the settlement-rule date."""
        expected = 5 if self.month >= FIVE_MINUTE_SETTLEMENT_START else 30
        return self.spacing_minutes == expected


def _measure_spacing(frame: pd.DataFrame) -> int:
    gaps = frame["SETTLEMENTDATE"].diff().dropna().mode()
    return int(gaps.iloc[0].total_seconds() // 60) if not gaps.empty else 0


def fetch_month(
    client: httpx.Client, region: str, month: date, timeout: float = 90.0
) -> MonthlySeries:
    """Fetch one region-month. Timestamps are converted from NEM local to UTC."""
    url = URL.format(ym=month.strftime("%Y%m"), region=region)
    response = client.get(url, timeout=timeout)
    response.raise_for_status()
    frame = pd.read_csv(io.BytesIO(response.content))
    frame["SETTLEMENTDATE"] = pd.to_datetime(frame["SETTLEMENTDATE"])
    frame = frame.sort_values("SETTLEMENTDATE").reset_index(drop=True)

    spacing = _measure_spacing(frame)
    series = MonthlySeries(region, month, frame, spacing, url)
    if not series.matches_expected_spacing:
        log.warning(
            "%s %s: measured %s-minute spacing, expected %s. Using the measurement.",
            region,
            month.strftime("%Y-%m"),
            spacing,
            5 if month >= FIVE_MINUTE_SETTLEMENT_START else 30,
        )
    return series


def months_between(start: date, end: date):
    cursor = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cursor <= last:
        yield cursor
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)


def ensure_table(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS raw.aggregated_price_demand (
                    region            text        NOT NULL,
                    settlement_end    timestamptz NOT NULL,
                    total_demand      double precision,
                    rrp               double precision,
                    period_type       text,
                    spacing_minutes   integer     NOT NULL,
                    source_url        text        NOT NULL,
                    ingested_at       timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (region, settlement_end)
                );
                """
            )
        )


def load_month(engine: Engine, series: MonthlySeries) -> int:
    """Write one region-month. Re-running a month replaces it.

    This table is a derived convenience rather than a vintage record — AEMO
    publishes one settled value per interval here and does not restate it
    through this endpoint — so unlike the MMS raw tables it is safe to replace
    in place. Revisions, if they happen, arrive as a corrected monthly file.
    """
    ensure_table(engine)
    frame = series.frame
    payload = pd.DataFrame(
        {
            "region": frame.get("REGION", series.region),
            "settlement_end": frame["SETTLEMENTDATE"]
            .dt.tz_localize(NEM_TZ)
            .dt.tz_convert(timezone.utc),
            "total_demand": pd.to_numeric(frame.get("TOTALDEMAND"), errors="coerce"),
            "rrp": pd.to_numeric(frame.get("RRP"), errors="coerce"),
            "period_type": frame.get("PERIODTYPE"),
            "spacing_minutes": series.spacing_minutes,
            "source_url": series.url,
        }
    )

    start = payload["settlement_end"].min()
    end = payload["settlement_end"].max()
    with engine.begin() as conn:
        conn.execute(
            text(
                "DELETE FROM raw.aggregated_price_demand "
                "WHERE region = :r AND settlement_end BETWEEN :a AND :b"
            ),
            {"r": series.region, "a": start, "b": end},
        )
    payload.to_sql(
        "aggregated_price_demand",
        engine,
        schema="raw",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=2000,
    )
    return len(payload)


def expected_rows(month: date, spacing_minutes: int, today: date | None = None) -> int:
    """How many rows a complete month should hold, for a gap check.

    The month in progress is counted only up to yesterday. Comparing it against
    a whole month makes the current month fail the check every time it runs,
    and a gap detector that always fires is a gap detector nobody reads.
    """
    following = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
    today = today or date.today()
    end = min(following, today)
    days = max((end - month).days, 0)
    return days * 24 * 60 // max(spacing_minutes, 1)


def nem_local(moment: datetime) -> datetime:
    return moment.astimezone(NEM_TZ)


def trading_day(settlement_end: datetime) -> date:
    """The NEM trading day a settlement interval belongs to.

    The trading day runs 04:00 to 04:00, which is why a DISPATCHIS row at 01:00
    on 18 August carried trading date 17 August. Calendar features built on the
    civil date would split every overnight period in the wrong place.
    """
    local = settlement_end.astimezone(NEM_TZ)
    return (local - timedelta(hours=4)).date()
