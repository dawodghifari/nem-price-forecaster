"""Backfill the NEMWeb archive, the long price history and the weather archive.

    make backfill                 # DISPATCHIS + P5MIN, allowlisted tables only
    make backfill ARGS="--report DISPATCHIS --days 30"
    make backfill-history         # aggregated price and demand, 2015 onward
    make backfill-weather

Resumable: every member already in raw.ingest_log is skipped, so an interrupted
run continues where it stopped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemfc import aggregated, weather
from nemfc.config import load_settings
from nemfc.db import get_engine
from nemfc.ingest import backfill
from nemfc.nemweb import NemwebClient
from nemfc.tables import PHASE1_TABLES

log = logging.getLogger("backfill")


def run_mms(args, settings) -> None:
    engine = get_engine(settings.database_url)
    tables = None if args.all_tables else PHASE1_TABLES
    start = date.today() - timedelta(days=args.days) if args.days else None

    with NemwebClient(settings.user_agent, min_interval=args.rate) as client:
        for report in args.report:
            log.info("backfilling %s from %s", report, start or "the start of the archive")
            days = 0
            rows = 0
            for filename, written in backfill(
                engine,
                client,
                report,
                settings.data_dir,
                tables=tables,
                start=start,
                keep_zip=args.keep_zip,
            ):
                days += 1
                rows += sum(written.values())
                log.info(
                    "%s: %s rows (%s days, %s rows total)",
                    filename,
                    sum(written.values()),
                    days,
                    rows,
                )
            log.info("%s done: %s days, %s rows", report, days, rows)


def run_history(args, settings) -> None:
    engine = get_engine(settings.database_url)
    start = date(args.from_year, 1, 1)
    end = date.today()
    total = 0
    with httpx.Client(headers={"User-Agent": settings.user_agent}, follow_redirects=True) as client:
        for month in aggregated.months_between(start, end):
            try:
                series = aggregated.fetch_month(client, settings.region, month)
            except httpx.HTTPStatusError as exc:
                log.warning("%s: %s", month.strftime("%Y-%m"), exc)
                continue
            written = aggregated.load_month(engine, series)
            total += written
            expected = aggregated.expected_rows(month, series.spacing_minutes)
            flag = "" if written >= expected * 0.98 else f"  GAP: expected ~{expected}"
            log.info(
                "%s %s: %s rows at %s-minute spacing%s",
                settings.region,
                month.strftime("%Y-%m"),
                written,
                series.spacing_minutes,
                flag,
            )
    log.info("aggregated history done: %s rows", total)


def run_weather(args, settings) -> None:
    engine = get_engine(settings.database_url)
    start = date.today() - timedelta(days=args.days)
    end = date.today() - timedelta(days=1)
    total = 0
    with httpx.Client(headers={"User-Agent": settings.user_agent}, follow_redirects=True) as client:
        for chunk_start, chunk_end in weather.chunk_dates(start, end):
            request = weather.WeatherRequest(
                region=settings.region, start=chunk_start, end=chunk_end
            )
            frame = weather.fetch(client, request)
            total += weather.load(engine, frame)
            log.info("weather %s to %s: %s rows", chunk_start, chunk_end, len(frame))
    log.info("weather done: %s rows", total)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--what",
        choices=["mms", "history", "weather"],
        default="mms",
        help="which source to backfill",
    )
    parser.add_argument(
        "--report",
        action="append",
        choices=["DISPATCHIS", "P5MIN"],
        help="repeatable; defaults to both",
    )
    parser.add_argument("--days", type=int, default=0, help="0 means the whole archive")
    parser.add_argument(
        "--all-tables",
        action="store_true",
        help="load every table. Adds ~1.35 billion rows; see src/nemfc/tables.py",
    )
    parser.add_argument(
        "--discard-zips",
        dest="keep_zip",
        action="store_false",
        help="delete each rollup after loading. P5MIN is ~20.6 GB kept",
    )
    parser.add_argument("--from-year", type=int, default=2015)
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="minimum seconds between requests to NEMWeb",
    )
    parser.set_defaults(keep_zip=True)
    args = parser.parse_args()
    args.report = args.report or ["DISPATCHIS", "P5MIN"]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()

    {"mms": run_mms, "history": run_history, "weather": run_weather}[args.what](args, settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
