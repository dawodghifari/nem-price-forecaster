"""Live five-minute poller.

    make poll            # loop until interrupted
    make poll ARGS="--once"

Each pass ingests every file in Reports/Current not already in the ingest log,
rather than only the newest one. Reports/Current holds about two days, so a
pass missed because the process was down, or because AEMO published late, is
picked up on the next pass without a separate catch-up path. That is what makes
"runs on a loop without dropping intervals" true rather than hoped for.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemfc.config import load_settings
from nemfc.db import get_engine
from nemfc.ingest import poll_once
from nemfc.nemweb import NemwebClient, RateLimited
from nemfc.tables import PHASE1_TABLES

log = logging.getLogger("poll")

_stop = False


def _handle_signal(*_: object) -> None:
    global _stop
    _stop = True
    log.info("stopping after the current pass")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=60, help="seconds between passes")
    parser.add_argument("--report", action="append", choices=["DISPATCHIS", "P5MIN"])
    parser.add_argument("--all-tables", action="store_true")
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="minimum seconds between requests to NEMWeb",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=60,
        help="cap per pass. A cold start spreads over several passes rather "
        "than pulling two days of intervals in one burst",
    )
    args = parser.parse_args()
    reports = args.report or ["DISPATCHIS", "P5MIN"]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    settings = load_settings()
    engine = get_engine(settings.database_url)
    tables = None if args.all_tables else PHASE1_TABLES

    # A pass costs one directory listing per report even when nothing is new,
    # which is far cheaper than the alternative of tracking state ourselves.
    backoff = 0.0
    with NemwebClient(settings.user_agent, min_interval=args.rate) as client:
        while True:
            started = time.monotonic()
            try:
                written = poll_once(
                    engine,
                    client,
                    reports,
                    settings.data_dir,
                    tables=tables,
                    max_files=args.max_files,
                )
                backoff = 0.0
                if written:
                    log.info("loaded %s", written)
                else:
                    log.debug("nothing new")
            except RateLimited as exc:
                # Never retry straight into a refusal. AEMO publishes this for
                # anyone to use, but it is a web server, not a metered API, and
                # the way to lose access is to argue with a 403.
                backoff = exc.retry_after or max(60.0, min(backoff * 2 or 60.0, 900.0))
                log.warning("%s -- backing off %.0f s. Consider a larger --rate.", exc, backoff)
            except Exception:
                log.exception("pass failed; retrying next interval")

            if args.once or _stop:
                return 0
            wait = max(args.interval, backoff) - (time.monotonic() - started)
            time.sleep(max(0.0, wait))


if __name__ == "__main__":
    raise SystemExit(main())
