"""Ingestion: the live poller and the archive backfill.

Both paths funnel through `ingest_zip`, so a file collected live and the same
file collected from the archive land identically apart from their vintage.

**The archive rollup carries a weaker vintage than a live file, and that is
recorded rather than papered over.** A live download knows when the file
appeared in the NEMWeb index, to the minute. A daily rollup is one zip
published the next morning, so per-interval listing times are gone; only the
generation timestamp inside each member survives. Since P5MIN publication ran
7.5 minutes behind generation at the median and 23 minutes at the tail, a
backfilled row's `known_at` would otherwise be optimistic by that much.

`ARCHIVE_PUBLISH_LAG` is added to the generation time for archive-sourced rows,
so a backfilled vintage stays conservative. Rows carry `_vintage_source` so the
distinction survives into the feature layer and can be audited.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Collection, Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import Engine, text

from .db import FileVintage, load_mms_file
from .mms import iter_mms_zip
from .nemweb import ARCHIVE, CURRENT, NEM_TZ, NemwebClient, NemwebFile, sha256_of
from .tables import REPORTS

log = logging.getLogger(__name__)

# The 90th percentile of observed P5MIN publication lag, 18 Aug 2026. Applied
# to archive-sourced rows, where the true listing time is unrecoverable.
ARCHIVE_PUBLISH_LAG = timedelta(minutes=12, seconds=30)


def member_key(source_name: str) -> str:
    """The identity of a dispatch interval, independent of how it was fetched.

    The same interval arrives by two routes with two different paths:

        live     PUBLIC_DISPATCHIS_202608160130_0000000532838629.zip!...CSV
        archive  PUBLIC_DISPATCHIS_20260816.zip!PUBLIC_DISPATCHIS_2026...zip!...CSV

    Keying on the full path made those look like different files, so a poller
    run after a backfill re-downloaded two days of intervals it already had —
    163 needless requests, which is most of what earned the 403. The innermost
    component is the same either way, so that is the key.
    """
    return source_name.rsplit("!", 1)[-1].strip().upper()


def _already_loaded(engine: Engine, source_name: str) -> bool:
    with engine.connect() as conn:
        return (
            conn.execute(
                text("SELECT 1 FROM raw.ingest_log WHERE member_key = :k LIMIT 1"),
                {"k": member_key(source_name)},
            ).first()
            is not None
        )


def loaded_member_keys(engine: Engine, report: str) -> set[str]:
    """Every interval already held, for cheap set membership during a pass."""
    with engine.connect() as conn:
        return {
            row[0]
            for row in conn.execute(
                text("SELECT member_key FROM raw.ingest_log WHERE report = :r"),
                {"r": report},
            )
        }


def ingest_zip(
    engine: Engine,
    path: Path,
    entry: NemwebFile | None = None,
    tables: Collection[str] | None = None,
    archive: bool = False,
    known: set[str] | None = None,
) -> dict[str, int]:
    """Parse a downloaded zip and load its allowlisted tables.

    A daily archive rollup holds 288 members, each of which is logged as its own
    source file so a partial backfill can resume mid-day.
    """
    totals: dict[str, int] = {}
    known = loaded_member_keys(engine, entry.report if entry else "") if known is None else known
    for parsed in iter_mms_zip(path, tables=tables):
        member = parsed.source_name or path.name
        key = member_key(member)
        if key in known:
            continue
        known.add(key)

        generated = parsed.header.generated_at_local
        listed = None if archive else (entry.listed_at_utc if entry else None)
        if archive and generated is not None:
            # No per-interval listing time survives in a rollup. Push the
            # vintage out by the observed publication lag instead of pretending
            # the file was readable the instant it was generated.
            listed = _as_utc(generated) + ARCHIVE_PUBLISH_LAG

        vintage = FileVintage(
            source_file=member,
            member_key=key,
            source_url=entry.url if entry else None,
            report=entry.report if entry else parsed.header.report_id,
            interval_label=entry.interval_label_utc if entry else None,
            listed_at=listed,
            size_bytes=entry.size_bytes if entry else None,
            vintage_source="archive" if archive else "live",
        )
        for table, count in load_mms_file(engine, parsed, vintage).items():
            totals[table] = totals.get(table, 0) + count
    return totals


def _as_utc(naive: datetime) -> datetime:
    """A C-header timestamp is NEM local; the database is UTC."""
    return naive.replace(tzinfo=NEM_TZ).astimezone(timezone.utc)


def poll_once(
    engine: Engine,
    client: NemwebClient,
    reports: Collection[str],
    data_dir: Path,
    tables: Collection[str] | None = None,
    max_files: int = 0,
) -> dict[str, int]:
    """Fetch every file in Reports/Current not already ingested.

    Deliberately not "fetch the latest". The exit condition for Day 2 is a
    poller that does not drop intervals, and a poller that only ever takes the
    newest file drops one every time a run is slow or the process restarts.
    Reports/Current holds about two days, so anything missed inside that window
    is recovered on the next pass without a separate catch-up path.
    """
    written: dict[str, int] = {}
    for report in reports:
        known = loaded_member_keys(engine, report)
        fetched = 0
        for entry in client.list_directory(f"{CURRENT}/{REPORTS[report]}/"):
            if predicted_member_key(entry.filename) in known:
                continue
            if max_files and fetched >= max_files:
                log.info("%s: stopping this pass at %s files, rest next time", report, fetched)
                break
            path = client.download(entry, data_dir / "current" / report)
            fetched += 1
            for table, count in ingest_zip(
                engine, path, entry=entry, tables=tables, known=known
            ).items():
                written[table] = written.get(table, 0) + count
    return written


def predicted_member_key(zip_filename: str) -> str:
    """The member key a current-report zip will yield, without opening it.

    NEMWeb wraps one CSV of the same name in each current-report zip, so the
    key is knowable from the listing. That is what lets a pass skip a file it
    already holds *before* spending a request on it.
    """
    return f"{zip_filename.rsplit('.', 1)[0]}.CSV".upper()


def archive_days(client: NemwebClient, report: str) -> list[NemwebFile]:
    """Daily rollups available in the archive, oldest first."""
    entries = client.list_directory(f"{ARCHIVE}/{REPORTS[report]}/")
    return sorted(entries, key=lambda e: e.filename)


def backfill(
    engine: Engine,
    client: NemwebClient,
    report: str,
    data_dir: Path,
    tables: Collection[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    keep_zip: bool = True,
) -> Iterator[tuple[str, dict[str, int]]]:
    """Download and load archive rollups, yielding progress per day.

    `keep_zip=False` deletes each rollup after loading. P5MIN is about 20.6 GB
    across the archive, and since the point of `make backfill` is that the data
    is reproducible rather than stored, keeping it is optional.
    """
    target = data_dir / "archive" / report
    known = loaded_member_keys(engine, report)
    per_day = Counter(d for d in (member_day(k) for k in known) if d)

    for entry in archive_days(client, report):
        day = _day_from(entry.filename)
        if day is None:
            log.warning("skipping unparseable archive filename %s", entry.filename)
            continue
        if (start and day < start) or (end and day > end):
            continue
        if _day_complete(per_day, day):
            log.info("%s: already loaded, not downloading", entry.filename)
            yield entry.filename, {}
            continue

        path = client.download(entry, target)
        try:
            written = ingest_zip(
                engine, path, entry=entry, tables=tables, archive=True, known=known
            )
        finally:
            if not keep_zip and path.exists():
                path.unlink()
        yield entry.filename, written


INTERVALS_PER_DAY = 288

# A rollup for day D holds the 287 intervals labelled D, plus the one labelled
# D+1 00:00 -- which physically covers 23:55 to midnight of D. So a calendar day
# is only complete once both its own rollup and the previous one are in.
OWN_INTERVALS = INTERVALS_PER_DAY - 1


def member_day(key: str) -> str | None:
    """The YYYYMMDD a member key belongs to, from its interval label."""
    parts = key.split("_")
    if len(parts) >= 3 and len(parts[2]) >= 8 and parts[2][:8].isdigit():
        return parts[2][:8]
    return None


def _day_complete(per_day: Counter, day: date) -> bool:
    """Whether a rollup can be skipped without leaving a hole.

    Requires both that the day itself is loaded, and that the single
    next-midnight interval this rollup carries is already present. Checking only
    the first would silently drop one interval per skipped day, which is exactly
    the kind of gap that is invisible until a rolling feature is quietly wrong.
    """
    following = day + timedelta(days=1)
    return (
        per_day.get(day.strftime("%Y%m%d"), 0) >= OWN_INTERVALS
        and per_day.get(following.strftime("%Y%m%d"), 0) >= 1
    )


def _day_from(filename: str) -> date | None:
    stem = filename.rsplit(".", 1)[0]
    digits = stem.rsplit("_", 1)[-1]
    try:
        return datetime.strptime(digits, "%Y%m%d").date()
    except ValueError:
        return None


def checksum(path: Path) -> str:
    return sha256_of(path)
