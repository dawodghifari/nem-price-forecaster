"""Loading parsed MMS tables into the raw schema.

Two decisions are worth stating, because everything downstream depends on
them.

**Raw tables are generated from the file, not declared in advance.** The
columns come from the I record, and the table name carries the report version,
so `DISPATCH.PRICE` version 5 and version 6 land in different tables. When
AEMO adds a column, the pipeline gets a new table rather than a silent column
shift, and the dbt staging model is the single place that has to notice.

**Raw is append only.** AEMO revises prices after the fact, so the archive
holds final values while the live poller sees first-published ones. Overwriting
would mean training on numbers that were not visible at prediction time, and
would also make the drift monitor in Phase 3 report a market regime change when
the real cause was the pipeline. Instead every row keeps the file it came from
and the time that file became public, and re-running a file is a no-op.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from .mms import MMSFile, TableKey

log = logging.getLogger(__name__)

VINTAGE_COLUMNS = ("_source_file", "_row_ordinal", "_generated_at", "_listed_at", "_known_at")

_SAFE = re.compile(r"[^a-z0-9_]")


@dataclass(frozen=True)
class FileVintage:
    """When a file's contents became knowable."""

    source_file: str
    source_url: str | None = None
    report: str | None = None
    interval_label: datetime | None = None
    generated_at: datetime | None = None
    listed_at: datetime | None = None
    size_bytes: int | None = None
    sha256: str | None = None

    @property
    def known_at(self) -> datetime | None:
        candidates = [t for t in (self.generated_at, self.listed_at) if t is not None]
        return max(candidates) if candidates else None


def get_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True, future=True)


def raw_table_name(key: TableKey) -> str:
    """`DISPATCH.PRICE` v5 becomes `dispatch_price_v5`."""
    parts = [key.namespace, key.table, f"v{key.version}"]
    return _SAFE.sub("_", "_".join(p.strip().lower() for p in parts))


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def ensure_raw_table(engine: Engine, key: TableKey, columns: Iterable[str]) -> str:
    """Create the raw table for a key if it does not exist, and add new columns.

    Everything lands as text. Typing is the staging layer's job, so a value
    AEMO writes in an unexpected shape fails a visible dbt test instead of
    being dropped at ingest.
    """
    table = raw_table_name(key)
    cols = [c.strip() for c in columns if c.strip()]

    ddl_cols = ",\n    ".join(f"{_quote(c)} text" for c in cols)
    create = f"""
    CREATE TABLE IF NOT EXISTS raw.{_quote(table)} (
        {ddl_cols},
        "_source_file"  text        NOT NULL,
        "_row_ordinal"  integer     NOT NULL,
        "_generated_at" timestamptz,
        "_listed_at"    timestamptz,
        "_known_at"     timestamptz,
        "_ingested_at"  timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY ("_source_file", "_row_ordinal")
    );
    """
    index = (
        f"CREATE INDEX IF NOT EXISTS {_quote(table + '_known_at_idx')} "
        f'ON raw.{_quote(table)} ("_known_at");'
    )

    with engine.begin() as conn:
        conn.execute(text(create))
        conn.execute(text(index))
        existing = {
            row[0]
            for row in conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'raw' AND table_name = :t"
                ),
                {"t": table},
            )
        }
        for col in cols:
            if col not in existing:
                # Reached when AEMO adds a column without bumping the version.
                log.warning("raw.%s: adding unseen column %s", table, col)
                conn.execute(text(f"ALTER TABLE raw.{_quote(table)} ADD COLUMN {_quote(col)} text"))
    return table


def load_mms_file(engine: Engine, parsed: MMSFile, vintage: FileVintage) -> dict[str, int]:
    """Load every table in a parsed file. Returns rows written per table.

    Re-running the same file writes nothing, because the primary key is
    (source file, row ordinal).
    """
    written: dict[str, int] = {}
    generated = vintage.generated_at or parsed.header.generated_at_local
    if generated is not None and generated.tzinfo is None:
        # A naive timestamp here came from the file header, which is NEM local.
        from .nemweb import NEM_TZ

        generated = generated.replace(tzinfo=NEM_TZ).astimezone(timezone.utc)

    known = max([t for t in (generated, vintage.listed_at) if t is not None], default=None)

    for key, df in parsed.tables.items():
        if df.empty:
            continue
        table = ensure_raw_table(engine, key, df.columns)
        payload = df.copy()
        payload["_source_file"] = vintage.source_file
        payload["_row_ordinal"] = range(len(payload))
        payload["_generated_at"] = generated
        payload["_listed_at"] = vintage.listed_at
        payload["_known_at"] = known

        with engine.begin() as conn:
            before = conn.execute(
                text(f"SELECT count(*) FROM raw.{_quote(table)} WHERE _source_file = :f"),
                {"f": vintage.source_file},
            ).scalar_one()
            if before:
                log.info("raw.%s already holds %s rows for %s", table, before, vintage.source_file)
                written[table] = 0
                continue

        payload.to_sql(
            raw_table_name(key),
            engine,
            schema="raw",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000,
        )
        written[table] = len(payload)

    _record_ingest(engine, parsed, vintage, written, known, generated)
    return written


def _record_ingest(
    engine: Engine,
    parsed: MMSFile,
    vintage: FileVintage,
    written: dict[str, int],
    known: datetime | None,
    generated: datetime | None,
) -> None:
    anomalies = [
        {"line": a.line_no, "kind": a.kind, "detail": a.detail} for a in parsed.anomalies[:200]
    ]
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO raw.ingest_log (
                    source_file, source_url, report, interval_label,
                    generated_at, listed_at, known_at, size_bytes, sha256,
                    tables_loaded, anomalies, row_count_ok
                ) VALUES (
                    :source_file, :source_url, :report, :interval_label,
                    :generated_at, :listed_at, :known_at, :size_bytes, :sha256,
                    CAST(:tables_loaded AS jsonb), CAST(:anomalies AS jsonb), :row_count_ok
                )
                ON CONFLICT (source_file) DO UPDATE SET
                    tables_loaded = EXCLUDED.tables_loaded,
                    anomalies     = EXCLUDED.anomalies,
                    row_count_ok  = EXCLUDED.row_count_ok
                """
            ),
            {
                "source_file": vintage.source_file,
                "source_url": vintage.source_url,
                "report": vintage.report or parsed.header.report_id,
                "interval_label": vintage.interval_label,
                "generated_at": generated,
                "listed_at": vintage.listed_at,
                "known_at": known,
                "size_bytes": vintage.size_bytes,
                "sha256": vintage.sha256,
                "tables_loaded": json.dumps(written),
                "anomalies": json.dumps(anomalies),
                "row_count_ok": parsed.row_count_matches,
            },
        )


def row_hash(values: Iterable[object]) -> str:
    """Stable hash of a row, for comparing a revision against its predecessor."""
    joined = "\x1f".join("" if v is None else str(v) for v in values)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def frame_is_empty(df: pd.DataFrame) -> bool:  # pragma: no cover - trivial
    return df is None or df.empty
