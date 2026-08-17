"""Parser for AEMO MMS flat files.

An MMS file is not a CSV with a header. It is a stream of records whose first
field states the record type:

    C  comment. The first line is the file header; the last line is a footer
       carrying the number of *records* the file claims to hold — every C, I
       and D line, not just the data rows. Confirmed against live files on
       18 Aug 2026; see `MMSFile.record_count_matches`.
    I  information. Declares a table and *is* that table's header row.
    D  data. One row belonging to the most recently declared table with the
       same (namespace, table, version) key.

A single file holds several tables. Each table is versioned, and AEMO bumps
that version when it changes the column set, so the columns must be read from
the I record rather than assumed. That is the whole reason this module reads
the header out of the file instead of hard-coding a schema: when AEMO ships a
new report version, the result is a new table key, not a silent column shift.

Example, abridged::

    C,NEMP.WORLD,DISPATCHIS,AEMO,PUBLIC,2026/08/17,20:08:00,533127148,...
    I,DISPATCH,PRICE,5,SETTLEMENTDATE,RUNNO,REGIONID,DISPATCHINTERVAL,RRP,...
    D,DISPATCH,PRICE,5,"2026/08/17 20:10:00",1,NSW1,244,89.63,...
    C,"END OF REPORT",412
"""

from __future__ import annotations

import contextlib
import csv
import io
import logging
import re
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO

import pandas as pd

log = logging.getLogger(__name__)

# AEMO writes timestamps as "2026/08/17 20:10:00" in data fields and as a
# separate date and time pair in the C header.
_DATE_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


@dataclass(frozen=True)
class TableKey:
    """Identifies one table inside one MMS file."""

    namespace: str
    table: str
    version: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.namespace}.{self.table}.v{self.version}"


@dataclass
class MMSHeader:
    """The C record at the top of the file."""

    system: str | None = None
    report_id: str | None = None
    origin: str | None = None
    audience: str | None = None
    generated_at_local: datetime | None = None
    serial: str | None = None
    raw: list[str] = field(default_factory=list)


@dataclass
class MMSAnomaly:
    """Something the file did that the format does not promise."""

    line_no: int
    kind: str
    detail: str


@dataclass
class MMSFile:
    """A parsed MMS file."""

    header: MMSHeader
    tables: dict[TableKey, pd.DataFrame]
    declared_record_count: int | None = None
    observed_record_count: int = 0
    observed_row_count: int = 0
    anomalies: list[MMSAnomaly] = field(default_factory=list)
    source_name: str | None = None

    def get(self, table: str, namespace: str | None = None) -> pd.DataFrame:
        """Fetch a table by name, ignoring version.

        Raises if the name is ambiguous, because silently picking one of two
        report versions is exactly the class of bug this parser exists to
        prevent.
        """
        matches = [
            k
            for k in self.tables
            if k.table.upper() == table.upper()
            and (namespace is None or k.namespace.upper() == namespace.upper())
        ]
        if not matches:
            available = ", ".join(str(k) for k in self.tables) or "(none)"
            raise KeyError(f"no table {table!r} in {self.source_name}. Present: {available}")
        if len(matches) > 1:
            found = ", ".join(str(k) for k in matches)
            raise KeyError(f"{table!r} is ambiguous in {self.source_name}: {found}")
        return self.tables[matches[0]]

    def inventory(self) -> pd.DataFrame:
        """One row per table: key and row count. Used by the Day 1 report."""
        return pd.DataFrame(
            [
                {
                    "namespace": k.namespace,
                    "table": k.table,
                    "version": k.version,
                    "rows": len(df),
                    "columns": len(df.columns),
                }
                for k, df in sorted(self.tables.items(), key=str)
            ]
        )

    @property
    def record_count_matches(self) -> bool | None:
        """Whether the footer's count matches every record parsed.

        The footer counts **all** records — the C header, every I record, every
        D record, and the footer itself — not just data rows. Measured on live
        files 18 Aug 2026: a DISPATCHIS footer claiming 997 held 988 D rows
        across 7 tables, and 988 + 7 + 2 = 997. A P5MIN footer claiming 11766
        held 11759 D rows across 5 tables, and 11759 + 5 + 2 = 11766.

        Reading it as a data-row count makes every well-formed file look
        corrupt, which would have trained us to ignore the one integrity check
        the format hands us for free.
        """
        if self.declared_record_count is None:
            return None
        return self.declared_record_count == self.observed_record_count


def _parse_header(row: list[str]) -> MMSHeader:
    """Read the C header without assuming field positions.

    Positions have moved between report families in the past, so the date and
    time are located by shape rather than by index.
    """
    header = MMSHeader(raw=list(row))
    fields = row[1:]
    for i, value in enumerate(fields):
        if _DATE_RE.match(value.strip()) and i + 1 < len(fields):
            nxt = fields[i + 1].strip()
            if _TIME_RE.match(nxt):
                with contextlib.suppress(ValueError):  # shape already checked above
                    header.generated_at_local = datetime.strptime(
                        f"{value.strip()} {nxt}", "%Y/%m/%d %H:%M:%S"
                    )
                remainder = fields[i + 2 :]
                if remainder:
                    header.serial = remainder[0].strip() or None
                break
    named = [f.strip() for f in fields[:4]]
    for attr, value in zip(("system", "report_id", "origin", "audience"), named, strict=False):
        setattr(header, attr, value or None)
    return header


def parse_mms(lines: Iterable[str], source_name: str | None = None) -> MMSFile:
    """Parse MMS records into one dataframe per table.

    Every value stays a string. Typing happens in the dbt staging layer, so
    that the raw layer is a faithful record of what AEMO published and a type
    change never loses data at ingest time.
    """
    header = MMSHeader()
    headers: dict[TableKey, list[str]] = {}
    rows: dict[TableKey, list[list[str | None]]] = {}
    anomalies: list[MMSAnomaly] = []
    declared: int | None = None
    observed = 0
    records = 0
    seen_file_header = False

    for line_no, row in enumerate(csv.reader(lines), start=1):
        if not row or not row[0].strip():
            continue
        kind = row[0].strip().upper()
        records += 1

        if kind == "C":
            if not seen_file_header:
                header = _parse_header(row)
                seen_file_header = True
            elif len(row) > 2 and "END OF REPORT" in row[1].upper():
                try:
                    declared = int(row[2].strip())
                except (ValueError, IndexError):
                    anomalies.append(
                        MMSAnomaly(line_no, "bad_footer", f"unreadable record count: {row[2:]!r}")
                    )
            continue

        if kind not in {"I", "D"}:
            anomalies.append(MMSAnomaly(line_no, "unknown_record", f"record type {kind!r}"))
            continue

        if len(row) < 4:
            anomalies.append(MMSAnomaly(line_no, "short_record", f"{len(row)} fields"))
            continue

        key = TableKey(row[1].strip(), row[2].strip(), row[3].strip())

        if kind == "I":
            cols = [c.strip() for c in row[4:]]
            if key in headers and headers[key] != cols:
                anomalies.append(
                    MMSAnomaly(line_no, "header_redeclared", f"{key} columns changed mid-file")
                )
            headers[key] = cols
            rows.setdefault(key, [])
            continue

        # kind == "D"
        if key not in headers:
            anomalies.append(MMSAnomaly(line_no, "orphan_data", f"{key} has no I record"))
            continue

        values: list[str | None] = [v.strip() for v in row[4:]]
        expected = len(headers[key])
        if len(values) < expected:
            # AEMO drops trailing empty fields on some rows. Pad rather than
            # discard, and record it so the Day 1 report can show how often it
            # happens.
            anomalies.append(
                MMSAnomaly(line_no, "short_row", f"{key}: {len(values)} of {expected} fields")
            )
            values = values + [None] * (expected - len(values))
        elif len(values) > expected:
            anomalies.append(
                MMSAnomaly(line_no, "long_row", f"{key}: {len(values)} of {expected} fields")
            )
            values = values[:expected]
        rows[key].append(values)
        observed += 1

    tables = {
        key: pd.DataFrame(rows.get(key, []), columns=cols, dtype="object")
        for key, cols in headers.items()
    }

    parsed = MMSFile(
        header=header,
        tables=tables,
        declared_record_count=declared,
        observed_record_count=records,
        observed_row_count=observed,
        anomalies=anomalies,
        source_name=source_name,
    )
    if parsed.record_count_matches is False:
        log.warning(
            "%s: footer claims %s records, parsed %s (%s data rows)",
            source_name,
            declared,
            records,
            observed,
        )
    return parsed


def parse_mms_text(text: str, source_name: str | None = None) -> MMSFile:
    return parse_mms(io.StringIO(text), source_name=source_name)


def _iter_csv_members(zf: zipfile.ZipFile, prefix: str = "") -> Iterator[tuple[str, bytes]]:
    """Yield (name, bytes) for every CSV in a zip, descending into nested zips.

    NEMWeb serves single-file zips for current reports and zip-of-zips daily
    rollups in the archive, so both shapes have to work.
    """
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = f"{prefix}{info.filename}"
        payload = zf.read(info)
        if info.filename.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                yield from _iter_csv_members(inner, prefix=f"{name}!")
        else:
            yield name, payload


def read_mms_zip(source: str | Path | IO[bytes]) -> list[MMSFile]:
    """Parse every MMS file inside a zip, including nested archive rollups."""
    label = str(source) if isinstance(source, (str, Path)) else "<stream>"
    parsed: list[MMSFile] = []
    with zipfile.ZipFile(source) as zf:
        for name, payload in _iter_csv_members(zf):
            text = payload.decode("utf-8", errors="replace")
            parsed.append(parse_mms_text(text, source_name=f"{Path(label).name}!{name}"))
    if not parsed:
        raise ValueError(f"no MMS members found in {label}")
    return parsed


def read_mms_zip_single(source: str | Path | IO[bytes]) -> MMSFile:
    """Parse a zip expected to hold exactly one MMS file."""
    files = read_mms_zip(source)
    if len(files) != 1:
        names = ", ".join(f.source_name or "?" for f in files)
        raise ValueError(f"expected one MMS member, found {len(files)}: {names}")
    return files[0]
