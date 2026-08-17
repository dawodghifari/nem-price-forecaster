"""Client for NEMWeb: directory listings, filename metadata, downloads.

NEMWeb is an IIS directory index. Each entry carries a timestamp for when the
file appeared on the site, and that timestamp is worth capturing, because it is
not the same as the timestamps inside the file.

A P5MIN filename holds two times already::

    PUBLIC_P5MIN_202608180025_20260818002033.zip
                 |            |
                 |            generated at 00:20:33
                 run labelled 00:25

and the directory index showed that file appearing at 00:24 — about three and a
half minutes after it was generated. The lag is not constant: the 23:40 run was
generated at 23:35:33 and did not appear until 23:48, a twelve minute gap.

That matters for one reason. The forecaster's headline claim is a comparison
against AEMO's own pre-dispatch, so AEMO's forecast is both a feature and the
benchmark. A point-in-time join keyed on the generated time would hand the
model a forecast several minutes before it was actually published, which
flatters the model on the one comparison the project is built around. So every
download records both times, and the feature layer uses the later of the two.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

BASE = "https://nemweb.com.au"
CURRENT = f"{BASE}/Reports/Current"
ARCHIVE = f"{BASE}/Reports/Archive"

# The NEM runs on Australian Eastern Standard Time year round and does not
# observe daylight saving, so a fixed +10:00 offset is correct rather than a
# simplification. Verify on Day 1 that the directory index is also stamped in
# this zone; it is an assumption until a listing time is compared against a
# known publication.
NEM_TZ = timezone(timedelta(hours=10), name="AEST")

# One entry of an IIS directory index:
#   Monday, August 17, 2026  8:08 PM        19017 <A HREF="...zip">...zip</A>
_ENTRY_RE = re.compile(
    r"(?P<listed>\w+,\s+\w+\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s+[AP]M)"
    r"\s+(?P<size>[\d,]+|&lt;dir&gt;|<dir>)"
    r"\s*<a\s+href=\"(?P<href>[^\"]+)\"",
    re.IGNORECASE,
)

_FILENAME_RE = re.compile(
    r"^PUBLIC_(?P<report>[A-Z0-9_]+?)_(?P<label>\d{12})_(?P<tail>\d+)\.zip$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NemwebFile:
    """One entry in a NEMWeb directory index."""

    filename: str
    url: str
    size_bytes: int | None
    listed_at_utc: datetime | None
    report: str | None
    interval_label_utc: datetime | None
    generated_at_utc: datetime | None
    serial: str | None

    @property
    def publish_lag(self) -> timedelta | None:
        """How long after generation the file appeared on the site."""
        if self.listed_at_utc is None or self.generated_at_utc is None:
            return None
        return self.listed_at_utc - self.generated_at_utc

    @property
    def known_at_utc(self) -> datetime | None:
        """The conservative vintage: the later of generated and listed.

        This is the timestamp the feature layer must join on.
        """
        candidates = [t for t in (self.listed_at_utc, self.generated_at_utc) if t is not None]
        return max(candidates) if candidates else None


def _nem_local_to_utc(naive: datetime) -> datetime:
    return naive.replace(tzinfo=NEM_TZ).astimezone(timezone.utc)


def parse_report_filename(filename: str) -> dict[str, object]:
    """Pull the report name and both timestamps out of a NEMWeb filename.

    The trailing field is a generation timestamp on some reports (P5MIN writes
    fourteen digits, YYYYMMDDHHMMSS) and an opaque sequence number on others
    (DISPATCHIS). Distinguish them by length and by whether the digits parse as
    a date, never by report name, so a new report family does not need code.
    """
    out: dict[str, object] = {
        "report": None,
        "interval_label_utc": None,
        "generated_at_utc": None,
        "serial": None,
    }
    m = _FILENAME_RE.match(filename)
    if not m:
        return out

    out["report"] = m.group("report").upper()
    with contextlib.suppress(ValueError):
        out["interval_label_utc"] = _nem_local_to_utc(
            datetime.strptime(m.group("label"), "%Y%m%d%H%M")
        )

    tail = m.group("tail")
    if len(tail) == 14:
        try:
            out["generated_at_utc"] = _nem_local_to_utc(datetime.strptime(tail, "%Y%m%d%H%M%S"))
        except ValueError:
            out["serial"] = tail
    else:
        out["serial"] = tail
    return out


def parse_directory_listing(html: str, base_url: str = BASE) -> list[NemwebFile]:
    """Parse an IIS directory index into structured entries."""
    files: list[NemwebFile] = []
    for m in _ENTRY_RE.finditer(html):
        href = m.group("href")
        filename = href.rsplit("/", 1)[-1]
        if not filename.lower().endswith(".zip"):
            continue

        size_raw = m.group("size").replace(",", "")
        size = int(size_raw) if size_raw.isdigit() else None

        listed_utc = None
        try:
            listed_utc = _nem_local_to_utc(
                datetime.strptime(
                    re.sub(r"\s+", " ", m.group("listed")).strip(), "%A, %B %d, %Y %I:%M %p"
                )
            )
        except ValueError:
            log.debug("unparsed listing time: %r", m.group("listed"))

        meta = parse_report_filename(filename)
        files.append(
            NemwebFile(
                filename=filename,
                url=href if href.startswith("http") else f"{base_url}{href}",
                size_bytes=size,
                listed_at_utc=listed_utc,
                report=meta["report"],  # type: ignore[arg-type]
                interval_label_utc=meta["interval_label_utc"],  # type: ignore[arg-type]
                generated_at_utc=meta["generated_at_utc"],  # type: ignore[arg-type]
                serial=meta["serial"],  # type: ignore[arg-type]
            )
        )
    return files


class NemwebClient:
    """Minimal HTTP client with retries and a contactable user agent."""

    def __init__(self, user_agent: str, timeout: float = 60.0) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            follow_redirects=True,
        )

    def __enter__(self) -> NemwebClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError,)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _get(self, url: str) -> httpx.Response:
        response = self._client.get(url)
        response.raise_for_status()
        return response

    def list_directory(self, url: str) -> list[NemwebFile]:
        return parse_directory_listing(self._get(url).text)

    def latest(self, url: str) -> NemwebFile:
        """The most recently listed file in a directory."""
        entries = self.list_directory(url)
        if not entries:
            raise RuntimeError(f"no zip files listed at {url}")
        return max(
            entries, key=lambda f: f.listed_at_utc or datetime.min.replace(tzinfo=timezone.utc)
        )

    def download(self, entry: NemwebFile, dest_dir: Path, overwrite: bool = False) -> Path:
        """Fetch a file and return its local path. Existing files are reused."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / entry.filename
        if target.exists() and not overwrite:
            log.debug("cached: %s", target)
            return target
        tmp = target.with_suffix(target.suffix + ".part")
        with self._client.stream("GET", entry.url) as response:
            response.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in response.iter_bytes(chunk_size=1 << 16):
                    fh.write(chunk)
        tmp.replace(target)
        return target


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
