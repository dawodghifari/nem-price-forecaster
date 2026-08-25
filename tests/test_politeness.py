"""Pacing, refusal handling, and cross-route deduplication.

All three exist because of one live incident on 18 Aug 2026: a first poller
pass fetched 163 files in 52 seconds and was refused with 403. Most of those
requests were for intervals already held from a backfill, which the dedupe was
failing to recognise because it keyed on the full path rather than the file.
"""

from __future__ import annotations

import itertools
import time

import httpx
import pytest

from nemfc.ingest import member_key, predicted_member_key
from nemfc.nemweb import NemwebClient, NemwebFile, RateLimited

LIVE = "PUBLIC_DISPATCHIS_202608160130_0000000532838629.zip!PUBLIC_DISPATCHIS_202608160130_0000000532838629.CSV"
ARCHIVE = (
    "PUBLIC_DISPATCHIS_20260816.zip"
    "!PUBLIC_DISPATCHIS_202608160130_0000000532838629.zip"
    "!PUBLIC_DISPATCHIS_202608160130_0000000532838629.CSV"
)


def test_the_same_interval_has_one_key_by_either_route():
    """The bug: these two are the same dispatch interval."""
    assert member_key(LIVE) == member_key(ARCHIVE)
    assert member_key(LIVE) == "PUBLIC_DISPATCHIS_202608160130_0000000532838629.CSV"


def test_different_intervals_keep_different_keys():
    other = ARCHIVE.replace("202608160130", "202608160135")
    assert member_key(ARCHIVE) != member_key(other)


def test_the_key_is_predictable_from_a_listing_without_downloading():
    """This is what lets a pass skip a file before spending a request on it."""
    zip_name = "PUBLIC_DISPATCHIS_202608160130_0000000532838629.zip"
    assert predicted_member_key(zip_name) == member_key(LIVE)


def _client(handler, **kwargs) -> NemwebClient:
    return NemwebClient("test-agent", transport=httpx.MockTransport(handler), **kwargs)


def test_requests_are_paced():
    calls = []

    def handler(request):
        calls.append(time.monotonic())
        return httpx.Response(200, text="<pre></pre>")

    with _client(handler, min_interval=0.05) as client:
        for _ in range(3):
            client.list_directory("https://nemweb.com.au/x/")

    gaps = [b - a for a, b in itertools.pairwise(calls)]
    assert all(gap >= 0.045 for gap in gaps), gaps


def test_a_403_raises_rate_limited_rather_than_a_generic_http_error():
    with _client(lambda r: httpx.Response(403)) as client, pytest.raises(RateLimited) as caught:
        client.list_directory("https://nemweb.com.au/x/")
    assert caught.value.status == 403


def test_a_429_is_treated_the_same_and_carries_retry_after():
    handler = lambda r: httpx.Response(429, headers={"retry-after": "120"})  # noqa: E731
    with _client(handler) as client, pytest.raises(RateLimited) as caught:
        client.list_directory("https://nemweb.com.au/x/")
    assert caught.value.retry_after == 120.0


def test_a_refusal_is_not_retried():
    """Retrying a 403 is how a rate limit becomes a ban."""
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(403)

    with _client(handler) as client, pytest.raises(RateLimited):
        client.list_directory("https://nemweb.com.au/x/")
    assert len(attempts) == 1


def test_a_404_is_not_retried_either():
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(404)

    with _client(handler) as client, pytest.raises(httpx.HTTPStatusError):
        client.list_directory("https://nemweb.com.au/x/")
    assert len(attempts) == 1


def test_transport_faults_are_retried():
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            raise httpx.ConnectError("flaky", request=request)
        return httpx.Response(200, text="<pre></pre>")

    with _client(handler) as client:
        client.list_directory("https://nemweb.com.au/x/")
    assert len(attempts) == 3


def test_download_reuses_a_cached_file_without_a_request(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, content=b"payload")

    entry = NemwebFile(
        filename="PUBLIC_DISPATCHIS_202608160130_1.zip",
        url="https://nemweb.com.au/x/PUBLIC_DISPATCHIS_202608160130_1.zip",
        size_bytes=7,
        listed_at_utc=None,
        report="DISPATCHIS",
        interval_label_utc=None,
        generated_at_utc=None,
        serial="1",
    )
    with _client(handler, min_interval=0.0) as client:
        first = client.download(entry, tmp_path)
        second = client.download(entry, tmp_path)

    assert first == second
    assert first.read_bytes() == b"payload"
    assert len(calls) == 1
