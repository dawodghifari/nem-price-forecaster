"""Independent verification of the feature table, against the raw layer.

`tests/test_no_leakage.py` proves the invariants fire on planted leaks, and the
dbt tests assert them in the warehouse. Neither is sufficient on its own, and
once the mart filters on the same condition a dbt test checks, that test can
only pass -- the table is certifying itself.

So this file recomputes the answers from `raw.*` in Python, reusing none of the
dbt SQL: its own timezone conversion, its own as-of logic, its own arithmetic.
Two implementations agreeing is evidence; one implementation agreeing with
itself is not.

Requires a populated database. Skipped, loudly, when there isn't one, because a
silently skipped verification is worse than none.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd
import pytest

pytest.importorskip("sqlalchemy")
from sqlalchemy import create_engine, text

SAMPLE_SIZE = int(os.getenv("NEMFC_LEAKAGE_SAMPLE", "200"))
REGION = os.getenv("NEMFC_REGION", "NSW1")

# AEST, fixed all year. Written out rather than imported from nemfc so that a
# bug in the project's own conversion cannot hide itself here.
NEM_OFFSET = timedelta(hours=10)


def _database_url() -> str:
    return (
        f"postgresql+psycopg://{os.getenv('POSTGRES_USER', 'nemfc')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'nemfc')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'nemfc')}"
    )


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(_database_url(), pool_pre_ping=True)
        with eng.connect() as conn:
            conn.execute(text("select 1 from staging.mart_spike_features limit 1"))
        return eng
    except Exception as exc:
        pytest.skip(f"no populated database to verify against: {exc}")


def _to_utc(series: pd.Series) -> pd.Series:
    """AEMO's naive local timestamps to UTC, computed here from scratch."""
    return pd.to_datetime(series, format="%Y/%m/%d %H:%M:%S") - NEM_OFFSET


@pytest.fixture(scope="module")
def sample(engine) -> pd.DataFrame:
    frame = pd.read_sql(
        text(
            """
            select prediction_at, anchor_settlement_end, rrp_now, rrp_lag_5m,
                   target_max_rrp, min_target_known_at, max_feature_known_at,
                   benchmark_run_at, target_window_start, target_window_end
            from staging.mart_spike_features
            order by md5(prediction_at::text)
            limit :n
            """
        ),
        engine,
        params={"n": SAMPLE_SIZE},
    )
    assert len(frame) > 0, "feature table is empty"
    for column in (
        "prediction_at",
        "anchor_settlement_end",
        "min_target_known_at",
        "max_feature_known_at",
        "benchmark_run_at",
        "target_window_start",
        "target_window_end",
    ):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


@pytest.fixture(scope="module")
def raw_prices(engine) -> pd.DataFrame:
    """Every dispatch price for the region, straight from raw."""
    frame = pd.read_sql(
        text(
            """
            select "SETTLEMENTDATE" as settlement_text, "RRP" as rrp_text,
                   _known_at as known_at
            from raw.dispatch_price_v5
            where "REGIONID" = :region and "INTERVENTION" = '0'
            """
        ),
        engine,
        params={"region": REGION},
    )
    frame["settlement_end"] = _to_utc(frame["settlement_text"]).dt.tz_localize("UTC")
    frame["rrp"] = frame["rrp_text"].astype(float)
    frame["known_at"] = pd.to_datetime(frame["known_at"], utc=True)
    return frame.set_index("settlement_end").sort_index()


def test_the_anchor_price_matches_raw(sample, raw_prices):
    """The price the features report is the price AEMO published."""
    merged = sample.join(raw_prices["rrp"].rename("raw_rrp"), on="anchor_settlement_end")
    assert merged["raw_rrp"].notna().all(), "an anchor interval is missing from raw"
    mismatched = merged[~merged["rrp_now"].sub(merged["raw_rrp"]).abs().lt(1e-9)]
    assert mismatched.empty, f"{len(mismatched)} anchors disagree with raw"


def test_the_five_minute_lag_is_the_previous_interval(sample, raw_prices):
    """A lag must be the actual previous interval, not merely the previous row.

    Positional lags bridge a missing interval silently. This checks the lag
    against the price stamped exactly five minutes earlier in raw.
    """
    wanted = sample["anchor_settlement_end"] - timedelta(minutes=5)
    expected = raw_prices["rrp"].reindex(wanted).to_numpy()

    both_present = sample["rrp_lag_5m"].notna() & pd.notna(expected)
    assert both_present.sum() > 0, "no comparable lags in the sample"

    diff = (sample["rrp_lag_5m"][both_present].to_numpy() - expected[both_present]).round(9)
    assert (diff == 0).all(), "a five-minute lag does not match the previous interval"

    # Where raw has no interval five minutes back, the feature must be null
    # rather than reaching further into the past.
    assert sample["rrp_lag_5m"][pd.isna(expected)].isna().all()


def test_no_target_interval_was_readable_at_prediction_time(sample, raw_prices):
    """The invariant PLAN §4.1 does not describe, recomputed from raw.

    For each sampled prediction, find every raw interval inside the target
    window and confirm none of them had been published yet.
    """
    violations = []
    for row in sample.itertuples():
        window = raw_prices[
            (raw_prices.index > row.anchor_settlement_end)
            & (raw_prices.index <= row.target_window_end)
        ]
        already_known = window[window["known_at"] <= row.prediction_at]
        if not already_known.empty:
            violations.append((row.prediction_at, already_known.index.tolist()))

    assert not violations, f"{len(violations)} predictions had a target already published"


def test_the_target_is_the_maximum_over_the_window(sample, raw_prices):
    """Recompute the label rather than trusting the mart's aggregate."""
    mismatched = []
    for row in sample.itertuples():
        window = raw_prices[
            (raw_prices.index > row.anchor_settlement_end)
            & (raw_prices.index <= row.target_window_end)
        ]
        if len(window) != 6:
            mismatched.append((row.prediction_at, f"{len(window)} intervals, expected 6"))
        elif abs(window["rrp"].max() - row.target_max_rrp) > 1e-9:
            mismatched.append((row.prediction_at, f"{window['rrp'].max()} vs {row.target_max_rrp}"))

    assert not mismatched, f"{len(mismatched)} targets disagree with raw: {mismatched[:3]}"


def test_the_benchmark_run_is_the_latest_one_readable(engine, sample):
    """The as-of join must pick AEMO's most recent readable forecast.

    Choosing a later run would hand the model a forecast nobody could have
    seen, on the single comparison the project is built around. Recomputed
    here from the raw pre-dispatch vintages.
    """
    runs = pd.read_sql(
        text(
            """
            select distinct "RUN_DATETIME" as run_text, _known_at as known_at
            from raw.p5min_regionsolution_v10
            where "REGIONID" = :region and "INTERVENTION" = '0'
            """
        ),
        engine,
        params={"region": REGION},
    )
    runs["run_at"] = _to_utc(runs["run_text"]).dt.tz_localize("UTC")
    runs["known_at"] = pd.to_datetime(runs["known_at"], utc=True)
    runs = runs.sort_values("known_at")

    checked = 0
    wrong = []
    for row in sample.itertuples():
        if pd.isna(row.benchmark_run_at):
            continue
        readable = runs[runs["known_at"] <= row.prediction_at]
        if readable.empty:
            continue
        latest = readable.sort_values(["known_at", "run_at"]).iloc[-1]["run_at"]
        checked += 1
        if latest != row.benchmark_run_at:
            wrong.append((row.prediction_at, row.benchmark_run_at, latest))

    assert checked > 0, "no benchmark runs were comparable"
    assert not wrong, f"{len(wrong)} predictions used the wrong forecast run: {wrong[:3]}"


def test_every_feature_source_predates_the_prediction(sample):
    """The feature-side invariant, over the sample."""
    late = sample[sample["max_feature_known_at"] > sample["prediction_at"]]
    assert late.empty, f"{len(late)} rows carry a feature published after prediction time"
