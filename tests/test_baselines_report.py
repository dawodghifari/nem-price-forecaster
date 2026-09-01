"""End-to-end exercise of `scripts/baselines.py` on synthetic data.

The database lives on a machine this suite cannot reach, so the script's
assembly -- episode labelling, fold construction, scoring, report rendering --
is run here against a frame with the same shape and the same column names.
This is the same reason the SQL is parsed offline: a runtime error found in the
sandbox costs a minute, and one found mid-run costs the run.

It also pins the two controls. A random score must land at the base rate and
perfect foresight must reach precision 1.0; if the harness cannot reproduce
that on data whose answer is known, nothing it says about real data counts.
"""

from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nemfc.evaluate import DEFAULT_K, score
from nemfc.folds import describe, make_folds

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "baselines.py"
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
THRESHOLD = 300.0


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("baselines_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _features(days: int = 200) -> pd.DataFrame:
    """A market shaped like the real one: rare evening spikes, in short runs."""
    rng = np.random.default_rng(4)
    rows = []
    for d in range(days):
        day = BASE + timedelta(days=d)
        for i in range(48):  # half-hourly, enough to exercise the machinery
            stamp = day + timedelta(minutes=30 * i)
            local_hour = (stamp + timedelta(hours=10)).hour
            spiking = (d % 4 == 0) and local_hour in (18, 19)
            rrp = 4000.0 if spiking else float(rng.normal(70, 15))
            rows.append(
                {
                    "prediction_at": stamp,
                    "anchor_settlement_end": stamp,
                    "region_id": "NSW1",
                    "rrp_now": rrp,
                    "target_max_rrp": np.nan,
                    "rolling_max_rrp_30m": rrp,
                    "benchmark_max_forecast_rrp": rrp * 0.6 if spiking else rrp,
                    "benchmark_intervals": 6,
                    "nem_hour": local_hour,
                    "nem_day_of_week": (stamp.weekday() + 1) % 7,
                    "nem_trading_date": date(day.year, day.month, day.day),
                }
            )
    frame = pd.DataFrame(rows)

    # Target: the max price over the next hour, at this fixture's resolution.
    future = frame["rrp_now"].rolling(2).max().shift(-2)
    frame["target_max_rrp"] = future
    frame["is_spike"] = (future > THRESHOLD).fillna(False)
    return frame.dropna(subset=["target_max_rrp"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def prepared(module):
    frame, n_episodes = module.add_episodes(_features(), THRESHOLD)
    folds = make_folds(frame, min_train_days=90, min_spiking_days=5)
    return frame, folds, n_episodes


def test_episodes_are_found_and_are_fewer_than_the_positive_rows(prepared):
    """The overlap the whole module exists to handle, on realistic shape."""
    frame, _, n_episodes = prepared
    assert n_episodes > 0
    assert n_episodes < int(frame["is_spike"].sum())


def test_the_folds_all_carry_events(prepared):
    _, folds, _ = prepared
    assert len(folds) >= 2
    assert all(f.n_test_positive_rows > 0 for f in folds)


def test_the_random_control_lands_at_the_base_rate(module, prepared):
    frame, folds, _ = prepared
    scores = module.build_scores(frame, folds, shape=None)
    in_scope = scores["Climatology"].notna()

    evaluated = frame[in_scope].copy()
    evaluated["Random (floor)"] = scores["Random (floor)"][in_scope]

    card = score(evaluated, "Random (floor)", "Random (floor)")
    assert card.pr_auc == pytest.approx(card.base_rate, abs=0.02)


def test_perfect_foresight_reaches_precision_one(module, prepared):
    frame, folds, _ = prepared
    scores = module.build_scores(frame, folds, shape=None)
    in_scope = scores["Climatology"].notna()

    evaluated = frame[in_scope].copy()
    evaluated["Perfect foresight (ceiling)"] = scores["Perfect foresight (ceiling)"][in_scope]

    card = score(evaluated, "Perfect foresight (ceiling)", "Perfect foresight (ceiling)")
    assert card.precision_at_k[DEFAULT_K[0]] == pytest.approx(1.0)


def test_climatology_beats_random_when_spikes_have_a_time_of_day(module, prepared):
    """The floor is only a floor if something clears it."""
    frame, folds, _ = prepared
    scores = module.build_scores(frame, folds, shape=None)
    in_scope = scores["Climatology"].notna()

    evaluated = frame[in_scope].copy()
    for name in ("Climatology", "Random (floor)"):
        evaluated[name] = scores[name][in_scope]

    clim = score(evaluated, "Climatology", "Climatology")
    rand = score(evaluated, "Random (floor)", "Random (floor)")
    assert clim.pr_auc > rand.pr_auc


def test_the_report_renders_every_section(module, prepared, tmp_path):
    frame, folds, _ = prepared
    scores = module.build_scores(frame, folds, shape=None)
    in_scope = scores["Climatology"].notna()

    evaluated = frame[in_scope].copy()
    for name, series in scores.items():
        evaluated[name] = series[in_scope]

    cards = [score(evaluated, name, name) for name in scores]
    cis = {c.name: (0.1, 0.2) for c in cards}
    points = {"AEMO P5MIN": module.operating_point(evaluated, evaluated["AEMO P5MIN"], THRESHOLD)}
    coverage = dict.fromkeys(
        [
            "total_rows",
            "scored_rows",
            "positive_rows",
            "episodes",
            "days_with_spikes",
            "has_benchmark",
            "positive_missing_benchmark",
            "has_persistence",
        ],
        1,
    )

    out = tmp_path / "day5.md"
    module.write_report(out, cards, points, describe(folds), coverage, cis, THRESHOLD)

    text = out.read_text()
    assert "## The comparison set" in text
    assert "## Results" in text
    assert "## Fold structure" in text
    assert "Perfect foresight (ceiling)" in text
    for k in DEFAULT_K:
        assert f"episode recall@{k}" in text
    assert "nan" not in text.lower().replace("n/a", "")


def test_the_report_needs_no_optional_dependency(module):
    """`to_markdown` would have required tabulate, which is not installed."""
    table = pd.DataFrame({"a": [1000], "b": [pd.Timestamp("2026-01-01", tz="UTC")]})
    rendered = module._markdown_table(table)
    assert "1,000" in rendered
    assert "2026-01-01 00:00" in rendered
