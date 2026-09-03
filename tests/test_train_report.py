"""End-to-end exercise of `scripts/train.py` on synthetic data.

Same reason as the Day 5 equivalent: the database is on a machine this suite
cannot reach, and a runtime error found here costs a minute where one found
mid-backtest costs the run. Seven folds of LightGBM is not a cheap thing to
restart because a report template had a typo in it.
"""

from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from nemfc.evaluate import onset_mask, score
from nemfc.folds import describe, make_folds
from nemfc.model import BASE_FEATURES, BENCHMARK_FEATURES, ModelSpec
from nemfc.model import predict_out_of_fold as _predict

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
BASE = datetime(2025, 9, 1, tzinfo=timezone.utc)
THRESHOLD = 300.0


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("train_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frame(days: int = 150, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        day = BASE + timedelta(days=d)
        tight = rng.random() < 0.3
        for i in range(48):
            stamp = day + timedelta(minutes=30 * i)
            hour = (stamp + timedelta(hours=10)).hour
            spiking = tight and hour in (17, 18, 19) and rng.random() < 0.4
            rrp = float(rng.uniform(400, 15000)) if spiking else float(rng.normal(70, 20))
            rows.append(
                {
                    "prediction_at": stamp,
                    "anchor_settlement_end": stamp,
                    "region_id": "NSW1",
                    "nem_trading_date": date(day.year, day.month, day.day),
                    "rrp_now": rrp,
                    "nem_hour": hour,
                    "nem_day_of_week": (stamp.weekday() + 1) % 7,
                    "nem_month": stamp.month,
                    "local_generation_balance": -800.0 if tight else 2500.0,
                    "benchmark_max_forecast_rrp": rrp * 0.5 if spiking else rrp,
                    "benchmark_intervals": 6,
                }
            )
    frame = pd.DataFrame(rows)
    future = frame["rrp_now"].shift(-1).rolling(2).max().shift(-1)
    frame["target_max_rrp"] = future
    frame["is_spike"] = (future > THRESHOLD).fillna(False)
    frame["rolling_max_rrp_30m"] = frame["rrp_now"].rolling(2, min_periods=2).max()

    for column in BASE_FEATURES:
        if column not in frame.columns:
            frame[column] = rng.normal(size=len(frame))
    for column in BENCHMARK_FEATURES:
        if column not in frame.columns:
            frame[column] = frame["benchmark_max_forecast_rrp"] > THRESHOLD

    return frame.dropna(subset=["target_max_rrp"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def scored(module):
    frame, _ = module.add_episodes(_frame(), THRESHOLD)
    folds = make_folds(frame, min_train_days=45, min_train_spiking_days=5, min_spiking_days=4)

    predictions, reports = _predict(frame, folds, ModelSpec("Model (+ AEMO feature)"))
    in_scope = predictions.notna()

    evaluated = frame[in_scope].copy()
    evaluated["Model (+ AEMO feature)"] = predictions[in_scope]
    evaluated["Persistence"] = evaluated["rolling_max_rrp_30m"]
    return evaluated, folds, reports


def test_the_onset_slice_survives_the_pipeline(scored):
    """It must be non-empty and hold events, or the headline metric is empty."""
    evaluated, _, _ = scored
    onset = evaluated[onset_mask(evaluated, THRESHOLD)]
    assert len(onset) > 0
    assert int(onset["is_spike"].sum()) > 0
    assert onset["is_spike"].mean() < evaluated["is_spike"].mean()


def test_the_results_table_renders_for_both_slices(module, scored):
    evaluated, _, _ = scored
    columns = ["Model (+ AEMO feature)", "Persistence"]

    full_cards, full_cis = module.score_everything(evaluated, columns, 20)
    onset = evaluated[onset_mask(evaluated, THRESHOLD)].copy()
    onset_cards, onset_cis = module.score_everything(onset, columns, 20)

    for cards, cis in ((full_cards, full_cis), (onset_cards, onset_cis)):
        rendered = "\n".join(module.results_table(cards, cis))
        assert "PR-AUC" in rendered
        assert "Model (+ AEMO feature)" in rendered


def test_the_report_renders_every_section(module, scored, tmp_path):
    evaluated, folds, reports = scored
    columns = ["Model (+ AEMO feature)", "Persistence"]
    cards, cis = module.score_everything(evaluated, columns, 20)
    fold_detail = pd.DataFrame([r.__dict__ for r in reports]).drop(columns=["importance"])

    out = tmp_path / "day6.md"
    module.write_report(
        out,
        {
            "full_rows": len(evaluated),
            "full_positives": int(evaluated["is_spike"].sum()),
            "full_episodes": 10,
            "full_days": 5,
            "full_table": module.results_table(cards, cis),
            "onset_rows": 1,
            "onset_positives": 1,
            "onset_episodes": 1,
            "onset_days": 1,
            "onset_table": module.results_table(cards, cis),
            "importance_without": module._table(
                pd.DataFrame({"feature": ["rrp_now"], "mean_gain": [1.0]})
            ),
            "importance_with": module._table(
                pd.DataFrame({"feature": ["benchmark_max_forecast_rrp"], "mean_gain": [1.0]})
            ),
            "folds": module._table(describe(folds)),
            "fold_detail": module._table(fold_detail),
        },
    )

    text = out.read_text()
    for heading in (
        "## Setup",
        "## The full evaluation set",
        "## The onset slice",
        "## Feature importance",
        "### Without the AEMO feature",
        "### With the AEMO feature",
        "## Fold structure",
        "## Per-fold training detail",
    ):
        assert heading in text
    assert "learning_rate" in text, "the parameters must be in the report, not just the code"


def test_mlflow_logging_never_breaks_a_run(module, scored, tmp_path, capsys):
    """A tracking failure is a lost record, not a lost result.

    Asserted in the branch that applies, rather than with an `or True` that
    would pass either way. An assertion that cannot fail is not a test.
    """
    evaluated, _, _ = scored
    cards, _ = module.score_everything(evaluated, ["Persistence"], 20)

    module.log_to_mlflow(cards, cards, tmp_path)
    printed = capsys.readouterr().out.lower()

    if importlib.util.find_spec("mlflow") is None:
        assert "mlflow not installed" in printed
        assert not (tmp_path / "mlruns").exists()
    else:
        assert (tmp_path / "mlruns").exists()


def test_mlflow_logging_swallows_a_broken_tracking_store(module, scored, monkeypatch, capsys):
    """The failure path, forced, because the happy path does not exercise it."""
    pytest.importorskip("mlflow")
    import mlflow

    evaluated, _, _ = scored
    cards, _ = module.score_everything(evaluated, ["Persistence"], 20)

    def explode(*_args, **_kwargs):
        raise RuntimeError("tracking server is down")

    monkeypatch.setattr(mlflow, "set_tracking_uri", explode)
    module.log_to_mlflow(cards, cards, Path("/tmp"))  # must not raise
    assert "mlflow logging failed" in capsys.readouterr().err.lower()


def test_the_shuffle_control_would_catch_a_leaking_harness(module, scored):
    """The control is only worth running if it can fail.

    Scoring a permuted label column against real predictions has to land at the
    base rate; if this passed regardless, the control in the script would be
    decoration.
    """
    evaluated, _, _ = scored
    rng = np.random.default_rng(2)
    shuffled = evaluated.assign(is_spike=rng.permutation(evaluated["is_spike"].to_numpy()))

    card = score(shuffled, "Model (+ AEMO feature)", "shuffled")
    assert card.pr_auc < card.base_rate * 3
