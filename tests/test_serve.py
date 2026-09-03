"""Tests for the API.

The interesting cases are the failures. A service that returns a plausible
number when the poller has stopped, or when the warehouse has drifted under the
model, is worse than one that errors -- the number looks like a forecast and is
not. So most of what follows checks that the bad paths announce themselves.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from nemfc import serve
from nemfc.model import ModelSpec, fit_fold

NOW = datetime(2026, 9, 2, 6, 0, tzinfo=timezone.utc)
FEATURES = ["rrp_now", "rolling_max_rrp_30m", "total_demand", "benchmark_max_forecast_rrp"]


def _training_frame(n: int = 4000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stamps = pd.date_range(NOW - timedelta(days=30), periods=n, freq="5min", tz="UTC")
    rrp = np.where(rng.random(n) < 0.02, rng.uniform(400, 15000, n), rng.normal(70, 20, n))
    frame = pd.DataFrame(
        {
            "prediction_at": stamps,
            "rrp_now": rrp,
            "rolling_max_rrp_30m": pd.Series(rrp).rolling(7, min_periods=7).max(),
            "total_demand": rng.normal(9000, 800, n),
            "benchmark_max_forecast_rrp": rrp * 0.6,
        }
    )
    frame["is_spike"] = pd.Series(rrp).shift(-1).rolling(6).max().shift(-5) > 300
    return frame.dropna().reset_index(drop=True)


def _live_row(age_minutes: float = 2.0) -> pd.DataFrame:
    prediction_at = NOW - timedelta(minutes=age_minutes)
    return pd.DataFrame(
        [
            {
                "region_id": "NSW1",
                "prediction_at": prediction_at,
                "anchor_settlement_end": prediction_at + timedelta(minutes=2),
                "target_window_start": prediction_at + timedelta(minutes=7),
                "target_window_end": prediction_at + timedelta(minutes=32),
                "rrp_now": 250.0,
                "rolling_max_rrp_30m": 260.0,
                "total_demand": 11000.0,
                "benchmark_max_forecast_rrp": 310.0,
                "age": pd.Timedelta(minutes=age_minutes),
            }
        ]
    )


@pytest.fixture
def deployed(tmp_path, monkeypatch):
    """A trained artifact on disk and a stubbed warehouse."""
    spec = ModelSpec("test", use_benchmark=True)
    monkeypatch.setattr(
        "nemfc.model.BASE_FEATURES", ("rrp_now", "rolling_max_rrp_30m", "total_demand")
    )
    monkeypatch.setattr("nemfc.model.BENCHMARK_FEATURES", ("benchmark_max_forecast_rrp",))
    monkeypatch.setattr("nemfc.model.CATEGORICAL_FEATURES", ())

    booster, _ = fit_fold(_training_frame(), spec)

    model_file = tmp_path / "spike_model.txt"
    meta_file = tmp_path / "spike_model.json"
    booster.save_model(str(model_file))
    meta_file.write_text(
        json.dumps(
            {
                "git_sha": "abc1234",
                "created_at": NOW.isoformat(),
                "trained_to": NOW.isoformat(),
                "features": FEATURES,
                "spike_threshold": 300.0,
                "horizon_minutes": 30,
                "out_of_sample_score": None,
                "scoring_note": "This artifact has no out-of-sample score.",
            }
        )
    )
    monkeypatch.setattr(serve, "MODEL_FILE", model_file)
    monkeypatch.setattr(serve, "METADATA_FILE", meta_file)
    monkeypatch.setattr(serve, "create_engine", lambda *a, **k: object())
    return tmp_path


def _client(monkeypatch, row: pd.DataFrame | None = None):
    frame = _live_row() if row is None else row
    monkeypatch.setattr(
        serve.pd,
        "read_sql",
        lambda sql, engine, **kw: frame.iloc[:0] if "limit 0" in str(sql) else frame,
    )
    return TestClient(serve.create_app())


def test_health_reports_ok_on_fresh_data(deployed, monkeypatch):
    with _client(monkeypatch) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["data_age_seconds"] == pytest.approx(120.0)
    assert body["model"]["git_sha"] == "abc1234"


def test_health_says_the_served_model_has_no_measured_score(deployed, monkeypatch):
    """The one claim nobody should be able to make by accident."""
    with _client(monkeypatch) as client:
        body = client.get("/health").json()
    assert body["model"]["out_of_sample_score"] is None
    assert "no out-of-sample score" in body["model"]["scoring_note"]


def test_health_reports_whether_the_aemo_forecast_reached_the_row(deployed, monkeypatch):
    """It went missing once, three models upstream, and nothing complained.

    A null benchmark costs the model its second-largest source of gain and
    changes nothing else about the response, so it has to be visible.
    """
    with _client(monkeypatch) as client:
        assert client.get("/health").json()["benchmark_present"] is True

    without = _live_row()
    without["benchmark_max_forecast_rrp"] = None
    with _client(monkeypatch, without) as client:
        assert client.get("/health").json()["benchmark_present"] is False


def test_a_stopped_poller_is_degraded_not_ok(deployed, monkeypatch):
    """A stale row scored silently looks exactly like a working forecaster."""
    with _client(monkeypatch, _live_row(age_minutes=90)) as client:
        body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert "poller" in body["reason"]


def test_the_forecast_carries_its_own_staleness(deployed, monkeypatch):
    with _client(monkeypatch, _live_row(age_minutes=90)) as client:
        body = client.get("/forecast/latest").json()
    assert body["stale"] is True
    assert body["data_age_seconds"] == pytest.approx(5400.0)


def test_the_forecast_returns_a_probability_and_its_provenance(deployed, monkeypatch):
    with _client(monkeypatch) as client:
        body = client.get("/forecast/latest").json()

    assert 0.0 <= body["probability_of_spike"] <= 1.0
    assert body["spike_threshold"] == 300.0
    assert body["horizon_minutes"] == 30
    assert body["rrp_now"] == 250.0
    assert body["aemo_forecast_max_rrp"] == 310.0
    assert body["model_git_sha"] == "abc1234"
    # The window predicted must start after the interval already published.
    assert body["target_window_start"] > body["anchor_settlement_end"]


def test_predict_accepts_explicit_features(deployed, monkeypatch):
    with _client(monkeypatch) as client:
        body = client.post(
            "/predict",
            json={"features": {"rrp_now": 5000.0, "rolling_max_rrp_30m": 5000.0}},
        ).json()

    assert 0.0 <= body["probability_of_spike"] <= 1.0
    assert body["features_supplied"] == 2
    assert body["features_expected"] == len(FEATURES)


def test_predict_names_the_features_it_ignored(deployed, monkeypatch):
    """Silently dropping an unknown key would hide a caller's typo."""
    with _client(monkeypatch) as client:
        body = client.post("/predict", json={"features": {"rrp_nwo": 5000.0}}).json()
    assert body["features_ignored"] == ["rrp_nwo"]
    assert body["features_supplied"] == 0


def test_a_missing_model_is_unhealthy_rather_than_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "MODEL_FILE", tmp_path / "absent.txt")
    monkeypatch.setattr(serve, "METADATA_FILE", tmp_path / "absent.json")

    with TestClient(serve.create_app()) as client:
        body = client.get("/health").json()
        assert body["status"] == "unhealthy"
        assert "make fit" in body["error"]

        assert client.get("/forecast/latest").status_code == 503
        assert client.post("/predict", json={"features": {}}).status_code == 503


def test_warehouse_drift_is_caught_at_startup_not_on_the_first_request(deployed, monkeypatch):
    """The Day 6 failure, in the shape it would take inside a service.

    There, a disagreement between the feature list and the table killed a
    backtest with a stack trace. Here it would return a number instead, which
    is worse, so it has to fail before the service reports itself ready.
    """
    thin = _live_row().drop(columns=["benchmark_max_forecast_rrp"])
    monkeypatch.setattr(
        serve.pd,
        "read_sql",
        lambda sql, engine, **kw: thin.iloc[:0] if "limit 0" in str(sql) else thin,
    )

    with TestClient(serve.create_app()) as client:
        body = client.get("/health").json()

    assert body["status"] == "unhealthy"
    assert "benchmark_max_forecast_rrp" in body["error"]
    assert "drifted" in body["error"]
