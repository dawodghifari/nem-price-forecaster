"""The API. Scores a warehouse row rather than rebuilding one.

The usual way a project like this fails after a good backtest is train/serve
skew: features recomputed in Python that drift from the SQL the model was
trained on. Nothing here recomputes anything. `mart_live_features` is the same
feature definition the training mart uses -- one macro, two callers -- so
`/forecast/latest` reads a row and hands it to the booster.

Day 4's decision to anchor predictions to publication events is what makes that
possible. A prediction happens when a new dispatch price appears, in training
and in production alike, so "the newest row" is a well-formed request rather
than a clock-time approximation.

Three things the service refuses to do quietly:

- **Start with a stale feature list.** The model's features are compared against
  the live mart's columns at startup. A column renamed in dbt fails the health
  check instead of producing predictions from a silently wrong matrix.
- **Serve an old row without saying so.** Every response carries the age of the
  data it used. A stopped poller looks like a healthy API otherwise.
- **Drop a feature quietly.** `/health` reports whether AEMO's forecast reached
  the served row. It went missing once already, through a materialisation
  choice three models upstream, and the prediction that came back looked
  entirely reasonable.
- **Imply the served model has a measured score.** `/health` reports the
  metadata's scoring note verbatim.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from nemfc.config import load_settings
from nemfc.model import prepare

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_FILE = MODEL_DIR / "spike_model.txt"
METADATA_FILE = MODEL_DIR / "spike_model.json"

#: Beyond this the newest row is not a live forecast. The poller runs every five
#: minutes, so fifteen allows two missed passes before the service says so.
STALE_AFTER = timedelta(minutes=15)

#: Every connection this service opens gives up after ten seconds.
#:
#: Not a nicety. Without it, a slow query on the live mart ran for ninety
#: minutes; the container healthcheck fired another every thirty seconds; and
#: eighteen copies accumulated, saturated the buffer pool, and starved the
#: database badly enough that an unrelated `dbt build` took three hours and
#: forty-five minutes. Nothing reported an error at any point -- the API simply
#: never answered, and the symptom appeared somewhere else entirely.
#:
#: A serving query that cannot finish in ten seconds is not going to become
#: useful at sixty. Failing fast turns an invisible pile-up into a visible 503.
STATEMENT_TIMEOUT_MS = 10_000

#: Ordered here, not only inside the view: Postgres does not promise to keep a
#: view's ORDER BY through an outer LIMIT.
LATEST_SQL = "select * from staging.mart_live_features order by prediction_at desc limit 1"
COLUMNS_SQL = "select * from staging.mart_live_features limit 0"


@dataclass
class State:
    """Everything loaded once at startup."""

    booster: Any = None
    metadata: dict = field(default_factory=dict)
    engine: Any = None
    startup_error: str | None = None


state = State()


def load_model() -> tuple[Any, dict]:
    import lightgbm as lgb

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"no model at {MODEL_FILE}. Run `make fit` to train and persist one."
        )
    booster = lgb.Booster(model_file=str(MODEL_FILE))
    metadata = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else {}
    return booster, metadata


def check_features_against_warehouse(engine, features: list[str]) -> None:
    """Fail at startup, not on the first request.

    The Day 6 backtest died mid-run because the loader and the feature list
    disagreed. The same disagreement in a service would return predictions
    instead of an error, which is worse.
    """
    available = set(pd.read_sql(text(COLUMNS_SQL), engine).columns)
    missing = sorted(set(features) - available)
    if missing:
        raise RuntimeError(
            f"the model needs {len(missing)} columns the live mart does not have: "
            f"{missing}. The warehouse and the model artifact have drifted; "
            "rebuild dbt or refit the model."
        )


@asynccontextmanager
async def lifespan(app):
    try:
        state.booster, state.metadata = load_model()
        settings = load_settings()
        state.engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            # Bound every statement, and cap the pool so a slow query cannot be
            # answered by opening more connections.
            pool_size=4,
            max_overflow=0,
            connect_args={"options": f"-c statement_timeout={STATEMENT_TIMEOUT_MS}"},
        )
        check_features_against_warehouse(state.engine, state.metadata.get("features", []))
        state.startup_error = None
    except Exception as exc:
        state.startup_error = str(exc)
    yield


class Features(BaseModel):
    """Body of a `/predict` request.

    Declared at module scope, not inside `create_app`. This file uses postponed
    annotation evaluation, so FastAPI resolves `payload: Features` by looking
    the name up in the module's globals -- a class local to the factory is
    invisible there, and the body silently becomes a required query parameter.
    The symptom is a 422 on every well-formed request.
    """

    features: dict[str, float | None] = Field(
        description="Feature name to value. Missing features are sent as null "
        "and routed by the model, which is what a short rolling window "
        "produces in training too."
    )


def create_app():
    app = FastAPI(
        title="NEM price spike forecaster",
        description=(
            "Probability that the NSW spot price exceeds the threshold in the "
            "next 30 minutes, from AEMO's live five-minute dispatch feed."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    def _score(frame: pd.DataFrame) -> float:
        matrix = prepare(frame, state.metadata["features"])
        return float(state.booster.predict(matrix)[0])

    def _require_ready() -> None:
        if state.startup_error:
            raise HTTPException(503, f"service is not ready: {state.startup_error}")

    @app.get("/health")
    def health() -> dict:
        """Readiness, data freshness, and what the served model actually is."""
        if state.startup_error:
            return {"status": "unhealthy", "error": state.startup_error}

        try:
            latest = pd.read_sql(text(LATEST_SQL), state.engine)
        except Exception as exc:
            return {"status": "unhealthy", "error": f"database unreachable: {exc}"}

        if latest.empty:
            return {"status": "degraded", "reason": "no rows in the live feature mart"}

        age = pd.to_timedelta(latest["age"].iloc[0])
        fresh = age <= STALE_AFTER
        return {
            "status": "ok" if fresh else "degraded",
            "reason": None if fresh else "the newest feature row is stale; is the poller running?",
            "data_age_seconds": round(age.total_seconds(), 1),
            "latest_prediction_at": latest["prediction_at"].iloc[0].isoformat(),
            "benchmark_present": bool(pd.notna(latest["benchmark_max_forecast_rrp"].iloc[0])),
            "model": {
                "git_sha": state.metadata.get("git_sha"),
                "created_at": state.metadata.get("created_at"),
                "trained_to": state.metadata.get("trained_to"),
                "n_features": len(state.metadata.get("features", [])),
                "spike_threshold": state.metadata.get("spike_threshold"),
                "out_of_sample_score": state.metadata.get("out_of_sample_score"),
                "scoring_note": state.metadata.get("scoring_note"),
            },
        }

    @app.get("/forecast/latest")
    def forecast_latest() -> dict:
        """Score the newest published dispatch interval."""
        _require_ready()

        latest = pd.read_sql(text(LATEST_SQL), state.engine)
        if latest.empty:
            raise HTTPException(503, "no rows in the live feature mart")

        age = pd.to_timedelta(latest["age"].iloc[0])
        probability = _score(latest)

        return {
            "region": latest["region_id"].iloc[0],
            "probability_of_spike": round(probability, 6),
            "spike_threshold": state.metadata.get("spike_threshold"),
            "horizon_minutes": state.metadata.get("horizon_minutes", 30),
            "prediction_at": latest["prediction_at"].iloc[0].isoformat(),
            "anchor_settlement_end": latest["anchor_settlement_end"].iloc[0].isoformat(),
            "target_window_start": latest["target_window_start"].iloc[0].isoformat(),
            "target_window_end": latest["target_window_end"].iloc[0].isoformat(),
            "rrp_now": float(latest["rrp_now"].iloc[0]),
            "aemo_forecast_max_rrp": (
                None
                if pd.isna(latest["benchmark_max_forecast_rrp"].iloc[0])
                else float(latest["benchmark_max_forecast_rrp"].iloc[0])
            ),
            "data_age_seconds": round(age.total_seconds(), 1),
            "stale": bool(age > STALE_AFTER),
            "model_git_sha": state.metadata.get("git_sha"),
        }

    @app.post("/predict")
    def predict(payload: Features) -> dict:
        """Score explicit features.

        The escape hatch, for testing and for callers holding their own feature
        values. `/forecast/latest` is the path that cannot skew, because it
        reads the same table training did.
        """
        _require_ready()

        wanted = state.metadata["features"]
        frame = pd.DataFrame([{name: payload.features.get(name) for name in wanted}])
        unknown = sorted(set(payload.features) - set(wanted))
        supplied = sorted(set(payload.features) & set(wanted))

        return {
            "probability_of_spike": round(_score(frame), 6),
            "spike_threshold": state.metadata.get("spike_threshold"),
            "horizon_minutes": state.metadata.get("horizon_minutes", 30),
            "features_supplied": len(supplied),
            "features_expected": len(wanted),
            "features_ignored": unknown,
            "model_git_sha": state.metadata.get("git_sha"),
        }

    return app


app = create_app()
