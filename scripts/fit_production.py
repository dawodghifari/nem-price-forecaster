#!/usr/bin/env python3
"""Fit the model that gets served, and write down exactly what it is.

This is a different object from the seven models the backtest trains. Those
each see a slice of history and are thrown away; this one sees everything up to
now, because a served model should use every observation available to it.

That has a consequence worth being blunt about, and the metadata says so in
words rather than leaving it to be inferred: **this model has no
out-of-sample score of its own.** It was trained through the period the backtest
evaluated on, so quoting `docs/day6_model.md`'s numbers as if this artifact had
earned them would be wrong. Those numbers describe the *procedure* -- these
features, these parameters, this fold structure -- and the claim they support is
that the procedure generalises, not that this particular file does.

The sidecar records the feature list too, so the API can refuse to start when
the warehouse has drifted underneath it rather than serving nonsense on the
first request.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemfc.config import load_settings
from nemfc.dataset import load
from nemfc.model import DEFAULT_PARAMS, ModelSpec, fit_fold

DEFAULT_DIR = Path("models")
MODEL_FILE = "spike_model.txt"
METADATA_FILE = "spike_model.json"


def git_sha(repo: Path) -> str:
    """The commit this artifact came from, or a marker that it is unknown."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Exclude AEMO's forecast from the features. The served default "
        "includes it: it scored better on both slices, and at serving time it "
        "is available for free.",
    )
    args = parser.parse_args()

    settings = load_settings()
    engine = create_engine(settings.database_url, pool_pre_ping=True)

    features, _ = load(engine, settings.region)
    if features.empty:
        print("feature table is empty", file=sys.stderr)
        return 1

    spec = ModelSpec("production", use_benchmark=not args.no_benchmark)
    booster, report = fit_fold(features, spec)

    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    booster.save_model(str(out_dir / MODEL_FILE), num_iteration=booster.best_iteration or None)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(repo),
        "region": settings.region,
        "spike_threshold": settings.spike_threshold,
        "horizon_minutes": 30,
        "features": spec.features(),
        "params": DEFAULT_PARAMS,
        "num_boost_round": spec.num_boost_round,
        "best_iteration": report.best_iteration,
        "early_stopped": report.early_stopped,
        "trained_rows": len(features),
        "trained_positives": int(features["is_spike"].sum()),
        "trained_from": features["prediction_at"].min().isoformat(),
        "trained_to": features["prediction_at"].max().isoformat(),
        "validation_rows": report.n_valid,
        "validation_positives": report.n_valid_positives,
        "out_of_sample_score": None,
        "scoring_note": (
            "This artifact has no out-of-sample score. It was trained through "
            "the period docs/day6_model.md evaluates on. Those figures measure "
            "the procedure -- these features, these parameters, this fold "
            "structure -- not this file. Do not quote them as this model's "
            "performance."
        ),
    }
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out_dir / MODEL_FILE}")
    print(f"wrote {out_dir / METADATA_FILE}")
    print(
        f"trained on {metadata['trained_rows']:,} rows "
        f"({metadata['trained_positives']} positive) "
        f"through {metadata['trained_to']}, {len(spec.features())} features"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
