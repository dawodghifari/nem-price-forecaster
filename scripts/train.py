#!/usr/bin/env python3
"""Day 6: train the model, backtest it, and write down what it scored.

Output goes to `docs/day6_model.md`, generated rather than hand-edited.

Everything is scored twice. Once on the full evaluation set, which is what
PLAN §2.3 asks for, and once on the **onset slice** -- rows where nothing in the
trailing thirty minutes had already spiked. Day 5 measured persistence at 84x
lift, because episodes run 3.3 intervals and a high price stays high, so a model
can post a strong overall number having learned only that. The onset slice was
defined and committed before this script existed, precisely so it could not be
chosen afterwards as the cut where the model happens to look good.

Two models rather than one. PLAN §2.1 never resolves that AEMO's forecast is
both a feature and the benchmark, so it is run both ways: without it, the
question is whether pre-dispatch can be replaced; with it, whether it can be
improved on.

Three controls run alongside, and the script exits non-zero if any fails:

    random              must score at the base rate
    perfect foresight   must reach precision 1.0
    shuffled labels     a full backtest on permuted labels, which must also
                        score at the base rate

The third is the expensive one and the one worth having. It exercises the
entire training path -- folds, purge, inner validation, prediction -- against
data with no signal in it. If that comes back above chance, the harness leaks,
and every other number here is worthless.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemfc.baselines import (
    aemo_score,
    climatology_out_of_fold,
    long_history_shape,
    persistence_score,
)
from nemfc.config import load_settings
from nemfc.dataset import add_episodes, load
from nemfc.evaluate import (
    DEFAULT_K,
    NO_EPISODE,
    ScoreCard,
    blocked_bootstrap_pr_auc,
    onset_mask,
    score,
)
from nemfc.folds import (
    DEFAULT_MIN_SPIKING_DAYS,
    DEFAULT_MIN_TRAIN_DAYS,
    DEFAULT_MIN_TRAIN_SPIKING_DAYS,
    describe,
    make_folds,
)
from nemfc.model import (
    DEFAULT_PARAMS,
    ModelSpec,
    aggregate_importance,
    predict_out_of_fold,
)

MODEL_WITHOUT = "Model (no AEMO feature)"
MODEL_WITH = "Model (+ AEMO feature)"
SHUFFLED = "Shuffled labels (leak control)"
RANDOM = "Random (floor)"
PERFECT = "Perfect foresight (ceiling)"


def _fmt(value: float, places: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{places}f}"


def _table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(str(c) for c in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, pd.Timestamp):
                cells.append(f"{value:%Y-%m-%d %H:%M}")
            elif isinstance(value, float):
                cells.append(f"{value:,.4f}")
            elif isinstance(value, int | np.integer):
                cells.append(f"{value:,}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def results_table(cards: list[ScoreCard], cis: dict[str, tuple[float, float]]) -> list[str]:
    header = "| Score | PR-AUC | 95% CI |"
    divider = "|---|---|---|"
    for k in DEFAULT_K:
        header += f" p@{k} | lift@{k} | ep.recall@{k} |"
        divider += "---|---|---|"

    lines = [header, divider]
    for card in cards:
        low, high = cis.get(card.name, (float("nan"), float("nan")))
        row = f"| {card.name} | {_fmt(card.pr_auc)} | {_fmt(low)} to {_fmt(high)} |"
        for k in DEFAULT_K:
            if k in card.precision_at_k:
                row += (
                    f" {_fmt(card.precision_at_k[k])} |"
                    f" {_fmt(card.lift_at_k(k), 1)}x |"
                    f" {_fmt(card.episode_recall_at_k[k])} |"
                )
            else:
                row += " n/a | n/a | n/a |"
        lines.append(row)
    return lines


def score_everything(
    frame: pd.DataFrame, columns: list[str], resamples: int
) -> tuple[list[ScoreCard], dict[str, tuple[float, float]]]:
    cards, cis = [], {}
    for name in columns:
        cards.append(score(frame, name, name))
        cis[name] = blocked_bootstrap_pr_auc(frame, name, n_resamples=resamples)
    return cards, cis


def log_to_mlflow(cards: list[ScoreCard], onset_cards: list[ScoreCard], repo: Path) -> None:
    """Record the run, if MLflow is installed. Never fatal.

    PLAN's Day 6 asks for MLflow logging, and the registry with its promotion
    gate is Phase 3. What is wrong here is letting either one abort a backtest:
    a tracking failure is a lost record, not a lost result, and the generated
    report already carries every number. So this is best-effort against a local
    file store, and a missing MLflow degrades to a line of output.
    """
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed: skipping experiment logging (pip install -r ml.txt)")
        return

    try:
        mlflow.set_tracking_uri(f"file://{repo / 'mlruns'}")
        mlflow.set_experiment("nem-spike-forecaster")
        for card in cards:
            onset = next((c for c in onset_cards if c.name == card.name), None)
            with mlflow.start_run(run_name=card.name):
                mlflow.log_params({**DEFAULT_PARAMS, "score": card.name})
                mlflow.log_metrics(
                    {
                        "pr_auc": card.pr_auc,
                        "base_rate": card.base_rate,
                        "n_episodes": card.n_episodes,
                        **{f"precision_at_{k}": v for k, v in card.precision_at_k.items()},
                        **{
                            f"episode_recall_at_{k}": v for k, v in card.episode_recall_at_k.items()
                        },
                        **({"onset_pr_auc": onset.pr_auc} if onset else {}),
                    }
                )
    except Exception as exc:
        print(f"mlflow logging failed, continuing: {exc}", file=sys.stderr)


def write_report(path: Path, sections: dict[str, object]) -> None:
    lines = [
        "# Day 6: model and backtest",
        "",
        f"Generated {datetime.now(timezone.utc):%d %b %Y %H:%M} UTC by "
        "`scripts/train.py`. Not hand-edited.",
        "",
        "## Setup",
        "",
        "LightGBM, expanding-window walk-forward, folds sized by spiking days,",
        "24-hour purge between train and test, early stopping on a purged tail of",
        "each fold's own training window. No hyperparameter search: against this",
        "many events a search tunes to the fold structure and the reported score",
        "stops being out-of-sample in the sense it claims. Training is unweighted,",
        "so the output probabilities stay calibrated.",
        "",
        "```",
        *[f"{k}: {v}" for k, v in sorted(DEFAULT_PARAMS.items())],
        "```",
        "",
        "## The full evaluation set",
        "",
        f"{sections['full_rows']:,} rows, {sections['full_positives']:,} positive, "
        f"{sections['full_episodes']} episodes across {sections['full_days']} days.",
        "",
        *sections["full_table"],
        "",
        "## The onset slice",
        "",
        "Rows where nothing in the trailing thirty minutes had already exceeded the",
        "threshold, so persistence is silent by construction. Day 5 measured",
        "persistence at 84x lift because most positives are continuations of an",
        "episode already in progress; this slice keeps only the forecasts that are",
        "genuinely forecasts. It was defined before this script was written.",
        "",
        f"{sections['onset_rows']:,} rows, {sections['onset_positives']:,} positive, "
        f"{sections['onset_episodes']} episodes across {sections['onset_days']} days.",
        "",
        *sections["onset_table"],
        "",
        "## Feature importance",
        "",
        "Mean gain across folds, as a share of the total. One fold's ranking is",
        "close to arbitrary at this event count; only the average is worth reading.",
        "",
        "Both models are shown. The one without AEMO's forecast is the more",
        "informative of the two: it has to find the shortage some other way, and",
        "what it reaches for is the answer to what pre-dispatch is actually",
        "contributing.",
        "",
        "### Without the AEMO feature",
        "",
        sections["importance_without"],
        "",
        "### With the AEMO feature",
        "",
        sections["importance_with"],
        "",
        "## Fold structure",
        "",
        sections["folds"],
        "",
        "## Per-fold training detail",
        "",
        "`early_stopped` false means the validation tail held too few positives to",
        "stop on, and the fold ran the full round count rather than stopping on noise.",
        "",
        sections["fold_detail"],
        "",
    ]
    path.write_text("\n".join(str(line) for line in lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-spiking-days", type=int, default=DEFAULT_MIN_SPIKING_DAYS)
    parser.add_argument("--min-train-days", type=int, default=DEFAULT_MIN_TRAIN_DAYS)
    parser.add_argument(
        "--min-train-spiking-days", type=int, default=DEFAULT_MIN_TRAIN_SPIKING_DAYS
    )
    # Same count as scripts/baselines.py. With the same seed and rows but 500
    # draws here and 1,000 there, the two reports disagreed in the third decimal
    # on AEMO's interval (0.222-0.484 against 0.219-0.483).
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("docs/day6_model.md"))
    parser.add_argument(
        "--skip-shuffle-control",
        action="store_true",
        help="Skip the shuffled-label backtest. Halves the runtime, and removes "
        "the only end-to-end check that the training path does not leak.",
    )
    args = parser.parse_args()

    settings = load_settings()
    engine = create_engine(settings.database_url, pool_pre_ping=True)

    features, history = load(engine, settings.region)
    if features.empty:
        print("feature table is empty", file=sys.stderr)
        return 1

    features, n_episodes_total = add_episodes(features, settings.spike_threshold)
    shape = long_history_shape(history) if not history.empty else None

    folds = make_folds(
        features,
        min_spiking_days=args.min_spiking_days,
        min_train_days=args.min_train_days,
        min_train_spiking_days=args.min_train_spiking_days,
    )
    print(f"{len(folds)} folds, {n_episodes_total} episodes in the full period")

    rng = np.random.default_rng(0)
    scores: dict[str, pd.Series] = {
        "Climatology": climatology_out_of_fold(features, folds, shape),
        "Persistence": persistence_score(features),
        "AEMO P5MIN": aemo_score(features),
        RANDOM: pd.Series(rng.random(len(features)), index=features.index),
        PERFECT: features["target_max_rrp"].astype(float),
    }

    print(f"training {MODEL_WITHOUT} ...")
    without, reports_without = predict_out_of_fold(
        features, folds, ModelSpec(MODEL_WITHOUT, use_benchmark=False)
    )
    scores[MODEL_WITHOUT] = without

    print(f"training {MODEL_WITH} ...")
    with_bench, reports_with = predict_out_of_fold(
        features, folds, ModelSpec(MODEL_WITH, use_benchmark=True)
    )
    scores[MODEL_WITH] = with_bench

    if not args.skip_shuffle_control:
        print(f"training {SHUFFLED} ...")
        shuffled_frame = features.assign(
            is_spike=np.random.default_rng(11).permutation(features["is_spike"].to_numpy())
        )
        shuffled, _ = predict_out_of_fold(
            shuffled_frame, folds, ModelSpec(SHUFFLED, use_benchmark=True)
        )
        scores[SHUFFLED] = shuffled

    in_scope = scores["Climatology"].notna()
    evaluated = features[in_scope].copy()
    for name, series in scores.items():
        evaluated[name] = series[in_scope]

    columns = list(scores.keys())
    print("scoring the full set ...")
    full_cards, full_cis = score_everything(evaluated, columns, args.resamples)
    for card in full_cards:
        print("  " + card.headline())

    onset = evaluated[onset_mask(evaluated, settings.spike_threshold)].copy()
    print(f"scoring the onset slice ({len(onset):,} rows) ...")
    onset_cards, onset_cis = score_everything(onset, columns, args.resamples)
    for card in onset_cards:
        print("  " + card.headline())

    def denominators(frame: pd.DataFrame) -> dict[str, int]:
        positive = frame["is_spike"].astype(bool)
        return {
            "rows": len(frame),
            "positives": int(positive.sum()),
            "episodes": len(set(frame.loc[positive, "episode_id"]) - {NO_EPISODE}),
            "days": int(frame.loc[positive, "nem_trading_date"].nunique()),
        }

    full_d, onset_d = denominators(evaluated), denominators(onset)
    fold_detail = pd.DataFrame([r.__dict__ for r in reports_with]).drop(columns=["importance"])

    out = Path(__file__).resolve().parents[1] / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(
        out,
        {
            "full_rows": full_d["rows"],
            "full_positives": full_d["positives"],
            "full_episodes": full_d["episodes"],
            "full_days": full_d["days"],
            "full_table": results_table(full_cards, full_cis),
            "onset_rows": onset_d["rows"],
            "onset_positives": onset_d["positives"],
            "onset_episodes": onset_d["episodes"],
            "onset_days": onset_d["days"],
            "onset_table": results_table(onset_cards, onset_cis),
            "importance_without": _table(aggregate_importance(reports_without)),
            "importance_with": _table(aggregate_importance(reports_with)),
            "folds": _table(describe(folds)),
            "fold_detail": _table(fold_detail),
        },
    )
    print(f"\nwrote {out}")
    log_to_mlflow(full_cards, onset_cards, Path(__file__).resolve().parents[1])

    # Controls. Nothing above is trustworthy if any of these misses.
    lookup = {c.name: c for c in full_cards}
    ok = True
    floor, ceiling = lookup[RANDOM], lookup[PERFECT]

    if not np.isclose(floor.pr_auc, floor.base_rate, atol=0.01):
        print(
            f"CONTROL FAILED: random scored {floor.pr_auc:.4f} against a base "
            f"rate of {floor.base_rate:.4f}",
            file=sys.stderr,
        )
        ok = False
    if not np.isclose(ceiling.precision_at_k[DEFAULT_K[0]], 1.0):
        print(
            f"CONTROL FAILED: perfect foresight scored "
            f"{ceiling.precision_at_k[DEFAULT_K[0]]:.4f} at k={DEFAULT_K[0]}",
            file=sys.stderr,
        )
        ok = False
    if SHUFFLED in lookup:
        leak = lookup[SHUFFLED]
        if leak.pr_auc > leak.base_rate * 3:
            print(
                f"CONTROL FAILED: a model trained on shuffled labels scored "
                f"{leak.pr_auc:.4f} against a base rate of {leak.base_rate:.4f}. "
                "The training path is leaking; disregard every other number.",
                file=sys.stderr,
            )
            ok = False
    else:
        print("WARNING: the shuffled-label control was skipped", file=sys.stderr)

    print("controls passed" if ok else "controls failed")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
