#!/usr/bin/env python3
"""Day 5: measure the three baselines, and write down what they scored.

Output goes to `docs/day5_baselines.md`, generated rather than hand-edited, so
no number in it can be one nobody produced.

Two controls run alongside the real baselines. A uniformly random score must
land at the base rate, and perfect foresight must reach precision 1.0 -- if
either misses, the scoring is wrong and the three real numbers mean nothing.
They are printed with the rest rather than hidden in a test, because a floor
and a ceiling make the middle three readable.

All four scored baselines see the identical row set. Where a baseline has no
value for a row -- AEMO published no forecast, the rolling window was too short
-- the row keeps its place and the score ranks last. Dropping those rows would
let a baseline improve its own metric by declining to answer on the hard ones.
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
    operating_point,
    persistence_score,
)
from nemfc.config import load_settings
from nemfc.dataset import add_episodes, load
from nemfc.evaluate import (
    DEFAULT_K,
    NO_EPISODE,
    ScoreCard,
    blocked_bootstrap_pr_auc,
    score,
)
from nemfc.folds import (
    DEFAULT_MIN_SPIKING_DAYS,
    DEFAULT_MIN_TRAIN_DAYS,
    DEFAULT_MIN_TRAIN_SPIKING_DAYS,
    describe,
    make_folds,
)


def build_scores(frame: pd.DataFrame, folds, shape, seed: int = 0) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    return {
        "Climatology": climatology_out_of_fold(frame, folds, shape),
        "Persistence": persistence_score(frame),
        "AEMO P5MIN": aemo_score(frame),
        "Random (floor)": pd.Series(rng.random(len(frame)), index=frame.index),
        "Perfect foresight (ceiling)": frame["target_max_rrp"].astype(float),
    }


def _fmt(value: float, places: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{places}f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    """A markdown table without pulling in `tabulate` for one call."""
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
            elif isinstance(value, int | np.integer):
                cells.append(f"{value:,}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    cards: list[ScoreCard],
    points: dict[str, dict],
    fold_table: pd.DataFrame,
    coverage: dict[str, int],
    cis: dict[str, tuple[float, float]],
    threshold: float,
) -> None:
    lines = [
        "# Day 5: baselines",
        "",
        f"Generated {datetime.now(timezone.utc):%d %b %Y %H:%M} UTC by "
        "`scripts/baselines.py`. Not hand-edited.",
        "",
        "## The comparison set",
        "",
        f"- Rows scored: {coverage['scored_rows']:,} "
        f"(of {coverage['total_rows']:,} in the feature table)",
        f"- Positive rows: {coverage['positive_rows']:,}",
        f"- Spike episodes: {coverage['episodes']} across {coverage['days_with_spikes']} days",
        f"- Rows AEMO published a forecast for: {coverage['has_benchmark']:,} "
        f"({100 * coverage['has_benchmark'] / max(coverage['scored_rows'], 1):.1f}%)",
        f"- Positive rows with no AEMO forecast: {coverage['positive_missing_benchmark']:,}",
        f"- Rows with a full 30-minute rolling window: {coverage['has_persistence']:,}",
        "",
        "Rows before the first test fold are training data for every fold and are",
        "excluded. Rows a baseline has no value for keep their place and rank last:",
        "abstaining is not rewarded.",
        "",
        "## Results",
        "",
        "Every figure is pooled across the out-of-fold predictions and scored once,",
        "never averaged over folds. `precision@k` is the share of the top k alerts",
        "followed by a spike; `episode recall@k` is the share of distinct episodes",
        "with at least one alert inside it. The two answer different questions and",
        "the second is the one an operator would ask.",
        "",
    ]

    header = "| Baseline | PR-AUC | 95% CI |"
    divider = "|---|---|---|"
    for k in DEFAULT_K:
        header += f" precision@{k} | lift@{k} | episode recall@{k} |"
        divider += "---|---|---|"
    lines += [header, divider]

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

    base_rate = cards[0].base_rate if cards else float("nan")
    lines += [
        "",
        f"Base rate {_fmt(base_rate, 4)}. Lift is precision@k over that: 1.0x is no",
        "better than guessing.",
        "",
        "### Where the tie-break carries the result",
        "",
        "`precision@k` breaks ties chronologically. The column below is the same",
        "figure averaged over every possible tie ordering. A gap means the score is",
        "too coarse to rank at that k, which is a fact about the baseline.",
        "",
        "| Baseline | k | chronological | expected under random ties |",
        "|---|---|---|---|",
    ]
    for card in cards:
        for k in DEFAULT_K:
            if k in card.precision_at_k:
                lines.append(
                    f"| {card.name} | {k} | {_fmt(card.precision_at_k[k])} | "
                    f"{_fmt(card.expected_precision_at_k[k])} |"
                )

    lines += [
        "",
        f"### The boolean operating point at ${threshold:,.0f}",
        "",
        "PLAN §2.1 defines the AEMO and persistence baselines as booleans. Scored",
        "that way they cannot be ranked, so the tables above use the underlying",
        "magnitude. This is the decision the boolean rule actually makes.",
        "",
        "| Baseline | alerts | precision | recall |",
        "|---|---|---|---|",
    ]
    for name, point in points.items():
        lines.append(
            f"| {name} | {point['n_alerts']:,} | {_fmt(point['precision'])} | "
            f"{_fmt(point['recall'])} |"
        )

    lines += [
        "",
        "## Fold structure",
        "",
        "Expanding window, folds sized by spiking days rather than by calendar or",
        "by episode count, with a 24-hour purge gap between train and test.",
        "",
        _markdown_table(fold_table),
        "",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-spiking-days", type=int, default=DEFAULT_MIN_SPIKING_DAYS)
    parser.add_argument("--min-train-days", type=int, default=DEFAULT_MIN_TRAIN_DAYS)
    parser.add_argument(
        "--min-train-spiking-days", type=int, default=DEFAULT_MIN_TRAIN_SPIKING_DAYS
    )
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("docs/day5_baselines.md"))
    args = parser.parse_args()

    settings = load_settings()
    engine = create_engine(settings.database_url, pool_pre_ping=True)

    features, history = load(engine, settings.region)
    if features.empty:
        print("feature table is empty", file=sys.stderr)
        return 1

    features, n_episodes_total = add_episodes(features, settings.spike_threshold)
    shape = long_history_shape(history) if not history.empty else None
    if shape is None:
        print("no long history found: climatology falls back to a flat prior")

    folds = make_folds(
        features,
        min_spiking_days=args.min_spiking_days,
        min_train_days=args.min_train_days,
        min_train_spiking_days=args.min_train_spiking_days,
    )
    print(f"{len(folds)} folds, {n_episodes_total} episodes in the full period")

    scores = build_scores(features, folds, shape)

    # The evaluation set is where climatology exists: out of fold, for all.
    in_scope = scores["Climatology"].notna()
    evaluated = features[in_scope].copy()
    for name, series in scores.items():
        evaluated[name] = series[in_scope]

    coverage = {
        "total_rows": len(features),
        "scored_rows": len(evaluated),
        "positive_rows": int(evaluated["is_spike"].sum()),
        "episodes": len(set(evaluated.loc[evaluated["is_spike"], "episode_id"]) - {NO_EPISODE}),
        "days_with_spikes": int(evaluated.loc[evaluated["is_spike"], "nem_trading_date"].nunique()),
        "has_benchmark": int(evaluated["benchmark_max_forecast_rrp"].notna().sum()),
        "positive_missing_benchmark": int(
            (evaluated["is_spike"] & evaluated["benchmark_max_forecast_rrp"].isna()).sum()
        ),
        "has_persistence": int(evaluated["rolling_max_rrp_30m"].notna().sum()),
    }

    cards, cis = [], {}
    for name in scores:
        card = score(evaluated, name, name)
        cards.append(card)
        cis[name] = blocked_bootstrap_pr_auc(evaluated, name, n_resamples=args.resamples)
        print(card.headline())

    points = {
        "AEMO P5MIN": operating_point(evaluated, evaluated["AEMO P5MIN"], settings.spike_threshold),
        "Persistence": operating_point(
            evaluated, evaluated["Persistence"], settings.spike_threshold
        ),
    }

    out = Path(__file__).resolve().parents[1] / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(out, cards, points, describe(folds), coverage, cis, settings.spike_threshold)
    print(f"\nwrote {out}")

    # The controls. If either fails, nothing above is trustworthy.
    floor = next(c for c in cards if c.name.startswith("Random"))
    ceiling = next(c for c in cards if c.name.startswith("Perfect"))
    ok = True
    if not np.isclose(floor.pr_auc, floor.base_rate, atol=0.01):
        print(
            f"CONTROL FAILED: random scored PR-AUC {floor.pr_auc:.4f}, "
            f"base rate is {floor.base_rate:.4f}",
            file=sys.stderr,
        )
        ok = False
    if not np.isclose(ceiling.precision_at_k[DEFAULT_K[0]], 1.0):
        print(
            f"CONTROL FAILED: perfect foresight scored precision@{DEFAULT_K[0]} "
            f"{ceiling.precision_at_k[DEFAULT_K[0]]:.4f}, expected 1.0",
            file=sys.stderr,
        )
        ok = False
    print("controls passed" if ok else "controls failed")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
