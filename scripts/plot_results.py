"""Draw the README's results figure from docs/day6_model.md.

    python scripts/plot_results.py

The numbers are read from the generated report rather than typed in, so the
figure cannot drift from the backtest that produced it. Writes a light and a
dark rendering to docs/figures/ for the README's <picture> element.

PR-AUC with its 95% blocked-bootstrap interval, for every score, on all rows
and on the onset slice. The point of the picture is the overlap: the model
leads, and the intervals say the lead cannot be certified.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = Path("docs/day6_model.md")
OUT_DIR = Path("docs/figures")

# Rows to draw, top to bottom, with the label shown on the axis. The floor and
# ceiling controls are left out: random sits on the base-rate line drawn in
# each panel, and perfect foresight is 1.0 by construction.
ROWS = [
    ("Model (+ AEMO feature)", "Model, with AEMO feature"),
    ("Model (no AEMO feature)", "Model, without it"),
    ("AEMO P5MIN", "AEMO pre-dispatch"),
    ("Persistence", "Persistence"),
    ("Climatology", "Climatology"),
]

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "model": "#2a78d6",
        "aemo": "#eb6834",
        "other": "#8a8984",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#33322f",
        "model": "#3987e5",
        "aemo": "#d95926",
        "other": "#8f8e88",
    },
}


@dataclass
class Score:
    pr_auc: float
    low: float
    high: float


@dataclass
class Section:
    rows: int
    positives: int
    episodes: int
    scores: dict[str, Score]

    @property
    def base_rate(self) -> float:
        return self.positives / self.rows


def parse_section(text: str, heading: str) -> Section:
    """Read one '## heading' block: its row counts and its score table."""
    block = text.split(f"## {heading}", 1)[1].split("\n## ", 1)[0]
    counts = re.search(r"([\d,]+) rows, ([\d,]+) positive, (\d+) episodes", block)
    if counts is None:
        raise ValueError(f"no row counts under '## {heading}' in {REPORT}")
    n_rows, n_pos, n_ep = (int(g.replace(",", "")) for g in counts.groups())
    scores = {}
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or " to " not in cells[2]:
            continue
        low, high = (float(v) for v in cells[2].split(" to "))
        scores[cells[0]] = Score(float(cells[1]), low, high)
    missing = [name for name, _ in ROWS if name not in scores]
    if missing:
        raise ValueError(f"'## {heading}' in {REPORT} has no row for {missing}")
    return Section(n_rows, n_pos, n_ep, scores)


def color_for(name: str, theme: dict[str, str]) -> str:
    if name.startswith("Model"):
        return theme["model"]
    if name == "AEMO P5MIN":
        return theme["aemo"]
    return theme["other"]


def draw_panel(ax, section: Section, title: str, theme: dict[str, str]) -> None:
    ys = list(range(len(ROWS)))[::-1]
    xmax = max(section.scores[name].high for name, _ in ROWS) * 1.18
    for y, (name, _) in zip(ys, ROWS, strict=True):
        s = section.scores[name]
        c = color_for(name, theme)
        ax.hlines(y, s.low, s.high, color=c, linewidth=2, zorder=2)
        ax.plot(
            s.pr_auc,
            y,
            "o",
            markersize=9,
            color=c,
            zorder=3,
            markeredgecolor=theme["surface"],
            markeredgewidth=2,
        )
        ax.text(
            s.high + xmax * 0.015,
            y,
            f"{s.pr_auc:.3f}",
            va="center",
            fontsize=9,
            color=theme["text"],
        )
    ax.axvline(
        section.base_rate, color=theme["muted"], linewidth=1, linestyle=(0, (3, 3)), zorder=1
    )
    ax.text(
        section.base_rate,
        len(ROWS) - 0.45,
        f" base rate {section.base_rate:.2%}",
        fontsize=8,
        color=theme["muted"],
        va="bottom",
    )
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(ROWS) - 0.1)
    ax.set_title(
        f"{title}\n{section.episodes} spike episodes, " f"{section.positives:,} positive rows",
        loc="left",
        fontsize=10,
        color=theme["text"],
    )
    ax.set_xlabel("PR-AUC (95% blocked-bootstrap interval)", fontsize=9, color=theme["muted"])
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])


def render(full: Section, onset: Section, mode: str, out: Path) -> None:
    theme = THEMES[mode]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10, 3.9), sharey=True, gridspec_kw={"wspace": 0.12}
    )
    fig.patch.set_facecolor(theme["surface"])
    for ax in (left, right):
        ax.set_facecolor(theme["surface"])
    draw_panel(left, full, "All rows", theme)
    draw_panel(right, onset, "Onset rows (no spike in the previous 30 min)", theme)
    left.set_yticks(list(range(len(ROWS)))[::-1], [label for _, label in ROWS])
    for tick in left.get_yticklabels():
        tick.set_color(theme["text"])
    fig.suptitle(
        "The model leads AEMO's forecast, and the intervals overlap",
        x=0.012,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=theme["text"],
    )
    fig.text(
        0.012,
        0.005,
        "Pooled out-of-fold scores from docs/day6_model.md. "
        "The two panels use different x scales.",
        fontsize=8,
        color=theme["muted"],
    )
    fig.subplots_adjust(left=0.19, right=0.98, top=0.78, bottom=0.17)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=theme["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    text = args.report.read_text()
    full = parse_section(text, "The full evaluation set")
    onset = parse_section(text, "The onset slice")
    for mode in THEMES:
        render(full, onset, mode, args.out_dir / f"results-{mode}.png")


if __name__ == "__main__":
    main()
