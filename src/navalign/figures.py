"""Paper figures F1–F4 from the files `navalign.analysis`, `navalign.score` and `navalign.validity`
write into one output dir. Needs the `analysis` extra (matplotlib).

- F1 `f1_breakdown.png`          failure breakdown per repo (100% stacked bars), in the plain-language groups
                                 of `BREAKDOWN`
- F2 `f2_first_read.png`         share of tasks that have read a fix (target) file by call k, agent vs random
                                 (needs `random_k.csv` from `navalign.validity baselines`)
- F3 `f3_scores_by_outcome.png`  S_file, focus and S_node of the runs that found a target file, by outcome
- F4 `f4_trajectories.png`       d of every read call in the window for three runs
                                 (needs `steps.jsonl` from `navalign.score --steps`)

Colors are a categorical palette validated for colour-vision deficiency, used in fixed slot order.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from .outcomes import CATEGORIES

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
FONTS = ("Plus Jakarta Sans", "Segoe UI", "DejaVu Sans")
DPI = 200
LINE = 1.5  # points; ~2px at the size the figures are shown
GAP = 1.5  # surface gap between touching fills, and the ring around dots
DOT = 18  # marker area (points²), ~8px across
OUTCOME_GROUPS = {"B": ("B",), "C": ("C1", "C2"), "D": ("D",), "CD_ng": ("CD_ng",)}
CASE_CATEGORIES = ("D", "B", "A")
# F1's legend, for readers who haven't met the category codes; escaping failures (C1) count as failed edits
BREAKDOWN = {
    "Never read a fix file": ("A",),
    "Read it, never edited it": ("B",),
    "Edited it, tests failed": ("C1", "C2"),
    "Edited it, tests passed": ("D",),
    "Edited it, can't be graded": ("CD_ng",),
    "Stopped by the context limit": ("X",),
}


def _rc() -> dict:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return {
        "font.family": next(f for f in FONTS if f in available or f == FONTS[-1]),
        "font.size": 8.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 9.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False, "legend.fontsize": 8,
    }


def _label_color(fill: str) -> str:
    """White or ink, whichever contrasts more with the fill."""
    channels = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(fill)]
    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    return INK if (luminance + 0.05) / 0.05 > 1.05 / (luminance + 0.05) else "white"


def _save(fig: Figure, path: Path) -> Path:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    return path


def breakdown_counts(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Runs per repo (rows, fewest first, then "all") and `BREAKDOWN` group (columns)."""
    by_category = pd.crosstab(outcomes["repo"], outcomes["category"]).reindex(columns=list(CATEGORIES), fill_value=0)
    counts = pd.DataFrame({label: by_category[list(categories)].sum(axis=1) for label, categories in BREAKDOWN.items()})
    counts = counts.loc[counts.sum(axis=1).sort_values().index]
    counts.loc["all"] = counts.sum()
    return counts


def fig_breakdown(outcomes: pd.DataFrame, path: Path) -> Path:
    counts = breakdown_counts(outcomes)
    totals = counts.sum(axis=1)
    shares = counts.div(totals, axis=0)
    with mpl.rc_context(_rc()):
        fig = Figure(figsize=(7.2, 2.9))
        ax = fig.subplots()
        y = np.arange(len(counts))
        left = np.zeros(len(counts))
        for color, label in zip(SERIES, BREAKDOWN, strict=False):
            width = shares[label].to_numpy()
            ax.barh(y, width, left=left, height=0.56, color=color, edgecolor=SURFACE, linewidth=GAP, label=label)
            for yi, (w, n) in enumerate(zip(width, counts[label], strict=True)):
                if w >= 0.06:  # only where the count fits inside the segment
                    ax.text(left[yi] + w / 2, yi, str(n), ha="center", va="center", fontsize=7.5, color=_label_color(color))
            left += width
        ax.set_yticks(y, [f"{repo}  (n = {n})" for repo, n in totals.items()])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"Where each agent run ended up, by repository ({totals['all']} tasks)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3, handlelength=1, columnspacing=1.4)
        return _save(fig, path)


def fig_first_read(outcomes: pd.DataFrame, random_k: pd.DataFrame, path: Path) -> Path:
    real = outcomes.loc[outcomes["s_file"].notna(), "k"].astype(float)
    rand = random_k["k"].astype(float)
    x_max = int(np.ceil(np.nanmax(real) / 5) * 5)
    xs = np.arange(1, x_max + 1)
    with mpl.rc_context(_rc()):
        fig = Figure(figsize=(6.4, 3.2))
        ax = fig.subplots()
        for (name, ks), color in zip({"agent": real, "random reads": rand}.items(), SERIES, strict=False):
            share = np.array([(ks <= x).mean() for x in xs])
            ax.step(xs, share, where="post", color=color, linewidth=LINE, label=name)
            never = ks.isna().mean()
            ax.annotate(f"{name}: {share[-1]:.0%} by call {x_max}\n({never:.0%} never in the window)",
                        (xs[-1], share[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlim(1, x_max)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel("call number in the trace (k)")
        ax.set_ylabel("tasks that have read a fix file")
        ax.set_title(f"First read of a fix file, {len(real)} tasks")
        ax.legend(loc="lower right", bbox_to_anchor=(1, 0.08))
        return _save(fig, path)


def fig_scores_by_outcome(outcomes: pd.DataFrame, path: Path, seed: int = 0) -> Path:
    found = outcomes[~outcomes["category"].isin(["A", "X"])]
    rng = np.random.default_rng(seed)
    metrics = {"s_file": "S_file", "focus": "focus", "s_node": "S_node"}
    with mpl.rc_context(_rc()):
        fig = Figure(figsize=(7.2, 2.8))
        axes = fig.subplots(1, len(metrics), sharey=True)
        for ax, (column, title) in zip(axes, metrics.items(), strict=True):
            labels = []
            for i, (group, categories) in enumerate(OUTCOME_GROUPS.items()):
                values = found.loc[found["category"].isin(categories), column].dropna().to_numpy(float)
                ax.scatter(i + rng.uniform(-0.18, 0.18, len(values)), values, s=DOT, color=SERIES[0], alpha=0.6,
                           edgecolors=SURFACE, linewidths=GAP / 2, zorder=2)
                if len(values):
                    ax.hlines(np.median(values), i - 0.3, i + 0.3, color=INK, linewidth=LINE, zorder=3)
                labels.append(f"{group}\nn = {len(values)}")
            ax.set_xticks(range(len(labels)), labels, fontsize=7.5)
            ax.set_xlim(-0.6, len(labels) - 0.4)
            ax.set_ylim(-0.03, 1.03)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
            ax.set_title(title)
        axes[0].set_ylabel("score (dots: runs; bar: median)")
        fig.suptitle("Navigation scores of the runs that found a target file, by outcome", x=0.01, ha="left",
                     fontsize=9.5, fontweight="semibold", color=INK, y=1.04)
        return _save(fig, path)


def pick_cases(outcomes: pd.DataFrame) -> list[str]:
    """One run per category in CASE_CATEGORIES: the one with the most read calls, on a task with target nodes."""
    eligible = outcomes[outcomes["cap"].notna()].sort_values(["read_calls", "instance_id"], ascending=[False, True])
    return [eligible.loc[eligible["category"] == c, "instance_id"].iloc[0]
            for c in CASE_CATEGORIES if (eligible["category"] == c).any()]


def fig_trajectories(outcomes: pd.DataFrame, steps: dict[str, list[dict]], path: Path, cases: list[str]) -> Path:
    by_id = outcomes.set_index("instance_id")
    y_max = int(by_id.loc[cases, "cap"].max())
    with mpl.rc_context(_rc()):
        fig = Figure(figsize=(7.2, 2.7))
        axes = fig.subplots(1, len(cases), sharey=True, squeeze=False)
        for ax, tid in zip(axes[0], cases, strict=True):
            reads = [s for s in steps[tid] if s["d"] is not None]
            x, d = [s["index"] + 1 for s in reads], [s["d"] for s in reads]
            cap = int(by_id.loc[tid, "cap"])
            ax.axhline(cap, color=AXIS, linewidth=0.6, zorder=1)
            ax.text(0.01, cap, "cap", va="bottom", fontsize=7, color=MUTED, transform=ax.get_yaxis_transform())
            ax.plot(x, d, color=SERIES[0], linewidth=LINE, solid_joinstyle="round", solid_capstyle="round", zorder=2)
            hits = [xi for xi, di in zip(x, d, strict=True) if di == 0]
            ax.scatter(hits, [0] * len(hits), s=DOT * 1.5, color=SERIES[0], edgecolors=SURFACE, linewidths=GAP, zorder=3)
            category = by_id.loc[tid, "category"]
            ax.set_title(f"{tid}\n{category}: {CATEGORIES[category]}", fontsize=8.5)
            ax.set_xlabel("call number (read calls only)")
            ax.set_ylim(-0.5, y_max + 1)
            ax.yaxis.grid(True)
            ax.set_axisbelow(True)
        axes[0][0].set_ylabel("hops to the nearest target node (d)")
        fig.suptitle("Distance of each read from the fix, inside the exploration window "
                     "(dots: reads that show a target node)", x=0.01, ha="left",
                     fontsize=9.5, fontweight="semibold", color=INK, y=1.08)
        return _save(fig, path)


def draw_all(out: Path, outcomes: pd.DataFrame) -> list[Path]:
    """Every figure whose inputs are in `out`; prints which were skipped and why."""
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written = [fig_breakdown(outcomes, figures / "f1_breakdown.png")]
    if (out / "random_k.csv").exists():
        written.append(fig_first_read(outcomes, pd.read_csv(out / "random_k.csv"), figures / "f2_first_read.png"))
    else:
        print(f"skipped F2: no {out / 'random_k.csv'} (run navalign.validity baselines)", file=sys.stderr)
    written.append(fig_scores_by_outcome(outcomes, figures / "f3_scores_by_outcome.png"))
    if (out / "steps.jsonl").exists():
        steps = {row["instance_id"]: row["steps"] for row in map(json.loads, (out / "steps.jsonl").read_text(encoding="utf-8").splitlines())}
        written.append(fig_trajectories(outcomes, steps, figures / "f4_trajectories.png", pick_cases(outcomes)))
    else:
        print(f"skipped F4: no {out / 'steps.jsonl'} (run navalign.score --steps)", file=sys.stderr)
    return written
