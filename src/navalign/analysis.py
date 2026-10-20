"""Result tables for the paper (Methodology step 5), from a run, its ground truth and its grades.

Written to `--out`:

- `outcomes.csv`        one row per run: scores, category (`navalign.outcomes`), finer outcomes
- `t1_groundtruth.csv`  ground truth and gradability per repo
- `t2_breakdown.csv`    failure breakdown per repo, the main result
- `t4_predictive.csv`   does navigation predict the outcome? AUC, bootstrap 95% CI, permutation p
- `figures/`            F1–F4 with `--figures` (needs the `analysis` extra)

T3 (validity) comes from `navalign.validity`. Circularity guard: the score never predicts A vs the
rest, since A is the score's own file-level test. It predicts (i) among runs that found a target
file, editing one (C/D vs B) and the patch touching a target node, and (ii) among graded runs that
edited one, passing (D vs C). A run that never edits a target file has a longer window, with
more room to stray, so window length is listed as a reference and the scores are also taken over
every run's first `--prefix` calls (default 10), an equal span for all runs.

    python -m navalign.analysis --run DIR [--run DIR] --groundtruth DIR --grades DIR --gradable CSV --out DIR [--prefix N] [--figures]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_jsonl
from .grade import load_grades
from .outcomes import CATEGORIES, PREFIX, agent_grades, run_outcomes
from .runs import load_runs
from .score import GroundTruth, load_groundtruth

FALLBACK_RULES = frozenset({"enclosing", "file", "new_file"})
FOUND_EXCLUDED = ("A", "X")
GRADED_EDITS = ("C1", "C2", "D")


def _auc_from_ranks(ranks: np.ndarray, labels: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(a positive scores above a negative), ties counted half; NaN unless both classes are present."""
    return _auc_from_ranks(pd.Series(np.asarray(scores, dtype=float)).rank().to_numpy(), np.asarray(labels, dtype=bool))


def bootstrap_ci(stat: Callable[..., float], *arrays: np.ndarray, n_boot: int = 2000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of `stat` over resamples of the rows (one row per task)."""
    rng = np.random.default_rng(seed)
    arrays = tuple(np.asarray(a) for a in arrays)
    size = len(arrays[0])
    values = [stat(*(a[idx] for a in arrays)) for idx in (rng.integers(0, size, size) for _ in range(n_boot))]
    lo, hi = np.nanpercentile(values, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def permutation_p(scores: np.ndarray, labels: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided p of the AUC's distance from 0.5 under shuffled labels."""
    rng = np.random.default_rng(seed)
    ranks = pd.Series(np.asarray(scores, dtype=float)).rank().to_numpy()
    labels = np.asarray(labels, dtype=bool)
    observed = abs(_auc_from_ranks(ranks, labels) - 0.5)
    hits = sum(abs(_auc_from_ranks(ranks, rng.permutation(labels)) - 0.5) >= observed - 1e-12 for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values for one family of tests (NaN p-values stay NaN and don't count)."""
    p = np.asarray(p, dtype=float)
    adjusted = np.full_like(p, np.nan)
    keep = np.flatnonzero(~np.isnan(p))
    order = keep[np.argsort(p[keep], kind="stable")]
    m = len(order)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adjusted[idx] = running
    return adjusted


def _repo_summary(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "tasks": len(g),
        "gradable": int(g["gradable"].sum()),
        "with_target_node": int(g["target_node"].sum()),
        "fully_node_covered": int(g["fully_node_covered"].sum()),
        "file_level_only": int((~g["target_node"]).sum()),
        "new_file_only": int(g["new_file_only"].sum()),
        "fallback_rule": int(g["fallback_rule"].sum()),
        "span_coverage_mean": round(g["span_coverage"].mean(), 3),
        "cap_range": f"{g['cap'].min()}-{g['cap'].max()}",
    })


def groundtruth_table(targets: dict[str, dict], groundtruth: dict[str, GroundTruth], gradable: set[str]) -> pd.DataFrame:
    """T1: tasks, gradable tasks, target-node coverage, fallback rules, span coverage and caps per repo."""
    rows = [{
        "repo": tid.rsplit("_", 1)[0],
        "gradable": tid in gradable,
        "target_node": bool(t["target_nodes"]),
        "fully_node_covered": t["node_coverage"] == 1,
        "new_file_only": groundtruth[tid].new_file_only,
        "fallback_rule": bool(set(t["rules"]) & FALLBACK_RULES),
        "span_coverage": groundtruth[tid].spans.coverage,
        "cap": groundtruth[tid].distances.cap,
    } for tid, t in targets.items() if tid in groundtruth]
    df = pd.DataFrame(rows)
    table = pd.DataFrame({repo: _repo_summary(g) for repo, g in df.groupby("repo")} | {"all": _repo_summary(df)}).T
    return table.rename_axis("repo").reset_index()


def breakdown_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    """T2: runs per category and repo, with an `all` row."""
    counts = pd.crosstab(outcomes["repo"], outcomes["category"]).reindex(columns=list(CATEGORIES), fill_value=0)
    counts.loc["all"] = counts.sum()
    counts.insert(0, "n", counts.sum(axis=1))
    return counts.rename_axis(columns=None).reset_index()


def _predictors(t: pd.DataFrame, prefix: int) -> dict[str, pd.Series]:
    """Navigation measures oriented so that higher = better navigation (over the window, then over its
    first `prefix` calls), plus window length as a reference."""
    k_node = t["k_node"].astype(float)
    return {
        "s_file": t["s_file"].astype(float),
        "focus": t["focus"].astype(float),
        "s_node": t["s_node"].astype(float),
        "-k_node": (-k_node.fillna(np.inf)).where(t["cap"].notna()),  # never showing a target node ranks last
        f"s_file_first{prefix}": t["s_file_prefix"].astype(float),
        f"focus_first{prefix}": t["focus_prefix"].astype(float),
        f"s_node_first{prefix}": t["s_node_prefix"].astype(float),
        "window_end": t["window_end"].astype(float),
    }


def predictive_table(outcomes: pd.DataFrame, prefix: int = PREFIX, n_boot: int = 2000, n_perm: int = 5000,
                     seed: int = 0) -> pd.DataFrame:
    """T4: AUC of each navigation measure for each outcome, on the populations where it isn't circular.

    `p_holm` adjusts the permutation p-values over every test in the table except window length, which is a
    reference, not a navigation measure.
    """
    found = outcomes[~outcomes["category"].isin(FOUND_EXCLUDED)]
    with_nodes = found[found["cap"].notna()]
    graded = outcomes[outcomes["category"].isin(GRADED_EDITS)]
    questions = [
        ("edited a target file (C/D vs B)", "found a target file", found, found["category"] != "B"),
        ("patch touches a target node", "found a target file, task has target nodes", with_nodes,
         with_nodes["patch_touches_target_node"].astype(bool)),
        ("passed (D vs C)", "graded, edited a target file", graded, graded["category"] == "D"),
    ]
    rows = []
    for outcome, population, sub, y in questions:
        for name, x in _predictors(sub, prefix).items():
            keep = x.notna().to_numpy()
            xs, ys = x.to_numpy(float)[keep], y.to_numpy(bool)[keep]
            lo, hi = bootstrap_ci(auc, xs, ys, n_boot=n_boot, seed=seed)
            rows.append({"outcome": outcome, "population": population, "predictor": name, "n": len(xs),
                         "n_positive": int(ys.sum()), "auc": auc(xs, ys), "ci_low": lo, "ci_high": hi,
                         "p": permutation_p(xs, ys, n_perm, seed)})
    table = pd.DataFrame(rows)
    tested = table["predictor"] != "window_end"
    table["p_holm"] = np.nan
    table.loc[tested, "p_holm"] = holm(table.loc[tested, "p"].to_numpy())
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="navalign analysis", description="Outcomes and result tables (T1, T2, T4) for a scored run.")
    parser.add_argument("--run", required=True, action="append", help="run directory; repeat for each shard")
    parser.add_argument("--groundtruth", required=True, help="navalign.build output dir")
    parser.add_argument("--grades", required=True, help="dir with the grades.jsonl that holds the run's agent grades")
    parser.add_argument("--gradable", required=True, help="gradable_tasks.csv")
    parser.add_argument("--out", required=True, help="output dir")
    parser.add_argument("--figures", action="store_true", help="also draw F1-F4 into <out>/figures (needs matplotlib)")
    parser.add_argument("--prefix", type=int, default=PREFIX, help="calls for the equal-span prefix scores")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run = load_runs(args.run)
    groundtruth = load_groundtruth(args.groundtruth, run.latest)
    gradable = set(pd.read_csv(args.gradable).query("gradable")["instance_id"])
    outcomes = run_outcomes(run, groundtruth, agent_grades(load_grades(args.grades)), gradable, args.prefix)
    t1 = groundtruth_table(load_jsonl(Path(args.groundtruth) / "targets.jsonl"), groundtruth, gradable)
    t2 = breakdown_table(outcomes)
    t4 = predictive_table(outcomes, args.prefix, seed=args.seed)
    outcomes.to_csv(out / "outcomes.csv", index=False)
    t1.to_csv(out / "t1_groundtruth.csv", index=False)
    t2.to_csv(out / "t2_breakdown.csv", index=False)
    t4.to_csv(out / "t4_predictive.csv", index=False)
    if args.figures:
        from .figures import draw_all

        for path in draw_all(out, outcomes):
            print(f"wrote {path}", file=sys.stderr)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("T1 ground truth\n" + t1.to_string(index=False))
        print("\nT2 failure breakdown (" + ", ".join(f"{k} = {v}" for k, v in CATEGORIES.items()) + ")\n" + t2.to_string(index=False))
        print("\nT4 navigation -> outcome\n" + t4.to_string(index=False, float_format="%.3f"))
    missing = sorted(set(run.latest) - set(outcomes["instance_id"]))
    for tid in missing:
        print(f"PROBLEM {tid}: no trace or no ground truth", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
