"""Compare two analysed runs of the same tasks (e.g. run 1 with a 50-call budget, run 2 with 10).

    navalign compare --a results/analysis/run1 --a-run <run1 shard> [--a-run ...] \\
                     --b results/analysis/run2 --b-run <run2 dir> --out results/analysis/compare

Each side is a `navalign analysis` output dir (outcomes.csv, t4_predictive.csv) and the run
directories it was made from (results.jsonl and traces). Writes:

- `compare.csv`: one row per measure, a column per run (outcomes, scores, the T4 AUCs).
- `paired.csv`: per-task agreement on the tasks both runs have: rank correlation of S_file,
  same k, found in both or one, and how many calls the two traces share from the start (with
  sampling above temperature 0, a high score correlation is not one run replaying the other).
- `transitions.csv`: each task's category in run a against run b.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .outcomes import CATEGORIES
from .runs import Run, load_runs

SCORES = ("s_file", "focus", "s_node")
AUC_PREDICTORS = ("s_file", "-k_node")


def call_budget(run: Run) -> int | None:
    manifests = run.manifest.get("shards") or [run.manifest]
    budgets = {(m.get("evaluation") or {}).get("max_tool_calls") for m in manifests}
    return budgets.pop() if len(budgets) == 1 else None


def summary(outcomes: pd.DataFrame, t4: pd.DataFrame, run: Run) -> dict:
    """The run's outcomes, scores and AUCs as {measure: value}."""
    rows = run.latest
    o = outcomes.set_index("instance_id")
    calls = pd.Series({tid: r.get("tool_calls") for tid, r in rows.items()}, dtype=float)
    budget = call_budget(run)
    graded = o[o["gradable"] & o["tests_ran"].astype(bool)]
    out = {
        "tasks": len(o),
        "call budget": budget,
        "resolved (harness)": sum(bool(r.get("resolved")) for r in rows.values()),
        "passed (regraded)": int(o["passed"].fillna(False).astype(bool).sum()),
        "passed, gradable tasks whose tests ran": int(graded["passed"].astype(bool).sum()),
        "gradable tasks whose tests ran": len(graded),
        "non-empty patch": sum(bool((r.get("agent_patch") or "").strip()) for r in rows.values()),
        "median tool calls (harness count)": calls.median(),
        "median trace calls": o["n_calls"].median(),
        "median window length": o["window_end"].median(),
    }
    out |= {f"category {c}": int((o["category"] == c).sum()) for c in CATEGORIES}
    if budget is not None:
        at_budget = calls.reindex(o.index) >= budget
        out["runs at the call budget"] = int(at_budget.sum())
        out["B runs at the call budget"] = int((at_budget & (o["category"] == "B")).sum())
    out |= {f"mean {s}": o[s].mean() for s in SCORES}
    out |= {f"n {s}": int(o[s].notna().sum()) for s in SCORES}
    out |= {"median k": o["k"].median(), "median k_node": o["k_node"].median()}
    for _, r in t4[t4["predictor"].isin(AUC_PREDICTORS)].iterrows():
        name = f"AUC {r['predictor']} -> {r['outcome']}"
        out |= {name: r["auc"], f"{name} CI low": r["ci_low"], f"{name} CI high": r["ci_high"],
                f"{name} n": r["n"], f"{name} n positive": r["n_positive"]}
    return out


def _call_key(call) -> tuple[str, str]:
    return call.name, json.dumps(call.arguments, sort_keys=True)


def shared_prefix(a: list, b: list) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def paired(outcomes_a: pd.DataFrame, outcomes_b: pd.DataFrame, run_a: Run, run_b: Run) -> dict:
    a, b = outcomes_a.set_index("instance_id"), outcomes_b.set_index("instance_id")
    both = a.index.intersection(b.index)
    a, b = a.loc[both], b.loc[both]
    scored = a["s_file"].notna() & b["s_file"].notna()
    # Spearman as Pearson on ranks, as in navalign.analysis (no scipy)
    rho = a.loc[scored, "s_file"].rank().corr(b.loc[scored, "s_file"].rank())
    prefixes = []
    for tid in both:
        ta, tb = run_a.trace(tid), run_b.trace(tid)
        if ta is not None and tb is not None:
            prefixes.append(shared_prefix([_call_key(c) for c in ta.calls], [_call_key(c) for c in tb.calls]))
    prefixes = pd.Series(prefixes, dtype=float)
    found_a, found_b = a["found"].astype(bool), b["found"].astype(bool)
    return {
        "tasks in both runs": len(both),
        "S_file rank correlation": rho,
        "S_file rank correlation n": int(scored.sum()),
        "same k": int((a["k"] == b["k"]).sum()),
        "found a target file in both": int((found_a & found_b).sum()),
        "found in a only": int((found_a & ~found_b).sum()),
        "found in b only": int((~found_a & found_b).sum()),
        "trace pairs": len(prefixes),
        "same first call": int((prefixes >= 1).sum()),
        "median shared leading calls": prefixes.median(),
        "max shared leading calls": prefixes.max(),
    }


def transitions(outcomes_a: pd.DataFrame, outcomes_b: pd.DataFrame, names: tuple[str, str]) -> pd.DataFrame:
    m = outcomes_a[["instance_id", "category"]].merge(outcomes_b[["instance_id", "category"]], on="instance_id")
    order = list(CATEGORIES)
    t = pd.crosstab(m["category_x"], m["category_y"]).reindex(index=order, columns=order, fill_value=0)
    return t.rename_axis(index=names[0], columns=names[1])


def _show(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="navalign compare", description="Compare two analysed runs of the same tasks.")
    parser.add_argument("--a", required=True, help="navalign analysis output dir of the first run")
    parser.add_argument("--a-run", required=True, action="append", help="its run directory; repeat for each shard")
    parser.add_argument("--b", required=True, help="navalign analysis output dir of the second run")
    parser.add_argument("--b-run", required=True, action="append", help="its run directory; repeat for each shard")
    parser.add_argument("--out", required=True, help="output dir")
    args = parser.parse_args(argv)

    dirs = Path(args.a), Path(args.b)
    names = (dirs[0].name, dirs[1].name) if dirs[0].name != dirs[1].name else ("a", "b")
    outcomes = [pd.read_csv(d / "outcomes.csv") for d in dirs]
    t4 = [pd.read_csv(d / "t4_predictive.csv") for d in dirs]
    runs = load_runs(args.a_run), load_runs(args.b_run)
    table = pd.DataFrame({name: pd.Series(summary(o, t, r), dtype=object)
                          for name, o, t, r in zip(names, outcomes, t4, runs)})
    pairs = pd.Series(paired(*outcomes, *runs), name="value", dtype=object)
    moves = transitions(*outcomes, names)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.rename_axis("measure").to_csv(out / "compare.csv")
    pairs.rename_axis("measure").to_csv(out / "paired.csv")
    moves.to_csv(out / "transitions.csv")
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_rows", None):
        print(table.map(_show).to_string())
        print(f"\nPaired tasks (a = {names[0]}, b = {names[1]})\n" + pairs.map(_show).to_string())
        print("\nCategory transitions\n" + moves.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
