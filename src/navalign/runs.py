"""Load, check and summarise a trace-collection run downloaded from Kaggle.

A run directory is what `navalign.collect` and the harness write: `manifest.json`,
`results.jsonl`, and `trace_<id>.json` / `<id>.log` files anywhere below it. A folder with
traces but no `results.jsonl` (the harness's own `results/traces/`) is a run of every trace in it.

    python -m navalign.runs --run results/traces/val1 [--csv summary.csv]

A run collected in shards is checked as one: repeat `--run` once per shard directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .collect import SKIPPED, read_results
from .trace import ToolCall, Trace, load_trace

CONTEXT_ERROR = "ContextWindowExceeded"
# The harness's `tool_calls` count leaves these out, though the trace records them: `submit_patch`
# (val1), `get_status` and the root agent's call that launches the `code_analyzer_agent` sub-agent
# (run 1; the sub-agent's own calls are counted).
UNCOUNTED_TOOLS = frozenset({"submit_patch", "get_status", "code_analyzer_agent"})
# Calls that never reached a tool: the agent framework rejected the arguments, or the budget was spent.
REJECTED_ARGS = "mandatory input parameters are not present"
BUDGET_EXCEEDED = "BudgetExceeded"


def reached_tool(call: ToolCall) -> bool:
    """False for a call the framework rejected (malformed arguments) or the harness blocked (budget spent)."""
    try:
        obs = json.loads(call.output) if call.output else {}
    except json.JSONDecodeError:
        return True
    if not isinstance(obs, dict):
        return True
    return REJECTED_ARGS not in str(obs.get("error", "")) and obs.get("error_type") != BUDGET_EXCEEDED
SUMMARY_COLUMNS = ["instance_id", "resolved", "test_exit_code", "patch_chars", "tool_calls", "trace_calls",
                   "duration_seconds", "context_limit", "error"]


@dataclass
class Run:
    run_dir: Path
    manifest: dict
    rows: list[dict]  # every row in file order; a resumed task has several
    trace_paths: dict[str, Path]
    log_paths: dict[str, Path]
    _traces: dict[str, Trace] = field(default_factory=dict, repr=False)

    @property
    def latest(self) -> dict[str, dict]:
        """The last row per task: the attempt that counts."""
        return {r["instance_id"]: r for r in self.rows}

    def trace(self, instance_id: str) -> Trace | None:
        if instance_id not in self.trace_paths:
            return None
        if instance_id not in self._traces:
            self._traces[instance_id] = load_trace(self.trace_paths[instance_id])
        return self._traces[instance_id]


def load_run(run_dir: str | Path) -> Run:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    results = run_dir / "results.jsonl"
    trace_paths = {p.stem.removeprefix("trace_"): p for p in sorted(run_dir.rglob("trace_*.json"))}
    return Run(
        run_dir=run_dir,
        manifest=manifest,
        rows=read_results(results) if results.exists() else [{"instance_id": tid} for tid in trace_paths],
        trace_paths=trace_paths,
        log_paths={p.stem: p for p in sorted(run_dir.rglob("*.log"))},
    )


def load_runs(run_dirs: list[str | Path]) -> Run:
    """One Run over several run directories (the shards of one run); every manifest's task ids count."""
    runs = [load_run(d) for d in run_dirs]
    if len(runs) == 1:
        return runs[0]
    run_ids = dict.fromkeys(r.manifest.get("run_id", r.run_dir.name) for r in runs)
    return Run(
        run_dir=runs[0].run_dir,
        manifest={
            "run_id": "+".join(run_ids),
            "task_ids": sorted({tid for r in runs for tid in r.manifest.get("task_ids", [])}),
            "shards": [r.manifest for r in runs],
        },
        rows=[row for r in runs for row in r.rows],
        trace_paths={k: v for r in runs for k, v in r.trace_paths.items()},
        log_paths={k: v for r in runs for k, v in r.log_paths.items()},
    )


def check_run(run: Run) -> list[str]:
    """One message per problem; an empty list means every task ran and its trace matches its result."""
    problems = []
    latest = run.latest
    for tid in run.manifest.get("task_ids", []):
        if tid not in latest:
            problems.append(f"{tid}: no result row")
    completed = Counter(r["instance_id"] for r in run.rows if not r.get("error"))
    for tid, row in latest.items():
        if row.get("error") == SKIPPED:
            problems.append(f"{tid}: not run (deadline)")
            continue
        if row.get("error"):
            problems.append(f"{tid}: failed: {row['error']}")
            continue
        if completed[tid] > 1:
            problems.append(f"{tid}: {completed[tid]} completed rows")
        trace = run.trace(tid)
        if trace is None:
            problems.append(f"{tid}: no trace file")
        else:
            counted = sum(c.name not in UNCOUNTED_TOOLS and reached_tool(c) for c in trace.calls)
            if "tool_calls" in row and row["tool_calls"] != counted:
                problems.append(f"{tid}: result has {row.get('tool_calls')} tool calls, trace has {counted}")
    return problems


def hit_context_limit(run: Run, row: dict) -> bool:
    if CONTEXT_ERROR in json.dumps(row, default=str):
        return True
    log = run.log_paths.get(row["instance_id"])
    return log is not None and CONTEXT_ERROR in log.read_text(encoding="utf-8", errors="replace")


def summarize_run(run: Run) -> pd.DataFrame:
    """One row per task (its latest attempt), with a `calls_<tool>` count column per tool used."""
    records = []
    for tid, row in run.latest.items():
        trace = run.trace(tid)
        names = Counter(c.name for c in trace.calls) if trace else Counter()
        records.append({
            "instance_id": tid,
            "resolved": row.get("resolved"),
            "test_exit_code": row.get("test_exit_code"),
            "patch_chars": len(row.get("agent_patch") or ""),
            "tool_calls": row.get("tool_calls"),
            "trace_calls": len(trace.calls) if trace else None,
            "duration_seconds": row.get("duration_seconds"),
            "context_limit": hit_context_limit(run, row),
            "error": row.get("error", ""),
        } | {f"calls_{name}": n for name, n in sorted(names.items())})
    df = pd.DataFrame.from_records(records) if records else pd.DataFrame(columns=SUMMARY_COLUMNS)
    call_cols = [c for c in df.columns if c.startswith("calls_")]
    df[call_cols] = df[call_cols].fillna(0).astype(int)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="navalign runs", description="Check and summarise a downloaded trace-collection run.")
    parser.add_argument("--run", required=True, action="append",
                        help="run directory (manifest.json, results.jsonl, traces); repeat for each shard")
    parser.add_argument("--csv", help="also write the per-task summary here")
    args = parser.parse_args(argv)

    run = load_runs(args.run)
    summary = summarize_run(run)
    if args.csv:
        summary.to_csv(args.csv, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.drop(columns="error").to_string(index=False))
    done = summary[summary["error"] == ""]
    print(f"\n{run.manifest.get('run_id', run.run_dir.name)}: {len(done)}/{len(summary)} tasks ran, "
          f"{int(done['resolved'].fillna(False).astype(bool).sum())} resolved, "
          f"{int(summary['context_limit'].sum())} hit the context limit")
    problems = check_run(run)
    for p in problems:
        print(f"PROBLEM {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
