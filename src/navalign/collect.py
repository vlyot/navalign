"""Bookkeeping for agent trace collection in the Kaggle notebook.

The notebook passes in `run_one(task) -> result`, which wraps the harness `Evaluator`; nothing
here imports the harness, so the loop can be tested locally with a fake agent.

Output layout of a run directory:

- `manifest.json`  run id, shard, task ids, sampling and eval settings, start time
- `results.jsonl`  one row per attempted task, written and fsynced as soon as the task ends
- `traces/`, `logs/`  written by the harness (`trace_<id>.json`, `<id>.log`)
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from collections.abc import Callable, Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

RESULT_FIELDS = ("resolved", "test_exit_code", "agent_patch", "tool_calls", "duration_seconds")
SKIPPED = "skipped: deadline"


def select_tasks(ids: Iterable[str], shard: int = 0, n_shards: int = 1, done: Iterable[str] = ()) -> list[str]:
    """Every `n_shards`-th id of the sorted list, starting at `shard`, minus the ids in `done`.

    Shards are disjoint and together cover every id, whatever order `ids` comes in.
    """
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} is not in 0..{n_shards - 1}")
    done = set(done)
    return [tid for tid in sorted(ids)[shard::n_shards] if tid not in done]


def read_results(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    with open(results_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(results_path: Path, retry_errors: bool = True) -> set[str]:
    """Ids that already have a row in `results_path`; rows with an error don't count unless `retry_errors` is False."""
    return {r["instance_id"] for r in read_results(results_path) if not (retry_errors and r.get("error"))}


def as_dict(result: Any) -> dict:
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if dataclasses.is_dataclass(result):
        return dataclasses.asdict(result)
    return {k: v for k, v in vars(result).items() if not k.startswith("_")}


def result_record(instance_id: str, result: Any, started: float, finished: float) -> dict:
    """One JSON-ready row for an Evaluator result, given as a dict or an object.

    `tool_calls` is stored as a count (the trace holds the calls themselves). Fields other than
    the known ones go under `extra`, so nothing the harness returns is lost.
    """
    fields = as_dict(result)
    calls = fields.get("tool_calls")
    row = {"instance_id": instance_id} | {k: fields.get(k) for k in RESULT_FIELDS}
    row["tool_calls"] = len(calls) if isinstance(calls, (list, tuple)) else calls
    row["extra"] = {k: v for k, v in fields.items() if k not in RESULT_FIELDS and k != "instance_id"}
    row |= {"started": round(started, 3), "finished": round(finished, 3), "error": ""}
    return row


def _navalign_version() -> str | None:
    try:
        return version("navalign")
    except PackageNotFoundError:
        return None


def write_manifest(out_dir: Path, config: dict) -> Path:
    """Write `manifest.json`: the caller's config plus the navalign version and start time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = config | {"navalign_version": _navalign_version(), "written_at": time.time()}
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return path


def append_row(f, row: dict) -> None:
    f.write(json.dumps(row, default=str) + "\n")
    f.flush()
    os.fsync(f.fileno())


def collect_run(tasks: dict[str, Any], ids: list[str], run_one: Callable[[Any], Any], out_dir: Path,
                deadline: float, max_task_s: float, trace_dir: Path | None = None) -> list[dict]:
    """Run `run_one` on each task in `ids` and append one row per task to `out_dir/results.jsonl`.

    A task is not started if it could run past `deadline` (a `time.time()` value), given that
    one task takes at most `max_task_s`; it gets a `skipped: deadline` row instead, so a resumed
    run picks it up. An exception from `run_one` becomes an error row and the loop continues.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = trace_dir or out_dir / "traces"
    rows = []
    with open(out_dir / "results.jsonl", "a", encoding="utf-8") as f:
        for n, tid in enumerate(ids, 1):
            started = time.time()
            if started + max_task_s > deadline:
                row = {"instance_id": tid, "started": round(started, 3), "error": SKIPPED}
            else:
                try:
                    result = run_one(tasks[tid])
                except Exception as exc:  # one broken task must not end a 12-hour run
                    row = {"instance_id": tid, "started": round(started, 3), "finished": round(time.time(), 3),
                           "error": f"{type(exc).__name__}: {exc}"}
                else:
                    row = result_record(tid, result, started, time.time())
                row["trace_found"] = (trace_dir / f"trace_{tid}.json").exists()
            append_row(f, row)
            rows.append(row)
            status = row["error"] or f"resolved={row['resolved']} calls={row['tool_calls']}"
            print(f"[{n}/{len(ids)}] {tid}: {status} ({time.time() - started:.0f}s)", file=sys.stderr)
    return rows
