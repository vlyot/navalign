"""Build the ground-truth artifacts for a set of tasks in one pass.

Writes to `--out`:

- `targets.jsonl`    mapper output per task (target files, nodes, rule per changed line)
- `spans.jsonl`      node → (file, start, end) index per task
- `distances.jsonl`  distance from each reachable node to the nearest target, with the cap
- `summary.csv`      one row per task, including any error

Rows are written as each task finishes, so an interrupted run keeps what it has done.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from .data import graph_path, load_graph, load_tasks
from .distance import DistanceMap, distance_map, graph_cap, undirected_calls
from .mapper import TaskTargets, map_task
from .spans import SpanIndex, build_span_index

SUMMARY_FIELDS = [
    "instance_id", "graph_file", "n_files", "n_target_nodes", "node_coverage", "rules",
    "cap", "n_nodes", "n_reachable", "span_coverage", "n_unlocated", "n_skipped_files",
    "seconds", "error",
]


def snapshot_path(snapshots_dir: Path, instance_id: str) -> Path | None:
    """`<id>.tgz` or an extracted `<id>/` directory, whichever exists."""
    for p in (snapshots_dir / f"{instance_id}.tgz", snapshots_dir / instance_id):
        if p.exists():
            return p
    return None


def build_task(task: dict, graphs_dir: Path, snapshot: Path, cap_cache: dict[str, int]) -> tuple[TaskTargets, SpanIndex, DistanceMap]:
    """Targets, span index and distance map for one task; the graph cap is cached per graph file."""
    gpath = graph_path(graphs_dir, task)
    graph = load_graph(gpath)
    targets = map_task(task, graph, snapshot)
    spans = build_span_index(snapshot, graph, task["instance_id"])
    u = undirected_calls(graph)
    if gpath.name not in cap_cache:
        cap_cache[gpath.name] = graph_cap(u)
    distances = distance_map(u, targets.target_nodes, cap_cache[gpath.name], task["instance_id"], gpath.name)
    return targets, spans, distances


def _summary_row(targets: TaskTargets, spans: SpanIndex, distances: DistanceMap) -> dict:
    return {
        "instance_id": targets.instance_id,
        "graph_file": distances.graph_file,
        "n_files": len(targets.files),
        "n_target_nodes": len(targets.target_nodes),
        "node_coverage": round(targets.node_coverage, 4),
        "rules": json.dumps(dict(targets.rules), sort_keys=True),
        "cap": distances.cap,
        "n_nodes": distances.n_nodes,
        "n_reachable": distances.n_reachable,
        "span_coverage": round(spans.coverage, 4),
        "n_unlocated": len(spans.unlocated),
        "n_skipped_files": len(spans.skipped_files),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build target maps, node span indexes and distance maps.")
    parser.add_argument("--data", default="data", help="dir with tasks.jsonl, graphs/, snapshots/")
    parser.add_argument("--tasks", help="tasks.jsonl (default: <data>/tasks.jsonl)")
    parser.add_argument("--graphs", help="graph dir (default: <data>/graphs)")
    parser.add_argument("--snapshots", help="snapshot dir (default: <data>/snapshots)")
    parser.add_argument("--task", action="append", help="instance_id (repeatable); default: every task with a snapshot")
    parser.add_argument("--out", required=True, help="output dir")
    args = parser.parse_args(argv)

    data = Path(args.data)
    tasks = load_tasks(Path(args.tasks) if args.tasks else data / "tasks.jsonl")
    graphs_dir = Path(args.graphs) if args.graphs else data / "graphs"
    snapshots_dir = Path(args.snapshots) if args.snapshots else data / "snapshots"
    ids = args.task or sorted(tid for tid in tasks if snapshot_path(snapshots_dir, tid))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cap_cache: dict[str, int] = {}
    failures = 0
    with (open(out / "targets.jsonl", "w", encoding="utf-8") as f_targets,
          open(out / "spans.jsonl", "w", encoding="utf-8") as f_spans,
          open(out / "distances.jsonl", "w", encoding="utf-8") as f_dist,
          open(out / "summary.csv", "w", encoding="utf-8", newline="") as f_summary):
        summary = csv.DictWriter(f_summary, fieldnames=SUMMARY_FIELDS)
        summary.writeheader()
        for n, tid in enumerate(ids, 1):
            start = time.perf_counter()
            try:
                if tid not in tasks:
                    raise KeyError(f"{tid} not in tasks.jsonl")
                snapshot = snapshot_path(snapshots_dir, tid)
                if snapshot is None:
                    raise FileNotFoundError(f"no snapshot for {tid} in {snapshots_dir}")
                targets, spans, distances = build_task(tasks[tid], graphs_dir, snapshot, cap_cache)
            except Exception as exc:  # one broken task must not stop a 129-task run
                failures += 1
                row = {"instance_id": tid, "error": f"{type(exc).__name__}: {exc}"}
            else:
                f_targets.write(json.dumps(targets.to_dict()) + "\n")
                f_spans.write(json.dumps(spans.to_dict()) + "\n")
                f_dist.write(json.dumps(distances.to_dict()) + "\n")
                row = _summary_row(targets, spans, distances) | {"error": ""}
            row["seconds"] = round(time.perf_counter() - start, 1)
            summary.writerow(row)
            for f in (f_targets, f_spans, f_dist, f_summary):
                f.flush()
            status = row["error"] or "ok"
            print(f"[{n}/{len(ids)}] {tid}: {status} ({row['seconds']}s)", file=sys.stderr)

    print(f"{len(ids) - failures}/{len(ids)} tasks built into {out}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
