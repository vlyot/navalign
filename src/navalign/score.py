"""Navigation scores per task and run (Methodology step 4), over the exploration window.

File level, the main score:

- k: 1-based position in the trace of the first call that reads a target file (None if none does)
- S_file = 1/k, or 0 when no call in the window reads a target file
- focus: share of the window's read calls that read a target file

Node level, for tasks with at least one target node: each read call gets d, the hops on the
undirected `calls` graph from what it shows to the nearest target node, and
S_node = 1 − mean(d)/C over the window's read calls (C = the graph's cap).

- a read shows the graph nodes whose spans overlap the lines it returned (`read_nodes`)
- d of a read is the smallest distance among the nodes it shows, or C if it shows none
- a read of a target file whose changes map to no node (module-level, async or added code, a
  non-Python file) gets d = 0: that file has no finer location to measure against
- a call with several reads takes the smallest d

Reads of paths that are neither in the snapshot nor target files (the agent's scratch scripts)
are not navigation of the codebase and count toward neither focus nor S_node. Tasks whose only
targets are new files get no file-level score; `created_target` says whether the agent wrote one.

    python -m navalign.score --run DIR [--run DIR] --groundtruth DIR [--out scores.csv] [--steps steps.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .build import build_task
from .data import load_jsonl, snapshot_files
from .distance import DistanceMap
from .explore import Exploration, ReadSpan, explore_trace
from .mapper import NODE_RULES
from .runs import Run, load_runs
from .spans import SpanIndex
from .trace import Trace

GROUND_TRUTH = ("targets", "spans", "distances", "files")
COUNT_COLUMNS = ("k", "k_node", "k_seen", "cap")
PREFIX_COLUMNS = ("s_file_prefix", "focus_prefix", "s_node_prefix")


@dataclass
class GroundTruth:
    instance_id: str
    target_files: set[str]
    target_nodes: set[str]
    flat_files: set[str]  # target files with no target node: any read of one gives d = 0
    new_file_only: bool
    snapshot_files: set[str]
    spans: SpanIndex
    distances: DistanceMap

    @classmethod
    def from_rows(cls, targets: dict, spans: dict, distances: dict, files: dict) -> GroundTruth:
        """From one task's rows of targets.jsonl, spans.jsonl, distances.jsonl and files.jsonl."""
        target_files = set(targets["files"])
        node_files = {t["file"] for t in targets["targets"] if t["rule"] in NODE_RULES}
        return cls(
            instance_id=targets["instance_id"],
            target_files=target_files,
            target_nodes=set(targets["target_nodes"]),
            flat_files=target_files - node_files,
            new_file_only=bool(targets["targets"]) and all(t["rule"] == "new_file" for t in targets["targets"]),
            snapshot_files=set(files["files"]),
            spans=SpanIndex.from_dict(spans),
            distances=DistanceMap.from_dict(distances),
        )


def load_groundtruth(directory: str | Path, ids: Iterable[str] | None = None) -> dict[str, GroundTruth]:
    """Ground truth per task from a `navalign.build` output dir (only `ids` if given)."""
    rows = {name: load_jsonl(Path(directory) / f"{name}.jsonl") for name in GROUND_TRUTH}
    wanted = sorted(rows["targets"] if ids is None else ids)
    return {tid: GroundTruth.from_rows(*(rows[name][tid] for name in GROUND_TRUTH))
            for tid in wanted if all(tid in rows[name] for name in GROUND_TRUTH)}


@dataclass
class Step:
    index: int
    tool: str
    author: str | None
    files: list[str]  # files read (scratch files left out); empty for a call that reads nothing
    target_file: bool
    target_node: bool  # shows a target node
    d: int | None  # None: not a read call, or the task has no node level


@dataclass
class TaskScore:
    instance_id: str
    n_calls: int
    window_end: int
    read_calls: int
    k: int | None
    s_file: float | None
    focus: float | None
    k_node: int | None
    s_node: float | None
    mean_d: float | None
    cap: int | None
    k_seen: int | None  # first call whose results show a target file or node, read or only seen
    created_target: bool
    steps: list[Step] = field(default_factory=list)

    def row(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "steps"}


def read_nodes(read: ReadSpan, spans: SpanIndex) -> set[str]:
    """Graph nodes a read shows. `tail -n N` is placed at the file's last located span, since the
    ground truth has no file lengths: an estimate (code after the last node shifts it)."""
    if read.last is not None:
        ends = [s.end for s in spans.spans.get(read.file, [])]
        return spans.nodes_in(read.file, max(ends) - read.last + 1) if ends else set()
    return spans.nodes_in(read.file, read.start, read.end)


def read_distance(read: ReadSpan, nodes: set[str], gt: GroundTruth) -> int:
    """d for one read that shows `nodes`."""
    if read.file in gt.flat_files:
        return 0
    return min((gt.distances.distance(n) for n in nodes), default=gt.distances.cap)


def score_exploration(ex: Exploration, gt: GroundTruth) -> TaskScore:
    node_level = bool(gt.target_nodes)
    steps = []
    for e in ex.window:
        reads = [r for r in e.reads if r.file in gt.snapshot_files or r.file in gt.target_files]
        shown = [read_nodes(r, gt.spans) for r in reads]
        steps.append(Step(
            index=e.index, tool=e.tool, author=e.author,
            files=sorted({r.file for r in reads}),
            target_file=any(r.file in gt.target_files for r in reads),
            target_node=any(nodes & gt.target_nodes for nodes in shown),
            d=min(read_distance(r, nodes, gt) for r, nodes in zip(reads, shown, strict=True)) if reads and node_level else None,
        ))
    read_steps = [s for s in steps if s.files]
    file_level = not gt.new_file_only
    k = next((s.index + 1 for s in steps if s.target_file), None) if file_level else None
    ds = [s.d for s in read_steps if s.d is not None]
    mean_d = sum(ds) / len(ds) if ds else None
    seen = (e.index + 1 for e in ex.window
            if (e.seen_files | {r.file for r in e.reads}) & gt.target_files or e.seen_nodes & gt.target_nodes)
    return TaskScore(
        instance_id=ex.instance_id,
        n_calls=len(ex.events),
        window_end=ex.window_end,
        read_calls=len(read_steps),
        k=k,
        s_file=(1 / k if k else 0.0) if file_level else None,
        focus=sum(s.target_file for s in read_steps) / len(read_steps) if file_level and read_steps else None,
        k_node=next((s.index + 1 for s in steps if s.target_node), None) if node_level else None,
        s_node=1 - mean_d / gt.distances.cap if mean_d is not None else None,
        mean_d=mean_d,
        cap=gt.distances.cap if node_level else None,
        k_seen=next(seen, None),
        created_target=bool(set(ex.created_files) & gt.target_files),
        steps=steps,
    )


def prefix_scores(score: TaskScore, n: int) -> dict[str, float | None]:
    """S_file, focus and S_node over trace positions 1..n of the window only, so that runs are compared
    over the same span: a run that never edits a target file has a long window, and more room to stray."""
    if score.s_file is None:
        return dict.fromkeys(PREFIX_COLUMNS)
    reads = [s for s in score.steps if s.index < n and s.files]
    ds = [s.d for s in reads if s.d is not None]
    return {
        "s_file_prefix": 1 / score.k if score.k is not None and score.k <= n else 0.0,
        "focus_prefix": sum(s.target_file for s in reads) / len(reads) if reads else None,
        "s_node_prefix": 1 - sum(ds) / len(ds) / score.cap if ds else None,
    }


def explore_run(run: Run, groundtruth: dict[str, GroundTruth]) -> tuple[dict[str, Exploration], list[str]]:
    """Exploration per task with a trace and ground truth, and one problem line per task without."""
    explorations, problems = {}, []
    for tid in sorted(run.latest):
        trace = run.trace(tid)
        if trace is None or tid not in groundtruth:
            problems.append(f"{tid}: " + ("no trace" if trace is None else "no ground truth"))
            continue
        gt = groundtruth[tid]
        explorations[tid] = explore_trace(trace, gt.snapshot_files, gt.target_files, tid)
    return explorations, problems


def score_task(task: dict, graphs_dir: str | Path, snapshot: str | Path, trace: Trace) -> TaskScore:
    """Score one ATIF trace straight from a task, its code graph and its snapshot (no prebuilt ground truth)."""
    targets, spans, distances = build_task(task, Path(graphs_dir), Path(snapshot), {})
    gt = GroundTruth.from_rows(targets.to_dict(), spans.to_dict(), distances.to_dict(), {"files": snapshot_files(snapshot)})
    return score_exploration(explore_trace(trace, gt.snapshot_files, gt.target_files, task["instance_id"]), gt)


def scores_table(scores: Iterable[TaskScore]) -> pd.DataFrame:
    table = pd.DataFrame([s.row() for s in scores], columns=[f for f in TaskScore.__dataclass_fields__ if f != "steps"])
    for col in COUNT_COLUMNS:
        table[col] = table[col].astype("Int64")
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="File- and node-level navigation scores for every trace of a run.")
    parser.add_argument("--run", required=True, action="append", help="run directory; repeat for each shard")
    parser.add_argument("--groundtruth", required=True, help="navalign.build output dir")
    parser.add_argument("--out", help="write the per-task scores here (CSV)")
    parser.add_argument("--steps", help="write each task's per-step reads and distances here (JSON lines)")
    args = parser.parse_args(argv)

    run = load_runs(args.run)
    groundtruth = load_groundtruth(args.groundtruth, run.latest)
    explorations, problems = explore_run(run, groundtruth)
    scores = [score_exploration(ex, groundtruth[tid]) for tid, ex in explorations.items()]
    table = scores_table(scores)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
    if args.steps:
        Path(args.steps).parent.mkdir(parents=True, exist_ok=True)
        Path(args.steps).write_text("".join(json.dumps({"instance_id": s.instance_id, "steps": [asdict(st) for st in s.steps]}) + "\n"
                                            for s in scores), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.drop(columns=["n_calls", "cap"]).to_string(index=False, float_format="%.3f"))
    file_level = table[table["s_file"].notna()]
    print(f"\n{len(table)} tasks: S_file mean {file_level['s_file'].mean():.3f} over {len(file_level)} "
          f"({int(file_level['k'].isna().sum())} never read a target file), focus mean {table['focus'].mean():.3f}, "
          f"S_node mean {table['s_node'].mean():.3f} over {int(table['s_node'].notna().sum())} tasks with target nodes")
    for p in problems:
        print(f"PROBLEM {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
