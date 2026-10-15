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
    navalign score-trace TRACE (--groundtruth DIR | --data DIR) [--task ID] [--steps] [--json]
    navalign score-trace --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .build import build_task, snapshot_path
from .data import example_dir, load_jsonl, load_tasks, snapshot_files
from .distance import DistanceMap
from .explore import Exploration, ReadSpan, explore_trace
from .mapper import NODE_RULES
from .runs import Run, load_runs
from .spans import SpanIndex
from .trace import Trace, load_trace

GROUND_TRUTH = ("targets", "spans", "distances", "files")
COUNT_COLUMNS = ("k", "k_node", "k_seen", "cap")
PREFIX_COLUMNS = ("s_file_prefix", "focus_prefix", "s_node_prefix")
DEMO_TASK = "toyshop_1"
METHODOLOGY = "https://github.com/vlyot/navalign/blob/main/docs/methodology.md"


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


def score_one(trace: str | Path, task_id: str, groundtruth: str | Path | None = None, tasks: str | Path | None = None,
              graphs: str | Path | None = None, snapshots: str | Path | None = None) -> TaskScore:
    """Score one trace file against prebuilt ground truth, or against ground truth built from the task's
    `tasks` file, graph dir and snapshot dir."""
    if groundtruth is not None:
        gt = load_groundtruth(groundtruth, [task_id]).get(task_id)
        if gt is None:
            raise KeyError(f"{task_id} is not in the ground truth in {groundtruth}")
        return score_exploration(explore_trace(load_trace(trace), gt.snapshot_files, gt.target_files, task_id), gt)
    task = load_tasks(tasks).get(task_id)
    if task is None:
        raise KeyError(f"{task_id} is not in {tasks}")
    snapshot = snapshot_path(Path(snapshots), task_id)
    if snapshot is None:
        raise FileNotFoundError(f"no snapshot of {task_id} in {snapshots}")
    return score_task(task, graphs, snapshot, load_trace(trace))


def _at_call(k: int | None) -> str:
    return f"call {k}" if k else "never"


def describe_score(score: TaskScore, steps: bool = False) -> str:
    """Plain-English summary of one task's scores; `steps` adds a row per call in the exploration window."""
    if score.window_end == score.n_calls:
        window = "no call edits a repo file, so the exploration window is the whole trace"
    else:
        window = f"the exploration window is calls 1-{score.window_end}, before the first edit of a repo file"
    reads = [s for s in score.steps if s.files]
    rows = []
    if score.s_file is None:
        rows.append(("fixed files", "the fix only adds new files; the agent " +
                     ("created one" if score.created_target else "created none"), ""))
    else:
        rows.append(("first read of a fixed file", _at_call(score.k), f"S_file {score.s_file:.3f}"))
        rows.append(("reads that hit a fixed file", f"{sum(s.target_file for s in reads)} of {len(reads)}",
                     "" if score.focus is None else f"focus  {score.focus:.3f}"))
    if score.cap is None:
        rows.append(("fixed code", "no fixed function or class is in the graph", ""))
    else:
        rows.append(("first read of fixed code", _at_call(score.k_node), ""))
        rows.append(("mean distance of reads", "no reads" if score.mean_d is None else f"{score.mean_d:.2f} hops",
                     "" if score.s_node is None else f"S_node {score.s_node:.3f}  (cap {score.cap} hops)"))
    rows.append(("first sight of the fix", _at_call(score.k_seen), "(search results or reads)"))
    widths = [max(len(r[i]) for r in rows) for i in (0, 1)]
    out = [f"{score.instance_id}: {score.n_calls} tool calls; {window}", ""]
    out += [f"  {a:<{widths[0]}}  {b:<{widths[1]}}  {c}".rstrip() for a, b, c in rows]
    if steps:
        table = [("call", "tool", "files read", "fixed file", "fixed code", "d")] + [
            (str(s.index + 1), s.tool, ", ".join(s.files) or "-", "yes" if s.target_file else "",
             "yes" if s.target_node else "", "" if s.d is None else str(s.d)) for s in score.steps]
        widths = [max(len(r[i]) for r in table) for i in range(6)]
        out += [""] + [("  " + "  ".join(f"{v:<{w}}" for v, w in zip(r, widths, strict=True))).rstrip() for r in table]
    out += ["", "S_file = 1/k for the first read of a fixed file; focus = share of reads that hit one;",
            f"S_node = 1 - mean hops/cap (1: every read showed fixed code). Definitions: {METHODOLOGY}"]
    return "\n".join(out)


def trace_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="navalign score-trace", description="Navigation scores for one agent trace.")
    parser.add_argument("trace", nargs="?", help="ATIF trace file, e.g. trace_<instance_id>.json")
    parser.add_argument("--demo", action="store_true", help="score the bundled toy example")
    parser.add_argument("--task", help="instance_id (default: from the trace file name)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--groundtruth", help="prebuilt ground truth: `navalign build` output or the navalign-groundtruth Dataset")
    source.add_argument("--data", help="dir with tasks.jsonl, graphs/ and snapshots/, to build the ground truth on the fly")
    parser.add_argument("--tasks", help="tasks.jsonl (default: <data>/tasks.jsonl)")
    parser.add_argument("--graphs", help="graph dir (default: <data>/graphs)")
    parser.add_argument("--snapshots", help="snapshot dir (default: <data>/snapshots)")
    parser.add_argument("--steps", action="store_true", help="also list every call in the exploration window")
    parser.add_argument("--json", action="store_true", help="print the scores and steps as JSON")
    args = parser.parse_args(argv)

    if args.demo:
        if args.trace or args.groundtruth or args.data:
            parser.error("--demo takes no trace, --groundtruth or --data")
        args.data = example_dir()
        args.trace = args.data / "traces" / f"trace_{DEMO_TASK}.json"
    data = Path(args.data) if args.data else None
    raw = {name: getattr(args, name) or (data / default if data else None)
           for name, default in (("tasks", "tasks.jsonl"), ("graphs", "graphs"), ("snapshots", "snapshots"))}
    if not args.trace or not (args.groundtruth or None not in raw.values()):
        parser.error("give a trace and either --groundtruth DIR or --data DIR (or try --demo)")
    task_id = args.task or Path(args.trace).stem.removeprefix("trace_")
    try:
        score = score_one(args.trace, task_id, args.groundtruth, **raw)
    except KeyError as exc:
        parser.error(exc.args[0] + ("" if args.task else "; if the file isn't trace_<instance_id>.json, give the id with --task"))
    except FileNotFoundError as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(score), indent=2) if args.json else describe_score(score, args.steps))
    return 0


def scores_table(scores: Iterable[TaskScore]) -> pd.DataFrame:
    table = pd.DataFrame([s.row() for s in scores], columns=[f for f in TaskScore.__dataclass_fields__ if f != "steps"])
    for col in COUNT_COLUMNS:
        table[col] = table[col].astype("Int64")
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="navalign score", description="File- and node-level navigation scores for every trace of a run.")
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
