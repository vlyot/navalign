"""Checkpoint 2: is the score valid? Three checks, run before the analysis.

- Oracle: the real window's length, every call a read that shows a target node. It must score
  best: S_file = focus = S_node = 1 on every task.
- Random: the real window with each read call pointed at a random Python file of the snapshot
  (150 lines from line 1 or a random node's start). The real runs must score clearly above it.
- Issue mentions and BM25 (`--rankings`): cheap heuristics an agent could follow without exploring.
  The real window's i-th read becomes 150 lines from the i-th entry of a fixed ranking: the files
  and functions the issue names, in order of mention, or the graph functions ranked by BM25 against
  the issue text. A ranking that runs out reads nothing more.
- Hand check: ~10 traces labelled by hand from the raw calls, 0 (never read a target file),
  1 (read one, but no target node's lines) or 2 (read a target node's lines), compared with the
  level the scorer derives from the same window (Cohen's kappa).

    python -m navalign.validity rankings --tasks FILE --graphs DIR --groundtruth DIR --out rankings.jsonl
    python -m navalign.validity baselines --run DIR [--run DIR] --groundtruth DIR --out DIR [--draws 200] [--rankings FILE]
    python -m navalign.validity label-sheet --run DIR [--run DIR] --groundtruth DIR --out DIR [--n 10]
                                          [--sample random] [--exclude labels.csv]
    python -m navalign.validity agreement --run DIR [--run DIR] --groundtruth DIR --labels CSV [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import bootstrap_ci
from .data import graph_path, load_graph, load_jsonl, load_tasks
from .explore import CallEvent, Exploration, ReadSpan, observation
from .runs import Run, load_runs
from .score import (
    GroundTruth,
    TaskScore,
    explore_run,
    load_groundtruth,
    score_exploration,
)
from .trace import ToolCall

METRICS = ("s_file", "focus", "s_node")
RANKED = ("issue", "bm25")
READ_LINES = 150
PY_PATH = re.compile(r"[\w./-]+\.py\b")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
WORD = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")  # splits snake_case and CamelCase
MIN_NAME = 4  # shorter names (get, run, ...) say nothing about where the fix is
MAX_NAME_MATCHES = 3  # a name defined more often than this is too common to point anywhere
BM25_K1, BM25_B = 1.2, 0.75
LEVELS = {0: "never read a target file", 1: "read a target file, no target node", 2: "read a target node"}
MAX_LABEL_WINDOW = 40  # keeps a hand-labelled trace readable
CODE_INDENT = " " * 8  # an indented code block inside a numbered list item, up to "99. "


def _target_read(gt: GroundTruth) -> ReadSpan:
    """A read showing a located target node; a target file's first lines when no target node is located."""
    for file, spans in sorted(gt.spans.spans.items()):
        for s in spans:
            if s.node in gt.target_nodes:
                return ReadSpan(file, s.start, s.end)
    return ReadSpan(min(gt.target_files), 1, READ_LINES)


def oracle_exploration(ex: Exploration, gt: GroundTruth) -> Exploration:
    read = _target_read(gt)
    events = [CallEvent(i, "read_file", True, reads=[read]) for i in range(ex.window_end)]
    return Exploration(ex.instance_id, events, ex.window_end, [])


def random_exploration(ex: Exploration, gt: GroundTruth, rng: np.random.Generator, python_files: list[str]) -> Exploration:
    """The real window, each call that reads snapshot code instead reading one random Python file."""
    events = []
    for e in ex.window:
        reads = []
        if any(r.file in gt.snapshot_files or r.file in gt.target_files for r in e.reads):
            file = python_files[rng.integers(len(python_files))]
            starts = [1] + [s.start for s in gt.spans.spans.get(file, [])]
            start = starts[rng.integers(len(starts))]
            reads = [ReadSpan(file, start, start + READ_LINES - 1)]
        events.append(CallEvent(e.index, e.tool, e.ok, reads=reads, author=e.author))
    return Exploration(ex.instance_id, events, ex.window_end, [])


Read = tuple[str, int]  # (file, first line)


def issue_text(task: dict) -> str:
    return "\n".join(t for t in (task.get("problem_statement"), task.get("hints_text")) if t)


def issue_mentions(text: str, gt: GroundTruth) -> list[Read]:
    """Files and functions the issue names, as reads in order of first mention.

    A file is named by its path (`requests/utils.py`, or its tail) or its module (`requests.utils`); a
    function or class by its name, if that name has at least MIN_NAME characters and at most
    MAX_NAME_MATCHES located graph nodes carry it.
    """
    files = sorted(f for f in gt.snapshot_files if f.endswith(".py"))
    by_name: dict[str, list[Read]] = {}
    for file, spans in gt.spans.spans.items():
        for span in spans:
            by_name.setdefault(span.node.rsplit(".", 1)[-1], []).append((file, span.start))

    def named(tail: str) -> list[Read]:
        return [(f, 1) for f in files if f == tail or f.endswith("/" + tail)]

    found: list[tuple[int, Read]] = []
    for m in PY_PATH.finditer(text):
        found += [(m.start(), r) for r in named(m.group().lstrip("./"))]
    for m in IDENTIFIER.finditer(text):
        dotted = m.group()
        if "." in dotted:
            module = dotted.replace(".", "/")
            found += [(m.start(), r) for r in named(module + ".py") + named(module + "/__init__.py")]
        name = dotted.rsplit(".", 1)[-1]
        spots = by_name.get(name, [])
        if len(name) >= MIN_NAME and len(spots) <= MAX_NAME_MATCHES:
            found += [(m.start(), r) for r in spots]
    return list(dict.fromkeys(r for _, r in sorted(found, key=lambda x: x[0])))


def words(text: str) -> list[str]:
    return [w.lower() for w in WORD.findall(text) if len(w) > 1]


def bm25_ranking(text: str, node_texts: dict[str, str], gt: GroundTruth) -> list[Read]:
    """Located graph nodes ranked by BM25 between the issue text and each node's id and source; zero scores left out."""
    located = {s.node: (file, s.start) for file, spans in gt.spans.spans.items() for s in spans}
    docs = {n: Counter(words(n + " " + t)) for n, t in node_texts.items() if n in located}
    if not docs:
        return []
    lengths = {n: sum(c.values()) for n, c in docs.items()}
    avg = sum(lengths.values()) / len(docs) or 1
    df = Counter(w for c in docs.values() for w in c)
    idf = {w: math.log(1 + (len(docs) - d + 0.5) / (d + 0.5)) for w, d in df.items()}
    query = set(words(text)) & set(idf)
    scores = {}
    for n, counts in docs.items():
        norm = BM25_K1 * (1 - BM25_B + BM25_B * lengths[n] / avg)
        score = sum(idf[w] * counts[w] * (BM25_K1 + 1) / (counts[w] + norm) for w in query if w in counts)
        if score > 0:
            scores[n] = score
    return list(dict.fromkeys(located[n] for n in sorted(scores, key=lambda n: (-scores[n], n))))


def task_rankings(task: dict, node_texts: dict[str, str], gt: GroundTruth) -> dict[str, list[Read]]:
    text = issue_text(task)
    return {"issue": issue_mentions(text, gt), "bm25": bm25_ranking(text, node_texts, gt)}


def ranked_exploration(ex: Exploration, gt: GroundTruth, ranking: list[Read]) -> Exploration:
    """The real window, its i-th call that reads snapshot code instead reading 150 lines from `ranking[i]`."""
    events, i = [], 0
    for e in ex.window:
        reads = []
        if any(r.file in gt.snapshot_files or r.file in gt.target_files for r in e.reads):
            if i < len(ranking):
                file, start = ranking[i]
                reads = [ReadSpan(file, start, start + READ_LINES - 1)]
            i += 1
        events.append(CallEvent(e.index, e.tool, e.ok, reads=reads, author=e.author))
    return Exploration(ex.instance_id, events, ex.window_end, [])


def baselines(explorations: dict[str, Exploration], groundtruth: dict[str, GroundTruth], draws: int = 200,
              seed: int = 0, rankings: dict[str, dict] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per task: real, oracle and mean random scores, plus each ranked baseline's when `rankings` has the task;
    and every random draw's k (for the first-read ECDF).

    New-file-only tasks have no file-level score and are left out.
    """
    rng = np.random.default_rng(seed)
    rows, ks = [], []
    for tid, ex in explorations.items():
        gt = groundtruth[tid]
        if gt.new_file_only:
            continue
        python_files = sorted(f for f in gt.snapshot_files if f.endswith(".py"))
        real, oracle = score_exploration(ex, gt), score_exploration(oracle_exploration(ex, gt), gt)
        randoms = [score_exploration(random_exploration(ex, gt, rng, python_files), gt) for _ in range(draws)]
        row = {"instance_id": tid}
        for m in METRICS:
            values = [getattr(r, m) for r in randoms if getattr(r, m) is not None]
            row |= {f"real_{m}": getattr(real, m), f"oracle_{m}": getattr(oracle, m),
                    f"random_{m}": float(np.mean(values)) if values else None}
        for kind in RANKED:
            if kind in (rankings or {}).get(tid, {}):
                ranking = [tuple(r) for r in rankings[tid][kind]]
                ranked = score_exploration(ranked_exploration(ex, gt, ranking), gt)
                row |= {f"{kind}_{m}": getattr(ranked, m) for m in METRICS}
        rows.append(row)
        ks += [{"instance_id": tid, "draw": i, "k": r.k} for i, r in enumerate(randoms)]
    per_task = pd.DataFrame(rows)
    if rankings:
        per_task = per_task.reindex(columns=[*per_task.columns, *(f"{kind}_{m}" for kind in RANKED for m in METRICS
                                                                 if f"{kind}_{m}" not in per_task)])
    per_task = per_task.astype({c: float for c in per_task.columns if c != "instance_id"})
    return per_task, pd.DataFrame(ks, columns=["instance_id", "draw", "k"]).astype({"k": "Int64"})


def validity_summary(per_task: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """T3 (baselines): per metric, the means, the oracle's minimum, and real − random with a bootstrap CI;
    the same difference for each ranked baseline present (`issue_*`, `bm25_*` columns)."""
    rows = []
    for m in METRICS:
        sub = per_task[[f"real_{m}", f"oracle_{m}", f"random_{m}"]].dropna()
        real, oracle, rand = (sub[f"{kind}_{m}"].to_numpy() for kind in ("real", "oracle", "random"))
        lo, hi = bootstrap_ci(lambda r, x: float(np.mean(r - x)), real, rand, seed=seed)
        row = {"metric": m, "n": len(sub), "oracle_mean": oracle.mean(), "oracle_min": oracle.min(),
               "real_mean": real.mean(), "random_mean": rand.mean(), "real_minus_random": float(np.mean(real - rand)),
               "ci_low": lo, "ci_high": hi, "share_real_above_random": float(np.mean(real > rand))}
        for kind in RANKED:
            if f"{kind}_{m}" not in per_task:
                continue
            pair = per_task[[f"real_{m}", f"{kind}_{m}"]].dropna()
            real_k, base = pair[f"real_{m}"].to_numpy(), pair[f"{kind}_{m}"].to_numpy()
            klo, khi = bootstrap_ci(lambda r, x: float(np.mean(r - x)), real_k, base, seed=seed)
            row |= {f"{kind}_mean": base.mean(), f"real_minus_{kind}": float(np.mean(real_k - base)),
                    f"{kind}_ci_low": klo, f"{kind}_ci_high": khi}
        rows.append(row)
    return pd.DataFrame(rows)


def checkpoint_passes(summary: pd.DataFrame) -> bool:
    """Oracle scores 1 everywhere, and the real runs beat random on every metric (CI above 0)."""
    return bool((summary["oracle_min"] >= 1 - 1e-9).all() and (summary["ci_low"] > 0).all())


def score_level(score: TaskScore) -> int:
    """The hand-check level the scorer derives: 0 no target-file read, 1 target file only, 2 a target node shown."""
    if score.k is None:
        return 0
    return 2 if score.k_node is not None else 1


def _describe_call(call: ToolCall) -> str:
    obs = observation(call)
    status = obs.get("status", "?")
    if call.name == "read_file":
        return (f"`read_file` {call.arguments.get('filepath')!s} → {status}, returned lines "
                f"{obs.get('start_line')}–{obs.get('end_line')} of {obs.get('total_lines')}")
    if call.name == "run_command":
        lines = (obs.get("stdout") or "").splitlines()
        shown = "".join(f"\n{CODE_INDENT}{line[:160]}" for line in lines[:8])
        more = f"\n{CODE_INDENT}… ({len(lines) - 8} more lines)" if len(lines) > 8 else ""
        command = call.arguments.get("command", "").replace("\n", f"\n{CODE_INDENT}")
        return f"`run_command` → {status}\n\n{CODE_INDENT}$ {command[:400]}{shown}{more}\n"
    args = {k: call.arguments[k] for k in sorted(call.arguments, key=lambda k: k != "filepath")}
    return f"`{call.name}` {json.dumps(args)[:200]} → {status}"


def _digest(number: int, tid: str, run: Run, ex: Exploration, gt: GroundTruth) -> str:
    spans = {s.node: (file, s.start, s.end) for file, file_spans in gt.spans.spans.items() for s in file_spans}
    nodes = [f"`{n}` ({spans[n][0]} lines {spans[n][1]}–{spans[n][2]})" if n in spans else f"`{n}` (not located)"
             for n in sorted(gt.target_nodes)]
    calls = run.trace(tid).calls[: ex.window_end]
    body = "\n".join(f"{c.index + 1}. {_describe_call(c)}" for c in calls)
    return (f"## {number}. {tid}\n\nTarget files: {', '.join(f'`{f}`' for f in sorted(gt.target_files))}\n\n"
            f"Target nodes: {'; '.join(nodes)}\n\nWindow: calls 1–{ex.window_end} of {len(ex.events)}\n\n{body}\n")


SHEET_INTRO = """# Hand check: did the agent read the code the fix changes?

For each trace below, read the calls in its exploration window (every call before its first edit
of a real file) and put one label per task in `human_labels.csv`:

- **0**: never read any target file
- **1**: read a target file, but never any line of a target node
- **2**: read at least one line of a target node

Judge from what each call returned (line ranges, printed output), not from what the agent meant
to do. A file named only in search output (`grep`, `find`, `ls`) is not a read. Use `notes` for
anything that made the call hard.
"""


def label_sheet(run: Run, explorations: dict[str, Exploration], groundtruth: dict[str, GroundTruth], n: int = 10,
                seed: int = 0, sample: str = "stratified", exclude: frozenset[str] = frozenset()) -> tuple[str, pd.DataFrame]:
    """`n` traces to label and a blank label table: stratified by the scorer's level (which the sheet doesn't show),
    or a uniform random sample; tasks in `exclude` (already labelled) are never drawn."""
    rng = np.random.default_rng(seed)
    by_level: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for tid, ex in explorations.items():
        gt = groundtruth[tid]
        if tid not in exclude and gt.target_nodes and not gt.new_file_only and 1 <= ex.window_end <= MAX_LABEL_WINDOW:
            by_level[score_level(score_exploration(ex, gt))].append(tid)
    if sample == "random":
        chosen = [str(t) for t in rng.permutation(sorted(t for ids in by_level.values() for t in ids))[:n]]
    else:
        pools = [list(rng.permutation(sorted(ids))) for ids in by_level.values()]
        chosen = []
        while len(chosen) < n and any(pools):
            for pool in pools:
                if pool and len(chosen) < n:
                    chosen.append(str(pool.pop()))
    chosen = [str(t) for t in rng.permutation(chosen)]
    sheet = SHEET_INTRO + "\n" + "\n".join(_digest(i, tid, run, explorations[tid], groundtruth[tid])
                                           for i, tid in enumerate(chosen, 1))
    return sheet, pd.DataFrame({"instance_id": chosen, "label": pd.Series([pd.NA] * len(chosen), dtype="Int64"), "notes": ""})


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    observed = float(np.mean(a == b))
    expected = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in np.union1d(a, b))
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement(labels: pd.DataFrame, scores: dict[str, TaskScore]) -> tuple[pd.DataFrame, dict]:
    """Per task human vs scorer level, and kappa, exact agreement and Spearman(human, S_node)."""
    table = labels[["instance_id", "label"]].astype({"label": int}).assign(
        scorer=lambda d: [score_level(scores[t]) for t in d["instance_id"]],
        s_node=lambda d: [scores[t].s_node for t in d["instance_id"]],
    )
    summary = {
        "n": len(table),
        "kappa": cohen_kappa(table["label"], table["scorer"]),
        "agreement": float((table["label"] == table["scorer"]).mean()),
        "spearman_s_node": float(table["label"].rank().corr(table["s_node"].astype(float).rank())),  # no scipy
    }
    return table, summary


def rankings_main(args: argparse.Namespace) -> int:
    tasks = load_tasks(args.tasks)
    groundtruth = load_groundtruth(args.groundtruth)
    texts: dict[Path, dict[str, str]] = {}  # one graph in memory at a time; tasks on a commit are consecutive
    with open(args.out, "w", encoding="utf-8") as f:
        for tid, gt in groundtruth.items():
            path = graph_path(args.graphs, tasks[tid])
            if path not in texts:
                texts = {path: {n: d.get("text") or "" for n, d in load_graph(path).nodes(data=True)}}
            f.write(json.dumps({"instance_id": tid} | task_rankings(tasks[tid], texts[path], gt)) + "\n")
    print(f"rankings for {len(groundtruth)} tasks written to {args.out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", required=True, action="append", help="run directory; repeat for each shard")
    common.add_argument("--groundtruth", required=True, help="navalign.build output dir")
    common.add_argument("--seed", type=int, default=0)
    parser = argparse.ArgumentParser(prog="navalign validity", description="Checkpoint 2 validity checks for the navigation score.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("rankings", help="issue-mention and BM25 read rankings per task")
    p.add_argument("--tasks", required=True, help="tasks.jsonl (problem_statement, hints_text)")
    p.add_argument("--graphs", required=True, help="graph dir; node text is the BM25 document")
    p.add_argument("--groundtruth", required=True, help="navalign.build output dir (spans locate each node)")
    p.add_argument("--out", required=True, help="rankings.jsonl")
    p = sub.add_parser("baselines", parents=[common], help="oracle, random and (with --rankings) issue/BM25 baselines")
    p.add_argument("--out", required=True, help="writes baselines.csv, validity.csv, random_k.csv")
    p.add_argument("--draws", type=int, default=200, help="random draws per task")
    p.add_argument("--rankings", help="rankings.jsonl from `validity rankings`")
    p = sub.add_parser("label-sheet", parents=[common], help="traces to label by hand")
    p.add_argument("--out", required=True, help="writes human_check.md and a blank human_labels.csv")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--sample", choices=("stratified", "random"), default="stratified")
    p.add_argument("--exclude", action="append", default=[], help="a labels CSV whose tasks are never drawn (repeatable)")
    p = sub.add_parser("agreement", parents=[common], help="hand labels vs the scorer")
    p.add_argument("--labels", required=True, help="the filled-in human_labels.csv")
    p.add_argument("--out", help="also write agreement.csv here")
    args = parser.parse_args(argv)
    if args.command == "rankings":
        return rankings_main(args)

    run = load_runs(args.run)
    groundtruth = load_groundtruth(args.groundtruth, run.latest)
    explorations, problems = explore_run(run, groundtruth)
    for problem in problems:
        print(f"PROBLEM {problem}", file=sys.stderr)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    if args.command == "baselines":
        rankings = load_jsonl(args.rankings) if args.rankings else None
        per_task, random_k = baselines(explorations, groundtruth, args.draws, args.seed, rankings)
        summary = validity_summary(per_task, args.seed)
        per_task.to_csv(out / "baselines.csv", index=False)
        summary.to_csv(out / "validity.csv", index=False)
        random_k.to_csv(out / "random_k.csv", index=False)
        print(summary.to_string(index=False, float_format="%.3f"))
        passed = checkpoint_passes(summary)
        print("\nCheckpoint 2 baselines: " + ("PASS" if passed else "FAIL") +
              " (oracle scores 1 on every task; real - random CI above 0 for every metric)")
        return 0 if passed and not problems else 1

    if args.command == "label-sheet":
        labels_path = out / "human_labels.csv"
        if labels_path.exists():
            print(f"{labels_path} already exists; move it away to draw a new sheet", file=sys.stderr)
            return 1
        exclude = frozenset(t for path in args.exclude for t in pd.read_csv(path)["instance_id"])
        sheet, labels = label_sheet(run, explorations, groundtruth, args.n, args.seed, args.sample, exclude)
        (out / "human_check.md").write_text(sheet, encoding="utf-8")
        labels.to_csv(labels_path, index=False)
        print(f"wrote {out / 'human_check.md'} and {labels_path} ({len(labels)} traces)")
        return 1 if problems else 0

    labels = pd.read_csv(args.labels)
    unlabelled = labels.loc[labels["label"].isna(), "instance_id"].tolist()
    invalid = labels.loc[labels["label"].notna() & ~labels["label"].isin(list(LEVELS)), "instance_id"].tolist()
    if unlabelled or invalid:
        print(f"labels must be 0, 1 or 2: missing {unlabelled}, invalid {invalid}", file=sys.stderr)
        return 1
    scores = {tid: score_exploration(explorations[tid], groundtruth[tid]) for tid in labels["instance_id"]}
    table, summary = agreement(labels, scores)
    if out:
        table.to_csv(out / "agreement.csv", index=False)
    print(table.to_string(index=False, float_format="%.3f"))
    print("\n" + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in summary.items()))
    print("confusion (rows human, columns scorer):\n" + pd.crosstab(table["label"], table["scorer"]).to_string())
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
