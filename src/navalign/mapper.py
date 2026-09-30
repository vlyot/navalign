"""Map a task's reference patch to target files and graph nodes.

Every changed line is resolved to one rule:

- ``exact``      the innermost function/class containing the line is a graph node
- ``enclosing``  the innermost symbol is missing (async, added by the patch, ...) but an
                 enclosing symbol is a node; that node is the target
- ``file``       no containing symbol is a node (module-level line, or a top-level added/async
                 function); the file is the target, and its graph nodes are listed separately
- ``new_file``   the patch creates the file, so the base graph has nothing for it
- ``non_python`` docs, config, etc.; file-level target only

Removed lines are resolved against the base file, added lines against the post-patch file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

from .data import graph_path, load_graph, load_tasks, read_snapshot_files
from .patch import FileChange, apply_file_change, parse_patch
from .symbols import Symbol, enclosing_chain, index_symbols, module_candidates

NODE_RULES = ("exact", "enclosing")
SOURCE_ROOTS = ("src", "docs_src")


@dataclass
class Target:
    file: str
    rule: str
    node: str | None = None  # graph node id for exact/enclosing
    symbol: str | None = None  # qualname of the innermost changed symbol, if any
    reason: str | None = None  # why the innermost symbol is not a node
    lines: list[str] = field(default_factory=list)  # "-12" base line, "+14" post line


@dataclass
class TaskTargets:
    instance_id: str
    files: list[str]
    targets: list[Target]
    file_nodes: dict[str, list[str]]  # graph nodes defined in each changed file (base side)

    @property
    def target_nodes(self) -> set[str]:
        return {t.node for t in self.targets if t.rule in NODE_RULES}

    @property
    def rules(self) -> Counter:
        """Changed-line count per rule."""
        return Counter({r: sum(len(t.lines) for t in self.targets if t.rule == r)
                        for r in {t.rule for t in self.targets}})

    @property
    def node_coverage(self) -> float:
        """Share of changed lines that resolve to a graph node (exact or enclosing)."""
        counts = self.rules
        total = sum(counts.values())
        return sum(counts[r] for r in NODE_RULES) / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "files": self.files,
            "target_nodes": sorted(self.target_nodes),
            "node_coverage": round(self.node_coverage, 4),
            "rules": dict(self.rules),
            "targets": [asdict(t) for t in self.targets],
            "file_nodes": self.file_nodes,
        }


def module_prefixes(node_ids: set[str]) -> set[str]:
    """Every proper dotted prefix of every node id (`a.b.c` → `a`, `a.b`)."""
    return {i.rsplit(".", k)[0] for i in node_ids for k in range(1, i.count(".") + 1)}


def resolve_module(path: str, node_ids: set[str], prefixes: set[str] | None = None) -> str:
    """Module name the graph uses for `path` (e.g. `docs_src/security/x.py` → `security.x`).

    Takes the longest dotted candidate that prefixes some node id; for files with no nodes
    (e.g. new files) falls back to the path with a leading source root stripped. Pass
    `prefixes` (from `module_prefixes`) when resolving many files against one graph.
    """
    if prefixes is None:
        prefixes = module_prefixes(node_ids)
    candidates = module_candidates(path)
    for cand in candidates:
        if cand in prefixes:
            return cand
    first = path.replace("\\", "/").split("/", 1)[0]
    return candidates[1] if first in SOURCE_ROOTS and len(candidates) > 1 else candidates[0]


def _resolve_chain(chain: list[Symbol], node_ids: set[str], base_ids: set[str]) -> tuple[str, str | None, str | None]:
    """(rule, node, reason) for the symbols enclosing one changed line, innermost first."""
    if not chain:
        return "file", None, "module_level"
    inner = chain[0]
    if inner.graph_id in node_ids:
        return "exact", inner.graph_id, None
    if inner.graph_id not in base_ids:
        reason = "added"
    elif inner.is_async:
        reason = "async"
    else:
        reason = "not_in_graph"
    for sym in chain[1:]:
        if sym.graph_id in node_ids:
            return "enclosing", sym.graph_id, reason
    return "file", None, reason


def map_file(change: FileChange, base_text: str | None, graph: nx.Graph) -> tuple[list[Target], list[str]]:
    """Targets for one changed file, plus the graph nodes defined in its base version."""
    if change.is_new:
        return [Target(change.path, "new_file", lines=[f"+{n}" for n in change.added_lines])], []
    if not change.is_python:
        lines = [f"-{n}" for n in change.removed_lines] + [f"+{n}" for n in change.added_lines]
        return [Target(change.path, "non_python", lines=lines)], []

    node_ids = set(graph.nodes)
    module = resolve_module(change.path, node_ids)
    base_syms = index_symbols(base_text, module)
    base_ids = {s.graph_id for s in base_syms}
    post_syms = [] if change.is_deleted else index_symbols(apply_file_change(base_text, change), module)

    grouped: dict[tuple, Target] = {}
    sides = [("-", n, base_syms) for n in change.removed_lines] + [("+", n, post_syms) for n in change.added_lines]
    for sign, line, syms in sides:
        chain = enclosing_chain(syms, line)
        rule, node, reason = _resolve_chain(chain, node_ids, base_ids)
        symbol = chain[0].qualname if chain else None
        key = (rule, node, symbol, reason)
        if key not in grouped:
            grouped[key] = Target(change.path, rule, node=node, symbol=symbol, reason=reason)
        grouped[key].lines.append(f"{sign}{line}")

    file_nodes = sorted(s.graph_id for s in base_syms if s.graph_id in node_ids)
    return list(grouped.values()), file_nodes


def map_task(task: dict, graph: nx.Graph, snapshot: str | Path) -> TaskTargets:
    """Resolve a task's reference patch against its base snapshot and graph."""
    changes = parse_patch(task["patch"])
    needed = [c.path for c in changes if c.is_python and not c.is_new]
    texts = read_snapshot_files(snapshot, needed)
    missing = set(needed) - set(texts)
    if missing:
        raise FileNotFoundError(f"{task['instance_id']}: not in snapshot: {sorted(missing)}")

    targets: list[Target] = []
    file_nodes: dict[str, list[str]] = {}
    for change in changes:
        file_targets, nodes = map_file(change, texts.get(change.path), graph)
        targets.extend(file_targets)
        file_nodes[change.path] = nodes
    return TaskTargets(task["instance_id"], [c.path for c in changes], targets, file_nodes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Map reference patches to target files and graph nodes.")
    parser.add_argument("--data", default="data", help="dir with tasks.jsonl, graphs/, snapshots/")
    parser.add_argument("--task", action="append", help="instance_id (repeatable); default: every task with a snapshot")
    parser.add_argument("--out", help="write JSON lines here instead of stdout")
    args = parser.parse_args(argv)

    data = Path(args.data)
    tasks = load_tasks(data / "tasks.jsonl")
    ids = args.task or sorted(p.name.removesuffix(".tgz") for p in (data / "snapshots").glob("*.tgz"))
    lines = []
    for tid in ids:
        task = tasks[tid]
        graph = load_graph(graph_path(data / "graphs", task))
        lines.append(json.dumps(map_task(task, graph, data / "snapshots" / f"{tid}.tgz").to_dict()))
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
