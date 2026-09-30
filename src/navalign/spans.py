"""Locate every graph node in the snapshot source: node → (file, first line, last line).

The graphs carry no line numbers, but the node-level read rule needs them (a read covers the
nodes whose spans overlap the lines shown). The index is built once where the snapshots are,
so scoring can run without them.

Nodes that no file defines under the graph's naming, or that two files both claim, are listed
as `unlocated` rather than guessed. Known case: in requests, test methods appear in the graph
without their test class (`tests.test_requests.test_unicode_get`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

from .data import iter_python_files
from .mapper import module_prefixes, resolve_module
from .symbols import index_symbols


@dataclass(frozen=True)
class NodeSpan:
    node: str
    start: int  # first line, including decorators
    end: int


@dataclass
class SpanIndex:
    instance_id: str
    n_nodes: int
    spans: dict[str, list[NodeSpan]]  # repo-relative file → nodes defined in it, in source order
    unlocated: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)  # files `ast` could not parse

    @property
    def coverage(self) -> float:
        """Share of graph nodes located in some file."""
        return 1 - len(self.unlocated) / self.n_nodes if self.n_nodes else 0.0

    def nodes_in(self, file: str, start: int | None = None, end: int | None = None) -> set[str]:
        """Nodes in `file` whose span overlaps lines [start, end]; the whole file if no range."""
        lo = start if start is not None else 1
        hi = end if end is not None else float("inf")
        return {s.node for s in self.spans.get(file, []) if s.start <= hi and s.end >= lo}

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "n_nodes": self.n_nodes,
            "coverage": round(self.coverage, 4),
            "spans": {f: [[s.node, s.start, s.end] for s in spans] for f, spans in self.spans.items()},
            "unlocated": self.unlocated,
            "skipped_files": self.skipped_files,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SpanIndex:
        return cls(
            instance_id=data["instance_id"],
            n_nodes=data["n_nodes"],
            spans={f: [NodeSpan(*s) for s in spans] for f, spans in data["spans"].items()},
            unlocated=data["unlocated"],
            skipped_files=data["skipped_files"],
        )


def build_span_index(snapshot: str | Path, graph: nx.Graph, instance_id: str = "") -> SpanIndex:
    """Index every `.py` file of the snapshot and keep the symbols that are graph nodes."""
    node_ids = set(graph.nodes)
    prefixes = module_prefixes(node_ids)
    found: dict[str, list[tuple[str, NodeSpan]]] = {}
    skipped: list[str] = []
    for path, text in iter_python_files(snapshot):
        try:
            symbols = index_symbols(text, resolve_module(path, node_ids, prefixes))
        except SyntaxError:
            skipped.append(path)
            continue
        for sym in symbols:
            if sym.graph_id in node_ids:
                found.setdefault(sym.graph_id, []).append((path, NodeSpan(sym.graph_id, sym.start, sym.end)))

    spans: dict[str, list[NodeSpan]] = {}
    for places in found.values():
        # one file may define a node more than once (conditional defs, collapsed nested functions)
        if len({p for p, _ in places}) == 1:
            spans.setdefault(places[0][0], []).extend(span for _, span in places)
    for file_spans in spans.values():
        file_spans.sort(key=lambda s: (s.start, s.end))
    located = {s.node for file_spans in spans.values() for s in file_spans}
    return SpanIndex(
        instance_id=instance_id,
        n_nodes=len(node_ids),
        spans=dict(sorted(spans.items())),
        unlocated=sorted(node_ids - located),
        skipped_files=sorted(skipped),
    )
