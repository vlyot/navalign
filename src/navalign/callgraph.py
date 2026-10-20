"""Build a code graph from source with `ast`, in the competition's node-link format.

The shipped graphs leave out async functions and miss many calls (requests' `Session.request` has
three call edges, none of them to `self.send`). This builder resolves each call site statically,
trying in order: the defs nested in the calling function, the module's defs and imports (relative
imports and package re-exports followed), `self`/`cls`/`super()` methods through same-repo base
classes, and finally a repo-wide unique name. A call that stays ambiguous makes no edge.

Node ids follow `symbols`: `<module>.<class chain>.<name>`, enclosing functions dropped. Calling a
class adds an edge to the class and to its `__init__`. Module-level code has no node, so its calls
make no edge. Like the shipped graphs, whose `calls` edges also join a class to its methods and to
its base classes, the graph has `contains` (class → member) and `inherits` (class → same-repo base)
edges; distances use every edge.

    navalign graph --tasks tasks.jsonl --snapshots DIR --out DIR [--task ID]
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .build import snapshot_path
from .data import graph_path, iter_python_files, load_tasks
from .mapper import resolve_module

MAX_HOPS = 8  # re-export and base-class chains longer than this are not followed
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
# never resolved by unique name: a call to one is far likelier the builtin than a repo def
BUILTIN_NAMES = frozenset(dir(builtins)) | frozenset(
    name for t in (object, list, dict, set, str, bytes, int, float, tuple, frozenset) for name in dir(t)
)


@dataclass
class _Module:
    name: str
    path: str
    tree: ast.Module
    lines: list[str]
    bindings: dict[str, str] = field(default_factory=dict)  # local name → dotted name it refers to


@dataclass
class _Scope:
    graph_id: str | None  # innermost function/class node; None at module level
    class_id: str | None  # enclosing class, for self/cls
    local_defs: dict[str, str]  # defs nested in the enclosing functions


@dataclass
class CallGraph:
    name: str
    nodes: dict[str, str]  # graph id → source text
    edges: dict[tuple[str, str], str]  # (source, target) → calls | contains | inherits
    skipped_files: list[str] = field(default_factory=list)

    def to_node_link(self) -> dict:
        return {
            "directed": True,
            "multigraph": True,
            "graph": {"repo_name": self.name, "builder": "navalign.callgraph"},
            "nodes": [{"name": n, "text": t, "id": n} for n, t in sorted(self.nodes.items())],
            "edges": [{"type": kind, "source": a, "target": b, "key": 0} for (a, b), kind in sorted(self.edges.items())],
        }


def _package(module: _Module) -> str:
    return module.name if Path(module.path).name == "__init__.py" else module.name.rpartition(".")[0]


def _import_bindings(module: _Module) -> dict[str, str]:
    """Names bound by every import in the module, at any depth (a function-level import counts too)."""
    bindings = {}
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bindings[alias.asname] = alias.name
                else:
                    head = alias.name.split(".")[0]
                    bindings[head] = head
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                pkg = _package(module).split(".") if _package(module) else []
                pkg = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                base = ".".join([*pkg, base] if base else pkg)
            for alias in node.names:
                if alias.name != "*":
                    bindings[alias.asname or alias.name] = f"{base}.{alias.name}" if base else alias.name
    return bindings


def _nested_defs(func: ast.AST, class_prefix: str) -> dict[str, str]:
    """Defs anywhere in a function's body (not inside a nested def), by name → graph id."""
    found, stack = {}, list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        if isinstance(node, DEFS):
            found[node.name] = f"{class_prefix}.{node.name}"
        else:
            stack.extend(ast.iter_child_nodes(node))
    return found


def _chain(expr: ast.expr) -> list[str] | None:
    """`a.b.c` → ["a", "b", "c"]; None for anything that isn't a dotted name."""
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        return [expr.id, *reversed(parts)]
    return None


class _Builder:
    def __init__(self, name: str, modules: list[_Module]):
        self.modules = {m.name: m for m in modules}
        self.graph = CallGraph(name, {}, {})
        self.bases: dict[str, list[str]] = {}
        self.base_exprs: dict[str, tuple[_Module, _Scope, list[ast.expr]]] = {}
        self.short: dict[str, set[str]] = {}

    # pass 1: nodes, module bindings, class bases (unresolved)
    def collect(self) -> None:
        for m in self.modules.values():
            m.bindings = _import_bindings(m)
            self._collect(m, m.tree, m.name, _Scope(None, None, {}))
        for nid in self.graph.nodes:
            self.short.setdefault(nid.rsplit(".", 1)[-1], set()).add(nid)

    def _collect(self, m: _Module, node: ast.AST, class_prefix: str, scope: _Scope) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, DEFS):
                gid = f"{class_prefix}.{child.name}"
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                self.graph.nodes.setdefault(gid, "\n".join(m.lines[start - 1: child.end_lineno]))
                if scope.graph_id is None:
                    m.bindings.setdefault(child.name, gid)
                if isinstance(child, ast.ClassDef):
                    self.base_exprs.setdefault(gid, (m, scope, child.bases))
                    self._collect(m, child, gid, _Scope(gid, gid, scope.local_defs))
                else:
                    self._collect(m, child, class_prefix, _Scope(gid, scope.class_id, scope.local_defs))
            else:
                self._collect(m, child, class_prefix, scope)

    # name resolution
    def lookup(self, dotted: str, hops: int = 0) -> str | None:
        """Graph node a fully qualified dotted name refers to, following re-exports and inherited members."""
        if dotted in self.graph.nodes:
            return dotted
        if hops > MAX_HOPS:
            return None
        parts = dotted.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in self.graph.nodes:  # a class: the rest is a member
                found = self.member(prefix, parts[i]) if i == len(parts) - 1 else None
                return found
            if prefix in self.modules:
                target = self.modules[prefix].bindings.get(parts[i])
                if target and target != dotted:
                    return self.lookup(".".join([target, *parts[i + 1:]]), hops + 1)
                return None
        return None

    def member(self, class_id: str, attr: str, hops: int = 0) -> str | None:
        """`class_id.attr`, else the first base class (same repo, depth first) that defines it."""
        if f"{class_id}.{attr}" in self.graph.nodes:
            return f"{class_id}.{attr}"
        if hops > MAX_HOPS:
            return None
        for base in self.bases.get(class_id, []):
            found = self.member(base, attr, hops + 1)
            if found:
                return found
        return None

    def resolve(self, func: ast.expr, m: _Module, scope: _Scope) -> str | None:
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call)
                and isinstance(func.value.func, ast.Name) and func.value.func.id == "super" and scope.class_id):
            return next((f for b in self.bases.get(scope.class_id, []) if (f := self.member(b, func.attr))), None)
        parts = _chain(func)
        if parts is None:
            return self._unique(func.attr) if isinstance(func, ast.Attribute) else None
        head, rest = parts[0], parts[1:]
        if head in ("self", "cls") and scope.class_id and len(rest) == 1:
            found = self.member(scope.class_id, rest[0])
            if found:
                return found
        elif head in scope.local_defs:
            found = self.lookup(".".join([scope.local_defs[head], *rest]))
            if found:
                return found
        elif head in m.bindings:
            found = self.lookup(".".join([m.bindings[head], *rest]))
            if found:
                return found
        return self._unique(parts[-1])

    def _unique(self, name: str) -> str | None:
        if name in BUILTIN_NAMES:
            return None
        candidates = self.short.get(name, set())
        return next(iter(candidates)) if len(candidates) == 1 else None

    # pass 2: bases, then calls
    def link(self) -> None:
        for cid, (m, scope, exprs) in self.base_exprs.items():
            resolved = [self.resolve(e, m, scope) for e in exprs]
            self.bases[cid] = [b for b in resolved if b in self.base_exprs and b != cid]
            for base in self.bases[cid]:
                self._edge(cid, base, "inherits")
        for m in self.modules.values():
            self._calls(m, m.tree, m.name, _Scope(None, None, {}))

    def _calls(self, m: _Module, node: ast.AST, class_prefix: str, scope: _Scope) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, DEFS):
                gid = f"{class_prefix}.{child.name}"
                if scope.graph_id is not None and scope.graph_id == scope.class_id:
                    self._edge(scope.graph_id, gid, "contains")
                if isinstance(child, ast.ClassDef):
                    self._calls(m, child, gid, _Scope(gid, gid, scope.local_defs))
                else:
                    local = scope.local_defs | _nested_defs(child, class_prefix)
                    self._calls(m, child, class_prefix, _Scope(gid, scope.class_id, local))
                continue
            if isinstance(child, ast.Call) and scope.graph_id is not None:
                callee = self.resolve(child.func, m, scope)
                if callee:
                    self._edge(scope.graph_id, callee)
                    if callee in self.base_exprs:
                        init = self.member(callee, "__init__")
                        if init:
                            self._edge(scope.graph_id, init)
            self._calls(m, child, class_prefix, scope)

    def _edge(self, a: str, b: str, kind: str = "calls") -> None:
        if a != b:
            self.graph.edges.setdefault((a, b), kind)


def build_callgraph(snapshot: str | Path, name: str = "") -> CallGraph:
    """The call graph of every parseable `.py` file in a `.tgz` snapshot or a directory."""
    modules, skipped = [], []
    for path, text in iter_python_files(snapshot):
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            skipped.append(path)
            continue
        modules.append(_Module(resolve_module(path, set(), set()), path, tree, text.splitlines()))
    builder = _Builder(name, modules)
    builder.collect()
    builder.link()
    builder.graph.skipped_files = sorted(skipped)
    return builder.graph


def write_graph(graph: CallGraph, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph.to_node_link(), f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="navalign graph", description="Build ast call graphs, one per repository commit.")
    parser.add_argument("--tasks", required=True, help="tasks.jsonl (repo and base_commit per task)")
    parser.add_argument("--snapshots", required=True, help="dir with <instance_id>.tgz or <instance_id>/")
    parser.add_argument("--task", action="append", help="instance_id (repeatable); default: every task with a snapshot")
    parser.add_argument("--out", required=True, help="graph dir, files named <repo>_<base_commit>.json")
    args = parser.parse_args(argv)

    tasks = load_tasks(args.tasks)
    snapshots, out = Path(args.snapshots), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = args.task or sorted(tid for tid in tasks if snapshot_path(snapshots, tid))
    built: set[Path] = set()
    for n, tid in enumerate(ids, 1):
        target = graph_path(out, tasks[tid])
        if target in built:
            continue
        snapshot = snapshot_path(snapshots, tid)
        if snapshot is None:
            print(f"[{n}/{len(ids)}] {tid}: no snapshot", file=sys.stderr)
            return 1
        graph = build_callgraph(snapshot, target.stem)
        write_graph(graph, target)
        built.add(target)
        print(f"[{n}/{len(ids)}] {target.name}: {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
              f"{len(graph.skipped_files)} files skipped", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
