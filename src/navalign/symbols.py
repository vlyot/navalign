"""Index Python source into functions/classes with line spans and graph-style node ids.

The competition graphs name a symbol `<module>.<class chain>.<name>`: enclosing classes are kept,
enclosing functions are dropped (a function nested in `should_bypass_proxies` in requests/utils.py
is `requests.utils.get_proxy`). Async functions are never in the graphs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class Symbol:
    graph_id: str  # id the graph would use for this symbol
    qualname: str  # full dotted chain including enclosing functions
    kind: str  # "function" or "class"
    is_async: bool
    start: int  # first line, including decorators
    end: int
    parent: "Symbol | None"


def module_candidates(path: str) -> list[str]:
    """Dotted module names for a repo path, longest first (`src/requests/utils.py` →
    `src.requests.utils`, `requests.utils`, `utils`). `__init__.py` maps to its package."""
    parts = path.replace("\\", "/").removesuffix(".py").split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return [".".join(parts[i:]) for i in range(len(parts)) if parts[i:]]


def index_symbols(source: str, module: str) -> list[Symbol]:
    """All functions and classes in `source`, outermost first."""
    symbols: list[Symbol] = []

    def walk(node: ast.AST, class_prefix: str, qual_prefix: str, parent: Symbol | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                is_class = isinstance(child, ast.ClassDef)
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                sym = Symbol(
                    graph_id=f"{class_prefix}.{child.name}",
                    qualname=f"{qual_prefix}.{child.name}",
                    kind="class" if is_class else "function",
                    is_async=isinstance(child, ast.AsyncFunctionDef),
                    start=start,
                    end=child.end_lineno,
                    parent=parent,
                )
                symbols.append(sym)
                walk(child, sym.graph_id if is_class else class_prefix, sym.qualname, sym)
            else:
                walk(child, class_prefix, qual_prefix, parent)

    walk(ast.parse(source), module, module, None)
    return symbols


def enclosing_chain(symbols: list[Symbol], line: int) -> list[Symbol]:
    """Symbols whose span contains `line`, innermost first (empty for module-level lines)."""
    containing = [s for s in symbols if s.start <= line <= s.end]
    return sorted(containing, key=lambda s: s.end - s.start)
