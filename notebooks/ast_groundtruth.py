# Source of the Kaggle notebook kaic87/navalign-ground-truth-build, version "ast" (CPU, no Internet).
# Inputs: the gemma-4-developer-agent competition data and the kaic87/navalign-src and
# kaic87/navalign-groundtruth Datasets. Each `# %%` block is one cell.
#
# Builds, for all 129 tasks: an `ast` call graph per repository commit (`navalign graph`), the ground
# truth on those graphs (`groundtruth-ast/`), the issue-mention and BM25 read rankings on the shipped
# graphs (`rankings.jsonl`), and a per-commit comparison of the two graphs (`graph_compare.csv`). The
# `ast` graphs themselves are derived from the competition snapshots and too large to keep, so they
# are written to /tmp and left out of the output.
#
# Download the output with
#   kaggle kernels output kaic87/navalign-ground-truth-build -p results/groundtruth-ast-out

# %%
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

INPUT = Path("/kaggle/input")
COMPETITION = INPUT / "competitions" / "gemma-4-developer-agent"
SRC = next(INPUT.rglob("navalign/cli.py")).parents[1]
GROUNDTRUTH = next(INPUT.rglob("navalign_groundtruth.md")).parent
OUT = Path("/kaggle/working")
AST_GRAPHS = Path("/tmp/ast-graphs")

sys.path.insert(0, str(SRC))
try:
    import unidiff  # noqa: F401
except ImportError:  # notebooks attached to a competition have no Internet: use the wheel in navalign-src
    wheels = next(INPUT.rglob("unidiff-*.whl")).parent
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", str(wheels), "unidiff"],
                   check=True)

from navalign import cli
from navalign.data import graph_path, load_graph, load_tasks


def navalign(*args, check=True):
    code = cli.main([str(a) for a in args])
    if code and check:
        raise RuntimeError(f"navalign {args[0]} exited with {code}")
    return code


# %%
navalign("graph", "--tasks", COMPETITION / "tasks.jsonl", "--snapshots", COMPETITION / "snapshots", "--out", AST_GRAPHS)

# %%
# A task that fails is recorded in summary.csv and the build carries on (exit code 1).
navalign("build", "--data", COMPETITION, "--graphs", AST_GRAPHS, "--out", OUT / "groundtruth-ast", check=False)
summary = pd.read_csv(OUT / "groundtruth-ast" / "summary.csv")
print(len(summary), "tasks;", int(summary["error"].notna().sum()), "errors")
print(summary[summary["error"].notna()][["instance_id", "error"]].to_string())

# %%
navalign("validity", "rankings", "--tasks", COMPETITION / "tasks.jsonl", "--graphs", COMPETITION / "graphs",
         "--groundtruth", GROUNDTRUTH, "--out", OUT / "rankings.jsonl")

# %%
# Per commit: how many shipped edges the ast graph also has (on the nodes both name alike), how many
# nodes have no edge at all, and the ast graph's edge kinds.
def undirected(edges):
    return {frozenset(e) for e in edges if e[0] != e[1]}


rows, seen = [], set()
for task in load_tasks(COMPETITION / "tasks.jsonl").values():
    shipped_path, ast_path = graph_path(COMPETITION / "graphs", task), graph_path(AST_GRAPHS, task)
    if shipped_path.name in seen:
        continue
    seen.add(shipped_path.name)
    shipped, built = load_graph(shipped_path), load_graph(ast_path)
    common = set(shipped.nodes) & set(built.nodes)
    s_edges, a_edges = undirected(shipped.edges()), undirected(built.edges())
    s_common = {e for e in s_edges if e <= common}
    kinds = Counter(d["type"] for *_, d in built.edges(data=True))
    rows.append({
        "graph_file": shipped_path.name, "repo": task["repo"].split("/")[-1],
        "shipped_nodes": shipped.number_of_nodes(), "ast_nodes": built.number_of_nodes(), "common_nodes": len(common),
        "shipped_edges": len(s_edges), "ast_edges": len(a_edges),
        "shipped_edges_in_ast": len(s_common & a_edges) / len(s_common) if s_common else None,
        "shipped_isolated": sum(1 for n in shipped.nodes if shipped.degree(n) == 0) / shipped.number_of_nodes(),
        "ast_isolated": sum(1 for n in built.nodes if built.degree(n) == 0) / built.number_of_nodes(),
    } | {f"ast_{k}": kinds[k] for k in ("calls", "contains", "inherits")})
compare = pd.DataFrame(rows)
compare.to_csv(OUT / "graph_compare.csv", index=False)
print(compare.groupby("repo").mean(numeric_only=True).round(3).T.to_string())
