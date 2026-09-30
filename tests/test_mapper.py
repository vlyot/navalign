from pathlib import Path

import networkx as nx
import pytest

from navalign.data import graph_path, load_graph, load_tasks
from navalign.mapper import map_file, map_task, resolve_module
from navalign.patch import parse_patch

BASE = """\
X = 1


def sync_fn():
    return 1


async def async_fn():
    def helper():
        return 2
    y = 3
    return y
"""

PATCH = """\
--- a/src/pkg/mod.py
+++ b/src/pkg/mod.py
@@ -1,4 +1,4 @@
-X = 1
+X = 2


 def sync_fn():
@@ -5,8 +5,12 @@
     return 1


+def added_fn():
+    return 0
+
+
 async def async_fn():
     def helper():
-        return 2
+        return 5
-    y = 3
+    y = 4
     return y
"""


def _graph(*ids):
    g = nx.MultiDiGraph()
    g.add_nodes_from(ids)
    return g


def test_resolve_module_strips_source_roots():
    ids = {"pkg.mod.sync_fn", "security.tutorial001.User"}
    assert resolve_module("src/pkg/mod.py", ids) == "pkg.mod"
    assert resolve_module("docs_src/security/tutorial001.py", ids) == "security.tutorial001"
    assert resolve_module("src/pkg/brand_new.py", ids) == "pkg.brand_new"


def test_rules_on_synthetic_patch():
    (change,) = parse_patch(PATCH)
    graph = _graph("pkg.mod.sync_fn", "pkg.mod.helper")  # async_fn absent, like the real graphs
    targets, file_nodes = map_file(change, BASE, graph)
    got = {(t.rule, t.node, t.symbol, t.reason): t.lines for t in targets}
    assert got == {
        ("file", None, None, "module_level"): ["-1", "+1"],
        ("exact", "pkg.mod.helper", "pkg.mod.async_fn.helper", None): ["-10", "+14"],
        ("file", None, "pkg.mod.async_fn", "async"): ["-11", "+15"],
        ("file", None, "pkg.mod.added_fn", "added"): ["+8", "+9"],
    }
    assert file_nodes == ["pkg.mod.helper", "pkg.mod.sync_fn"]


def test_enclosing_rule_for_method_added_to_class():
    base = "class K:\n    def a(self):\n        pass\n"
    patch = """\
--- a/k.py
+++ b/k.py
@@ -3 +3,4 @@
         pass
+
+    def b(self):
+        pass
"""
    (change,) = parse_patch(patch)
    targets, _ = map_file(change, base, _graph("k.K", "k.K.a"))
    (t,) = targets
    assert (t.rule, t.node, t.symbol, t.reason) == ("enclosing", "k.K", "k.K.b", "added")


# --- Acceptance: the 5 local dev tasks (needs the git-ignored data/ subset) ---------------------

DATA = Path(__file__).parent.parent / "data"
needs_data = pytest.mark.skipif(not (DATA / "tasks.jsonl").exists(), reason="local data/ subset not present")

# Written by hand from reading each patch against its base snapshot.
EXPECTED = {
    # fix adds a new script: no node can exist at base_commit
    "fastapi_15661": dict(nodes=set(), rules={"new_file"}),
    # edits the removed validator and the ServerSentEvent class body; three added top-level
    # functions fall back to file level
    "fastapi_15588": dict(
        nodes={"fastapi.sse._check_id_no_null", "fastapi.sse.ServerSentEvent"},
        rules={"exact", "file"},
        added={"fastapi.sse._check_single_line", "fastapi.sse._check_event_single_line", "fastapi.sse._check_id_valid"},
    ),
    # lines sit in async _extract_form_body (the hunk header names process_fn, which is only
    # the nearest def above); async defs are never in the graph
    "fastapi_14303": dict(nodes=set(), rules={"file"}),
    # hunk header says get_proxy, but the lines are in its parent should_bypass_proxies
    "requests_7427": dict(nodes={"requests.utils.should_bypass_proxies"}, rules={"exact"}),
    "rich_3944": dict(nodes={"rich.cells.get_character_cell_size"}, rules={"exact"}),
}


@needs_data
@pytest.mark.parametrize("instance_id", sorted(EXPECTED))
def test_dev_task_targets(instance_id):
    task = load_tasks(DATA / "tasks.jsonl")[instance_id]
    graph = load_graph(graph_path(DATA / "graphs", task))
    result = map_task(task, graph, DATA / "snapshots" / f"{instance_id}.tgz")
    exp = EXPECTED[instance_id]
    assert result.target_nodes == exp["nodes"]
    assert set(result.rules) == exp["rules"]
    assert result.target_nodes <= set(graph.nodes)
    if "added" in exp:
        assert {t.symbol for t in result.targets if t.reason == "added"} == exp["added"]


@needs_data
def test_fastapi_14303_reason_is_async():
    task = load_tasks(DATA / "tasks.jsonl")["fastapi_14303"]
    graph = load_graph(graph_path(DATA / "graphs", task))
    result = map_task(task, graph, DATA / "snapshots" / "fastapi_14303.tgz")
    assert {(t.symbol, t.reason) for t in result.targets} == {
        ("fastapi.dependencies.utils._extract_form_body", "async")
    }
