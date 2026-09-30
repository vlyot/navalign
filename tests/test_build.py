import csv
import json
from pathlib import Path

import networkx as nx
import pytest
from networkx.readwrite import json_graph

from navalign.build import main

MOD = """\
def target():
    return 1


def caller():
    return target()


def lonely():
    return 0
"""

PATCH = """\
--- a/src/pkg/mod.py
+++ b/src/pkg/mod.py
@@ -1,2 +1,2 @@
 def target():
-    return 1
+    return 2
"""


@pytest.fixture
def data_dir(tmp_path):
    data = tmp_path / "data"
    (data / "graphs").mkdir(parents=True)
    (data / "snapshots" / "pkg_1" / "src" / "pkg").mkdir(parents=True)
    (data / "snapshots" / "pkg_1" / "src" / "pkg" / "mod.py").write_bytes(MOD.encode())
    task = {"instance_id": "pkg_1", "repo": "org/pkg", "base_commit": "abc", "patch": PATCH}
    (data / "tasks.jsonl").write_text(json.dumps(task) + "\n", encoding="utf-8")
    g = nx.MultiDiGraph()
    g.add_nodes_from(["pkg.mod.target", "pkg.mod.caller", "pkg.mod.lonely"])
    g.add_edge("pkg.mod.caller", "pkg.mod.target", type="calls")
    (data / "graphs" / "pkg_abc.json").write_text(json.dumps(json_graph.node_link_data(g, edges="edges")), encoding="utf-8")
    return data


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _summary(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_build_writes_all_artifacts(data_dir, tmp_path):
    out = tmp_path / "out"
    assert main(["--data", str(data_dir), "--out", str(out)]) == 0

    (targets,) = _jsonl(out / "targets.jsonl")
    assert targets["target_nodes"] == ["pkg.mod.target"]
    (spans,) = _jsonl(out / "spans.jsonl")
    assert spans["coverage"] == 1.0
    assert spans["spans"]["src/pkg/mod.py"][0] == ["pkg.mod.target", 1, 2]
    (dist,) = _jsonl(out / "distances.jsonl")
    assert dist["cap"] == 2
    assert dist["distances"] == {"pkg.mod.caller": 1, "pkg.mod.target": 0}
    (row,) = _summary(out / "summary.csv")
    assert row["error"] == "" and row["n_target_nodes"] == "1" and row["graph_file"] == "pkg_abc.json"


def test_failing_task_is_recorded_and_run_continues(data_dir, tmp_path):
    out = tmp_path / "out"
    assert main(["--data", str(data_dir), "--task", "ghost_9", "--task", "pkg_1", "--out", str(out)]) == 1
    rows = {r["instance_id"]: r for r in _summary(out / "summary.csv")}
    assert rows["ghost_9"]["error"].startswith("KeyError")
    assert rows["pkg_1"]["error"] == ""
    assert len(_jsonl(out / "targets.jsonl")) == 1


DATA = Path(__file__).parent.parent / "data"


@pytest.mark.skipif(not (DATA / "tasks.jsonl").exists(), reason="local data/ subset not present")
def test_build_on_dev_task(tmp_path):
    out = tmp_path / "out"
    assert main(["--data", str(DATA), "--task", "rich_3944", "--out", str(out)]) == 0
    for name in ("targets.jsonl", "spans.jsonl", "distances.jsonl"):
        assert len(_jsonl(out / name)) == 1
    (row,) = _summary(out / "summary.csv")
    assert row["cap"] == "13" and row["span_coverage"] == "1.0"
    assert _jsonl(out / "distances.jsonl")[0]["distances"]["rich.cells.get_character_cell_size"] == 0
