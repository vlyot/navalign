import io
import tarfile
from pathlib import Path

import networkx as nx
import pytest

from navalign.data import graph_path, iter_python_files, load_graph, load_tasks
from navalign.mapper import map_task
from navalign.spans import NodeSpan, SpanIndex, build_span_index

MOD = """\
class K:
    def method(self):
        def inner():
            return 1
        return inner()


def top():
    return 2
"""

DUP = """\
if FLAG:
    def f():
        return 1
else:
    def f():
        return 2
"""


def _graph(*ids):
    g = nx.MultiDiGraph()
    g.add_nodes_from(ids)
    return g


@pytest.fixture
def snapshot_dir(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "mod.py").write_text(MOD, encoding="utf-8")
    (tmp_path / "src" / "pkg" / "dup.py").write_text(DUP, encoding="utf-8")
    (tmp_path / "src" / "pkg" / "broken.py").write_text("def (:\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("not python", encoding="utf-8")
    return tmp_path


def test_iter_python_files_dir_and_tgz_agree(snapshot_dir, tmp_path_factory):
    archive = tmp_path_factory.mktemp("tgz") / "snap.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        for f in sorted(snapshot_dir.rglob("*")):
            if f.is_file():
                data = f.read_bytes()
                info = tarfile.TarInfo("./" + f.relative_to(snapshot_dir).as_posix())
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
    from_dir = sorted(iter_python_files(snapshot_dir))
    assert [p for p, _ in from_dir] == ["src/pkg/broken.py", "src/pkg/dup.py", "src/pkg/mod.py"]
    assert sorted(iter_python_files(archive)) == from_dir


def test_graph_naming_and_spans(snapshot_dir):
    graph = _graph("pkg.mod.K", "pkg.mod.K.method", "pkg.mod.K.inner", "pkg.mod.top", "pkg.dup.f")
    index = build_span_index(snapshot_dir, graph, "t_1")
    assert index.spans["src/pkg/mod.py"] == [
        NodeSpan("pkg.mod.K", 1, 5),
        NodeSpan("pkg.mod.K.method", 2, 5),
        NodeSpan("pkg.mod.K.inner", 3, 4),  # enclosing method dropped, class kept
        NodeSpan("pkg.mod.top", 8, 9),
    ]
    assert index.spans["src/pkg/dup.py"] == [NodeSpan("pkg.dup.f", 2, 3), NodeSpan("pkg.dup.f", 5, 6)]
    assert index.skipped_files == ["src/pkg/broken.py"]
    assert index.unlocated == [] and index.coverage == 1.0


def test_range_inside_method_includes_its_class(snapshot_dir):
    index = build_span_index(snapshot_dir, _graph("pkg.mod.K", "pkg.mod.K.method", "pkg.mod.top"))
    assert index.nodes_in("src/pkg/mod.py", 4, 4) == {"pkg.mod.K", "pkg.mod.K.method"}
    assert index.nodes_in("src/pkg/mod.py", 6, 7) == set()
    assert index.nodes_in("src/pkg/mod.py", 5, 8) == {"pkg.mod.K", "pkg.mod.K.method", "pkg.mod.top"}


def test_no_range_returns_whole_file(snapshot_dir):
    index = build_span_index(snapshot_dir, _graph("pkg.mod.K", "pkg.mod.top"))
    assert index.nodes_in("src/pkg/mod.py") == {"pkg.mod.K", "pkg.mod.top"}
    assert index.nodes_in("src/pkg/mod.py", start=8) == {"pkg.mod.top"}


def test_unknown_file_is_empty(snapshot_dir):
    index = build_span_index(snapshot_dir, _graph("pkg.mod.K"))
    assert index.nodes_in("src/pkg/nope.py") == set()
    assert index.nodes_in("README.md") == set()


def test_nodes_not_defined_anywhere_are_unlocated(snapshot_dir):
    index = build_span_index(snapshot_dir, _graph("pkg.mod.K", "pkg.mod.method", "tests.test_x.test_y"))
    assert index.unlocated == ["pkg.mod.method", "tests.test_x.test_y"]
    assert index.coverage == pytest.approx(1 / 3)


def test_node_claimed_by_two_files_is_unlocated(tmp_path):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "util.py").write_text("def helper():\n    pass\n", encoding="utf-8")
    # neither a.util nor b.util prefixes a node, so both files resolve to module "util"
    index = build_span_index(tmp_path, _graph("util.helper"))
    assert index.unlocated == ["util.helper"]
    assert index.spans == {}


def test_round_trip(snapshot_dir):
    index = build_span_index(snapshot_dir, _graph("pkg.mod.K", "pkg.mod.top", "missing.node"), "t_1")
    assert SpanIndex.from_dict(index.to_dict()) == index


# --- Dev tasks (needs the git-ignored data/ subset) ----------------------------------------------

DATA = Path(__file__).parent.parent / "data"
needs_data = pytest.mark.skipif(not (DATA / "tasks.jsonl").exists(), reason="local data/ subset not present")


def _dev(instance_id):
    task = load_tasks(DATA / "tasks.jsonl")[instance_id]
    graph = load_graph(graph_path(DATA / "graphs", task))
    snapshot = DATA / "snapshots" / f"{instance_id}.tgz"
    return map_task(task, graph, snapshot), build_span_index(snapshot, graph, instance_id)


@needs_data
@pytest.mark.parametrize("instance_id", ["fastapi_15588", "requests_7427", "rich_3944"])
def test_dev_targets_located_in_mapped_file(instance_id):
    targets, index = _dev(instance_id)
    for t in targets.targets:
        if t.node:
            assert t.node in index.nodes_in(t.file), (t.node, t.file)


@needs_data
@pytest.mark.parametrize("instance_id,minimum", [("rich_3944", 1.0), ("fastapi_14303", 0.99)])
def test_dev_span_coverage(instance_id, minimum):
    _, index = _dev(instance_id)
    assert index.coverage >= minimum


@needs_data
def test_requests_unlocated_nodes_are_tests():
    _, index = _dev("requests_7427")
    assert index.unlocated
    assert all(n.startswith("tests.") for n in index.unlocated)
