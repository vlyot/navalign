from pathlib import Path

import networkx as nx
import pytest

from navalign.data import graph_path, load_graph, load_tasks
from navalign.distance import DistanceMap, distance_map, graph_cap, undirected_calls
from navalign.mapper import map_task


def _calls(*edges, nodes=()):
    g = nx.MultiDiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges, type="calls")
    return g


def test_path_graph_counts_hops_from_target():
    u = undirected_calls(_calls(("a", "b"), ("b", "c"), ("c", "d")))
    dm = distance_map(u, {"a"}, graph_cap(u))
    assert [dm.distance(n) for n in "abcd"] == [0, 1, 2, 3]


def test_nearest_of_several_targets():
    u = undirected_calls(_calls(("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")))
    dm = distance_map(u, {"a", "e"}, graph_cap(u))
    assert [dm.distance(n) for n in "abcde"] == [0, 1, 2, 1, 0]


def test_edge_direction_is_ignored():
    u = undirected_calls(_calls(("caller", "target"), ("other", "caller")))
    dm = distance_map(u, {"target"}, graph_cap(u))
    assert dm.distance("caller") == 1
    assert dm.distance("other") == 2


def test_disconnected_node_gets_cap():
    u = undirected_calls(_calls(("a", "b"), nodes=["lonely"]))
    dm = distance_map(u, {"a"}, graph_cap(u))
    assert dm.cap == 2
    assert dm.distance("lonely") == 2
    assert dm.distance("not_a_node") == 2
    assert "lonely" not in dm.distances


def test_cap_is_largest_component_diameter_plus_one():
    # component 1 is a 4-node path (diameter 3), component 2 a single edge (diameter 1)
    u = undirected_calls(_calls(("a", "b"), ("b", "c"), ("c", "d"), ("x", "y")))
    assert graph_cap(u) == 4


def test_cap_of_edgeless_graph_is_one():
    assert graph_cap(undirected_calls(_calls(nodes=["a", "b"]))) == 1


def test_self_loops_and_parallel_edges_do_not_change_distances():
    plain = undirected_calls(_calls(("a", "b"), ("b", "c")))
    noisy = undirected_calls(_calls(("a", "b"), ("a", "b"), ("b", "a"), ("b", "b"), ("b", "c")))
    assert set(noisy.edges) == set(plain.edges)
    assert distance_map(noisy, {"a"}, 3).distances == distance_map(plain, {"a"}, 3).distances


def test_target_missing_from_graph_raises():
    u = undirected_calls(_calls(("a", "b")))
    with pytest.raises(ValueError, match="ghost"):
        distance_map(u, {"a", "ghost"}, 2)


def test_empty_targets_give_empty_map():
    u = undirected_calls(_calls(("a", "b")))
    dm = distance_map(u, set(), 2)
    assert dm.distances == {} and dm.targets == []
    assert dm.distance("a") == 2


def test_round_trip():
    u = undirected_calls(_calls(("a", "b"), nodes=["c"]))
    dm = distance_map(u, {"a"}, graph_cap(u), instance_id="t_1", graph_file="g.json")
    data = dm.to_dict()
    assert data["n_reachable"] == 2
    assert DistanceMap.from_dict(data) == dm


# --- Dev tasks (needs the git-ignored data/ subset) ----------------------------------------------

DATA = Path(__file__).parent.parent / "data"
needs_data = pytest.mark.skipif(not (DATA / "tasks.jsonl").exists(), reason="local data/ subset not present")

CAPS = {"fastapi": 17, "requests": 14, "rich": 13}


def _dev(instance_id):
    task = load_tasks(DATA / "tasks.jsonl")[instance_id]
    graph = load_graph(graph_path(DATA / "graphs", task))
    targets = map_task(task, graph, DATA / "snapshots" / f"{instance_id}.tgz").target_nodes
    u = undirected_calls(graph)
    return distance_map(u, targets, graph_cap(u), instance_id)


@needs_data
@pytest.mark.parametrize("instance_id", ["fastapi_15588", "requests_7427", "rich_3944"])
def test_dev_targets_at_zero_and_cap_per_repo(instance_id):
    dm = _dev(instance_id)
    assert dm.targets
    assert all(dm.distance(t) == 0 for t in dm.targets)
    assert dm.cap == CAPS[instance_id.split("_")[0]]
    assert max(dm.distances.values()) < dm.cap


@needs_data
def test_fastapi_15588_isolated_target_and_class_component():
    dm = _dev("fastapi_15588")
    assert dm.distance("fastapi.sse._check_id_no_null") == 0
    # ServerSentEvent's component is reached, so more nodes than the two targets have finite distance
    assert dm.n_reachable > 2


@needs_data
@pytest.mark.parametrize("instance_id", ["fastapi_15661", "fastapi_14303"])
def test_dev_tasks_without_target_nodes(instance_id):
    dm = _dev(instance_id)
    assert dm.targets == [] and dm.distances == {}
