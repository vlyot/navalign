"""Distance from every graph node to the nearest target node (Methodology step 4, node level).

Distances are shortest paths on the `calls` graph with edge direction ignored, because an agent
can follow a call either way. Nodes that can't reach any target get the cap
C = (largest finite distance in the graph) + 1, which is the largest component diameter + 1.
The cap is a property of the graph, so every task on the same commit shares it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import networkx as nx


def undirected_calls(graph: nx.Graph) -> nx.Graph:
    """Simple undirected copy of `graph`: every node kept, parallel edges merged, self-loops dropped."""
    u = nx.Graph()
    u.add_nodes_from(graph.nodes)
    u.add_edges_from((a, b) for a, b in graph.edges() if a != b)
    return u


def graph_cap(u: nx.Graph) -> int:
    """Largest finite shortest-path distance in `u`, plus one (1 for a graph with no edges)."""
    diameters = [
        nx.diameter(u.subgraph(c), usebounds=True)
        for c in nx.connected_components(u)
        if len(c) > 1
    ]
    return max(diameters, default=0) + 1


@dataclass
class DistanceMap:
    instance_id: str
    graph_file: str
    targets: list[str]
    cap: int
    n_nodes: int
    distances: dict[str, int] = field(default_factory=dict)  # reachable nodes only

    @property
    def n_reachable(self) -> int:
        return len(self.distances)

    def distance(self, node: str) -> int:
        """Hops to the nearest target, or the cap if unreachable or not a graph node."""
        return self.distances.get(node, self.cap)

    def to_dict(self) -> dict:
        return asdict(self) | {"n_reachable": self.n_reachable}

    @classmethod
    def from_dict(cls, data: dict) -> DistanceMap:
        return cls(**{k: v for k, v in data.items() if k != "n_reachable"})


def distance_map(u: nx.Graph, targets: set[str], cap: int, instance_id: str = "", graph_file: str = "") -> DistanceMap:
    """Multi-source BFS from `targets` over the undirected graph `u`.

    Empty `targets` gives an empty map: the task has no node-level score.
    """
    missing = targets - set(u.nodes)
    if missing:
        raise ValueError(f"{instance_id}: targets not in graph: {sorted(missing)}")
    distances = nx.multi_source_dijkstra_path_length(u, targets) if targets else {}
    return DistanceMap(
        instance_id=instance_id,
        graph_file=graph_file,
        targets=sorted(targets),
        cap=cap,
        n_nodes=u.number_of_nodes(),
        distances={n: int(d) for n, d in sorted(distances.items())},
    )
