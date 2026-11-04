"""Static topology and id resolution (stage S3, DC-07a, DC-07i).

Covers the deterministic node/edge order and the structural queries the
converter and the hand-engineered features rely on: ``index_of``,
``link_neighbors`` (either direction), transitive ``dependents``, ``diameter``
and ``hops`` over the undirected ``network_link`` graph, and that every query
raises :class:`GraphConversionError` with :data:`ErrorCode.UNKNOWN_ID` for an
identifier that is not in the topology.
"""

from __future__ import annotations

import pytest

from cyberswarm.contracts import ErrorCode, Topology
from cyberswarm.graph import GraphConversionError, TopologyGraph

_V1_NODE_ORDER = [
    "node-app-01",
    "node-auth-01",
    "node-db-01",
    "node-file-01",
    "node-gw-01",
    "node-web-01",
    "node-web-02",
    "node-ws-01",
]


# --------------------------------------------------------------------------- #
# Canonical order and index resolution
# --------------------------------------------------------------------------- #
def test_node_and_edge_order_equal_the_topology_order(topology_v1: Topology) -> None:
    graph = TopologyGraph(topology_v1)
    assert graph.node_order == list(topology_v1.node_ids)
    assert graph.edge_order == list(topology_v1.edge_ids)
    assert graph.node_order == _V1_NODE_ORDER
    # The canonical order is exactly ascending by id.
    assert graph.node_order == sorted(graph.node_order)
    assert graph.edge_order == sorted(graph.edge_order)


def test_index_of_matches_position_in_node_order(topology_v1: Topology) -> None:
    graph = TopologyGraph(topology_v1)
    for position, node_id in enumerate(graph.node_order):
        assert graph.index_of(node_id) == position
    assert graph.index_of("node-gw-01") == 4


def test_criticality_of_returns_the_static_value(topology_v1: Topology) -> None:
    graph = TopologyGraph(topology_v1)
    assert graph.criticality_of("node-db-01") == pytest.approx(1.0)
    assert graph.criticality_of("node-ws-01") == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# Structural queries used by the features
# --------------------------------------------------------------------------- #
def test_link_neighbors_are_undirected_and_sorted(topology_v1: Topology) -> None:
    graph = TopologyGraph(topology_v1)
    # app-01 is linked to auth, db, file, web-01, web-02 (edges in both senses).
    assert graph.link_neighbors("node-app-01") == [
        "node-auth-01",
        "node-db-01",
        "node-file-01",
        "node-web-01",
        "node-web-02",
    ]
    assert graph.link_neighbors("node-ws-01") == ["node-file-01"]
    # Neighbour membership is symmetric over the link graph.
    for node_id in graph.node_order:
        for other in graph.link_neighbors(node_id):
            assert node_id in graph.link_neighbors(other)


def test_link_neighbors_are_never_empty(topology_v1: Topology) -> None:
    # The connected link graph with N >= 2 guarantees L(n) is non-empty.
    graph = TopologyGraph(topology_v1)
    for node_id in graph.node_order:
        assert graph.link_neighbors(node_id)


def test_dependents_are_transitive(topology_v1: Topology) -> None:
    graph = TopologyGraph(topology_v1)
    # web-01 depends on app-01, app-01 depends on auth-01 and db-01, so both
    # auth-01 and db-01 have web-01 as a transitive dependant.
    assert graph.dependents("node-auth-01") == ["node-app-01", "node-web-01"]
    assert graph.dependents("node-db-01") == ["node-app-01", "node-web-01"]
    assert graph.dependents("node-app-01") == ["node-web-01"]
    # A leaf of the dependency graph has no dependants.
    assert graph.dependents("node-web-01") == []
    assert graph.dependents("node-ws-01") == []


def test_dependents_exclude_the_node_itself(tiny_topology: Topology) -> None:
    graph = TopologyGraph(tiny_topology)
    # node-a depends on node-c, so node-c's dependants are {node-a} only.
    assert graph.dependents("node-c") == ["node-a"]
    assert "node-c" not in graph.dependents("node-c")
    assert graph.dependents("node-a") == []


def test_diameter_and_hops_over_the_link_graph(topology_v1: Topology) -> None:
    graph = TopologyGraph(topology_v1)
    assert graph.diameter == 4
    assert graph.hops("node-app-01", "node-app-01") == 0
    assert graph.hops("node-gw-01", "node-db-01") == 3
    # Distance is symmetric and bounded by the diameter.
    assert graph.hops("node-ws-01", "node-gw-01") == graph.hops("node-gw-01", "node-ws-01")
    for a in graph.node_order:
        for b in graph.node_order:
            assert 0 <= graph.hops(a, b) <= graph.diameter


# --------------------------------------------------------------------------- #
# Unknown id resolution (DC-07i)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.index_of("node-ghost"),
        lambda g: g.criticality_of("node-ghost"),
        lambda g: g.link_neighbors("node-ghost"),
        lambda g: g.dependents("node-ghost"),
        lambda g: g.hops("node-ghost", "node-app-01"),
        lambda g: g.hops("node-app-01", "node-ghost"),
    ],
)
def test_unknown_id_raises_graph_conversion_error(topology_v1: Topology, call) -> None:
    graph = TopologyGraph(topology_v1)
    with pytest.raises(GraphConversionError) as excinfo:
        call(graph)
    assert excinfo.value.code is ErrorCode.UNKNOWN_ID


def test_unknown_id_message_names_the_id(topology_v1: Topology) -> None:
    graph = TopologyGraph(topology_v1)
    with pytest.raises(GraphConversionError, match="node-ghost"):
        graph.index_of("node-ghost")
