"""Hand-engineered ``graph-derived-v1`` features (CONTRACTS 10.2, DESIGN 4.5).

:func:`compute_features` turns one :class:`~cyberswarm.contracts.GraphSnapshot`
and its :class:`~cyberswarm.contracts.Topology` into a
:class:`~cyberswarm.contracts.GraphFeatureRecord`. It receives **no** observation
and **no** ground-truth object: the snapshot already carries the agent-visible
values and the masks, and the topology carries the public structure. Features
0-3 depend on the topology only (mask always 1); features 4-7 depend on the
agent's observation and fall back to the documented default with mask 0.

The eight columns (CONTRACTS 10.2):

0. ``degree_norm`` = ``|L(n)| / (N - 1)``
1. ``neighbor_crit_mean`` = mean ``criticality`` over ``L(n)``
2. ``neighbor_crit_max`` = max ``criticality`` over ``L(n)``
3. ``dependents_crit_norm`` = sum of dependants' criticality / sum of all
4. ``neighbor_alert_mean`` = mean ``alert_count_norm`` over ``L(n) ∩ obs``
5. ``neighbor_alert_max`` = max ``alert_count_norm`` over ``L(n) ∩ obs``
6. ``hop_to_alerted_norm`` = hop distance to nearest observed alerted node / diameter
7. ``blocked_edge_frac`` = fraction of observed incident link edges that are blocked

``L(n)`` is the ``network_link`` neighbourhood and ``obs`` the agent's observed
nodes, both read from the topology and the snapshot masks.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from ..contracts import (
    DERIVED_FEATURE_NAMES_V1,
    EdgeType,
    GraphFeatureRecord,
    GraphSnapshot,
    Topology,
    feature_record_id,
)
from .topology_graph import TopologyGraph

# Snapshot node-feature column indices used as inputs (CONTRACTS 10.1).
_STATUS_UP_COL = 1  # mask 1 here iff the node was observed
_ALERT_COUNT_NORM_COL = 6
# Snapshot edge-feature column indices.
_EDGE_STATE_ACTIVE_COL = 2  # mask 1 here iff the edge was observed
_EDGE_STATE_BLOCKED_COL = 4

_DERIVED_COLS = len(DERIVED_FEATURE_NAMES_V1)


def _f32(value: float) -> float:
    """Round ``value`` to float32 and return it as a Python float (in [0, 1])."""

    return float(np.float32(value))


def compute_features(
    snapshot: GraphSnapshot | Topology,
    topology: Topology | TopologyGraph | GraphSnapshot,
) -> GraphFeatureRecord:
    """Compute ``graph-derived-v1`` features from a snapshot and its topology.

    Accepts either argument order; it takes no ground-truth parameter.
    """

    # Tolerate the DESIGN argument order compute_features(topology, snapshot).
    if isinstance(snapshot, (Topology, TopologyGraph)) and isinstance(
        topology, GraphSnapshot
    ):
        snapshot, topology = topology, snapshot
    assert isinstance(snapshot, GraphSnapshot)

    graph = (
        topology if isinstance(topology, TopologyGraph) else TopologyGraph(topology)
    )

    node_order = snapshot.node_order
    node_count = len(node_order)
    criticality = {node_id: graph.criticality_of(node_id) for node_id in node_order}
    total_criticality = sum(criticality.values())

    observed, alert_norm = _observed_and_alerts(snapshot)
    alerted = {node_id for node_id in observed if alert_norm.get(node_id, 0.0) > 0.0}
    distance_to_alerted = _distance_to_nearest(graph.link_undirected, alerted)
    diameter = graph.diameter
    incident_blocked = _incident_blocked_link_edges(snapshot, graph)

    values: list[list[float]] = []
    mask: list[list[int]] = []
    for node_id in node_order:
        neighbors = graph.link_neighbors(node_id)
        row = [0.0] * _DERIVED_COLS
        row_mask = [0] * _DERIVED_COLS

        # 0: degree_norm (topology only).
        row[0] = _f32(len(neighbors) / (node_count - 1))
        row_mask[0] = 1

        # 1, 2: neighbour criticality mean/max (L(n) never empty; mask always 1).
        neighbor_crit = [criticality[other] for other in neighbors]
        row[1] = _f32(sum(neighbor_crit) / len(neighbor_crit))
        row[2] = _f32(max(neighbor_crit))
        row_mask[1] = 1
        row_mask[2] = 1

        # 3: dependents_crit_norm (topology only; 0.0 means "no dependants").
        dependants = graph.dependents(node_id)
        if total_criticality > 0.0:
            row[3] = _f32(
                sum(criticality[node] for node in dependants) / total_criticality
            )
        row_mask[3] = 1

        # 4, 5: neighbour alert mean/max over observed neighbours.
        observed_neighbors = [other for other in neighbors if other in observed]
        if observed_neighbors:
            alerts = [alert_norm[other] for other in observed_neighbors]
            row[4] = _f32(sum(alerts) / len(alerts))
            row[5] = _f32(max(alerts))
            row_mask[4] = 1
            row_mask[5] = 1

        # 6: hop_to_alerted_norm (default 1.0, mask 0, when nothing is alerted).
        if alerted:
            row[6] = _f32(distance_to_alerted[node_id] / diameter)
            row_mask[6] = 1
        else:
            row[6] = 1.0

        # 7: blocked_edge_frac over observed incident link edges.
        incident = incident_blocked[node_id]
        if incident:
            row[7] = _f32(sum(1 for blocked in incident if blocked) / len(incident))
            row_mask[7] = 1

        values.append(row)
        mask.append(row_mask)

    return GraphFeatureRecord(
        feature_record_id=feature_record_id(snapshot.step, snapshot.agent_id),
        episode_id=snapshot.episode_id,
        step=snapshot.step,
        agent_id=snapshot.agent_id,
        topology_id=snapshot.topology_id,
        snapshot_id=snapshot.snapshot_id,
        node_order=list(node_order),
        feature_names=list(DERIVED_FEATURE_NAMES_V1),
        values=values,
        mask=mask,
    )


def _distance_to_nearest(
    link_graph: nx.Graph, sources: set[str]
) -> dict[str, int]:
    """Shortest hop distance from every node to the nearest node in ``sources``.

    Computed as the per-source minimum of unweighted single-source distances so
    it relies only on the always-present ``single_source_shortest_path_length``.
    The link graph is connected, so every node is reachable from any source.
    """

    nearest: dict[str, int] = {}
    for source in sources:
        for node, distance in nx.single_source_shortest_path_length(
            link_graph, source
        ).items():
            current = nearest.get(node)
            if current is None or distance < current:
                nearest[node] = distance
    return nearest


def _observed_and_alerts(
    snapshot: GraphSnapshot,
) -> tuple[set[str], dict[str, float]]:
    """Return the observed node set and their ``alert_count_norm`` values.

    A node is observed iff its status one-hot carries mask 1 (CONTRACTS 10.1:
    an unobserved node has every dynamic column imputed with mask 0).
    """

    observed: set[str] = set()
    alert_norm: dict[str, float] = {}
    for index, node_id in enumerate(snapshot.node_order):
        if snapshot.node_mask[index][_STATUS_UP_COL] == 1:
            observed.add(node_id)
            alert_norm[node_id] = snapshot.node_features[index][_ALERT_COUNT_NORM_COL]
    return observed, alert_norm


def _incident_blocked_link_edges(
    snapshot: GraphSnapshot, graph: TopologyGraph
) -> dict[str, list[bool]]:
    """Map each node to the blocked flags of its observed incident link edges."""

    edge_type_by_id = {edge.edge_id: edge.edge_type for edge in graph.topology.edges}
    endpoints_by_id = {
        edge.edge_id: (edge.source, edge.target) for edge in graph.topology.edges
    }
    incident: dict[str, list[bool]] = {node_id: [] for node_id in snapshot.node_order}
    for index, edge_id in enumerate(snapshot.edge_order):
        if edge_type_by_id[edge_id] is not EdgeType.network_link:
            continue
        if snapshot.edge_mask[index][_EDGE_STATE_ACTIVE_COL] != 1:
            continue  # edge state unobserved -> edge not observed
        blocked = snapshot.edge_features[index][_EDGE_STATE_BLOCKED_COL] == 1.0
        source, target = endpoints_by_id[edge_id]
        incident[source].append(blocked)
        incident[target].append(blocked)
    return incident
