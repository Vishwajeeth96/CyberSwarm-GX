"""Agent observation to :class:`GraphSnapshot` (CONTRACTS 10.1, DESIGN 4.5).

:func:`convert` turns one agent-visible :class:`~cyberswarm.contracts.AgentObservation`
into a deterministic :class:`~cyberswarm.contracts.GraphSnapshot` built with the
``graph-base-v1`` feature registry and the ``impute-v1`` imputation policy.

Leakage boundary (Requirement 5.3, CONTRACTS section 10 preamble): the converter
reads only the topology and the agent-visible observation (whose events already
satisfy ``step <= t``). It takes **no** ground-truth parameter and never touches
``cyberswarm.contracts.evaluator_only``. An optional ``events`` argument lets a
caller pass an explicit list of agent-visible events to consider for
``max_event_step``; any event with ``step`` beyond the observation's step is
excluded rather than trusted.

Determinism: node and edge order equal the topology order; every feature value
is cast through ``float32`` so the stored matrices are exactly what
:func:`~cyberswarm.graph.tensors.to_tensors` yields and a repeated conversion is
bit-identical.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..contracts import (
    EDGE_FEATURE_NAMES_BASE_V1,
    NODE_FEATURE_NAMES_BASE_V1,
    AgentObservation,
    EdgeState,
    EdgeType,
    ErrorCode,
    EventRecord,
    GraphSnapshot,
    NodeStatus,
    Topology,
    snapshot_id,
)
from .errors import GraphConversionError
from .topology_graph import TopologyGraph

#: Column order of the five node-status one-hots (CONTRACTS 10.1, columns 1-5).
_STATUS_ORDER: tuple[NodeStatus, ...] = (
    NodeStatus.up,
    NodeStatus.rate_limited,
    NodeStatus.quarantined,
    NodeStatus.isolated,
    NodeStatus.restoring,
)

#: Column order of the three edge-state one-hots (CONTRACTS 10.1, columns 2-4).
_EDGE_STATE_ORDER: tuple[EdgeState, ...] = (
    EdgeState.active,
    EdgeState.limited,
    EdgeState.blocked,
)

#: Default converter id written to the snapshot's ``converter`` field.
CONVERTER_ID = "graph.converter-v1"

_NODE_COLS = len(NODE_FEATURE_NAMES_BASE_V1)
_EDGE_COLS = len(EDGE_FEATURE_NAMES_BASE_V1)


def _f32(value: float) -> float:
    """Round ``value`` to the nearest ``float32`` and return it as a Python float.

    Every stored feature is float32-representable (CONTRACTS 10.1: ``dtype`` is
    ``float32``), so a snapshot survives the numpy/torch round trip unchanged.
    """

    return float(np.float32(value))


class GraphConverter:
    """Reusable converter bound to one topology (DESIGN section 4.5)."""

    def __init__(
        self, topology: Topology | TopologyGraph, *, converter_id: str = CONVERTER_ID
    ) -> None:
        self.graph = (
            topology if isinstance(topology, TopologyGraph) else TopologyGraph(topology)
        )
        self.topology = self.graph.topology
        self.converter_id = converter_id

    def convert(
        self,
        observation: AgentObservation,
        events: Iterable[EventRecord] | None = None,
    ) -> GraphSnapshot:
        """Convert one agent observation into a :class:`GraphSnapshot`."""

        return _convert(self.graph, observation, events, self.converter_id)


def convert(
    observation: AgentObservation,
    topology: Topology | TopologyGraph,
    events: Iterable[EventRecord] | None = None,
) -> GraphSnapshot:
    """Convert ``observation`` against ``topology`` into a :class:`GraphSnapshot`.

    This free function takes no ground-truth argument. ``events`` is an optional
    list of agent-visible events; future events (``step`` beyond the
    observation) are excluded.
    """

    # Tolerate the DESIGN method order convert(topology, observation) as well.
    if isinstance(observation, (Topology, TopologyGraph)) and isinstance(
        topology, AgentObservation
    ):
        observation, topology = topology, observation
    graph = (
        topology if isinstance(topology, TopologyGraph) else TopologyGraph(topology)
    )
    return _convert(graph, observation, events, CONVERTER_ID)


def _convert(
    graph: TopologyGraph,
    observation: AgentObservation,
    events: Iterable[EventRecord] | None,
    converter_id: str,
) -> GraphSnapshot:
    topology = graph.topology
    known_nodes = set(graph.node_order)
    known_edges = set(graph.edge_order)

    node_obs = {node.node_id: node for node in observation.nodes}
    edge_obs = {edge.edge_id: edge for edge in observation.edges}
    for node_id in node_obs:
        if node_id not in known_nodes:
            raise GraphConversionError(
                ErrorCode.UNKNOWN_ID,
                f"observation node {node_id!r} is not in topology {topology.topology_id!r}",
            )
    for edge_id in edge_obs:
        if edge_id not in known_edges:
            raise GraphConversionError(
                ErrorCode.UNKNOWN_ID,
                f"observation edge {edge_id!r} is not in topology {topology.topology_id!r}",
            )

    node_features, node_mask = _build_node_matrix(graph, node_obs)
    edge_features, edge_mask, edge_index = _build_edge_matrix(graph, edge_obs)

    return GraphSnapshot(
        snapshot_id=snapshot_id(observation.step, observation.agent_id),
        episode_id=observation.episode_id,
        step=observation.step,
        agent_id=observation.agent_id,
        topology_id=topology.topology_id,
        node_order=list(graph.node_order),
        edge_order=list(graph.edge_order),
        edge_index=edge_index,
        node_feature_names=list(NODE_FEATURE_NAMES_BASE_V1),
        node_features=node_features,
        node_mask=node_mask,
        edge_feature_names=list(EDGE_FEATURE_NAMES_BASE_V1),
        edge_features=edge_features,
        edge_mask=edge_mask,
        source_observation_id=observation.observation_id,
        converter=converter_id,
        max_event_step=_max_event_step(observation, events),
    )


def _build_node_matrix(
    graph: TopologyGraph, node_obs: dict
) -> tuple[list[list[float]], list[list[int]]]:
    node_features: list[list[float]] = []
    node_mask: list[list[int]] = []
    for node_id in graph.node_order:
        row = [0.0] * _NODE_COLS
        mask = [0] * _NODE_COLS

        # Column 0 is public topology information: always observed.
        row[0] = _f32(graph.criticality_of(node_id))
        mask[0] = 1

        observed = node_obs.get(node_id)
        if observed is not None:
            # Columns 1-5: status one-hot. Columns 1-7 always observed.
            row[1 + _STATUS_ORDER.index(observed.visible_status)] = 1.0
            for column in range(1, 8):
                mask[column] = 1
            row[6] = _f32(min(observed.alert_count_recent, 5) / 5)
            row[7] = _f32(observed.max_alert_severity / 3)
            # Columns 8-10 observed only when the private measurement is present.
            if observed.evidence_level is not None:
                row[8] = _f32(observed.evidence_level / 3)
                mask[8] = 1
            if observed.service_health is not None:
                row[9] = _f32(observed.service_health)
                mask[9] = 1
            if observed.steps_since_investigated is not None:
                row[10] = _f32(min(observed.steps_since_investigated, 10) / 10)
                mask[10] = 1

        node_features.append(row)
        node_mask.append(mask)
    return node_features, node_mask


def _build_edge_matrix(
    graph: TopologyGraph, edge_obs: dict
) -> tuple[list[list[float]], list[list[int]], list[list[int]]]:
    edge_type_by_id = {edge.edge_id: edge.edge_type for edge in graph.topology.edges}
    endpoints_by_id = {
        edge.edge_id: (edge.source, edge.target) for edge in graph.topology.edges
    }

    edge_features: list[list[float]] = []
    edge_mask: list[list[int]] = []
    sources: list[int] = []
    targets: list[int] = []

    for edge_id in graph.edge_order:
        row = [0.0] * _EDGE_COLS
        mask = [0] * _EDGE_COLS

        # Columns 0-1: edge-type one-hot (static, always observed).
        if edge_type_by_id[edge_id] is EdgeType.network_link:
            row[0] = 1.0
        else:
            row[1] = 1.0
        mask[0] = 1
        mask[1] = 1

        observed = edge_obs.get(edge_id)
        if observed is not None:
            row[2 + _EDGE_STATE_ORDER.index(observed.edge_state)] = 1.0
            for column in range(2, 5):
                mask[column] = 1
            if observed.traffic_level is not None:
                row[5] = _f32(observed.traffic_level)
                mask[5] = 1

        edge_features.append(row)
        edge_mask.append(mask)
        source, target = endpoints_by_id[edge_id]
        sources.append(graph.index_of(source))
        targets.append(graph.index_of(target))

    return edge_features, edge_mask, [sources, targets]


def _max_event_step(
    observation: AgentObservation, events: Iterable[EventRecord] | None
) -> int | None:
    visible = list(observation.recent_events)
    if events is not None:
        visible.extend(events)
    steps = [event.step for event in visible if event.step <= observation.step]
    return max(steps) if steps else None
