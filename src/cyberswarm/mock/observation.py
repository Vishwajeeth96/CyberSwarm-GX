"""MOCK simulator - agent-visible observation building (lead-owned, task 5.2).

**This is MOCK, not a real simulator.** This module turns the hidden
:class:`~cyberswarm.mock.effects.NodeState` into the agent-visible records a
decision agent is allowed to see, and nothing more. By construction it emits no
evaluator-only field: it builds :class:`~cyberswarm.contracts.AgentObservation`
and :class:`~cyberswarm.contracts.AvailableActions` records whose schemas forbid
a ground-truth field, and it decides each event's ``visible_to`` set from the
AD-19 table.

What it implements:

* :class:`TopologyIndex` - the derived structure of a topology (zones, agents,
  link adjacency, service dependencies, own/boundary partition, per-node event
  visibility). Built once per episode with NetworkX.
* :func:`build_observation` - own-zone nodes carry the full per-node signals;
  one-hop boundary nodes carry only status and alert counts, with the private
  fields (``evidence_level``, ``service_health``, ``steps_since_investigated``)
  withheld as ``null``.
* :func:`build_available_actions` - the state-dependent action menu.
* the weakened ``service_health`` channel (AD-18: compromise factor ``0.9`` via
  ``service_availability``, truncated-normal noise ``0.08``) and the edge
  ``traffic_level`` channel (compromise offset ``0.05``, noise ``0.08``), both
  made ``null`` at the scenario ``missing_value_rate``.

It imports only :mod:`cyberswarm.contracts`, the sibling :mod:`.effects`,
NetworkX and the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx

from ..contracts import (
    ACTION_ORDER,
    ActionKind,
    AgentObservation,
    AvailableAction,
    AvailableActions,
    EdgeObservation,
    EdgeState,
    EdgeType,
    EventRecord,
    EventType,
    NodeObservation,
    NodeScope,
    NodeStatus,
    RateLimitLevel,
    Severity,
    Topology,
    agent_id_for_zone,
    available_actions_id,
    default_parameters,
    observation_id,
)
from ..contracts.vocabulary import ACTION_DEFS
from .effects import NodeState, action_precondition

__all__ = [
    "SERVICE_HEALTH_NOISE",
    "EDGE_TRAFFIC_NOISE",
    "EDGE_COMPROMISE_OFFSET",
    "EVIDENCE_TTL_STEPS",
    "SEVERITY_RANK",
    "TopologyIndex",
    "edge_state_of",
    "service_health_value",
    "traffic_level_value",
    "build_observation",
    "build_available_actions",
]

#: AD-18 weakened-channel constants.
SERVICE_HEALTH_NOISE: float = 0.08
EDGE_TRAFFIC_NOISE: float = 0.08
EDGE_COMPROMISE_OFFSET: float = 0.05

#: An investigation result stays visible for this many steps, then reverts to
#: null (DESIGN 5.3). Measured from the step the investigation was executed.
EVIDENCE_TTL_STEPS: int = 5

#: Map an alert severity to the 0..3 integer used by ``max_alert_severity``.
SEVERITY_RANK: Mapping[Severity, int] = {
    Severity.low: 1,
    Severity.medium: 2,
    Severity.high: 3,
}

#: Edge-state throughput factor used by ``traffic_level``.
_STATE_FACTOR: Mapping[EdgeState, float] = {
    EdgeState.active: 1.0,
    EdgeState.limited: 0.6,
    EdgeState.blocked: 0.0,
}


def _clamp_unit(value: float) -> float:
    """Clamp a float to the ``[0, 1]`` ``UnitFloat`` range."""

    return min(1.0, max(0.0, value))


# --------------------------------------------------------------------------- #
# Derived topology structure
# --------------------------------------------------------------------------- #
class TopologyIndex:
    """Derived, read-only structure of a :class:`Topology` for the MOCK.

    Built once per episode. It exposes the orderings the common-random-number
    tables are indexed by, the zone/authority partition, the link adjacency used
    for propagation and event visibility, and the service-dependency map used
    for availability.
    """

    def __init__(self, topology: Topology) -> None:
        self.topology = topology
        self.node_order: tuple[str, ...] = topology.node_ids
        self.edge_order: tuple[str, ...] = topology.edge_ids
        self.node_index = {node_id: i for i, node_id in enumerate(self.node_order)}
        self.edge_index = {edge_id: i for i, edge_id in enumerate(self.edge_order)}
        self.agents: tuple[str, ...] = topology.agent_ids
        self.edge_by_id = {edge.edge_id: edge for edge in topology.edges}

        self.zone_of = {node.node_id: node.zone_id for node in topology.nodes}
        self.agent_of_node = {
            node_id: agent_id_for_zone(zone) for node_id, zone in self.zone_of.items()
        }
        #: The zone each agent owns (``agent-dmz`` -> ``zone-dmz``).
        self.zone_of_agent = {
            agent_id_for_zone(zone): zone for zone in topology.zone_ids
        }

        link = nx.Graph()
        link.add_nodes_from(self.node_order)
        dependency = nx.DiGraph()
        dependency.add_nodes_from(self.node_order)
        link_out: dict[str, list[tuple[str, str]]] = {n: [] for n in self.node_order}
        for edge in topology.edges:
            if edge.edge_type is EdgeType.network_link:
                link.add_edge(edge.source, edge.target)
                link_out[edge.source].append((edge.edge_id, edge.target))
            elif edge.edge_type is EdgeType.service_dependency:
                dependency.add_edge(edge.source, edge.target)
        for node_id in link_out:
            link_out[node_id].sort()
        self._link = link
        self._link_out = link_out
        #: ``dependencies[n]`` are the nodes ``n`` depends on (service edges).
        self.dependencies = {
            node_id: tuple(sorted(dependency.successors(node_id)))
            for node_id in self.node_order
        }

        self.own_nodes = {
            agent: tuple(n for n in self.node_order if self.agent_of_node[n] == agent)
            for agent in self.agents
        }
        self.boundary_nodes: dict[str, tuple[str, ...]] = {}
        self.agent_edges: dict[str, tuple[str, ...]] = {}
        for agent in self.agents:
            own = set(self.own_nodes[agent])
            boundary = {
                neighbour
                for node_id in own
                for neighbour in link.neighbors(node_id)
                if neighbour not in own
            }
            self.boundary_nodes[agent] = tuple(sorted(boundary))
            self.agent_edges[agent] = tuple(
                edge.edge_id
                for edge in topology.edges
                if edge.source in own or edge.target in own
            )

        self.node_visible_agents: dict[str, tuple[str, ...]] = {}
        for node_id in self.node_order:
            agents = {self.agent_of_node[node_id]}
            agents.update(self.agent_of_node[m] for m in link.neighbors(node_id))
            self.node_visible_agents[node_id] = tuple(sorted(agents))

    def link_out(self, node_id: str) -> list[tuple[str, str]]:
        """Outgoing ``network_link`` edges ``(edge_id, target)``, ascending."""

        return self._link_out[node_id]

    def event_visible_to(
        self,
        event_type: EventType,
        *,
        node_id: str | None = None,
        acting_agent: str | None = None,
    ) -> list[str]:
        """Return the ascending ``visible_to`` set for an event (AD-19)."""

        if event_type in (EventType.episode_started, EventType.episode_ended):
            return list(self.agents)
        if event_type in (
            EventType.alert_raised,
            EventType.node_status_changed,
            EventType.restore_completed,
        ):
            assert node_id is not None
            return list(self.node_visible_agents[node_id])
        if event_type in (EventType.action_applied, EventType.investigation_result):
            assert acting_agent is not None
            return [acting_agent]
        # recovery_verified / recovery_failed (coordinator events): zone agent.
        assert node_id is not None
        return [self.agent_of_node[node_id]]


# --------------------------------------------------------------------------- #
# Observation measurements
# --------------------------------------------------------------------------- #
def edge_state_of(source: NodeState, target: NodeState) -> EdgeState:
    """Return the visible edge state from its endpoints' statuses (DESIGN 5.3)."""

    blocking = (NodeStatus.isolated, NodeStatus.restoring)
    if (
        source.status in blocking
        or target.status in blocking
        or source.status is NodeStatus.quarantined
    ):
        return EdgeState.blocked
    if source.status is NodeStatus.rate_limited or target.status is NodeStatus.rate_limited:
        return EdgeState.limited
    return EdgeState.active


def service_health_value(
    service_avail: float, noise: float, *, monitored: bool
) -> float:
    """Weakened ``service_health`` proxy (AD-18).

    ``service_avail`` already carries the ``0.9`` compromise factor, so the
    channel reflects compromise only weakly. The truncated-normal ``noise`` is
    scaled by ``0.08`` and halved while the node is monitored.
    """

    scale = SERVICE_HEALTH_NOISE * (0.5 if monitored else 1.0)
    return _clamp_unit(service_avail + scale * noise)


def traffic_level_value(
    state: EdgeState, *, source_compromised: bool, noise: float
) -> float:
    """Weakened edge ``traffic_level`` proxy (AD-18)."""

    base = 0.5 * _STATE_FACTOR[state]
    if source_compromised and state is EdgeState.active:
        base += EDGE_COMPROMISE_OFFSET
    return _clamp_unit(base + EDGE_TRAFFIC_NOISE * noise)


def _visible_alerts_for_node(
    events: Sequence[EventRecord], node_id: str
) -> tuple[int, int]:
    """Return ``(alert_count_recent, max_alert_severity)`` for one node."""

    count = 0
    max_rank = 0
    for event in events:
        if event.event_type is EventType.alert_raised and event.node_id == node_id:
            count += 1
            if event.severity is not None:
                max_rank = max(max_rank, SEVERITY_RANK[event.severity])
    return min(count, 50), max_rank


def _node_observation(
    node_id: str,
    state: NodeState,
    *,
    scope: NodeScope,
    step: int,
    visible_alerts: tuple[int, int],
    service_avail: float,
    health_noise: float,
    missing: bool,
) -> NodeObservation:
    """Build one :class:`NodeObservation`, withholding private fields on a boundary."""

    alert_count, max_severity = visible_alerts
    if scope is NodeScope.boundary:
        return NodeObservation(
            node_id=node_id,
            scope=scope,
            visible_status=state.status,
            alert_count_recent=alert_count,
            max_alert_severity=max_severity,
            evidence_level=None,
            service_health=None,
            steps_since_investigated=None,
        )

    evidence_visible = (
        state.evidence_step is not None
        and 0 <= step - state.evidence_step <= EVIDENCE_TTL_STEPS
    )
    evidence_level = state.evidence_level if evidence_visible else None
    steps_since = (
        step - state.last_investigated if state.last_investigated is not None else None
    )
    service_health = (
        None
        if missing
        else service_health_value(service_avail, health_noise, monitored=state.monitored)
    )
    return NodeObservation(
        node_id=node_id,
        scope=scope,
        visible_status=state.status,
        alert_count_recent=alert_count,
        max_alert_severity=max_severity,
        evidence_level=evidence_level,
        service_health=service_health,
        steps_since_investigated=steps_since,
    )


def build_observation(
    *,
    agent_id: str,
    step: int,
    max_steps: int,
    window: int,
    episode_id: str,
    zone_id: str,
    index: TopologyIndex,
    node_states: Mapping[str, NodeState],
    service_avail: Mapping[str, float],
    events: Sequence[EventRecord],
    health_noise: Mapping[str, float],
    node_missing: Mapping[str, bool],
    edge_noise: Mapping[str, float],
    edge_missing: Mapping[str, bool],
) -> AgentObservation:
    """Assemble the agent-visible :class:`AgentObservation` for one agent.

    ``events`` must already be the step's P1 view (events emitted up to and
    including P0 of this step); it is filtered here to the window and to this
    agent's ``visible_to`` set. ``last_step_feedback`` is left empty: the
    simulator does not see coordinator decisions, so the engine attaches the
    feedback of the previous step.
    """

    low = step - window
    visible_events = sorted(
        (
            event
            for event in events
            if agent_id in event.visible_to and low < event.step <= step
        ),
        key=lambda event: (event.step, event.event_id),
    )

    own = set(index.own_nodes[agent_id])
    nodes: list[NodeObservation] = []
    for node_id in index.own_nodes[agent_id]:
        nodes.append(
            _node_observation(
                node_id,
                node_states[node_id],
                scope=NodeScope.own,
                step=step,
                visible_alerts=_visible_alerts_for_node(visible_events, node_id),
                service_avail=service_avail[node_id],
                health_noise=health_noise[node_id],
                missing=node_missing[node_id],
            )
        )
    for node_id in index.boundary_nodes[agent_id]:
        nodes.append(
            _node_observation(
                node_id,
                node_states[node_id],
                scope=NodeScope.boundary,
                step=step,
                visible_alerts=_visible_alerts_for_node(visible_events, node_id),
                service_avail=service_avail[node_id],
                health_noise=health_noise[node_id],
                missing=node_missing[node_id],
            )
        )
    nodes.sort(key=lambda node: node.node_id)

    edges: list[EdgeObservation] = []
    for edge_id in index.agent_edges[agent_id]:
        edge = index.edge_by_id[edge_id]
        source = node_states[edge.source]
        target = node_states[edge.target]
        state = edge_state_of(source, target)
        traffic = (
            None
            if edge_missing[edge_id]
            else traffic_level_value(
                state,
                source_compromised=source.compromised,
                noise=edge_noise[edge_id],
            )
        )
        edges.append(
            EdgeObservation(edge_id=edge_id, edge_state=state, traffic_level=traffic)
        )
    edges.sort(key=lambda edge: edge.edge_id)

    return AgentObservation(
        observation_id=observation_id(step, agent_id),
        episode_id=episode_id,
        step=step,
        agent_id=agent_id,
        zone_id=zone_id,
        max_steps=max_steps,
        observation_window_steps=window,
        nodes=nodes,
        edges=edges,
        recent_events=list(visible_events),
        last_step_feedback=[],
    )


# --------------------------------------------------------------------------- #
# Available actions
# --------------------------------------------------------------------------- #
def _available_action(kind: ActionKind, target_node_id: str | None) -> AvailableAction:
    """Build one :class:`AvailableAction` with the vocabulary metadata."""

    spec = ACTION_DEFS[kind]
    return AvailableAction(
        action_kind=kind,
        target_node_id=target_node_id,
        parameters=default_parameters(kind),
        impact_class=spec.impact_class,
        high_impact=spec.high_impact,
        cost_units=spec.cost_units,
    )


def build_available_actions(
    *,
    agent_id: str,
    step: int,
    episode_id: str,
    index: TopologyIndex,
    node_states: Mapping[str, NodeState],
    final: bool = False,
) -> AvailableActions:
    """Build the state-dependent action menu for one agent (DESIGN 5.3 P4).

    ``no_op`` is always first. The remaining actions follow the vocabulary order
    and, within each kind, ascending ``node_id``; only own-zone nodes appear as
    targets, and only where the action's precondition holds. The final
    observation after the episode ends offers ``no_op`` only.
    """

    actions: list[AvailableAction] = [_available_action(ActionKind.no_op, None)]
    if not final:
        for kind in ACTION_ORDER:
            if kind is ActionKind.no_op:
                continue
            for node_id in index.own_nodes[agent_id]:
                if action_precondition(kind, node_states[node_id].status):
                    actions.append(_available_action(kind, node_id))

    return AvailableActions(
        available_actions_id=available_actions_id(step, agent_id),
        episode_id=episode_id,
        step=step,
        agent_id=agent_id,
        actions=actions,
    )
