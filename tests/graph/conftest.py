"""Shared setup for the graph-pipeline tests (stage S3, DC-07a..DC-07j).

Exposes the two committed topologies (the reference ``topo-mock-small-v1`` and
the unseen ``topo-mock-small-v2``), the committed step-5 DMZ observation and the
snapshot/feature record derived from it, small hand-built topologies with known
structure for per-feature hand computation, and builder callables for
observations. The builders mirror the agent-visible contract: only
``AgentObservation``/``NodeObservation``/``EdgeObservation`` fields, never any
ground-truth input (the converter and feature functions take no truth argument).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from cyberswarm.contracts import (
    AgentObservation,
    Edge,
    EdgeObservation,
    EdgeState,
    EdgeType,
    GraphFeatureRecord,
    GraphSnapshot,
    Node,
    NodeObservation,
    NodeScope,
    NodeStatus,
    NodeType,
    Topology,
    load_topology,
    observation_id,
)
from cyberswarm.graph import compute_features, convert

#: Repository root: ``tests/graph/conftest.py`` -> ``<repo>``.
REPO_ROOT = Path(__file__).resolve().parents[2]
_TOPO_DIR = REPO_ROOT / "configs" / "topologies"
VALID_FIXTURES = REPO_ROOT / "fixtures" / "records" / "valid"


# --------------------------------------------------------------------------- #
# Committed topologies and the step-5 DMZ records
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def topology_v1() -> Topology:
    """The reference 8-node, 2-zone topology (``topo-mock-small-v1``)."""

    return load_topology(_TOPO_DIR / "topo-mock-small-v1.json")


@pytest.fixture(scope="session")
def topology_v2() -> Topology:
    """The unseen variant (same node ids/zones, different edges/criticalities)."""

    return load_topology(_TOPO_DIR / "topo-mock-small-v2.json")


@pytest.fixture(scope="session")
def obs_step5() -> AgentObservation:
    """The committed step-5 DMZ agent observation (four visible nodes)."""

    return AgentObservation.model_validate_json(
        (VALID_FIXTURES / "AgentObservation.dmz-step5.json").read_text()
    )


@pytest.fixture(scope="session")
def snapshot_step5(obs_step5: AgentObservation, topology_v1: Topology) -> GraphSnapshot:
    """``convert(obs_step5, topo_v1)`` - the full 8-node base snapshot."""

    return convert(obs_step5, topology_v1)


@pytest.fixture(scope="session")
def features_step5(
    snapshot_step5: GraphSnapshot, topology_v1: Topology
) -> GraphFeatureRecord:
    """``compute_features`` over :func:`snapshot_step5`."""

    return compute_features(snapshot_step5, topology_v1)


# --------------------------------------------------------------------------- #
# Agent-observation builders (no ground-truth input anywhere)
# --------------------------------------------------------------------------- #
@pytest.fixture
def make_node() -> Callable[..., NodeObservation]:
    """Return a builder for a single :class:`NodeObservation`."""

    def _node(
        node_id: str,
        *,
        scope: NodeScope = NodeScope.own,
        status: NodeStatus = NodeStatus.up,
        alert_count: int = 0,
        severity: int = 0,
        evidence_level: int | None = None,
        service_health: float | None = None,
        steps_since: int | None = None,
    ) -> NodeObservation:
        return NodeObservation(
            node_id=node_id,
            scope=scope,
            visible_status=status,
            alert_count_recent=alert_count,
            max_alert_severity=severity,
            evidence_level=evidence_level,
            service_health=service_health,
            steps_since_investigated=steps_since,
        )

    return _node


@pytest.fixture
def make_obs() -> Callable[..., AgentObservation]:
    """Return a builder for an :class:`AgentObservation` for ``agent-dmz``."""

    def _obs(
        nodes: list[NodeObservation],
        *,
        step: int = 5,
        agent_id: str = "agent-dmz",
        zone_id: str = "zone-dmz",
        edges: list[EdgeObservation] | None = None,
        episode_id: str = "ep-rule-conv-v1-mock-lateral-eval-01-s2001",
    ) -> AgentObservation:
        return AgentObservation(
            observation_id=observation_id(step, agent_id),
            episode_id=episode_id,
            step=step,
            agent_id=agent_id,
            zone_id=zone_id,
            max_steps=40,
            observation_window_steps=5,
            nodes=nodes,
            edges=edges or [],
        )

    return _obs


# --------------------------------------------------------------------------- #
# Small hand-built topologies with fully known structure
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def tiny_topology() -> Topology:
    """A 3-node line ``a - b - c`` (network_link) with ``a`` depending on ``c``.

    Criticalities ``a=0.2``, ``b=0.6``, ``c=1.0`` (total 1.8). Edge ids follow
    the canonical rank of ``(source, target, edge_type)``:
    ``edge-001`` a->b link, ``edge-002`` a->c service_dependency,
    ``edge-003`` b->c link.
    """

    return Topology(
        topology_id="topo-tiny-v1",
        nodes=[
            Node(node_id="node-a", node_type=NodeType.web, zone_id="zone-dmz", criticality=0.2),
            Node(node_id="node-b", node_type=NodeType.app, zone_id="zone-dmz", criticality=0.6),
            Node(node_id="node-c", node_type=NodeType.database, zone_id="zone-dmz", criticality=1.0),
        ],
        edges=[
            Edge(edge_id="edge-001", source="node-a", target="node-b", edge_type=EdgeType.network_link),
            Edge(edge_id="edge-002", source="node-a", target="node-c", edge_type=EdgeType.service_dependency),
            Edge(edge_id="edge-003", source="node-b", target="node-c", edge_type=EdgeType.network_link),
        ],
    )


@pytest.fixture(scope="session")
def tiny_observation() -> AgentObservation:
    """Observe all three tiny-topology nodes; only ``node-a`` is alerted.

    ``node-a`` has five alerts at severity 3 (so ``alert_count_norm = 1.0``),
    ``edge-001`` (a-b) is blocked and ``edge-003`` (b-c) is active.
    """

    return AgentObservation(
        observation_id=observation_id(5, "agent-dmz"),
        episode_id="ep-rule-conv-v1-mock-lateral-tiny-s1",
        step=5,
        agent_id="agent-dmz",
        zone_id="zone-dmz",
        max_steps=40,
        observation_window_steps=5,
        nodes=[
            NodeObservation(node_id="node-a", scope=NodeScope.own, visible_status=NodeStatus.up, alert_count_recent=5, max_alert_severity=3),
            NodeObservation(node_id="node-b", scope=NodeScope.own, visible_status=NodeStatus.up, alert_count_recent=0, max_alert_severity=0),
            NodeObservation(node_id="node-c", scope=NodeScope.own, visible_status=NodeStatus.up, alert_count_recent=0, max_alert_severity=0),
        ],
        edges=[
            EdgeObservation(edge_id="edge-001", edge_state=EdgeState.blocked),
            EdgeObservation(edge_id="edge-003", edge_state=EdgeState.active),
        ],
    )


@pytest.fixture(scope="session")
def pair_topology() -> Topology:
    """A 2-node topology ``a - b`` (one network_link, no dependency edges).

    Criticalities ``a=0.4``, ``b=0.9``; used to exercise the mask-0 derived
    defaults when a neighbour is unobserved.
    """

    return Topology(
        topology_id="topo-pair-v1",
        nodes=[
            Node(node_id="node-a", node_type=NodeType.web, zone_id="zone-dmz", criticality=0.4),
            Node(node_id="node-b", node_type=NodeType.app, zone_id="zone-dmz", criticality=0.9),
        ],
        edges=[
            Edge(edge_id="edge-001", source="node-a", target="node-b", edge_type=EdgeType.network_link),
        ],
    )
