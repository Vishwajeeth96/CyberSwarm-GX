"""Unit tests for :mod:`cyberswarm.roles.evidence` (stage S3, DC-10).

Covers the per-own-node evidence view and the ``insufficient`` (investigation
advisable) flag, own nodes with null evidence, exclusion of boundary nodes
whose private evidence is withheld, the shared ``e_min`` threshold through
:class:`EvidenceConfig`, node ordering, determinism, and input validation.
"""

from __future__ import annotations

import pathlib

import pytest

from cyberswarm.contracts import (
    AgentObservation,
    NodeObservation,
    NodeScope,
    NodeStatus,
    observation_id,
)
from cyberswarm.interfaces import EvidenceView, NodeEvidence
from cyberswarm.roles import DEFAULT_E_MIN, EvidenceConfig, evidence_view

_FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "records" / "valid"
_EPISODE_ID = "ep-rule-conv-v1-mock-lateral-eval-01-s2001"


def _node(
    node_id: str,
    *,
    scope: NodeScope = NodeScope.own,
    alert_count: int = 0,
    severity: int = 0,
    evidence_level: int | None = None,
    steps_since: int | None = None,
    service_health: float | None = None,
    status: NodeStatus = NodeStatus.up,
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


def _obs(nodes: list[NodeObservation], *, step: int = 5, agent_id: str = "agent-dmz") -> AgentObservation:
    return AgentObservation(
        observation_id=observation_id(step, agent_id),
        episode_id=_EPISODE_ID,
        step=step,
        agent_id=agent_id,
        zone_id="zone-dmz",
        max_steps=40,
        observation_window_steps=5,
        nodes=nodes,
    )


# --------------------------------------------------------------------------- #
# Per-node evidence values
# --------------------------------------------------------------------------- #
def test_evidence_view_reports_hand_values_for_own_nodes() -> None:
    view = evidence_view(
        _obs(
            [
                _node("node-gw-01", alert_count=1, severity=1, evidence_level=0, steps_since=3),
                _node("node-web-01", alert_count=0, severity=0),
                _node("node-web-02", alert_count=1, severity=2),
            ]
        )
    )
    assert isinstance(view, EvidenceView)
    assert view.nodes["node-gw-01"] == NodeEvidence(
        evidence_level=0, steps_since_investigated=3, alerted=True, insufficient=True
    )
    assert view.nodes["node-web-01"] == NodeEvidence(
        evidence_level=None, steps_since_investigated=None, alerted=False, insufficient=False
    )
    assert view.nodes["node-web-02"] == NodeEvidence(
        evidence_level=None, steps_since_investigated=None, alerted=True, insufficient=True
    )


# --------------------------------------------------------------------------- #
# insufficient / investigation advisable
# --------------------------------------------------------------------------- #
def test_advises_investigation_when_alerted_and_evidence_below_e_min() -> None:
    view = evidence_view(
        _obs(
            [
                _node("node-a", alert_count=1, evidence_level=None),  # null evidence
                _node("node-b", alert_count=1, evidence_level=1),  # 1 < e_min(2)
                _node("node-c", alert_count=1, evidence_level=2),  # 2 == e_min -> enough
                _node("node-d", alert_count=1, evidence_level=3),  # above e_min
            ]
        )
    )
    advisable = {nid: ev.insufficient for nid, ev in view.nodes.items()}
    assert advisable == {
        "node-a": True,  # null evidence, alerted -> investigate
        "node-b": True,  # weak evidence below e_min -> investigate
        "node-c": False,  # evidence at e_min is sufficient
        "node-d": False,  # strong evidence
    }


def test_no_advice_when_not_alerted_even_with_null_evidence() -> None:
    # A quiet own node with no visible evidence is not a reason to investigate.
    view = evidence_view(_obs([_node("node-quiet", alert_count=0, evidence_level=None)]))
    assert view.nodes["node-quiet"].alerted is False
    assert view.nodes["node-quiet"].insufficient is False


def test_own_node_with_null_evidence_is_handled() -> None:
    view = evidence_view(_obs([_node("node-web-02", alert_count=2, severity=2, evidence_level=None)]))
    node = view.nodes["node-web-02"]
    assert node.evidence_level is None
    assert node.steps_since_investigated is None
    assert node.alerted is True
    assert node.insufficient is True


# --------------------------------------------------------------------------- #
# Boundary nodes (withheld private evidence) are excluded
# --------------------------------------------------------------------------- #
def test_boundary_nodes_are_excluded_from_the_view() -> None:
    view = evidence_view(
        _obs(
            [
                _node("node-app-01", scope=NodeScope.boundary, alert_count=1, severity=2),
                _node("node-gw-01", scope=NodeScope.own, alert_count=1, evidence_level=0),
            ]
        )
    )
    assert set(view.nodes) == {"node-gw-01"}
    assert "node-app-01" not in view.nodes


def test_evidence_view_matches_the_storyboard_fixture() -> None:
    obs = AgentObservation.model_validate_json(
        (_FIXTURES / "AgentObservation.dmz-step5.json").read_text()
    )
    view = evidence_view(obs)
    # Only the three own-scope nodes; the boundary node-app-01 is withheld.
    assert set(view.nodes) == {"node-gw-01", "node-web-01", "node-web-02"}
    assert view.nodes["node-gw-01"].evidence_level == 0
    assert view.nodes["node-gw-01"].steps_since_investigated == 3
    assert view.nodes["node-gw-01"].insufficient is True
    assert view.nodes["node-web-02"].insufficient is True
    assert view.nodes["node-web-01"].insufficient is False


# --------------------------------------------------------------------------- #
# Shared e_min threshold
# --------------------------------------------------------------------------- #
def test_e_min_threshold_changes_insufficiency() -> None:
    obs = _obs([_node("node-a", alert_count=1, evidence_level=2)])
    assert evidence_view(obs, e_min=2).nodes["node-a"].insufficient is False
    assert evidence_view(obs, e_min=3).nodes["node-a"].insufficient is True


def test_config_default_matches_function_default() -> None:
    obs = _obs([_node("node-a", alert_count=1, evidence_level=1)])
    config = EvidenceConfig()
    assert config.e_min == DEFAULT_E_MIN == 2
    with_config = evidence_view(obs, e_min=config.e_min)
    with_default = evidence_view(obs)
    assert with_config.nodes == with_default.nodes


def test_evidence_config_validates_range() -> None:
    assert EvidenceConfig(e_min=0).e_min == 0
    assert EvidenceConfig(e_min=3).e_min == 3
    with pytest.raises(ValueError):
        EvidenceConfig(e_min=-1)
    with pytest.raises(ValueError):
        EvidenceConfig(e_min=4)


@pytest.mark.parametrize("bad", [-1, 4, 10])
def test_evidence_view_rejects_out_of_range_e_min(bad: int) -> None:
    with pytest.raises(ValueError):
        evidence_view(_obs([_node("node-a", alert_count=1)]), e_min=bad)


# --------------------------------------------------------------------------- #
# Ordering and determinism
# --------------------------------------------------------------------------- #
def test_view_preserves_ascending_own_node_order() -> None:
    view = evidence_view(
        _obs(
            [
                _node("node-a", alert_count=1),
                _node("node-b", scope=NodeScope.boundary, alert_count=1),
                _node("node-c", alert_count=1),
            ]
        )
    )
    assert list(view.nodes) == ["node-a", "node-c"]


def test_evidence_view_is_deterministic() -> None:
    obs = _obs(
        [
            _node("node-gw-01", alert_count=1, severity=1, evidence_level=0, steps_since=3),
            _node("node-web-02", alert_count=1, severity=2),
        ]
    )
    assert evidence_view(obs).nodes == evidence_view(obs).nodes
