"""Unit tests for :mod:`cyberswarm.roles.detection` (stage S3, DC-10).

Covers the CONTRACTS 8.4 suspicion formula on hand examples, the ``alerted``
flag, boundary nodes with withheld (null) private evidence, node ordering and
id/metadata propagation, determinism, the empty observation, and that an
invalid ``producer`` raises instead of guessing.
"""

from __future__ import annotations

import pathlib

import pytest

from cyberswarm.contracts import (
    AgentObservation,
    ContractError,
    DetectionSummary,
    NodeObservation,
    NodeScope,
    NodeStatus,
    detection_id,
    observation_id,
)
from cyberswarm.roles import DEFAULT_PRODUCER, DetectionRole, suspicion_score

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
# Suspicion formula
# --------------------------------------------------------------------------- #
# Hand-computed from suspicion = min(1, 0.5*min(alerts,5)/5 + 0.5*severity/3).
_HAND_EXAMPLES = [
    (0, 0, 0.0),
    (2, 0, 0.2),
    (5, 0, 0.5),
    (10, 0, 0.5),  # alert count caps at 5
    (0, 3, 0.5),
    (3, 3, 0.8),
    (5, 3, 1.0),
    (6, 3, 1.0),  # cap keeps the sum at the ceiling, min() holds it at 1
    (50, 3, 1.0),
    (1, 1, 0.26666666666666666),  # node-gw-01 in the storyboard fixture
    (1, 2, 0.43333333333333335),  # node-web-02 in the storyboard fixture
]


@pytest.mark.parametrize(("alerts", "severity", "expected"), _HAND_EXAMPLES)
def test_suspicion_matches_formula_on_hand_examples(
    alerts: int, severity: int, expected: float
) -> None:
    assert suspicion_score(alerts, severity) == pytest.approx(expected)


@pytest.mark.parametrize(("alerts", "severity", "expected"), _HAND_EXAMPLES)
def test_summary_suspicion_matches_hand_examples(
    alerts: int, severity: int, expected: float
) -> None:
    summary = DetectionRole().summarize(
        _obs([_node("node-x", alert_count=alerts, severity=severity)])
    )
    assert summary.nodes[0].suspicion == pytest.approx(expected)


def test_suspicion_is_in_unit_range_over_the_domain() -> None:
    for alerts in range(0, 51):
        for severity in range(0, 4):
            value = suspicion_score(alerts, severity)
            assert 0.0 <= value <= 1.0


def test_suspicion_caps_alert_count_at_five() -> None:
    for severity in range(0, 4):
        capped = suspicion_score(5, severity)
        for alerts in (6, 7, 20, 50):
            assert suspicion_score(alerts, severity) == capped


def test_suspicion_is_monotonic_non_decreasing() -> None:
    for severity in range(0, 4):
        values = [suspicion_score(alerts, severity) for alerts in range(0, 11)]
        assert values == sorted(values)
    for alerts in range(0, 11):
        values = [suspicion_score(alerts, severity) for severity in range(0, 4)]
        assert values == sorted(values)


# --------------------------------------------------------------------------- #
# alerted flag
# --------------------------------------------------------------------------- #
def test_alerted_flag_true_iff_alert_present() -> None:
    summary = DetectionRole().summarize(
        _obs(
            [
                _node("node-a", alert_count=0, severity=0),
                _node("node-b", alert_count=1, severity=0),
                _node("node-c", alert_count=4, severity=3),
            ]
        )
    )
    flags = {node.node_id: node.alerted for node in summary.nodes}
    assert flags == {"node-a": False, "node-b": True, "node-c": True}


# --------------------------------------------------------------------------- #
# Boundary nodes (null private evidence) are summarised from visible fields
# --------------------------------------------------------------------------- #
def test_boundary_node_is_summarised_from_visible_alert_fields() -> None:
    summary = DetectionRole().summarize(
        _obs(
            [
                _node("node-app-01", scope=NodeScope.boundary, alert_count=1, severity=2),
                _node("node-web-02", scope=NodeScope.own, alert_count=0, severity=0),
            ]
        )
    )
    by_id = {node.node_id: node for node in summary.nodes}
    # The boundary node is present even though its private evidence is withheld.
    assert set(by_id) == {"node-app-01", "node-web-02"}
    assert by_id["node-app-01"].suspicion == pytest.approx(0.43333333333333335)
    assert by_id["node-app-01"].alerted is True


# --------------------------------------------------------------------------- #
# Ordering, ids and metadata
# --------------------------------------------------------------------------- #
def test_summary_preserves_observation_node_order() -> None:
    obs = _obs(
        [
            _node("node-a", alert_count=1),
            _node("node-b", alert_count=2),
            _node("node-c", alert_count=3),
        ]
    )
    summary = DetectionRole().summarize(obs)
    assert [n.node_id for n in summary.nodes] == [n.node_id for n in obs.nodes]


def test_summary_carries_ids_and_default_producer() -> None:
    obs = _obs([_node("node-a")], step=7, agent_id="agent-core")
    summary = DetectionRole().summarize(obs)
    assert summary.detection_id == detection_id(7, "agent-core")
    assert summary.detection_id == "det-0007-agent-core"
    assert summary.episode_id == _EPISODE_ID
    assert summary.step == 7
    assert summary.agent_id == "agent-core"
    assert summary.producer == DEFAULT_PRODUCER == "detection.rule-v1"


def test_custom_producer_is_recorded() -> None:
    summary = DetectionRole(producer="detection.rule-v2").summarize(_obs([_node("node-a")]))
    assert summary.producer == "detection.rule-v2"


def test_empty_observation_yields_empty_summary() -> None:
    summary = DetectionRole().summarize(_obs([]))
    assert summary.nodes == []
    assert isinstance(summary, DetectionSummary)


# --------------------------------------------------------------------------- #
# Determinism and fixture reproduction
# --------------------------------------------------------------------------- #
def test_summarize_is_deterministic() -> None:
    obs = _obs(
        [
            _node("node-gw-01", alert_count=1, severity=1, evidence_level=0, steps_since=3),
            _node("node-web-02", alert_count=1, severity=2),
        ]
    )
    role = DetectionRole()
    first = role.summarize(obs)
    second = role.summarize(obs)
    assert first.model_dump_json() == second.model_dump_json()
    # A second, independent instance with the same producer is identical too.
    assert DetectionRole().summarize(obs).model_dump_json() == first.model_dump_json()


def test_summarize_reproduces_the_storyboard_fixture_exactly() -> None:
    obs = AgentObservation.model_validate_json(
        (_FIXTURES / "AgentObservation.dmz-step5.json").read_text()
    )
    expected = DetectionSummary.model_validate_json(
        (_FIXTURES / "DetectionSummary.dmz-step5.json").read_text()
    )
    produced = DetectionRole().summarize(obs)
    assert produced.model_dump_json() == expected.model_dump_json()


# --------------------------------------------------------------------------- #
# Invalid input raises instead of guessing
# --------------------------------------------------------------------------- #
def test_invalid_producer_raises_on_summarize() -> None:
    # A dotted-quad producer is rejected by the ShortText security refinement.
    with pytest.raises(ContractError):
        DetectionRole(producer="1.2.3.4").summarize(_obs([_node("node-a")]))
