"""Step semantics at the contract layer (DELTA check DC-02).

Covers the phase/producer pairing and id-prefix rule for events, the embedded
step numbers that each per-step record must agree with, the "no future events"
rule on observations, the step range bound, and the canonical ascending node
ordering.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cyberswarm.contracts import (
    ActionKind,
    ActionSpec,
    AgentObservation,
    AvailableAction,
    AvailableActions,
    DetectionNode,
    DetectionSummary,
    EventProducer,
    EventRecord,
    GroundTruthNode,
    GroundTruthSnapshot,
    ImpactClass,
    NodeObservation,
    NodeScope,
    NodeScore,
    NodeScoreRecord,
    NodeStatus,
    Phase,
    ScorerKind,
)

EP = "ep-rule-conv-v1-mock-lateral-eval-01-s2001"


def _ev(event_id: str, phase: str, event_type: str, producer: str, **fields: object) -> dict:
    return {
        "schema_version": "1.0",
        "event_id": event_id,
        "episode_id": EP,
        "step": 5,
        "phase": phase,
        "event_type": event_type,
        "producer": producer,
        "visible_to": ["agent-dmz"],
        **fields,
    }


# event_type -> (valid payload, expected phase, expected producer, a wrong phase)
_EVENT_CASES = {
    "episode_started": (
        _ev("evt-000001", "lifecycle", "episode_started", "simulator"),
        Phase.lifecycle,
        EventProducer.simulator,
        "dynamics",
    ),
    "alert_raised": (
        _ev(
            "evt-000002", "dynamics", "alert_raised", "simulator",
            node_id="node-web-02", severity="medium", alert_category="traffic_anomaly",
        ),
        Phase.dynamics,
        EventProducer.simulator,
        "outcome",
    ),
    "restore_completed": (
        _ev(
            "evt-000003", "dynamics", "restore_completed", "simulator",
            node_id="node-web-02", status_from="restoring", status_to="up",
        ),
        Phase.dynamics,
        EventProducer.simulator,
        "outcome",
    ),
    "node_status_changed": (
        _ev(
            "evt-000004", "outcome", "node_status_changed", "simulator",
            node_id="node-web-02", status_from="up", status_to="isolated",
            execution_id="exe-000005",
        ),
        Phase.outcome,
        EventProducer.simulator,
        "dynamics",
    ),
    "action_applied": (
        _ev(
            "evt-000005", "outcome", "action_applied", "simulator",
            node_id="node-web-02", agent_id="agent-dmz", action_kind="isolate_node",
            proposal_id="prop-000006", execution_id="exe-000005",
        ),
        Phase.outcome,
        EventProducer.simulator,
        "dynamics",
    ),
    "investigation_result": (
        _ev(
            "evt-000006", "outcome", "investigation_result", "simulator",
            node_id="node-web-02", evidence_level=2, agent_id="agent-dmz",
            execution_id="exe-000003",
        ),
        Phase.outcome,
        EventProducer.simulator,
        "dynamics",
    ),
    "recovery_verified": (
        _ev(
            "cev-000001", "outcome", "recovery_verified", "coordinator",
            node_id="node-web-02", execution_id="exe-000007",
        ),
        Phase.outcome,
        EventProducer.coordinator,
        "dynamics",
    ),
    "recovery_failed": (
        _ev(
            "cev-000002", "outcome", "recovery_failed", "coordinator",
            node_id="node-web-02", execution_id="exe-000007",
        ),
        Phase.outcome,
        EventProducer.coordinator,
        "dynamics",
    ),
    "episode_ended": (
        _ev(
            "evt-000007", "lifecycle", "episode_ended", "simulator",
            terminated=False, truncated=True,
        ),
        Phase.lifecycle,
        EventProducer.simulator,
        "dynamics",
    ),
}


@pytest.mark.parametrize("event_type", list(_EVENT_CASES))
def test_event_phase_and_producer_pairing(event_type: str) -> None:
    payload, phase, producer, _wrong = _EVENT_CASES[event_type]
    event = EventRecord.model_validate(payload)
    assert event.phase is phase
    assert event.producer is producer


@pytest.mark.parametrize("event_type", list(_EVENT_CASES))
def test_event_wrong_phase_rejected(event_type: str) -> None:
    payload, _phase, _producer, wrong_phase = _EVENT_CASES[event_type]
    bad = dict(payload)
    bad["phase"] = wrong_phase
    with pytest.raises(ValidationError):
        EventRecord.model_validate(bad)


def test_simulator_event_must_use_evt_prefix() -> None:
    bad = dict(_EVENT_CASES["alert_raised"][0])
    bad["event_id"] = "cev-000002"  # coordinator prefix on a simulator event
    with pytest.raises(ValidationError):
        EventRecord.model_validate(bad)


@pytest.mark.parametrize("step", [-1, 10001])
def test_step_out_of_range_rejected(step: int) -> None:
    bad = dict(_EVENT_CASES["alert_raised"][0])
    bad["step"] = step
    with pytest.raises(ValidationError):
        EventRecord.model_validate(bad)


# --------------------------------------------------------------------------- #
# No future events in an observation
# --------------------------------------------------------------------------- #
def _observation(step: int, event_step: int) -> AgentObservation:
    event = EventRecord.model_validate(
        _ev(
            "evt-000002", "dynamics", "alert_raised", "simulator",
            node_id="node-web-02", severity="medium", alert_category="traffic_anomaly",
        )
        | {"step": event_step}
    )
    return AgentObservation(
        observation_id=f"obs-{step:04d}-agent-dmz",
        episode_id=EP,
        step=step,
        agent_id="agent-dmz",
        zone_id="zone-dmz",
        max_steps=40,
        observation_window_steps=5,
        nodes=[
            NodeObservation(
                node_id="node-web-02",
                scope=NodeScope.own,
                visible_status=NodeStatus.up,
                alert_count_recent=1,
                max_alert_severity=2,
                service_health=0.9,
            )
        ],
        edges=[],
        recent_events=[event],
        last_step_feedback=[],
    )


def test_observation_accepts_current_and_past_events() -> None:
    assert _observation(step=5, event_step=5).step == 5


def test_observation_rejects_future_event() -> None:
    with pytest.raises(ValidationError):
        _observation(step=5, event_step=6)


# --------------------------------------------------------------------------- #
# Embedded step numbers must match the record's step and agent
# --------------------------------------------------------------------------- #
def test_observation_id_must_encode_step_and_agent() -> None:
    with pytest.raises(ValidationError):
        AgentObservation(
            observation_id="obs-0004-agent-dmz",  # step says 5
            episode_id=EP,
            step=5,
            agent_id="agent-dmz",
            zone_id="zone-dmz",
            max_steps=40,
            observation_window_steps=5,
            nodes=[],
            edges=[],
            recent_events=[],
            last_step_feedback=[],
        )


def test_detection_id_must_encode_step_and_agent() -> None:
    with pytest.raises(ValidationError):
        DetectionSummary(
            detection_id="det-0004-agent-dmz",
            episode_id=EP,
            step=5,
            agent_id="agent-dmz",
            producer="detection.rule-v1",
            nodes=[DetectionNode(node_id="node-web-02", suspicion=0.1, alerted=True)],
        )


def test_available_actions_id_must_encode_step_and_agent() -> None:
    no_op = AvailableAction(
        action_kind=ActionKind.no_op,
        target_node_id=None,
        impact_class=ImpactClass.none,
        high_impact=False,
        cost_units=0.0,
    )
    with pytest.raises(ValidationError):
        AvailableActions(
            available_actions_id="avl-0004-agent-dmz",
            episode_id=EP,
            step=5,
            agent_id="agent-dmz",
            actions=[no_op],
        )


def test_node_score_id_prefix_must_encode_step_and_agent() -> None:
    with pytest.raises(ValidationError):
        NodeScoreRecord(
            node_score_id="nsc-0004-agent-dmz-detection-rule-v1",
            episode_id=EP,
            step=5,
            agent_id="agent-dmz",
            scorer_id="detection.rule-v1",
            scorer_kind=ScorerKind.detection_rule,
            nodes=[NodeScore(node_id="node-web-02", score=0.1)],
        )


def test_proposal_observation_id_must_match_step_and_agent() -> None:
    from cyberswarm.contracts import PolicyProposal

    with pytest.raises(ValidationError):
        PolicyProposal(
            proposal_id="prop-000001",
            episode_id=EP,
            step=5,
            agent_id="agent-dmz",
            policy_id="pol-rule-conv-v1",
            observation_id="obs-0004-agent-dmz",  # step says 5
            action=ActionSpec(action_kind=ActionKind.no_op),
        )


# --------------------------------------------------------------------------- #
# Canonical ascending node ordering
# --------------------------------------------------------------------------- #
def test_observation_nodes_must_be_ascending() -> None:
    with pytest.raises(ValidationError):
        AgentObservation(
            observation_id="obs-0005-agent-dmz",
            episode_id=EP,
            step=5,
            agent_id="agent-dmz",
            zone_id="zone-dmz",
            max_steps=40,
            observation_window_steps=5,
            nodes=[
                NodeObservation(
                    node_id="node-web-02",
                    scope=NodeScope.own,
                    visible_status=NodeStatus.up,
                    alert_count_recent=0,
                    max_alert_severity=0,
                ),
                NodeObservation(
                    node_id="node-app-01",  # out of ascending order
                    scope=NodeScope.own,
                    visible_status=NodeStatus.up,
                    alert_count_recent=0,
                    max_alert_severity=0,
                ),
            ],
            edges=[],
            recent_events=[],
            last_step_feedback=[],
        )


def test_ground_truth_nodes_must_be_ascending() -> None:
    def node(node_id: str) -> GroundTruthNode:
        return GroundTruthNode(
            node_id=node_id,
            compromised=False,
            service_availability=1.0,
            defense_availability=1.0,
            contained=False,
            status=NodeStatus.up,
        )

    with pytest.raises(ValidationError):
        GroundTruthSnapshot(
            episode_id=EP,
            step=5,
            incident_active=False,
            nodes=[node("node-web-02"), node("node-app-01")],
        )
