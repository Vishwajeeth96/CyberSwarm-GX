"""MOCK action effects: the eight allowlisted actions and their preconditions.

Covers every row of the DESIGN section 5.3 P4 effect table through the public
simulator (status transitions, emitted events, the ``ActionApplication`` shape),
the applicability rule that turns an out-of-state or out-of-authority action
into a ``failed`` result that changes nothing (conformance SC-9), the
state-dependent availability menu (sizes 22/36, ``no_op`` first, own-zone
targets only), and the evidence-draw mapping. Every action is an abstract state
transition: nothing here shells out, opens a socket or writes a file.
"""

from __future__ import annotations

import pytest

from cyberswarm.contracts import (
    ActionKind,
    IncidentSpec,
    NodeStatus,
    ObservabilitySpec,
    RateLimitLevel,
    ScenarioConfig,
)
from cyberswarm.mock.effects import (
    NodeState,
    action_precondition,
    apply_action_effect,
    draw_evidence_level,
)

# A quiet scenario: the incident starts near the horizon and never spreads, so
# every node is ``up`` and controllable at step 0.
_QUIET = ScenarioConfig(
    scenario_id="scn-mock-quiet-probe",
    topology_id="topo-mock-small-v1",
    max_steps=15,
    observation_window_steps=5,
    incident=IncidentSpec(
        entry_node_id="node-web-02", start_step=14, spread_probability=0.0
    ),
    observability=ObservabilitySpec(
        alert_detect_probability=0.0,
        false_alert_rate=0.0,
        alert_delay_max_steps=0,
        missing_value_rate=0.0,
    ),
    restore_duration_steps=2,
    objective_node_id=None,
)

_DMZ = "agent-dmz"
_TARGET = "node-web-02"

# (kind, target, expected status changes, expected emitted event types).
_ACTION_CASES = [
    (ActionKind.no_op, None, [], ["action_applied"]),
    (ActionKind.monitor_node, _TARGET, [], ["action_applied"]),
    (ActionKind.investigate_node, _TARGET, [], ["action_applied", "investigation_result"]),
    (ActionKind.request_evidence, _TARGET, [], ["action_applied", "investigation_result"]),
    (ActionKind.rate_limit_node, _TARGET, [("up", "rate_limited")], ["action_applied", "node_status_changed"]),
    (ActionKind.quarantine_node, _TARGET, [("up", "quarantined")], ["action_applied", "node_status_changed"]),
    (ActionKind.isolate_node, _TARGET, [("up", "isolated")], ["action_applied", "node_status_changed"]),
    (ActionKind.restore_node, _TARGET, [("up", "restoring")], ["action_applied", "node_status_changed"]),
]


def _reset_quiet(new_simulator, episode_id_for, seed: int = 5):
    sim, gt = new_simulator()
    episode_id = episode_id_for(_QUIET.scenario_id, seed)
    sim.reset(_QUIET, seed, episode_id)
    sim.drain_events()  # discard episode_started
    return sim, gt, episode_id


# --------------------------------------------------------------------------- #
# The eight action effects through the public simulator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kind", "target", "expected_changes", "expected_events"),
    _ACTION_CASES,
    ids=[case[0].value for case in _ACTION_CASES],
)
def test_action_effect(
    new_simulator,
    episode_id_for,
    validated_action,
    kind,
    target,
    expected_changes,
    expected_events,
) -> None:
    sim, _gt, episode_id = _reset_quiet(new_simulator, episode_id_for)
    before = sim.state_digest()

    app = sim.apply_action(validated_action(episode_id, 0, _DMZ, kind, target))

    assert app.status == "executed"
    assert app.error_code is None
    changes = [(c.status_from.value, c.status_to.value) for c in app.status_changes]
    assert changes == expected_changes
    events = [e.event_type.value for e in sim.drain_events()]
    assert events == expected_events
    assert len(app.outcome_event_ids) == len(expected_events)
    # Every executed action moves the state digest (a new execution id at least).
    assert sim.state_digest() != before


def test_no_op_event_has_null_node_and_action_metadata(
    new_simulator, episode_id_for, validated_action
) -> None:
    sim, _gt, episode_id = _reset_quiet(new_simulator, episode_id_for)
    app = sim.apply_action(validated_action(episode_id, 0, _DMZ, ActionKind.no_op))
    (event,) = sim.drain_events()
    assert event.event_type.value == "action_applied"
    assert event.node_id is None, "no_op carries no target"
    assert event.action_kind is ActionKind.no_op
    assert event.agent_id == _DMZ
    assert event.execution_id is not None
    assert app.affected_node_ids == ()


def test_targeted_action_reports_affected_node_and_events(
    new_simulator, episode_id_for, validated_action
) -> None:
    sim, _gt, episode_id = _reset_quiet(new_simulator, episode_id_for)
    app = sim.apply_action(
        validated_action(episode_id, 0, _DMZ, ActionKind.isolate_node, _TARGET)
    )
    assert app.affected_node_ids == (_TARGET,)
    applied, changed = sim.drain_events()
    assert applied.event_type.value == "action_applied"
    assert applied.node_id == _TARGET and applied.agent_id == _DMZ
    assert changed.event_type.value == "node_status_changed"
    assert changed.status_from is NodeStatus.up and changed.status_to is NodeStatus.isolated
    assert changed.execution_id == applied.execution_id


def test_investigation_reports_evidence_level(
    new_simulator, episode_id_for, validated_action
) -> None:
    sim, _gt, episode_id = _reset_quiet(new_simulator, episode_id_for)
    sim.apply_action(
        validated_action(episode_id, 0, _DMZ, ActionKind.investigate_node, _TARGET)
    )
    _applied, result = sim.drain_events()
    assert result.event_type.value == "investigation_result"
    assert result.node_id == _TARGET
    assert result.evidence_level is not None and 0 <= result.evidence_level <= 3


# --------------------------------------------------------------------------- #
# Preconditions: an inapplicable action fails and changes nothing (SC-9)
# --------------------------------------------------------------------------- #
def test_isolate_twice_second_fails_and_leaves_state_unchanged(
    new_simulator, episode_id_for, validated_action
) -> None:
    sim, _gt, episode_id = _reset_quiet(new_simulator, episode_id_for)
    first = sim.apply_action(
        validated_action(episode_id, 0, _DMZ, ActionKind.isolate_node, _TARGET)
    )
    assert first.status == "executed"
    digest_after_first = sim.state_digest()

    second = sim.apply_action(
        validated_action(episode_id, 0, _DMZ, ActionKind.isolate_node, _TARGET, counter=2)
    )
    assert second.status == "failed"
    assert second.error_code is not None
    assert second.status_changes == ()
    assert second.outcome_event_ids == ()
    assert sim.state_digest() == digest_after_first, "a failed action must not change state"


def test_out_of_authority_action_fails(
    new_simulator, episode_id_for, validated_action
) -> None:
    sim, _gt, episode_id = _reset_quiet(new_simulator, episode_id_for)
    before = sim.state_digest()
    # agent-dmz cannot touch a core-zone node.
    app = sim.apply_action(
        validated_action(episode_id, 0, _DMZ, ActionKind.investigate_node, "node-db-01")
    )
    assert app.status == "failed"
    assert sim.state_digest() == before


def test_unknown_agent_action_fails(
    new_simulator, episode_id_for, validated_action
) -> None:
    sim, _gt, episode_id = _reset_quiet(new_simulator, episode_id_for)
    before = sim.state_digest()
    app = sim.apply_action(
        validated_action(episode_id, 0, "agent-ghost", ActionKind.no_op)
    )
    assert app.status == "failed"
    assert sim.state_digest() == before


# --------------------------------------------------------------------------- #
# State-dependent availability menu (CONTRACTS 8.3, DESIGN 5.3)
# --------------------------------------------------------------------------- #
def test_available_actions_sizes_and_shape(new_simulator, episode_id_for) -> None:
    sim, _gt, _episode_id = _reset_quiet(new_simulator, episode_id_for)
    sizes = {}
    for agent_id in sim.agent_ids:
        available = sim.available_actions(agent_id)
        sizes[agent_id] = len(available.actions)
        assert available.actions[0].action_kind is ActionKind.no_op
        own = set(sim.authority(agent_id))
        # Every targeted action points at one of the agent's own nodes, and
        # every one of the eight kinds is offered while the nodes are all up.
        kinds = set()
        for action in available.actions:
            kinds.add(action.action_kind)
            if action.target_node_id is not None:
                assert action.target_node_id in own
        assert kinds == set(ActionKind)
    # 1 no_op + 7 targeted kinds * nodes: core has 5 nodes, dmz has 3.
    assert sizes == {"agent-core": 1 + 7 * 5, "agent-dmz": 1 + 7 * 3}


def test_available_actions_follow_state_after_containment(
    new_simulator, episode_id_for, validated_action
) -> None:
    sim, _gt, episode_id = _reset_quiet(new_simulator, episode_id_for)
    sim.apply_action(
        validated_action(episode_id, 0, _DMZ, ActionKind.isolate_node, _TARGET)
    )
    available = sim.available_actions(_DMZ)
    offered = {(a.action_kind, a.target_node_id) for a in available.actions}
    # An isolated node can no longer be isolated/quarantined/rate-limited, but it
    # can still be restored and investigated.
    assert (ActionKind.isolate_node, _TARGET) not in offered
    assert (ActionKind.quarantine_node, _TARGET) not in offered
    assert (ActionKind.rate_limit_node, _TARGET) not in offered
    assert (ActionKind.restore_node, _TARGET) in offered
    assert (ActionKind.investigate_node, _TARGET) in offered


def test_final_observation_offers_only_no_op(
    new_simulator, episode_id_for, validated_action
) -> None:
    sim, _gt, episode_id = _reset_quiet(new_simulator, episode_id_for)
    guard = 0
    while not sim.done and guard <= _QUIET.max_steps + 2:
        sim.observe()
        for agent_id in sim.agent_ids:
            sim.apply_action(validated_action(episode_id, sim.step, agent_id, ActionKind.no_op))
        sim.end_step()
        guard += 1
    assert sim.done
    for agent_id in sim.agent_ids:
        available = sim.available_actions(agent_id)
        assert [a.action_kind for a in available.actions] == [ActionKind.no_op]


# --------------------------------------------------------------------------- #
# Pure-function checks: precondition table and evidence draw mapping
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kind", "status", "expected"),
    [
        (ActionKind.rate_limit_node, NodeStatus.up, True),
        (ActionKind.rate_limit_node, NodeStatus.rate_limited, False),
        (ActionKind.rate_limit_node, NodeStatus.isolated, False),
        (ActionKind.quarantine_node, NodeStatus.up, True),
        (ActionKind.quarantine_node, NodeStatus.rate_limited, True),
        (ActionKind.quarantine_node, NodeStatus.isolated, False),
        (ActionKind.isolate_node, NodeStatus.quarantined, True),
        (ActionKind.isolate_node, NodeStatus.isolated, False),
        (ActionKind.restore_node, NodeStatus.isolated, True),
        (ActionKind.restore_node, NodeStatus.restoring, False),
        (ActionKind.investigate_node, NodeStatus.restoring, False),
        (ActionKind.investigate_node, NodeStatus.up, True),
        (ActionKind.no_op, NodeStatus.restoring, True),
    ],
)
def test_action_precondition_table(kind, status, expected) -> None:
    assert action_precondition(kind, status) is expected


def test_apply_effect_leaves_state_untouched_when_inapplicable() -> None:
    state = NodeState(node_id="node-web-02", status=NodeStatus.restoring, restore_left=2)
    outcome = apply_action_effect(
        state,
        ActionKind.investigate_node,
        step=4,
        duration_steps=None,
        level=None,
        restore_duration_steps=3,
        evidence_uniform=0.1,
    )
    assert outcome.applied is False
    assert state.status is NodeStatus.restoring and state.restore_left == 2


def test_rate_limit_sets_level() -> None:
    state = NodeState(node_id="node-web-02")
    outcome = apply_action_effect(
        state,
        ActionKind.rate_limit_node,
        step=0,
        duration_steps=None,
        level=RateLimitLevel.heavy,
        restore_duration_steps=3,
        evidence_uniform=0.0,
    )
    assert outcome.applied is True
    assert state.status is NodeStatus.rate_limited
    assert state.rate_level is RateLimitLevel.heavy


@pytest.mark.parametrize(
    ("uniform", "compromised", "strong", "expected"),
    [
        # investigate_node (strong=False), compromised node tends high.
        (0.0, True, False, 3),
        (0.59, True, False, 3),
        (0.60, True, False, 2),
        (0.84, True, False, 2),
        (0.85, True, False, 1),
        # investigate_node, healthy node tends to 0.
        (0.0, False, False, 0),
        (0.69, False, False, 0),
        (0.70, False, False, 1),
        (0.89, False, False, 1),
        (0.90, False, False, 2),
        (0.98, False, False, 3),
        # request_evidence (strong=True) is sharper.
        (0.0, True, True, 3),
        (0.89, True, True, 3),
        (0.90, True, True, 2),
        (0.0, False, True, 0),
        (0.90, False, True, 1),
    ],
)
def test_draw_evidence_level_mapping(uniform, compromised, strong, expected) -> None:
    assert draw_evidence_level(uniform, compromised=compromised, strong=strong) == expected


def test_evidence_draw_separates_compromise_in_expectation() -> None:
    # Averaged over the unit interval, a compromised node yields a higher
    # evidence level than a healthy one for both investigation kinds.
    grid = [i / 1000 for i in range(1000)]
    for strong in (False, True):
        comp = sum(draw_evidence_level(u, compromised=True, strong=strong) for u in grid)
        healthy = sum(draw_evidence_level(u, compromised=False, strong=strong) for u in grid)
        assert comp > healthy
