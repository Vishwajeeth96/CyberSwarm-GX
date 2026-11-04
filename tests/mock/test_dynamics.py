"""MOCK P0 dynamics: incident spread, restore completion and alert truth.

The attacker is a seeded random process over the in-memory graph (DESIGN section
5.3): one ``propagating_compromise`` incident, no learning adversary, no packets
or hosts. These tests read the evaluator-only ground-truth stream to pin the
hidden dynamics - incident onset, propagation over ``network_link`` edges,
restore completion timing, and the true/false-positive labelling of alerts - and
check that the common-random-number tables depend only on ``(seed, H, N, E)``.
"""

from __future__ import annotations

import numpy as np

from cyberswarm.contracts import (
    ActionKind,
    IncidentSpec,
    NodeStatus,
    ObservabilitySpec,
    ScenarioConfig,
)
from cyberswarm.mock.dynamics import CommonRandomTables


def _scn(scenario_id: str, **overrides) -> ScenarioConfig:
    """Build a v1 scenario with quiet defaults and the given overrides."""

    params = {
        "max_steps": 12,
        "observation_window_steps": 5,
        "entry_node_id": "node-web-02",
        "start_step": 3,
        "spread_probability": 0.0,
        "alert_detect_probability": 0.0,
        "false_alert_rate": 0.0,
        "alert_delay_max_steps": 0,
        "missing_value_rate": 0.0,
        "restore_duration_steps": 3,
        "objective_node_id": None,
    }
    params.update(overrides)
    return ScenarioConfig(
        scenario_id=scenario_id,
        topology_id="topo-mock-small-v1",
        max_steps=params["max_steps"],
        observation_window_steps=params["observation_window_steps"],
        incident=IncidentSpec(
            entry_node_id=params["entry_node_id"],
            start_step=params["start_step"],
            spread_probability=params["spread_probability"],
        ),
        observability=ObservabilitySpec(
            alert_detect_probability=params["alert_detect_probability"],
            false_alert_rate=params["false_alert_rate"],
            alert_delay_max_steps=params["alert_delay_max_steps"],
            missing_value_rate=params["missing_value_rate"],
        ),
        restore_duration_steps=params["restore_duration_steps"],
        objective_node_id=params["objective_node_id"],
    )


def _drive(sim, gt, episode_id, validated_action, *, actions=None, max_iters=12):
    """Run ``max_iters`` steps, applying scripted actions; collect truth/events.

    ``actions`` maps a step to ``{agent_id: (kind, target)}``; unlisted agents
    play ``no_op``. Returns ``(snapshots_by_step, all_events)``.
    """

    actions = actions or {}
    snapshots: dict[int, object] = {}
    events: list = []
    guard = 0
    while not sim.done and guard < max_iters:
        guard += 1
        sim.observe()
        step = sim.step
        step_actions = actions.get(step, {})
        for counter, agent_id in enumerate(sim.agent_ids, start=1):
            kind, target = step_actions.get(agent_id, (ActionKind.no_op, None))
            sim.apply_action(
                validated_action(episode_id, step, agent_id, kind, target, counter=counter)
            )
        result = sim.end_step()
        snapshots[result.step_closed] = gt.ground_truth()
        events.extend(sim.drain_events())
    return snapshots, events


def _compromised(snapshot) -> set[str]:
    return {node.node_id for node in snapshot.nodes if node.compromised}


def _status(snapshot, node_id: str) -> NodeStatus:
    return next(n.status for n in snapshot.nodes if n.node_id == node_id)


# --------------------------------------------------------------------------- #
# Incident onset and propagation
# --------------------------------------------------------------------------- #
def test_incident_starts_at_start_step_with_no_spread(
    new_simulator, validated_action, episode_id_for
) -> None:
    scenario = _scn("scn-mock-start-probe", start_step=3, spread_probability=0.0)
    sim, gt = new_simulator()
    episode_id = episode_id_for(scenario.scenario_id, 11)
    sim.reset(scenario, 11, episode_id)
    snapshots, _events = _drive(sim, gt, episode_id, validated_action, max_iters=8)
    sim.close()

    # The entry node is clean through step 2 and compromised from step 3; with
    # spread_probability 0.0 nothing else is ever compromised.
    assert _compromised(snapshots[2]) == set()
    assert _compromised(snapshots[3]) == {"node-web-02"}
    for step, snapshot in snapshots.items():
        assert _compromised(snapshot) <= {"node-web-02"}
        assert snapshots[step].incident_active == (step >= 3)


def test_incident_propagates_over_network_links(
    new_simulator, validated_action, episode_id_for
) -> None:
    scenario = _scn(
        "scn-mock-spread-probe", start_step=1, spread_probability=1.0, max_steps=10
    )
    sim, gt = new_simulator()
    episode_id = episode_id_for(scenario.scenario_id, 11)
    sim.reset(scenario, 11, episode_id)
    snapshots, _events = _drive(sim, gt, episode_id, validated_action, max_iters=5)
    sim.close()

    # Step 1: only the entry node. Step 2: its network_link neighbours fall.
    assert _compromised(snapshots[1]) == {"node-web-02"}
    assert {"node-web-02", "node-app-01", "node-gw-01"} <= _compromised(snapshots[2])
    # Without containment the compromised set only grows.
    ordered = [snapshots[s] for s in sorted(snapshots)]
    for earlier, later in zip(ordered, ordered[1:]):
        assert _compromised(earlier) <= _compromised(later)


# --------------------------------------------------------------------------- #
# Restore completion timing
# --------------------------------------------------------------------------- #
def test_restore_completes_after_restore_duration_steps(
    new_simulator, validated_action, episode_id_for
) -> None:
    scenario = _scn(
        "scn-mock-restore-window",
        start_step=1,
        spread_probability=0.0,
        restore_duration_steps=3,
        max_steps=12,
    )
    sim, gt = new_simulator()
    episode_id = episode_id_for(scenario.scenario_id, 11)
    sim.reset(scenario, 11, episode_id)
    # Restore the entry node at step 1, right after it is compromised.
    actions = {1: {"agent-dmz": (ActionKind.restore_node, "node-web-02")}}
    snapshots, events = _drive(
        sim, gt, episode_id, validated_action, actions=actions, max_iters=8
    )
    sim.close()

    # restore_duration_steps == 3: restoring at steps 1, 2 and 3, recovered at 4.
    for step in (1, 2, 3):
        assert _status(snapshots[step], "node-web-02") is NodeStatus.restoring
        assert "node-web-02" in _compromised(snapshots[step])
    assert _status(snapshots[4], "node-web-02") is NodeStatus.up
    assert "node-web-02" not in _compromised(snapshots[4])

    restore_events = [
        e for e in events
        if e.event_type is e.event_type.restore_completed and e.node_id == "node-web-02"
    ]
    assert len(restore_events) == 1
    assert restore_events[0].step == 4


# --------------------------------------------------------------------------- #
# Alert truth labelling
# --------------------------------------------------------------------------- #
def _alert_truth_by_event(snapshots) -> dict[str, bool]:
    table: dict[str, bool] = {}
    for snapshot in snapshots.values():
        for entry in snapshot.alert_truth:
            table[entry.event_id] = entry.true_positive
    return table


def _alert_events(events):
    return [e for e in events if e.event_type is e.event_type.alert_raised]


def test_compromised_node_raises_true_positive_alerts(
    new_simulator, validated_action, episode_id_for
) -> None:
    scenario = _scn(
        "scn-mock-true-alerts",
        start_step=1,
        spread_probability=0.0,
        alert_detect_probability=1.0,
        false_alert_rate=0.0,
        alert_delay_max_steps=0,
    )
    sim, gt = new_simulator()
    episode_id = episode_id_for(scenario.scenario_id, 11)
    sim.reset(scenario, 11, episode_id)
    snapshots, events = _drive(sim, gt, episode_id, validated_action, max_iters=8)
    sim.close()

    # An alert emitted in P0 of step t is labelled in the step-t snapshot; keep
    # only alerts whose step was actually closed and captured.
    alerts = [e for e in _alert_events(events) if e.step in snapshots]
    assert alerts, "a reliably-detected compromise must raise alerts"
    assert {e.node_id for e in alerts} == {"node-web-02"}
    truth = _alert_truth_by_event(snapshots)
    assert truth, "each emitted alert is labelled in the ground-truth stream"
    assert all(truth[e.event_id] is True for e in alerts)


def test_healthy_node_raises_false_positive_alerts(
    new_simulator, validated_action, episode_id_for
) -> None:
    scenario = _scn(
        "scn-mock-false-alerts",
        start_step=11,  # effectively no incident within the window
        spread_probability=0.0,
        alert_detect_probability=0.0,
        false_alert_rate=1.0,
        alert_delay_max_steps=0,
        max_steps=12,
    )
    sim, gt = new_simulator()
    episode_id = episode_id_for(scenario.scenario_id, 11)
    sim.reset(scenario, 11, episode_id)
    snapshots, events = _drive(sim, gt, episode_id, validated_action, max_iters=6)
    sim.close()

    alerts = [e for e in _alert_events(events) if e.step in snapshots]
    assert alerts, "a high false-alert rate on healthy nodes must raise alerts"
    truth = _alert_truth_by_event(snapshots)
    assert all(truth[e.event_id] is False for e in alerts)
    # None of the alerted nodes were actually compromised.
    compromised_anywhere: set[str] = set()
    for snapshot in snapshots.values():
        compromised_anywhere |= _compromised(snapshot)
    assert compromised_anywhere == set()


# --------------------------------------------------------------------------- #
# Common random numbers
# --------------------------------------------------------------------------- #
def test_common_random_tables_depend_only_on_seed_and_shape() -> None:
    same_a = CommonRandomTables(123, 40, 8, 19)
    same_b = CommonRandomTables(123, 40, 8, 19)
    other = CommonRandomTables(124, 40, 8, 19)

    for name in (
        "spread", "alert", "delay", "severity", "category", "false_alert",
        "health_noise", "traffic_noise", "missing_node", "missing_edge", "evidence",
    ):
        assert np.array_equal(getattr(same_a, name), getattr(same_b, name)), name
    # A different seed changes the draws.
    assert not np.array_equal(same_a.spread, other.spread)
    assert not np.array_equal(same_a.alert, other.alert)

    # Shapes are indexed by (step, node) or (step, edge); evidence carries two
    # draws per node (investigate and request_evidence).
    assert same_a.spread.shape == (40, 19)
    assert same_a.alert.shape == (40, 8)
    assert same_a.evidence.shape == (40, 8, 2)

    # AD-18: the health/traffic noise is a standard normal truncated at +/-3.
    for noise in (same_a.health_noise, same_a.traffic_noise):
        assert noise.min() >= -3.0
        assert noise.max() <= 3.0
