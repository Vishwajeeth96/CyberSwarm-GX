"""MOCK ground-truth separation: hidden truth never reaches agent-visible data.

Every agent-visible record the mock emits - observations, available actions and
events - must carry no evaluator-only field, both as Python objects and once
serialized to JSON (CONTRACTS section 9, rule X-10, conformance SC-6/SC-10). The
ground truth exists, but only behind the separate
:class:`~cyberswarm.interfaces.GroundTruthAccessor`. These checks drive a real
episode with varied actions so the scan sees alerts, investigations, status
changes and a restore, then confirm the split holds and that the scanner would
actually catch a leak.
"""

from __future__ import annotations

import json

from cyberswarm.contracts import (
    EVALUATOR_ONLY_FIELD_NAMES,
    ActionKind,
    AgentObservation,
    check_no_evaluator_fields,
)

_BASE = "scn-mock-lateral-eval-01"


def _scan_keys(payload: object) -> set[str]:
    """Recursively collect evaluator-only field names used as keys in ``payload``."""

    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in EVALUATOR_ONLY_FIELD_NAMES:
                found.add(key)
            found |= _scan_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            found |= _scan_keys(item)
    return found


def _policy(step: int, agent_id: str, available):
    own_investigate = sorted(
        a.target_node_id
        for a in available.actions
        if a.action_kind is ActionKind.investigate_node and a.target_node_id is not None
    )
    own_isolate = sorted(
        a.target_node_id
        for a in available.actions
        if a.action_kind is ActionKind.isolate_node and a.target_node_id is not None
    )
    own_restore = sorted(
        a.target_node_id
        for a in available.actions
        if a.action_kind is ActionKind.restore_node and a.target_node_id is not None
    )
    if step == 4 and own_isolate:
        return ActionKind.isolate_node, own_isolate[0]
    if step == 7 and own_restore:
        return ActionKind.restore_node, own_restore[0]
    if own_investigate:
        return ActionKind.investigate_node, own_investigate[0]
    return ActionKind.no_op, None


def _collect(new_simulator, load_scenario, validated_action, episode_id_for, steps=15):
    scenario = load_scenario(_BASE)
    sim, gt = new_simulator()
    episode_id = episode_id_for(_BASE, 2001)
    sim.reset(scenario, 2001, episode_id)

    observations: list = []
    available: list = []
    events: list = []
    truth: list = []
    guard = 0
    while not sim.done and guard < steps:
        guard += 1
        obs = sim.observe()
        observations.extend(obs.values())
        for counter, agent_id in enumerate(sim.agent_ids, start=1):
            available.append(sim.available_actions(agent_id))
            kind, target = _policy(sim.step, agent_id, sim.available_actions(agent_id))
            sim.apply_action(
                validated_action(episode_id, sim.step, agent_id, kind, target, counter=counter)
            )
        sim.end_step()
        events.extend(sim.drain_events())
        truth.append(gt.ground_truth())
    sim.close()
    return observations, available, events, truth


def test_agent_visible_records_carry_no_evaluator_field(
    new_simulator, load_scenario, validated_action, episode_id_for
) -> None:
    observations, available, events, _truth = _collect(
        new_simulator, load_scenario, validated_action, episode_id_for
    )
    # The episode must actually have produced varied content, or the scan would
    # be vacuous.
    assert observations and available and events
    event_types = {e.event_type.value for e in events}
    assert "action_applied" in event_types
    assert "alert_raised" in event_types

    for record in [*observations, *available, *events]:
        as_python = record.model_dump()
        as_json = json.loads(record.model_dump_json())
        assert _scan_keys(as_python) == set(), f"{type(record).__name__} leaks truth (python)"
        assert _scan_keys(as_json) == set(), f"{type(record).__name__} leaks truth (json)"


def test_observation_object_exposes_no_truth_attribute(
    new_simulator, load_scenario, validated_action, episode_id_for
) -> None:
    observations, _available, _events, _truth = _collect(
        new_simulator, load_scenario, validated_action, episode_id_for
    )
    for observation in observations:
        for attribute in ("compromised", "service_availability", "defense_availability"):
            assert not hasattr(observation, attribute)


def test_ground_truth_snapshot_does_carry_the_hidden_fields(
    new_simulator, load_scenario, validated_action, episode_id_for
) -> None:
    # The separation is only meaningful if the truth really lives somewhere: the
    # evaluator-only snapshot must contain exactly the fields withheld above.
    _obs, _available, _events, truth = _collect(
        new_simulator, load_scenario, validated_action, episode_id_for
    )
    assert truth
    present = _scan_keys(json.loads(truth[-1].model_dump_json()))
    assert {
        "compromised",
        "service_availability",
        "defense_availability",
        "incident_active",
    } <= present


def test_observation_model_rejects_injected_truth_field(
    new_simulator, load_scenario, validated_action, episode_id_for
) -> None:
    observations, _available, _events, _truth = _collect(
        new_simulator, load_scenario, validated_action, episode_id_for
    )
    leaked = observations[0].model_dump()
    leaked["compromised"] = True
    try:
        AgentObservation.model_validate(leaked)
    except Exception:
        pass
    else:
        raise AssertionError("AgentObservation must reject an injected 'compromised' field")


def test_check_no_evaluator_fields_on_written_files(
    tmp_path, new_simulator, load_scenario, validated_action, episode_id_for
) -> None:
    observations, available, events, truth = _collect(
        new_simulator, load_scenario, validated_action, episode_id_for
    )

    def _write_jsonl(path, records) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(record.model_dump_json() + "\n" for record in records),
            encoding="utf-8",
        )

    obs_file = tmp_path / "observations.jsonl"
    avl_file = tmp_path / "available.jsonl"
    evt_file = tmp_path / "events.jsonl"
    truth_file = tmp_path / "evaluator_only" / "ground_truth.jsonl"
    _write_jsonl(obs_file, observations)
    _write_jsonl(avl_file, available)
    _write_jsonl(evt_file, events)
    _write_jsonl(truth_file, truth)

    # Agent-visible files: no violation. The evaluator_only file is skipped.
    assert check_no_evaluator_fields([obs_file, avl_file, evt_file, truth_file]) == []

    # The guard is real: a hand-crafted agent-visible leak is reported.
    leaked = tmp_path / "leaky.jsonl"
    leaked.write_text(json.dumps({"compromised": True}) + "\n", encoding="utf-8")
    violations = check_no_evaluator_fields([leaked])
    assert violations and "compromised" in violations[0]
