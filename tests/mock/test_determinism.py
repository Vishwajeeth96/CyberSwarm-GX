"""MOCK determinism: common random numbers make an episode reproducible.

A fixed ``(scenario, seed)`` and a fixed action script must reproduce the exact
observation, event and ``state_digest()`` sequences byte for byte, and a
different seed must change the episode (DESIGN section 5.3, conformance SC-5,
design Property 7). These checks drive the public simulator loop rather than any
internal, so they also guard the determinism a matched evaluation relies on.
"""

from __future__ import annotations

from collections.abc import Callable

from cyberswarm.contracts import ActionKind, AvailableActions, NodeStatus

_BASE = "scn-mock-lateral-eval-01"
_STEPS = 15


def _noop_policy(step: int, agent_id: str, available: AvailableActions):
    return ActionKind.no_op, None


def _containment_policy(step: int, agent_id: str, available: AvailableActions):
    """A deterministic state-changing script: isolate, then investigate.

    On even steps isolate the lowest-id own node that is still ``up``; on odd
    steps investigate the lowest-id own node. Everything is a pure function of
    ``(step, agent_id, available)``, so two runs with the same seed must match.
    """

    isolate_targets = sorted(
        a.target_node_id
        for a in available.actions
        if a.action_kind is ActionKind.isolate_node and a.target_node_id is not None
    )
    investigate_targets = sorted(
        a.target_node_id
        for a in available.actions
        if a.action_kind is ActionKind.investigate_node and a.target_node_id is not None
    )
    if step % 2 == 0 and isolate_targets:
        return ActionKind.isolate_node, isolate_targets[0]
    if investigate_targets:
        return ActionKind.investigate_node, investigate_targets[0]
    return ActionKind.no_op, None


def _rollout(
    new_simulator,
    load_scenario,
    validated_action,
    episode_id_for,
    *,
    scenario_id: str,
    seed: int,
    policy: Callable,
    steps: int = _STEPS,
) -> tuple[list, list, list]:
    """Run one episode under ``policy`` and return (observations, events, digests)."""

    scenario = load_scenario(scenario_id)
    sim, _gt = new_simulator()
    episode_id = episode_id_for(scenario_id, seed)
    sim.reset(scenario, seed, episode_id)

    observations: list = []
    events: list = []
    digests: list = []
    for _ in range(steps):
        if sim.done:
            break
        obs = sim.observe()
        observations.append(
            {agent_id: ob.model_dump_json() for agent_id, ob in obs.items()}
        )
        for counter, agent_id in enumerate(sim.agent_ids, start=1):
            kind, target = policy(sim.step, agent_id, sim.available_actions(agent_id))
            sim.apply_action(
                validated_action(episode_id, sim.step, agent_id, kind, target, counter=counter)
            )
        sim.end_step()
        events.append([ev.model_dump_json() for ev in sim.drain_events()])
        digests.append(sim.state_digest())
    sim.close()
    return observations, events, digests


def test_same_seed_reproduces_noop_rollout(
    new_simulator, load_scenario, validated_action, episode_id_for
) -> None:
    args = (new_simulator, load_scenario, validated_action, episode_id_for)
    first = _rollout(*args, scenario_id=_BASE, seed=2001, policy=_noop_policy)
    second = _rollout(*args, scenario_id=_BASE, seed=2001, policy=_noop_policy)
    obs_a, ev_a, dg_a = first
    obs_b, ev_b, dg_b = second
    assert obs_a == obs_b, "observations are not reproducible for a fixed seed"
    assert ev_a == ev_b, "events are not reproducible for a fixed seed"
    assert dg_a == dg_b, "state digests are not reproducible for a fixed seed"
    assert len(dg_a) == _STEPS


def test_same_seed_reproduces_action_rollout(
    new_simulator, load_scenario, validated_action, episode_id_for
) -> None:
    # Determinism must hold when the agents actually change the state, not only
    # under no_ops: the common-random-number draws are independent of actions.
    args = (new_simulator, load_scenario, validated_action, episode_id_for)
    first = _rollout(*args, scenario_id=_BASE, seed=2001, policy=_containment_policy)
    second = _rollout(*args, scenario_id=_BASE, seed=2001, policy=_containment_policy)
    assert first == second


def test_different_seed_changes_the_episode(
    new_simulator, load_scenario, validated_action, episode_id_for
) -> None:
    args = (new_simulator, load_scenario, validated_action, episode_id_for)
    obs_a, _ev_a, dg_a = _rollout(*args, scenario_id=_BASE, seed=2001, policy=_noop_policy)
    obs_b, _ev_b, dg_b = _rollout(*args, scenario_id=_BASE, seed=2002, policy=_noop_policy)
    # At least the digest sequence must differ for a different seed (SC-5); in
    # practice the observations differ too once alerts and spread diverge.
    assert dg_a != dg_b, "a different seed must change the digest sequence"
    assert obs_a != obs_b, "a different seed must change the observation sequence"


def test_two_fresh_instances_same_seed_are_identical(
    new_simulator, load_scenario, validated_action, episode_id_for
) -> None:
    # Independent MockSimulator instances (the make_simulator contract) must not
    # share hidden state: same seed, same script, identical output.
    args = (new_simulator, load_scenario, validated_action, episode_id_for)
    run_a = _rollout(*args, scenario_id="scn-mock-lateral-train-01", seed=1000, policy=_containment_policy)
    run_b = _rollout(*args, scenario_id="scn-mock-lateral-train-01", seed=1000, policy=_containment_policy)
    assert run_a == run_b
