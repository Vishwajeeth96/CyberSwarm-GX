"""Episode lifecycle and invalid-action handling of the RL environment (task 7.4).

Covers the ``reset`` -> ``step`` -> end sequence: termination versus truncation,
the agents list emptying at the end, the ``EnvStateError`` raised when ``step``
or ``state`` is called out of the lifecycle, the rejection of malformed action
dicts, the ``no_op``-only final observation mask, and the masked-but-in-range
substitution (flagged ``invalid_action`` and never forwarded to the coordinator).

Validates design Property 12 (action-mask invariants: an in-range masked index
is substituted with ``no_op`` and flagged, and the masked proposal never reaches
the coordinator).
"""

from __future__ import annotations

import pytest

from cyberswarm.contracts import ActionKind
from cyberswarm.rl.env import EnvStateError


def _drive_to_end(env, seed, scenario_id):
    """Reset and step ``no_op`` until the episode ends; return per-step outcomes.

    Each entry is ``(observations, rewards, terminations, truncations, infos)``.
    """

    env.reset(seed=seed, options={"scenario_id": scenario_id})
    outcomes = []
    while env.agents:
        outcomes.append(env.step({agent: 0 for agent in env.agents}))
    return outcomes


def test_reset_returns_observations_and_infos_for_every_agent(make_env):
    env = make_env().env
    observations, infos = env.reset(seed=0, options={"scenario_id": "scn-rl-env-trunc"})
    assert set(observations) == set(env.possible_agents)
    assert set(infos) == set(env.possible_agents)
    assert env.agents == env.possible_agents


def test_episode_terminates_on_objective_reached(make_env):
    env = make_env().env
    outcomes = _drive_to_end(env, seed=1, scenario_id="scn-rl-env-obj")

    # Only the last step ends the episode, and it ends by termination.
    for _obs, _rew, term, trunc, _info in outcomes[:-1]:
        assert not any(term.values())
        assert not any(trunc.values())

    _obs, _rew, term, trunc, _info = outcomes[-1]
    assert all(term.values())  # terminated for every agent
    assert not any(trunc.values())  # not a truncation
    assert env.agents == []  # the agents list empties at the end


def test_episode_truncates_at_the_horizon(make_env):
    env = make_env().env  # SCENARIO_TRUNCATE, max_steps = 6, never terminates
    outcomes = _drive_to_end(env, seed=1, scenario_id="scn-rl-env-trunc")

    assert len(outcomes) == 6  # one step per horizon slot
    for _obs, _rew, term, trunc, _info in outcomes[:-1]:
        assert not any(term.values())
        assert not any(trunc.values())

    _obs, _rew, term, trunc, _info = outcomes[-1]
    assert not any(term.values())  # truncation, not termination
    assert all(trunc.values())
    assert env.agents == []


@pytest.mark.parametrize("scenario_id", ["scn-rl-env-trunc", "scn-rl-env-obj"])
def test_final_observation_mask_allows_only_no_op(make_env, scenario_id):
    env = make_env().env
    outcomes = _drive_to_end(env, seed=2, scenario_id=scenario_id)

    final_observations = outcomes[-1][0]
    for agent, observation in final_observations.items():
        mask = observation["action_mask"]
        assert int(mask[0]) == 1  # no_op stays available
        assert int(mask.sum()) == 1  # and it is the *only* available action


def test_step_before_reset_raises(make_env):
    env = make_env().env
    with pytest.raises(EnvStateError):
        env.step({})


def test_step_after_episode_end_raises(make_env):
    env = make_env().env
    _drive_to_end(env, seed=1, scenario_id="scn-rl-env-trunc")
    assert env.agents == []
    with pytest.raises(EnvStateError):
        env.step({"agent-core": 0, "agent-dmz": 0})


def test_missing_agent_key_raises(make_env):
    env = make_env().env
    env.reset(seed=0, options={"scenario_id": "scn-rl-env-trunc"})
    with pytest.raises(ValueError):
        env.step({"agent-dmz": 0})  # agent-core omitted


def test_unknown_agent_key_raises(make_env):
    env = make_env().env
    env.reset(seed=0, options={"scenario_id": "scn-rl-env-trunc"})
    with pytest.raises(ValueError):
        env.step({"agent-core": 0, "agent-dmz": 0, "agent-ghost": 0})


def test_out_of_range_action_index_raises(make_env):
    env = make_env().env
    env.reset(seed=0, options={"scenario_id": "scn-rl-env-trunc"})
    with pytest.raises(ValueError):
        env.step({"agent-core": 0, "agent-dmz": 999})


def test_masked_in_range_index_is_substituted_and_never_reaches_the_coordinator(
    make_env,
):
    harness = make_env()
    env = harness.env
    observations, _infos = env.reset(seed=3, options={"scenario_id": "scn-rl-env-trunc"})

    # Isolating a healthy own node is available at step 0; afterwards the
    # containment actions on that node (rate_limit/quarantine/isolate) are masked.
    table = env.action_table("agent-dmz")
    isolate_index = next(
        i
        for i, spec in enumerate(table.specs)
        if spec.action_kind is ActionKind.isolate_node
        and spec.target_node_id == "node-web-01"
    )
    assert int(observations["agent-dmz"]["action_mask"][isolate_index]) == 1

    observations, _rew, _term, _trunc, _info = env.step(
        {"agent-dmz": isolate_index, "agent-core": 0}
    )
    assert harness.coordinator.last_actions["agent-dmz"].action_kind is (
        ActionKind.isolate_node
    )

    # At the next step that index is masked; the env must substitute no_op.
    mask = observations["agent-dmz"]["action_mask"]
    masked_index = next(i for i, bit in enumerate(mask) if not bit)
    assert masked_index != 0  # no_op is never masked
    masked_kind = table.spec_of(masked_index).action_kind
    assert masked_kind is not ActionKind.no_op

    observations, _rew, _term, _trunc, infos = env.step(
        {"agent-dmz": masked_index, "agent-core": 0}
    )

    # The masked agent is flagged; the other agent (a valid no_op) is not.
    assert infos["agent-dmz"]["invalid_action"] is True
    assert infos["agent-core"]["invalid_action"] is False

    # The coordinator received a no_op for the masked agent, never the masked
    # action: the masked proposal never reaches the safety boundary.
    seen = harness.coordinator.last_actions["agent-dmz"]
    assert seen.action_kind is ActionKind.no_op
    assert seen.action_kind is not masked_kind

    # The reward model charged the invalid-action guard for the substitution.
    breakdown = env.last_reward_breakdown
    assert breakdown is not None
    assert breakdown.raw["invalid_action"] >= 1.0
