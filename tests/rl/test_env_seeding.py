"""Seeding and determinism of the RL environment (task 7.4).

The environment makes reproducibility mandatory: ``reset`` requires a
non-negative integer seed (``None``, a float, a bool or a negative int raise),
a fixed seed reproduces the trajectory exactly, and a different seed changes it.

Validates the seed-reproducibility part of design Property 12.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rollout(env, seed, scenario_id="scn-rl-env-trunc"):
    """Reset under ``seed`` and step ``no_op`` to the end.

    Returns ``(obs_trajectory, rewards)`` as stacked float arrays: the
    ascending-agent concatenation of every observation vector, and the per-step
    team reward.
    """

    observations, _infos = env.reset(seed=seed, options={"scenario_id": scenario_id})
    vectors = [_concat(observations)]
    rewards = []
    while env.agents:
        observations, step_rewards, _term, _trunc, _infos = env.step(
            {agent: 0 for agent in env.agents}
        )
        vectors.append(_concat(observations))
        rewards.append(sum(step_rewards.values()))
    return np.concatenate(vectors), np.asarray(rewards, dtype=np.float64)


def _concat(observations):
    return np.concatenate([observations[agent]["obs"] for agent in sorted(observations)])


@pytest.mark.parametrize("bad_seed", [None, 1.0, -1, True])
def test_reset_rejects_a_non_integer_or_negative_seed(make_env, bad_seed):
    env = make_env().env
    with pytest.raises(ValueError):
        env.reset(seed=bad_seed)


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_reset_accepts_a_non_negative_integer_seed(make_env, seed):
    env = make_env().env
    observations, infos = env.reset(seed=seed, options={"scenario_id": "scn-rl-env-trunc"})
    assert set(observations) == set(env.possible_agents)
    assert set(infos) == set(env.possible_agents)


def test_a_fixed_seed_reproduces_the_trajectory(make_env):
    obs_a, rewards_a = _rollout(make_env().env, seed=7)
    obs_b, rewards_b = _rollout(make_env().env, seed=7)
    assert np.array_equal(obs_a, obs_b)
    assert np.array_equal(rewards_a, rewards_b)


def test_a_different_seed_changes_the_trajectory(make_env):
    obs_a, rewards_a = _rollout(make_env().env, seed=7)
    obs_b, rewards_b = _rollout(make_env().env, seed=99)
    # The scenario is pinned, so the divergence comes purely from the simulator
    # seed stream (alerts, evidence and the weakened service-health channel).
    assert not np.array_equal(obs_a, obs_b) or not np.array_equal(rewards_a, rewards_b)
