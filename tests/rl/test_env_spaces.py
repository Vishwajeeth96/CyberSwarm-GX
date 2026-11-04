"""Observation and action spaces of the RL environment (task 7.4).

Covers the PettingZoo/Gymnasium space declarations and the well-formedness of
the vectors the env actually returns: the per-agent ``Discrete`` sizes (22 and
36 on the mock), the ``Dict({action_mask, obs})`` observation space, the
documented dimensions of both representations (138 conventional, 202 graph), and
the invariant that every observed value is finite and in ``[0, 1]`` with the
``no_op`` index always unmasked.

Validates design Property 11 (observation well-formedness) and the action-mask
part of Property 12.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

# The mock topology has N = 8 nodes split into a 3-node DMZ zone (agent-dmz) and
# a 5-node core zone (agent-core); A = 1 + 7 * |own nodes|.
EXPECTED_ACTION_SIZES = {"agent-dmz": 22, "agent-core": 36}
MODE_DIMS = {"conventional": 138, "graph_features": 202}


def _rollout_observations(env, seed):
    """Reset under ``seed`` and drive ``no_op`` to the end, collecting every obs."""

    observations, _ = env.reset(seed=seed, options={"scenario_id": "scn-rl-env-trunc"})
    collected = [observations]
    while env.agents:
        observations, _rewards, _term, _trunc, _infos = env.step(
            {agent: 0 for agent in env.agents}
        )
        collected.append(observations)
    return collected


def test_possible_agents_are_the_two_sorted_zone_agents(make_env):
    env = make_env().env
    assert env.possible_agents == ["agent-core", "agent-dmz"]


def test_action_spaces_are_discrete_with_the_documented_sizes(make_env):
    env = make_env().env
    for agent, size in EXPECTED_ACTION_SIZES.items():
        space = env.action_space(agent)
        assert isinstance(space, spaces.Discrete)
        assert space.n == size
        # The action table and the Discrete space agree on the size.
        assert env.action_table(agent).n == size


@pytest.mark.parametrize("mode, dim", MODE_DIMS.items())
def test_observation_space_structure_matches_the_mode(make_env, mode, dim):
    env = make_env(mode=mode).env
    for agent, size in EXPECTED_ACTION_SIZES.items():
        space = env.observation_space(agent)
        assert isinstance(space, spaces.Dict)
        assert set(space.spaces) == {"action_mask", "obs"}

        mask_space = space["action_mask"]
        assert isinstance(mask_space, spaces.MultiBinary)
        assert mask_space.n == size

        obs_space = space["obs"]
        assert isinstance(obs_space, spaces.Box)
        assert obs_space.shape == (dim,)
        assert obs_space.dtype == np.float32
        assert float(obs_space.low.min()) == 0.0
        assert float(obs_space.high.max()) == 1.0


@pytest.mark.parametrize("mode, dim", MODE_DIMS.items())
def test_reset_observations_are_well_formed(make_env, mode, dim):
    harness = make_env(mode=mode)
    env = harness.env
    observations, _infos = env.reset(seed=0, options={"scenario_id": "scn-rl-env-trunc"})

    assert set(observations) == set(env.possible_agents)
    for agent, observation in observations.items():
        obs = observation["obs"]
        mask = observation["action_mask"]

        # Documented dimension, float32, finite, and bounded to [0, 1].
        assert obs.shape == (dim,)
        assert obs.dtype == np.float32
        assert np.isfinite(obs).all()
        assert float(obs.min()) >= 0.0
        assert float(obs.max()) <= 1.0

        # The mask aligns with the Discrete size and no_op (index 0) is unmasked.
        assert mask.shape == (EXPECTED_ACTION_SIZES[agent],)
        assert int(mask[0]) == 1

        # The whole dict is a member of the declared observation space.
        assert env.observation_space(agent).contains(observation)


@pytest.mark.parametrize("mode", MODE_DIMS)
def test_every_observation_of_an_episode_is_well_formed(make_env, mode):
    """Well-formedness holds at every step, including the final observation."""

    env = make_env(mode=mode).env
    dim = MODE_DIMS[mode]
    for observations in _rollout_observations(env, seed=5):
        for agent, observation in observations.items():
            obs = observation["obs"]
            mask = observation["action_mask"]
            assert obs.shape == (dim,)
            assert np.isfinite(obs).all()
            assert float(obs.min()) >= 0.0
            assert float(obs.max()) <= 1.0
            assert mask.shape == (EXPECTED_ACTION_SIZES[agent],)
            # no_op is always available, so a valid action always exists.
            assert int(mask[0]) == 1
