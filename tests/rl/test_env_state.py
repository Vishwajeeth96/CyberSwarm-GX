"""The centralized-critic ``state()`` hook of the RL environment (task 7.4).

``state()`` returns the ascending-agent-order concatenation of the agents'
current observation vectors (agent-visible data only), shape ``(n_agents * D,)``
and ``float32`` in ``[0, 1]``. It is available only between ``reset`` and the
episode end; it raises :class:`EnvStateError` before the first ``reset`` and
after the episode is done. It is the hook a later MAPPO critic uses; the IPPO
critics never call it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cyberswarm.rl.env import EnvStateError

#: ``n_agents * D`` for the two mock agents in each observation mode.
MODE_STATE_LEN = {"conventional": 2 * 138, "graph_features": 2 * 202}


def _ascending_concat(observations):
    return np.concatenate([observations[agent]["obs"] for agent in sorted(observations)])


@pytest.mark.parametrize("mode, length", MODE_STATE_LEN.items())
def test_state_has_the_expected_length_dtype_and_range(make_env, mode, length):
    env = make_env(mode=mode).env
    env.reset(seed=0, options={"scenario_id": "scn-rl-env-trunc"})
    state = env.state()
    assert state.shape == (length,)
    assert state.dtype == np.float32
    assert np.isfinite(state).all()
    assert float(state.min()) >= 0.0
    assert float(state.max()) <= 1.0


def test_state_is_the_ascending_agent_concatenation_of_obs(make_env):
    env = make_env().env
    observations, _infos = env.reset(seed=1, options={"scenario_id": "scn-rl-env-trunc"})
    assert np.array_equal(env.state(), _ascending_concat(observations))


def test_state_tracks_the_current_step_observations(make_env):
    env = make_env().env
    env.reset(seed=1, options={"scenario_id": "scn-rl-env-trunc"})
    observations, _rew, _term, _trunc, _infos = env.step(
        {agent: 0 for agent in env.agents}
    )
    # After a step, state() reflects the new observations, not the old ones.
    assert np.array_equal(env.state(), _ascending_concat(observations))


def test_state_raises_before_reset(make_env):
    env = make_env().env
    with pytest.raises(EnvStateError):
        env.state()


def test_state_raises_after_the_episode_ends(make_env):
    env = make_env().env
    env.reset(seed=1, options={"scenario_id": "scn-rl-env-trunc"})
    while env.agents:
        env.step({agent: 0 for agent in env.agents})
    with pytest.raises(EnvStateError):
        env.state()
