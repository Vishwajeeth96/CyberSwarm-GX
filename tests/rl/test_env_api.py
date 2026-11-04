"""PettingZoo parallel-API conformance for the RL environment (task 7.4).

``pettingzoo.test.parallel_api_test`` exercises the parallel API contract
(spaces, the five step dicts, the agent lifecycle, and that observations stay in
their spaces) over several cycles. It calls ``reset()`` without a seed, but this
environment makes a seed mandatory (reproducibility is required and
``reset(None)`` raises). The conformance check is therefore run against a thin
subclass whose ``reset`` falls back to ``DEFAULT_ENV_SEED`` when no seed is
given; the production environment still requires an explicit seed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pettingzoo.test import parallel_api_test

from cyberswarm.rl.env import DEFAULT_ENV_SEED, CyberSwarmParallelEnv


class _SeedDefaultingEnv(CyberSwarmParallelEnv):
    """A test-only subclass that tolerates ``reset()`` without a seed.

    The parallel-API test resets without a seed; the real environment mandates
    one. Supplying the default here keeps the check deterministic without
    weakening the production contract.
    """

    def reset(
        self, seed: int | None = None, options: Mapping[str, Any] | None = None
    ):
        if seed is None:
            seed = DEFAULT_ENV_SEED
        return super().reset(seed=seed, options=options)


def test_parallel_api_conformance(make_env):
    env = make_env(env_cls=_SeedDefaultingEnv).env
    parallel_api_test(env, num_cycles=20)
