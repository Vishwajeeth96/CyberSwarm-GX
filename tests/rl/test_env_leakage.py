"""Agent-visibility and leakage guard for the RL environment (task 7.4).

The ``info`` dictionaries and observations the environment hands a policy carry
agent-visible data only. ``info[agent]`` has exactly the six documented keys
(``step``, ``episode_id``, ``proposal_id``, ``validation``, ``reason_codes``,
``invalid_action``) and so cannot carry ground truth, a reward breakdown, a
termination reason or future events; and a scan of the serialized infos and
observations finds no ``GroundTruthSnapshot`` field name.

Validates design Property 5 (no evaluator-only leakage), at the RL boundary.
"""

from __future__ import annotations

import json

import pytest

from cyberswarm.contracts import check_no_evaluator_fields
from cyberswarm.contracts.consistency import EVALUATOR_ONLY_FIELD_NAMES

#: The only keys the environment is allowed to place in ``info[agent]`` (env
#: docstring / RL_IMPLEMENTATION 2.7).
DOCUMENTED_INFO_KEYS = frozenset(
    {"step", "episode_id", "proposal_id", "validation", "reason_codes", "invalid_action"}
)

#: Names that must never surface at the policy boundary besides ground truth.
FORBIDDEN_INFO_KEYS = frozenset(
    {
        "reward",
        "rewards",
        "reward_breakdown",
        "term_totals",
        "termination_reason",
        "terminated",
        "truncated",
        "recent_events",
        "ground_truth",
    }
)


def _collect(env, seed, scenario_id):
    """Drive ``no_op`` to the end, returning every ``(observations, infos)`` pair."""

    observations, infos = env.reset(seed=seed, options={"scenario_id": scenario_id})
    collected = [(observations, infos)]
    while env.agents:
        observations, _rew, _term, _trunc, infos = env.step(
            {agent: 0 for agent in env.agents}
        )
        collected.append((observations, infos))
    return collected


@pytest.mark.parametrize("mode", ["conventional", "graph_features"])
def test_info_has_exactly_the_documented_keys(make_env, mode):
    env = make_env(mode=mode).env
    for _observations, infos in _collect(env, seed=4, scenario_id="scn-rl-env-trunc"):
        for agent, info in infos.items():
            assert set(info) == DOCUMENTED_INFO_KEYS
            # No reward breakdown, termination reason, event stream or truth.
            assert set(info).isdisjoint(FORBIDDEN_INFO_KEYS)
            assert set(info).isdisjoint(EVALUATOR_ONLY_FIELD_NAMES)


@pytest.mark.parametrize("mode", ["conventional", "graph_features"])
def test_serialized_infos_and_observations_hide_evaluator_fields(
    make_env, mode, tmp_path
):
    env = make_env(mode=mode).env
    records: list[dict] = []
    for observations, infos in _collect(env, seed=4, scenario_id="scn-rl-env-obj"):
        for info in infos.values():
            records.append(info)
        for observation in observations.values():
            records.append(
                {
                    "action_mask": observation["action_mask"].tolist(),
                    "obs": observation["obs"].tolist(),
                }
            )

    stream = tmp_path / "env_boundary_stream.jsonl"
    stream.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    # The canonical X-10 scanner finds no evaluator-only field name anywhere in
    # the infos or observations the policy receives.
    assert check_no_evaluator_fields([stream]) == []


def test_reward_breakdown_is_available_to_the_harness_but_not_in_info(make_env):
    env = make_env().env
    env.reset(seed=4, options={"scenario_id": "scn-rl-env-trunc"})
    _obs, _rew, _term, _trunc, infos = env.step({agent: 0 for agent in env.agents})

    # The per-step reward score is reachable by the training harness ...
    assert env.last_reward_breakdown is not None
    # ... but it never appears in any agent's info dictionary.
    for info in infos.values():
        assert "reward" not in info
        assert set(info) == DOCUMENTED_INFO_KEYS
