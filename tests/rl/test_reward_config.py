"""Reward configuration: defaults, validation, and linearity in the weights.

Covers loading the committed ``reward_default.toml``, the non-negative-weight
rule, rejection of unknown keys, the ``StepScorer`` protocol shape of
:class:`RewardModel`, and the reweight identity that design Property 13 relies
on (the reward is a linear function of the weight vector).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cyberswarm.contracts.common import ContractError, NodeStatus
from cyberswarm.contracts.results import RewardTermTotals
from cyberswarm.interfaces.ports import StepScore
from cyberswarm.rl.reward import (
    TERM_NAMES,
    RewardConfig,
    RewardModel,
    RewardWeights,
    signed_weighted,
    weighted_total,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REWARD_DEFAULT_TOML = REPO_ROOT / "configs" / "experiments" / "reward_default.toml"

_VALID_WEIGHTS = {
    "attack_impact": 1.0,
    "response_delay": 0.5,
    "containment": 3.0,
    "false_isolation": 2.0,
    "healthy_disruption": 0.5,
    "investigation_cost": 0.05,
    "invalid_action": 0.1,
}


def _toml_text(*, extra: str = "", weights: dict[str, float] | None = None) -> str:
    weights = weights or _VALID_WEIGHTS
    lines = [
        'weights_id = "reward-default-v1"',
        'sharing = "team"',
        "d_max = 20",
        "false_isolation_base = 0.5",
    ]
    if extra:
        lines.append(extra)
    lines.append("[weights]")
    lines.extend(f"{name} = {value}" for name, value in weights.items())
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Defaults from the committed file
# --------------------------------------------------------------------------- #
def test_default_file_has_the_documented_values(default_config):
    assert default_config.weights_id == "reward-default-v1"
    assert default_config.sharing == "team"
    assert default_config.d_max == 20
    assert default_config.false_isolation_base == pytest.approx(0.5)

    weights = default_config.weights
    assert weights.attack_impact == pytest.approx(1.0)
    assert weights.response_delay == pytest.approx(0.5)
    assert weights.containment == pytest.approx(3.0)
    assert weights.false_isolation == pytest.approx(2.0)
    assert weights.healthy_disruption == pytest.approx(0.5)
    assert weights.investigation_cost == pytest.approx(0.05)
    assert weights.invalid_action == pytest.approx(0.1)


def test_from_toml_round_trips_the_committed_file():
    config = RewardConfig.from_toml(REWARD_DEFAULT_TOML)
    assert config.weights.model_dump() == _VALID_WEIGHTS


def test_weight_names_match_the_reward_term_totals_contract():
    # The stored magnitudes feed RewardTermTotals, so the names must line up.
    assert set(TERM_NAMES) == set(RewardTermTotals.model_fields)
    assert set(RewardWeights.model_fields) == set(TERM_NAMES)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_negative_weight_is_rejected():
    with pytest.raises(ValidationError):
        RewardWeights(**{**_VALID_WEIGHTS, "attack_impact": -1.0})


def test_non_positive_d_max_is_rejected():
    with pytest.raises(ValidationError):
        RewardConfig(
            weights_id="reward-default-v1",
            sharing="team",
            d_max=0,
            false_isolation_base=0.5,
            weights=RewardWeights(**_VALID_WEIGHTS),
        )


def test_unknown_top_level_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(_toml_text(extra="mystery = 1"), encoding="utf-8")
    with pytest.raises(ContractError) as excinfo:
        RewardConfig.from_toml(bad)
    assert excinfo.value.code == "CONFIG_INVALID"


def test_unknown_weight_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        _toml_text(weights={**_VALID_WEIGHTS, "mystery": 1.0}), encoding="utf-8"
    )
    with pytest.raises(ContractError) as excinfo:
        RewardConfig.from_toml(bad)
    assert excinfo.value.code == "CONFIG_INVALID"


def test_default_sharing_is_team_when_omitted():
    config = RewardConfig(
        weights_id="reward-test-v1",
        d_max=20,
        false_isolation_base=0.5,
        weights=RewardWeights(**_VALID_WEIGHTS),
    )
    assert config.sharing == "team"


# --------------------------------------------------------------------------- #
# StepScorer protocol shape
# --------------------------------------------------------------------------- #
def test_reward_model_scores_as_a_step_scorer(
    default_config, make_node, make_snapshot, make_topology
):
    model = RewardModel(default_config)
    model.reset(make_topology({"node-a": 0.5, "node-b": 1.0}), max_steps=40)
    score = model.score(
        step=0,
        prev=None,
        curr=make_snapshot([make_node("node-a"), make_node("node-b")]),
        executions=[],
        invalid_count=0,
        termination_reason=None,
    )
    assert isinstance(score, StepScore)
    assert set(score.raw) == set(TERM_NAMES)
    assert set(score.weighted) == set(TERM_NAMES)
    assert score.total == pytest.approx(sum(score.weighted.values()))
    # A quiet healthy network scores exactly zero.
    assert score.total == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Reward is linear in the weights (design Property 13 / reweight identity)
# --------------------------------------------------------------------------- #
def test_weighted_total_equals_direct_signed_sum(default_config):
    raw = {
        "containment": 2.0,
        "response_delay": 0.3,
        "attack_impact": 1.5,
        "false_isolation": 0.8,
        "healthy_disruption": 0.35,
        "investigation_cost": 3.0,
        "invalid_action": 1.0,
    }
    weights = default_config.weights
    direct = (
        weights.containment * raw["containment"]
        - weights.response_delay * raw["response_delay"]
        - weights.attack_impact * raw["attack_impact"]
        - weights.false_isolation * raw["false_isolation"]
        - weights.healthy_disruption * raw["healthy_disruption"]
        - weights.investigation_cost * raw["investigation_cost"]
        - weights.invalid_action * raw["invalid_action"]
    )
    assert weighted_total(raw, default_config) == pytest.approx(direct)
    assert sum(signed_weighted(raw, default_config).values()) == pytest.approx(direct)


def test_scaling_all_weights_scales_the_reward(default_config):
    raw = {name: float(index + 1) for index, name in enumerate(TERM_NAMES)}
    scaled = default_config.model_copy(
        update={
            "weights": RewardWeights(
                **{
                    name: 2.0 * getattr(default_config.weights, name)
                    for name in TERM_NAMES
                }
            )
        }
    )
    assert weighted_total(raw, scaled) == pytest.approx(
        2.0 * weighted_total(raw, default_config)
    )


def test_zero_weights_give_zero_reward_for_any_magnitudes():
    config = RewardConfig(
        weights_id="reward-zero-v1",
        d_max=20,
        false_isolation_base=0.5,
        weights=RewardWeights(**dict.fromkeys(TERM_NAMES, 0.0)),
    )
    raw = {name: float(index + 1) for index, name in enumerate(TERM_NAMES)}
    assert weighted_total(raw, config) == pytest.approx(0.0)
