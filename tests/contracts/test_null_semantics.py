"""Unavailable optional values stay null, never a filled default (DC-03).

An optional uncertainty signal, learned-model output, episode reward or
comparison statistic that no producer computed must serialize as ``null`` and
survive a round trip as ``None`` (never ``0.0`` and never a silent default).
"""

from __future__ import annotations

import json
from pathlib import Path

from cyberswarm.contracts import (
    ActionKind,
    ActionSpec,
    EpisodeResult,
    GraphModelOutput,
    PolicyProposal,
    RuleOutcome,
    UncertaintyInfo,
    dumps_canonical,
    loads_record,
    read_json,
)
from cyberswarm.contracts.results import MetricDifference, MetricSummary

REPO_ROOT = Path(__file__).resolve().parents[2]
_MINI = REPO_ROOT / "fixtures" / "mini_episode"


def test_proposal_without_uncertainty_round_trips_null() -> None:
    proposal = PolicyProposal(
        proposal_id="prop-000001",
        episode_id="ep-rule-conv-v1-mock-lateral-eval-01-s2001",
        step=5,
        agent_id="agent-dmz",
        policy_id="pol-rule-conv-v1",
        observation_id="obs-0005-agent-dmz",
        action=ActionSpec(
            action_kind=ActionKind.investigate_node, target_node_id="node-web-02"
        ),
    )
    assert proposal.uncertainty is None
    assert proposal.action_probability is None

    text = dumps_canonical(proposal)
    assert '"uncertainty":null' in text
    assert '"action_probability":null' in text

    restored = loads_record("PolicyProposal", text)
    assert restored.uncertainty is None
    assert restored.action_probability is None


def test_uncertainty_info_null_value_round_trips() -> None:
    info = UncertaintyInfo(rule_outcome=RuleOutcome.not_applied)
    assert info.value is None
    assert info.rule_id is None
    assert info.original_action is None

    text = dumps_canonical(info)
    assert '"value":null' in text
    assert '"value":0.0' not in text

    restored = UncertaintyInfo.model_validate(json.loads(text))
    assert restored == info
    assert restored.value is None


def test_graph_model_output_not_computed_round_trips_null() -> None:
    output = GraphModelOutput(
        graph_model_output_id="gmo-0005-agent-dmz",
        episode_id="ep-rule-conv-v1-mock-lateral-eval-01-s2001",
        step=5,
        agent_id="agent-dmz",
        snapshot_id="gsn-0005-agent-dmz",
        status="not_computed",
        model_id=None,
        output_kind=None,
        node_order=None,
        values=None,
    )
    text = dumps_canonical(output)
    for field in ("model_id", "output_kind", "node_order", "values"):
        assert f'"{field}":null' in text

    restored = loads_record("GraphModelOutput", text)
    assert restored.model_id is None
    assert restored.output_kind is None
    assert restored.node_order is None
    assert restored.values is None


def test_metric_summary_undefined_stays_null_not_zero() -> None:
    summary = MetricSummary(name="attack impact", n=0, n_undefined=3)
    text = dumps_canonical(summary)
    for field in ("mean", "std", "ci_low", "ci_high"):
        assert f'"{field}":null' in text
        assert f'"{field}":0.0' not in text
    restored = MetricSummary.model_validate(json.loads(text))
    assert restored.mean is None and restored.std is None


def test_metric_difference_undefined_stays_null() -> None:
    diff = MetricDifference(name="episode reward", n_pairs=4)
    text = dumps_canonical(diff)
    assert '"mean_difference":null' in text
    restored = MetricDifference.model_validate(json.loads(text))
    assert restored.mean_difference is None
    assert restored.ci_low is None and restored.ci_high is None


def test_episode_result_without_reward_round_trips_null() -> None:
    # The committed mini-episode result was produced by a policy that ran no
    # reward model and no uncertainty policy.
    result = read_json(_MINI / "episode_result.json", EpisodeResult)
    assert result.reward is None
    assert result.uncertainty_summary is None

    text = dumps_canonical(result)
    assert '"reward":null' in text
    assert '"uncertainty_summary":null' in text

    restored = loads_record("EpisodeResult", text)
    assert restored.reward is None
    assert restored.uncertainty_summary is None
