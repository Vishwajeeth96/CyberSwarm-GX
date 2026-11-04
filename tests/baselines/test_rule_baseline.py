"""Rule-based conventional baseline tests (task 8.3; design Property 16).

Covers, for :class:`~cyberswarm.baselines.rule.RuleBasedPolicy`:

* every proposal is contract-valid and its action is one the simulator offered
  this step (Property 16, conventional arm);
* an alerted node with insufficient or null evidence is *investigated*, never
  isolated - the investigate-before-contain discipline, which never isolates on
  null evidence even under a high-severity alert;
* across a real episode, an isolate is only ever proposed on decisive evidence
  (so null evidence never drives an isolation);
* the policy is deterministic (same context -> same proposal);
* the proposal is invariant to any perturbation of the graph channels, which the
  conventional baseline does not read.
"""

from __future__ import annotations

import pytest

from cyberswarm.baselines.rule import DEFAULT_E_CONTAIN, RuleBasedPolicy
from cyberswarm.contracts import ActionKind, NodeScope, NodeStatus, RationaleCode

from .conftest import (
    action_in_available,
    build_context,
    make_node,
    perturb_features,
    revalidate,
    with_graph_features,
)

_GATHER = {ActionKind.investigate_node, ActionKind.request_evidence}
_CONTAIN = {
    ActionKind.rate_limit_node,
    ActionKind.quarantine_node,
    ActionKind.isolate_node,
}


def _entry(ctx, node_id: str):
    return next(node for node in ctx.observation.nodes if node.node_id == node_id)


# --------------------------------------------------------------------------- #
# Property 16: contract-valid proposal whose action is in the menu
# --------------------------------------------------------------------------- #
def test_every_proposal_is_contract_valid_and_in_the_menu(mock_contexts) -> None:
    policy = RuleBasedPolicy()
    assert mock_contexts  # the episode produced contexts to check
    for ctx in mock_contexts:
        proposal = policy.propose(ctx)
        # Contract-valid: survives a JSON round trip through every validator.
        assert revalidate(proposal) == proposal
        assert proposal.agent_id == ctx.agent_id
        assert proposal.step == ctx.step
        # The chosen action (kind, target and parameters) is one offered now.
        assert action_in_available(proposal.action, ctx.available_actions)


@pytest.mark.parametrize(
    "nodes",
    [
        [make_node("node-web-02", alert_count=3, severity=3)],
        [
            make_node("node-gw-01"),
            make_node("node-web-01", status=NodeStatus.rate_limited),
            make_node("node-web-02", alert_count=5, severity=2, evidence_level=3),
        ],
    ],
)
def test_proposal_is_valid_and_in_menu_for_synthetic_contexts(nodes) -> None:
    ctx = build_context(nodes)
    proposal = RuleBasedPolicy().propose(ctx)
    assert revalidate(proposal) == proposal
    assert action_in_available(proposal.action, ctx.available_actions)


# --------------------------------------------------------------------------- #
# Investigate, never isolate, on insufficient / null evidence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("severity", [1, 3])
@pytest.mark.parametrize("evidence_level", [None, 1])
def test_alerted_node_with_insufficient_evidence_is_investigated_not_contained(
    severity, evidence_level
) -> None:
    ctx = build_context(
        [
            make_node("node-gw-01"),
            make_node("node-web-01"),
            make_node(
                "node-web-02",
                alert_count=4,
                severity=severity,
                evidence_level=evidence_level,
            ),
        ]
    )
    proposal = RuleBasedPolicy().propose(ctx)
    action = proposal.action
    # Gather evidence, never contain, even when the alert severity is high.
    assert action.action_kind in _GATHER
    assert action.action_kind not in _CONTAIN
    assert action.target_node_id == "node-web-02"
    assert RationaleCode.evidence_insufficient in proposal.rationale_codes


def test_null_evidence_never_isolates_even_at_max_severity() -> None:
    ctx = build_context(
        [make_node("node-web-02", alert_count=5, severity=3, evidence_level=None)]
    )
    proposal = RuleBasedPolicy().propose(ctx)
    assert proposal.action.action_kind is not ActionKind.isolate_node
    assert proposal.action.action_kind in _GATHER


def test_isolate_is_only_ever_proposed_on_decisive_evidence(mock_contexts) -> None:
    """Across a real episode, no isolate is proposed while evidence is null."""

    policy = RuleBasedPolicy()
    saw_isolate = False
    for ctx in mock_contexts:
        action = policy.propose(ctx).action
        if action.action_kind is ActionKind.isolate_node:
            saw_isolate = True
            target = _entry(ctx, action.target_node_id)
            assert target.evidence_level is not None
            assert target.evidence_level >= DEFAULT_E_CONTAIN
    assert saw_isolate  # the rule-2f episode does reach an isolate at least once


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_same_context_yields_identical_proposals(mock_contexts) -> None:
    # Two independent instances prove the decision is a pure function of ctx.
    first, second = RuleBasedPolicy(), RuleBasedPolicy()
    for ctx in mock_contexts:
        assert first.propose(ctx) == second.propose(ctx)


def test_repeated_calls_are_stable() -> None:
    ctx = build_context(
        [make_node("node-web-02", alert_count=2, severity=2, evidence_level=3)]
    )
    policy = RuleBasedPolicy()
    proposals = [policy.propose(ctx) for _ in range(5)]
    assert all(proposal == proposals[0] for proposal in proposals)


# --------------------------------------------------------------------------- #
# Invariance to graph-feature perturbation (the conventional arm ignores them)
# --------------------------------------------------------------------------- #
def test_proposal_is_invariant_to_graph_feature_perturbation(mock_contexts) -> None:
    policy = RuleBasedPolicy()
    checked_with_features = False
    for ctx in mock_contexts:
        if ctx.graph_features is None:
            continue
        checked_with_features = True
        baseline = policy.propose(ctx)
        # Dropping the features entirely must not change the conventional choice.
        assert policy.propose(with_graph_features(ctx, None)) == baseline
        # Nor must perturbing every graph-feature value.
        perturbed = with_graph_features(ctx, perturb_features(ctx.graph_features))
        assert policy.propose(perturbed) == baseline
    assert checked_with_features


def test_perturbing_graph_features_does_not_change_a_quiet_decision() -> None:
    # Quiet own nodes with an alerted out-of-zone neighbour: the conventional
    # baseline returns no_op regardless of what the graph channel says.
    ctx = build_context(
        [
            make_node("node-app-01", scope=NodeScope.boundary, alert_count=3, severity=2),
            make_node("node-gw-01"),
            make_node("node-web-01"),
            make_node("node-web-02"),
        ]
    )
    policy = RuleBasedPolicy()
    baseline = policy.propose(ctx)
    assert baseline.action.action_kind is ActionKind.no_op
    perturbed = with_graph_features(ctx, perturb_features(ctx.graph_features))
    assert policy.propose(perturbed) == baseline
    assert policy.propose(with_graph_features(ctx, None)) == baseline
