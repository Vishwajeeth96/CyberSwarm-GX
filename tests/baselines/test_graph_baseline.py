"""Non-neural graph-feature baseline tests (task 8.3; design Property 16).

Covers, for :class:`~cyberswarm.baselines.graph_rule.GraphFeaturePolicy`:

* every proposal is contract-valid and its action is in the menu (Property 16,
  graph arm);
* the policy raises :class:`PolicyError` when ``ctx.graph_features`` is ``None``
  - a graph-representation policy cannot act without the features it is defined
  over;
* it actually reads the graph features: on a context whose own nodes are all
  quiet but whose out-of-zone neighbour is alerting, it pre-emptively monitors
  the own node nearest the alerted frontier, where the conventional baseline
  (blind to the graph channel) proposes ``no_op`` - so the two baselines differ;
  and zeroing the neighbour-alert features removes that difference;
* the policy is deterministic (same context -> same proposal).
"""

from __future__ import annotations

import pytest

from cyberswarm.baselines.graph_rule import GraphFeaturePolicy, PolicyError
from cyberswarm.baselines.rule import RuleBasedPolicy
from cyberswarm.contracts import (
    DERIVED_FEATURE_NAMES_V1,
    ActionKind,
    ErrorCode,
    NodeScope,
)

from .conftest import (
    action_in_available,
    build_context,
    make_node,
    revalidate,
    with_graph_features,
)


def _quiet_boundary_context():
    """Quiet own DMZ nodes with an alerted boundary neighbour (node-app-01)."""

    return build_context(
        [
            make_node("node-app-01", scope=NodeScope.boundary, alert_count=3, severity=2),
            make_node("node-gw-01"),
            make_node("node-web-01"),
            make_node("node-web-02"),
        ]
    )


def _without_neighbor_alerts(features):
    """Copy ``features`` with the two neighbour-alert columns set to zero."""

    mean_col = DERIVED_FEATURE_NAMES_V1.index("neighbor_alert_mean")
    max_col = DERIVED_FEATURE_NAMES_V1.index("neighbor_alert_max")
    values = [list(row) for row in features.values]
    for row in values:
        row[mean_col] = 0.0
        row[max_col] = 0.0
    return features.model_copy(update={"values": values})


# --------------------------------------------------------------------------- #
# Property 16: contract-valid proposal whose action is in the menu
# --------------------------------------------------------------------------- #
def test_every_proposal_is_contract_valid_and_in_the_menu(mock_contexts) -> None:
    policy = GraphFeaturePolicy()
    assert mock_contexts
    for ctx in mock_contexts:
        assert ctx.graph_features is not None  # the episode ran in graph mode
        proposal = policy.propose(ctx)
        assert revalidate(proposal) == proposal
        assert proposal.agent_id == ctx.agent_id
        assert action_in_available(proposal.action, ctx.available_actions)


# --------------------------------------------------------------------------- #
# Refuses to run without the features it is defined over
# --------------------------------------------------------------------------- #
def test_raises_policy_error_when_graph_features_absent() -> None:
    ctx = build_context(
        [make_node("node-web-02", alert_count=3, severity=3)], with_graph=False
    )
    assert ctx.graph_features is None
    with pytest.raises(PolicyError) as excinfo:
        GraphFeaturePolicy().propose(ctx)
    assert excinfo.value.code is ErrorCode.POLICY_FAILURE


def test_raises_when_features_stripped_from_an_otherwise_valid_context(
    mock_contexts,
) -> None:
    ctx = next(c for c in mock_contexts if c.graph_features is not None)
    with pytest.raises(PolicyError):
        GraphFeaturePolicy().propose(with_graph_features(ctx, None))


# --------------------------------------------------------------------------- #
# It reads the features: differs from the conventional baseline
# --------------------------------------------------------------------------- #
def test_graph_baseline_differs_from_conventional_on_a_graph_cue() -> None:
    ctx = _quiet_boundary_context()
    conventional = RuleBasedPolicy().propose(ctx)
    graph = GraphFeaturePolicy().propose(ctx)

    # The conventional baseline, blind to the graph channel, stands down.
    assert conventional.action.action_kind is ActionKind.no_op
    # The graph baseline pre-emptively monitors an own node near the alerts.
    assert graph.action.action_kind is ActionKind.monitor_node
    assert graph.action.target_node_id in {"node-gw-01", "node-web-01", "node-web-02"}
    assert action_in_available(graph.action, ctx.available_actions)
    assert graph.action != conventional.action


def test_graph_decision_depends_on_the_neighbor_alert_features() -> None:
    ctx = _quiet_boundary_context()
    policy = GraphFeaturePolicy()
    with_cue = policy.propose(ctx)
    assert with_cue.action.action_kind is ActionKind.monitor_node

    # Remove the graph cue (zero the neighbour-alert columns): the pre-emptive
    # monitor no longer fires, so the graph baseline falls back to no_op.
    without_cue_ctx = with_graph_features(ctx, _without_neighbor_alerts(ctx.graph_features))
    without_cue = policy.propose(without_cue_ctx)
    assert without_cue.action.action_kind is ActionKind.no_op
    assert without_cue.action != with_cue.action


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_same_context_yields_identical_proposals(mock_contexts) -> None:
    first, second = GraphFeaturePolicy(), GraphFeaturePolicy()
    for ctx in mock_contexts:
        assert first.propose(ctx) == second.propose(ctx)


def test_repeated_calls_are_stable() -> None:
    ctx = _quiet_boundary_context()
    policy = GraphFeaturePolicy()
    proposals = [policy.propose(ctx) for _ in range(5)]
    assert all(proposal == proposals[0] for proposal in proposals)
