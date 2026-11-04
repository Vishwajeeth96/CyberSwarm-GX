"""End-to-end rule-2f behaviour of the conventional baseline (task 8.3; AD-22).

Rule 2f (DECISIONS AD-22) makes the rule baselines re-investigate a *contained*
but still-compromised node whose evidence is null, weak or moderate, so a
quarantined compromised node whose evidence expired is looked at again within
``investigate_cooldown + 1`` steps and the incident can be cleared instead of
sitting in quarantine until the horizon.

These tests drive the real :class:`~cyberswarm.baselines.rule.RuleBasedPolicy`
through the real :class:`~cyberswarm.engine.EpisodeEngine` over the
:class:`~cyberswarm.mock.MockSimulator` (with an accept-all coordinator double),
on a single-incident DMZ scenario with no spread. They assert that:

* the contained, still-compromised node is re-investigated within
  ``investigate_cooldown + 1`` steps while it stays contained;
* the episode reaches ``incident_resolved`` rather than truncating at
  ``max_steps``; and
* removing rule 2f (the :class:`NoRule2fPolicy` counterfactual) makes the very
  same episode stall in quarantine until ``max_steps`` - which is the handicap
  AD-22 exists to remove.
"""

from __future__ import annotations

from cyberswarm.baselines.rule import DEFAULT_INVESTIGATE_COOLDOWN, RuleBasedPolicy
from cyberswarm.contracts import ActionKind, NodeStatus, TerminationReason

from .conftest import RULE2F_SCENARIO

_CONTAINED = {NodeStatus.quarantined, NodeStatus.isolated}
_GATHER = {ActionKind.investigate_node, ActionKind.request_evidence}


def _nondecisive(evidence_level: int | None) -> bool:
    """Evidence that neither confirms (>= e_contain) nor clears (== 0) the node."""

    return evidence_level is None or (evidence_level != 0 and evidence_level < 3)


class NoRule2fPolicy(RuleBasedPolicy):
    """:class:`RuleBasedPolicy` with the contained-node re-investigation removed.

    Overrides the per-node ladder to drop *only* the rule-2f branch: a contained
    node is never handed a gather action, so once a compromised node is
    quarantined on non-decisive evidence it can never be refreshed to the strong
    evidence a ``restore_node`` needs. Everything else (investigate-before-
    contain, the containment rungs, restore on strong evidence) is unchanged.
    """

    def _node_candidate(
        self,
        *,
        node_id,
        status,
        severity,
        evidence,
        suspicion,
        criticality,
        available,
    ):
        candidate = super()._node_candidate(
            node_id=node_id,
            status=status,
            severity=severity,
            evidence=evidence,
            suspicion=suspicion,
            criticality=criticality,
            available=available,
        )
        if (
            candidate is not None
            and status in _CONTAINED
            and candidate.action.action_kind in _GATHER
        ):
            return None
        return candidate


def test_contained_compromised_node_is_reinvestigated_within_cooldown_plus_one(
    rule2f_runner,
) -> None:
    trace = rule2f_runner(RuleBasedPolicy())

    contained_weak = [
        row
        for row in trace.trajectory
        if row.visible_status in _CONTAINED
        and row.truly_compromised
        and _nondecisive(row.evidence_level)
    ]
    assert contained_weak, (
        "scenario did not produce a quarantined, still-compromised node with "
        "non-decisive evidence (the rule-2f precondition)"
    )

    first_step = contained_weak[0].step
    window_end = first_step + DEFAULT_INVESTIGATE_COOLDOWN + 1
    reinvestigations = [
        row
        for row in trace.trajectory
        if first_step <= row.step <= window_end
        and row.visible_status in _CONTAINED
        and row.truly_compromised
        and row.action in _GATHER
    ]
    assert reinvestigations, (
        "rule 2f did not re-investigate the contained compromised node within "
        f"investigate_cooldown + 1 (= {DEFAULT_INVESTIGATE_COOLDOWN + 1}) steps"
    )


def test_episode_reaches_incident_resolved_not_truncation(rule2f_runner) -> None:
    trace = rule2f_runner(RuleBasedPolicy())
    assert trace.termination_reason == TerminationReason.incident_resolved.value
    assert trace.termination_reason != TerminationReason.max_steps.value
    assert trace.steps < RULE2F_SCENARIO.max_steps


def test_without_rule_2f_the_same_episode_stalls_to_truncation(rule2f_runner) -> None:
    trace = rule2f_runner(NoRule2fPolicy())
    # The node is quarantined while compromised but never re-investigated, so it
    # is never restored and the incident never clears: the episode truncates.
    assert trace.termination_reason == TerminationReason.max_steps.value
    assert any(
        row.visible_status in _CONTAINED and row.truly_compromised
        for row in trace.trajectory
    )


def test_driven_episode_is_deterministic(rule2f_runner) -> None:
    assert rule2f_runner(RuleBasedPolicy()) == rule2f_runner(RuleBasedPolicy())
