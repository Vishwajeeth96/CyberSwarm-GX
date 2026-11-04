"""Rule-based conventional baseline policy (stage S4, task 8.1).

:class:`RuleBasedPolicy` is a deterministic, non-learned decision policy that
implements the :class:`~cyberswarm.interfaces.Policy` Protocol. It consumes
*conventional* agent-visible inputs only - the :class:`AgentObservation`, the
:class:`DetectionSummary` and the :class:`~cyberswarm.interfaces.EvidenceView`
assembled into the :class:`~cyberswarm.interfaces.DecisionContext`. It never
reads ``ctx.graph_snapshot`` or ``ctx.graph_features`` (that is the job of the
graph-feature baseline, task 8.2), so its proposal is invariant to any
perturbation of the graph channels (design Property 16).

The escalation ladder (RL_IMPLEMENTATION section 5 / DESIGN section 4) chooses,
for one agent, a single action per step over the agent's own-zone nodes:

* When a node is alerted but its evidence is insufficient (absent or weak), the
  policy *investigates* (``investigate_node``) or asks for deeper evidence
  (``request_evidence``) before taking any containment action. It never
  isolates a node on null or weak evidence.
* As evidence and alert severity warrant, containment escalates up the rungs
  ``rate_limit_node`` -> ``quarantine_node`` -> ``isolate_node``: moderate
  evidence (``>= e_min``) earns the light rung, strong evidence (``>= e_contain``)
  the heavier ones, and a high-severity alert the top rung.
* A *contained* node is recovered: once its evidence is decisive it is
  ``restore_node``-d; while its evidence is still null, weak or moderate it is
  re-investigated (rule 2f, AD-22) so a quarantined compromised node whose
  evidence expired is looked at again within ``investigate_cooldown + 1`` steps
  and the episode can reach ``incident_resolved`` instead of stalling.
* When nothing warrants action the policy proposes ``no_op``.

Every proposal is a contract-valid :class:`PolicyProposal` whose action is taken
*only* from ``ctx.available_actions`` (the policy mirrors the available action's
parameters), carries explanatory ``rationale_codes`` and a fixed-template
``rationale_text`` into which only node ids are interpolated, and is a pure
function of the context (identical contexts yield identical proposals).

Per-node suspicion: the rule baseline's suspicion *is* the detection role's
suspicion (``ctx.detection``), which the episode engine already streams as the
detection :class:`~cyberswarm.contracts.NodeScoreRecord` (scorer kind
``detection_rule``). The :class:`~cyberswarm.interfaces.Policy` Protocol returns
only a proposal, so this baseline does not re-emit that score and never
duplicates the engine's record (task 8.1 note; AD-25).

Imports stay within the baselines' boundary (DESIGN section 3.2): ``contracts``,
``interfaces``, ``common`` and ``roles`` only - never ``graph``, ``rl``,
``marl`` or ``mock``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..contracts import (
    ActionKind,
    ActionParameters,
    ActionSpec,
    AvailableActions,
    DetectionSummary,
    EpisodeId,
    NodeStatus,
    PolicyId,
    PolicyProposal,
    RationaleCode,
    Representation,
)
from ..interfaces import DecisionContext, NodeEvidence
from ..roles import EvidenceConfig

__all__ = [
    "DEFAULT_POLICY_ID",
    "DEFAULT_E_CONTAIN",
    "DEFAULT_INVESTIGATE_COOLDOWN",
    "DEFAULT_HIGH_SEVERITY",
    "RuleBasedPolicy",
]

#: Default policy id (CONTRACTS section 4 example; ``^pol-<slug>$``).
DEFAULT_POLICY_ID: PolicyId = "pol-rule-conv-v1"

#: Evidence level that counts as *strong* - enough to take a high-impact
#: containment action or to confirm a compromise before restoring. The evidence
#: domain is ``0..3``; ``e_min`` (moderate, from :class:`EvidenceConfig`) is the
#: lower, weak/sufficient threshold shared with the uncertainty rule.
DEFAULT_E_CONTAIN: int = 3

#: A contained node is re-investigated (rule 2f) with ``investigate_node`` when
#: it was last investigated at least this many steps ago (or never), otherwise
#: with the deeper ``request_evidence`` (RL_IMPLEMENTATION section 7.2.1).
DEFAULT_INVESTIGATE_COOLDOWN: int = 2

#: ``max_alert_severity`` (0..3) at or above which a strong-evidence containment
#: reaches for the top ``isolate_node`` rung rather than ``quarantine_node``.
DEFAULT_HIGH_SEVERITY: int = 3

# Priority tiers of a per-node candidate action (higher acts first). Stopping an
# active, evidenced compromise outranks gathering evidence, which outranks
# recovering an already-contained node; recovery therefore runs once the active
# incident is handled, which is exactly when the episode can resolve.
_TIER_CONTAIN_STRONG = 5
_TIER_CONTAIN_MODERATE = 4
_TIER_INVESTIGATE_ACTIVE = 3
_TIER_RESTORE = 2
_TIER_REINVESTIGATE = 1

#: Node statuses in which a node counts as contained (CONTRACTS section 9);
#: ``restoring`` is already recovering and accepts none of these actions, so the
#: recovery ladder targets ``quarantined`` and ``isolated`` nodes.
_CONTAINED = frozenset({NodeStatus.quarantined, NodeStatus.isolated})

#: Fixed verb per action kind for the rationale text (no free text is
#: interpolated - only the node id is).
_VERB: Mapping[ActionKind, str] = {
    ActionKind.investigate_node: "investigate",
    ActionKind.request_evidence: "request evidence for",
    ActionKind.rate_limit_node: "rate limit",
    ActionKind.quarantine_node: "quarantine",
    ActionKind.isolate_node: "isolate",
    ActionKind.restore_node: "restore",
}


@dataclass(frozen=True)
class _Candidate:
    """One node's desired action, with the keys used to pick between nodes."""

    tier: int
    suspicion: float
    criticality: float
    node_id: str
    action: ActionSpec
    rationale: tuple[RationaleCode, ...]
    text: str

    def sort_key(self) -> tuple[int, float, float, str]:
        """Descending ``(tier, suspicion, criticality)`` then ascending id.

        Expressed as a ``min`` key by negating the descending components, so the
        selection is total and deterministic (ties break on the lowest node id).
        """

        return (-self.tier, -self.suspicion, -self.criticality, self.node_id)


class RuleBasedPolicy:
    """Deterministic rule-based baseline over conventional inputs (task 8.1).

    Construct with an optional ``policy_id`` and the evidence thresholds. The
    policy holds no episode state that affects its decision; ``reset`` records
    the episode id and seed only so the object satisfies the Protocol and reads
    identically across runs.
    """

    representation: Representation = Representation.conventional

    def __init__(
        self,
        *,
        policy_id: PolicyId = DEFAULT_POLICY_ID,
        evidence: EvidenceConfig | None = None,
        e_contain: int = DEFAULT_E_CONTAIN,
        investigate_cooldown: int = DEFAULT_INVESTIGATE_COOLDOWN,
        high_severity: int = DEFAULT_HIGH_SEVERITY,
    ) -> None:
        self.policy_id: PolicyId = policy_id
        self._evidence = evidence if evidence is not None else EvidenceConfig()
        if not self._evidence.e_min <= e_contain <= 3:
            raise ValueError(
                f"e_contain must be in e_min..3, got {e_contain} (e_min={self._evidence.e_min})"
            )
        if investigate_cooldown < 0:
            raise ValueError("investigate_cooldown must be non-negative")
        self._e_contain = e_contain
        self._investigate_cooldown = investigate_cooldown
        self._high_severity = high_severity
        self._episode_id: EpisodeId | None = None
        self._seed: int | None = None

    @property
    def e_min(self) -> int:
        """The moderate/sufficient evidence threshold (from the shared config)."""

        return self._evidence.e_min

    # ------------------------------------------------------------------ #
    # Policy Protocol
    # ------------------------------------------------------------------ #
    def reset(self, episode_id: EpisodeId, seed: int) -> None:
        """Record the episode id and seed (the policy draws no random numbers)."""

        self._episode_id = episode_id
        self._seed = seed

    def propose(self, ctx: DecisionContext) -> PolicyProposal:
        """Return the contract-valid proposal for this agent at this step.

        Builds one candidate action per own-zone node, picks the
        highest-priority candidate deterministically, and falls back (by
        default to ``no_op``) when no node warrants action.

        Only three things vary between this baseline and the graph-feature one
        (task 8.2): the per-node *suspicion* and *priority* used to rank
        candidates (:meth:`_suspicion_by_node`, :meth:`_priority_by_node`) and
        the no-candidate *fallback* (:meth:`_fallback`). The escalation ladder
        itself - what action a node in a given state warrants, the
        investigate-before-isolate discipline and rule 2f - lives in
        :meth:`_node_candidate` and is shared verbatim, so the two baselines
        differ only in how they read the features, never in the safety rules.
        """

        available = _AvailableIndex(ctx.available_actions)
        suspicion = self._suspicion_by_node(ctx)
        priority = self._priority_by_node(ctx)
        evidence = ctx.evidence.nodes

        candidates: list[_Candidate] = []
        for node in ctx.observation.nodes:
            if node.scope.value != "own":
                continue  # the agent acts only within its own authority
            candidate = self._node_candidate(
                node_id=node.node_id,
                status=node.visible_status,
                severity=node.max_alert_severity,
                evidence=evidence.get(node.node_id),
                suspicion=suspicion.get(node.node_id, 0.0),
                criticality=priority.get(node.node_id, 0.0),
                available=available,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return self._fallback(ctx, available)

        best = min(candidates, key=_Candidate.sort_key)
        return self._build(ctx, best.action, best.rationale, best.text)

    # ------------------------------------------------------------------ #
    # Feature-derivation hooks (the only points a sibling baseline overrides)
    # ------------------------------------------------------------------ #
    def _suspicion_by_node(self, ctx: DecisionContext) -> dict[str, float]:
        """Per-node suspicion used to rank candidate nodes (higher acts first).

        The conventional baseline uses the detection role's suspicion directly;
        the graph baseline blends in neighbour alert aggregation.
        """

        return {node.node_id: node.suspicion for node in ctx.detection.nodes}

    def _priority_by_node(self, ctx: DecisionContext) -> dict[str, float]:
        """Per-node priority that breaks suspicion ties (higher acts first).

        The conventional baseline uses the node's own topology criticality; the
        graph baseline uses its exposure of critical dependents.
        """

        return {node.node_id: node.criticality for node in ctx.topology.nodes}

    def _fallback(
        self, ctx: DecisionContext, available: "_AvailableIndex"
    ) -> PolicyProposal:
        """Proposal when no own node warrants a ladder action (default ``no_op``)."""

        return self._build(
            ctx,
            available.spec(ActionKind.no_op, None),
            (RationaleCode.no_action_needed,),
            "no action warranted",
        )

    # ------------------------------------------------------------------ #
    # Per-node ladder
    # ------------------------------------------------------------------ #
    def _node_candidate(
        self,
        *,
        node_id: str,
        status: NodeStatus,
        severity: int,
        evidence: NodeEvidence | None,
        suspicion: float,
        criticality: float,
        available: "_AvailableIndex",
    ) -> _Candidate | None:
        """Return the single desired action for one own node, or ``None``.

        ``evidence`` is the node's :class:`~cyberswarm.interfaces.NodeEvidence`
        from the evidence view (always present for an own node); a defensive
        ``None`` is treated as no evidence and not alerted.
        """

        level = evidence.evidence_level if evidence is not None else None
        steps_since = evidence.steps_since_investigated if evidence is not None else None
        alerted = evidence.alerted if evidence is not None else False

        def make(tier: int, kind: ActionKind, rationale, phrase: str) -> _Candidate:
            return _Candidate(
                tier=tier,
                suspicion=suspicion,
                criticality=criticality,
                node_id=node_id,
                action=available.spec(kind, node_id),
                rationale=tuple(rationale),
                text=f"{_VERB[kind]} {node_id} {phrase}",
            )

        # -- Recovery of a contained node (rule 2f + restore). ------------- #
        if status in _CONTAINED:
            if level is not None and level >= self._e_contain:
                # Compromise confirmed; clear it from containment.
                if available.has(ActionKind.restore_node, node_id):
                    return make(
                        _TIER_RESTORE,
                        ActionKind.restore_node,
                        (
                            RationaleCode.evidence_strong,
                            RationaleCode.recovery_after_containment,
                            RationaleCode.containment_ladder,
                        ),
                        "after containment",
                    )
            elif level == 0:
                # Investigated clean; undo the containment so the node returns up.
                if available.has(ActionKind.restore_node, node_id):
                    return make(
                        _TIER_RESTORE,
                        ActionKind.restore_node,
                        (
                            RationaleCode.no_evidence,
                            RationaleCode.recovery_after_containment,
                        ),
                        "after containment",
                    )
            # Evidence null, weak or moderate (or restore unavailable): rule 2f.
            kind = self._gather_kind(node_id, steps_since, available)
            if kind is not None:
                return make(
                    _TIER_REINVESTIGATE,
                    kind,
                    (
                        RationaleCode.evidence_insufficient,
                        RationaleCode.recovery_after_containment,
                    ),
                    "after containment",
                )
            return None

        # -- Active node (up or rate_limited): investigate, then escalate. - #
        if level is not None and level >= self._e_contain:
            # Strong evidence: heavier containment, top rung on a severe alert.
            if severity >= self._high_severity:
                order = (
                    ActionKind.isolate_node,
                    ActionKind.quarantine_node,
                    ActionKind.rate_limit_node,
                )
            else:
                order = (
                    ActionKind.quarantine_node,
                    ActionKind.isolate_node,
                    ActionKind.rate_limit_node,
                )
            kind = available.first(order, node_id)
            if kind is not None:
                return make(
                    _TIER_CONTAIN_STRONG,
                    kind,
                    (RationaleCode.evidence_strong, RationaleCode.containment_ladder),
                    "on strong evidence",
                )
            return None

        if level is not None and level >= self.e_min:
            # Moderate evidence: the lightest rung the node has not yet reached.
            if status is NodeStatus.up:
                order = (ActionKind.rate_limit_node, ActionKind.quarantine_node)
            else:  # already rate_limited: escalate one rung
                order = (ActionKind.quarantine_node, ActionKind.rate_limit_node)
            kind = available.first(order, node_id)
            if kind is not None:
                return make(
                    _TIER_CONTAIN_MODERATE,
                    kind,
                    (RationaleCode.evidence_moderate, RationaleCode.containment_ladder),
                    "on moderate evidence",
                )
            return None

        # Insufficient evidence (null or weak level 1): gather before containing,
        # but only when the node is alerted. A level-0 (clean) reading is left
        # alone, so a false alert does not drive endless investigation.
        if alerted and (level is None or level == 1):
            kind = self._gather_kind(node_id, steps_since, available)
            if kind is not None:
                return make(
                    _TIER_INVESTIGATE_ACTIVE,
                    kind,
                    (RationaleCode.alerts_present, RationaleCode.evidence_insufficient),
                    "to gather evidence",
                )
        return None

    def _gather_kind(
        self, node_id: str, steps_since: int | None, available: "_AvailableIndex"
    ) -> ActionKind | None:
        """Choose the evidence action (RL_IMPLEMENTATION 7.2.1), honouring availability.

        A cheap ``investigate_node`` when the node was last looked at long ago
        (or never), otherwise the deeper ``request_evidence``; fall back to the
        other if the first is not available, and give up if neither is.
        """

        if steps_since is None or steps_since >= self._investigate_cooldown:
            order = (ActionKind.investigate_node, ActionKind.request_evidence)
        else:
            order = (ActionKind.request_evidence, ActionKind.investigate_node)
        return available.first(order, node_id)

    # ------------------------------------------------------------------ #
    # Proposal construction
    # ------------------------------------------------------------------ #
    def _build(
        self,
        ctx: DecisionContext,
        action: ActionSpec,
        rationale: tuple[RationaleCode, ...],
        text: str,
    ) -> PolicyProposal:
        """Wrap a chosen action in a contract-valid :class:`PolicyProposal`."""

        return PolicyProposal(
            proposal_id=ctx.next_proposal_id,
            episode_id=ctx.episode_id,
            step=ctx.step,
            agent_id=ctx.agent_id,
            policy_id=self.policy_id,
            observation_id=ctx.observation.observation_id,
            action=action,
            rationale_codes=list(rationale),
            rationale_text=text,
        )


class _AvailableIndex:
    """Fast lookup over an :class:`AvailableActions` menu.

    Keys are ``(action_kind, target_node_id)`` pairs (``no_op`` keys on
    ``None``). :meth:`spec` rebuilds an :class:`ActionSpec` that mirrors the
    menu's parameters exactly, so a selected action is always one the simulator
    offered this step.
    """

    def __init__(self, available: AvailableActions) -> None:
        self._params: dict[tuple[ActionKind, str | None], ActionParameters] = {
            (action.action_kind, action.target_node_id): action.parameters
            for action in available.actions
        }

    def has(self, kind: ActionKind, node_id: str | None) -> bool:
        """Whether ``(kind, node_id)`` is in the available-action menu."""

        return (kind, node_id) in self._params

    def first(self, kinds: tuple[ActionKind, ...], node_id: str) -> ActionKind | None:
        """Return the first of ``kinds`` available for ``node_id``, else ``None``."""

        for kind in kinds:
            if (kind, node_id) in self._params:
                return kind
        return None

    def spec(self, kind: ActionKind, node_id: str | None) -> ActionSpec:
        """Build the :class:`ActionSpec` for an available ``(kind, node_id)``."""

        return ActionSpec(
            action_kind=kind,
            target_node_id=node_id,
            parameters=self._params[(kind, node_id)],
        )
