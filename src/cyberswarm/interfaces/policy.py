"""Decision-policy interface (stage S2, DESIGN section 4.2).

A :class:`Policy` turns a :class:`DecisionContext` (everything one agent may see
at a step, already assembled from agent-visible records) into a
:class:`~cyberswarm.contracts.PolicyProposal`. A :class:`DistributionPolicy`
additionally exposes the action distribution it sampled, which the
uncertainty-aware wrapper and the matched control consume.

The context carries *only* agent-visible data: the observation, the available
actions, the detection summary, the evidence view and the optional graph
snapshot/features. It never carries ground truth. This module imports only
:mod:`cyberswarm.contracts` and the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..contracts import (
    ActionSpec,
    AgentId,
    AgentObservation,
    AvailableActions,
    DetectionSummary,
    EpisodeId,
    GraphFeatureRecord,
    GraphSnapshot,
    NodeId,
    PolicyId,
    PolicyProposal,
    ProposalId,
    RationaleCode,
    Representation,
    Topology,
    UncertaintyInfo,
)

__all__ = [
    "NodeEvidence",
    "EvidenceView",
    "DecisionContext",
    "Policy",
    "PolicyDecision",
    "DistributionPolicy",
]


@dataclass(frozen=True)
class NodeEvidence:
    """The evidence state of one node as a policy sees it.

    ``insufficient`` is ``True`` when the node is alerted and the evidence is
    absent or weak (``evidence_level`` is ``None`` or below the evidence
    threshold); it is the signal the uncertainty rule and the baselines use to
    decide whether to gather more evidence.
    """

    evidence_level: int | None
    steps_since_investigated: int | None
    alerted: bool
    insufficient: bool


@dataclass(frozen=True)
class EvidenceView:
    """Per-node evidence state for one agent's own nodes."""

    nodes: Mapping[NodeId, NodeEvidence]


@dataclass(frozen=True)
class DecisionContext:
    """Everything one agent may use to choose an action this step.

    Assembled by the engine from agent-visible records only. ``graph_snapshot``
    and ``graph_features`` are ``None`` for a conventional-only run.
    """

    episode_id: EpisodeId
    step: int
    max_steps: int
    agent_id: AgentId
    topology: Topology
    observation: AgentObservation
    available_actions: AvailableActions
    detection: DetectionSummary
    evidence: EvidenceView
    graph_snapshot: GraphSnapshot | None
    graph_features: GraphFeatureRecord | None
    next_proposal_id: ProposalId


class Policy(Protocol):
    """A decision policy: one proposal per agent per step.

    ``reset`` is called at the start of each episode with the episode id and the
    policy seed. ``propose`` returns a contract-valid
    :class:`~cyberswarm.contracts.PolicyProposal` whose action is present in the
    context's ``available_actions``.
    """

    policy_id: PolicyId
    representation: Representation

    def reset(self, episode_id: EpisodeId, seed: int) -> None: ...

    def propose(self, ctx: DecisionContext) -> PolicyProposal: ...


@dataclass(frozen=True)
class PolicyDecision:
    """The action a :class:`DistributionPolicy` chose and the distribution.

    ``probs`` is aligned with ``valid_actions`` and sums to ``1.0`` in float64.
    The uncertainty signal is computed from this distribution.
    """

    chosen: ActionSpec
    valid_actions: tuple[ActionSpec, ...]
    probs: tuple[float, ...]


class DistributionPolicy(Policy, Protocol):
    """A policy that exposes its action distribution.

    ``decide`` returns the sampled :class:`PolicyDecision`; ``build_proposal``
    turns a decision (possibly after the uncertainty rule substituted the
    action) into the final proposal, attaching the rationale codes and the
    optional uncertainty signal.
    """

    def decide(self, ctx: DecisionContext) -> PolicyDecision: ...

    def build_proposal(
        self,
        ctx: DecisionContext,
        decision: PolicyDecision,
        *,
        action: ActionSpec,
        rationale_codes: Sequence[RationaleCode],
        uncertainty: UncertaintyInfo | None,
    ) -> PolicyProposal: ...
