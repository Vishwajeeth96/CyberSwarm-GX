"""Action and decision-loop records (CONTRACTS section 6 and 8.5 to 8.7).

Contains the action payload types (:class:`ActionParameters`,
:class:`ActionSpec`) and the three decision-loop records that are three
different things with three different ids: :class:`PolicyProposal` (a request),
:class:`ValidationResult` (the coordinator's decision) and
:class:`ExecutedAction` (what the simulator did).

An :class:`ActionSpec` enforces only its own structural rules (target present
per kind, parameters applicable per kind). The coordinator is the sole
validation boundary for state-dependent and authority checks and assigns the
reason codes; it inspects raw input so that it can report, for example,
``TARGET_NOT_ALLOWED`` rather than only ``SCHEMA_INVALID``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .common import (
    ContractModel,
    Decision,
    ErrorCode,
    ExecutionStatus,
    Hex64,
    NodeStatus,
    ReasonCode,
    ShortText,
    UnitFloat,
    reason_codes_in_order,
)
from .ids import (
    AgentId,
    EpisodeId,
    EventId,
    ExecutionId,
    NodeId,
    PolicyId,
    ProposalId,
    ValidationId,
    observation_id,
)
from .vocabulary import ACTION_DEFS, ActionKind


class RateLimitLevel(StrEnum):
    """Allowed ``level`` parameter of ``rate_limit_node``."""

    light = "light"
    heavy = "heavy"


class ActionParameters(ContractModel):
    """Action parameters; ``{}`` means defaults (CONTRACTS section 6)."""

    duration_steps: int | None = Field(default=None, ge=1, le=5)
    level: RateLimitLevel | None = None


def _check_action_params(action_kind: ActionKind, parameters: ActionParameters) -> None:
    """Enforce that each parameter is allowed only for its action kind."""

    allowed = ACTION_DEFS[action_kind].allowed_parameters
    if parameters.duration_steps is not None and "duration_steps" not in allowed:
        raise ValueError(
            f"duration_steps is not allowed for {action_kind.value}"
        )
    if parameters.level is not None and "level" not in allowed:
        raise ValueError(f"level is not allowed for {action_kind.value}")


class ActionSpec(ContractModel):
    """A fully specified abstract action (CONTRACTS section 6)."""

    action_kind: ActionKind
    target_node_id: NodeId | None = None
    parameters: ActionParameters = Field(default_factory=ActionParameters)

    @model_validator(mode="after")
    def _validate_target_and_params(self) -> "ActionSpec":
        needs_target = ACTION_DEFS[self.action_kind].needs_target
        if needs_target and self.target_node_id is None:
            raise ValueError(
                f"{self.action_kind.value} requires a target_node_id"
            )
        if not needs_target and self.target_node_id is not None:
            raise ValueError(
                f"{self.action_kind.value} must not carry a target_node_id"
            )
        _check_action_params(self.action_kind, self.parameters)
        return self


class RationaleCode(StrEnum):
    """Why a policy chose an action (CONTRACTS section 8.5)."""

    no_evidence = "no_evidence"
    alerts_present = "alerts_present"
    evidence_insufficient = "evidence_insufficient"
    evidence_moderate = "evidence_moderate"
    evidence_strong = "evidence_strong"
    containment_ladder = "containment_ladder"
    recovery_after_containment = "recovery_after_containment"
    learned_policy = "learned_policy"
    uncertainty_evidence_request = "uncertainty_evidence_request"
    uncertainty_deferral = "uncertainty_deferral"
    strong_evidence_override = "strong_evidence_override"
    no_action_needed = "no_action_needed"


class RuleOutcome(StrEnum):
    """Outcome of the uncertainty rule (CONTRACTS section 8.5)."""

    not_applied = "not_applied"
    unavailable_pass_through = "unavailable_pass_through"
    pass_through = "pass_through"
    evidence_request = "evidence_request"
    deferred_high_impact = "deferred_high_impact"
    containment_allowed_strong_evidence = "containment_allowed_strong_evidence"


class UncertaintyInfo(ContractModel):
    """The uncertainty signal attached to a proposal, or null when none ran."""

    method: Literal["normalized_entropy"] = "normalized_entropy"
    value: UnitFloat | None = None
    rule_id: ShortText | None = None
    rule_outcome: RuleOutcome
    original_action: ActionSpec | None = None


class PolicyProposal(ContractModel):
    """One agent's requested action for a step (CONTRACTS section 8.5)."""

    schema_version: Literal["1.0"] = "1.0"
    proposal_id: ProposalId
    episode_id: EpisodeId
    step: int = Field(ge=0, le=10000)
    agent_id: AgentId
    role: Literal["decision"] = "decision"
    policy_id: PolicyId
    observation_id: str
    action: ActionSpec
    rationale_codes: list[RationaleCode] = Field(default_factory=list)
    rationale_text: ShortText | None = None
    action_probability: UnitFloat | None = None
    uncertainty: UncertaintyInfo | None = None

    @model_validator(mode="after")
    def _observation_id_matches(self) -> "PolicyProposal":
        expected = observation_id(self.step, self.agent_id)
        if self.observation_id != expected:
            raise ValueError(
                f"observation_id must equal {expected!r}, got {self.observation_id!r}"
            )
        return self


class ValidationResult(ContractModel):
    """The coordinator's decision on one proposal (CONTRACTS section 8.6)."""

    schema_version: Literal["1.0"] = "1.0"
    validation_id: ValidationId
    episode_id: EpisodeId
    step: int = Field(ge=0, le=10000)
    proposal_id: ProposalId | None
    agent_id: AgentId | None
    decision: Decision
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    resolved_action: ActionSpec | None
    allowlist_id: ShortText
    state_digest: Hex64 | None = None

    @model_validator(mode="after")
    def _validate_decision_consistency(self) -> "ValidationResult":
        if self.decision is Decision.accepted:
            if self.reason_codes:
                raise ValueError("an accepted validation must have no reason codes")
            if self.resolved_action is None:
                raise ValueError("an accepted validation must have a resolved_action")
        else:  # rejected
            if not self.reason_codes:
                raise ValueError("a rejected validation must list at least one reason code")
            if self.resolved_action is not None:
                raise ValueError("a rejected validation must have a null resolved_action")
        if not reason_codes_in_order(self.reason_codes):
            raise ValueError(
                "reason_codes must be listed in the canonical check order without repeats"
            )
        return self


class StatusChange(ContractModel):
    """A visible node status transition caused by an execution."""

    node_id: NodeId
    status_from: NodeStatus
    status_to: NodeStatus


class ExecutedAction(ContractModel):
    """One accepted action as executed by the simulator (CONTRACTS section 8.7)."""

    schema_version: Literal["1.0"] = "1.0"
    execution_id: ExecutionId
    episode_id: EpisodeId
    step: int = Field(ge=0, le=10000)
    proposal_id: ProposalId
    validation_id: ValidationId
    agent_id: AgentId
    action: ActionSpec
    status: ExecutionStatus
    error_code: ErrorCode | None
    affected_node_ids: list[NodeId] = Field(default_factory=list)
    status_changes: list[StatusChange] = Field(default_factory=list)
    outcome_event_ids: list[EventId] = Field(default_factory=list)
    state_digest_before: Hex64 | None = None
    state_digest_after: Hex64 | None = None

    @model_validator(mode="after")
    def _validate_status(self) -> "ExecutedAction":
        if self.status is ExecutionStatus.failed:
            if self.error_code is None:
                raise ValueError("a failed execution must carry an error_code")
        else:  # executed
            if self.error_code is not None:
                raise ValueError("an executed action must have a null error_code")
        return self
