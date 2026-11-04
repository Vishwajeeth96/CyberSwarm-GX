"""Agent-visible observation records (CONTRACTS sections 8.2 to 8.4).

These records are what one decision agent may see at a step. By construction
they contain no evaluator-only ground truth (``extra="forbid"`` makes a field
such as ``compromised`` a validation error) and no event from a future step.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .common import (
    ContractModel,
    Decision,
    ExecutionStatus,
    ImpactClass,
    NodeStatus,
    ReasonCode,
    ShortText,
    UnitFloat,
)
from .actions import ActionParameters, _check_action_params
from .events import EventRecord
from .ids import (
    AgentId,
    AvailableActionsId,
    DetectionId,
    EdgeId,
    EpisodeId,
    NodeId,
    ObservationId,
    ProposalId,
    ZoneId,
    available_actions_id,
    detection_id,
    observation_id,
)
from .vocabulary import ACTION_DEFS, ACTION_ORDER, ActionKind


class NodeScope(StrEnum):
    """Whether a node is in the agent's own zone or on its boundary."""

    own = "own"
    boundary = "boundary"


class EdgeState(StrEnum):
    """Agent-visible edge state."""

    active = "active"
    limited = "limited"
    blocked = "blocked"


class NodeObservation(ContractModel):
    """One node as an agent sees it (CONTRACTS section 8.2).

    For a boundary node the private measurements (``evidence_level``,
    ``service_health``, ``steps_since_investigated``) are withheld (null).
    """

    node_id: NodeId
    scope: NodeScope
    visible_status: NodeStatus
    alert_count_recent: int = Field(ge=0, le=50)
    max_alert_severity: int = Field(ge=0, le=3)
    evidence_level: int | None = Field(default=None, ge=0, le=3)
    service_health: UnitFloat | None = None
    steps_since_investigated: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _boundary_withholds_private(self) -> "NodeObservation":
        if self.scope is NodeScope.boundary:
            for field_name in ("evidence_level", "service_health", "steps_since_investigated"):
                if getattr(self, field_name) is not None:
                    raise ValueError(
                        f"boundary node must have null {field_name}"
                    )
        return self


class EdgeObservation(ContractModel):
    """One edge as an agent sees it (CONTRACTS section 8.2)."""

    edge_id: EdgeId
    edge_state: EdgeState
    traffic_level: UnitFloat | None = None


class ProposalFeedback(ContractModel):
    """What happened to one of the agent's proposals of the previous step."""

    proposal_id: ProposalId
    decision: Decision
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    execution_status: ExecutionStatus | None = None


class AgentObservation(ContractModel):
    """What one decision agent may see at step ``t`` (CONTRACTS section 8.2)."""

    schema_version: Literal["1.0"] = "1.0"
    observation_id: ObservationId
    episode_id: EpisodeId
    step: int = Field(ge=0, le=10000)
    agent_id: AgentId
    zone_id: ZoneId
    max_steps: int = Field(ge=1)
    observation_window_steps: int = Field(ge=1, le=20)
    nodes: list[NodeObservation] = Field(default_factory=list)
    edges: list[EdgeObservation] = Field(default_factory=list)
    recent_events: list[EventRecord] = Field(default_factory=list)
    last_step_feedback: list[ProposalFeedback] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_observation(self) -> "AgentObservation":
        expected = observation_id(self.step, self.agent_id)
        if self.observation_id != expected:
            raise ValueError(
                f"observation_id must equal {expected!r}, got {self.observation_id!r}"
            )

        node_ids = [node.node_id for node in self.nodes]
        if node_ids != sorted(node_ids):
            raise ValueError("observation nodes must be ascending by node_id")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("observation nodes must be unique")

        edge_ids = [edge.edge_id for edge in self.edges]
        if edge_ids != sorted(edge_ids):
            raise ValueError("observation edges must be ascending by edge_id")
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("observation edges must be unique")

        for event in self.recent_events:
            if event.step > self.step:
                raise ValueError(
                    "observation must not contain an event from a future step"
                )
        return self


class DetectionNode(ContractModel):
    """Per-node detection output (CONTRACTS section 8.4)."""

    node_id: NodeId
    suspicion: UnitFloat
    alerted: bool


class DetectionSummary(ContractModel):
    """Output of the deterministic detection role (CONTRACTS section 8.4)."""

    schema_version: Literal["1.0"] = "1.0"
    detection_id: DetectionId
    episode_id: EpisodeId
    step: int = Field(ge=0, le=10000)
    agent_id: AgentId
    producer: ShortText
    nodes: list[DetectionNode] = Field(default_factory=list)

    @model_validator(mode="after")
    def _detection_id_matches(self) -> "DetectionSummary":
        expected = detection_id(self.step, self.agent_id)
        if self.detection_id != expected:
            raise ValueError(
                f"detection_id must equal {expected!r}, got {self.detection_id!r}"
            )
        return self


class AvailableAction(ContractModel):
    """One action an agent may propose this step (CONTRACTS section 8.3)."""

    action_kind: ActionKind
    target_node_id: NodeId | None = None
    parameters: ActionParameters = Field(default_factory=ActionParameters)
    impact_class: ImpactClass
    high_impact: bool
    cost_units: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _matches_vocabulary(self) -> "AvailableAction":
        spec = ACTION_DEFS[self.action_kind]
        if spec.needs_target and self.target_node_id is None:
            raise ValueError(f"{self.action_kind.value} requires a target_node_id")
        if not spec.needs_target and self.target_node_id is not None:
            raise ValueError(f"{self.action_kind.value} must not carry a target_node_id")
        if self.impact_class is not spec.impact_class:
            raise ValueError(
                f"{self.action_kind.value} impact_class must be {spec.impact_class.value}"
            )
        if self.high_impact != spec.high_impact:
            raise ValueError(
                f"{self.action_kind.value} high_impact must be {spec.high_impact}"
            )
        if self.cost_units != spec.cost_units:
            raise ValueError(
                f"{self.action_kind.value} cost_units must be {spec.cost_units}"
            )
        _check_action_params(self.action_kind, self.parameters)
        return self


class AvailableActions(ContractModel):
    """The actions one agent may propose this step (CONTRACTS section 8.3)."""

    schema_version: Literal["1.0"] = "1.0"
    available_actions_id: AvailableActionsId
    episode_id: EpisodeId
    step: int = Field(ge=0, le=10000)
    agent_id: AgentId
    actions: list[AvailableAction] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_available(self) -> "AvailableActions":
        expected = available_actions_id(self.step, self.agent_id)
        if self.available_actions_id != expected:
            raise ValueError(
                f"available_actions_id must equal {expected!r}, got {self.available_actions_id!r}"
            )
        if not self.actions:
            raise ValueError("available actions must include at least no_op")
        if self.actions[0].action_kind is not ActionKind.no_op:
            raise ValueError("no_op must be present and first")
        return self
