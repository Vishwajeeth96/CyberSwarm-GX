"""Agent-visible event records (CONTRACTS section 8.1).

An :class:`EventRecord` carries no ground truth: there is deliberately no field
for "was this alert a true positive" or "is this node compromised". The
per-event-type field rules and the ``producer``/``phase`` pairing are enforced
here; the ``visible_to`` set itself is produced by the simulator from the
visibility table and is checked for shape only (non-empty, ascending, unique).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import Field, model_validator

from .common import ContractModel, NodeStatus
from .ids import AgentId, EpisodeId, EventId, ExecutionId, NodeId, ProposalId
from .vocabulary import ActionKind


class Phase(StrEnum):
    """The step phase in which an event was produced."""

    dynamics = "dynamics"
    outcome = "outcome"
    lifecycle = "lifecycle"


class EventProducer(StrEnum):
    """Who produced the event."""

    simulator = "simulator"
    coordinator = "coordinator"


class EventType(StrEnum):
    """Closed set of agent-visible event types (CONTRACTS section 8.1)."""

    episode_started = "episode_started"
    alert_raised = "alert_raised"
    restore_completed = "restore_completed"
    node_status_changed = "node_status_changed"
    action_applied = "action_applied"
    investigation_result = "investigation_result"
    recovery_verified = "recovery_verified"
    recovery_failed = "recovery_failed"
    episode_ended = "episode_ended"


class Severity(StrEnum):
    """Abstract alert severity."""

    low = "low"
    medium = "medium"
    high = "high"


class AlertCategory(StrEnum):
    """Abstract alert class; no technique names."""

    traffic_anomaly = "traffic_anomaly"
    access_anomaly = "access_anomaly"
    service_anomaly = "service_anomaly"


# Optional, type-specific fields. For each event type, the fields in the "required"
# set must be non-null and every other field in this tuple must be null.
_TYPE_SPECIFIC_FIELDS: Final[tuple[str, ...]] = (
    "node_id",
    "agent_id",
    "severity",
    "alert_category",
    "evidence_level",
    "status_from",
    "status_to",
    "action_kind",
    "proposal_id",
    "execution_id",
    "terminated",
    "truncated",
)

# event_type -> (phase, producer, required non-null fields)
_EVENT_RULES: Final[dict[EventType, tuple[Phase, EventProducer, frozenset[str]]]] = {
    EventType.episode_started: (Phase.lifecycle, EventProducer.simulator, frozenset()),
    EventType.alert_raised: (
        Phase.dynamics,
        EventProducer.simulator,
        frozenset({"node_id", "severity", "alert_category"}),
    ),
    EventType.restore_completed: (
        Phase.dynamics,
        EventProducer.simulator,
        frozenset({"node_id", "status_from", "status_to"}),
    ),
    EventType.node_status_changed: (
        Phase.outcome,
        EventProducer.simulator,
        frozenset({"node_id", "status_from", "status_to", "execution_id"}),
    ),
    EventType.action_applied: (
        Phase.outcome,
        EventProducer.simulator,
        # node_id is required too, except for no_op (handled specially below).
        frozenset({"action_kind", "agent_id", "proposal_id", "execution_id"}),
    ),
    EventType.investigation_result: (
        Phase.outcome,
        EventProducer.simulator,
        frozenset({"node_id", "evidence_level", "agent_id", "execution_id"}),
    ),
    EventType.recovery_verified: (
        Phase.outcome,
        EventProducer.coordinator,
        frozenset({"node_id", "execution_id"}),
    ),
    EventType.recovery_failed: (
        Phase.outcome,
        EventProducer.coordinator,
        frozenset({"node_id", "execution_id"}),
    ),
    EventType.episode_ended: (
        Phase.lifecycle,
        EventProducer.simulator,
        frozenset({"terminated", "truncated"}),
    ),
}


class EventRecord(ContractModel):
    """One agent-visible event (CONTRACTS section 8.1)."""

    schema_version: Literal["1.0"] = "1.0"
    event_id: EventId
    episode_id: EpisodeId
    step: int = Field(ge=0, le=10000)
    phase: Phase
    event_type: EventType
    producer: EventProducer
    visible_to: list[AgentId]
    node_id: NodeId | None = None
    agent_id: AgentId | None = None
    severity: Severity | None = None
    alert_category: AlertCategory | None = None
    evidence_level: int | None = Field(default=None, ge=0, le=3)
    status_from: NodeStatus | None = None
    status_to: NodeStatus | None = None
    action_kind: ActionKind | None = None
    proposal_id: ProposalId | None = None
    execution_id: ExecutionId | None = None
    terminated: bool | None = None
    truncated: bool | None = None

    @model_validator(mode="after")
    def _validate_visible_to(self) -> "EventRecord":
        if not self.visible_to:
            raise ValueError("visible_to must list at least one agent")
        if self.visible_to != sorted(self.visible_to):
            raise ValueError("visible_to must be ascending")
        if len(set(self.visible_to)) != len(self.visible_to):
            raise ValueError("visible_to must not repeat an agent")
        return self

    @model_validator(mode="after")
    def _validate_type_rules(self) -> "EventRecord":
        phase, producer, required = _EVENT_RULES[self.event_type]
        if self.phase is not phase:
            raise ValueError(
                f"{self.event_type} must use phase {phase.value}, got {self.phase.value}"
            )
        if self.producer is not producer:
            raise ValueError(
                f"{self.event_type} must be produced by {producer.value}"
            )

        required_effective = set(required)
        if self.event_type is EventType.action_applied:
            # node_id is required unless the action is no_op; it must be null for no_op.
            if self.action_kind is ActionKind.no_op:
                if self.node_id is not None:
                    raise ValueError("action_applied for no_op must have null node_id")
            else:
                required_effective.add("node_id")

        for field_name in required_effective:
            if getattr(self, field_name) is None:
                raise ValueError(
                    f"{self.event_type} requires non-null {field_name}"
                )
        for field_name in _TYPE_SPECIFIC_FIELDS:
            if field_name in required_effective:
                continue
            if getattr(self, field_name) is not None:
                raise ValueError(
                    f"{self.event_type} must have null {field_name}"
                )
        return self

    @model_validator(mode="after")
    def _validate_id_prefix(self) -> "EventRecord":
        # Simulator events use the evt- counter; coordinator events use cev-.
        expected_prefix = "cev-" if self.producer is EventProducer.coordinator else "evt-"
        if not self.event_id.startswith(expected_prefix):
            raise ValueError(
                f"{self.producer.value} events must use the {expected_prefix!r} id prefix"
            )
        return self
