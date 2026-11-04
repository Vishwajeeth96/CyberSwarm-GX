"""Error, log, run-manifest and stage-decision records (CONTRACTS 11.4 to 11.6, 11.9).

This is the only contracts module that mentions the deferred second training
stage by name, both as the ``RunManifest.status`` field and as a
:class:`StageDecision.component` value; the import-boundary test allows that
token here and in ``runner/export.py`` only.

``RunManifest.status`` is execution-derived: the runner computes each state from
what a run actually executed and from the committed :class:`StageDecision`
files. A ``reason`` is required exactly when a state is ``no_go`` or
``deferred``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .common import (
    ComponentStatus,
    ContractModel,
    ErrorCode,
    Hex64,
    Provenance,
    RelPath,
    ShortText,
    SimulatorKind,
)
from .ids import (
    DecisionId,
    EpisodeId,
    ErrorId,
    ExecutionId,
    NodeId,
    PolicyId,
    ProposalId,
    RunId,
    ScenarioId,
    TopologyId,
    ValidationId,
)


class Component(StrEnum):
    """Which component produced an error or log line (CONTRACTS section 11.4)."""

    contracts = "contracts"
    simulator = "simulator"
    mock = "mock"
    roles = "roles"
    graph = "graph"
    engine = "engine"
    coordinator = "coordinator"
    rl_env = "rl_env"
    baseline = "baseline"
    marl = "marl"
    uncertainty = "uncertainty"
    gnn = "gnn"
    evaluation = "evaluation"
    runner = "runner"
    config = "config"


class ErrorSeverity(StrEnum):
    """Severity of an error record."""

    warning = "warning"
    error = "error"
    fatal = "fatal"


class LogLevel(StrEnum):
    """Log level of a log record."""

    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"


class EpisodeStatus(StrEnum):
    """Whether an episode completed or aborted."""

    completed = "completed"
    aborted = "aborted"


# --------------------------------------------------------------------------- #
# ErrorRecord (CONTRACTS section 11.4)
# --------------------------------------------------------------------------- #
class ErrorRecord(ContractModel):
    """A structured error (CONTRACTS section 11.4)."""

    schema_version: Literal["1.0"] = "1.0"
    error_id: ErrorId
    run_id: RunId | None = None
    episode_id: EpisodeId | None = None
    step: int | None = Field(default=None, ge=0, le=10000)
    component: Component
    code: ErrorCode
    severity: ErrorSeverity
    recoverable: bool
    message: ShortText
    proposal_id: ProposalId | None = None
    validation_id: ValidationId | None = None
    execution_id: ExecutionId | None = None
    node_id: NodeId | None = None


# --------------------------------------------------------------------------- #
# LogRecord (CONTRACTS section 11.5)
# --------------------------------------------------------------------------- #
class LogRecord(ContractModel):
    """One JSON log line; ``ts`` is the only wall-clock value in the system."""

    schema_version: Literal["1.0"] = "1.0"
    ts: str
    level: LogLevel
    component: Component
    run_id: RunId | None = None
    episode_id: EpisodeId | None = None
    step: int | None = Field(default=None, ge=0, le=10000)
    code: ErrorCode | None = None
    message: ShortText


# --------------------------------------------------------------------------- #
# RunManifest (CONTRACTS section 11.6)
# --------------------------------------------------------------------------- #
class EpisodeEntry(ContractModel):
    """One episode entry of a run manifest."""

    episode_id: EpisodeId
    directory: RelPath
    status: EpisodeStatus


class FileEntry(ContractModel):
    """One record file of a run manifest."""

    path: RelPath
    record_type: ShortText
    count: int = Field(ge=0)


# Allowed status states per staged component (CONTRACTS section 11.6).
_ALLOWED_STATES: dict[str, frozenset[ComponentStatus]] = {
    "ippo": frozenset({ComponentStatus.not_run, ComponentStatus.implemented}),
    "learned_gnn": frozenset(
        {ComponentStatus.not_run, ComponentStatus.implemented, ComponentStatus.no_go}
    ),
    "mappo": frozenset(
        {ComponentStatus.not_run, ComponentStatus.implemented, ComponentStatus.deferred}
    ),
}
_NEEDS_REASON: frozenset[ComponentStatus] = frozenset(
    {ComponentStatus.no_go, ComponentStatus.deferred}
)


class StatusEntry(ContractModel):
    """Status of one staged component in a run manifest."""

    state: ComponentStatus
    reason: ShortText | None = None
    decision_id: DecisionId | None = None

    @model_validator(mode="after")
    def _reason_required_when_set(self) -> "StatusEntry":
        if self.state in _NEEDS_REASON:
            if self.reason is None:
                raise ValueError(
                    f"state {self.state.value} requires a reason"
                )
        else:
            if self.reason is not None:
                raise ValueError(
                    f"state {self.state.value} must have a null reason"
                )
            if self.decision_id is not None:
                raise ValueError(
                    f"state {self.state.value} must have a null decision_id"
                )
        return self


class StatusBlock(ContractModel):
    """The IPPO, deferred-stage and learned-GNN status entries."""

    ippo: StatusEntry
    mappo: StatusEntry
    learned_gnn: StatusEntry

    @model_validator(mode="after")
    def _states_allowed(self) -> "StatusBlock":
        for field_name, entry in (
            ("ippo", self.ippo),
            ("mappo", self.mappo),
            ("learned_gnn", self.learned_gnn),
        ):
            if entry.state not in _ALLOWED_STATES[field_name]:
                allowed = ", ".join(sorted(s.value for s in _ALLOWED_STATES[field_name]))
                raise ValueError(
                    f"status.{field_name} state {entry.state.value} is not one of: {allowed}"
                )
        return self


class RunManifest(ContractModel):
    """Index of one run directory (CONTRACTS section 11.6)."""

    schema_version: Literal["1.0"] = "1.0"
    run_id: RunId
    simulator_kind: SimulatorKind
    topology_ids: list[TopologyId] = Field(default_factory=list)
    scenario_ids: list[ScenarioId] = Field(default_factory=list)
    policy_ids: list[PolicyId] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    config_digest: Hex64
    package_version: ShortText
    episodes: list[EpisodeEntry] = Field(default_factory=list)
    files: list[FileEntry] = Field(default_factory=list)
    graph_model_outputs: Literal["not_computed", "present"]
    status: StatusBlock
    provenance: Provenance


# --------------------------------------------------------------------------- #
# StageDecision (CONTRACTS section 11.9)
# --------------------------------------------------------------------------- #
class StageComponent(StrEnum):
    """The staged component a decision is about."""

    learned_gnn = "learned_gnn"
    mappo = "mappo"


class StageName(StrEnum):
    """The workflow stage that decided."""

    S7 = "S7"
    S8 = "S8"


class StageCriterion(ContractModel):
    """One go/no-go criterion with its measured value."""

    criterion_id: ShortText
    passed: bool | None = None
    measured: float | None = None
    threshold: float | None = None
    note: ShortText | None = None


_STAGE_ALLOWED_STATES: dict[StageComponent, frozenset[ComponentStatus]] = {
    StageComponent.learned_gnn: frozenset(
        {ComponentStatus.implemented, ComponentStatus.no_go}
    ),
    StageComponent.mappo: frozenset(
        {ComponentStatus.implemented, ComponentStatus.deferred}
    ),
}


class StageDecision(ContractModel):
    """Recorded go, no-go or deferral of a staged component (CONTRACTS 11.9)."""

    schema_version: Literal["1.0"] = "1.0"
    decision_id: DecisionId
    component: StageComponent
    stage: StageName
    state: ComponentStatus
    reason: ShortText | None
    criteria: list[StageCriterion] = Field(default_factory=list)
    evidence_paths: list[RelPath] = Field(default_factory=list)
    provenance: Provenance

    @model_validator(mode="after")
    def _validate_decision(self) -> "StageDecision":
        allowed = _STAGE_ALLOWED_STATES[self.component]
        if self.state not in allowed:
            allowed_str = ", ".join(sorted(s.value for s in allowed))
            raise ValueError(
                f"{self.component.value} state must be one of: {allowed_str}"
            )
        needs_reason = self.state in _NEEDS_REASON
        if needs_reason:
            if self.reason is None:
                raise ValueError(f"state {self.state.value} requires a reason")
            if not self.evidence_paths:
                raise ValueError(
                    f"state {self.state.value} requires at least one evidence path"
                )
        else:
            if self.reason is not None:
                raise ValueError(
                    f"state {self.state.value} must have a null reason"
                )
        return self
