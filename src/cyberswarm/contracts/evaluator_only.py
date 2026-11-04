"""Evaluator-only ground truth (CONTRACTS section 9).

:class:`GroundTruthSnapshot` is hidden truth about the simulated network. It is
never part of an observation, a graph record, a model input, a policy input, an
``info`` dictionary or any decision-time data. It lives only under
``evaluator_only/`` and may be imported only by the packages listed in the
single evaluator-only allowlist (DESIGN section 3.2).

Each node carries both ``service_availability`` (including the compromise
penalty; used by the attack-impact view) and ``defense_availability``
(availability from status and dependency effects only, compromise factor fixed
at 1.0). The healthy-disruption reward term and the two disruption metrics use
``defense_availability`` so that damage done by the attack is not charged to the
defender.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .common import ContractModel, NodeStatus, UnitFloat
from .ids import EpisodeId, EventId, NodeId


class CompromiseCause(StrEnum):
    """Why a node became compromised during a step."""

    initial_entry = "initial_entry"
    propagation = "propagation"


class GroundTruthNode(ContractModel):
    """Hidden per-node truth (CONTRACTS section 9)."""

    node_id: NodeId
    compromised: bool
    service_availability: UnitFloat
    defense_availability: UnitFloat
    contained: bool
    status: NodeStatus


class NewCompromise(ContractModel):
    """A node that became compromised during this step."""

    node_id: NodeId
    cause: CompromiseCause


class AlertTruth(ContractModel):
    """Whether one ``alert_raised`` event of this step was a true positive."""

    event_id: EventId
    true_positive: bool


class GroundTruthSnapshot(ContractModel):
    """Hidden truth captured at the end of a step (CONTRACTS section 9).

    One record per step, captured by ``end_step()`` of that step after P4 and
    before P0 of the next step; ``step`` equals ``StepResult.step_closed``.
    """

    schema_version: Literal["1.0"] = "1.0"
    episode_id: EpisodeId
    step: int = Field(ge=0, le=10000)
    visibility: Literal["evaluator_only"] = "evaluator_only"
    incident_active: bool
    nodes: list[GroundTruthNode] = Field(default_factory=list)
    new_compromises: list[NewCompromise] = Field(default_factory=list)
    alert_truth: list[AlertTruth] = Field(default_factory=list)

    @model_validator(mode="after")
    def _nodes_ascending(self) -> "GroundTruthSnapshot":
        node_ids = [node.node_id for node in self.nodes]
        if node_ids != sorted(node_ids):
            raise ValueError("ground-truth nodes must be ascending by node_id")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("ground-truth nodes must be unique")
        return self
