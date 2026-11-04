"""Scenario configuration records (CONTRACTS section 7.4).

Every field is a requirement on any simulator: a simulator that cannot honour a
field must refuse the scenario with ``SCENARIO_UNSUPPORTED`` rather than ignore
it. The random seed is not part of the scenario; it is passed to
``reset(scenario, seed)``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .common import ContractModel, ShortText, UnitFloat
from .ids import NodeId, ScenarioId, TopologyId


class IncidentSpec(ContractModel):
    """The single abstract incident type of version 1.0: propagating compromise."""

    incident_type: Literal["propagating_compromise"] = "propagating_compromise"
    entry_node_id: NodeId
    start_step: int = Field(ge=1)
    spread_probability: UnitFloat


class ObservabilitySpec(ContractModel):
    """How observable the incident is (alerts, delays, missing values)."""

    alert_detect_probability: UnitFloat
    false_alert_rate: UnitFloat
    alert_delay_max_steps: int = Field(ge=0, le=5)
    missing_value_rate: UnitFloat


class ScenarioConfig(ContractModel):
    """A scenario configuration (CONTRACTS section 7.4)."""

    schema_version: Literal["1.0"] = "1.0"
    scenario_id: ScenarioId
    topology_id: TopologyId
    description: ShortText | None = None
    max_steps: int = Field(ge=5, le=500)
    observation_window_steps: int = Field(ge=1, le=20)
    incident: IncidentSpec
    observability: ObservabilitySpec
    restore_duration_steps: int = Field(ge=1, le=10)
    objective_node_id: NodeId | None

    @model_validator(mode="after")
    def _incident_starts_before_horizon(self) -> "ScenarioConfig":
        # start_step is 1 .. max_steps - 1; start_step >= max_steps is rejected.
        if self.incident.start_step >= self.max_steps:
            raise ValueError(
                "incident.start_step must be less than max_steps"
            )
        return self
