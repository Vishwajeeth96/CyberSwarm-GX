"""Record-name to model mapping used by schema export, validators and tests.

``RECORD_TYPES`` names every top-level record that is serialized to disk and
exported as ``schemas/<Name>.schema.json``. It includes ``NodeScoreRecord`` and
the revision-2 additions ``AggregateMetrics`` and ``StageDecision`` (each has a
documented ``json contract=<Name>`` example that must validate against an
exported schema).
"""

from __future__ import annotations

from .actions import ActionSpec, ExecutedAction, PolicyProposal, ValidationResult
from .common import ContractModel
from .events import EventRecord
from .evaluator_only import GroundTruthSnapshot
from .graph import GraphFeatureRecord, GraphModelOutput, GraphSnapshot
from .manifest import ErrorRecord, LogRecord, RunManifest, StageDecision
from .observation import AgentObservation, AvailableActions, DetectionSummary
from .results import (
    AggregateMetrics,
    CalibrationReport,
    ComparisonRecord,
    EpisodeResult,
    EvaluationMetrics,
    NodeScoreRecord,
)
from .scenario import ScenarioConfig
from .topology import Edge, Node, Topology

RECORD_TYPES: dict[str, type[ContractModel]] = {
    # Static topology and scenario.
    "Node": Node,
    "Edge": Edge,
    "Topology": Topology,
    "ScenarioConfig": ScenarioConfig,
    # Decision-loop records.
    "EventRecord": EventRecord,
    "AgentObservation": AgentObservation,
    "AvailableActions": AvailableActions,
    "DetectionSummary": DetectionSummary,
    "ActionSpec": ActionSpec,
    "PolicyProposal": PolicyProposal,
    "ValidationResult": ValidationResult,
    "ExecutedAction": ExecutedAction,
    "NodeScoreRecord": NodeScoreRecord,
    # Evaluator-only ground truth.
    "GroundTruthSnapshot": GroundTruthSnapshot,
    # Graph records.
    "GraphSnapshot": GraphSnapshot,
    "GraphFeatureRecord": GraphFeatureRecord,
    "GraphModelOutput": GraphModelOutput,
    # Episode, evaluation and comparison records.
    "EpisodeResult": EpisodeResult,
    "EvaluationMetrics": EvaluationMetrics,
    "AggregateMetrics": AggregateMetrics,
    "ComparisonRecord": ComparisonRecord,
    "CalibrationReport": CalibrationReport,
    "StageDecision": StageDecision,
    # Error, log and run-manifest records.
    "ErrorRecord": ErrorRecord,
    "LogRecord": LogRecord,
    "RunManifest": RunManifest,
}
