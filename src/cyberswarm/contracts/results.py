"""Episode, evaluation and comparison records (CONTRACTS sections 8.8, 11.1 to 11.3, 11.7, 11.8).

These records are produced after an episode or a run and are never read by a
decision-time component. Optional values that no producer computed are ``null``
(CONTRACTS C4): ``reward`` is null when no reward model ran, the ranking fields
``origin_rank_top1``/``origin_mrr`` are null when undefined, and the per-metric
summaries hold ``null`` (never ``0.0``) for an undefined metric.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from .common import (
    ContractModel,
    Provenance,
    Representation,
    ShortText,
    SimulatorKind,
    Split,
    UnitFloat,
)
from .ids import (
    AggregateId,
    CalibrationReportId,
    ComparisonId,
    EpisodeId,
    MatchedSetId,
    MetricsId,
    NodeId,
    NodeScoreId,
    PolicyFamilyId,
    PolicyId,
    RunId,
    ScenarioId,
)


class ScorerKind(StrEnum):
    """Which component produced a per-node suspicion score."""

    detection_rule = "detection_rule"
    graph_feature_rule = "graph_feature_rule"
    learned_graph_model = "learned_graph_model"


class TerminationReason(StrEnum):
    """Why an episode ended."""

    objective_reached = "objective_reached"
    incident_resolved = "incident_resolved"
    max_steps = "max_steps"


class RewardSharing(StrEnum):
    """How reward is shared among agents."""

    team = "team"
    zone_local = "zone_local"


class CIMethod(StrEnum):
    """Confidence-interval method."""

    percentile_bootstrap = "percentile_bootstrap"


class ComparisonUnit(StrEnum):
    """The pairing unit of a comparison."""

    matched_pair = "matched_pair"
    training_seed = "training_seed"


class TrainingStatus(StrEnum):
    """Whether training of a seed completed."""

    completed = "completed"
    aborted = "aborted"


class LearningCheck(StrEnum):
    """Descriptive training-return check; says nothing about policy quality."""

    improved = "improved"
    not_improved = "not_improved"
    not_assessed = "not_assessed"


class CalibrationStatus(StrEnum):
    """Status of the held-out calibration check."""

    ok = "ok"
    insufficient_data = "insufficient_data"
    not_informative = "not_informative"


class ThresholdSelection(StrEnum):
    """Whether thresholds are defaults or selected on the calibration split."""

    default = "default"
    selected_on_cal = "selected_on_cal"


# --------------------------------------------------------------------------- #
# NodeScoreRecord (CONTRACTS section 8.8)
# --------------------------------------------------------------------------- #
class NodeScore(ContractModel):
    """One predicted suspicion score for a node."""

    node_id: NodeId
    score: UnitFloat


class NodeScoreRecord(ContractModel):
    """Per-node predicted suspicion, one per agent per step per scorer (8.8).

    A score is a predicted suspicion from agent-visible inputs; a ranking of
    scores is not a root-cause or causal finding.
    """

    schema_version: Literal["1.0"] = "1.0"
    node_score_id: NodeScoreId
    episode_id: EpisodeId
    step: int = Field(ge=0, le=10000)
    agent_id: str
    scorer_id: ShortText
    scorer_kind: ScorerKind
    nodes: list[NodeScore] = Field(default_factory=list)

    @model_validator(mode="after")
    def _id_prefix(self) -> "NodeScoreRecord":
        prefix = f"nsc-{self.step:04d}-{self.agent_id}-"
        if not self.node_score_id.startswith(prefix):
            raise ValueError(
                f"node_score_id must start with {prefix!r}, got {self.node_score_id!r}"
            )
        return self


# --------------------------------------------------------------------------- #
# EpisodeResult (CONTRACTS section 11.1)
# --------------------------------------------------------------------------- #
class Counts(ContractModel):
    """Proposal/validation/execution counts of an episode."""

    proposals: int = Field(ge=0)
    accepted: int = Field(ge=0)
    rejected: int = Field(ge=0)
    executed: int = Field(ge=0)
    failed: int = Field(ge=0)

    @model_validator(mode="after")
    def _arithmetic(self) -> "Counts":
        if self.accepted != self.executed + self.failed:
            raise ValueError("accepted must equal executed + failed")
        if self.proposals != self.accepted + self.rejected:
            raise ValueError("proposals must equal accepted + rejected")
        return self


class RewardTermTotals(ContractModel):
    """The seven raw, unweighted reward-term magnitudes (RL_IMPLEMENTATION 4)."""

    containment: float
    response_delay: float
    attack_impact: float
    false_isolation: float
    healthy_disruption: float
    investigation_cost: float
    invalid_action: float


class RewardSummary(ContractModel):
    """Episode reward summary; null on the episode when no reward model ran."""

    sharing: RewardSharing
    weights_id: ShortText
    total: float
    term_totals: RewardTermTotals


class NodeScoring(ContractModel):
    """Per-scorer origin-ranking summary for an episode (CONTRACTS 11.1/11.2)."""

    scorer_id: ShortText
    scorer_kind: ScorerKind
    origin_rank_top1: float | None = None
    origin_mrr: float | None = None
    n_ranked_pairs: int = Field(ge=0)


class OutcomeSummary(ContractModel):
    """Ground-truth-derived episode outcome (CONTRACTS section 11.1)."""

    compromised_nodes: int = Field(ge=0)
    attack_impact: float
    healthy_services_disrupted: int = Field(ge=0)
    healthy_disruption_cost: float
    false_isolation_count: int = Field(ge=0)
    true_positive_alerts: int = Field(ge=0)
    false_positive_alerts: int = Field(ge=0)
    compromised_nodes_detected: int = Field(ge=0)
    first_compromise_step: int | None = Field(default=None, ge=0)
    first_true_alert_step: int | None = Field(default=None, ge=0)
    containment_step: int | None = Field(default=None, ge=0)
    detection_to_containment_steps: int | None = None
    objective_reached: bool
    origin_node_id: NodeId | None = None
    node_scoring: list[NodeScoring] = Field(default_factory=list)


class UncertaintySummary(ContractModel):
    """Uncertainty activity summary; null unless an uncertainty policy ran."""

    deferral_count: int = Field(ge=0)
    evidence_request_count: int = Field(ge=0)
    containment_allowed_count: int = Field(ge=0)
    mean_uncertainty: UnitFloat | None = None


class EpisodeResult(ContractModel):
    """Post-hoc result of one episode (CONTRACTS section 11.1)."""

    schema_version: Literal["1.0"] = "1.0"
    episode_id: EpisodeId
    run_id: RunId
    scenario_id: ScenarioId
    seed: int = Field(ge=0)
    policy_id: PolicyId
    representation: Representation
    split: Split
    simulator_kind: SimulatorKind
    visibility: Literal["post_hoc_evaluation"] = "post_hoc_evaluation"
    steps_taken: int = Field(ge=0)
    terminated: bool
    truncated: bool
    termination_reason: TerminationReason
    counts: Counts
    reward: RewardSummary | None
    outcome: OutcomeSummary
    uncertainty_summary: UncertaintySummary | None
    provenance: Provenance

    @model_validator(mode="after")
    def _terminated_xor_truncated(self) -> "EpisodeResult":
        if self.terminated == self.truncated:
            raise ValueError("exactly one of terminated and truncated must be true")
        return self


# --------------------------------------------------------------------------- #
# EvaluationMetrics (CONTRACTS section 11.2)
# --------------------------------------------------------------------------- #
class MetricSummary(ContractModel):
    """Summary of one metric over a matched set (CONTRACTS section 11.2)."""

    name: ShortText
    scorer_id: ShortText | None = None
    mean: float | None = None
    std: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n: int = Field(ge=0)
    n_undefined: int = Field(ge=0)


class EvaluationMetrics(ContractModel):
    """Metrics for one policy, representation and split (CONTRACTS 11.2)."""

    schema_version: Literal["1.0"] = "1.0"
    metrics_id: MetricsId
    run_id: RunId
    policy_id: PolicyId
    representation: Representation
    split: Split
    policy_family_id: PolicyFamilyId
    training_seed: int | None = Field(default=None, ge=0)
    matched_set_id: MatchedSetId
    n_episodes: int = Field(ge=1)
    n_aborted: int = Field(ge=0)
    scenario_ids: list[ScenarioId] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    ci_method: CIMethod | None = None
    bootstrap_resamples: int | None = Field(default=None, ge=1)
    metrics: list[MetricSummary] = Field(default_factory=list)
    provenance: Provenance


# --------------------------------------------------------------------------- #
# AggregateMetrics (CONTRACTS section 11.8)
# --------------------------------------------------------------------------- #
class TrainingOutcome(ContractModel):
    """Training status and learning check for one training seed."""

    training_seed: int = Field(ge=0)
    training_status: TrainingStatus
    learning_check: LearningCheck


class AggregateMetricSummary(ContractModel):
    """Summary of one metric aggregated across training seeds (11.8)."""

    name: ShortText
    scorer_id: ShortText | None = None
    per_seed_means: list[float | None] = Field(default_factory=list)
    mean: float | None = None
    std: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_units_defined: int = Field(ge=0)


class AggregateMetrics(ContractModel):
    """Statistics over training seeds for a learned arm (CONTRACTS 11.8)."""

    schema_version: Literal["1.0"] = "1.0"
    aggregate_id: AggregateId
    run_id: RunId
    policy_family_id: PolicyFamilyId
    representation: Representation
    split: Split
    matched_set_id: MatchedSetId
    training_seeds: list[int] = Field(min_length=1)
    training_outcomes: list[TrainingOutcome] = Field(default_factory=list)
    n_units: int = Field(ge=1)
    ci_method: CIMethod | None = None
    bootstrap_resamples: int | None = Field(default=None, ge=1)
    metrics: list[AggregateMetricSummary] = Field(default_factory=list)
    provenance: Provenance

    @model_validator(mode="after")
    def _training_seeds_sorted_unique(self) -> "AggregateMetrics":
        if self.training_seeds != sorted(self.training_seeds):
            raise ValueError("training_seeds must be sorted ascending")
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("training_seeds must be unique")
        if any(seed < 0 for seed in self.training_seeds):
            raise ValueError("training_seeds must be non-negative")
        return self


# --------------------------------------------------------------------------- #
# ComparisonRecord (CONTRACTS section 11.3)
# --------------------------------------------------------------------------- #
class MetricDifference(ContractModel):
    """A paired difference for one metric (candidate minus baseline)."""

    name: ShortText
    scorer_id: ShortText | None = None
    mean_difference: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_pairs: int = Field(ge=1)


class ComparisonRecord(ContractModel):
    """Paired comparison of two policy families (CONTRACTS section 11.3)."""

    schema_version: Literal["1.0"] = "1.0"
    comparison_id: ComparisonId
    run_id: RunId
    matched_set_id: MatchedSetId
    split: Split
    baseline_policy_family_id: PolicyFamilyId
    candidate_policy_family_id: PolicyFamilyId
    baseline_representation: Representation
    candidate_representation: Representation
    unit: ComparisonUnit
    n_pairs: int = Field(ge=1)
    differences: list[MetricDifference] = Field(default_factory=list)
    provenance: Provenance


# --------------------------------------------------------------------------- #
# CalibrationReport (CONTRACTS section 11.7)
# --------------------------------------------------------------------------- #
class ReliabilityBin(ContractModel):
    """One confidence bin of the reliability diagram."""

    lower: float
    upper: float
    n: int = Field(ge=0)
    mean_confidence: UnitFloat | None = None
    accuracy: UnitFloat | None = None


class CalibrationReport(ContractModel):
    """Held-out calibration check of the uncertainty signal (CONTRACTS 11.7)."""

    schema_version: Literal["1.0"] = "1.0"
    report_id: CalibrationReportId
    run_id: RunId
    policy_id: PolicyId
    cal_split: Literal["cal"] = "cal"
    eval_split: Literal["eval", "eval_utopo"]
    n_cal: int = Field(ge=0)
    n_eval: int = Field(ge=0)
    bins: list[ReliabilityBin] = Field(default_factory=list)
    ece_raw: float | None = None
    ece_calibrated: float | None = None
    mce: float | None = None
    brier: float | None = None
    auroc: float | None = None
    auroc_ci_low: float | None = None
    auroc_ci_high: float | None = None
    status: CalibrationStatus
    tau_evidence: float = Field(ge=0.0, le=1.01)
    tau_defer: float = Field(ge=0.0, le=1.01)
    threshold_selection: ThresholdSelection
    provenance: Provenance
