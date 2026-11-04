"""Graph records (CONTRACTS section 10).

The graph pipeline turns agent-visible records into deterministic matrices. The
contract models here validate shape, the feature-name registries, the mask
rules (0/1, imputed entries equal ``0.0`` for the base snapshot) and the
``max_event_step <= step`` rule. :class:`GraphModelOutput` carries an explicit
``not_computed`` shape so "no learned model ran" has one agreed form; its
payload fields are null exactly when ``status == not_computed``.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field, model_validator

from .common import ContractModel, ShortText
from .ids import (
    AgentId,
    EdgeId,
    EpisodeId,
    FeatureRecordId,
    GraphModelOutputId,
    NodeId,
    ObservationId,
    SnapshotId,
    TopologyId,
)
from .observation import EdgeState  # noqa: F401  (documents the edge-state source)

#: Node feature registry ``graph-base-v1`` (11 columns, CONTRACTS section 10.1).
NODE_FEATURE_NAMES_BASE_V1: Final[tuple[str, ...]] = (
    "criticality",
    "status_up",
    "status_rate_limited",
    "status_quarantined",
    "status_isolated",
    "status_restoring",
    "alert_count_norm",
    "max_alert_severity_norm",
    "evidence_level_norm",
    "service_health",
    "steps_since_investigated_norm",
)

#: Edge feature registry ``graph-base-v1`` (6 columns, CONTRACTS section 10.1).
EDGE_FEATURE_NAMES_BASE_V1: Final[tuple[str, ...]] = (
    "type_network_link",
    "type_service_dependency",
    "state_active",
    "state_limited",
    "state_blocked",
    "traffic_level",
)

#: Derived feature registry ``graph-derived-v1`` (8 columns, CONTRACTS 10.2).
DERIVED_FEATURE_NAMES_V1: Final[tuple[str, ...]] = (
    "degree_norm",
    "neighbor_crit_mean",
    "neighbor_crit_max",
    "dependents_crit_norm",
    "neighbor_alert_mean",
    "neighbor_alert_max",
    "hop_to_alerted_norm",
    "blocked_edge_frac",
)


def _validate_mask_matrix(
    mask: list[list[int]],
    rows: int,
    cols: int,
    *,
    label: str,
) -> None:
    if len(mask) != rows:
        raise ValueError(f"{label} mask must have {rows} rows, got {len(mask)}")
    for row in mask:
        if len(row) != cols:
            raise ValueError(f"{label} mask rows must have {cols} columns")
        for value in row:
            if value not in (0, 1):
                raise ValueError(f"{label} mask values must be 0 or 1")


def _validate_float_matrix(
    matrix: list[list[float]],
    rows: int,
    cols: int,
    *,
    label: str,
    unit_range: bool = False,
) -> None:
    if len(matrix) != rows:
        raise ValueError(f"{label} must have {rows} rows, got {len(matrix)}")
    for row in matrix:
        if len(row) != cols:
            raise ValueError(f"{label} rows must have {cols} columns")
        if unit_range:
            for value in row:
                if not (0.0 <= value <= 1.0):
                    raise ValueError(f"{label} values must be in [0, 1]")


class GraphSnapshot(ContractModel):
    """Deterministic per-agent, per-step graph matrices (CONTRACTS section 10.1)."""

    schema_version: Literal["1.0"] = "1.0"
    snapshot_id: SnapshotId
    episode_id: EpisodeId
    step: int = Field(ge=0, le=10000)
    agent_id: AgentId
    topology_id: TopologyId
    feature_set_id: Literal["graph-base-v1"] = "graph-base-v1"
    node_order: list[NodeId]
    edge_order: list[EdgeId]
    edge_index: list[list[int]]
    node_feature_names: list[str]
    node_features: list[list[float]]
    node_mask: list[list[int]]
    edge_feature_names: list[str]
    edge_features: list[list[float]]
    edge_mask: list[list[int]]
    dtype: Literal["float32"] = "float32"
    imputation_policy_id: Literal["impute-v1"] = "impute-v1"
    source_observation_id: ObservationId
    converter: ShortText
    max_event_step: int | None = Field(default=None, ge=0, le=10000)

    @model_validator(mode="after")
    def _validate_snapshot(self) -> "GraphSnapshot":
        n = len(self.node_order)
        e = len(self.edge_order)

        if tuple(self.node_feature_names) != NODE_FEATURE_NAMES_BASE_V1:
            raise ValueError("node_feature_names must equal the graph-base-v1 registry")
        if tuple(self.edge_feature_names) != EDGE_FEATURE_NAMES_BASE_V1:
            raise ValueError("edge_feature_names must equal the graph-base-v1 registry")

        node_cols = len(NODE_FEATURE_NAMES_BASE_V1)
        edge_cols = len(EDGE_FEATURE_NAMES_BASE_V1)
        _validate_float_matrix(self.node_features, n, node_cols, label="node_features")
        _validate_mask_matrix(self.node_mask, n, node_cols, label="node")
        _validate_float_matrix(self.edge_features, e, edge_cols, label="edge_features")
        _validate_mask_matrix(self.edge_mask, e, edge_cols, label="edge")

        # Imputed entries (mask 0) must be exactly 0.0 under impute-v1.
        self._check_imputed_zero(self.node_features, self.node_mask, label="node")
        self._check_imputed_zero(self.edge_features, self.edge_mask, label="edge")

        if len(self.edge_index) != 2:
            raise ValueError("edge_index must be [sources, targets]")
        for axis in self.edge_index:
            if len(axis) != e:
                raise ValueError("each edge_index row must have length E")
            for index in axis:
                if not (0 <= index < n):
                    raise ValueError("edge_index values must be indices into node_order")

        if self.max_event_step is not None and self.max_event_step > self.step:
            raise ValueError("max_event_step must be <= step")
        return self

    @staticmethod
    def _check_imputed_zero(
        matrix: list[list[float]],
        mask: list[list[int]],
        *,
        label: str,
    ) -> None:
        for row_values, row_mask in zip(matrix, mask):
            for value, flag in zip(row_values, row_mask):
                if flag == 0 and value != 0.0:
                    raise ValueError(
                        f"{label} imputed entries (mask 0) must equal 0.0"
                    )


class GraphFeatureRecord(ContractModel):
    """Hand-engineered graph-derived features (CONTRACTS section 10.2)."""

    schema_version: Literal["1.0"] = "1.0"
    feature_record_id: FeatureRecordId
    episode_id: EpisodeId
    step: int = Field(ge=0, le=10000)
    agent_id: AgentId
    topology_id: TopologyId
    snapshot_id: SnapshotId
    feature_set_id: Literal["graph-derived-v1"] = "graph-derived-v1"
    node_order: list[NodeId]
    feature_names: list[str]
    values: list[list[float]]
    mask: list[list[int]]

    @model_validator(mode="after")
    def _validate_features(self) -> "GraphFeatureRecord":
        if tuple(self.feature_names) != DERIVED_FEATURE_NAMES_V1:
            raise ValueError("feature_names must equal the graph-derived-v1 registry")
        n = len(self.node_order)
        cols = len(DERIVED_FEATURE_NAMES_V1)
        _validate_float_matrix(self.values, n, cols, label="values", unit_range=True)
        _validate_mask_matrix(self.mask, n, cols, label="feature")
        return self


class GraphModelOutput(ContractModel):
    """Optional learned-graph output; null payload when no model ran (10.3)."""

    schema_version: Literal["1.0"] = "1.0"
    graph_model_output_id: GraphModelOutputId
    episode_id: EpisodeId
    step: int = Field(ge=0, le=10000)
    agent_id: AgentId
    snapshot_id: SnapshotId
    status: Literal["computed", "not_computed"]
    model_id: ShortText | None
    output_kind: Literal["anomaly_score", "embedding"] | None
    node_order: list[NodeId] | None
    values: list[list[float]] | None

    @model_validator(mode="after")
    def _validate_model_output(self) -> "GraphModelOutput":
        payload = (self.model_id, self.output_kind, self.node_order, self.values)
        if self.status == "not_computed":
            if any(field is not None for field in payload):
                raise ValueError(
                    "a not_computed graph model output must have null model_id, "
                    "output_kind, node_order and values"
                )
            return self

        # status == "computed"
        if any(field is None for field in payload):
            raise ValueError(
                "a computed graph model output must have non-null model_id, "
                "output_kind, node_order and values"
            )
        assert self.node_order is not None and self.values is not None
        if len(self.values) != len(self.node_order):
            raise ValueError("values must have one row per node in node_order")
        if self.output_kind == "anomaly_score":
            for row in self.values:
                if len(row) != 1:
                    raise ValueError("anomaly_score values must be N x 1")
                if not (0.0 <= row[0] <= 1.0):
                    raise ValueError("anomaly_score values must be in [0, 1]")
        return self
