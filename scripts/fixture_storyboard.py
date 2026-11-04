"""Literal storyboard data for the S1 CyberSwarm fixtures (synthetic, neutral).

This module is the single source of the *values* from which ``build_fixtures.py``
writes every stage-S1 fixture. It is deliberately a pure data source:

* It imports **only** :mod:`cyberswarm.contracts` and the standard library. It
  does not import the mock simulator, the graph package, the coordinator or the
  evaluator, so the fixtures stay independent of the components they test.
* Every value is illustrative and hand-written. Nothing here is a research
  result or a simulator run; records that carry :class:`~cyberswarm.contracts.Provenance`
  are labelled ``fixture_synthetic``.

It exposes these families of fixtures (DESIGN section 8):

``valid_fixtures()``
    one valid example per S1 record, built through the real pydantic models so
    construction itself proves validity.
``invalid_fixtures()``
    one example per S1 record that breaks exactly one rule. These are raw
    payloads (an invalid record cannot be built through its model); each is
    annotated with the single reason it must fail.
``mini_episode()``
    a small, internally consistent episode (healthy topology, a short event
    stream, observations, available actions, detection summaries, node scores,
    a proposal/validation/execution chain that includes one rejected proposal,
    an evaluator-only ground-truth stream and an episode result) used for the
    cross-record consistency checks.
``graph_observation_inputs()``
    the two agent-visible ``agent-dmz`` observations (step 5 and step 6) that
    the S3 graph fixtures are *derived* from. These are literal, contracts-only
    inputs; the derivation through the real converter lives in the separate
    ``graph_fixtures`` builder so this storyboard never imports the graph
    package. Step 5 is the committed ``AgentObservation.dmz-step5`` fixture;
    step 6 is an internal input (not itself a committed fixture) that shows
    changed dynamic node and edge state at an identical topology ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cyberswarm.contracts import (
    ACTION_DEFS,
    ActionKind,
    ActionParameters,
    ActionSpec,
    AgentObservation,
    AlertCategory,
    AlertTruth,
    AvailableAction,
    AvailableActions,
    CalibrationReport,
    CalibrationStatus,
    Component,
    CompromiseCause,
    ContractModel,
    Counts,
    Decision,
    DetectionNode,
    DetectionSummary,
    Edge,
    EdgeObservation,
    EdgeState,
    EdgeType,
    EpisodeResult,
    ErrorCode,
    ErrorRecord,
    ErrorSeverity,
    EventProducer,
    EventRecord,
    EventType,
    ExecutedAction,
    ExecutionStatus,
    GraphModelOutput,
    GroundTruthNode,
    GroundTruthSnapshot,
    ImpactClass,
    IncidentSpec,
    LogLevel,
    LogRecord,
    NewCompromise,
    Node,
    NodeObservation,
    NodeScope,
    NodeScore,
    NodeScoreRecord,
    NodeScoring,
    NodeStatus,
    NodeType,
    ObservabilitySpec,
    OutcomeSummary,
    Phase,
    PolicyProposal,
    ProposalFeedback,
    Provenance,
    ProvenanceKind,
    RationaleCode,
    ReasonCode,
    Representation,
    RuleOutcome,
    ScenarioConfig,
    ScorerKind,
    Severity,
    SimulatorKind,
    Split,
    StatusChange,
    TerminationReason,
    ThresholdSelection,
    Topology,
    UncertaintyInfo,
    ValidationResult,
    available_actions_id,
    detection_id,
    make_episode_id,
    node_score_id,
    observation_id,
)

# --------------------------------------------------------------------------- #
# Shared synthetic constants
# --------------------------------------------------------------------------- #
#: Episode id used by the single-record demo fixtures (the CONTRACTS examples).
DEMO_EPISODE = "ep-rule-conv-v1-mock-lateral-eval-01-s2001"
#: A short, repository-relative provenance note for records that carry one.
SYNTHETIC_NOTE = "Illustrative values, not a result"
ALLOWLIST_ID = "allowlist-default-v1"
DETECTION_SCORER = "detection.rule-v1"

# Suspicion values of the detection formula, written as literals so the data
# stays independent of any component (CONTRACTS section 8.4). They match
# min(1, 0.5 * min(alerts, 5) / 5 + 0.5 * severity / 3).
SUSPICION_A1_S1 = 0.26666666666666666  # one alert, severity 1
SUSPICION_A1_S2 = 0.43333333333333335  # one alert, severity 2


def _fixture_provenance() -> Provenance:
    return Provenance(kind=ProvenanceKind.fixture_synthetic, note=SYNTHETIC_NOTE)


# --------------------------------------------------------------------------- #
# Fixture specifications
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ValidFixture:
    """A valid single-record fixture: a repository-relative path and a model."""

    rel_path: str
    model: ContractModel


@dataclass(frozen=True, slots=True)
class InvalidFixture:
    """An invalid single-record fixture that must fail for exactly one reason."""

    rel_path: str
    record_type: str
    payload: dict[str, Any]
    reason: str


# --------------------------------------------------------------------------- #
# Valid single-record fixtures (CONTRACTS examples, built through the models)
# --------------------------------------------------------------------------- #
def _valid_node() -> Node:
    return Node(
        node_id="node-web-02",
        node_type=NodeType.web,
        zone_id="zone-dmz",
        criticality=0.2,
        label="Public web server B",
    )


def _valid_edge() -> Edge:
    return Edge(
        edge_id="edge-014",
        source="node-web-01",
        target="node-app-01",
        edge_type=EdgeType.network_link,
    )


def _valid_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="scn-mock-lateral-eval-01",
        topology_id="topo-mock-small-v1",
        description="Synthetic propagating incident entering at node-web-02",
        max_steps=40,
        observation_window_steps=5,
        incident=IncidentSpec(
            entry_node_id="node-web-02",
            start_step=3,
            spread_probability=0.25,
        ),
        observability=ObservabilitySpec(
            alert_detect_probability=0.35,
            false_alert_rate=0.02,
            alert_delay_max_steps=1,
            missing_value_rate=0.05,
        ),
        restore_duration_steps=3,
        objective_node_id="node-db-01",
    )


def _alert_event() -> EventRecord:
    return EventRecord(
        event_id="evt-000014",
        episode_id=DEMO_EPISODE,
        step=5,
        phase=Phase.dynamics,
        event_type=EventType.alert_raised,
        producer=EventProducer.simulator,
        visible_to=["agent-core", "agent-dmz"],
        node_id="node-web-02",
        severity=Severity.medium,
        alert_category=AlertCategory.traffic_anomaly,
    )


def _action_applied_event() -> EventRecord:
    return EventRecord(
        event_id="evt-000018",
        episode_id=DEMO_EPISODE,
        step=6,
        phase=Phase.outcome,
        event_type=EventType.action_applied,
        producer=EventProducer.simulator,
        visible_to=["agent-dmz"],
        node_id="node-web-02",
        agent_id="agent-dmz",
        action_kind=ActionKind.isolate_node,
        proposal_id="prop-000014",
        execution_id="exe-000013",
    )


def _valid_observation() -> AgentObservation:
    return AgentObservation(
        observation_id="obs-0005-agent-dmz",
        episode_id=DEMO_EPISODE,
        step=5,
        agent_id="agent-dmz",
        zone_id="zone-dmz",
        max_steps=40,
        observation_window_steps=5,
        nodes=[
            NodeObservation(
                node_id="node-app-01",
                scope=NodeScope.boundary,
                visible_status=NodeStatus.up,
                alert_count_recent=0,
                max_alert_severity=0,
            ),
            NodeObservation(
                node_id="node-gw-01",
                scope=NodeScope.own,
                visible_status=NodeStatus.up,
                alert_count_recent=1,
                max_alert_severity=1,
                evidence_level=0,
                service_health=0.97,
                steps_since_investigated=3,
            ),
            NodeObservation(
                node_id="node-web-01",
                scope=NodeScope.own,
                visible_status=NodeStatus.up,
                alert_count_recent=0,
                max_alert_severity=0,
                service_health=0.99,
            ),
            NodeObservation(
                node_id="node-web-02",
                scope=NodeScope.own,
                visible_status=NodeStatus.up,
                alert_count_recent=1,
                max_alert_severity=2,
                service_health=0.88,
            ),
        ],
        edges=[
            EdgeObservation(
                edge_id="edge-014",
                edge_state=EdgeState.active,
                traffic_level=0.52,
            ),
        ],
        recent_events=[_alert_event()],
        last_step_feedback=[],
    )


def _valid_available_actions() -> AvailableActions:
    return AvailableActions(
        available_actions_id="avl-0005-agent-dmz",
        episode_id=DEMO_EPISODE,
        step=5,
        agent_id="agent-dmz",
        actions=[
            _available(ActionKind.no_op),
            _available(ActionKind.investigate_node, "node-web-02"),
            _available(ActionKind.isolate_node, "node-web-02"),
        ],
    )


def _valid_detection() -> DetectionSummary:
    return DetectionSummary(
        detection_id="det-0005-agent-dmz",
        episode_id=DEMO_EPISODE,
        step=5,
        agent_id="agent-dmz",
        producer=DETECTION_SCORER,
        nodes=[
            DetectionNode(node_id="node-app-01", suspicion=0.0, alerted=False),
            DetectionNode(
                node_id="node-gw-01", suspicion=SUSPICION_A1_S1, alerted=True
            ),
            DetectionNode(node_id="node-web-01", suspicion=0.0, alerted=False),
            DetectionNode(
                node_id="node-web-02", suspicion=SUSPICION_A1_S2, alerted=True
            ),
        ],
    )


def _valid_proposal_investigate() -> PolicyProposal:
    return PolicyProposal(
        proposal_id="prop-000012",
        episode_id=DEMO_EPISODE,
        step=5,
        agent_id="agent-dmz",
        policy_id="pol-rule-conv-v1",
        observation_id="obs-0005-agent-dmz",
        action=ActionSpec(
            action_kind=ActionKind.investigate_node, target_node_id="node-web-02"
        ),
        rationale_codes=[
            RationaleCode.alerts_present,
            RationaleCode.evidence_insufficient,
        ],
        rationale_text="Alert on node-web-02 with no evidence yet",
    )


def _valid_proposal_uncertainty() -> PolicyProposal:
    return PolicyProposal(
        proposal_id="prop-000015",
        episode_id=DEMO_EPISODE,
        step=5,
        agent_id="agent-dmz",
        policy_id="pol-ippo-conv-s0",
        observation_id="obs-0005-agent-dmz",
        action=ActionSpec(
            action_kind=ActionKind.request_evidence, target_node_id="node-web-02"
        ),
        rationale_codes=[
            RationaleCode.alerts_present,
            RationaleCode.uncertainty_evidence_request,
        ],
        rationale_text="High uncertainty, requesting more evidence before containment",
        action_probability=0.41,
        uncertainty=UncertaintyInfo(
            value=0.62,
            rule_id="uncertainty-rule-v1",
            rule_outcome=RuleOutcome.evidence_request,
            original_action=ActionSpec(
                action_kind=ActionKind.isolate_node, target_node_id="node-web-02"
            ),
        ),
    )


def _valid_validation_accepted() -> ValidationResult:
    return ValidationResult(
        validation_id="val-000012",
        episode_id=DEMO_EPISODE,
        step=5,
        proposal_id="prop-000012",
        agent_id="agent-dmz",
        decision=Decision.accepted,
        reason_codes=[],
        resolved_action=ActionSpec(
            action_kind=ActionKind.investigate_node, target_node_id="node-web-02"
        ),
        allowlist_id=ALLOWLIST_ID,
    )


def _valid_validation_rejected() -> ValidationResult:
    return ValidationResult(
        validation_id="val-000013",
        episode_id=DEMO_EPISODE,
        step=6,
        proposal_id="prop-000013",
        agent_id="agent-core",
        decision=Decision.rejected,
        reason_codes=[ReasonCode.TARGET_OUT_OF_AUTHORITY],
        resolved_action=None,
        allowlist_id=ALLOWLIST_ID,
    )


def _valid_execution_isolate() -> ExecutedAction:
    return ExecutedAction(
        execution_id="exe-000013",
        episode_id=DEMO_EPISODE,
        step=6,
        proposal_id="prop-000014",
        validation_id="val-000014",
        agent_id="agent-dmz",
        action=ActionSpec(
            action_kind=ActionKind.isolate_node, target_node_id="node-web-02"
        ),
        status=ExecutionStatus.executed,
        error_code=None,
        affected_node_ids=["node-web-02"],
        status_changes=[
            StatusChange(
                node_id="node-web-02",
                status_from=NodeStatus.up,
                status_to=NodeStatus.isolated,
            )
        ],
        outcome_event_ids=["evt-000018", "evt-000019"],
        state_digest_before="a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90",
        state_digest_after="0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0",
    )


def _valid_execution_failed() -> ExecutedAction:
    digest = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"
    return ExecutedAction(
        execution_id="exe-000020",
        episode_id=DEMO_EPISODE,
        step=9,
        proposal_id="prop-000021",
        validation_id="val-000021",
        agent_id="agent-core",
        action=ActionSpec(
            action_kind=ActionKind.restore_node, target_node_id="node-app-01"
        ),
        status=ExecutionStatus.failed,
        error_code=ErrorCode.ACTION_EXECUTION_FAILED,
        affected_node_ids=[],
        status_changes=[],
        outcome_event_ids=[],
        state_digest_before=digest,
        state_digest_after=digest,
    )


def _valid_ground_truth() -> GroundTruthSnapshot:
    return GroundTruthSnapshot(
        episode_id=DEMO_EPISODE,
        step=5,
        incident_active=True,
        nodes=[
            GroundTruthNode(
                node_id="node-web-02",
                compromised=True,
                service_availability=0.9,
                defense_availability=1.0,
                contained=False,
                status=NodeStatus.up,
            )
        ],
        new_compromises=[],
        alert_truth=[AlertTruth(event_id="evt-000014", true_positive=True)],
    )


def _valid_graph_model_output() -> GraphModelOutput:
    return GraphModelOutput(
        graph_model_output_id="gmo-0005-agent-dmz",
        episode_id=DEMO_EPISODE,
        step=5,
        agent_id="agent-dmz",
        snapshot_id="gsn-0005-agent-dmz",
        status="not_computed",
        model_id=None,
        output_kind=None,
        node_order=None,
        values=None,
    )


def _valid_error_record() -> ErrorRecord:
    return ErrorRecord(
        error_id="err-000001",
        run_id="run-smoke",
        episode_id=DEMO_EPISODE,
        step=5,
        component=Component.coordinator,
        code=ErrorCode.SCHEMA_INVALID,
        severity=ErrorSeverity.warning,
        recoverable=True,
        message="Proposal rejected, not parseable as a PolicyProposal",
    )


def _valid_log_record() -> LogRecord:
    return LogRecord(
        ts="2026-10-30T20:00:00Z",
        level=LogLevel.info,
        component=Component.runner,
        run_id="run-smoke",
        episode_id=None,
        step=None,
        code=None,
        message="Run started",
    )


def _valid_calibration_report() -> CalibrationReport:
    return CalibrationReport(
        report_id="cbr-fixture-demo",
        run_id="run-fixture-demo",
        policy_id="pol-ippo-conv-s0-ctl",
        eval_split="eval",
        n_cal=0,
        n_eval=0,
        bins=[],
        ece_raw=None,
        ece_calibrated=None,
        mce=None,
        brier=None,
        auroc=None,
        auroc_ci_low=None,
        auroc_ci_high=None,
        status=CalibrationStatus.insufficient_data,
        tau_evidence=0.5,
        tau_defer=0.3,
        threshold_selection=ThresholdSelection.default,
        provenance=_fixture_provenance(),
    )


def _available(kind: ActionKind, target: str | None = None) -> AvailableAction:
    """Build an :class:`AvailableAction` with vocabulary-correct metadata."""

    spec = ACTION_DEFS[kind]
    return AvailableAction(
        action_kind=kind,
        target_node_id=target,
        parameters=ActionParameters(),
        impact_class=spec.impact_class,
        high_impact=spec.high_impact,
        cost_units=spec.cost_units,
    )


def valid_fixtures() -> list[ValidFixture]:
    """Return every valid S1 single-record fixture (path and model)."""

    root = "fixtures/records/valid"
    return [
        ValidFixture(f"{root}/Node.web-02.json", _valid_node()),
        ValidFixture(f"{root}/Edge.link.json", _valid_edge()),
        ValidFixture(f"{root}/ScenarioConfig.eval-01.json", _valid_scenario()),
        ValidFixture(f"{root}/EventRecord.alert.json", _alert_event()),
        ValidFixture(
            f"{root}/EventRecord.action_applied.json", _action_applied_event()
        ),
        ValidFixture(f"{root}/AgentObservation.dmz-step5.json", _valid_observation()),
        ValidFixture(
            f"{root}/AvailableActions.dmz-step5.json", _valid_available_actions()
        ),
        ValidFixture(f"{root}/DetectionSummary.dmz-step5.json", _valid_detection()),
        ValidFixture(
            f"{root}/PolicyProposal.investigate.json", _valid_proposal_investigate()
        ),
        ValidFixture(
            f"{root}/PolicyProposal.uncertainty-evidence-request.json",
            _valid_proposal_uncertainty(),
        ),
        ValidFixture(
            f"{root}/ValidationResult.accepted.json", _valid_validation_accepted()
        ),
        ValidFixture(
            f"{root}/ValidationResult.rejected.json", _valid_validation_rejected()
        ),
        ValidFixture(f"{root}/ExecutedAction.isolate.json", _valid_execution_isolate()),
        ValidFixture(f"{root}/ExecutedAction.failed.json", _valid_execution_failed()),
        ValidFixture(f"{root}/GroundTruthSnapshot.step5.json", _valid_ground_truth()),
        ValidFixture(
            f"{root}/GraphModelOutput.not_computed.json", _valid_graph_model_output()
        ),
        ValidFixture(f"{root}/ErrorRecord.coordinator.json", _valid_error_record()),
        ValidFixture(f"{root}/LogRecord.info.json", _valid_log_record()),
        ValidFixture(
            f"{root}/CalibrationReport.insufficient.json",
            _valid_calibration_report(),
        ),
    ]


# --------------------------------------------------------------------------- #
# Graph observation inputs (literal; the S3 graph fixtures derive from these)
# --------------------------------------------------------------------------- #
def _graph_step6_isolation_event() -> EventRecord:
    """The ``node_status_changed`` event for node-web-02 isolated at step 6."""

    return EventRecord(
        event_id="evt-000020",
        episode_id=DEMO_EPISODE,
        step=6,
        phase=Phase.outcome,
        event_type=EventType.node_status_changed,
        producer=EventProducer.simulator,
        visible_to=["agent-core", "agent-dmz"],
        node_id="node-web-02",
        status_from=NodeStatus.up,
        status_to=NodeStatus.isolated,
        execution_id="exe-000013",
    )


def _graph_observation_step6() -> AgentObservation:
    """The agent-dmz observation one step after ``_valid_observation`` (step 6).

    Same four observed nodes and the same topology as step 5, with changed
    dynamic state only: node-web-02 has now been investigated (evidence 3), its
    service health has dropped, and it has just been isolated; the gateway alert
    has aged out of the window; and two of node-web-02's link edges now read as
    blocked. node-app-01 stays a boundary node with no private fields (the
    hidden spread to it is not visible to agent-dmz), so the derived snapshot
    carries no evaluator-only data. The node and edge ordering of the derived
    snapshot is the topology order, identical to the step-5 snapshot.
    """

    return AgentObservation(
        observation_id=observation_id(6, "agent-dmz"),
        episode_id=DEMO_EPISODE,
        step=6,
        agent_id="agent-dmz",
        zone_id="zone-dmz",
        max_steps=40,
        observation_window_steps=5,
        nodes=[
            NodeObservation(
                node_id="node-app-01",
                scope=NodeScope.boundary,
                visible_status=NodeStatus.up,
                alert_count_recent=0,
                max_alert_severity=0,
            ),
            NodeObservation(
                node_id="node-gw-01",
                scope=NodeScope.own,
                visible_status=NodeStatus.up,
                alert_count_recent=0,
                max_alert_severity=0,
                evidence_level=0,
                service_health=0.98,
                steps_since_investigated=4,
            ),
            NodeObservation(
                node_id="node-web-01",
                scope=NodeScope.own,
                visible_status=NodeStatus.up,
                alert_count_recent=0,
                max_alert_severity=0,
                service_health=0.99,
            ),
            NodeObservation(
                node_id="node-web-02",
                scope=NodeScope.own,
                visible_status=NodeStatus.isolated,
                alert_count_recent=1,
                max_alert_severity=2,
                evidence_level=3,
                service_health=0.82,
                steps_since_investigated=0,
            ),
        ],
        edges=[
            EdgeObservation(
                edge_id="edge-013", edge_state=EdgeState.blocked, traffic_level=None
            ),
            EdgeObservation(
                edge_id="edge-014", edge_state=EdgeState.active, traffic_level=0.48
            ),
            EdgeObservation(
                edge_id="edge-018", edge_state=EdgeState.blocked, traffic_level=None
            ),
        ],
        recent_events=[_alert_event(), _graph_step6_isolation_event()],
        last_step_feedback=[],
    )


def graph_observation_inputs() -> tuple[AgentObservation, AgentObservation]:
    """Return the (step 5, step 6) agent-dmz observations the graph fixtures use.

    Step 5 is the exact model behind the committed ``AgentObservation.dmz-step5``
    fixture, so the derived step-5 snapshot matches that committed observation.
    Step 6 is an internal input (not a committed fixture) that advances the
    dynamic state while keeping the same topology.
    """

    return _valid_observation(), _graph_observation_step6()


# --------------------------------------------------------------------------- #
# Invalid single-record fixtures (raw payloads, one broken rule each)
# --------------------------------------------------------------------------- #
def _dump(model: ContractModel) -> dict[str, Any]:
    """Serialize a valid base record to a JSON-ready dict for mutation."""

    return model.model_dump(mode="json")


def _invalid_action_unknown_kind() -> dict[str, Any]:
    payload = _dump(
        ActionSpec(action_kind=ActionKind.isolate_node, target_node_id="node-web-02")
    )
    payload["action_kind"] = "firewall_node"
    return payload


def _invalid_action_raw_ip() -> dict[str, Any]:
    payload = _dump(
        ActionSpec(action_kind=ActionKind.isolate_node, target_node_id="node-web-02")
    )
    payload["target_node_id"] = "10.0.0.5"
    return payload


def _invalid_action_ip_lookalike() -> dict[str, Any]:
    payload = _dump(
        ActionSpec(action_kind=ActionKind.isolate_node, target_node_id="node-web-02")
    )
    payload["target_node_id"] = "node-10-0-0-1"
    return payload


def _invalid_action_extra_field() -> dict[str, Any]:
    payload = _dump(
        ActionSpec(action_kind=ActionKind.isolate_node, target_node_id="node-web-02")
    )
    payload["command"] = "placeholder"
    return payload


def _invalid_action_wrong_parameter() -> dict[str, Any]:
    payload = _dump(
        ActionSpec(action_kind=ActionKind.monitor_node, target_node_id="node-web-02")
    )
    payload["parameters"]["level"] = "light"
    return payload


def _base_observation() -> AgentObservation:
    """A minimal valid observation used as a base for invalid mutations."""

    return AgentObservation(
        observation_id="obs-0005-agent-dmz",
        episode_id=DEMO_EPISODE,
        step=5,
        agent_id="agent-dmz",
        zone_id="zone-dmz",
        max_steps=40,
        observation_window_steps=5,
        nodes=[
            NodeObservation(
                node_id="node-app-01",
                scope=NodeScope.boundary,
                visible_status=NodeStatus.up,
                alert_count_recent=0,
                max_alert_severity=0,
            ),
            NodeObservation(
                node_id="node-web-02",
                scope=NodeScope.own,
                visible_status=NodeStatus.up,
                alert_count_recent=1,
                max_alert_severity=2,
                evidence_level=0,
                service_health=0.88,
                steps_since_investigated=2,
            ),
        ],
        edges=[],
        recent_events=[_alert_event()],
        last_step_feedback=[],
    )


def _invalid_observation_extra_ground_truth() -> dict[str, Any]:
    payload = _dump(_base_observation())
    payload["nodes"][1]["compromised"] = True
    return payload


def _invalid_observation_future_event() -> dict[str, Any]:
    payload = _dump(_base_observation())
    payload["recent_events"][0]["step"] = 6
    return payload


def _invalid_observation_boundary_private() -> dict[str, Any]:
    payload = _dump(_base_observation())
    payload["nodes"][0]["service_health"] = 0.5
    return payload


def _invalid_observation_nan() -> dict[str, Any]:
    payload = _dump(_base_observation())
    payload["nodes"][1]["service_health"] = float("nan")
    return payload


def _invalid_proposal_text(rationale_text: str) -> dict[str, Any]:
    payload = _dump(_valid_proposal_investigate())
    payload["rationale_text"] = rationale_text
    return payload


def _invalid_event_missing_node() -> dict[str, Any]:
    payload = _dump(_alert_event())
    payload["node_id"] = None
    return payload


def _invalid_validation_rejected_without_reason() -> dict[str, Any]:
    payload = _dump(_valid_validation_rejected())
    payload["reason_codes"] = []
    return payload


def _invalid_validation_accepted_without_action() -> dict[str, Any]:
    payload = _dump(_valid_validation_accepted())
    payload["resolved_action"] = None
    return payload


def _invalid_execution_failed_without_code() -> dict[str, Any]:
    payload = _dump(_valid_execution_failed())
    payload["error_code"] = None
    return payload


def _invalid_scenario_start_after_end() -> dict[str, Any]:
    payload = _dump(_valid_scenario())
    payload["max_steps"] = 5
    payload["incident"]["start_step"] = 5
    return payload


def _invalid_topology_dangling_edge() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "topology_id": "topo-fixture-bad-v1",
        "nodes": [
            {
                "node_id": "node-app-01",
                "node_type": "app",
                "zone_id": "zone-core",
                "criticality": 0.7,
                "label": None,
            },
            {
                "node_id": "node-web-01",
                "node_type": "web",
                "zone_id": "zone-dmz",
                "criticality": 0.5,
                "label": None,
            },
        ],
        "edges": [
            {
                "edge_id": "edge-001",
                "source": "node-app-01",
                "target": "node-ghost-01",
                "edge_type": "network_link",
            }
        ],
    }


def invalid_fixtures() -> list[InvalidFixture]:
    """Return every invalid S1 single-record fixture with its one failure."""

    root = "fixtures/records/invalid"
    return [
        InvalidFixture(
            f"{root}/ActionSpec.unknown-kind.json",
            "ActionSpec",
            _invalid_action_unknown_kind(),
            "action_kind is not in the vocabulary",
        ),
        InvalidFixture(
            f"{root}/ActionSpec.raw-ip-target.json",
            "ActionSpec",
            _invalid_action_raw_ip(),
            "target_node_id is a raw dotted-quad IP address",
        ),
        InvalidFixture(
            f"{root}/ActionSpec.ip-lookalike-target.json",
            "ActionSpec",
            _invalid_action_ip_lookalike(),
            "target_node_id looks like an IP address",
        ),
        InvalidFixture(
            f"{root}/ActionSpec.extra-command-field.json",
            "ActionSpec",
            _invalid_action_extra_field(),
            "unknown extra field command is forbidden",
        ),
        InvalidFixture(
            f"{root}/ActionSpec.wrong-parameter.json",
            "ActionSpec",
            _invalid_action_wrong_parameter(),
            "level parameter is not allowed for monitor_node",
        ),
        InvalidFixture(
            f"{root}/AgentObservation.extra-ground-truth-field.json",
            "AgentObservation",
            _invalid_observation_extra_ground_truth(),
            "extra evaluator-only field compromised is forbidden",
        ),
        InvalidFixture(
            f"{root}/AgentObservation.future-event.json",
            "AgentObservation",
            _invalid_observation_future_event(),
            "recent event has a step greater than the observation step",
        ),
        InvalidFixture(
            f"{root}/AgentObservation.boundary-private-value.json",
            "AgentObservation",
            _invalid_observation_boundary_private(),
            "boundary node carries a non-null service_health",
        ),
        InvalidFixture(
            f"{root}/AgentObservation.nan-service-health.json",
            "AgentObservation",
            _invalid_observation_nan(),
            "service_health is NaN (non-finite float)",
        ),
        InvalidFixture(
            f"{root}/PolicyProposal.unsafe-rationale-text.json",
            "PolicyProposal",
            _invalid_proposal_text("escalate; page oncall"),
            "rationale_text uses a character outside ShortText",
        ),
        InvalidFixture(
            f"{root}/PolicyProposal.ip-in-rationale.json",
            "PolicyProposal",
            _invalid_proposal_text("Block scanner at 10.0.0.5"),
            "rationale_text contains a dotted-quad IP",
        ),
        InvalidFixture(
            f"{root}/PolicyProposal.path-in-rationale.json",
            "PolicyProposal",
            _invalid_proposal_text("Read config at ..etc_passwd"),
            "rationale_text contains the sequence .. (path traversal)",
        ),
        InvalidFixture(
            f"{root}/EventRecord.alert-missing-node.json",
            "EventRecord",
            _invalid_event_missing_node(),
            "alert_raised is missing its required node_id",
        ),
        InvalidFixture(
            f"{root}/ValidationResult.rejected-without-reason.json",
            "ValidationResult",
            _invalid_validation_rejected_without_reason(),
            "rejected validation has an empty reason_codes list",
        ),
        InvalidFixture(
            f"{root}/ValidationResult.accepted-without-action.json",
            "ValidationResult",
            _invalid_validation_accepted_without_action(),
            "accepted validation has a null resolved_action",
        ),
        InvalidFixture(
            f"{root}/ExecutedAction.failed-without-error-code.json",
            "ExecutedAction",
            _invalid_execution_failed_without_code(),
            "failed execution has a null error_code",
        ),
        InvalidFixture(
            f"{root}/ScenarioConfig.start-after-end.json",
            "ScenarioConfig",
            _invalid_scenario_start_after_end(),
            "incident.start_step is not less than max_steps",
        ),
        InvalidFixture(
            f"{root}/Topology.dangling-edge.json",
            "Topology",
            _invalid_topology_dangling_edge(),
            "edge references an unknown endpoint node",
        ),
    ]


# --------------------------------------------------------------------------- #
# Primary mini-episode (internally consistent, for cross-record checks)
# --------------------------------------------------------------------------- #
MINI_TOPOLOGY_ID = "topo-fixture-mini-v1"
MINI_SCENARIO_ID = "scn-example-01"
MINI_POLICY_ID = "pol-rule-conv-v1"
MINI_RUN_ID = "run-fixture-mini"
MINI_SEED = 7
MINI_EPISODE_ID = make_episode_id(MINI_POLICY_ID, MINI_SCENARIO_ID, MINI_SEED)
MINI_HORIZON = 3
MINI_WINDOW = 3

AGENT_CORE = "agent-core"
AGENT_DMZ = "agent-dmz"
ZONE_CORE = "zone-core"
ZONE_DMZ = "zone-dmz"
NODE_APP = "node-app-01"
NODE_DB = "node-db-01"
NODE_WEB = "node-web-01"


@dataclass(frozen=True, slots=True)
class MiniEpisode:
    """A small, internally consistent episode across every agent-visible record."""

    topology: Topology
    events: list[EventRecord]
    observations: list[AgentObservation]
    available_actions: list[AvailableActions]
    detection: list[DetectionSummary]
    node_scores: list[NodeScoreRecord]
    proposals: list[PolicyProposal]
    validations: list[ValidationResult]
    executions: list[ExecutedAction]
    ground_truth: list[GroundTruthSnapshot]
    episode_result: EpisodeResult


def _mini_topology() -> Topology:
    return Topology(
        topology_id=MINI_TOPOLOGY_ID,
        nodes=[
            Node(
                node_id=NODE_APP,
                node_type=NodeType.app,
                zone_id=ZONE_CORE,
                criticality=0.7,
                label="Application server",
            ),
            Node(
                node_id=NODE_DB,
                node_type=NodeType.database,
                zone_id=ZONE_CORE,
                criticality=1.0,
                label=None,
            ),
            Node(
                node_id=NODE_WEB,
                node_type=NodeType.web,
                zone_id=ZONE_DMZ,
                criticality=0.5,
                label="Public web server",
            ),
        ],
        edges=[
            Edge(
                edge_id="edge-001",
                source=NODE_APP,
                target=NODE_DB,
                edge_type=EdgeType.network_link,
            ),
            Edge(
                edge_id="edge-002",
                source=NODE_APP,
                target=NODE_DB,
                edge_type=EdgeType.service_dependency,
            ),
            Edge(
                edge_id="edge-003",
                source=NODE_APP,
                target=NODE_WEB,
                edge_type=EdgeType.network_link,
            ),
            Edge(
                edge_id="edge-004",
                source=NODE_DB,
                target=NODE_APP,
                edge_type=EdgeType.network_link,
            ),
            Edge(
                edge_id="edge-005",
                source=NODE_WEB,
                target=NODE_APP,
                edge_type=EdgeType.network_link,
            ),
            Edge(
                edge_id="edge-006",
                source=NODE_WEB,
                target=NODE_APP,
                edge_type=EdgeType.service_dependency,
            ),
        ],
    )


def _node_obs(
    node_id: str,
    scope: NodeScope,
    *,
    status: NodeStatus = NodeStatus.up,
    alerts: int = 0,
    severity: int = 0,
    evidence: int | None = None,
    health: float | None = None,
    since: int | None = None,
) -> NodeObservation:
    return NodeObservation(
        node_id=node_id,
        scope=scope,
        visible_status=status,
        alert_count_recent=alerts,
        max_alert_severity=severity,
        evidence_level=evidence,
        service_health=health,
        steps_since_investigated=since,
    )


def _core_edges() -> list[EdgeObservation]:
    return [
        EdgeObservation(edge_id=eid, edge_state=EdgeState.active, traffic_level=None)
        for eid in ("edge-001", "edge-002", "edge-003", "edge-004", "edge-005", "edge-006")
    ]


def _dmz_edges() -> list[EdgeObservation]:
    return [
        EdgeObservation(edge_id=eid, edge_state=EdgeState.active, traffic_level=None)
        for eid in ("edge-003", "edge-005", "edge-006")
    ]


def _observation(
    step: int,
    agent_id: str,
    zone_id: str,
    nodes: list[NodeObservation],
    edges: list[EdgeObservation],
    events: list[EventRecord],
    feedback: list[ProposalFeedback],
) -> AgentObservation:
    return AgentObservation(
        observation_id=observation_id(step, agent_id),
        episode_id=MINI_EPISODE_ID,
        step=step,
        agent_id=agent_id,
        zone_id=zone_id,
        max_steps=MINI_HORIZON,
        observation_window_steps=MINI_WINDOW,
        nodes=nodes,
        edges=edges,
        recent_events=events,
        last_step_feedback=feedback,
    )


def _event(
    counter: int,
    step: int,
    phase: Phase,
    event_type: EventType,
    visible_to: list[str],
    *,
    producer: EventProducer = EventProducer.simulator,
    **fields: Any,
) -> EventRecord:
    prefix = "cev" if producer is EventProducer.coordinator else "evt"
    return EventRecord(
        event_id=f"{prefix}-{counter:06d}",
        episode_id=MINI_EPISODE_ID,
        step=step,
        phase=phase,
        event_type=event_type,
        producer=producer,
        visible_to=visible_to,
        **fields,
    )


def _mini_events() -> list[EventRecord]:
    both = [AGENT_CORE, AGENT_DMZ]
    return [
        _event(1, 0, Phase.lifecycle, EventType.episode_started, both),
        _event(
            2,
            1,
            Phase.dynamics,
            EventType.alert_raised,
            both,
            node_id=NODE_WEB,
            severity=Severity.medium,
            alert_category=AlertCategory.traffic_anomaly,
        ),
        _event(
            3,
            1,
            Phase.outcome,
            EventType.action_applied,
            [AGENT_DMZ],
            node_id=NODE_WEB,
            agent_id=AGENT_DMZ,
            action_kind=ActionKind.investigate_node,
            proposal_id="prop-000004",
            execution_id="exe-000003",
        ),
        _event(
            4,
            1,
            Phase.outcome,
            EventType.investigation_result,
            [AGENT_DMZ],
            node_id=NODE_WEB,
            evidence_level=2,
            agent_id=AGENT_DMZ,
            execution_id="exe-000003",
        ),
        _event(
            5,
            2,
            Phase.outcome,
            EventType.action_applied,
            [AGENT_DMZ],
            node_id=NODE_WEB,
            agent_id=AGENT_DMZ,
            action_kind=ActionKind.isolate_node,
            proposal_id="prop-000006",
            execution_id="exe-000005",
        ),
        _event(
            6,
            2,
            Phase.outcome,
            EventType.node_status_changed,
            both,
            node_id=NODE_WEB,
            status_from=NodeStatus.up,
            status_to=NodeStatus.isolated,
            execution_id="exe-000005",
        ),
        _event(
            7,
            2,
            Phase.lifecycle,
            EventType.episode_ended,
            both,
            terminated=False,
            truncated=True,
        ),
    ]


def _mini_observations(events: list[EventRecord]) -> list[AgentObservation]:
    by_id = {event.event_id: event for event in events}

    def pick(*event_ids: str) -> list[EventRecord]:
        return [by_id[event_id] for event_id in event_ids]

    accepted_executed = ProposalFeedback(
        proposal_id="prop-000001",
        decision=Decision.accepted,
        reason_codes=[],
        execution_status=ExecutionStatus.executed,
    )
    return [
        # Step 0 - healthy network.
        _observation(
            0,
            AGENT_CORE,
            ZONE_CORE,
            [
                _node_obs(NODE_APP, NodeScope.own, health=1.0),
                _node_obs(NODE_DB, NodeScope.own, health=1.0),
                _node_obs(NODE_WEB, NodeScope.boundary),
            ],
            _core_edges(),
            pick("evt-000001"),
            [],
        ),
        _observation(
            0,
            AGENT_DMZ,
            ZONE_DMZ,
            [
                _node_obs(NODE_APP, NodeScope.boundary),
                _node_obs(NODE_WEB, NodeScope.own, health=1.0),
            ],
            _dmz_edges(),
            pick("evt-000001"),
            [],
        ),
        # Step 1 - alert raised on the web node; dmz investigates.
        _observation(
            1,
            AGENT_CORE,
            ZONE_CORE,
            [
                _node_obs(NODE_APP, NodeScope.own, health=1.0),
                _node_obs(NODE_DB, NodeScope.own, health=1.0),
                _node_obs(NODE_WEB, NodeScope.boundary, alerts=1, severity=2),
            ],
            _core_edges(),
            pick("evt-000001", "evt-000002"),
            [
                ProposalFeedback(
                    proposal_id="prop-000001",
                    decision=Decision.accepted,
                    reason_codes=[],
                    execution_status=ExecutionStatus.executed,
                )
            ],
        ),
        _observation(
            1,
            AGENT_DMZ,
            ZONE_DMZ,
            [
                _node_obs(NODE_APP, NodeScope.boundary),
                _node_obs(
                    NODE_WEB, NodeScope.own, alerts=1, severity=2, health=0.95
                ),
            ],
            _dmz_edges(),
            pick("evt-000001", "evt-000002"),
            [
                ProposalFeedback(
                    proposal_id="prop-000002",
                    decision=Decision.accepted,
                    reason_codes=[],
                    execution_status=ExecutionStatus.executed,
                )
            ],
        ),
        # Step 2 - dmz contains the web node; core's faulty proposal was rejected.
        _observation(
            2,
            AGENT_CORE,
            ZONE_CORE,
            [
                _node_obs(NODE_APP, NodeScope.own, health=1.0),
                _node_obs(NODE_DB, NodeScope.own, health=1.0),
                _node_obs(NODE_WEB, NodeScope.boundary, alerts=1, severity=2),
            ],
            _core_edges(),
            pick("evt-000001", "evt-000002"),
            [
                ProposalFeedback(
                    proposal_id="prop-000003",
                    decision=Decision.rejected,
                    reason_codes=[ReasonCode.TARGET_OUT_OF_AUTHORITY],
                    execution_status=None,
                )
            ],
        ),
        _observation(
            2,
            AGENT_DMZ,
            ZONE_DMZ,
            [
                _node_obs(NODE_APP, NodeScope.boundary),
                _node_obs(
                    NODE_WEB,
                    NodeScope.own,
                    alerts=1,
                    severity=2,
                    evidence=2,
                    health=0.6,
                    since=1,
                ),
            ],
            _dmz_edges(),
            pick("evt-000001", "evt-000002", "evt-000003", "evt-000004"),
            [
                ProposalFeedback(
                    proposal_id="prop-000004",
                    decision=Decision.accepted,
                    reason_codes=[],
                    execution_status=ExecutionStatus.executed,
                )
            ],
        ),
    ]


def _detection_nodes(entries: list[tuple[str, float, bool]]) -> list[DetectionNode]:
    return [
        DetectionNode(node_id=node_id, suspicion=suspicion, alerted=alerted)
        for node_id, suspicion, alerted in entries
    ]


def _mini_detection() -> list[DetectionSummary]:
    healthy_core = [
        (NODE_APP, 0.0, False),
        (NODE_DB, 0.0, False),
        (NODE_WEB, 0.0, False),
    ]
    healthy_dmz = [(NODE_APP, 0.0, False), (NODE_WEB, 0.0, False)]
    alerted_core = [
        (NODE_APP, 0.0, False),
        (NODE_DB, 0.0, False),
        (NODE_WEB, SUSPICION_A1_S2, True),
    ]
    alerted_dmz = [(NODE_APP, 0.0, False), (NODE_WEB, SUSPICION_A1_S2, True)]
    plan = [
        (0, AGENT_CORE, healthy_core),
        (0, AGENT_DMZ, healthy_dmz),
        (1, AGENT_CORE, alerted_core),
        (1, AGENT_DMZ, alerted_dmz),
        (2, AGENT_CORE, alerted_core),
        (2, AGENT_DMZ, alerted_dmz),
    ]
    return [
        DetectionSummary(
            detection_id=detection_id(step, agent_id),
            episode_id=MINI_EPISODE_ID,
            step=step,
            agent_id=agent_id,
            producer=DETECTION_SCORER,
            nodes=_detection_nodes(entries),
        )
        for step, agent_id, entries in plan
    ]


def _mini_node_scores(detection: list[DetectionSummary]) -> list[NodeScoreRecord]:
    """One detection-scorer record per detection summary (scores == suspicion)."""

    scorer_slug = "detection-rule-v1"
    records: list[NodeScoreRecord] = []
    for summary in detection:
        records.append(
            NodeScoreRecord(
                node_score_id=node_score_id(
                    summary.step, summary.agent_id, scorer_slug
                ),
                episode_id=MINI_EPISODE_ID,
                step=summary.step,
                agent_id=summary.agent_id,
                scorer_id=DETECTION_SCORER,
                scorer_kind=ScorerKind.detection_rule,
                nodes=[
                    NodeScore(node_id=node.node_id, score=node.suspicion)
                    for node in summary.nodes
                ],
            )
        )
    return records


def _mini_available_actions() -> list[AvailableActions]:
    core_actions = [
        _available(ActionKind.no_op),
        _available(ActionKind.investigate_node, NODE_APP),
    ]
    dmz_actions = [
        _available(ActionKind.no_op),
        _available(ActionKind.investigate_node, NODE_WEB),
        _available(ActionKind.isolate_node, NODE_WEB),
    ]
    records: list[AvailableActions] = []
    for step in range(MINI_HORIZON):
        for agent_id, actions in ((AGENT_CORE, core_actions), (AGENT_DMZ, dmz_actions)):
            records.append(
                AvailableActions(
                    available_actions_id=available_actions_id(step, agent_id),
                    episode_id=MINI_EPISODE_ID,
                    step=step,
                    agent_id=agent_id,
                    actions=list(actions),
                )
            )
    return records


def _no_op() -> ActionSpec:
    return ActionSpec(action_kind=ActionKind.no_op)


def _proposal(
    counter: int,
    step: int,
    agent_id: str,
    action: ActionSpec,
    rationale: list[RationaleCode],
    text: str | None,
) -> PolicyProposal:
    return PolicyProposal(
        proposal_id=f"prop-{counter:06d}",
        episode_id=MINI_EPISODE_ID,
        step=step,
        agent_id=agent_id,
        policy_id=MINI_POLICY_ID,
        observation_id=observation_id(step, agent_id),
        action=action,
        rationale_codes=rationale,
        rationale_text=text,
    )


def _mini_proposals() -> list[PolicyProposal]:
    investigate_web = ActionSpec(
        action_kind=ActionKind.investigate_node, target_node_id=NODE_WEB
    )
    isolate_web = ActionSpec(
        action_kind=ActionKind.isolate_node, target_node_id=NODE_WEB
    )
    return [
        _proposal(1, 0, AGENT_CORE, _no_op(), [RationaleCode.no_action_needed], None),
        _proposal(2, 0, AGENT_DMZ, _no_op(), [RationaleCode.no_action_needed], None),
        _proposal(
            3,
            1,
            AGENT_CORE,
            isolate_web,
            [RationaleCode.containment_ladder],
            "Injected faulty cross zone proposal",
        ),
        _proposal(
            4,
            1,
            AGENT_DMZ,
            investigate_web,
            [RationaleCode.alerts_present, RationaleCode.evidence_insufficient],
            "Alert on node-web-01 with no evidence yet",
        ),
        _proposal(5, 2, AGENT_CORE, _no_op(), [RationaleCode.no_action_needed], None),
        _proposal(
            6,
            2,
            AGENT_DMZ,
            isolate_web,
            [RationaleCode.evidence_moderate, RationaleCode.containment_ladder],
            "Containing node-web-01 after moderate evidence",
        ),
    ]


def _validation(
    counter: int,
    step: int,
    agent_id: str,
    decision: Decision,
    resolved_action: ActionSpec | None,
    reason_codes: list[ReasonCode],
) -> ValidationResult:
    return ValidationResult(
        validation_id=f"val-{counter:06d}",
        episode_id=MINI_EPISODE_ID,
        step=step,
        proposal_id=f"prop-{counter:06d}",
        agent_id=agent_id,
        decision=decision,
        reason_codes=reason_codes,
        resolved_action=resolved_action,
        allowlist_id=ALLOWLIST_ID,
    )


def _mini_validations() -> list[ValidationResult]:
    investigate_web = ActionSpec(
        action_kind=ActionKind.investigate_node, target_node_id=NODE_WEB
    )
    isolate_web = ActionSpec(
        action_kind=ActionKind.isolate_node, target_node_id=NODE_WEB
    )
    return [
        _validation(1, 0, AGENT_CORE, Decision.accepted, _no_op(), []),
        _validation(2, 0, AGENT_DMZ, Decision.accepted, _no_op(), []),
        _validation(
            3,
            1,
            AGENT_CORE,
            Decision.rejected,
            None,
            [ReasonCode.TARGET_OUT_OF_AUTHORITY],
        ),
        _validation(4, 1, AGENT_DMZ, Decision.accepted, investigate_web, []),
        _validation(5, 2, AGENT_CORE, Decision.accepted, _no_op(), []),
        _validation(6, 2, AGENT_DMZ, Decision.accepted, isolate_web, []),
    ]


def _execution(
    counter: int,
    step: int,
    agent_id: str,
    proposal_counter: int,
    validation_counter: int,
    action: ActionSpec,
    *,
    affected: list[str] | None = None,
    status_changes: list[StatusChange] | None = None,
    outcome_event_ids: list[str] | None = None,
) -> ExecutedAction:
    return ExecutedAction(
        execution_id=f"exe-{counter:06d}",
        episode_id=MINI_EPISODE_ID,
        step=step,
        proposal_id=f"prop-{proposal_counter:06d}",
        validation_id=f"val-{validation_counter:06d}",
        agent_id=agent_id,
        action=action,
        status=ExecutionStatus.executed,
        error_code=None,
        affected_node_ids=affected or [],
        status_changes=status_changes or [],
        outcome_event_ids=outcome_event_ids or [],
    )


def _mini_executions() -> list[ExecutedAction]:
    investigate_web = ActionSpec(
        action_kind=ActionKind.investigate_node, target_node_id=NODE_WEB
    )
    isolate_web = ActionSpec(
        action_kind=ActionKind.isolate_node, target_node_id=NODE_WEB
    )
    return [
        _execution(1, 0, AGENT_CORE, 1, 1, _no_op()),
        _execution(2, 0, AGENT_DMZ, 2, 2, _no_op()),
        _execution(
            3,
            1,
            AGENT_DMZ,
            4,
            4,
            investigate_web,
            outcome_event_ids=["evt-000003", "evt-000004"],
        ),
        _execution(4, 2, AGENT_CORE, 5, 5, _no_op()),
        _execution(
            5,
            2,
            AGENT_DMZ,
            6,
            6,
            isolate_web,
            affected=[NODE_WEB],
            status_changes=[
                StatusChange(
                    node_id=NODE_WEB,
                    status_from=NodeStatus.up,
                    status_to=NodeStatus.isolated,
                )
            ],
            outcome_event_ids=["evt-000005", "evt-000006"],
        ),
    ]


def _ground_truth_node(
    node_id: str,
    *,
    compromised: bool,
    service: float,
    defense: float,
    contained: bool,
    status: NodeStatus,
) -> GroundTruthNode:
    return GroundTruthNode(
        node_id=node_id,
        compromised=compromised,
        service_availability=service,
        defense_availability=defense,
        contained=contained,
        status=status,
    )


def _mini_ground_truth() -> list[GroundTruthSnapshot]:
    clean_app = _ground_truth_node(
        NODE_APP,
        compromised=False,
        service=1.0,
        defense=1.0,
        contained=False,
        status=NodeStatus.up,
    )
    clean_db = _ground_truth_node(
        NODE_DB,
        compromised=False,
        service=1.0,
        defense=1.0,
        contained=False,
        status=NodeStatus.up,
    )
    return [
        GroundTruthSnapshot(
            episode_id=MINI_EPISODE_ID,
            step=0,
            incident_active=False,
            nodes=[
                clean_app,
                clean_db,
                _ground_truth_node(
                    NODE_WEB,
                    compromised=False,
                    service=1.0,
                    defense=1.0,
                    contained=False,
                    status=NodeStatus.up,
                ),
            ],
            new_compromises=[],
            alert_truth=[],
        ),
        GroundTruthSnapshot(
            episode_id=MINI_EPISODE_ID,
            step=1,
            incident_active=True,
            nodes=[
                clean_app,
                clean_db,
                _ground_truth_node(
                    NODE_WEB,
                    compromised=True,
                    service=0.9,
                    defense=1.0,
                    contained=False,
                    status=NodeStatus.up,
                ),
            ],
            new_compromises=[
                NewCompromise(node_id=NODE_WEB, cause=CompromiseCause.initial_entry)
            ],
            alert_truth=[AlertTruth(event_id="evt-000002", true_positive=True)],
        ),
        GroundTruthSnapshot(
            episode_id=MINI_EPISODE_ID,
            step=2,
            incident_active=True,
            nodes=[
                clean_app,
                clean_db,
                _ground_truth_node(
                    NODE_WEB,
                    compromised=True,
                    service=0.0,
                    defense=0.0,
                    contained=True,
                    status=NodeStatus.isolated,
                ),
            ],
            new_compromises=[],
            alert_truth=[],
        ),
    ]


def _mini_episode_result() -> EpisodeResult:
    return EpisodeResult(
        episode_id=MINI_EPISODE_ID,
        run_id=MINI_RUN_ID,
        scenario_id=MINI_SCENARIO_ID,
        seed=MINI_SEED,
        policy_id=MINI_POLICY_ID,
        representation=Representation.conventional,
        split=Split.fixture,
        simulator_kind=SimulatorKind.fixture,
        steps_taken=MINI_HORIZON,
        terminated=False,
        truncated=True,
        termination_reason=TerminationReason.max_steps,
        counts=Counts(proposals=6, accepted=5, rejected=1, executed=5, failed=0),
        reward=None,
        outcome=OutcomeSummary(
            compromised_nodes=1,
            attack_impact=1.0,
            healthy_services_disrupted=0,
            healthy_disruption_cost=0.0,
            false_isolation_count=0,
            true_positive_alerts=1,
            false_positive_alerts=0,
            compromised_nodes_detected=1,
            first_compromise_step=1,
            first_true_alert_step=1,
            containment_step=2,
            detection_to_containment_steps=1,
            objective_reached=False,
            origin_node_id=NODE_WEB,
            node_scoring=[
                NodeScoring(
                    scorer_id=DETECTION_SCORER,
                    scorer_kind=ScorerKind.detection_rule,
                    origin_rank_top1=1.0,
                    origin_mrr=1.0,
                    n_ranked_pairs=4,
                )
            ],
        ),
        uncertainty_summary=None,
        provenance=_fixture_provenance(),
    )


def mini_episode() -> MiniEpisode:
    """Build the internally consistent primary mini-episode."""

    events = _mini_events()
    detection = _mini_detection()
    return MiniEpisode(
        topology=_mini_topology(),
        events=events,
        observations=_mini_observations(events),
        available_actions=_mini_available_actions(),
        detection=detection,
        node_scores=_mini_node_scores(detection),
        proposals=_mini_proposals(),
        validations=_mini_validations(),
        executions=_mini_executions(),
        ground_truth=_mini_ground_truth(),
        episode_result=_mini_episode_result(),
    )


@dataclass(frozen=True, slots=True)
class MiniEpisodeFile:
    """One mini-episode output file: path, format and the records it holds."""

    rel_path: str
    is_jsonl: bool
    records: list[ContractModel]


def mini_episode_files() -> list[MiniEpisodeFile]:
    """Return the mini-episode records laid out as repository-relative files."""

    episode = mini_episode()
    root = "fixtures/mini_episode"
    return [
        MiniEpisodeFile(f"{root}/topology.json", False, [episode.topology]),
        MiniEpisodeFile(f"{root}/events.jsonl", True, list(episode.events)),
        MiniEpisodeFile(
            f"{root}/observations.jsonl", True, list(episode.observations)
        ),
        MiniEpisodeFile(
            f"{root}/available_actions.jsonl", True, list(episode.available_actions)
        ),
        MiniEpisodeFile(f"{root}/detection.jsonl", True, list(episode.detection)),
        MiniEpisodeFile(f"{root}/node_scores.jsonl", True, list(episode.node_scores)),
        MiniEpisodeFile(f"{root}/proposals.jsonl", True, list(episode.proposals)),
        MiniEpisodeFile(f"{root}/validations.jsonl", True, list(episode.validations)),
        MiniEpisodeFile(f"{root}/executions.jsonl", True, list(episode.executions)),
        MiniEpisodeFile(
            f"{root}/episode_result.json", False, [episode.episode_result]
        ),
        MiniEpisodeFile(
            f"{root}/evaluator_only/ground_truth.jsonl",
            True,
            list(episode.ground_truth),
        ),
    ]
