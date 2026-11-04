"""Cross-record consistency checks (CONTRACTS section 12, rules X-1 to X-17).

These functions take already-parsed records (each individually valid against its
own schema) and check the relationships *between* records that no single schema
can express. They never raise on well-formed input; they return a list of
:class:`ConsistencyIssue`. Run-level rules that need more than one episode
(X-13 evaluation-metric recomputation, X-17 manifest status) are checked by the
bundle-level tests, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable

from .actions import ExecutedAction, PolicyProposal, ValidationResult
from .common import Decision
from .events import EventRecord, EventType, Phase
from .evaluator_only import GroundTruthSnapshot
from .graph import GraphFeatureRecord, GraphSnapshot
from .ids import observation_id
from .observation import AgentObservation, AvailableActions, DetectionSummary
from .results import EpisodeResult, NodeScoreRecord
from .topology import Topology
from .vocabulary import ActionKind

#: Field names that may appear only in the evaluator-only schema (CONTRACTS X-10).
EVALUATOR_ONLY_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {
        "compromised",
        "true_positive",
        "service_availability",
        "defense_availability",
        "new_compromises",
        "alert_truth",
        "incident_active",
    }
)


@dataclass(frozen=True, slots=True)
class ConsistencyIssue:
    """A single cross-record problem found by :func:`check_episode`."""

    rule: str
    message: str


@dataclass(frozen=True, slots=True)
class EpisodeRecordSet:
    """All agent-visible and evaluator records of one episode, parsed."""

    events: list[EventRecord] = field(default_factory=list)
    observations: list[AgentObservation] = field(default_factory=list)
    available_actions: list[AvailableActions] = field(default_factory=list)
    detection: list[DetectionSummary] = field(default_factory=list)
    node_scores: list[NodeScoreRecord] = field(default_factory=list)
    graph_snapshots: list[GraphSnapshot] = field(default_factory=list)
    graph_features: list[GraphFeatureRecord] = field(default_factory=list)
    proposals: list[PolicyProposal] = field(default_factory=list)
    validations: list[ValidationResult] = field(default_factory=list)
    executions: list[ExecutedAction] = field(default_factory=list)
    ground_truth: list[GroundTruthSnapshot] = field(default_factory=list)
    episode_result: EpisodeResult | None = None


def check_episode(
    records: EpisodeRecordSet, topology: Topology
) -> list[ConsistencyIssue]:
    """Return every cross-record issue in one episode (CONTRACTS X-1 to X-16)."""

    issues: list[ConsistencyIssue] = []
    _check_id_resolution(records, topology, issues)  # X-1, X-2
    _check_episode_and_steps(records, issues)  # X-3
    _check_event_contiguity(records, issues)  # X-4
    _check_observation_events(records, issues)  # X-5, X-16
    _check_proposal_chain(records, issues)  # X-6, X-7
    _check_outcome_events(records, issues)  # X-8
    _check_accepted_available(records, issues)  # X-9
    _check_graph_order(records, topology, issues)  # X-11
    _check_counts(records, issues)  # X-12
    _check_node_scores(records, issues)  # X-15
    return issues


# --------------------------------------------------------------------------- #
# X-1 / X-2: id resolution
# --------------------------------------------------------------------------- #
def _check_id_resolution(
    records: EpisodeRecordSet,
    topology: Topology,
    issues: list[ConsistencyIssue],
) -> None:
    node_ids = set(topology.node_ids)
    edge_ids = set(topology.edge_ids)
    zone_ids = set(topology.zone_ids)
    agent_ids = set(topology.agent_ids)

    def check_node(node_id: str | None, where: str) -> None:
        if node_id is not None and node_id not in node_ids:
            issues.append(ConsistencyIssue("X-1", f"unknown node_id {node_id!r} in {where}"))

    def check_edge(edge_id: str, where: str) -> None:
        if edge_id not in edge_ids:
            issues.append(ConsistencyIssue("X-1", f"unknown edge_id {edge_id!r} in {where}"))

    def check_zone(zone_id: str, where: str) -> None:
        if zone_id not in zone_ids:
            issues.append(ConsistencyIssue("X-1", f"unknown zone_id {zone_id!r} in {where}"))

    def check_agent(agent_id: str | None, where: str) -> None:
        if agent_id is not None and agent_id not in agent_ids:
            issues.append(ConsistencyIssue("X-2", f"unknown agent_id {agent_id!r} in {where}"))

    for obs in records.observations:
        check_zone(obs.zone_id, obs.observation_id)
        check_agent(obs.agent_id, obs.observation_id)
        for node in obs.nodes:
            check_node(node.node_id, obs.observation_id)
        for edge in obs.edges:
            check_edge(edge.edge_id, obs.observation_id)
    for event in records.events:
        check_node(event.node_id, event.event_id)
        check_agent(event.agent_id, event.event_id)
        for agent in event.visible_to:
            check_agent(agent, f"{event.event_id}.visible_to")
    for proposal in records.proposals:
        check_agent(proposal.agent_id, proposal.proposal_id)
        check_node(proposal.action.target_node_id, proposal.proposal_id)
    for execution in records.executions:
        check_agent(execution.agent_id, execution.execution_id)
        check_node(execution.action.target_node_id, execution.execution_id)
        for node_id in execution.affected_node_ids:
            check_node(node_id, execution.execution_id)
    for summary in records.detection:
        check_agent(summary.agent_id, summary.detection_id)
        for node in summary.nodes:
            check_node(node.node_id, summary.detection_id)
    for score in records.node_scores:
        for node in score.nodes:
            check_node(node.node_id, score.node_score_id)
    for snapshot in records.graph_snapshots:
        for node_id in snapshot.node_order:
            check_node(node_id, snapshot.snapshot_id)
        for edge_id in snapshot.edge_order:
            check_edge(edge_id, snapshot.snapshot_id)
    for truth in records.ground_truth:
        for node in truth.nodes:
            check_node(node.node_id, f"ground_truth@{truth.step}")


# --------------------------------------------------------------------------- #
# X-3: episode id and step ranges
# --------------------------------------------------------------------------- #
def _iter_step_records(records: EpisodeRecordSet) -> Iterable[tuple[str, str, int]]:
    for obs in records.observations:
        yield (obs.episode_id, obs.observation_id, obs.step)
    for event in records.events:
        yield (event.episode_id, event.event_id, event.step)
    for proposal in records.proposals:
        yield (proposal.episode_id, proposal.proposal_id, proposal.step)
    for validation in records.validations:
        yield (validation.episode_id, validation.validation_id, validation.step)
    for execution in records.executions:
        yield (execution.episode_id, execution.execution_id, execution.step)


def _check_episode_and_steps(
    records: EpisodeRecordSet, issues: list[ConsistencyIssue]
) -> None:
    result = records.episode_result
    expected_episode = result.episode_id if result is not None else None
    for episode_id, record_id, step in _iter_step_records(records):
        if expected_episode is None:
            expected_episode = episode_id
        if episode_id != expected_episode:
            issues.append(
                ConsistencyIssue(
                    "X-3",
                    f"{record_id} has episode_id {episode_id!r}, expected {expected_episode!r}",
                )
            )
        if result is not None and not (0 <= step <= result.steps_taken - 1):
            issues.append(
                ConsistencyIssue(
                    "X-3",
                    f"{record_id} step {step} outside [0, {result.steps_taken - 1}]",
                )
            )


# --------------------------------------------------------------------------- #
# X-4: event id contiguity per prefix
# --------------------------------------------------------------------------- #
def _check_event_contiguity(
    records: EpisodeRecordSet, issues: list[ConsistencyIssue]
) -> None:
    for prefix in ("evt", "cev"):
        counters = sorted(
            int(event.event_id.split("-")[1])
            for event in records.events
            if event.event_id.startswith(prefix + "-")
        )
        expected = list(range(1, len(counters) + 1))
        if counters != expected:
            issues.append(
                ConsistencyIssue(
                    "X-4",
                    f"{prefix}- event ids are not contiguous starting at 1: {counters}",
                )
            )


# --------------------------------------------------------------------------- #
# X-5 / X-16: observation events
# --------------------------------------------------------------------------- #
def _check_observation_events(
    records: EpisodeRecordSet, issues: list[ConsistencyIssue]
) -> None:
    proposals_by_step_agent: dict[tuple[int, str], set[str]] = {}
    for proposal in records.proposals:
        proposals_by_step_agent.setdefault((proposal.step, proposal.agent_id), set()).add(
            proposal.proposal_id
        )

    for obs in records.observations:
        for event in obs.recent_events:
            if event.step > obs.step:
                issues.append(
                    ConsistencyIssue(
                        "X-5",
                        f"{obs.observation_id} contains future event {event.event_id}",
                    )
                )
            if obs.agent_id not in event.visible_to:
                issues.append(
                    ConsistencyIssue(
                        "X-16",
                        f"{obs.observation_id} contains event {event.event_id} not visible to it",
                    )
                )
            if event.event_type in (EventType.action_applied, EventType.investigation_result):
                if event.agent_id != obs.agent_id:
                    issues.append(
                        ConsistencyIssue(
                            "X-16",
                            f"{obs.observation_id} sees another agent's {event.event_type}",
                        )
                    )
        for feedback in obs.last_step_feedback:
            known = proposals_by_step_agent.get((obs.step - 1, obs.agent_id), set())
            if known and feedback.proposal_id not in known:
                issues.append(
                    ConsistencyIssue(
                        "X-5",
                        f"{obs.observation_id} feedback refers to {feedback.proposal_id}, "
                        "not a proposal of the previous step",
                    )
                )


# --------------------------------------------------------------------------- #
# X-6 / X-7: proposal / validation / execution chain
# --------------------------------------------------------------------------- #
def _check_proposal_chain(
    records: EpisodeRecordSet, issues: list[ConsistencyIssue]
) -> None:
    proposals = {p.proposal_id: p for p in records.proposals}
    observation_ids = {obs.observation_id for obs in records.observations}
    executions_by_proposal: dict[str, list[ExecutedAction]] = {}
    for execution in records.executions:
        executions_by_proposal.setdefault(execution.proposal_id, []).append(execution)

    # X-7: proposal.observation_id resolves and matches obs-<step>-<agent>.
    for proposal in records.proposals:
        expected = observation_id(proposal.step, proposal.agent_id)
        if proposal.observation_id != expected:
            issues.append(
                ConsistencyIssue(
                    "X-7",
                    f"{proposal.proposal_id} observation_id {proposal.observation_id!r} "
                    f"should be {expected!r}",
                )
            )
        elif observation_ids and proposal.observation_id not in observation_ids:
            issues.append(
                ConsistencyIssue(
                    "X-7",
                    f"{proposal.proposal_id} references missing observation "
                    f"{proposal.observation_id!r}",
                )
            )

    for validation in records.validations:
        proposal = proposals.get(validation.proposal_id) if validation.proposal_id else None
        if validation.proposal_id is not None and proposal is None:
            issues.append(
                ConsistencyIssue(
                    "X-6",
                    f"{validation.validation_id} references unknown proposal "
                    f"{validation.proposal_id!r}",
                )
            )
        elif proposal is not None:
            if proposal.step != validation.step or proposal.agent_id != validation.agent_id:
                issues.append(
                    ConsistencyIssue(
                        "X-6",
                        f"{validation.validation_id} does not match the step/agent of "
                        f"{validation.proposal_id}",
                    )
                )
        related = executions_by_proposal.get(validation.proposal_id or "", [])
        if validation.decision is Decision.accepted:
            if len(related) != 1:
                issues.append(
                    ConsistencyIssue(
                        "X-6",
                        f"accepted {validation.validation_id} must have exactly one execution",
                    )
                )
            elif related[0].action != validation.resolved_action:
                issues.append(
                    ConsistencyIssue(
                        "X-6",
                        f"execution {related[0].execution_id} action differs from the "
                        f"resolved action of {validation.validation_id}",
                    )
                )
        else:  # rejected
            if related:
                issues.append(
                    ConsistencyIssue(
                        "X-6",
                        f"rejected {validation.validation_id} must have no execution",
                    )
                )


# --------------------------------------------------------------------------- #
# X-8: outcome events
# --------------------------------------------------------------------------- #
def _check_outcome_events(
    records: EpisodeRecordSet, issues: list[ConsistencyIssue]
) -> None:
    events = {event.event_id: event for event in records.events}
    for execution in records.executions:
        for event_id in execution.outcome_event_ids:
            event = events.get(event_id)
            if event is None:
                issues.append(
                    ConsistencyIssue(
                        "X-8",
                        f"{execution.execution_id} references unknown event {event_id!r}",
                    )
                )
                continue
            if event.phase is not Phase.outcome or event.step != execution.step:
                issues.append(
                    ConsistencyIssue(
                        "X-8",
                        f"{event_id} must be an outcome event in step {execution.step}",
                    )
                )
            if event.event_type is EventType.action_applied and (
                event.execution_id != execution.execution_id
            ):
                issues.append(
                    ConsistencyIssue(
                        "X-8",
                        f"action_applied {event_id} execution_id must be "
                        f"{execution.execution_id}",
                    )
                )


# --------------------------------------------------------------------------- #
# X-9: accepted proposals appear in AvailableActions
# --------------------------------------------------------------------------- #
def _check_accepted_available(
    records: EpisodeRecordSet, issues: list[ConsistencyIssue]
) -> None:
    available: dict[tuple[int, str], set[tuple[str, str | None]]] = {}
    for record in records.available_actions:
        key = (record.step, record.agent_id)
        available.setdefault(key, set())
        for action in record.actions:
            available[key].add((action.action_kind.value, action.target_node_id))

    proposals = {p.proposal_id: p for p in records.proposals}
    for validation in records.validations:
        if validation.decision is not Decision.accepted or validation.resolved_action is None:
            continue
        proposal = proposals.get(validation.proposal_id or "")
        if proposal is None:
            continue
        key = (validation.step, proposal.agent_id)
        if key not in available:
            continue
        candidate = (
            validation.resolved_action.action_kind.value,
            validation.resolved_action.target_node_id,
        )
        if candidate not in available[key]:
            issues.append(
                ConsistencyIssue(
                    "X-9",
                    f"accepted action {candidate} for {proposal.agent_id} at step "
                    f"{validation.step} is not in AvailableActions",
                )
            )


# --------------------------------------------------------------------------- #
# X-11: graph order and shape vs topology
# --------------------------------------------------------------------------- #
def _check_graph_order(
    records: EpisodeRecordSet,
    topology: Topology,
    issues: list[ConsistencyIssue],
) -> None:
    node_order = list(topology.node_ids)
    edge_order = list(topology.edge_ids)
    for snapshot in records.graph_snapshots:
        if snapshot.node_order != node_order:
            issues.append(
                ConsistencyIssue("X-11", f"{snapshot.snapshot_id} node_order != topology nodes")
            )
        if snapshot.edge_order != edge_order:
            issues.append(
                ConsistencyIssue("X-11", f"{snapshot.snapshot_id} edge_order != topology edges")
            )
    for feature in records.graph_features:
        if feature.node_order != node_order:
            issues.append(
                ConsistencyIssue(
                    "X-11", f"{feature.feature_record_id} node_order != topology nodes"
                )
            )


# --------------------------------------------------------------------------- #
# X-12: episode-result counts
# --------------------------------------------------------------------------- #
def _check_counts(records: EpisodeRecordSet, issues: list[ConsistencyIssue]) -> None:
    result = records.episode_result
    if result is None:
        return
    counts = result.counts
    if counts.proposals != len(records.proposals):
        issues.append(
            ConsistencyIssue(
                "X-12",
                f"counts.proposals {counts.proposals} != {len(records.proposals)} proposal records",
            )
        )
    accepted = sum(1 for v in records.validations if v.decision is Decision.accepted)
    rejected = sum(1 for v in records.validations if v.decision is Decision.rejected)
    if counts.accepted != accepted:
        issues.append(
            ConsistencyIssue(
                "X-12", f"counts.accepted {counts.accepted} != {accepted} accepted validations"
            )
        )
    if counts.rejected != rejected:
        issues.append(
            ConsistencyIssue(
                "X-12", f"counts.rejected {counts.rejected} != {rejected} rejected validations"
            )
        )
    if counts.executed + counts.failed != len(records.executions):
        issues.append(
            ConsistencyIssue(
                "X-12",
                f"counts executed+failed {counts.executed + counts.failed} != "
                f"{len(records.executions)} execution records",
            )
        )


# --------------------------------------------------------------------------- #
# X-15: node-score lists match observation nodes; detection scorer == suspicion
# --------------------------------------------------------------------------- #
def _check_node_scores(records: EpisodeRecordSet, issues: list[ConsistencyIssue]) -> None:
    obs_nodes: dict[tuple[int, str], list[str]] = {}
    for obs in records.observations:
        obs_nodes[(obs.step, obs.agent_id)] = [node.node_id for node in obs.nodes]
    detection_suspicion: dict[tuple[int, str], dict[str, float]] = {}
    for summary in records.detection:
        detection_suspicion[(summary.step, summary.agent_id)] = {
            node.node_id: node.suspicion for node in summary.nodes
        }

    for score in records.node_scores:
        key = (score.step, score.agent_id)
        expected_nodes = obs_nodes.get(key)
        scored_nodes = [node.node_id for node in score.nodes]
        if expected_nodes is not None and scored_nodes != expected_nodes:
            issues.append(
                ConsistencyIssue(
                    "X-15",
                    f"{score.node_score_id} node list differs from the observation",
                )
            )
        if score.scorer_kind.value == "detection_rule":
            suspicion = detection_suspicion.get(key)
            if suspicion is not None:
                for node in score.nodes:
                    expected = suspicion.get(node.node_id)
                    if expected is not None and abs(expected - node.score) > 1e-9:
                        issues.append(
                            ConsistencyIssue(
                                "X-15",
                                f"{score.node_score_id} score for {node.node_id} "
                                "differs from the detection suspicion",
                            )
                        )
    return None


# --------------------------------------------------------------------------- #
# X-10: no evaluator-only fields in agent-visible files
# --------------------------------------------------------------------------- #
def _scan_keys(payload: object) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in EVALUATOR_ONLY_FIELD_NAMES:
                found.add(key)
            found |= _scan_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            found |= _scan_keys(item)
    return found


def check_no_evaluator_fields(paths: Iterable[Path]) -> list[str]:
    """Return ``"<path>: <field>"`` for each evaluator-only field found (X-10).

    Pass agent-visible files only. Files under an ``evaluator_only`` directory
    are skipped because they legitimately carry ground-truth fields.
    """

    violations: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if "evaluator_only" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        payloads: list[object] = []
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                stripped = line.strip()
                if stripped:
                    payloads.append(json.loads(stripped))
        else:
            payloads.append(json.loads(text))
        for payload in payloads:
            for field_name in sorted(_scan_keys(payload)):
                violations.append(f"{path}: {field_name}")
    return violations
