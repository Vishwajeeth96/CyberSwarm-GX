"""The episode engine: the six-phase step loop (stage S3, DESIGN section 4.6).

:class:`EpisodeEngine` drives one episode through the phases of CONTRACTS
section 5, exactly once per step and always in order:

* **P0 dynamics** - the simulator's exogenous events for the step (drained and
  recorded; for step 0 only ``episode_started``).
* **P1 observe** - the simulator's agent observations and available actions,
  then the deterministic graph conversion (:mod:`cyberswarm.graph`) and the
  detection summary (:mod:`cyberswarm.roles`); the engine also emits the
  always-present detection ``NodeScoreRecord`` and assembles one
  :class:`~cyberswarm.interfaces.DecisionContext` per agent.
* **P2 propose** - the engine collects the agents' proposals (substituting a
  ``no_op`` with an :class:`~cyberswarm.contracts.ErrorRecord` for a missing one)
  and records them.
* **P3 validate / P4 execute** - handed to a :class:`~cyberswarm.interfaces.CoordinatorPort`,
  which validates each proposal and executes the accepted ones through the
  simulator's public API; the engine records the validations, executions,
  outcome events and any coordinator recovery events.
* **end** - ``end_step()`` closes the step. This engine is the *only* module
  that then reads ``ground_truth()`` / ``termination_reason()`` (AD-16, DESIGN
  section 3.2): it records the end-of-step snapshot under ``evaluator_only/``
  and feeds it to the optional :class:`~cyberswarm.interfaces.StepScorer`.

Every record of step ``t`` carries ``step = t``. Feedback on an agent's
previous-step proposal is attached to its observation (the mock leaves
``last_step_feedback`` empty, CONTRACTS 5.2), and the final post-terminal
observation is returned in a ``final`` context but never recorded (CONTRACTS 5
rule 8).

Imports stay within the engine's boundary (DESIGN section 3.2): ``contracts``,
``interfaces``, ``common`` (none needed directly), ``roles`` and ``graph`` only
- never a concrete coordinator, reward model or RL package. The coordinator and
scorer arrive as the :class:`CoordinatorPort` / :class:`StepScorer` Protocols.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..contracts import (
    ActionKind,
    ActionSpec,
    Component,
    ErrorCode,
    ErrorRecord,
    ErrorSeverity,
    ExecutedAction,
    GroundTruthSnapshot,
    NodeScore,
    NodeScoreRecord,
    PolicyId,
    PolicyProposal,
    ProposalFeedback,
    RationaleCode,
    ScenarioConfig,
    ScorerKind,
    ValidationResult,
    error_id,
    make_episode_id,
    node_score_id,
    observation_id,
    proposal_id,
)
from ..graph import GraphConverter, compute_features
from ..interfaces import (
    CoordinatorPort,
    DecisionContext,
    GroundTruthAccessor,
    Simulator,
    StepScore,
    StepScorer,
)
from ..interfaces.policy import EvidenceView
from ..roles import DetectionRole, evidence_view
from .recorder import NullRecorder, Recorder

__all__ = [
    "StepContext",
    "StepOutcome",
    "EpisodeEngine",
    "DETECTION_SCORER_ID",
]

#: Scorer id and matching id-slug of the always-present detection node scorer.
DETECTION_SCORER_ID = "detection-rule-v1"


@dataclass(frozen=True)
class StepContext:
    """Everything the policies need to act at one step (DESIGN section 4.6).

    ``agents`` maps each agent id (ascending) to its
    :class:`~cyberswarm.interfaces.DecisionContext`. ``final`` is ``True`` for
    the single post-terminal observation (``step = last step + 1``) that is
    returned so a learner can bootstrap a value estimate; nothing in a final
    context is ever recorded and it has no matching proposals.
    """

    episode_id: str
    step: int
    max_steps: int
    agents: Mapping[str, DecisionContext]
    final: bool


@dataclass(frozen=True)
class StepOutcome:
    """The result of advancing the episode by one step (DESIGN section 4.6).

    ``context`` is the next step's :class:`StepContext` (or the final
    observation when the episode ended). ``feedback`` maps each agent to the
    outcome of its proposal this step, the same feedback attached to the next
    observation. ``score`` is ``None`` unless a :class:`StepScorer` is attached.
    """

    context: StepContext
    done: bool
    terminated: bool
    truncated: bool
    feedback: Mapping[str, tuple[ProposalFeedback, ...]]
    validations: tuple[ValidationResult, ...]
    executions: tuple[ExecutedAction, ...]
    score: StepScore | None


class EpisodeEngine:
    """Drive one episode through phases P0-P4 and the end-of-step snapshot.

    Construct it once with a :class:`~cyberswarm.interfaces.Simulator` and a
    :class:`~cyberswarm.interfaces.CoordinatorPort`; the optional
    :class:`~cyberswarm.interfaces.GroundTruthAccessor`,
    :class:`~cyberswarm.interfaces.StepScorer`, :class:`Recorder` and
    :class:`~cyberswarm.roles.DetectionRole` default to "none attached", a
    no-op reward, a :class:`~cyberswarm.engine.recorder.NullRecorder` and a
    default detection role. Set ``compute_graph=False`` to skip the graph
    snapshot/feature pipeline (a conventional-only run).

    A :class:`StepScorer` consumes ground-truth snapshots, so attaching one
    without a :class:`GroundTruthAccessor` is a configuration error.
    """

    def __init__(
        self,
        simulator: Simulator,
        coordinator: CoordinatorPort,
        *,
        truth: GroundTruthAccessor | None = None,
        scorer: StepScorer | None = None,
        recorder: Recorder | None = None,
        detection: DetectionRole | None = None,
        compute_graph: bool = True,
    ) -> None:
        if scorer is not None and truth is None:
            raise ValueError(
                "a StepScorer needs ground truth; pass truth= when scorer= is set"
            )
        self._sim = simulator
        self._coord = coordinator
        self._truth = truth
        self._scorer = scorer
        self._recorder: Recorder = recorder if recorder is not None else NullRecorder()
        self._detection = detection if detection is not None else DetectionRole()
        self._compute_graph = compute_graph

        # Per-episode state, (re)initialised by reset().
        self._policy_id: PolicyId = "pol-unset"
        self._episode_id = ""
        self._max_steps = 0
        self._converter: GraphConverter | None = None
        self._proposal_counter = 0
        self._error_counter = 0
        self._prev_snapshot: GroundTruthSnapshot | None = None
        self._termination_reason: str | None = None
        self._done = False
        self._started = False
        # Buffers describing the step the engine is currently waiting on.
        self._pending_proposal_ids: dict[str, str] = {}
        self._current_observations: dict[str, Any] = {}
        self._current_available: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reset(
        self, scenario: ScenarioConfig, seed: int, policy_id: PolicyId
    ) -> StepContext:
        """Start a new episode and return the step-0 context (DESIGN 4.6).

        Builds the deterministic episode id, resets the simulator, coordinator
        and scorer, then runs P0 (drained from ``reset``) and P1 of step 0.
        """

        self._policy_id = policy_id
        self._episode_id = make_episode_id(policy_id, scenario.scenario_id, seed)
        self._max_steps = scenario.max_steps
        self._sim.reset(scenario, seed, self._episode_id)
        self._coord.start_episode(self._episode_id, self._sim.topology)
        if self._scorer is not None:
            self._scorer.reset(self._sim.topology, scenario.max_steps)
        self._converter = (
            GraphConverter(self._sim.topology) if self._compute_graph else None
        )
        self._proposal_counter = 0
        self._error_counter = 0
        self._prev_snapshot = None
        self._termination_reason = None
        self._done = False
        self._started = True
        self._pending_proposal_ids = {}
        self._current_observations = {}
        self._current_available = {}
        # P0 of step 0 already ran inside reset() (episode_started). Observe it.
        return self._observe_and_record(self._sim.step, feedback={})

    def step(
        self,
        proposals: Mapping[str, PolicyProposal | Mapping[str, Any]],
        *,
        invalid_counts: Mapping[str, int] | None = None,
    ) -> StepOutcome:
        """Run P2-P4 and the end-of-step snapshot for the current step.

        ``proposals`` maps each agent to the action it proposed (a
        :class:`~cyberswarm.contracts.PolicyProposal` or a raw mapping the
        coordinator may reject). A missing agent is substituted with a ``no_op``
        proposal and an ``ErrorRecord(POLICY_FAILURE)``. ``invalid_counts`` lets
        the RL environment report masked-action substitutions for the scorer.
        """

        if not self._started:
            raise RuntimeError("call reset() before step()")
        if self._done:
            raise RuntimeError("the episode is done; call reset() to start another")

        step_t = self._sim.step
        agent_ids = tuple(sorted(self._sim.agent_ids))

        # ---- P2: collect and record proposals (ascending agent order). ---- #
        proposal_seq: list[PolicyProposal | Mapping[str, Any]] = []
        for agent_id in agent_ids:
            pid = self._pending_proposal_ids[agent_id]
            given = proposals.get(agent_id)
            if given is None:
                substitute = self._noop_proposal(agent_id, step_t, pid)
                self._recorder.write(
                    self._policy_failure_error(agent_id, step_t, pid)
                )
                self._recorder.write(substitute)
                proposal_seq.append(substitute)
            elif isinstance(given, PolicyProposal):
                self._recorder.write(given)
                proposal_seq.append(given)
            else:
                # A raw mapping: record it if it is a valid proposal, but always
                # hand the coordinator the original so it can reject a malformed
                # one with a reason code rather than crash.
                try:
                    parsed = PolicyProposal.model_validate(given)
                except Exception:  # noqa: BLE001 - malformed proposals are data
                    parsed = None
                if parsed is not None:
                    self._recorder.write(parsed)
                proposal_seq.append(given)

        # ---- P3 + P4: coordinator validates and executes. ---- #
        result = self._coord.process(
            proposal_seq, step=step_t, available=dict(self._current_available)
        )
        for validation in result.validations:
            self._recorder.write(validation)
        for execution in result.executions:
            self._recorder.write(execution)
        for error in result.errors:
            self._recorder.write(error)

        # P4 outcome events emitted by the simulator during execution.
        for event in self._sim.drain_events():
            self._recorder.write(event)

        # Coordinator recovery verification (recovery_verified/failed events).
        recovery_events = self._coord.verify_recovery(
            step_t, dict(self._current_observations)
        )
        for event in recovery_events:
            self._recorder.write(event)

        # ---- end: close the step, then read the evaluator snapshot. ---- #
        step_result = self._sim.end_step()

        snapshot: GroundTruthSnapshot | None = None
        reason = None
        if self._truth is not None:
            snapshot = self._truth.ground_truth()
            reason = self._truth.termination_reason()
            self._recorder.write(snapshot)
            self._termination_reason = reason.value if reason is not None else None

        score: StepScore | None = None
        if self._scorer is not None:
            assert snapshot is not None  # guaranteed: scorer implies truth
            invalid = sum(invalid_counts.values()) if invalid_counts else 0
            score = self._scorer.score(
                step=step_t,
                prev=self._prev_snapshot,
                curr=snapshot,
                executions=result.executions,
                invalid_count=invalid,
                termination_reason=reason,
            )
        self._prev_snapshot = snapshot

        feedback = self._build_feedback(agent_ids, result)
        done = step_result.terminated or step_result.truncated
        if done:
            self._done = True
            # end_step emitted episode_ended (and nothing else) for this step.
            for event in self._sim.drain_events():
                self._recorder.write(event)
            next_context = self._build_final_context(feedback)
        else:
            # end_step already advanced and ran P0 of t+1; its events are
            # buffered and recorded as step t+1 inside _observe_and_record.
            next_context = self._observe_and_record(self._sim.step, feedback)

        return StepOutcome(
            context=next_context,
            done=done,
            terminated=step_result.terminated,
            truncated=step_result.truncated,
            feedback=feedback,
            validations=tuple(result.validations),
            executions=tuple(result.executions),
            score=score,
        )

    @property
    def episode_id(self) -> str:
        """The current episode id, or ``""`` before the first ``reset``."""

        return self._episode_id

    @property
    def done(self) -> bool:
        """``True`` once the episode has terminated or truncated."""

        return self._done

    @property
    def termination_reason(self) -> str | None:
        """Why the episode ended (evaluator data); ``None`` without truth."""

        return self._termination_reason

    def close(self) -> None:
        """Flush the recorder and release the simulator's episode resources."""

        self._recorder.close()
        self._sim.close()

    # ------------------------------------------------------------------ #
    # Phase helpers
    # ------------------------------------------------------------------ #
    def _observe_and_record(
        self, step: int, feedback: Mapping[str, tuple[ProposalFeedback, ...]]
    ) -> StepContext:
        """Drain P0 events and build + record the P1 records for ``step``."""

        # P0: events of this step (dynamics/lifecycle) emitted before observe.
        for event in self._sim.drain_events():
            self._recorder.write(event)

        raw_observations = self._sim.observe()
        agent_ids = tuple(sorted(raw_observations))

        contexts: dict[str, DecisionContext] = {}
        self._pending_proposal_ids = {}
        self._current_observations = {}
        self._current_available = {}

        for agent_id in agent_ids:
            observation = raw_observations[agent_id]
            agent_feedback = feedback.get(agent_id, ())
            if agent_feedback:
                observation = observation.model_copy(
                    update={"last_step_feedback": list(agent_feedback)}
                )

            available = self._sim.available_actions(agent_id)
            detection = self._detection.summarize(observation)
            node_score = self._detection_node_score(detection)
            snapshot = None
            features = None
            if self._compute_graph:
                assert self._converter is not None
                snapshot = self._converter.convert(observation)
                features = compute_features(snapshot, self._converter.graph)
            evidence = evidence_view(observation)

            # Record the P1 records of this agent in phase order.
            self._recorder.write(observation)
            self._recorder.write(available)
            self._recorder.write(detection)
            self._recorder.write(node_score)
            if snapshot is not None:
                self._recorder.write(snapshot)
            if features is not None:
                self._recorder.write(features)

            self._proposal_counter += 1
            pid = proposal_id(self._proposal_counter)
            self._pending_proposal_ids[agent_id] = pid
            self._current_observations[agent_id] = observation
            self._current_available[agent_id] = available

            contexts[agent_id] = DecisionContext(
                episode_id=self._episode_id,
                step=step,
                max_steps=self._max_steps,
                agent_id=agent_id,
                topology=self._sim.topology,
                observation=observation,
                available_actions=available,
                detection=detection,
                evidence=evidence,
                graph_snapshot=snapshot,
                graph_features=features,
                next_proposal_id=pid,
            )

        return StepContext(
            episode_id=self._episode_id,
            step=step,
            max_steps=self._max_steps,
            agents=contexts,
            final=False,
        )

    def _build_final_context(
        self, feedback: Mapping[str, tuple[ProposalFeedback, ...]]
    ) -> StepContext:
        """Build the post-terminal observation context (never recorded)."""

        raw_observations = self._sim.observe()
        agent_ids = tuple(sorted(raw_observations))
        contexts: dict[str, DecisionContext] = {}
        final_step = self._sim.step + 1
        placeholder_pid = proposal_id(self._proposal_counter + 1)

        for agent_id in agent_ids:
            observation = raw_observations[agent_id]
            final_step = observation.step
            agent_feedback = feedback.get(agent_id, ())
            if agent_feedback:
                observation = observation.model_copy(
                    update={"last_step_feedback": list(agent_feedback)}
                )
            available = self._sim.available_actions(agent_id)
            detection = self._detection.summarize(observation)
            snapshot = None
            features = None
            if self._compute_graph:
                assert self._converter is not None
                snapshot = self._converter.convert(observation)
                features = compute_features(snapshot, self._converter.graph)
            evidence: EvidenceView = evidence_view(observation)
            contexts[agent_id] = DecisionContext(
                episode_id=self._episode_id,
                step=observation.step,
                max_steps=self._max_steps,
                agent_id=agent_id,
                topology=self._sim.topology,
                observation=observation,
                available_actions=available,
                detection=detection,
                evidence=evidence,
                graph_snapshot=snapshot,
                graph_features=features,
                next_proposal_id=placeholder_pid,
            )

        return StepContext(
            episode_id=self._episode_id,
            step=final_step,
            max_steps=self._max_steps,
            agents=contexts,
            final=True,
        )

    # ------------------------------------------------------------------ #
    # Record builders
    # ------------------------------------------------------------------ #
    def _detection_node_score(self, detection) -> NodeScoreRecord:
        """Build the always-present detection ``NodeScoreRecord`` (section 15)."""

        return NodeScoreRecord(
            node_score_id=node_score_id(
                detection.step, detection.agent_id, DETECTION_SCORER_ID
            ),
            episode_id=detection.episode_id,
            step=detection.step,
            agent_id=detection.agent_id,
            scorer_id=DETECTION_SCORER_ID,
            scorer_kind=ScorerKind.detection_rule,
            nodes=[
                NodeScore(node_id=node.node_id, score=node.suspicion)
                for node in detection.nodes
            ],
        )

    def _noop_proposal(
        self, agent_id: str, step: int, pid: str
    ) -> PolicyProposal:
        """A ``no_op`` stand-in for an agent that returned no proposal."""

        return PolicyProposal(
            proposal_id=pid,
            episode_id=self._episode_id,
            step=step,
            agent_id=agent_id,
            policy_id=self._policy_id,
            observation_id=observation_id(step, agent_id),
            action=ActionSpec(action_kind=ActionKind.no_op),
            rationale_codes=[RationaleCode.no_action_needed],
        )

    def _policy_failure_error(
        self, agent_id: str, step: int, pid: str
    ) -> ErrorRecord:
        """An ``ErrorRecord`` noting the missing-proposal substitution."""

        self._error_counter += 1
        return ErrorRecord(
            error_id=error_id(self._error_counter),
            episode_id=self._episode_id,
            step=step,
            component=Component.engine,
            code=ErrorCode.POLICY_FAILURE,
            severity=ErrorSeverity.warning,
            recoverable=True,
            message="missing proposal substituted with no_op",
            proposal_id=pid,
        )

    def _build_feedback(
        self, agent_ids: tuple[str, ...], result
    ) -> dict[str, tuple[ProposalFeedback, ...]]:
        """Map each agent to the feedback on its proposal of this step."""

        validation_by_proposal = {
            validation.proposal_id: validation
            for validation in result.validations
            if validation.proposal_id is not None
        }
        execution_by_proposal = {
            execution.proposal_id: execution for execution in result.executions
        }
        feedback: dict[str, tuple[ProposalFeedback, ...]] = {}
        for agent_id in agent_ids:
            pid = self._pending_proposal_ids.get(agent_id)
            validation = validation_by_proposal.get(pid) if pid else None
            if validation is None:
                continue
            execution = execution_by_proposal.get(pid)
            feedback[agent_id] = (
                ProposalFeedback(
                    proposal_id=pid,
                    decision=validation.decision,
                    reason_codes=list(validation.reason_codes),
                    execution_status=execution.status if execution else None,
                ),
            )
        return feedback
