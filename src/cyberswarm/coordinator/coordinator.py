"""The defense coordinator: the system's one safety boundary (stage S4).

:class:`DefenseCoordinator` implements the
:class:`~cyberswarm.interfaces.CoordinatorPort` the engine depends on. It is the
sole validator of proposals and the sole caller of the simulator's public
``apply_action`` (DESIGN sections 3.3, 4.8). Everything it does is pure,
in-memory validation and bookkeeping over the injected simulator's public
interface: it reaches no real host, opens no remote connection and mutates no
file. A rejected proposal changes nothing, so the simulator's ``state_digest``
is identical before and after a batch of rejections.

Processing order (AD-21). ``process`` reads every proposal, orders them ascending
by ``agent_id`` (an unparseable input sorts last, in input order), and validates
each in that order. The ``val-`` counter is assigned by processing position, so
the n-th processed proposal gets the n-th validation id (including one that
fails to parse). Accepted actions are executed in the same order through
``apply_action``; the resulting :class:`~cyberswarm.contracts.ExecutedAction`
adopts the simulator's own ``execution_id`` scheme - the mock stamps each
executed ``action_applied`` event from an executed-only counter, so mirroring an
executed-only counter here makes the execution record's id equal the
``action_applied`` event's id (cross-record rule X-8). A ``failed`` application
(which the simulator leaves eventless and state-preserving) takes a disjoint
high-range id so it can never collide with an executed one.

This module imports only :mod:`cyberswarm.contracts` and
:mod:`cyberswarm.interfaces` (and its sibling coordinator modules); it never
constructs a :class:`~cyberswarm.interfaces.ValidatedAction` itself - that is the
validator's sole responsibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import count
from typing import Any

from ..contracts import (
    ActionKind,
    AgentId,
    AgentObservation,
    AvailableActions,
    EpisodeId,
    EventRecord,
    ExecutedAction,
    ExecutionStatus,
    PolicyProposal,
    Topology,
    execution_id,
    validation_id,
)
from ..interfaces import CoordinatorResult, Simulator, ValidatedAction
from ..interfaces.simulator import SimulatorError
from .config import CoordinatorConfig
from .recovery import RecoveryVerifier
from .validator import ProposalValidator, ProposalView

__all__ = ["DefenseCoordinator"]

#: Base for the disjoint id range used by failed (eventless) executions, so a
#: failed application's id never collides with an executed one (X-8).
_FAILED_ID_BASE = 900_000


class DefenseCoordinator:
    """Validate and execute one step's proposals (implements ``CoordinatorPort``).

    Construct it once with a :class:`~cyberswarm.interfaces.Simulator` and a
    :class:`~cyberswarm.coordinator.config.CoordinatorConfig`; call
    :meth:`start_episode` at each ``reset`` and :meth:`process` once per step.
    """

    def __init__(self, simulator: Simulator, config: CoordinatorConfig) -> None:
        self._sim = simulator
        self._config = config
        self._validator: ProposalValidator | None = None
        self._recovery = RecoveryVerifier()
        self._episode_id: EpisodeId = "ep-unset-s0"
        self._started = False
        self._seen_proposal_ids: set[str] = set()
        self._validation_counter = 0
        self._execution_counter = 0
        self._failed_execution_counter = 0
        self._event_counter = count(1)

    # ------------------------------------------------------------------ #
    # CoordinatorPort
    # ------------------------------------------------------------------ #
    def start_episode(self, episode_id: EpisodeId, topology: Topology) -> None:
        """Begin a new episode: rebuild the validator and reset the counters."""

        self._episode_id = episode_id
        self._validator = ProposalValidator(topology, self._config)
        self._recovery.start_episode(episode_id)
        self._started = True
        self._seen_proposal_ids = set()
        self._validation_counter = 0
        self._execution_counter = 0
        self._failed_execution_counter = 0
        self._event_counter = count(1)

    def process(
        self,
        proposals: Sequence[PolicyProposal | Mapping[str, Any]],
        *,
        step: int,
        available: Mapping[AgentId, AvailableActions],
    ) -> CoordinatorResult:
        """Validate (P3) and execute (P4) one step's proposals."""

        if self._validator is None:
            raise RuntimeError("call start_episode() before process()")

        views = [self._validator.parse(proposal) for proposal in proposals]
        order = sorted(range(len(proposals)), key=lambda i: self._sort_key(views[i], i))
        active = self._started and not self._sim.done

        validations = []
        executions = []
        accepted: list[ValidatedAction] = []

        for index in order:
            view = views[index]
            self._validation_counter += 1
            vid = validation_id(self._validation_counter)
            digest = self._current_digest()
            result, validated = self._validator.validate(
                view,
                episode_id=self._episode_id,
                step=step,
                available=available,
                accepted_so_far=accepted,
                seen_proposal_ids=self._seen_proposal_ids,
                validation_id=vid,
                state_digest=digest,
                active=active,
            )
            validations.append(result)
            if view.proposal_id is not None:
                self._seen_proposal_ids.add(view.proposal_id)
            if validated is not None:
                accepted.append(validated)
                executions.append(self._execute(validated, step, digest))

        return CoordinatorResult(
            validations=tuple(validations),
            executions=tuple(executions),
            errors=(),
        )

    def verify_recovery(
        self, step: int, observations: Mapping[AgentId, AgentObservation]
    ) -> list[EventRecord]:
        """Emit any ``recovery_verified`` / ``recovery_failed`` events (AD-20)."""

        return self._recovery.check(step, observations, self._event_counter)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sort_key(view: ProposalView, index: int) -> tuple[int, str, int]:
        """Parseable proposals sort by agent then input order; others go last."""

        if view.schema_invalid or view.agent_id is None:
            return (1, "", index)
        return (0, view.agent_id, index)

    def _current_digest(self) -> str | None:
        try:
            return self._sim.state_digest()
        except SimulatorError:
            return None

    def _execute(
        self, validated: ValidatedAction, step: int, digest_before: str | None
    ) -> ExecutedAction:
        """Apply one accepted action and build its execution record."""

        application = self._sim.apply_action(validated)
        digest_after = self._current_digest()

        if application.status == "executed":
            self._execution_counter += 1
            exe_id = execution_id(self._execution_counter)
            status = ExecutionStatus.executed
            error_code = None
        else:
            self._failed_execution_counter += 1
            exe_id = execution_id(_FAILED_ID_BASE + self._failed_execution_counter)
            status = ExecutionStatus.failed
            error_code = application.error_code

        execution = ExecutedAction(
            execution_id=exe_id,
            episode_id=self._episode_id,
            step=step,
            proposal_id=validated.proposal_id,
            validation_id=validated.validation_id,
            agent_id=validated.agent_id,
            action=validated.action,
            status=status,
            error_code=error_code,
            affected_node_ids=list(application.affected_node_ids),
            status_changes=list(application.status_changes),
            outcome_event_ids=list(application.outcome_event_ids),
            state_digest_before=digest_before,
            state_digest_after=digest_after,
        )

        if (
            validated.action.action_kind is ActionKind.restore_node
            and status is ExecutionStatus.executed
        ):
            self._recovery.note_execution(execution)

        return execution
