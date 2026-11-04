"""Test doubles for the engine tests (task 7.1).

:class:`AcceptAllCoordinator` is a minimal :class:`~cyberswarm.interfaces.CoordinatorPort`
that accepts every proposal and executes it through the simulator's public
``apply_action``. It is deliberately *not* the real safety coordinator (that is
task 9); it exists only so the :class:`~cyberswarm.engine.EpisodeEngine` can be
exercised end to end.

Constructing a :class:`~cyberswarm.interfaces.simulator.ValidatedAction` is
permitted here because this is a test double under ``tests/`` (DESIGN section
3.2: production code constructs it only in ``coordinator.validator``).

Id discipline mirrors the contract (CONTRACTS section 4): validation and
execution ids are per-episode counters assigned in processing order (the engine
hands proposals over in ascending ``agent_id`` order). The execution counter is
advanced only for an *executed* application, so an ``action_applied`` event's
``execution_id`` (stamped by the mock from its own executed-only counter) always
matches the :class:`~cyberswarm.contracts.ExecutedAction` the double builds - the
cross-record rule X-8. A ``failed`` application (which emits no event) gets a
disjoint high-range id so it can never collide with an executed one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cyberswarm.contracts import (
    Decision,
    ExecutedAction,
    ExecutionStatus,
    PolicyProposal,
    ValidationResult,
    execution_id,
    validation_id,
)
from cyberswarm.interfaces import CoordinatorResult
from cyberswarm.interfaces.simulator import ValidatedAction

#: Base for the disjoint id range used by failed (eventless) executions.
_FAILED_ID_BASE = 900_000


class AcceptAllCoordinator:
    """A :class:`CoordinatorPort` that accepts and executes every proposal."""

    def __init__(self, simulator: Any) -> None:
        self._sim = simulator
        self._episode_id = ""
        self._validation_counter = 0
        self._execution_counter = 0
        self._failed_counter = 0

    def start_episode(self, episode_id: str, topology: Any) -> None:
        self._episode_id = episode_id
        self._validation_counter = 0
        self._execution_counter = 0
        self._failed_counter = 0

    def process(
        self,
        proposals: Sequence[PolicyProposal | Mapping[str, Any]],
        *,
        step: int,
        available: Mapping[str, Any],
    ) -> CoordinatorResult:
        validations: list[ValidationResult] = []
        executions: list[ExecutedAction] = []
        for given in proposals:
            proposal = (
                given
                if isinstance(given, PolicyProposal)
                else PolicyProposal.model_validate(given)
            )
            self._validation_counter += 1
            val_id = validation_id(self._validation_counter)
            resolved = proposal.action

            validations.append(
                ValidationResult(
                    validation_id=val_id,
                    episode_id=self._episode_id,
                    step=step,
                    proposal_id=proposal.proposal_id,
                    agent_id=proposal.agent_id,
                    decision=Decision.accepted,
                    reason_codes=[],
                    resolved_action=resolved,
                    allowlist_id="allow-all-test",
                )
            )

            application = self._sim.apply_action(
                ValidatedAction(
                    episode_id=self._episode_id,
                    step=step,
                    agent_id=proposal.agent_id,
                    proposal_id=proposal.proposal_id,
                    validation_id=val_id,
                    action=resolved,
                )
            )

            if application.status == "executed":
                self._execution_counter += 1
                exe_id = execution_id(self._execution_counter)
                status = ExecutionStatus.executed
                error_code = None
            else:
                self._failed_counter += 1
                exe_id = execution_id(_FAILED_ID_BASE + self._failed_counter)
                status = ExecutionStatus.failed
                error_code = application.error_code

            executions.append(
                ExecutedAction(
                    execution_id=exe_id,
                    episode_id=self._episode_id,
                    step=step,
                    proposal_id=proposal.proposal_id,
                    validation_id=val_id,
                    agent_id=proposal.agent_id,
                    action=resolved,
                    status=status,
                    error_code=error_code,
                    affected_node_ids=list(application.affected_node_ids),
                    status_changes=list(application.status_changes),
                    outcome_event_ids=list(application.outcome_event_ids),
                )
            )

        return CoordinatorResult(
            validations=tuple(validations),
            executions=tuple(executions),
            errors=(),
        )

    def verify_recovery(
        self, step: int, observations: Mapping[str, Any]
    ) -> list[Any]:
        """The accept-all double performs no recovery verification."""

        return []
