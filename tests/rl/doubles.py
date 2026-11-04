"""Test doubles for the RL environment tests (task 7.4).

:class:`AcceptAllCoordinator` is a minimal
:class:`~cyberswarm.interfaces.CoordinatorPort` that accepts every proposal and
executes it through the simulator's public ``apply_action``. It is deliberately
*not* the real safety coordinator (that is task 9); it exists only so the
:class:`~cyberswarm.rl.env.CyberSwarmParallelEnv` can be driven end to end over
the :class:`~cyberswarm.mock.MockSimulator`.

Beyond the engine's accept-all double it records, per step, the proposals it was
handed (``seen`` and ``last_actions``). The environment substitutes ``no_op``
for a masked-but-in-range index *before* building the proposal, so these records
let a test prove that a masked proposal never reaches the coordinator.

Constructing a :class:`~cyberswarm.interfaces.simulator.ValidatedAction` is
permitted here because this is a test double under ``tests/`` (DESIGN section
3.2: production code constructs it only in ``coordinator.validator``). Id
discipline mirrors the engine double: validation and execution ids are
per-episode counters assigned in processing order, and a ``failed`` application
gets a disjoint high-range execution id so it can never collide with an executed
one (the cross-record rule X-8).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cyberswarm.contracts import (
    ActionSpec,
    AgentId,
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
    """A recording :class:`CoordinatorPort` that accepts and executes everything."""

    def __init__(self, simulator: Any) -> None:
        self._sim = simulator
        self._episode_id = ""
        self._validation_counter = 0
        self._execution_counter = 0
        self._failed_counter = 0
        #: Per-step list of ``{agent_id: ActionSpec}`` the coordinator processed.
        self.seen: list[dict[AgentId, ActionSpec]] = []
        #: The actions of the most recently processed step, by agent id.
        self.last_actions: dict[AgentId, ActionSpec] = {}

    def start_episode(self, episode_id: str, topology: Any) -> None:
        self._episode_id = episode_id
        self._validation_counter = 0
        self._execution_counter = 0
        self._failed_counter = 0
        self.seen = []
        self.last_actions = {}

    def process(
        self,
        proposals: Sequence[PolicyProposal | Mapping[str, Any]],
        *,
        step: int,
        available: Mapping[str, Any],
    ) -> CoordinatorResult:
        validations: list[ValidationResult] = []
        executions: list[ExecutedAction] = []
        step_seen: dict[AgentId, ActionSpec] = {}
        for given in proposals:
            proposal = (
                given
                if isinstance(given, PolicyProposal)
                else PolicyProposal.model_validate(given)
            )
            step_seen[proposal.agent_id] = proposal.action
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

        self.seen.append(step_seen)
        self.last_actions = step_seen
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
