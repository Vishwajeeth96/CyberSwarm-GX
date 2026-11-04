"""The public simulator interface (stage S2, DESIGN section 4.2).

This module is *neutral*: it declares the boundary that both the lead-owned
MOCK simulator and Raksha's real simulator implement, without knowing anything
about either implementation. It imports only :mod:`cyberswarm.contracts` and the
standard library (import boundary of DESIGN section 3.2).

It defines:

* :class:`Simulator` - the runtime-checkable Protocol for the episode loop
  (``reset``, ``observe``, ``available_actions``, ``apply_action``,
  ``end_step``, ``drain_events``, ``state_digest``, ``close`` and the read-only
  ``topology``/``agent_ids``/``step``/``done`` views).
* :class:`GroundTruthAccessor` - the *separate*, evaluator-only Protocol that
  exposes hidden truth. It is deliberately a different object from the
  :class:`Simulator` so that no decision-time code can reach ground truth by
  holding a simulator reference (CONTRACTS section 9).
* :class:`ValidatedAction` - the only action the simulator accepts. It is
  constructed solely by ``coordinator.validator``; the conformance suite and
  test doubles under ``tests/`` may construct it for testing. No policy or
  decision-time component builds one.
* :class:`ActionApplication`, :class:`StepResult` - the simulator's replies to
  ``apply_action`` and ``end_step``.
* :class:`SimulatorError` - raised for misuse or an unsupported scenario (never
  for a merely inapplicable action, which is a normal ``failed`` result).

Ground-truth timing (AD-16, CONTRACTS section 5 rule 9, conformance SC-14):
``end_step()`` captures the evaluator snapshot of the step it closes *before* it
runs phase P0 of the next step. :meth:`GroundTruthAccessor.ground_truth`
therefore returns the end-of-step-``t`` snapshot, whose ``step`` equals
:attr:`StepResult.step_closed`, and that value is stable until the next
``end_step()`` call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable

from ..contracts import (
    ActionSpec,
    AgentId,
    AgentObservation,
    AvailableActions,
    ContractModel,
    ErrorCode,
    EpisodeId,
    EventId,
    EventRecord,
    GroundTruthSnapshot,
    NodeId,
    ProposalId,
    ScenarioConfig,
    StatusChange,
    Step,
    TerminationReason,
    Topology,
    ValidationId,
)

__all__ = [
    "SimulatorError",
    "ValidatedAction",
    "ActionApplication",
    "StepResult",
    "Simulator",
    "GroundTruthAccessor",
]


class SimulatorError(Exception):
    """A simulator was misused or asked to run an unsupported scenario.

    ``code`` is an :class:`~cyberswarm.contracts.ErrorCode`. ``reset`` raises
    :data:`ErrorCode.SCENARIO_UNSUPPORTED` for an unknown topology, node or
    unsupported scenario field; ``observe``, ``apply_action`` and ``end_step``
    raise :data:`ErrorCode.SIMULATOR_FAILURE` only for an internal fault or
    misuse (for example ``end_step`` after the episode is done). A rejected or
    inapplicable action is *not* an error: it is a normal ``failed``
    :class:`ActionApplication` that leaves ``state_digest()`` unchanged.
    """

    code: ErrorCode

    def __init__(self, code: ErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)


class ValidatedAction(ContractModel):
    """An action the coordinator has validated and the simulator may apply.

    Constructed only by ``coordinator.validator`` in production code (it is the
    sole validation boundary; see DESIGN section 3.2). The conformance suite and
    test doubles under ``tests/`` are the only other places that may build one,
    and only for testing. The ``action`` is fully resolved: defaults are filled
    and the target (when the kind needs one) is present.
    """

    episode_id: EpisodeId
    step: Step
    agent_id: AgentId
    proposal_id: ProposalId
    validation_id: ValidationId
    action: ActionSpec


class ActionApplication(ContractModel):
    """The simulator's reply to :meth:`Simulator.apply_action`.

    ``status`` is ``"executed"`` when the action changed state and
    ``"failed"`` when it was inapplicable or out of authority. A ``"failed"``
    application carries an ``error_code`` and leaves ``state_digest()``
    unchanged (conformance SC-9); it is never raised as an exception. The
    coordinator turns this reply into an :class:`~cyberswarm.contracts.ExecutedAction`
    record; the :class:`~cyberswarm.contracts.StatusChange` entries are reused
    verbatim.
    """

    status: Literal["executed", "failed"]
    error_code: ErrorCode | None
    affected_node_ids: tuple[NodeId, ...]
    status_changes: tuple[StatusChange, ...]
    outcome_event_ids: tuple[EventId, ...]


class StepResult(ContractModel):
    """The simulator's reply to :meth:`Simulator.end_step`.

    ``step_closed`` is the step just closed (the value that the end-of-step
    :class:`~cyberswarm.contracts.GroundTruthSnapshot` and
    :meth:`GroundTruthAccessor.ground_truth` report as their ``step``). Exactly
    the termination/truncation of that closed step is reported here.
    """

    step_closed: Step
    terminated: bool
    truncated: bool


@runtime_checkable
class Simulator(Protocol):
    """The episode-loop contract every simulator implements (DESIGN 4.2).

    The loop is: ``reset`` -> (``observe`` -> ``available_actions`` ->
    ``apply_action`` -> ``end_step``)* with ``drain_events`` and
    ``state_digest`` read as needed. ``apply_action`` never raises for an
    inapplicable action; it returns a ``failed`` :class:`ActionApplication`.
    ``end_step`` closes the current step, evaluates termination, and runs P0 of
    the next step when the episode continues.

    The Protocol is ``runtime_checkable`` so tests and the engine can assert an
    object satisfies it with ``isinstance``.
    """

    kind: Literal["mock", "simulator"]

    @property
    def topology(self) -> Topology: ...

    @property
    def agent_ids(self) -> tuple[AgentId, ...]:
        """Decision-agent ids in ascending order."""

    def authority(self, agent_id: AgentId) -> tuple[NodeId, ...]:
        """Node ids of the agent's zone, ascending (its authority)."""

    def reset(self, scenario: ScenarioConfig, seed: int, episode_id: EpisodeId) -> None:
        """Start a new episode. Raises ``SimulatorError(SCENARIO_UNSUPPORTED)``
        for an unknown topology/node or an unsupported scenario field, and
        leaves the simulator unusable until a successful ``reset``."""

    @property
    def step(self) -> int:
        """The current step ``t``; ``0`` immediately after ``reset``."""

    @property
    def done(self) -> bool:
        """``True`` once the episode has terminated or truncated."""

    def observe(self) -> Mapping[AgentId, AgentObservation]:
        """Agent-visible observations for the current step (phase P1).

        Idempotent within a step. After the episode is done it returns the
        single final observation with ``step = last step + 1``.
        """

    def available_actions(self, agent_id: AgentId) -> AvailableActions:
        """State-dependent available actions for one agent this step."""

    def apply_action(self, action: ValidatedAction) -> ActionApplication:
        """Apply one validated action (phase P4).

        Never raises for an inapplicable or out-of-authority action: it returns
        a ``failed`` :class:`ActionApplication` and leaves state unchanged.
        """

    def end_step(self) -> StepResult:
        """Close the current step.

        Captures the end-of-step evaluator snapshot, evaluates termination and
        truncation on that state, and - when the episode continues - advances to
        the next step and runs its P0. The ground-truth snapshot is captured
        before P0 of the next step (AD-16). Raises ``SimulatorError`` if called
        after the episode is done.
        """

    def drain_events(self) -> list[EventRecord]:
        """Events emitted since the last drain, in emission order."""

    def state_digest(self) -> str:
        """SHA-256 hex (64 lowercase chars) of the canonical internal state."""

    def close(self) -> None:
        """Release any resources held by the simulator."""


@runtime_checkable
class GroundTruthAccessor(Protocol):
    """Evaluator-only access to hidden truth (CONTRACTS section 9).

    This is a *separate* object from :class:`Simulator` on purpose: ground truth
    never travels with a decision-time simulator reference. Only
    ``engine.step_engine`` and tests may call these methods (DESIGN 3.2); the
    import-boundary test enforces it.
    """

    def ground_truth(self) -> GroundTruthSnapshot:
        """Return the end-of-step-``t`` snapshot of the most recently closed
        step.

        The snapshot is captured by ``end_step()`` *before* it runs phase P0 of
        step ``t+1`` (AD-16, CONTRACTS section 5 rule 9). Its ``step`` equals
        :attr:`StepResult.step_closed`, and it is stable until the next
        ``end_step()`` call. An event that fires at P0 of ``t+1`` (for example
        ``restore_completed``) is therefore not yet reflected here. Conformance
        check SC-14 verifies this timing.
        """

    def termination_reason(self) -> TerminationReason | None:
        """Why the closed step ended, or ``None`` if the episode continues.

        One of ``objective_reached``, ``incident_resolved`` or ``max_steps``;
        it refers to the same closed step as :meth:`ground_truth`.
        """
