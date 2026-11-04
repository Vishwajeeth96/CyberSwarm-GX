"""Engine ports: the coordinator and the step scorer (stage S2, DESIGN 4.2).

The episode engine depends on two abstractions so that it never imports a
concrete coordinator or reward model:

* :class:`CoordinatorPort` - the safety boundary. The engine hands it the
  step's proposals and the available actions; it returns the validations,
  executions and any error records, having executed accepted actions through
  the simulator's public API.
* :class:`StepScorer` - the reward boundary. The engine hands it the
  end-of-step ground-truth snapshots and the executions; it returns the per-step
  score. ``GroundTruthSnapshot`` appears here only as a type annotation
  (DESIGN 3.2 permits ``interfaces.ports`` to import ``evaluator_only`` for
  annotations only).

This module imports only :mod:`cyberswarm.contracts` and the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..contracts import (
    AgentId,
    AgentObservation,
    AvailableActions,
    EpisodeId,
    ErrorRecord,
    EventRecord,
    ExecutedAction,
    GroundTruthSnapshot,
    PolicyProposal,
    TerminationReason,
    Topology,
    ValidationResult,
)

__all__ = [
    "CoordinatorResult",
    "CoordinatorPort",
    "StepScore",
    "StepScorer",
]


@dataclass(frozen=True)
class CoordinatorResult:
    """What the coordinator produced for one step's proposals."""

    validations: tuple[ValidationResult, ...]
    executions: tuple[ExecutedAction, ...]
    errors: tuple[ErrorRecord, ...]


class CoordinatorPort(Protocol):
    """The engine's view of the defense coordinator (phases P3-P4).

    ``process`` accepts proposals as either
    :class:`~cyberswarm.contracts.PolicyProposal` objects or raw mappings (so a
    malformed proposal can be rejected with a reason code rather than crash),
    validates each in the defined order, and executes accepted actions through
    the simulator's public API. ``verify_recovery`` may add
    ``recovery_verified``/``recovery_failed`` events from agent-visible data.
    """

    def start_episode(self, episode_id: EpisodeId, topology: Topology) -> None: ...

    def process(
        self,
        proposals: Sequence[PolicyProposal | Mapping[str, Any]],
        *,
        step: int,
        available: Mapping[AgentId, AvailableActions],
    ) -> CoordinatorResult: ...

    def verify_recovery(
        self, step: int, observations: Mapping[AgentId, AgentObservation]
    ) -> list[EventRecord]: ...


@dataclass(frozen=True)
class StepScore:
    """One step's reward. ``raw`` holds the unweighted per-term magnitudes and
    ``weighted`` the signed contributions; ``total`` is their sum."""

    total: float
    raw: Mapping[str, float]
    weighted: Mapping[str, float]


class StepScorer(Protocol):
    """The engine's view of the reward model.

    ``score`` is called after each step closes, with the previous and current
    end-of-step ground-truth snapshots, the step's executions, the count of
    invalid (substituted) actions, and the termination reason when the episode
    ended on this step.
    """

    def reset(self, topology: Topology, max_steps: int) -> None: ...

    def score(
        self,
        *,
        step: int,
        prev: GroundTruthSnapshot | None,
        curr: GroundTruthSnapshot,
        executions: Sequence[ExecutedAction],
        invalid_count: int,
        termination_reason: TerminationReason | None,
    ) -> StepScore: ...
