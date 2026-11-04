"""Recovery verification from agent-visible data (stage S4, AD-20).

:class:`RecoveryVerifier` watches the nodes the coordinator has restored and
confirms, from agent-visible observations only, that each one actually came back
healthy. It tracks *executed* ``restore_node`` actions only (``quarantine`` and
``isolate`` are not verified, because their recovery is the later restore).

For each tracked restore (CONTRACTS section 8.6, AD-20):

1. It arms at step ``s``, the step of the ``restore_completed`` event for that
   node, read from the node's zone agent's observation. If no such event has
   arrived within ``restore_wait_steps`` of the execution, it emits
   ``recovery_failed`` and stops.
2. Once armed, at each step ``t >= s`` it reads the node's entry in the zone
   agent's observation. The step counts toward the streak when
   ``visible_status == up`` and ``service_health`` is non-null and at least
   ``health_min``; a null ``service_health`` or any other status resets the
   streak to 0 (a missing value is never treated as healthy).
3. When the streak reaches ``window`` consecutive steps, it emits
   ``recovery_verified`` (``producer = coordinator``, id prefix ``cev-``,
   carrying ``node_id`` and the restore ``execution_id``, visible to the node's
   zone agent) and stops.
4. If no ``recovery_verified`` has been emitted by step ``s + verify_timeout_steps``,
   it emits ``recovery_failed`` at that step and stops.

A verifier still armed when the episode ends emits nothing. The acting agent of
a ``restore_node`` is, by the authority rule, the node's own zone agent, so the
zone agent is read straight from the execution record; no topology or hidden
state is consulted. This module imports only :mod:`cyberswarm.contracts`, reads
no files and starts no external program.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from ..contracts import (
    ActionKind,
    AgentObservation,
    EventProducer,
    EventRecord,
    EventType,
    ExecutedAction,
    ExecutionStatus,
    NodeObservation,
    NodeStatus,
    Phase,
    event_id,
)

__all__ = ["RecoveryVerifier"]


@dataclass
class _Tracked:
    """The verification state of one restored node."""

    execution_id: str
    node_id: str
    zone_agent: str
    exec_step: int
    armed_step: int | None = None
    streak: int = 0
    resolved: bool = False


class RecoveryVerifier:
    """Verify restored nodes from agent-visible observations (AD-20).

    Defaults follow AD-20: ``health_min = 0.8``, ``window = 2``,
    ``verify_timeout_steps = 6``. ``restore_wait_steps`` is the
    ``restore_duration_steps + 2`` grace period for a ``restore_completed`` event
    to appear; the mock scenarios all restore in 3 steps, so it defaults to 5.
    The coordinator mints the ``cev-`` event ids and passes its own counter to
    :meth:`check`.
    """

    def __init__(
        self,
        *,
        health_min: float = 0.8,
        window: int = 2,
        verify_timeout_steps: int = 6,
        restore_wait_steps: int = 5,
    ) -> None:
        self._health_min = health_min
        self._window = window
        self._verify_timeout_steps = verify_timeout_steps
        self._restore_wait_steps = restore_wait_steps
        self._episode_id = ""
        self._tracked: list[_Tracked] = []

    def start_episode(self, episode_id: str) -> None:
        """Reset the verifier for a new episode."""

        self._episode_id = episode_id
        self._tracked = []

    def note_execution(self, execution: ExecutedAction) -> None:
        """Track an executed ``restore_node`` (ignores anything else)."""

        if execution.action.action_kind is not ActionKind.restore_node:
            return
        if execution.status is not ExecutionStatus.executed:
            return
        node_id = execution.action.target_node_id
        if node_id is None:
            return
        self._tracked.append(
            _Tracked(
                execution_id=execution.execution_id,
                node_id=node_id,
                zone_agent=execution.agent_id,
                exec_step=execution.step,
            )
        )

    def check(
        self,
        step: int,
        observations: Mapping[str, AgentObservation],
        event_counter: Iterator[int],
    ) -> list[EventRecord]:
        """Advance every tracked restore by one step; return any events emitted.

        ``event_counter`` yields the integers for the coordinator's ``cev-``
        event ids (``next(event_counter)`` per emitted event).
        """

        events: list[EventRecord] = []
        for tracked in self._tracked:
            if tracked.resolved:
                continue
            event = self._advance(tracked, step, observations, event_counter)
            if event is not None:
                events.append(event)
        return events

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _advance(
        self,
        tracked: _Tracked,
        step: int,
        observations: Mapping[str, AgentObservation],
        event_counter: Iterator[int],
    ) -> EventRecord | None:
        observation = observations.get(tracked.zone_agent)

        if tracked.armed_step is None:
            armed_at = (
                self._restore_completed_step(observation, tracked.node_id)
                if observation is not None
                else None
            )
            if armed_at is not None:
                tracked.armed_step = armed_at
                tracked.streak = 0
            elif step - tracked.exec_step >= self._restore_wait_steps:
                tracked.resolved = True
                return self._emit(EventType.recovery_failed, tracked, step, event_counter)

        if tracked.armed_step is not None:
            node_observation = self._node_observation(observation, tracked.node_id)
            if self._is_healthy(node_observation):
                tracked.streak += 1
            else:
                tracked.streak = 0
            if tracked.streak >= self._window:
                tracked.resolved = True
                return self._emit(
                    EventType.recovery_verified, tracked, step, event_counter
                )
            if step - tracked.armed_step >= self._verify_timeout_steps:
                tracked.resolved = True
                return self._emit(EventType.recovery_failed, tracked, step, event_counter)

        return None

    @staticmethod
    def _restore_completed_step(
        observation: AgentObservation, node_id: str
    ) -> int | None:
        """Return the step of the latest visible ``restore_completed`` for the node."""

        latest: int | None = None
        for event in observation.recent_events:
            if event.event_type is EventType.restore_completed and event.node_id == node_id:
                if latest is None or event.step > latest:
                    latest = event.step
        return latest

    @staticmethod
    def _node_observation(
        observation: AgentObservation | None, node_id: str
    ) -> NodeObservation | None:
        if observation is None:
            return None
        for node in observation.nodes:
            if node.node_id == node_id:
                return node
        return None

    def _is_healthy(self, node_observation: NodeObservation | None) -> bool:
        return (
            node_observation is not None
            and node_observation.visible_status is NodeStatus.up
            and node_observation.service_health is not None
            and node_observation.service_health >= self._health_min
        )

    def _emit(
        self,
        event_type: EventType,
        tracked: _Tracked,
        step: int,
        event_counter: Iterator[int],
    ) -> EventRecord:
        return EventRecord(
            event_id=event_id(next(event_counter), "cev"),
            episode_id=self._episode_id,
            step=step,
            phase=Phase.outcome,
            event_type=event_type,
            producer=EventProducer.coordinator,
            visible_to=[tracked.zone_agent],
            node_id=tracked.node_id,
            execution_id=tracked.execution_id,
        )
