"""MOCK simulator - the deterministic lead-owned simulator (task 5.2).

**This module is a MOCK, not the real simulator.** :class:`MockSimulator`
(``kind = "mock"``, ``simulator_kind = SimulatorKind.mock``) is the project
lead's clearly-labelled stand-in that lets the RL environment, coordinator,
uncertainty wrapper and smoke path run before Raksha delivers the real
simulator. It lives outside any ``simulator/`` package, implements *only* the
public :class:`~cyberswarm.interfaces.Simulator` and
:class:`~cyberswarm.interfaces.GroundTruthAccessor` protocols, and imports **no**
``cyberswarm.simulator`` and no other simulator's internals. Every "attack" and
"defense" is an abstract state transition: there is no shell, subprocess,
socket, network call, or file mutation anywhere in this package.

Design sources: DESIGN section 5 (mock scope and dynamics), CONTRACTS section 5
(step semantics), AD-16 (ground-truth timing), AD-17 (two availabilities),
AD-18 (weakened ``service_health``), AD-19 (event visibility), AD-20 (recovery,
verified by the coordinator, not here).

This module wires the phases together; the dynamics, effects, observation
building and ground-truth capture live in the sibling modules. It imports only
:mod:`cyberswarm.contracts`, :mod:`cyberswarm.interfaces`,
:mod:`cyberswarm.common`, the sibling mock modules and the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from ..common.hashing import sha256_json
from ..contracts import (
    ActionKind,
    AgentObservation,
    AvailableActions,
    ErrorCode,
    EventProducer,
    EventRecord,
    EventType,
    GroundTruthSnapshot,
    NodeStatus,
    Phase,
    ScenarioConfig,
    SimulatorKind,
    StatusChange,
    TerminationReason,
    Topology,
    event_id,
    execution_id,
    load_topology,
)
from ..interfaces.simulator import (
    ActionApplication,
    GroundTruthAccessor,
    SimulatorError,
    StepResult,
    ValidatedAction,
)
from .dynamics import CommonRandomTables, run_p0
from .effects import NodeState, apply_action_effect, service_availabilities
from .observation import TopologyIndex, build_available_actions, build_observation
from .truth import capture_snapshot

__all__ = ["MockSimulator"]

_INVESTIGATION_KINDS = frozenset(
    {ActionKind.investigate_node, ActionKind.request_evidence}
)


class _MockGroundTruthAccessor:
    """A separate evaluator-only view onto a :class:`MockSimulator`.

    Handing this out instead of the simulator itself keeps ground truth off any
    decision-time simulator reference (CONTRACTS section 9): a holder of this
    object can read truth but cannot step the episode.
    """

    __slots__ = ("_sim",)

    def __init__(self, sim: "MockSimulator") -> None:
        self._sim = sim

    def ground_truth(self) -> GroundTruthSnapshot:
        return self._sim.ground_truth()

    def termination_reason(self) -> TerminationReason | None:
        return self._sim.termination_reason()


class MockSimulator:
    """Deterministic MOCK simulator over one fixed topology (DESIGN section 5).

    Construct it with a :class:`~cyberswarm.contracts.Topology` (or load one by
    id with :meth:`from_topology_id`), then drive the episode loop ``reset`` ->
    (``observe`` -> ``available_actions`` -> ``apply_action`` -> ``end_step``)*.
    It is reproducible for a fixed ``(scenario, seed)`` and differs for a
    different seed.
    """

    #: Satisfies the ``Simulator`` protocol's ``kind`` member.
    kind: Literal["mock", "simulator"] = "mock"
    #: The manifest-facing simulator kind; a mock run is never a research result.
    simulator_kind: SimulatorKind = SimulatorKind.mock

    def __init__(self, topology: Topology) -> None:
        self._topology = topology
        self._index = TopologyIndex(topology)
        self._active = False

        # Per-episode state (populated by reset()).
        self._scenario: ScenarioConfig | None = None
        self._seed = 0
        self._episode_id = ""
        self._tables: CommonRandomTables | None = None
        self._node_states: dict[str, NodeState] = {}
        self._pending_alerts: list = []
        self._events: list[EventRecord] = []
        self._drain_index = 0
        self._evt_counter = 0
        self._exe_counter = 0
        self._step = 0
        self._done = False
        self._terminated = False
        self._truncated = False
        self._termination_reason: TerminationReason | None = None
        self._snapshot: GroundTruthSnapshot | None = None
        self._step_new_compromises: list = []
        self._step_alert_truth: list = []
        self._p1_cutoff = 0

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_topology_id(
        cls, topology_id: str, *, topologies_dir: Path | None = None
    ) -> "MockSimulator":
        """Load ``configs/topologies/<topology_id>.json`` via the contracts.

        ``topologies_dir`` defaults to the repository's ``configs/topologies``
        directory, resolved relative to this file so no absolute path is baked
        into committed code.
        """

        if topologies_dir is None:
            topologies_dir = Path(__file__).resolve().parents[3] / "configs" / "topologies"
        topology = load_topology(Path(topologies_dir) / f"{topology_id}.json")
        return cls(topology)

    def ground_truth_accessor(self) -> GroundTruthAccessor:
        """Return a separate evaluator-only accessor onto this simulator."""

        return _MockGroundTruthAccessor(self)

    # ------------------------------------------------------------------ #
    # Read-only views
    # ------------------------------------------------------------------ #
    @property
    def topology(self) -> Topology:
        return self._topology

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return self._index.agents

    def authority(self, agent_id: str) -> tuple[str, ...]:
        """Node ids of the agent's zone, ascending (its authority)."""

        return self._index.own_nodes.get(agent_id, ())

    @property
    def step(self) -> int:
        return self._step

    @property
    def done(self) -> bool:
        return self._done

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, scenario: ScenarioConfig, seed: int, episode_id: str) -> None:
        """Start a new episode (DESIGN section 5, CONTRACTS section 5).

        Raises ``SimulatorError(SCENARIO_UNSUPPORTED)`` for a scenario this
        topology cannot host, and leaves the simulator unusable until a
        successful reset.
        """

        self._active = False
        if scenario.topology_id != self._topology.topology_id:
            raise SimulatorError(
                ErrorCode.SCENARIO_UNSUPPORTED,
                f"scenario topology {scenario.topology_id!r} is not {self._topology.topology_id!r}",
            )
        node_ids = set(self._index.node_order)
        if scenario.incident.entry_node_id not in node_ids:
            raise SimulatorError(
                ErrorCode.SCENARIO_UNSUPPORTED,
                f"entry node {scenario.incident.entry_node_id!r} is not in the topology",
            )
        if scenario.objective_node_id is not None and scenario.objective_node_id not in node_ids:
            raise SimulatorError(
                ErrorCode.SCENARIO_UNSUPPORTED,
                f"objective node {scenario.objective_node_id!r} is not in the topology",
            )

        self._scenario = scenario
        self._seed = int(seed)
        self._episode_id = episode_id
        self._tables = CommonRandomTables(
            seed, scenario.max_steps, len(self._index.node_order), len(self._index.edge_order)
        )
        self._node_states = {
            node_id: NodeState(node_id=node_id) for node_id in self._index.node_order
        }
        self._pending_alerts = []
        self._events = []
        self._drain_index = 0
        self._evt_counter = 0
        self._exe_counter = 0
        self._step = 0
        self._done = False
        self._terminated = False
        self._truncated = False
        self._termination_reason = None
        self._snapshot = None
        self._step_new_compromises = []
        self._step_alert_truth = []

        # P0 of step 0 is skipped except for episode_started (CONTRACTS 5).
        self._emit(EventType.episode_started, Phase.lifecycle, step=0)
        self._p1_cutoff = len(self._events)
        self._active = True

    def observe(self) -> Mapping[str, AgentObservation]:
        """Agent-visible observations for the current step (phase P1).

        Idempotent within a step (built from the event history frozen at the end
        of P0). After the episode is done it returns the single final
        observation with ``step = last step + 1``.
        """

        self._require_active()
        assert self._scenario is not None and self._tables is not None
        obs_step = self._step + 1 if self._done else self._step
        window = self._scenario.observation_window_steps
        horizon = self._scenario.max_steps
        missing_rate = self._scenario.observability.missing_value_rate
        row = min(obs_step, horizon - 1)

        service_avail = service_availabilities(self._node_states, self._index.dependencies)
        health_noise = {
            node_id: float(self._tables.health_noise[row, self._index.node_index[node_id]])
            for node_id in self._index.node_order
        }
        node_missing = {
            node_id: float(self._tables.missing_node[row, self._index.node_index[node_id]])
            < missing_rate
            for node_id in self._index.node_order
        }
        edge_noise = {
            edge_id: float(self._tables.traffic_noise[row, self._index.edge_index[edge_id]])
            for edge_id in self._index.edge_order
        }
        edge_missing = {
            edge_id: float(self._tables.missing_edge[row, self._index.edge_index[edge_id]])
            < missing_rate
            for edge_id in self._index.edge_order
        }

        events_view = self._events[: self._p1_cutoff]
        observations: dict[str, AgentObservation] = {}
        for agent_id in self._index.agents:
            observations[agent_id] = build_observation(
                agent_id=agent_id,
                step=obs_step,
                max_steps=horizon,
                window=window,
                episode_id=self._episode_id,
                zone_id=self._index.zone_of_agent[agent_id],
                index=self._index,
                node_states=self._node_states,
                service_avail=service_avail,
                events=events_view,
                health_noise=health_noise,
                node_missing=node_missing,
                edge_noise=edge_noise,
                edge_missing=edge_missing,
            )
        return observations

    def available_actions(self, agent_id: str) -> AvailableActions:
        """State-dependent available actions for one agent this step."""

        self._require_active()
        if agent_id not in self._index.agents:
            raise SimulatorError(
                ErrorCode.SIMULATOR_FAILURE, f"unknown agent {agent_id!r}"
            )
        obs_step = self._step + 1 if self._done else self._step
        return build_available_actions(
            agent_id=agent_id,
            step=obs_step,
            episode_id=self._episode_id,
            index=self._index,
            node_states=self._node_states,
            final=self._done,
        )

    def apply_action(self, action: ValidatedAction) -> ActionApplication:
        """Apply one validated action (phase P4).

        Never raises for an inapplicable or out-of-authority action: it returns
        a ``failed`` :class:`ActionApplication` that emits no event and leaves
        ``state_digest()`` unchanged (conformance SC-9).
        """

        self._require_active()
        assert self._tables is not None and self._scenario is not None
        if self._done:
            return self._failed()

        agent_id = action.agent_id
        spec = action.action
        kind = spec.action_kind
        target = spec.target_node_id

        if agent_id not in self._index.agents:
            return self._failed()

        if kind is ActionKind.no_op:
            exid = self._next_execution_id()
            applied_event = self._emit(
                EventType.action_applied,
                Phase.outcome,
                step=self._step,
                node_id=None,
                acting_agent=agent_id,
                action_kind=ActionKind.no_op,
                agent_id=agent_id,
                proposal_id=action.proposal_id,
                execution_id=exid,
            )
            return ActionApplication(
                status="executed",
                error_code=None,
                affected_node_ids=(),
                status_changes=(),
                outcome_event_ids=(applied_event.event_id,),
            )

        # Targeted action: the target must be in the acting agent's zone.
        if target is None or target not in self._index.own_nodes.get(agent_id, ()):
            return self._failed()

        if kind in _INVESTIGATION_KINDS:
            table_idx = 1 if kind is ActionKind.request_evidence else 0
            evidence_uniform = float(
                self._tables.evidence[self._step, self._index.node_index[target], table_idx]
            )
        else:
            evidence_uniform = 0.0

        outcome = apply_action_effect(
            self._node_states[target],
            kind,
            step=self._step,
            duration_steps=spec.parameters.duration_steps,
            level=spec.parameters.level,
            restore_duration_steps=self._scenario.restore_duration_steps,
            evidence_uniform=evidence_uniform,
        )
        if not outcome.applied:
            return self._failed(outcome.error_code)

        exid = self._next_execution_id()
        event_ids: list[str] = []
        applied_event = self._emit(
            EventType.action_applied,
            Phase.outcome,
            step=self._step,
            node_id=target,
            acting_agent=agent_id,
            action_kind=kind,
            agent_id=agent_id,
            proposal_id=action.proposal_id,
            execution_id=exid,
        )
        event_ids.append(applied_event.event_id)

        status_changes: tuple[StatusChange, ...] = ()
        if kind in _INVESTIGATION_KINDS:
            result_event = self._emit(
                EventType.investigation_result,
                Phase.outcome,
                step=self._step,
                node_id=target,
                acting_agent=agent_id,
                evidence_level=outcome.investigation_evidence,
                agent_id=agent_id,
                execution_id=exid,
            )
            event_ids.append(result_event.event_id)
        elif outcome.status_changed:
            change_event = self._emit(
                EventType.node_status_changed,
                Phase.outcome,
                step=self._step,
                node_id=target,
                status_from=outcome.status_from,
                status_to=outcome.status_to,
                execution_id=exid,
            )
            event_ids.append(change_event.event_id)
            status_changes = (
                StatusChange(
                    node_id=target,
                    status_from=outcome.status_from,
                    status_to=outcome.status_to,
                ),
            )

        return ActionApplication(
            status="executed",
            error_code=None,
            affected_node_ids=outcome.affected,
            status_changes=status_changes,
            outcome_event_ids=tuple(event_ids),
        )

    def end_step(self) -> StepResult:
        """Close the current step (DESIGN section 5, AD-16).

        Captures the end-of-step-``t`` ground-truth snapshot *before* running P0
        of ``t+1``, evaluates termination on that state, and advances when the
        episode continues. Raises ``SimulatorError`` if called after the episode
        is done.
        """

        self._require_active()
        assert self._scenario is not None
        if self._done:
            raise SimulatorError(
                ErrorCode.SIMULATOR_FAILURE, "end_step called after the episode is done"
            )

        closing = self._step
        incident_active = closing >= self._scenario.incident.start_step
        self._snapshot = capture_snapshot(
            episode_id=self._episode_id,
            step=closing,
            node_order=self._index.node_order,
            node_states=self._node_states,
            dependencies=self._index.dependencies,
            incident_active=incident_active,
            new_compromises=self._step_new_compromises,
            alert_truth=self._step_alert_truth,
        )

        terminated, truncated, reason = self._evaluate_termination(closing)
        self._terminated = terminated
        self._truncated = truncated

        if terminated or truncated:
            self._done = True
            self._termination_reason = reason
            self._emit(
                EventType.episode_ended,
                Phase.lifecycle,
                step=closing,
                terminated=terminated,
                truncated=truncated,
            )
        else:
            self._termination_reason = None
            self._step = closing + 1
            self._run_p0(self._step)

        self._p1_cutoff = len(self._events)
        return StepResult(step_closed=closing, terminated=terminated, truncated=truncated)

    def drain_events(self) -> list[EventRecord]:
        """Events emitted since the last drain, in emission order."""

        drained = self._events[self._drain_index :]
        self._drain_index = len(self._events)
        return list(drained)

    def state_digest(self) -> str:
        """SHA-256 hex (64 lowercase chars) of the canonical internal state."""

        self._require_active()
        payload = {
            "episode_id": self._episode_id,
            "seed": self._seed,
            "step": self._step,
            "done": self._done,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "evt_counter": self._evt_counter,
            "exe_counter": self._exe_counter,
            "nodes": [self._node_digest(node_id) for node_id in self._index.node_order],
            "pending_alerts": sorted(
                [
                    [
                        alert.emit_step,
                        alert.node_id,
                        alert.severity.value,
                        alert.alert_category.value,
                        alert.true_positive,
                    ]
                    for alert in self._pending_alerts
                ]
            ),
        }
        return sha256_json(payload)

    def close(self) -> None:
        """Release episode resources; the simulator is unusable until reset."""

        self._active = False
        self._tables = None

    # ------------------------------------------------------------------ #
    # GroundTruthAccessor
    # ------------------------------------------------------------------ #
    def ground_truth(self) -> GroundTruthSnapshot:
        """Return the end-of-step snapshot of the most recently closed step."""

        if self._snapshot is None:
            raise SimulatorError(
                ErrorCode.SIMULATOR_FAILURE,
                "no ground-truth snapshot yet; call end_step() first",
            )
        return self._snapshot

    def termination_reason(self) -> TerminationReason | None:
        """Why the closed step ended, or ``None`` if the episode continues."""

        return self._termination_reason

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _require_active(self) -> None:
        if not self._active:
            raise SimulatorError(
                ErrorCode.SIMULATOR_FAILURE, "simulator is not active; call reset()"
            )

    def _failed(
        self, error_code: ErrorCode = ErrorCode.ACTION_EXECUTION_FAILED
    ) -> ActionApplication:
        return ActionApplication(
            status="failed",
            error_code=error_code,
            affected_node_ids=(),
            status_changes=(),
            outcome_event_ids=(),
        )

    def _next_execution_id(self) -> str:
        self._exe_counter += 1
        return execution_id(self._exe_counter)

    def _emit(
        self,
        event_type: EventType,
        phase: Phase,
        *,
        step: int,
        node_id: str | None = None,
        acting_agent: str | None = None,
        **fields: object,
    ) -> EventRecord:
        """Assign the next ``evt-`` id, compute ``visible_to`` and record the event."""

        self._evt_counter += 1
        visible_to = self._index.event_visible_to(
            event_type, node_id=node_id, acting_agent=acting_agent
        )
        event = EventRecord(
            event_id=event_id(self._evt_counter, "evt"),
            episode_id=self._episode_id,
            step=step,
            phase=phase,
            event_type=event_type,
            producer=EventProducer.simulator,
            visible_to=visible_to,
            node_id=node_id,
            **fields,
        )
        self._events.append(event)
        return event

    def _run_p0(self, step: int) -> None:
        """Run phase P0 of ``step`` and emit its events in the fixed id order."""

        assert self._tables is not None and self._scenario is not None
        result = run_p0(
            self._node_states,
            self._pending_alerts,
            self._tables,
            self._scenario,
            self._index,
            step,
        )
        for node_id in result.restores_completed:
            self._emit(
                EventType.restore_completed,
                Phase.dynamics,
                step=step,
                node_id=node_id,
                status_from=NodeStatus.restoring,
                status_to=NodeStatus.up,
            )
        alert_truth: list[tuple[str, bool]] = []
        for alert in result.due_alerts:
            event = self._emit(
                EventType.alert_raised,
                Phase.dynamics,
                step=step,
                node_id=alert.node_id,
                severity=alert.severity,
                alert_category=alert.alert_category,
            )
            alert_truth.append((event.event_id, alert.true_positive))
        self._step_new_compromises = result.new_compromises
        self._step_alert_truth = alert_truth

    def _evaluate_termination(
        self, closing: int
    ) -> tuple[bool, bool, TerminationReason | None]:
        assert self._scenario is not None
        states = self._node_states
        objective = self._scenario.objective_node_id
        if objective is not None and states[objective].compromised:
            return True, False, TerminationReason.objective_reached
        incident_started = closing >= self._scenario.incident.start_step
        if (
            incident_started
            and all(not state.compromised for state in states.values())
            and all(state.status is NodeStatus.up for state in states.values())
        ):
            return True, False, TerminationReason.incident_resolved
        if closing == self._scenario.max_steps - 1:
            return False, True, TerminationReason.max_steps
        return False, False, None

    def _node_digest(self, node_id: str) -> list[object]:
        state = self._node_states[node_id]
        return [
            state.node_id,
            state.status.value,
            state.compromised,
            state.restore_left,
            state.rate_level.value if state.rate_level is not None else None,
            state.monitor_left,
            state.evidence_level,
            state.evidence_step,
            state.last_investigated,
        ]
