"""Reusable simulator conformance checks (stage S2, DESIGN section 4.2).

:class:`SimulatorConformance` is a *pytest mixin*. It collects the checks
``SC-1`` through ``SC-15`` that every simulator must pass - the lead-owned MOCK
and, later, Raksha's real simulator - as ordinary ``test_*`` methods. A concrete
test class subclasses it and implements two hooks:

* :meth:`make_simulator` -> ``(Simulator, GroundTruthAccessor)`` (a fresh pair
  each call), and
* :meth:`scenarios` -> ``dict[str, ScenarioConfig]`` with the keys ``base``,
  ``no_missing``, ``all_missing`` and ``unsupported``.

``tests/interfaces/test_conformance_mock.py`` instantiates it for the mock
(task 5.3); Raksha subclasses it under ``tests/simulator/``.

**Collection.** This class is deliberately *not* named ``Test*`` so pytest does
not collect it on its own (its hooks raise :class:`NotImplementedError`). Only a
``Test*`` subclass that supplies the hooks is collected and run. The class sets
no ``__test__`` flag, so subclasses are collected normally.

**Imports / construction.** The module imports only
:mod:`cyberswarm.contracts`, the sibling :mod:`.simulator` and the standard
library - no pytest, so it also satisfies the ``interfaces`` import boundary. It
constructs :class:`~cyberswarm.interfaces.simulator.ValidatedAction` directly in
:meth:`_validated_action`; that is permitted because this mixin is conformance
*test support* used only by test subclasses (DESIGN 3.2: "the mock may construct
it only in tests").
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from ..contracts import (
    ACTION_DEFS,
    EVALUATOR_ONLY_FIELD_NAMES,
    ActionKind,
    ActionSpec,
    AgentObservation,
    AvailableActions,
    ErrorCode,
    EventType,
    NodeScope,
    NodeStatus,
    ScenarioConfig,
    make_episode_id,
    proposal_id,
    validation_id,
)
from .simulator import (
    GroundTruthAccessor,
    Simulator,
    SimulatorError,
    ValidatedAction,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")

#: Statuses the ground-truth ``contained`` flag must agree with (CONTRACTS 9).
_CONTAINMENT_STATUSES = frozenset(
    {NodeStatus.quarantined, NodeStatus.isolated, NodeStatus.restoring}
)

#: Boundary-scope node measurements that are always withheld (CONTRACTS 8.2).
_PRIVATE_FIELDS = ("evidence_level", "service_health", "steps_since_investigated")

#: Event types visible only to the acting agent (CONTRACTS 8.1, SC-15).
_ACTING_AGENT_ONLY = frozenset(
    {EventType.action_applied, EventType.investigation_result}
)


class SimulatorConformance:
    """The SC-1..SC-15 checks shared by every simulator (see module docstring)."""

    # A synthetic policy id used only to build conformance episode ids.
    _POLICY_ID = "pol-conformance"
    _SEED_A = 20240101
    _SEED_B = 20240102
    # Steps compared for determinism (bounded so the suite stays fast).
    _DETERMINISM_STEPS = 12

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    def make_simulator(self) -> tuple[Simulator, GroundTruthAccessor]:
        """Return a fresh ``(Simulator, GroundTruthAccessor)`` pair."""

        raise NotImplementedError(
            "conformance subclasses must implement make_simulator()"
        )

    def scenarios(self) -> dict[str, ScenarioConfig]:
        """Return the scenarios keyed ``base``/``no_missing``/``all_missing``/``unsupported``."""

        raise NotImplementedError(
            "conformance subclasses must implement scenarios()"
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _scenario(self, key: str) -> ScenarioConfig:
        return self.scenarios()[key]

    def _reset(self, sim: Simulator, scenario: ScenarioConfig, seed: int) -> str:
        """Reset ``sim`` on ``scenario``/``seed`` and return the episode id."""

        episode_id = make_episode_id(self._POLICY_ID, scenario.scenario_id, seed)
        sim.reset(scenario, seed, episode_id)
        return episode_id

    def _validated_action(
        self,
        episode_id: str,
        step: int,
        agent_id: str,
        kind: ActionKind,
        target: str | None = None,
    ) -> ValidatedAction:
        """Build a :class:`ValidatedAction` for a conformance step.

        Only conformance test support builds these directly; production code
        goes through ``coordinator.validator`` (DESIGN 3.2).
        """

        return ValidatedAction(
            episode_id=episode_id,
            step=step,
            agent_id=agent_id,
            proposal_id=proposal_id(1),
            validation_id=validation_id(1),
            action=ActionSpec(action_kind=kind, target_node_id=target),
        )

    def _apply_noops(self, sim: Simulator, episode_id: str) -> None:
        """Apply a ``no_op`` for every agent at the current step."""

        for agent_id in sim.agent_ids:
            sim.apply_action(
                self._validated_action(episode_id, sim.step, agent_id, ActionKind.no_op)
            )

    @staticmethod
    def _scan_evaluator_fields(obj: Any) -> set[str]:
        """Return any evaluator-only field name found as a key in ``obj``."""

        found: set[str] = set()
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                if key in EVALUATOR_ONLY_FIELD_NAMES:
                    found.add(str(key))
                found |= SimulatorConformance._scan_evaluator_fields(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                found |= SimulatorConformance._scan_evaluator_fields(item)
        return found

    @staticmethod
    def _expect_simulator_error(
        fn: Callable[[], Any], code: ErrorCode | None = None
    ) -> None:
        """Assert ``fn()`` raises :class:`SimulatorError` (optionally ``code``)."""

        try:
            fn()
        except SimulatorError as exc:
            if code is not None:
                assert exc.code == code, f"expected {code}, got {exc.code}"
            return
        raise AssertionError("expected SimulatorError to be raised")

    @staticmethod
    def _assert_contiguous(event_ids: list[str], prefix: str) -> None:
        """Assert the numeric parts of ``prefix``-ids form a gapless run."""

        numbers = sorted(int(eid[len(prefix):]) for eid in event_ids)
        assert len(set(numbers)) == len(numbers), f"{prefix} ids repeat"
        for earlier, later in zip(numbers, numbers[1:]):
            assert later == earlier + 1, (
                f"{prefix} ids are not contiguous near {earlier}/{later}"
            )

    def _run_scripted(
        self, scenario: ScenarioConfig, seed: int, steps: int
    ) -> tuple[list, list, list]:
        """Run a fresh simulator with scripted ``no_op``s; return the sequences."""

        sim, _gt = self.make_simulator()
        episode_id = self._reset(sim, scenario, seed)
        observations: list = []
        events: list = []
        digests: list = []
        for _ in range(steps):
            if sim.done:
                break
            obs = sim.observe()
            observations.append(
                {aid: ob.model_dump(mode="json") for aid, ob in obs.items()}
            )
            self._apply_noops(sim, episode_id)
            sim.end_step()
            events.append([ev.model_dump(mode="json") for ev in sim.drain_events()])
            digests.append(sim.state_digest())
        sim.close()
        return observations, events, digests

    # ------------------------------------------------------------------ #
    # SC-1 .. SC-15
    # ------------------------------------------------------------------ #
    def test_sc01_node_ids_match_topology(self) -> None:
        """SC-1: node ids equal the topology ids, are canonical and stable."""

        sim, _gt = self.make_simulator()
        base = self._scenario("base")
        self._reset(sim, base, self._SEED_A)
        node_ids = list(sim.topology.node_ids)
        assert node_ids == sorted(node_ids), "node ids must be ascending"
        assert len(set(node_ids)) == len(node_ids), "node ids must be unique"

        # Authorities partition the node set, each ascending.
        union: set[str] = set()
        for agent_id in sim.agent_ids:
            authority = list(sim.authority(agent_id))
            assert authority == sorted(authority), "authority must be ascending"
            assert set(authority) <= set(node_ids)
            assert union.isdisjoint(authority), "zones must not overlap"
            union |= set(authority)
        assert union == set(node_ids), "authorities must cover every node"

        # Identical across a second reset (different seed).
        self._reset(sim, base, self._SEED_B)
        assert list(sim.topology.node_ids) == node_ids
        sim.close()

    def test_sc02_edges_resolve_and_are_ordered(self) -> None:
        """SC-2: every edge endpoint resolves; edge ids are unique and ordered."""

        sim, _gt = self.make_simulator()
        self._reset(sim, self._scenario("base"), self._SEED_A)
        node_ids = set(sim.topology.node_ids)
        edges = sim.topology.edges
        edge_ids = [edge.edge_id for edge in edges]
        assert edge_ids == sorted(edge_ids), "edge ids must be ascending"
        assert len(set(edge_ids)) == len(edge_ids), "edge ids must be unique"
        for edge in edges:
            assert edge.source in node_ids
            assert edge.target in node_ids
        sim.close()

    def test_sc03_unsupported_scenario(self) -> None:
        """SC-3: an unsupported scenario raises and leaves the sim unusable."""

        sim, _gt = self.make_simulator()
        unsupported = self._scenario("unsupported")
        episode_id = make_episode_id(
            self._POLICY_ID, unsupported.scenario_id, self._SEED_A
        )
        self._expect_simulator_error(
            lambda: sim.reset(unsupported, self._SEED_A, episode_id),
            ErrorCode.SCENARIO_UNSUPPORTED,
        )
        # Unusable until a valid reset.
        self._expect_simulator_error(lambda: sim.observe())
        # A valid reset restores usability.
        self._reset(sim, self._scenario("base"), self._SEED_A)
        obs = sim.observe()
        assert set(obs.keys()) == set(sim.agent_ids)
        sim.close()

    def test_sc04_step_semantics(self) -> None:
        """SC-4: step starts at 0, observations carry the step, end_step advances."""

        sim, _gt = self.make_simulator()
        base = self._scenario("base")
        episode_id = self._reset(sim, base, self._SEED_A)
        assert sim.step == 0

        obs = sim.observe()
        assert set(obs.keys()) == set(sim.agent_ids)
        for agent_id, observation in obs.items():
            assert observation.step == sim.step
            assert observation.agent_id == agent_id

        before = sim.step
        self._apply_noops(sim, episode_id)
        result = sim.end_step()
        assert result.step_closed == before
        if not sim.done:
            assert sim.step == before + 1

        # Run to done, then end_step must raise.
        cap = base.max_steps + 2
        guard = 0
        while not sim.done and guard <= cap:
            sim.observe()
            self._apply_noops(sim, episode_id)
            sim.end_step()
            guard += 1
        assert sim.done, "episode did not finish within max_steps"
        self._expect_simulator_error(lambda: sim.end_step())
        sim.close()

    def test_sc05_determinism(self) -> None:
        """SC-5: same seed reproduces everything; a different seed changes digests."""

        base = self._scenario("base")
        steps = min(base.max_steps, self._DETERMINISM_STEPS)
        obs_a1, ev_a1, dg_a1 = self._run_scripted(base, self._SEED_A, steps)
        obs_a2, ev_a2, dg_a2 = self._run_scripted(base, self._SEED_A, steps)
        assert obs_a1 == obs_a2, "observations are not reproducible for a fixed seed"
        assert ev_a1 == ev_a2, "events are not reproducible for a fixed seed"
        assert dg_a1 == dg_a2, "state digests are not reproducible for a fixed seed"

        _obs_b, _ev_b, dg_b = self._run_scripted(base, self._SEED_B, steps)
        assert dg_a1 != dg_b, "a different seed must change the digest sequence"

    def test_sc06_records_validate_without_evaluator_fields(self) -> None:
        """SC-6: observations and available actions validate and leak no truth."""

        sim, _gt = self.make_simulator()
        self._reset(sim, self._scenario("base"), self._SEED_A)
        obs = sim.observe()
        for agent_id, observation in obs.items():
            dumped = observation.model_dump(mode="json")
            AgentObservation.model_validate(observation.model_dump())
            assert not self._scan_evaluator_fields(dumped), "observation leaks truth"

            available = sim.available_actions(agent_id)
            AvailableActions.model_validate(available.model_dump())
            assert not self._scan_evaluator_fields(available.model_dump(mode="json"))
        sim.close()

    def test_sc07_no_future_events_and_contiguous_ids(self) -> None:
        """SC-7: no event is from a future step; event ids are contiguous per prefix."""

        sim, _gt = self.make_simulator()
        base = self._scenario("base")
        episode_id = self._reset(sim, base, self._SEED_A)
        collected = []
        cap = base.max_steps + 2
        guard = 0
        while not sim.done and guard <= cap:
            obs = sim.observe()
            for observation in obs.values():
                for event in observation.recent_events:
                    assert event.step <= observation.step, "future event in observation"
            self._apply_noops(sim, episode_id)
            sim.end_step()
            collected.extend(sim.drain_events())
            guard += 1
        collected.extend(sim.drain_events())

        self._assert_contiguous(
            [e.event_id for e in collected if e.event_id.startswith("evt-")], "evt-"
        )
        self._assert_contiguous(
            [e.event_id for e in collected if e.event_id.startswith("cev-")], "cev-"
        )
        sim.close()

    def test_sc08_missing_value_semantics(self) -> None:
        """SC-8: rate 1.0 withholds (null, not 0.0); rate 0.0 fills; boundary is null."""

        # rate 1.0: own-node measurements are null, never 0.0.
        sim, _gt = self.make_simulator()
        self._reset(sim, self._scenario("all_missing"), self._SEED_A)
        for observation in sim.observe().values():
            for node in observation.nodes:
                if node.scope is NodeScope.own:
                    assert node.service_health is None, "missing value must be null"
                else:
                    for field in _PRIVATE_FIELDS:
                        assert getattr(node, field) is None
        sim.close()

        # rate 0.0: own-node measurements are present; boundary still null.
        sim2, _gt2 = self.make_simulator()
        self._reset(sim2, self._scenario("no_missing"), self._SEED_A)
        for observation in sim2.observe().values():
            for node in observation.nodes:
                if node.scope is NodeScope.own:
                    assert node.service_health is not None
                else:
                    for field in _PRIVATE_FIELDS:
                        assert getattr(node, field) is None
        sim2.close()

    def test_sc09_inapplicable_action_fails_without_state_change(self) -> None:
        """SC-9: inapplicable/out-of-authority actions fail and leave state unchanged."""

        sim, _gt = self.make_simulator()
        episode_id = self._reset(sim, self._scenario("base"), self._SEED_A)
        agent0 = sim.agent_ids[0]
        own = list(sim.authority(agent0))
        assert own, "the first agent must own at least one node"

        # Out of authority: a target in another zone (when one exists).
        foreign = [n for n in sim.topology.node_ids if n not in set(own)]
        if foreign:
            before = sim.state_digest()
            app = sim.apply_action(
                self._validated_action(
                    episode_id, sim.step, agent0, ActionKind.investigate_node, foreign[0]
                )
            )
            assert app.status == "failed"
            assert app.error_code is not None
            assert sim.state_digest() == before

        # Inapplicable by state: isolate the same own node twice.
        target = own[0]
        first = sim.apply_action(
            self._validated_action(
                episode_id, sim.step, agent0, ActionKind.isolate_node, target
            )
        )
        assert first.status == "executed"
        before = sim.state_digest()
        second = sim.apply_action(
            self._validated_action(
                episode_id, sim.step, agent0, ActionKind.isolate_node, target
            )
        )
        assert second.status == "failed"
        assert second.error_code is not None
        assert sim.state_digest() == before
        sim.close()

    def test_sc10_observe_hides_truth_and_ground_truth_covers_nodes(self) -> None:
        """SC-10: observations expose no truth; ground truth covers all nodes."""

        sim, gt = self.make_simulator()
        episode_id = self._reset(sim, self._scenario("base"), self._SEED_A)
        for observation in sim.observe().values():
            assert not self._scan_evaluator_fields(observation.model_dump(mode="json"))
            assert not hasattr(observation, "compromised")

        self._apply_noops(sim, episode_id)
        result = sim.end_step()
        snapshot = gt.ground_truth()
        assert snapshot.step == result.step_closed
        assert [n.node_id for n in snapshot.nodes] == list(sim.topology.node_ids)
        for node in snapshot.nodes:
            assert node.contained == (node.status in _CONTAINMENT_STATUSES)
        sim.close()

    def test_sc11_termination(self) -> None:
        """SC-11: termination is reported; the final observation is step last+1."""

        sim, _gt = self.make_simulator()
        base = self._scenario("base")
        episode_id = self._reset(sim, base, self._SEED_A)
        cap = base.max_steps + 2
        last_result = None
        guard = 0
        while not sim.done and guard <= cap:
            sim.observe()
            self._apply_noops(sim, episode_id)
            last_result = sim.end_step()
            guard += 1
        assert sim.done and last_result is not None
        assert last_result.terminated or last_result.truncated

        final = sim.observe()
        for observation in final.values():
            assert observation.step == last_result.step_closed + 1

        # apply_action after done is a failed result or a SimulatorError.
        try:
            app = sim.apply_action(
                self._validated_action(
                    episode_id, last_result.step_closed, sim.agent_ids[0], ActionKind.no_op
                )
            )
            assert app.status == "failed"
        except SimulatorError:
            pass
        sim.close()

    def test_sc12_available_actions_match_vocabulary(self) -> None:
        """SC-12: available actions match the vocabulary, start with no_op, in-zone."""

        sim, _gt = self.make_simulator()
        self._reset(sim, self._scenario("base"), self._SEED_A)
        for agent_id in sim.agent_ids:
            available = sim.available_actions(agent_id)
            assert available.actions, "must list at least no_op"
            assert available.actions[0].action_kind is ActionKind.no_op
            own = set(sim.authority(agent_id))
            for action in available.actions:
                spec = ACTION_DEFS[action.action_kind]
                assert action.impact_class is spec.impact_class
                assert action.high_impact == spec.high_impact
                assert action.cost_units == spec.cost_units
                if action.target_node_id is not None:
                    assert action.target_node_id in own, "target out of the agent's zone"
        sim.close()

    def test_sc13_kind_and_state_digest(self) -> None:
        """SC-13: ``kind`` is valid; ``state_digest`` is hex64 and reacts to change."""

        sim, _gt = self.make_simulator()
        episode_id = self._reset(sim, self._scenario("base"), self._SEED_A)
        assert sim.kind in ("mock", "simulator")

        before = sim.state_digest()
        assert _HEX64.match(before), "state_digest must be 64 lowercase hex chars"

        agent0 = sim.agent_ids[0]
        target = sim.authority(agent0)[0]
        app = sim.apply_action(
            self._validated_action(
                episode_id, sim.step, agent0, ActionKind.isolate_node, target
            )
        )
        assert app.status == "executed"
        after = sim.state_digest()
        assert _HEX64.match(after)
        assert before != after, "state_digest must change when state changes"
        sim.close()

    def test_sc14_ground_truth_timing(self) -> None:
        """SC-14: ground_truth() tracks step_closed and is stable until end_step."""

        sim, gt = self.make_simulator()
        base = self._scenario("base")
        episode_id = self._reset(sim, base, self._SEED_A)
        cap = min(base.max_steps, 6)
        previous = None
        for _ in range(cap):
            if sim.done:
                break
            closing = sim.step
            sim.observe()
            self._apply_noops(sim, episode_id)
            result = sim.end_step()
            assert result.step_closed == closing
            snapshot = gt.ground_truth()
            assert snapshot.step == result.step_closed
            # Stable until the next end_step().
            assert gt.ground_truth().step == result.step_closed
            assert [n.node_id for n in snapshot.nodes] == list(sim.topology.node_ids)
            if previous is not None:
                assert result.step_closed == previous + 1
            previous = result.step_closed
        sim.close()

    def test_sc15_event_visibility_no_cross_agent(self) -> None:
        """SC-15: action_applied/investigation_result reach only the acting agent."""

        sim, _gt = self.make_simulator()
        episode_id = self._reset(sim, self._scenario("base"), self._SEED_A)
        # Each agent investigates one of its own nodes at step 0.
        for agent_id in sim.agent_ids:
            own = sim.authority(agent_id)
            if own:
                sim.apply_action(
                    self._validated_action(
                        episode_id, sim.step, agent_id, ActionKind.investigate_node, own[0]
                    )
                )
        sim.end_step()

        # The raw events must be visible only to their acting agent.
        for event in sim.drain_events():
            if event.event_type in _ACTING_AGENT_ONLY:
                assert event.agent_id is not None
                assert event.visible_to == [event.agent_id], (
                    f"{event.event_type} must be visible only to {event.agent_id}"
                )

        # No agent's observation carries another agent's action/investigation.
        if not sim.done:
            for agent_id, observation in sim.observe().items():
                for event in observation.recent_events:
                    if event.event_type in _ACTING_AGENT_ONLY:
                        assert event.agent_id == agent_id, (
                            f"agent {agent_id} sees {event.event_type} of {event.agent_id}"
                        )
        sim.close()
