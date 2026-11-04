"""Run the shared simulator conformance suite against the MOCK (task 5.3).

:class:`TestMockConformance` is the concrete ``Test*`` subclass of
:class:`~cyberswarm.interfaces.conformance.SimulatorConformance`. By supplying
the two hooks the mixin needs - :meth:`make_simulator` and :meth:`scenarios` -
it inherits every ``SC-1`` .. ``SC-15`` check as a real test method and runs
them against :class:`~cyberswarm.mock.MockSimulator` over the lead-owned
``topo-mock-small-v1`` topology (DESIGN section 4.2, DELTA checks DC-04/DC-06).

Two extra methods pin down the concrete behaviour that the generic SC-14/SC-15
checks describe but do not script explicitly:

* :meth:`test_sc14_restore_completed_timing` builds a scenario where a node is
  restored at step ``t`` and its ``restore_completed`` fires in P0 of ``t + 1``;
  it asserts the end-of-step-``t`` ground-truth snapshot still shows the node as
  ``restoring`` and compromised, and the next snapshot shows it recovered
  (AD-16).
* :meth:`test_sc15_cross_agent_investigation_hidden` has both zone agents
  investigate their own nodes in the same step and asserts neither agent's
  observation ever carries the other's ``action_applied`` or
  ``investigation_result`` event (AD-19).

The four ``scenarios()`` keys are built from the committed configs: ``base`` is
the shipped held-out scenario ``scn-mock-lateral-eval-01`` on v1;
``no_missing``/``all_missing`` are that scenario with ``missing_value_rate`` set
to ``0.0``/``1.0``; ``unsupported`` is a real scenario authored for the *other*
topology (``scn-mock-lateral-utopo-01`` on ``topo-mock-small-v2``), which a v1
mock must refuse with ``SCENARIO_UNSUPPORTED``.
"""

from __future__ import annotations

from pathlib import Path

from cyberswarm.contracts import (
    ActionKind,
    EventType,
    IncidentSpec,
    NodeStatus,
    ObservabilitySpec,
    ScenarioConfig,
    load_topology,
)
from cyberswarm.interfaces.conformance import SimulatorConformance
from cyberswarm.interfaces.simulator import GroundTruthAccessor, Simulator
from cyberswarm.mock import MockSimulator

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO / "configs"


def _load_scenario(name: str) -> ScenarioConfig:
    path = _CONFIGS / "scenarios" / "mock" / f"{name}.json"
    return ScenarioConfig.model_validate_json(path.read_text(encoding="utf-8"))


def _with_missing_rate(base: ScenarioConfig, rate: float, scenario_id: str) -> ScenarioConfig:
    """Return a copy of ``base`` with a different ``missing_value_rate`` and id."""

    data = base.model_dump()
    data["scenario_id"] = scenario_id
    data["observability"]["missing_value_rate"] = rate
    return ScenarioConfig.model_validate(data)


# The mock's fixed topology and the four conformance scenarios. The topology and
# scenarios are frozen contract models, so sharing these module-level instances
# across the fresh simulators that make_simulator() builds is safe.
_TOPOLOGY = load_topology(_CONFIGS / "topologies" / "topo-mock-small-v1.json")
_BASE = _load_scenario("scn-mock-lateral-eval-01")
_NO_MISSING = _with_missing_rate(_BASE, 0.0, "scn-mock-nomissing")
_ALL_MISSING = _with_missing_rate(_BASE, 1.0, "scn-mock-allmissing")
# A scenario built for topo-mock-small-v2; a v1 mock cannot host it.
_UNSUPPORTED = _load_scenario("scn-mock-lateral-utopo-01")

# A dedicated scenario for the explicit SC-14 restore-timing check: the entry
# node is compromised at step 1, nothing spreads (spread_probability 0.0), and a
# restore finishes in a single step so restore_completed lands in P0 of t+1.
_RESTORE_PROBE = ScenarioConfig(
    scenario_id="scn-mock-restore-probe",
    topology_id="topo-mock-small-v1",
    description="SC-14 probe restore completes in P0 of the next step",
    max_steps=10,
    observation_window_steps=5,
    incident=IncidentSpec(
        entry_node_id="node-web-02", start_step=1, spread_probability=0.0
    ),
    observability=ObservabilitySpec(
        alert_detect_probability=0.0,
        false_alert_rate=0.0,
        alert_delay_max_steps=0,
        missing_value_rate=0.0,
    ),
    restore_duration_steps=1,
    objective_node_id=None,
)

_ACTING_AGENT_ONLY = (EventType.action_applied, EventType.investigation_result)


class TestMockConformance(SimulatorConformance):
    """The SC-1..SC-15 conformance suite, bound to the MOCK simulator."""

    def make_simulator(self) -> tuple[Simulator, GroundTruthAccessor]:
        sim = MockSimulator(_TOPOLOGY)
        return sim, sim.ground_truth_accessor()

    def scenarios(self) -> dict[str, ScenarioConfig]:
        return {
            "base": _BASE,
            "no_missing": _NO_MISSING,
            "all_missing": _ALL_MISSING,
            "unsupported": _UNSUPPORTED,
        }

    # ------------------------------------------------------------------ #
    # SC-14, scripted: ground_truth() does not leak into t from P0 of t+1.
    # ------------------------------------------------------------------ #
    def test_sc14_restore_completed_timing(self) -> None:
        """AD-16: the step-t snapshot predates the t+1 ``restore_completed``."""

        sim, gt = self.make_simulator()
        episode_id = self._reset(sim, _RESTORE_PROBE, self._SEED_A)
        target = "node-web-02"

        # Step 0: a quiet step. Its P0 of step 1 compromises the entry node.
        sim.observe()
        self._apply_noops(sim, episode_id)
        result0 = sim.end_step()
        assert result0.step_closed == 0
        snapshot0 = gt.ground_truth()
        web02_0 = self._node(snapshot0, target)
        assert not web02_0.compromised and web02_0.status is NodeStatus.up
        sim.drain_events()

        # Step 1: restore the (now compromised) entry node. It enters
        # ``restoring`` this step; the restore finishes only in P0 of step 2.
        sim.observe()
        app = sim.apply_action(
            self._validated_action(
                episode_id, sim.step, "agent-dmz", ActionKind.restore_node, target
            )
        )
        assert app.status == "executed"
        for agent_id in sim.agent_ids:
            if agent_id != "agent-dmz":
                sim.apply_action(
                    self._validated_action(
                        episode_id, sim.step, agent_id, ActionKind.no_op
                    )
                )
        result1 = sim.end_step()
        assert result1.step_closed == 1

        # The end-of-step-1 snapshot still shows the node restoring and
        # compromised: the restore completion has not happened yet.
        snapshot1 = gt.ground_truth()
        assert snapshot1.step == 1
        web02_1 = self._node(snapshot1, target)
        assert web02_1.status is NodeStatus.restoring
        assert web02_1.compromised is True
        assert web02_1.contained is True

        # The restore_completed event is emitted in P0 of step 2 (= t + 1).
        restore_events = [
            event
            for event in sim.drain_events()
            if event.event_type is EventType.restore_completed and event.node_id == target
        ]
        assert len(restore_events) == 1
        assert restore_events[0].step == 2

        # Step 2: the next snapshot reflects the completed restore.
        sim.observe()
        self._apply_noops(sim, episode_id)
        result2 = sim.end_step()
        assert result2.step_closed == 2
        snapshot2 = gt.ground_truth()
        assert snapshot2.step == 2
        web02_2 = self._node(snapshot2, target)
        assert web02_2.status is NodeStatus.up
        assert web02_2.compromised is False
        sim.close()

    # ------------------------------------------------------------------ #
    # SC-15, scripted: two agents acting in one step stay invisible to each
    # other for action_applied / investigation_result events.
    # ------------------------------------------------------------------ #
    def test_sc15_cross_agent_investigation_hidden(self) -> None:
        """AD-19: an agent never sees another agent's action/investigation."""

        sim, _gt = self.make_simulator()
        episode_id = self._reset(sim, _BASE, self._SEED_A)

        # Every agent investigates one of its own nodes in the same step.
        acting_node: dict[str, str] = {}
        for agent_id in sim.agent_ids:
            own = sim.authority(agent_id)
            assert own
            acting_node[agent_id] = own[0]
            sim.apply_action(
                self._validated_action(
                    episode_id, sim.step, agent_id, ActionKind.investigate_node, own[0]
                )
            )
        assert len(acting_node) >= 2, "need at least two zone agents for a cross-check"

        sim.end_step()

        # The raw events are visible only to the agent that caused them.
        raw = sim.drain_events()
        acting_events = [e for e in raw if e.event_type in _ACTING_AGENT_ONLY]
        assert acting_events, "both agents acted, so these events must exist"
        for event in acting_events:
            assert event.agent_id is not None
            assert event.visible_to == [event.agent_id]

        # No agent's observation carries another agent's action/investigation.
        if not sim.done:
            observations = sim.observe()
            for agent_id, observation in observations.items():
                seen = [
                    event
                    for event in observation.recent_events
                    if event.event_type in _ACTING_AGENT_ONLY
                ]
                # The acting agent does see its own investigation echoed back.
                assert any(
                    event.node_id == acting_node[agent_id] for event in seen
                ), f"{agent_id} should see its own investigation_result"
                for event in seen:
                    assert event.agent_id == agent_id, (
                        f"{agent_id} sees {event.event_type} of {event.agent_id}"
                    )
        sim.close()

    # ------------------------------------------------------------------ #
    # Helper
    # ------------------------------------------------------------------ #
    @staticmethod
    def _node(snapshot, node_id: str):
        for node in snapshot.nodes:
            if node.node_id == node_id:
                return node
        raise AssertionError(f"{node_id} missing from the ground-truth snapshot")
