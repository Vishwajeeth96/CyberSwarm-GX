"""Shared setup for the episode-engine tests (task 7.1).

Provides the lead-owned ``topo-mock-small-v1`` topology, two short inline
scenarios (a truncating one and one that terminates on ``objective_reached``),
an :class:`~cyberswarm.engine.EpisodeEngine` factory wired to a
:class:`~cyberswarm.mock.MockSimulator` and the
:class:`tests.engine.doubles.AcceptAllCoordinator`, two deterministic policies,
and a small driver that runs an episode to its end.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from cyberswarm.contracts import (
    ActionKind,
    ActionSpec,
    EpisodeRecordSet,
    IncidentSpec,
    ObservabilitySpec,
    PolicyProposal,
    ScenarioConfig,
    Topology,
    load_topology,
)
from cyberswarm.engine import EpisodeEngine
from cyberswarm.engine.recorder import MemoryRecorder
from cyberswarm.interfaces import DecisionContext
from cyberswarm.mock import MockSimulator

from .doubles import AcceptAllCoordinator

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOPOLOGY_V1 = REPO_ROOT / "configs" / "topologies" / "topo-mock-small-v1.json"

#: Policy id used by the engine (and the stand-in proposals) in these tests.
POLICY_ID = "pol-engine-test"


def _quiet_observability() -> ObservabilitySpec:
    """Observability with some alerts but no missing values (reproducible)."""

    return ObservabilitySpec(
        alert_detect_probability=0.5,
        false_alert_rate=0.05,
        alert_delay_max_steps=0,
        missing_value_rate=0.0,
    )


#: A short episode that never terminates and truncates at step max_steps-1.
SCENARIO_TRUNCATE = ScenarioConfig(
    scenario_id="scn-mock-engine-short",
    topology_id="topo-mock-small-v1",
    description="Short engine-test scenario truncating at the horizon",
    max_steps=6,
    observation_window_steps=5,
    incident=IncidentSpec(
        entry_node_id="node-web-02", start_step=1, spread_probability=0.0
    ),
    observability=_quiet_observability(),
    restore_duration_steps=2,
    objective_node_id=None,
)

#: A short episode that terminates with objective_reached at step 1.
SCENARIO_OBJECTIVE = ScenarioConfig(
    scenario_id="scn-mock-engine-obj",
    topology_id="topo-mock-small-v1",
    description="Short engine-test scenario reaching the objective at step 1",
    max_steps=8,
    observation_window_steps=5,
    incident=IncidentSpec(
        entry_node_id="node-web-02", start_step=1, spread_probability=0.0
    ),
    observability=_quiet_observability(),
    restore_duration_steps=2,
    objective_node_id="node-web-02",
)


@pytest.fixture(scope="session")
def topology_v1() -> Topology:
    """The MOCK's fixed 8-node, 2-zone reference topology."""

    return load_topology(_TOPOLOGY_V1)


@pytest.fixture
def make_engine(
    topology_v1: Topology,
) -> Callable[..., tuple[EpisodeEngine, MockSimulator, AcceptAllCoordinator]]:
    """Return a factory for ``(engine, simulator, coordinator)`` triples.

    By default a fresh :class:`~cyberswarm.interfaces.GroundTruthAccessor` is
    attached (so the engine records the ground-truth stream). Pass
    ``recorder=`` to capture records, ``compute_graph=False`` to skip the graph
    pipeline, ``scorer=`` to attach a reward, or ``truth=False`` to omit truth.
    """

    def _make(
        *,
        recorder=None,
        scorer=None,
        compute_graph: bool = True,
        truth: bool = True,
    ) -> tuple[EpisodeEngine, MockSimulator, AcceptAllCoordinator]:
        sim = MockSimulator(topology_v1)
        accessor = sim.ground_truth_accessor() if truth else None
        coordinator = AcceptAllCoordinator(sim)
        engine = EpisodeEngine(
            sim,
            coordinator,
            truth=accessor,
            scorer=scorer,
            recorder=recorder,
            compute_graph=compute_graph,
        )
        return engine, sim, coordinator

    return _make


def noop_policy(ctx: DecisionContext) -> ActionSpec:
    """Always propose ``no_op``."""

    return ActionSpec(action_kind=ActionKind.no_op)


def investigate_policy(ctx: DecisionContext) -> ActionSpec:
    """Investigate the lowest-id own node offered this step, else ``no_op``.

    A pure function of the (deterministic) available actions, so two runs with
    the same seed produce identical proposals.
    """

    targets = sorted(
        action.target_node_id
        for action in ctx.available_actions.actions
        if action.action_kind is ActionKind.investigate_node
        and action.target_node_id is not None
    )
    if targets:
        return ActionSpec(
            action_kind=ActionKind.investigate_node, target_node_id=targets[0]
        )
    return ActionSpec(action_kind=ActionKind.no_op)


def build_proposal(ctx: DecisionContext, action: ActionSpec) -> PolicyProposal:
    """Wrap one agent's chosen action in a contract-valid proposal."""

    return PolicyProposal(
        proposal_id=ctx.next_proposal_id,
        episode_id=ctx.episode_id,
        step=ctx.step,
        agent_id=ctx.agent_id,
        policy_id=POLICY_ID,
        observation_id=ctx.observation.observation_id,
        action=action,
    )


def make_record_set(recorder: MemoryRecorder) -> EpisodeRecordSet:
    """Assemble an :class:`EpisodeRecordSet` from a :class:`MemoryRecorder`."""

    return EpisodeRecordSet(
        events=recorder.by_type("EventRecord"),
        observations=recorder.by_type("AgentObservation"),
        available_actions=recorder.by_type("AvailableActions"),
        detection=recorder.by_type("DetectionSummary"),
        node_scores=recorder.by_type("NodeScoreRecord"),
        graph_snapshots=recorder.by_type("GraphSnapshot"),
        graph_features=recorder.by_type("GraphFeatureRecord"),
        proposals=recorder.by_type("PolicyProposal"),
        validations=recorder.by_type("ValidationResult"),
        executions=recorder.by_type("ExecutedAction"),
        ground_truth=recorder.by_type("GroundTruthSnapshot"),
        episode_result=None,
    )


def run_episode(
    engine: EpisodeEngine,
    first_context,
    policy: Callable[[DecisionContext], ActionSpec],
    *,
    max_iters: int = 200,
):
    """Drive ``engine`` under ``policy`` until the episode ends.

    Returns ``(outcomes, final_context)`` where ``outcomes`` is the list of each
    step's :class:`~cyberswarm.engine.StepOutcome` and ``final_context`` is the
    terminal (``final=True``) context.
    """

    outcomes = []
    context = first_context
    iterations = 0
    while not context.final and iterations < max_iters:
        proposals = {
            agent_id: build_proposal(agent_ctx, policy(agent_ctx))
            for agent_id, agent_ctx in context.agents.items()
        }
        outcome = engine.step(proposals)
        outcomes.append(outcome)
        context = outcome.context
        iterations += 1
    return outcomes, context
