"""Shared builders for the reward tests (task 7.2).

The reward cases of RL_IMPLEMENTATION section 4.3 build ground-truth snapshots
and executed-action lists directly, without a simulator, and check the exact
reward value under the committed default weights. These fixtures return the
minimal contract-valid records the reward model reads, plus a loader for the
committed ``reward_default.toml`` and a connected-topology builder for the
stateful ``RewardModel`` tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from cyberswarm.contracts import (
    IncidentSpec,
    ObservabilitySpec,
    ScenarioConfig,
    load_topology,
)
from cyberswarm.contracts.actions import ActionSpec, ExecutedAction
from cyberswarm.contracts.common import ErrorCode, ExecutionStatus, NodeStatus
from cyberswarm.contracts.evaluator_only import GroundTruthNode, GroundTruthSnapshot
from cyberswarm.contracts.topology import Edge, EdgeType, Node, NodeType, Topology
from cyberswarm.contracts.vocabulary import ActionKind
from cyberswarm.engine import EpisodeEngine
from cyberswarm.mock import MockSimulator
from cyberswarm.rl.env import CyberSwarmParallelEnv
from cyberswarm.rl.reward import CONTAINED_STATUSES, RewardConfig, RewardModel

from .doubles import AcceptAllCoordinator

#: Repository root: ``tests/rl/conftest.py`` -> ``<repo>``.
REPO_ROOT = Path(__file__).resolve().parents[2]
REWARD_DEFAULT_TOML = REPO_ROOT / "configs" / "experiments" / "reward_default.toml"

#: A valid episode id reused by every synthetic record in these tests.
EPISODE_ID = "ep-rule-conv-v1-mock-lateral-eval-01-s1"


@pytest.fixture(scope="session")
def default_config() -> RewardConfig:
    """The committed default reward weights (``reward-default-v1``)."""

    return RewardConfig.from_toml(REWARD_DEFAULT_TOML)


@pytest.fixture
def make_node() -> Callable[..., GroundTruthNode]:
    """Return a builder for a single :class:`GroundTruthNode`.

    ``contained`` defaults to ``True`` exactly when the status is a containment
    status, so the ``K`` set stays consistent with the node status.
    """

    def _build(
        node_id: str,
        *,
        compromised: bool = False,
        status: NodeStatus = NodeStatus.up,
        defense_availability: float = 1.0,
        service_availability: float = 1.0,
        contained: bool | None = None,
    ) -> GroundTruthNode:
        return GroundTruthNode(
            node_id=node_id,
            compromised=compromised,
            service_availability=service_availability,
            defense_availability=defense_availability,
            contained=(status in CONTAINED_STATUSES) if contained is None else contained,
            status=status,
        )

    return _build


@pytest.fixture
def make_snapshot() -> Callable[..., GroundTruthSnapshot]:
    """Return a builder for a :class:`GroundTruthSnapshot` from a node list.

    Nodes are sorted ascending (the contract requires it) and ``incident_active``
    defaults to whether any node is compromised.
    """

    def _build(
        nodes: Sequence[GroundTruthNode],
        *,
        step: int = 0,
        incident_active: bool | None = None,
        episode_id: str = EPISODE_ID,
    ) -> GroundTruthSnapshot:
        ordered = sorted(nodes, key=lambda node: node.node_id)
        active = (
            any(node.compromised for node in ordered)
            if incident_active is None
            else incident_active
        )
        return GroundTruthSnapshot(
            episode_id=episode_id,
            step=step,
            incident_active=active,
            nodes=list(ordered),
            new_compromises=[],
            alert_truth=[],
        )

    return _build


@pytest.fixture
def make_execution() -> Callable[..., ExecutedAction]:
    """Return a builder for one executed (or failed) :class:`ExecutedAction`."""

    def _build(
        kind: ActionKind,
        target: str | None = None,
        *,
        counter: int = 1,
        step: int = 0,
        agent_id: str = "agent-dmz",
        status: ExecutionStatus = ExecutionStatus.executed,
    ) -> ExecutedAction:
        failed = status is ExecutionStatus.failed
        return ExecutedAction(
            execution_id=f"exe-{counter:06d}",
            episode_id=EPISODE_ID,
            step=step,
            proposal_id=f"prop-{counter:06d}",
            validation_id=f"val-{counter:06d}",
            agent_id=agent_id,
            action=ActionSpec(action_kind=kind, target_node_id=target),
            status=status,
            error_code=ErrorCode.ACTION_EXECUTION_FAILED if failed else None,
            affected_node_ids=[target] if target is not None else [],
            status_changes=[],
            outcome_event_ids=[],
        )

    return _build


@pytest.fixture
def make_topology() -> Callable[[Mapping[str, float]], Topology]:
    """Return a builder for a connected topology from ``{node_id: criticality}``.

    Nodes are chained with ``network_link`` edges in ascending id order so the
    link graph is connected and the edge ids match the canonical rank; this is
    only the carrier the reward model reads criticalities from.
    """

    def _build(criticality: Mapping[str, float]) -> Topology:
        node_ids = sorted(criticality)
        nodes = [
            Node(
                node_id=node_id,
                node_type=NodeType.web,
                zone_id="zone-dmz",
                criticality=criticality[node_id],
            )
            for node_id in node_ids
        ]
        edges = [
            Edge(
                edge_id=f"edge-{rank:03d}",
                source=node_ids[rank - 1],
                target=node_ids[rank],
                edge_type=EdgeType.network_link,
            )
            for rank in range(1, len(node_ids))
        ]
        return Topology(topology_id="topo-reward-v1", nodes=nodes, edges=edges)

    return _build


# --------------------------------------------------------------------------- #
# Environment builders (task 7.4)
# --------------------------------------------------------------------------- #
# The env tests drive a real CyberSwarmParallelEnv over the MockSimulator. The
# engine is wired through an injected ``engine_factory`` closure (the env never
# imports a simulator), using the ``AcceptAllCoordinator`` double here and the
# committed default RewardModel as the StepScorer. The real safety coordinator
# is exercised separately in task 9.

_TOPOLOGY_V1 = REPO_ROOT / "configs" / "topologies" / "topo-mock-small-v1.json"


def _env_observability() -> ObservabilitySpec:
    """Observability with alerts (so different seeds diverge) and no nulls."""

    return ObservabilitySpec(
        alert_detect_probability=0.6,
        false_alert_rate=0.1,
        alert_delay_max_steps=0,
        missing_value_rate=0.0,
    )


#: A short episode that never terminates and truncates at step ``max_steps - 1``.
SCENARIO_TRUNCATE = ScenarioConfig(
    scenario_id="scn-rl-env-trunc",
    topology_id="topo-mock-small-v1",
    description="Short env-test scenario truncating at the horizon",
    max_steps=6,
    observation_window_steps=5,
    incident=IncidentSpec(
        entry_node_id="node-web-02", start_step=1, spread_probability=0.0
    ),
    observability=_env_observability(),
    restore_duration_steps=2,
    objective_node_id=None,
)

#: A short episode that terminates with ``objective_reached`` at step 1.
SCENARIO_OBJECTIVE = ScenarioConfig(
    scenario_id="scn-rl-env-obj",
    topology_id="topo-mock-small-v1",
    description="Short env-test scenario reaching the objective at step 1",
    max_steps=8,
    observation_window_steps=5,
    incident=IncidentSpec(
        entry_node_id="node-web-02", start_step=1, spread_probability=0.0
    ),
    observability=_env_observability(),
    restore_duration_steps=2,
    objective_node_id="node-web-02",
)

#: Both env-test scenarios, registered on every env so a test picks one by id
#: through ``options={"scenario_id": ...}``. ``SCENARIO_TRUNCATE`` is first, so
#: it is the one the env's construction-time bootstrap reset uses.
DEFAULT_SCENARIOS: tuple[ScenarioConfig, ...] = (SCENARIO_TRUNCATE, SCENARIO_OBJECTIVE)


@pytest.fixture(scope="session")
def topology_v1() -> Topology:
    """The MOCK's fixed 8-node, 2-zone reference topology."""

    return load_topology(_TOPOLOGY_V1)


@dataclass
class EnvHarness:
    """An env plus a handle on the single coordinator double its engine uses."""

    env: CyberSwarmParallelEnv
    coordinator: AcceptAllCoordinator


@pytest.fixture
def make_env(
    topology_v1: Topology, default_config: RewardConfig
) -> Callable[..., EnvHarness]:
    """Return a builder for an :class:`EnvHarness` over the mock.

    ``mode`` is the observation representation (``conventional`` /
    ``graph_features``); ``scenarios`` defaults to the single truncating
    scenario. The ``engine_factory`` closure wires a fresh
    :class:`~cyberswarm.mock.MockSimulator`, the
    :class:`~tests.rl.doubles.AcceptAllCoordinator`, the default
    :class:`~cyberswarm.rl.reward.RewardModel` and an
    :class:`~cyberswarm.engine.EpisodeEngine`; the env calls it once at
    construction, so the harness can keep a handle on that coordinator.
    """

    def _make(
        mode: str = "conventional",
        scenarios: Sequence[ScenarioConfig] = DEFAULT_SCENARIOS,
        env_cls: type[CyberSwarmParallelEnv] = CyberSwarmParallelEnv,
    ) -> EnvHarness:
        created: dict[str, AcceptAllCoordinator] = {}

        def engine_factory() -> EpisodeEngine:
            simulator = MockSimulator(topology_v1)
            coordinator = AcceptAllCoordinator(simulator)
            reward = RewardModel(default_config)
            created["coordinator"] = coordinator
            return EpisodeEngine(
                simulator,
                coordinator,
                truth=simulator.ground_truth_accessor(),
                scorer=reward,
            )

        env = env_cls(engine_factory, list(scenarios), observation_mode=mode)
        return EnvHarness(env=env, coordinator=created["coordinator"])

    return _make


def noop_actions(env: CyberSwarmParallelEnv) -> dict[str, int]:
    """Return a ``no_op`` (index 0) action for every currently-active agent."""

    return {agent: 0 for agent in env.agents}
