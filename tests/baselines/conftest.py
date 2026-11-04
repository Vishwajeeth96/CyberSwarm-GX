"""Shared setup for the baseline tests (stage S4, task 8.3; design Property 16).

Two kinds of :class:`~cyberswarm.interfaces.DecisionContext` feed these tests:

* **Synthetic** contexts built by :func:`build_context` from a hand-written node
  list, so a test can put a single node in a precise state (alerted with null
  evidence, quiet with an alerted boundary neighbour, ...) and read the exact
  proposal back. The detection summary, evidence view and available-action menu
  are assembled with the real production code (:mod:`cyberswarm.roles`, the
  mock's :func:`build_available_actions`), so the context is contract-faithful.
* **Realistic** contexts collected by :func:`collect_mock_contexts`, which
  drives the conventional baseline through the real
  :class:`~cyberswarm.engine.EpisodeEngine` over the
  :class:`~cyberswarm.mock.MockSimulator` in ``graph_features`` mode and returns
  every per-agent context of the episode (varied node states: up, rate-limited,
  quarantined, isolated, restoring, with and without evidence). Broad invariants
  (contract-valid proposals, action in the menu, determinism) are checked over
  this whole set.

The rule-2f end-to-end scenario (:data:`RULE2F_SCENARIO`) is a single compromised
DMZ node with no spread and reliable alerts, driven by :func:`run_episode`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from cyberswarm.contracts import (
    ActionKind,
    ActionSpec,
    AgentObservation,
    AvailableActions,
    GraphFeatureRecord,
    IncidentSpec,
    NodeObservation,
    NodeScope,
    NodeStatus,
    ObservabilitySpec,
    PolicyProposal,
    ScenarioConfig,
    Topology,
    load_topology,
    observation_id,
    proposal_id,
)
from cyberswarm.engine import EpisodeEngine
from cyberswarm.interfaces import DecisionContext
from cyberswarm.mock import MockSimulator
from cyberswarm.mock.effects import NodeState
from cyberswarm.mock.observation import TopologyIndex, build_available_actions
from cyberswarm.roles import DetectionRole, evidence_view
from cyberswarm.graph import compute_features, convert
from cyberswarm.baselines.rule import RuleBasedPolicy

from .doubles import AcceptAllCoordinator

# --------------------------------------------------------------------------- #
# Constants: the reference topology and a reusable episode id
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
TOPO_V1_PATH = REPO_ROOT / "configs" / "topologies" / "topo-mock-small-v1.json"

#: A contract-valid episode id slug reused by the synthetic contexts.
EPISODE_ID = "ep-rule-conv-v1-mock-lateral-eval-01-s2001"
AGENT_DMZ = "agent-dmz"
ZONE_DMZ = "zone-dmz"

#: Loaded once at import; the contexts and index below are read-only.
_TOPOLOGY = load_topology(TOPO_V1_PATH)
_INDEX = TopologyIndex(_TOPOLOGY)


# --------------------------------------------------------------------------- #
# Synthetic context builders (agent-visible inputs only)
# --------------------------------------------------------------------------- #
def make_node(
    node_id: str,
    *,
    scope: NodeScope = NodeScope.own,
    status: NodeStatus = NodeStatus.up,
    alert_count: int = 0,
    severity: int = 0,
    evidence_level: int | None = None,
    service_health: float | None = None,
    steps_since: int | None = None,
) -> NodeObservation:
    """Build one :class:`NodeObservation` (agent-visible fields only)."""

    return NodeObservation(
        node_id=node_id,
        scope=scope,
        visible_status=status,
        alert_count_recent=alert_count,
        max_alert_severity=severity,
        evidence_level=evidence_level,
        service_health=service_health,
        steps_since_investigated=steps_since,
    )


def _available_for(nodes: Sequence[NodeObservation], step: int) -> AvailableActions:
    """Build the real state-dependent menu for ``agent-dmz`` from node statuses.

    Own-node statuses come from the observation (so action preconditions match
    what the node shows); every other node defaults to ``up``.
    """

    states = {node_id: NodeState(node_id=node_id) for node_id in _INDEX.node_order}
    for node in nodes:
        if node.scope is NodeScope.own and node.node_id in states:
            states[node.node_id].status = node.visible_status
    return build_available_actions(
        agent_id=AGENT_DMZ,
        step=step,
        episode_id=EPISODE_ID,
        index=_INDEX,
        node_states=states,
    )


def build_context(
    nodes: Sequence[NodeObservation],
    *,
    with_graph: bool = True,
    step: int = 5,
    next_pid: int = 1,
) -> DecisionContext:
    """Assemble a complete :class:`DecisionContext` for ``agent-dmz``.

    The detection summary and evidence view are produced by the real roles; the
    available-action menu by the real mock builder; the graph snapshot/features
    by the real graph pipeline (``with_graph=False`` leaves both ``None``).
    """

    obs = AgentObservation(
        observation_id=observation_id(step, AGENT_DMZ),
        episode_id=EPISODE_ID,
        step=step,
        agent_id=AGENT_DMZ,
        zone_id=ZONE_DMZ,
        max_steps=40,
        observation_window_steps=5,
        nodes=list(nodes),
        edges=[],
    )
    snapshot = features = None
    if with_graph:
        snapshot = convert(obs, _TOPOLOGY)
        features = compute_features(snapshot, _TOPOLOGY)
    return DecisionContext(
        episode_id=EPISODE_ID,
        step=step,
        max_steps=40,
        agent_id=AGENT_DMZ,
        topology=_TOPOLOGY,
        observation=obs,
        available_actions=_available_for(nodes, step),
        detection=DetectionRole().summarize(obs),
        evidence=evidence_view(obs),
        graph_snapshot=snapshot,
        graph_features=features,
        next_proposal_id=proposal_id(next_pid),
    )


def with_graph_features(
    ctx: DecisionContext, features: GraphFeatureRecord | None
) -> DecisionContext:
    """Return a copy of ``ctx`` with ``graph_features`` replaced."""

    return replace(ctx, graph_features=features)


def perturb_features(features: GraphFeatureRecord, *, delta: float = 0.37) -> GraphFeatureRecord:
    """Return a copy of ``features`` with every value shifted inside ``[0, 1]``.

    The shift wraps within the unit range so the result is always a valid
    ``graph-derived-v1`` record but differs from the input in every column.
    """

    shifted = [[(value + delta) % 1.0 for value in row] for row in features.values]
    return features.model_copy(update={"values": shifted})


# --------------------------------------------------------------------------- #
# Proposal assertions
# --------------------------------------------------------------------------- #
def revalidate(proposal: PolicyProposal) -> PolicyProposal:
    """Round-trip a proposal through JSON, re-running every contract validator."""

    return PolicyProposal.model_validate_json(proposal.model_dump_json())


def action_in_available(action: ActionSpec, available: AvailableActions) -> bool:
    """Whether ``action`` (kind, target and parameters) is in the menu."""

    for offered in available.actions:
        if (
            offered.action_kind == action.action_kind
            and offered.target_node_id == action.target_node_id
        ):
            return offered.parameters == action.parameters
    return False


# --------------------------------------------------------------------------- #
# Realistic contexts collected from a real episode (graph_features mode)
# --------------------------------------------------------------------------- #
def collect_mock_contexts(
    *, seed: int = 13, scenario: ScenarioConfig | None = None
) -> list[DecisionContext]:
    """Drive the conventional baseline through the engine; return every context.

    The engine runs in ``graph_features`` mode (``compute_graph=True``), so each
    returned context carries a real graph snapshot and features as well as the
    conventional inputs. Includes the final (post-terminal) context.
    """

    scenario = scenario if scenario is not None else RULE2F_SCENARIO
    simulator = MockSimulator(_TOPOLOGY)
    coordinator = AcceptAllCoordinator(simulator)
    engine = EpisodeEngine(
        simulator, coordinator, truth=simulator.ground_truth_accessor()
    )
    policy = RuleBasedPolicy()
    context = engine.reset(scenario, seed, policy.policy_id)
    policy.reset(engine.episode_id, seed)

    contexts: list[DecisionContext] = []
    guard = 0
    while guard < 500:
        guard += 1
        contexts.extend(context.agents[agent] for agent in sorted(context.agents))
        if context.final:
            break
        proposals = {
            agent_id: policy.propose(agent_ctx)
            for agent_id, agent_ctx in context.agents.items()
        }
        context = engine.step(proposals).context
    engine.close()
    return contexts


@pytest.fixture
def mock_contexts() -> list[DecisionContext]:
    """Every per-agent context of a real rule-2f episode (graph mode)."""

    return collect_mock_contexts()


# --------------------------------------------------------------------------- #
# Rule-2f end-to-end scenario and driver
# --------------------------------------------------------------------------- #
#: The DMZ node the single-incident rule-2f scenario compromises.
ENTRY_NODE = "node-web-02"
#: A seed whose evidence draws put the entry node into quarantine on moderate
#: (level-2) evidence, so the contained-node re-investigation rule (2f) decides
#: whether the episode resolves (verified across the clean-pattern seed set).
RULE2F_SEED = 13

#: Single compromised DMZ node, no spread, reliable alerts, no missing values.
#: ``objective_node_id`` is ``None`` so the only ways to end are
#: ``incident_resolved`` (all nodes up, none compromised) or ``max_steps``.
RULE2F_SCENARIO = ScenarioConfig(
    scenario_id="scn-rule2f-e2e",
    topology_id="topo-mock-small-v1",
    description="Single compromised DMZ node with no spread, for the rule 2f path",
    max_steps=40,
    observation_window_steps=5,
    incident=IncidentSpec(
        entry_node_id=ENTRY_NODE, start_step=1, spread_probability=0.0
    ),
    observability=ObservabilitySpec(
        alert_detect_probability=1.0,
        false_alert_rate=0.0,
        alert_delay_max_steps=0,
        missing_value_rate=0.0,
    ),
    restore_duration_steps=2,
    objective_node_id=None,
)


@dataclass(frozen=True)
class TrajectoryRow:
    """One step of the entry node's story, as the policy saw it and the truth."""

    step: int
    visible_status: NodeStatus
    evidence_level: int | None
    steps_since_investigated: int | None
    action: ActionKind
    truly_compromised: bool


@dataclass(frozen=True)
class EpisodeTrace:
    """The outcome of a driven rule-2f episode."""

    termination_reason: str | None
    steps: int
    trajectory: tuple[TrajectoryRow, ...]


def _entry_node_obs(ctx: DecisionContext) -> NodeObservation | None:
    for node in ctx.observation.nodes:
        if node.node_id == ENTRY_NODE:
            return node
    return None


def run_episode(
    policy,
    *,
    seed: int = RULE2F_SEED,
    scenario: ScenarioConfig | None = None,
    max_iters: int = 200,
) -> EpisodeTrace:
    """Drive ``policy`` (both agents) through the engine to the episode end.

    Records, per step, the entry node's agent-visible status/evidence and the
    action the policy chose for ``agent-dmz``, plus the end-of-step ground-truth
    ``compromised`` flag read through the evaluator-only accessor (tests are
    permitted to read ground truth; DESIGN section 3.2).
    """

    scenario = scenario if scenario is not None else RULE2F_SCENARIO
    simulator = MockSimulator(_TOPOLOGY)
    coordinator = AcceptAllCoordinator(simulator)
    truth = simulator.ground_truth_accessor()
    engine = EpisodeEngine(simulator, coordinator, truth=truth)
    context = engine.reset(scenario, seed, policy.policy_id)
    policy.reset(engine.episode_id, seed)

    rows: list[TrajectoryRow] = []
    steps = 0
    while not context.final and steps < max_iters:
        dmz = context.agents[AGENT_DMZ]
        proposals = {
            agent_id: policy.propose(agent_ctx)
            for agent_id, agent_ctx in context.agents.items()
        }
        entry = _entry_node_obs(dmz)
        action_kind = proposals[AGENT_DMZ].action.action_kind
        outcome = engine.step(proposals)
        snapshot = truth.ground_truth()
        compromised = any(
            node.node_id == ENTRY_NODE and node.compromised for node in snapshot.nodes
        )
        if entry is not None:
            rows.append(
                TrajectoryRow(
                    step=dmz.step,
                    visible_status=entry.visible_status,
                    evidence_level=entry.evidence_level,
                    steps_since_investigated=entry.steps_since_investigated,
                    action=action_kind,
                    truly_compromised=compromised,
                )
            )
        steps += 1
        if outcome.done:
            break
        context = outcome.context

    reason = engine.termination_reason
    engine.close()
    return EpisodeTrace(termination_reason=reason, steps=steps, trajectory=tuple(rows))


@pytest.fixture
def rule2f_runner() -> Callable[..., EpisodeTrace]:
    """Expose :func:`run_episode` as a fixture for the end-to-end tests."""

    return run_episode


@pytest.fixture(scope="session")
def topology_v1() -> Topology:
    """The reference 8-node, 2-zone topology."""

    return _TOPOLOGY
