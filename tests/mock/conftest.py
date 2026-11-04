"""Shared setup for the MOCK simulator tests (task 5.3).

Exposes the lead-owned ``topo-mock-small-v1`` topology, a loader for the
committed mock scenarios, a factory that builds a fresh
:class:`~cyberswarm.mock.MockSimulator` with its separate
:class:`~cyberswarm.interfaces.GroundTruthAccessor`, and a builder for the
:class:`~cyberswarm.interfaces.simulator.ValidatedAction` records a test drives
the simulator with. Building ``ValidatedAction`` directly is permitted in tests
(DESIGN section 3.2); production code constructs it only in
``coordinator.validator``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from cyberswarm.contracts import (
    ActionKind,
    ActionSpec,
    ScenarioConfig,
    Topology,
    load_topology,
    make_episode_id,
    proposal_id,
    validation_id,
)
from cyberswarm.interfaces.simulator import GroundTruthAccessor, ValidatedAction
from cyberswarm.mock import MockSimulator

#: Repository root: ``tests/mock/conftest.py`` -> ``<repo>``.
REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = REPO_ROOT / "configs"
_SCENARIO_DIR = _CONFIGS / "scenarios" / "mock"
_TOPOLOGY_V1 = _CONFIGS / "topologies" / "topo-mock-small-v1.json"


@pytest.fixture(scope="session")
def topology_v1() -> Topology:
    """The MOCK's fixed 8-node, 2-zone reference topology."""

    return load_topology(_TOPOLOGY_V1)


@pytest.fixture(scope="session")
def load_scenario() -> Callable[[str], ScenarioConfig]:
    """Return a loader mapping a scenario id to its committed config."""

    def _load(scenario_id: str) -> ScenarioConfig:
        path = _SCENARIO_DIR / f"{scenario_id}.json"
        return ScenarioConfig.model_validate_json(path.read_text(encoding="utf-8"))

    return _load


@pytest.fixture
def new_simulator(
    topology_v1: Topology,
) -> Callable[[], tuple[MockSimulator, GroundTruthAccessor]]:
    """Return a factory for a fresh ``(MockSimulator, GroundTruthAccessor)`` pair."""

    def _make() -> tuple[MockSimulator, GroundTruthAccessor]:
        sim = MockSimulator(topology_v1)
        return sim, sim.ground_truth_accessor()

    return _make


@pytest.fixture
def validated_action() -> Callable[..., ValidatedAction]:
    """Return a builder for a :class:`ValidatedAction` (test-only construction)."""

    def _build(
        episode_id: str,
        step: int,
        agent_id: str,
        kind: ActionKind,
        target: str | None = None,
        *,
        counter: int = 1,
    ) -> ValidatedAction:
        return ValidatedAction(
            episode_id=episode_id,
            step=step,
            agent_id=agent_id,
            proposal_id=proposal_id(counter),
            validation_id=validation_id(counter),
            action=ActionSpec(action_kind=kind, target_node_id=target),
        )

    return _build


@pytest.fixture
def episode_id_for() -> Callable[[str, int], str]:
    """Return a helper building a deterministic episode id for a scenario/seed."""

    def _make(scenario_id: str, seed: int, policy_id: str = "pol-mock-test") -> str:
        return make_episode_id(policy_id, scenario_id, seed)

    return _make
