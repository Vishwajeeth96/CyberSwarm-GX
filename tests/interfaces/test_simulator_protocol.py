"""Shape tests for the public simulator/policy/engine interfaces (stage S2).

These tests need no simulator implementation. They check that the Protocols and
their boundary types import, that the two simulator Protocols are
``runtime_checkable`` while the policy/port Protocols are plain Protocols, that
each Protocol advertises the expected members, that the boundary records build
and are frozen, and that the :class:`SimulatorConformance` mixin is importable,
inert on its own and not collected as standalone tests.

Covers DELTA checks for I-03/I-04 at the interface level (DESIGN sections 3.2
and 4.2); the behavioural SC-1..SC-15 checks run against the mock in
``tests/interfaces/test_conformance_mock.py`` (task 5.3).
"""

from __future__ import annotations

import typing

import pytest

from cyberswarm.contracts import (
    ActionKind,
    ActionSpec,
    ContractModel,
    ErrorCode,
    NodeStatus,
    StatusChange,
    make_episode_id,
    proposal_id,
    validation_id,
)
from cyberswarm.interfaces import conformance as conformance_module
from cyberswarm.interfaces.conformance import SimulatorConformance
from cyberswarm.interfaces.policy import (
    DecisionContext,
    DistributionPolicy,
    EvidenceView,
    NodeEvidence,
    Policy,
    PolicyDecision,
)
from cyberswarm.interfaces.ports import (
    CoordinatorPort,
    CoordinatorResult,
    StepScore,
    StepScorer,
)
from cyberswarm.interfaces.simulator import (
    ActionApplication,
    GroundTruthAccessor,
    Simulator,
    SimulatorError,
    StepResult,
    ValidatedAction,
)

# Protocols that must be runtime-checkable, and those that must stay plain.
_RUNTIME_PROTOCOLS = (Simulator, GroundTruthAccessor)
_PLAIN_PROTOCOLS = (Policy, DistributionPolicy, CoordinatorPort, StepScorer)

_EXPECTED_MEMBERS = {
    Simulator: {
        "kind",
        "topology",
        "agent_ids",
        "authority",
        "reset",
        "step",
        "done",
        "observe",
        "available_actions",
        "apply_action",
        "end_step",
        "drain_events",
        "state_digest",
        "close",
    },
    GroundTruthAccessor: {"ground_truth", "termination_reason"},
    Policy: {"policy_id", "representation", "reset", "propose"},
    DistributionPolicy: {
        "policy_id",
        "representation",
        "reset",
        "propose",
        "decide",
        "build_proposal",
    },
    CoordinatorPort: {"start_episode", "process", "verify_recovery"},
    StepScorer: {"reset", "score"},
}


def _members(proto: type) -> set[str]:
    return set(typing.get_protocol_members(proto))


# --------------------------------------------------------------------------- #
# Protocol identity and runtime-checkability
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("proto", _RUNTIME_PROTOCOLS + _PLAIN_PROTOCOLS)
def test_declared_types_are_protocols(proto: type) -> None:
    assert getattr(proto, "_is_protocol", False) is True


@pytest.mark.parametrize("proto", _RUNTIME_PROTOCOLS)
def test_simulator_protocols_are_runtime_checkable(proto: type) -> None:
    assert getattr(proto, "_is_runtime_protocol", False) is True


@pytest.mark.parametrize("proto", _PLAIN_PROTOCOLS)
def test_policy_and_port_protocols_are_not_runtime_checkable(proto: type) -> None:
    # Declared without @runtime_checkable (DESIGN 4.2); an isinstance check
    # against them must raise rather than silently pass.
    assert getattr(proto, "_is_runtime_protocol", False) is False
    with pytest.raises(TypeError):
        isinstance(object(), proto)  # type: ignore[misc]


@pytest.mark.parametrize("proto, expected", list(_EXPECTED_MEMBERS.items()))
def test_protocol_members(proto: type, expected: set[str]) -> None:
    assert _members(proto) == expected


# --------------------------------------------------------------------------- #
# runtime_checkable behaviour with lightweight stubs (no real simulator)
# --------------------------------------------------------------------------- #
class _SimulatorStub:
    """Advertises every :class:`Simulator` member; not a working simulator."""

    kind = "mock"
    topology = None
    agent_ids: tuple = ()
    step = 0
    done = False

    def authority(self, agent_id):  # noqa: ANN001
        return ()

    def reset(self, scenario, seed, episode_id):  # noqa: ANN001
        return None

    def observe(self):
        return {}

    def available_actions(self, agent_id):  # noqa: ANN001
        return None

    def apply_action(self, action):  # noqa: ANN001
        return None

    def end_step(self):
        return None

    def drain_events(self):
        return []

    def state_digest(self):
        return "0" * 64

    def close(self):
        return None


class _GroundTruthStub:
    def ground_truth(self):
        return None

    def termination_reason(self):
        return None


def test_stub_satisfies_runtime_protocol() -> None:
    assert isinstance(_SimulatorStub(), Simulator)
    assert isinstance(_GroundTruthStub(), GroundTruthAccessor)


def test_objects_missing_members_fail_runtime_protocol() -> None:
    # A bare object advertises none of the members.
    assert not isinstance(object(), Simulator)
    assert not isinstance(object(), GroundTruthAccessor)
    # The ground-truth accessor is intentionally a different, smaller object:
    # it is not a Simulator, and a simulator stub is not a ground-truth accessor.
    assert not isinstance(_GroundTruthStub(), Simulator)
    assert not isinstance(_SimulatorStub(), GroundTruthAccessor)


# --------------------------------------------------------------------------- #
# Boundary record shapes
# --------------------------------------------------------------------------- #
def _sample_validated_action() -> ValidatedAction:
    return ValidatedAction(
        episode_id=make_episode_id("pol-rule-conv-v1", "scn-mock-lateral-eval-01", 2001),
        step=5,
        agent_id="agent-dmz",
        proposal_id=proposal_id(3),
        validation_id=validation_id(3),
        action=ActionSpec(
            action_kind=ActionKind.investigate_node, target_node_id="node-web-02"
        ),
    )


def test_boundary_records_are_frozen_contract_models() -> None:
    for model in (ValidatedAction, ActionApplication, StepResult):
        assert issubclass(model, ContractModel)
        assert model.model_config["frozen"] is True
        assert model.model_config["extra"] == "forbid"


def test_validated_action_builds_and_is_immutable() -> None:
    action = _sample_validated_action()
    assert action.step == 5
    assert action.agent_id == "agent-dmz"
    assert action.action.action_kind is ActionKind.investigate_node
    assert action.action.target_node_id == "node-web-02"
    with pytest.raises(Exception):
        action.step = 6  # frozen model


def test_validated_action_rejects_unknown_field() -> None:
    with pytest.raises(Exception):
        ValidatedAction(
            episode_id=make_episode_id("pol-x", "scn-y", 1),
            step=0,
            agent_id="agent-dmz",
            proposal_id=proposal_id(1),
            validation_id=validation_id(1),
            action=ActionSpec(action_kind=ActionKind.no_op),
            surprise=True,  # type: ignore[call-arg]
        )


def test_action_application_shapes() -> None:
    executed = ActionApplication(
        status="executed",
        error_code=None,
        affected_node_ids=("node-web-02",),
        status_changes=(
            StatusChange(
                node_id="node-web-02",
                status_from=NodeStatus.up,
                status_to=NodeStatus.isolated,
            ),
        ),
        outcome_event_ids=("evt-000014",),
    )
    assert executed.status == "executed"
    assert executed.affected_node_ids == ("node-web-02",)
    assert executed.status_changes[0].status_to is NodeStatus.isolated

    failed = ActionApplication(
        status="failed",
        error_code=ErrorCode.ACTION_EXECUTION_FAILED,
        affected_node_ids=(),
        status_changes=(),
        outcome_event_ids=(),
    )
    assert failed.status == "failed"
    assert failed.error_code is ErrorCode.ACTION_EXECUTION_FAILED


def test_step_result_shape() -> None:
    result = StepResult(step_closed=5, terminated=False, truncated=True)
    assert result.step_closed == 5
    assert result.terminated is False
    assert result.truncated is True


# --------------------------------------------------------------------------- #
# SimulatorError
# --------------------------------------------------------------------------- #
def test_simulator_error_carries_code() -> None:
    err = SimulatorError(ErrorCode.SCENARIO_UNSUPPORTED, "unknown topology")
    assert isinstance(err, Exception)
    assert err.code is ErrorCode.SCENARIO_UNSUPPORTED
    assert "unknown topology" in str(err)
    # The two codes the interface promises to use exist.
    assert ErrorCode.SCENARIO_UNSUPPORTED
    assert ErrorCode.SIMULATOR_FAILURE


# --------------------------------------------------------------------------- #
# Policy/ports dataclasses are constructible shells
# --------------------------------------------------------------------------- #
def test_policy_and_port_dataclasses_build() -> None:
    evidence = EvidenceView(
        nodes={
            "node-web-02": NodeEvidence(
                evidence_level=2,
                steps_since_investigated=1,
                alerted=True,
                insufficient=False,
            )
        }
    )
    assert evidence.nodes["node-web-02"].evidence_level == 2

    decision = PolicyDecision(
        chosen=ActionSpec(action_kind=ActionKind.no_op),
        valid_actions=(ActionSpec(action_kind=ActionKind.no_op),),
        probs=(1.0,),
    )
    assert decision.probs == (1.0,)

    result = CoordinatorResult(validations=(), executions=(), errors=())
    assert result.validations == ()

    score = StepScore(
        total=0.0, raw={"attack_impact": 0.0}, weighted={"attack_impact": 0.0}
    )
    assert score.total == 0.0


def test_decision_context_declares_agent_visible_fields() -> None:
    fields = set(DecisionContext.__dataclass_fields__)
    assert {
        "episode_id",
        "step",
        "max_steps",
        "agent_id",
        "topology",
        "observation",
        "available_actions",
        "detection",
        "evidence",
        "graph_snapshot",
        "graph_features",
        "next_proposal_id",
    } == fields


# --------------------------------------------------------------------------- #
# Conformance mixin: importable, inert, not collected on its own
# --------------------------------------------------------------------------- #
def test_conformance_mixin_is_not_named_test() -> None:
    # pytest collects classes matching ``Test*`` only; the mixin must not match.
    assert not SimulatorConformance.__name__.startswith("Test")
    assert SimulatorConformance.__module__ == conformance_module.__name__


def test_conformance_mixin_hooks_are_abstract() -> None:
    instance = SimulatorConformance()
    with pytest.raises(NotImplementedError):
        instance.make_simulator()
    with pytest.raises(NotImplementedError):
        instance.scenarios()


def test_conformance_mixin_declares_all_fifteen_checks() -> None:
    checks = sorted(n for n in dir(SimulatorConformance) if n.startswith("test_sc"))
    assert len(checks) == 15
    for index in range(1, 16):
        prefix = f"test_sc{index:02d}_"
        assert any(name.startswith(prefix) for name in checks), f"missing {prefix}"
