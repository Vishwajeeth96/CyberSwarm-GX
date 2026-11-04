"""The single, closed action vocabulary (DELTA check DC-02).

Pins every row of the CONTRACTS section 6 table (target need, allowed
parameters, impact class, high-impact flag, cost), the derived action sets, the
fixed order, the default parameters, and the structural rules an
:class:`~cyberswarm.contracts.ActionSpec` and an
:class:`~cyberswarm.contracts.AvailableAction` enforce on their own.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cyberswarm.contracts import (
    ACTION_DEFS,
    ACTION_ORDER,
    CONTAINMENT_CLASS,
    HIGH_IMPACT,
    INFORMATION_GATHERING,
    ActionKind,
    ActionParameters,
    ActionSpec,
    AvailableAction,
    ImpactClass,
    RateLimitLevel,
    default_parameters,
)

# The authoritative table, transcribed from CONTRACTS section 6:
# action_kind -> (needs_target, allowed_parameters, impact_class, high_impact, cost)
EXPECTED_TABLE = {
    ActionKind.no_op: (False, (), ImpactClass.none, False, 0.0),
    ActionKind.monitor_node: (True, ("duration_steps",), ImpactClass.low, False, 0.5),
    ActionKind.investigate_node: (True, (), ImpactClass.low, False, 1.0),
    ActionKind.request_evidence: (True, (), ImpactClass.low, False, 3.0),
    ActionKind.rate_limit_node: (True, ("level",), ImpactClass.medium, False, 0.0),
    ActionKind.quarantine_node: (True, (), ImpactClass.medium, False, 0.0),
    ActionKind.isolate_node: (True, (), ImpactClass.high, True, 0.0),
    ActionKind.restore_node: (True, (), ImpactClass.high, True, 0.0),
}


def test_exactly_eight_action_kinds() -> None:
    assert len(ActionKind) == 8
    assert set(ACTION_DEFS) == set(ActionKind)
    assert set(EXPECTED_TABLE) == set(ActionKind)


def test_action_order_starts_with_no_op_and_is_complete() -> None:
    assert ACTION_ORDER[0] is ActionKind.no_op
    assert len(ACTION_ORDER) == 8
    assert set(ACTION_ORDER) == set(ActionKind)


@pytest.mark.parametrize("kind", list(ActionKind))
def test_action_def_matches_contracts_table(kind: ActionKind) -> None:
    needs_target, params, impact, high, cost = EXPECTED_TABLE[kind]
    spec = ACTION_DEFS[kind]
    assert spec.needs_target is needs_target
    assert spec.allowed_parameters == params
    assert spec.impact_class is impact
    assert spec.high_impact is high
    assert spec.cost_units == cost


def test_derived_action_sets() -> None:
    assert HIGH_IMPACT == frozenset({ActionKind.isolate_node, ActionKind.restore_node})
    assert CONTAINMENT_CLASS == frozenset(
        {ActionKind.quarantine_node, ActionKind.isolate_node, ActionKind.restore_node}
    )
    assert INFORMATION_GATHERING == frozenset(
        {
            ActionKind.monitor_node,
            ActionKind.investigate_node,
            ActionKind.request_evidence,
        }
    )


def test_default_parameters() -> None:
    assert default_parameters(ActionKind.monitor_node).duration_steps == 3
    assert default_parameters(ActionKind.monitor_node).level is None
    assert default_parameters(ActionKind.rate_limit_node).level is RateLimitLevel.light
    assert default_parameters(ActionKind.rate_limit_node).duration_steps is None
    assert default_parameters(ActionKind.investigate_node) == ActionParameters()


# --------------------------------------------------------------------------- #
# ActionSpec structural rules
# --------------------------------------------------------------------------- #
def test_no_op_must_not_carry_a_target() -> None:
    assert ActionSpec(action_kind=ActionKind.no_op).target_node_id is None
    with pytest.raises(ValidationError):
        ActionSpec(action_kind=ActionKind.no_op, target_node_id="node-web-01")


def test_targeted_action_requires_a_target() -> None:
    with pytest.raises(ValidationError):
        ActionSpec(action_kind=ActionKind.isolate_node)


def test_parameter_only_allowed_for_its_kind() -> None:
    # duration_steps belongs to monitor_node only.
    with pytest.raises(ValidationError):
        ActionSpec(
            action_kind=ActionKind.investigate_node,
            target_node_id="node-web-01",
            parameters=ActionParameters(duration_steps=3),
        )
    # level belongs to rate_limit_node only.
    with pytest.raises(ValidationError):
        ActionSpec(
            action_kind=ActionKind.monitor_node,
            target_node_id="node-web-01",
            parameters=ActionParameters(level=RateLimitLevel.light),
        )


def test_monitor_node_accepts_its_parameter() -> None:
    spec = ActionSpec(
        action_kind=ActionKind.monitor_node,
        target_node_id="node-web-01",
        parameters=ActionParameters(duration_steps=4),
    )
    assert spec.parameters.duration_steps == 4


# --------------------------------------------------------------------------- #
# AvailableAction must mirror the vocabulary metadata
# --------------------------------------------------------------------------- #
def _available_payload(kind: ActionKind) -> dict:
    spec = ACTION_DEFS[kind]
    return {
        "action_kind": kind.value,
        "target_node_id": "node-web-01" if spec.needs_target else None,
        "parameters": {},
        "impact_class": spec.impact_class.value,
        "high_impact": spec.high_impact,
        "cost_units": spec.cost_units,
    }


@pytest.mark.parametrize("kind", list(ActionKind))
def test_available_action_accepts_correct_metadata(kind: ActionKind) -> None:
    AvailableAction.model_validate(_available_payload(kind))


def test_available_action_rejects_wrong_cost() -> None:
    payload = _available_payload(ActionKind.investigate_node)
    payload["cost_units"] = 0.0  # real cost is 1.0
    with pytest.raises(ValidationError):
        AvailableAction.model_validate(payload)


def test_available_action_rejects_wrong_high_impact_flag() -> None:
    payload = _available_payload(ActionKind.isolate_node)
    payload["high_impact"] = False  # isolate_node is high-impact
    with pytest.raises(ValidationError):
        AvailableAction.model_validate(payload)
