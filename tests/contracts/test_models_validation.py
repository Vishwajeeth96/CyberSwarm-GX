"""Per-schema model validation (DELTA checks DC-01, DC-04).

Confirms the structural guarantees every record inherits from
:class:`~cyberswarm.contracts.common.ContractModel`: an unknown field is
rejected (``extra="forbid"``), a non-finite float is rejected
(``allow_inf_nan=False``), the tightened :data:`~cyberswarm.contracts.common.ShortText`
refuses an IP look-alike, ``..``, a leading dash and an out-of-charset
character such as a colon, an unknown enum value is rejected, a malformed or
address-shaped identifier is rejected, the per-event-type field rules hold, and
an evaluator-only ground-truth field cannot ride along on an agent-visible
record.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cyberswarm.contracts import (
    AgentObservation,
    EdgeObservation,
    EventProducer,
    EventRecord,
    EventType,
    Node,
    NodeObservation,
    NodeScope,
    NodeStatus,
    Phase,
    Provenance,
    ProvenanceKind,
    Severity,
    AlertCategory,
)
from cyberswarm.contracts.common import ContractError


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _valid_node_payload() -> dict:
    return {
        "node_id": "node-web-02",
        "node_type": "web",
        "zone_id": "zone-dmz",
        "criticality": 0.2,
        "label": "Public web server B",
    }


def _alert_event(**overrides: object) -> dict:
    payload = {
        "schema_version": "1.0",
        "event_id": "evt-000014",
        "episode_id": "ep-rule-conv-v1-mock-lateral-eval-01-s2001",
        "step": 5,
        "phase": "dynamics",
        "event_type": "alert_raised",
        "producer": "simulator",
        "visible_to": ["agent-core", "agent-dmz"],
        "node_id": "node-web-02",
        "severity": "medium",
        "alert_category": "traffic_anomaly",
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# Positive controls (the valid records really do construct)
# --------------------------------------------------------------------------- #
def test_valid_node_constructs() -> None:
    node = Node.model_validate(_valid_node_payload())
    assert node.criticality == 0.2
    assert node.label == "Public web server B"


def test_valid_alert_event_constructs() -> None:
    event = EventRecord.model_validate(_alert_event())
    assert event.event_type is EventType.alert_raised
    assert event.phase is Phase.dynamics
    assert event.producer is EventProducer.simulator


# --------------------------------------------------------------------------- #
# extra="forbid"
# --------------------------------------------------------------------------- #
def test_unknown_field_rejected() -> None:
    payload = _valid_node_payload()
    payload["mystery"] = 1
    with pytest.raises(ValidationError):
        Node.model_validate(payload)


def test_ground_truth_field_rejected_on_agent_visible_node() -> None:
    # ``compromised`` is an evaluator-only field; it must not ride along on an
    # agent-visible NodeObservation (DC-04).
    with pytest.raises(ValidationError):
        NodeObservation.model_validate(
            {
                "node_id": "node-web-02",
                "scope": "own",
                "visible_status": "up",
                "alert_count_recent": 1,
                "max_alert_severity": 2,
                "compromised": True,
            }
        )


# --------------------------------------------------------------------------- #
# allow_inf_nan=False
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_rejected(bad: float) -> None:
    payload = _valid_node_payload()
    payload["criticality"] = bad
    with pytest.raises(ValidationError):
        Node.model_validate(payload)


# --------------------------------------------------------------------------- #
# ShortText refinements
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("Blocked scanner at 10.0.0.5", "dotted_quad_rejected"),
        ("Read ..etc_passwd", "double_dot_rejected"),
        ("-rf everything", "leading_dash_rejected"),
    ],
)
def test_shorttext_refinements_raise_contract_error(value: str, reason: str) -> None:
    with pytest.raises(ContractError) as excinfo:
        Provenance(kind=ProvenanceKind.fixture_synthetic, note=value)
    assert excinfo.value.reason == reason


@pytest.mark.parametrize("bad_char", [":", ";", "/", "=", "+", "|", "@", "'"])
def test_shorttext_rejects_out_of_charset(bad_char: str) -> None:
    # Characters outside [A-Za-z0-9 .,()_-] fail the pattern before the
    # refinement validator runs, so pydantic raises a ValidationError.
    with pytest.raises(ValidationError):
        Provenance(kind=ProvenanceKind.fixture_synthetic, note=f"note{bad_char}here")


def test_shorttext_accepts_plain_text() -> None:
    note = Provenance(kind=ProvenanceKind.fixture_synthetic, note="All fine (really).")
    assert note.note == "All fine (really)."


# --------------------------------------------------------------------------- #
# Unknown enum values
# --------------------------------------------------------------------------- #
def test_unknown_enum_rejected() -> None:
    payload = _valid_node_payload()
    payload["node_type"] = "router"  # not a NodeType member
    with pytest.raises(ValidationError):
        Node.model_validate(payload)


# --------------------------------------------------------------------------- #
# Identifier shape
# --------------------------------------------------------------------------- #
def test_address_shaped_node_id_rejected() -> None:
    payload = _valid_node_payload()
    payload["node_id"] = "node-10-0-0-1"  # four numeric groups: an IP look-alike
    with pytest.raises(ContractError) as excinfo:
        Node.model_validate(payload)
    assert excinfo.value.reason == "node_address_lookalike_rejected"


def test_malformed_node_id_rejected() -> None:
    payload = _valid_node_payload()
    payload["node_id"] = "Node_Web"  # uppercase and underscore: off-pattern
    with pytest.raises(ValidationError):
        Node.model_validate(payload)


# --------------------------------------------------------------------------- #
# Per-event-type field rules
# --------------------------------------------------------------------------- #
def test_alert_event_requires_node_id() -> None:
    with pytest.raises(ValidationError):
        EventRecord.model_validate(_alert_event(node_id=None))


def test_alert_event_wrong_phase_rejected() -> None:
    with pytest.raises(ValidationError):
        EventRecord.model_validate(_alert_event(phase="outcome"))


def test_coordinator_event_must_use_cev_prefix() -> None:
    # recovery_verified is a coordinator event; an evt- id is wrong.
    with pytest.raises(ValidationError):
        EventRecord.model_validate(
            {
                "schema_version": "1.0",
                "event_id": "evt-000020",
                "episode_id": "ep-rule-conv-v1-mock-lateral-eval-01-s2001",
                "step": 6,
                "phase": "outcome",
                "event_type": "recovery_verified",
                "producer": "coordinator",
                "visible_to": ["agent-dmz"],
                "node_id": "node-web-02",
                "execution_id": "exe-000013",
            }
        )


# --------------------------------------------------------------------------- #
# Agent-visible observation rules
# --------------------------------------------------------------------------- #
def test_boundary_node_withholds_private_measurements() -> None:
    with pytest.raises(ValidationError):
        NodeObservation.model_validate(
            {
                "node_id": "node-app-01",
                "scope": "boundary",
                "visible_status": "up",
                "alert_count_recent": 0,
                "max_alert_severity": 0,
                "service_health": 0.5,  # withheld on a boundary node
            }
        )
