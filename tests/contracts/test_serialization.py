"""Canonical serialization, round trips, NaN rejection and version policy (DC-02).

Checks the one serialization approach promised by CONTRACTS section 2: compact
canonical JSON whose field order follows the model declaration, lossless record
and JSON-Lines round trips, rejection on read of a non-finite number, and the
MAJOR/MINOR compatibility rule of :func:`cyberswarm.contracts.io.check_version`.
"""

from __future__ import annotations

import json

import pytest

from cyberswarm.contracts import (
    JsonlWriter,
    Node,
    SCHEMA_VERSION,
    check_version,
    dumps_canonical,
    loads_record,
    read_json,
    read_jsonl,
    write_json,
)
from cyberswarm.contracts.common import ContractError
from cyberswarm.contracts.events import EventRecord


def _node() -> Node:
    return Node(
        node_id="node-web-02",
        node_type="web",
        zone_id="zone-dmz",
        criticality=0.2,
        label="Public web server B",
    )


def _alert_event(counter: int) -> EventRecord:
    return EventRecord.model_validate(
        {
            "schema_version": "1.0",
            "event_id": f"evt-{counter:06d}",
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
    )


# --------------------------------------------------------------------------- #
# Canonical form
# --------------------------------------------------------------------------- #
def test_dumps_canonical_is_compact() -> None:
    text = dumps_canonical(_node())
    assert ", " not in text
    assert '": ' not in text
    assert "\n" not in text
    # It is valid JSON.
    json.loads(text)


def test_field_order_follows_model_declaration() -> None:
    text = dumps_canonical(_node())
    keys = list(json.loads(text).keys())
    assert keys == ["node_id", "node_type", "zone_id", "criticality", "label"]


def test_record_round_trip() -> None:
    node = _node()
    assert loads_record("Node", dumps_canonical(node)) == node


def test_canonical_bytes_are_stable() -> None:
    node = _node()
    once = dumps_canonical(node)
    twice = dumps_canonical(loads_record("Node", once))
    assert once == twice


def test_float_round_trips_shortest_repr() -> None:
    node = Node(
        node_id="node-web-02",
        node_type="web",
        zone_id="zone-dmz",
        criticality=0.43333333333333335,
        label=None,
    )
    text = dumps_canonical(node)
    assert '"criticality":0.43333333333333335' in text
    assert loads_record("Node", text).criticality == 0.43333333333333335


# --------------------------------------------------------------------------- #
# Non-finite numbers rejected on read
# --------------------------------------------------------------------------- #
def test_nan_rejected_on_read() -> None:
    raw = (
        '{"node_id":"node-web-02","node_type":"web","zone_id":"zone-dmz",'
        '"criticality":NaN,"label":null}'
    )
    with pytest.raises(ContractError):
        loads_record("Node", raw)


def test_infinity_rejected_on_read() -> None:
    raw = (
        '{"node_id":"node-web-02","node_type":"web","zone_id":"zone-dmz",'
        '"criticality":Infinity,"label":null}'
    )
    with pytest.raises(ContractError):
        loads_record("Node", raw)


# --------------------------------------------------------------------------- #
# File and JSON-Lines round trips
# --------------------------------------------------------------------------- #
def test_write_then_read_json_file(tmp_path) -> None:
    path = tmp_path / "node.json"
    node = _node()
    write_json(path, node)
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert read_json(path, Node) == node


def test_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    events = [_alert_event(14), _alert_event(15)]
    with JsonlWriter(path) as writer:
        for event in events:
            writer.write(event)
    read_back = list(read_jsonl(path, EventRecord))
    assert read_back == events


# --------------------------------------------------------------------------- #
# Compatibility policy (CONTRACTS section 13)
# --------------------------------------------------------------------------- #
def test_check_version_accepts_current() -> None:
    assert SCHEMA_VERSION == "1.0"
    check_version("1.0")  # does not raise


@pytest.mark.parametrize("version", ["2.0", "1.1", "0.9", "abc", "1"])
def test_check_version_rejects_unreadable(version: str) -> None:
    with pytest.raises(ContractError) as excinfo:
        check_version(version)
    assert excinfo.value.code == "SCHEMA_VERSION_MISMATCH"


def test_loads_record_rejects_future_version() -> None:
    payload = _alert_event(14).model_dump(mode="json")
    payload["schema_version"] = "2.0"
    with pytest.raises(ContractError) as excinfo:
        loads_record("EventRecord", json.dumps(payload))
    assert excinfo.value.code == "SCHEMA_VERSION_MISMATCH"
