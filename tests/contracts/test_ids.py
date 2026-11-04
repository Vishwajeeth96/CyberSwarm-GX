"""Identifier patterns and deterministic builders (DELTA check DC-02).

Exercises the ``Annotated`` identifier types through a pydantic
:class:`~pydantic.TypeAdapter`, the deterministic id builders, and the two
security refinements: a ``node-`` id that looks like an IP address is refused,
and so are off-pattern (shell-ish) identifiers.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from cyberswarm.contracts.common import ContractError
from cyberswarm.contracts.ids import (
    AgentId,
    EdgeId,
    EventId,
    NodeId,
    ScenarioId,
    TopologyId,
    ZoneId,
    agent_id_for_zone,
    available_actions_id,
    detection_id,
    error_id,
    event_id,
    execution_id,
    feature_record_id,
    graph_model_output_id,
    make_episode_id,
    node_score_id,
    observation_id,
    proposal_id,
    snapshot_id,
    validation_id,
)


@pytest.mark.parametrize(
    ("adapter", "value"),
    [
        (NodeId, "node-web-02"),
        (NodeId, "node-db-01"),
        (EdgeId, "edge-014"),
        (EdgeId, "edge-0001"),
        (ZoneId, "zone-dmz"),
        (AgentId, "agent-core"),
        (TopologyId, "topo-mock-small-v1"),
        (ScenarioId, "scn-mock-lateral-eval-01"),
        (EventId, "evt-000014"),
        (EventId, "cev-000003"),
    ],
)
def test_valid_ids_accepted(adapter: object, value: str) -> None:
    assert TypeAdapter(adapter).validate_python(value) == value


@pytest.mark.parametrize(
    ("adapter", "value"),
    [
        (NodeId, "Node-Web"),          # uppercase
        (NodeId, "node_web_02"),       # underscores
        (NodeId, "web-02"),            # missing prefix
        (EdgeId, "edge-14"),           # too few digits
        (ZoneId, "zone-"),             # empty slug
        (AgentId, "agent-DMZ"),        # uppercase
        (EventId, "evt-14"),           # wrong width
        (EventId, "xyz-000001"),       # wrong prefix
    ],
)
def test_off_pattern_ids_rejected(adapter: object, value: str) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(adapter).validate_python(value)


def test_node_id_ip_lookalike_refused() -> None:
    with pytest.raises(ContractError) as excinfo:
        TypeAdapter(NodeId).validate_python("node-10-0-0-1")
    assert excinfo.value.reason == "node_address_lookalike_rejected"


# --------------------------------------------------------------------------- #
# Deterministic builders
# --------------------------------------------------------------------------- #
def test_per_step_id_builders() -> None:
    assert observation_id(5, "agent-dmz") == "obs-0005-agent-dmz"
    assert available_actions_id(5, "agent-dmz") == "avl-0005-agent-dmz"
    assert detection_id(5, "agent-dmz") == "det-0005-agent-dmz"
    assert snapshot_id(5, "agent-dmz") == "gsn-0005-agent-dmz"
    assert feature_record_id(5, "agent-dmz") == "gft-0005-agent-dmz"
    assert graph_model_output_id(5, "agent-dmz") == "gmo-0005-agent-dmz"
    assert node_score_id(5, "agent-dmz", "detection-rule-v1") == (
        "nsc-0005-agent-dmz-detection-rule-v1"
    )


def test_counter_id_builders() -> None:
    assert event_id(14) == "evt-000014"
    assert event_id(3, "cev") == "cev-000003"
    assert proposal_id(12) == "prop-000012"
    assert validation_id(12) == "val-000012"
    assert execution_id(13) == "exe-000013"
    assert error_id(1) == "err-000001"


def test_agent_id_for_zone() -> None:
    assert agent_id_for_zone("zone-dmz") == "agent-dmz"
    assert agent_id_for_zone("zone-core") == "agent-core"


def test_make_episode_id_is_deterministic() -> None:
    episode = make_episode_id(
        "pol-rule-conv-v1", "scn-mock-lateral-eval-01", 2001
    )
    assert episode == "ep-rule-conv-v1-mock-lateral-eval-01-s2001"


def test_make_episode_id_rejects_negative_seed() -> None:
    with pytest.raises(ContractError):
        make_episode_id("pol-rule-conv-v1", "scn-mock-lateral-eval-01", -1)


def test_event_id_rejects_unknown_prefix() -> None:
    with pytest.raises(ContractError):
        event_id(1, "bad")


def test_built_per_step_ids_are_accepted_by_their_types() -> None:
    # The builders and the Annotated id patterns agree.
    assert TypeAdapter(EventId).validate_python(event_id(14)) == "evt-000014"
    assert TypeAdapter(EventId).validate_python(event_id(3, "cev")) == "cev-000003"
