"""Dynamic state updates across steps (stage S3, DC-07b).

Two things must hold as the episode advances: the node and edge order stay
identical (ids are stable, so a model sees the same rows at every step), while
the feature values track the changing agent-visible state. This is checked both
on the two committed snapshots (step 5 and step 6 of the DMZ agent) and through
the converter on two hand-built observations that differ only in dynamic state.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from cyberswarm.contracts import (
    AgentObservation,
    GraphSnapshot,
    NodeObservation,
    NodeStatus,
    Topology,
)
from cyberswarm.graph import convert

_VALID_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "records" / "valid"


# --------------------------------------------------------------------------- #
# The two committed snapshots share order but differ in values
# --------------------------------------------------------------------------- #
def _load_snapshot(name: str) -> GraphSnapshot:
    return GraphSnapshot.model_validate_json((_VALID_FIXTURES / name).read_text())


def test_committed_steps_share_node_and_edge_order() -> None:
    step5 = _load_snapshot("GraphSnapshot.dmz-step5.json")
    step6 = _load_snapshot("GraphSnapshot.dmz-step6.json")
    assert step5.step == 5
    assert step6.step == 6
    assert step5.node_order == step6.node_order
    assert step5.edge_order == step6.edge_order
    assert step5.edge_index == step6.edge_index
    # Per-step identity still advances with the step.
    assert step5.snapshot_id != step6.snapshot_id


def test_committed_steps_differ_in_feature_values() -> None:
    step5 = _load_snapshot("GraphSnapshot.dmz-step5.json")
    step6 = _load_snapshot("GraphSnapshot.dmz-step6.json")
    assert step5.node_features != step6.node_features
    # node-web-02 moves from `up` at step 5 to `isolated` at step 6.
    index = step5.node_order.index("node-web-02")
    status_up, status_isolated = 1, 4
    assert step5.node_features[index][status_up] == 1.0
    assert step6.node_features[index][status_isolated] == 1.0
    # An edge state also changes (edge-013 becomes blocked at step 6).
    edge_index = step5.edge_order.index("edge-013")
    blocked_col = 4
    assert step5.edge_features[edge_index][blocked_col] == 0.0
    assert step6.edge_features[edge_index][blocked_col] == 1.0


# --------------------------------------------------------------------------- #
# The converter keeps order stable while values follow the observation
# --------------------------------------------------------------------------- #
def _dmz_nodes(
    make_node: Callable[..., NodeObservation], *, web02_status: NodeStatus, web02_alerts: int
) -> list[NodeObservation]:
    return [
        make_node("node-app-01", alert_count=0),
        make_node("node-gw-01", alert_count=1, severity=1, evidence_level=0, service_health=0.97, steps_since=3),
        make_node("node-web-01", service_health=0.99),
        make_node("node-web-02", status=web02_status, alert_count=web02_alerts, severity=2, service_health=0.88),
    ]


def test_converter_keeps_order_while_state_changes(
    make_node: Callable[..., NodeObservation],
    make_obs: Callable[..., AgentObservation],
    topology_v1: Topology,
) -> None:
    obs_t = make_obs(_dmz_nodes(make_node, web02_status=NodeStatus.up, web02_alerts=1), step=5)
    obs_t1 = make_obs(
        _dmz_nodes(make_node, web02_status=NodeStatus.isolated, web02_alerts=0), step=6
    )

    snap_t = convert(obs_t, topology_v1)
    snap_t1 = convert(obs_t1, topology_v1)

    # Order (and therefore every row's identity) is identical across the step.
    assert snap_t.node_order == snap_t1.node_order == list(topology_v1.node_ids)
    assert snap_t.edge_order == snap_t1.edge_order == list(topology_v1.edge_ids)
    assert snap_t.edge_index == snap_t1.edge_index

    # The dynamic state changed, so the matrices differ for node-web-02.
    index = snap_t.node_order.index("node-web-02")
    assert snap_t.node_features[index] != snap_t1.node_features[index]
    status_up, status_isolated = 1, 4
    assert snap_t.node_features[index][status_up] == 1.0
    assert snap_t1.node_features[index][status_isolated] == 1.0

    # A node whose observed state did not change keeps an identical row.
    unchanged = snap_t.node_order.index("node-web-01")
    assert snap_t.node_features[unchanged] == snap_t1.node_features[unchanged]
