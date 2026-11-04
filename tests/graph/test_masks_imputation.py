"""Masks and the ``impute-v1`` policy (stage S3, DC-07f).

The imputation policy is deterministic and non-leaking: every imputed entry is
exactly ``0.0`` with mask ``0``; the public ``criticality`` column stays
observed even for a node the agent did not see; a null private measurement
imputes only its own column; and an unobserved edge imputes only its state
columns. The derived-feature defaults (mask ``0``) are checked by hand on a
two-node case so "no observed neighbour" (default) is distinguished from
"observed zero" (a real value with mask 1).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from cyberswarm.contracts import (
    AgentObservation,
    GraphSnapshot,
    NodeObservation,
    Topology,
)
from cyberswarm.graph import compute_features, convert

# Snapshot node columns (CONTRACTS 10.1).
_CRITICALITY = 0
_ALERT_COUNT_NORM = 6
_EVIDENCE_LEVEL_NORM = 8
_SERVICE_HEALTH = 9


# --------------------------------------------------------------------------- #
# impute-v1 invariants on the full step-5 snapshot
# --------------------------------------------------------------------------- #
def test_every_imputed_entry_is_zero(snapshot_step5: GraphSnapshot) -> None:
    for row, mask in zip(snapshot_step5.node_features, snapshot_step5.node_mask):
        for value, flag in zip(row, mask):
            if flag == 0:
                assert value == 0.0
    for row, mask in zip(snapshot_step5.edge_features, snapshot_step5.edge_mask):
        for value, flag in zip(row, mask):
            if flag == 0:
                assert value == 0.0


def test_criticality_is_always_observed(
    snapshot_step5: GraphSnapshot, topology_v1: Topology
) -> None:
    criticality = {node.node_id: float(node.criticality) for node in topology_v1.nodes}
    for index, node_id in enumerate(snapshot_step5.node_order):
        assert snapshot_step5.node_mask[index][_CRITICALITY] == 1
        assert snapshot_step5.node_features[index][_CRITICALITY] == pytest.approx(
            criticality[node_id]
        )


def test_unobserved_node_imputes_every_dynamic_column(
    snapshot_step5: GraphSnapshot,
) -> None:
    # node-auth-01 is not in the step-5 observation.
    index = snapshot_step5.node_order.index("node-auth-01")
    mask = snapshot_step5.node_mask[index]
    assert mask[_CRITICALITY] == 1  # public topology column stays observed
    assert mask[1:] == [0] * 10  # every dynamic column imputed
    assert snapshot_step5.node_features[index][1:] == [0.0] * 10


def test_observed_zero_differs_from_imputed_zero(
    snapshot_step5: GraphSnapshot,
) -> None:
    # Boundary node-app-01 is observed with zero alerts: value 0.0, mask 1.
    observed = snapshot_step5.node_order.index("node-app-01")
    assert snapshot_step5.node_features[observed][_ALERT_COUNT_NORM] == 0.0
    assert snapshot_step5.node_mask[observed][_ALERT_COUNT_NORM] == 1
    # Unobserved node-auth-01 also has 0.0 there, but mask 0 (imputed).
    imputed = snapshot_step5.node_order.index("node-auth-01")
    assert snapshot_step5.node_features[imputed][_ALERT_COUNT_NORM] == 0.0
    assert snapshot_step5.node_mask[imputed][_ALERT_COUNT_NORM] == 0


def test_null_measurement_imputes_only_that_column(
    snapshot_step5: GraphSnapshot,
) -> None:
    # node-web-01: evidence_level is null (imputed) but service_health is present.
    index = snapshot_step5.node_order.index("node-web-01")
    assert snapshot_step5.node_mask[index][_EVIDENCE_LEVEL_NORM] == 0
    assert snapshot_step5.node_features[index][_EVIDENCE_LEVEL_NORM] == 0.0
    assert snapshot_step5.node_mask[index][_SERVICE_HEALTH] == 1
    assert snapshot_step5.node_features[index][_SERVICE_HEALTH] == pytest.approx(0.99)


def test_unobserved_edge_imputes_only_state_columns(
    snapshot_step5: GraphSnapshot,
) -> None:
    # edge-001 is not in the step-5 observation: type one-hot stays, state null.
    index = snapshot_step5.edge_order.index("edge-001")
    assert snapshot_step5.edge_mask[index] == [1, 1, 0, 0, 0, 0]
    assert snapshot_step5.edge_features[index][2:] == [0.0, 0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# Derived-feature defaults by hand (two-node case)
# --------------------------------------------------------------------------- #
def test_derived_defaults_on_unobserved_neighbour(
    make_node: Callable[..., NodeObservation],
    make_obs: Callable[..., AgentObservation],
    pair_topology: Topology,
) -> None:
    # Observe only node-a (alerts 0); node-b and all edges stay unobserved.
    obs = make_obs([make_node("node-a", alert_count=0)], step=2)
    features = compute_features(convert(obs, pair_topology), pair_topology)
    by_node = dict(zip(features.node_order, zip(features.values, features.mask)))

    a_values, a_mask = by_node["node-a"]
    # Topology-only columns 0-3 are always present.
    assert a_mask[:4] == [1, 1, 1, 1]
    assert a_values[0] == pytest.approx(1.0)  # degree_norm = 1 / (N - 1)
    assert a_values[1] == pytest.approx(0.9)  # neighbour crit mean (node-b)
    assert a_values[2] == pytest.approx(0.9)  # neighbour crit max
    assert a_values[3] == pytest.approx(0.0)  # no dependants
    # node-b is unobserved -> neighbour-alert columns fall back to default.
    assert a_mask[4] == 0 and a_values[4] == 0.0
    assert a_mask[5] == 0 and a_values[5] == 0.0
    # No observed node is alerted -> hop default 1.0 with mask 0.
    assert a_mask[6] == 0 and a_values[6] == pytest.approx(1.0)
    # No incident edge observed -> blocked fraction default 0.0 with mask 0.
    assert a_mask[7] == 0 and a_values[7] == 0.0

    # node-b (unobserved) still gets neighbour-alert from its observed neighbour.
    b_values, b_mask = by_node["node-b"]
    assert b_mask[4] == 1 and b_values[4] == pytest.approx(0.0)  # neighbour node-a, 0 alerts
    assert b_mask[5] == 1 and b_values[5] == pytest.approx(0.0)
    assert b_mask[6] == 0 and b_values[6] == pytest.approx(1.0)
    assert b_mask[7] == 0 and b_values[7] == 0.0


def test_derived_defaults_on_committed_feature_record(
    features_step5,
) -> None:
    # node-ws-01's only link neighbour (node-file-01) is unobserved at step 5,
    # so neighbour-alert (4, 5) and blocked-edge (7) use their mask-0 defaults.
    index = features_step5.node_order.index("node-ws-01")
    assert features_step5.mask[index] == [1, 1, 1, 1, 0, 0, 1, 0]
    assert features_step5.values[index][4] == 0.0
    assert features_step5.values[index][5] == 0.0
    assert features_step5.values[index][7] == 0.0
