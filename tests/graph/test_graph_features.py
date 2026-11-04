"""Hand-engineered ``graph-derived-v1`` features (stage S3, DC-07).

Each of the eight derived columns is checked by hand on a three-node line
``a - b - c`` with known criticalities and one service dependency, so the exact
value of every formula is derived in the test rather than read back from the
code. The committed golden feature record pins the formulas on the real 8-node
topology, and the unseen-variant test shows that the same observation yields
different features on ``topo-mock-small-v2`` (Property 10: graph structure
feeds the features, so a different topology changes them).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cyberswarm.contracts import (
    AgentObservation,
    DERIVED_FEATURE_NAMES_V1,
    Topology,
    dumps_canonical,
)
from cyberswarm.graph import compute_features, convert

_VALID_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "records" / "valid"

# Hand-computed graph-derived-v1 values for the tiny topology / observation.
# N = 3, so N - 1 = 2. Link graph is a - b - c (diameter 2).
# criticality a=0.2, b=0.6, c=1.0 (total 1.8). Only node-a is alerted
# (alert_count_norm = 1.0); edge-001 (a-b) blocked, edge-003 (b-c) active.
_TINY_EXPECTED = {
    #           degree  crit_mean crit_max  dep_norm  al_mean  al_max  hop   blocked
    "node-a": [0.5, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 1.0],
    "node-b": [1.0, 0.6, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5],
    "node-c": [0.5, 0.6, 0.6, 0.2 / 1.8, 0.0, 0.0, 1.0, 0.0],
}


@pytest.fixture
def tiny_features(tiny_observation: AgentObservation, tiny_topology: Topology):
    return compute_features(convert(tiny_observation, tiny_topology), tiny_topology)


def test_feature_names_are_the_registry(tiny_features) -> None:
    assert tuple(tiny_features.feature_names) == DERIVED_FEATURE_NAMES_V1
    assert tiny_features.node_order == ["node-a", "node-b", "node-c"]


@pytest.mark.parametrize("node_id", ["node-a", "node-b", "node-c"])
def test_hand_computed_derived_values(tiny_features, node_id: str) -> None:
    index = tiny_features.node_order.index(node_id)
    produced = tiny_features.values[index]
    expected = _TINY_EXPECTED[node_id]
    assert produced == pytest.approx(expected)
    # Every column is computed from a real input on this fully observed case.
    assert tiny_features.mask[index] == [1] * 8


def test_degree_norm_column(tiny_features) -> None:
    # |L(n)| / (N - 1): a and c have one neighbour, b has two.
    column = [row[0] for row in tiny_features.values]
    assert column == pytest.approx([0.5, 1.0, 0.5])


def test_dependents_crit_norm_column(tiny_features) -> None:
    # Only node-a depends on node-c, so dependents_crit_norm(c) = crit(a)/total.
    index = tiny_features.node_order.index("node-c")
    assert tiny_features.values[index][3] == pytest.approx(0.2 / 1.8)
    # a and b have no dependants.
    assert tiny_features.values[tiny_features.node_order.index("node-a")][3] == 0.0
    assert tiny_features.values[tiny_features.node_order.index("node-b")][3] == 0.0


def test_hop_to_alerted_column(tiny_features) -> None:
    # node-a is the only alerted node (hop 0); b is one hop, c is two hops,
    # normalised by the diameter 2.
    hops = {nid: tiny_features.values[i][6] for i, nid in enumerate(tiny_features.node_order)}
    assert hops == pytest.approx({"node-a": 0.0, "node-b": 0.5, "node-c": 1.0})


def test_blocked_edge_frac_column(tiny_features) -> None:
    # edge-001 (a-b) is blocked, edge-003 (b-c) is active.
    frac = {nid: tiny_features.values[i][7] for i, nid in enumerate(tiny_features.node_order)}
    # a: 1/1 blocked; b: 1/2 blocked; c: 0/1 blocked.
    assert frac == pytest.approx({"node-a": 1.0, "node-b": 0.5, "node-c": 0.0})


def test_all_derived_values_are_in_unit_range(tiny_features, features_step5) -> None:
    for record in (tiny_features, features_step5):
        for row in record.values:
            for value in row:
                assert 0.0 <= value <= 1.0


# --------------------------------------------------------------------------- #
# Golden match on the real topology
# --------------------------------------------------------------------------- #
def test_features_match_the_committed_golden_record(features_step5) -> None:
    golden = (_VALID_FIXTURES / "GraphFeatureRecord.dmz-step5.json").read_text().strip()
    assert dumps_canonical(features_step5) == golden


# --------------------------------------------------------------------------- #
# Unseen topology: same observation, different features (Property 10)
# --------------------------------------------------------------------------- #
def test_unseen_topology_changes_derived_features(
    obs_step5: AgentObservation, topology_v1: Topology, topology_v2: Topology
) -> None:
    features_v1 = compute_features(convert(obs_step5, topology_v1), topology_v1)
    features_v2 = compute_features(convert(obs_step5, topology_v2), topology_v2)

    # Same node rows (ids are shared between the variants)...
    assert features_v1.node_order == features_v2.node_order
    # ...but the derived features differ for at least one node, because v2 has
    # different edges and criticalities.
    differing = [
        node_id
        for index, node_id in enumerate(features_v1.node_order)
        if features_v1.values[index] != features_v2.values[index]
    ]
    assert differing, "expected v1 and v2 to produce different features"

    # node-web-02 links to app-01 in v1 but to db-01 in v2, so its structural
    # neighbour-criticality features must change.
    index = features_v1.node_order.index("node-web-02")
    assert features_v1.values[index] != features_v2.values[index]
