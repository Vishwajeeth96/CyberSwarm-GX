#!/usr/bin/env python
"""Focused graph-evidence command: observation -> snapshot -> features -> tensors.

Loads a committed, agent-visible observation fixture and the demo topology
(repository-relative paths only), runs the real
``convert -> compute_features -> to_tensors`` pipeline, and prints the resulting
tensor shapes, dtypes and masks. Everything printed is agent-visible: the
converter and the feature function take no ground-truth argument, so there are
no hidden labels in the output. A leakage guard confirms that no evaluator-only
field name appears in the serialized snapshot or feature record.

It then loads the two committed step snapshots (step 5 and step 6) and shows
that they share an identical ``node_order`` even though the dynamic node and
edge state changed between the two steps (stable ordering by topology).

Run from the repository root with the repo-local interpreter::

    .venv/bin/python scripts/graph_evidence.py
"""

from __future__ import annotations

import json
from pathlib import Path

from cyberswarm.contracts import (
    DERIVED_FEATURE_NAMES_V1,
    EDGE_FEATURE_NAMES_BASE_V1,
    EVALUATOR_ONLY_FIELD_NAMES,
    NODE_FEATURE_NAMES_BASE_V1,
    AgentObservation,
    GraphSnapshot,
    Topology,
    dumps_canonical,
    read_json,
)
from cyberswarm.graph import compute_features, convert, to_tensors

_REPO_ROOT = Path(__file__).resolve().parent.parent

TOPOLOGY_REL = "configs/topologies/topo-mock-small-v1.json"
OBS_STEP5_REL = "fixtures/records/valid/AgentObservation.dmz-step5.json"
SNAPSHOT_STEP5_REL = "fixtures/records/valid/GraphSnapshot.dmz-step5.json"
SNAPSHOT_STEP6_REL = "fixtures/records/valid/GraphSnapshot.dmz-step6.json"

#: Node status one-hot columns (CONTRACTS 10.1 columns 1-5).
_STATUS_COLUMNS = {
    "status_up": "up",
    "status_rate_limited": "rate_limited",
    "status_quarantined": "quarantined",
    "status_isolated": "isolated",
    "status_restoring": "restoring",
}


def _scan_evaluator_only_keys(payload: object) -> set[str]:
    """Recursively collect any evaluator-only field names used as dict keys."""

    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in EVALUATOR_ONLY_FIELD_NAMES:
                found.add(key)
            found |= _scan_evaluator_only_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            found |= _scan_evaluator_only_keys(item)
    return found


def _status_label(node_row: list[float]) -> str:
    """Read the active status one-hot from a node feature row (or 'imputed')."""

    for column, label in _STATUS_COLUMNS.items():
        if node_row[NODE_FEATURE_NAMES_BASE_V1.index(column)] == 1.0:
            return label
    return "imputed"


def _print_mask_matrix(title: str, node_order, mask, names) -> None:
    width = len(names)
    print(f"\n{title} (1 = observed, 0 = imputed; columns = {width} features):")
    label_width = max(len(node_id) for node_id in node_order)
    for node_id, row in zip(node_order, mask):
        flags = " ".join(str(int(flag)) for flag in row)
        observed = sum(int(flag) for flag in row)
        print(f"  {node_id:<{label_width}}  {flags}   ({observed}/{width} observed)")


def main() -> int:
    topology = read_json(_REPO_ROOT / TOPOLOGY_REL, Topology)
    observation = read_json(_REPO_ROOT / OBS_STEP5_REL, AgentObservation)

    print("CyberSwarm graph-evidence (synthetic fixture, agent-visible only)")
    print("=" * 65)
    print("\nInputs (repository-relative):")
    print(
        f"  topology    : {TOPOLOGY_REL}  "
        f"({len(topology.nodes)} nodes, {len(topology.edges)} edges)"
    )
    print(
        f"  observation : {OBS_STEP5_REL}  "
        f"(agent {observation.agent_id}, step {observation.step})"
    )
    print("\nPipeline: convert -> compute_features -> to_tensors")

    snapshot = convert(observation, topology)
    features = compute_features(snapshot, topology)
    tensors = to_tensors(snapshot, features)

    print(f"\nGraphSnapshot {snapshot.snapshot_id}")
    print(f"  topology_id    : {snapshot.topology_id}")
    print(f"  feature_set_id : {snapshot.feature_set_id}")
    print(f"  imputation     : {snapshot.imputation_policy_id}")
    print(f"  step           : {snapshot.step}")
    print(f"  max_event_step : {snapshot.max_event_step}")
    print(f"  node_order     : {snapshot.node_order}")
    print(f"  edge_order     : {len(snapshot.edge_order)} edges")
    print(f"GraphFeatureRecord {features.feature_record_id}  ({features.feature_set_id})")

    print("\nTensor shapes and dtypes:")
    for name in (
        "x",
        "x_mask",
        "edge_index",
        "edge_attr",
        "edge_mask",
        "derived",
        "derived_mask",
    ):
        array = getattr(tensors, name)
        print(f"  {name:<13} {str(tuple(array.shape)):<10} {array.dtype}")

    _print_mask_matrix(
        "Node-feature mask x_mask",
        snapshot.node_order,
        snapshot.node_mask,
        NODE_FEATURE_NAMES_BASE_V1,
    )
    _print_mask_matrix(
        "Derived-feature mask derived_mask",
        features.node_order,
        features.mask,
        DERIVED_FEATURE_NAMES_V1,
    )

    width = len(EDGE_FEATURE_NAMES_BASE_V1)
    observed_edges = [
        edge_id
        for edge_id, row in zip(snapshot.edge_order, snapshot.edge_mask)
        # Edge state one-hot (columns 2-4) is set only for observed edges.
        if row[2] == 1
    ]
    print(f"\nEdge mask edge_mask (columns = {width} features):")
    print(
        f"  observed edges (state known): {observed_edges or '[none]'}  "
        f"({len(observed_edges)}/{len(snapshot.edge_order)} observed)"
    )

    serialized = dumps_canonical(snapshot) + dumps_canonical(features)
    leaked = _scan_evaluator_only_keys(json.loads(dumps_canonical(snapshot)))
    leaked |= _scan_evaluator_only_keys(json.loads(dumps_canonical(features)))
    assert not leaked, f"evaluator-only fields leaked: {sorted(leaked)}"
    assert "service_availability" not in serialized
    print(
        "\nLeakage guard: evaluator-only field names in serialized "
        "snapshot + features -> none (OK, no hidden labels)"
    )

    print("\nStable ordering across a dynamic state change")
    print("-" * 45)
    snapshot_step5 = read_json(_REPO_ROOT / SNAPSHOT_STEP5_REL, GraphSnapshot)
    snapshot_step6 = read_json(_REPO_ROOT / SNAPSHOT_STEP6_REL, GraphSnapshot)
    print(f"  step 5 snapshot node_order: {snapshot_step5.node_order}")
    print(f"  step 6 snapshot node_order: {snapshot_step6.node_order}")
    identical = snapshot_step5.node_order == snapshot_step6.node_order
    print(f"  identical node_order      : {identical}")
    print(
        f"  identical edge_order      : "
        f"{snapshot_step5.edge_order == snapshot_step6.edge_order}"
    )
    if not identical:
        raise SystemExit("ERROR: the two step snapshots disagree on node_order")

    index = snapshot_step5.node_order.index("node-web-02")
    status5 = _status_label(snapshot_step5.node_features[index])
    status6 = _status_label(snapshot_step6.node_features[index])
    evidence_col = NODE_FEATURE_NAMES_BASE_V1.index("evidence_level_norm")
    evidence5 = (
        snapshot_step5.node_features[index][evidence_col]
        if snapshot_step5.node_mask[index][evidence_col]
        else "imputed"
    )
    evidence6 = (
        snapshot_step6.node_features[index][evidence_col]
        if snapshot_step6.node_mask[index][evidence_col]
        else "imputed"
    )
    print("  example dynamic change at node-web-02 (same row index in both):")
    print(f"    status           step5 {status5!r} -> step6 {status6!r}")
    print(f"    evidence_lvl_norm step5 {evidence5!r} -> step6 {evidence6!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
