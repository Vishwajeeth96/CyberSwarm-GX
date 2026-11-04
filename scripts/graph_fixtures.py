#!/usr/bin/env python
"""Derived graph-record fixtures (stage S3), produced by the real converter.

Design choice (DESIGN section 8, "derived fixtures"): the graph snapshot and
feature fixtures are *derived* by running the real :mod:`cyberswarm.graph`
pipeline on the committed primary observation inputs, not hand-written. To keep
:mod:`fixture_storyboard` importing only ``cyberswarm.contracts`` (DELTA check
DC-05), the graph derivation lives here, in a separate, clearly-labelled builder
that imports ``cyberswarm.graph``. ``build_fixtures.py`` merges these with the
storyboard's S1 fixtures, so a single ``--check`` regenerates the whole set
*through the converter* and detects any drift (the golden-file guarantee).

Fixtures produced (DESIGN section 8.2):

``fixtures/records/valid/GraphSnapshot.dmz-step5.json``
    whole-topology snapshot (8 nodes, 19 edges) converted from the committed
    ``AgentObservation.dmz-step5`` observation, with masks and ``impute-v1``
    values.
``fixtures/records/valid/GraphSnapshot.dmz-step6.json``
    the next step with changed dynamic node and edge state and an identical
    ``node_order`` / ``edge_order`` (both are the topology order).
``fixtures/records/valid/GraphFeatureRecord.dmz-step5.json``
    the ``graph-derived-v1`` features computed from the step-5 snapshot.
``fixtures/records/invalid/GraphSnapshot.mask-shape-mismatch.json``
    a node-mask row whose length no longer matches the feature-matrix width.
``fixtures/records/invalid/GraphSnapshot.imputed-nonzero.json``
    an imputed entry (mask 0) whose value is not ``0.0``.

Every input is agent-visible: :func:`cyberswarm.graph.convert` and
:func:`cyberswarm.graph.compute_features` take no ground-truth argument, so the
derived fixtures carry no evaluator-only data.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

# The storyboard (literal, contracts-only data) lives next to this file; make it
# importable whatever the working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cyberswarm.contracts import Topology, read_json  # noqa: E402
from cyberswarm.graph import compute_features, convert  # noqa: E402

import fixture_storyboard as sb  # noqa: E402

#: Repository-relative topology the demo observations resolve against.
TOPOLOGY_REL = "configs/topologies/topo-mock-small-v1.json"

_VALID_ROOT = "fixtures/records/valid"
_INVALID_ROOT = "fixtures/records/invalid"


def topology() -> Topology:
    """Load the committed 8-node demo topology (repository-relative path)."""

    return read_json(_REPO_ROOT / TOPOLOGY_REL, Topology)


def valid_graph_fixtures() -> list[sb.ValidFixture]:
    """Derive the three valid graph fixtures by running the real converter."""

    topo = topology()
    obs_step5, obs_step6 = sb.graph_observation_inputs()
    snapshot_step5 = convert(obs_step5, topo)
    snapshot_step6 = convert(obs_step6, topo)
    features_step5 = compute_features(snapshot_step5, topo)
    return [
        sb.ValidFixture(
            f"{_VALID_ROOT}/GraphSnapshot.dmz-step5.json", snapshot_step5
        ),
        sb.ValidFixture(
            f"{_VALID_ROOT}/GraphSnapshot.dmz-step6.json", snapshot_step6
        ),
        sb.ValidFixture(
            f"{_VALID_ROOT}/GraphFeatureRecord.dmz-step5.json", features_step5
        ),
    ]


def invalid_graph_fixtures() -> list[sb.InvalidFixture]:
    """Build the two invalid graph fixtures as raw payloads, one broken rule each.

    Both start from the valid step-5 snapshot payload and mutate exactly one
    thing, so each fails validation for a single, named reason.
    """

    topo = topology()
    obs_step5, _ = sb.graph_observation_inputs()
    base = convert(obs_step5, topo).model_dump(mode="json")
    return [
        sb.InvalidFixture(
            f"{_INVALID_ROOT}/GraphSnapshot.mask-shape-mismatch.json",
            "GraphSnapshot",
            _mask_shape_mismatch(base),
            "a node_mask row length differs from the feature-matrix width",
        ),
        sb.InvalidFixture(
            f"{_INVALID_ROOT}/GraphSnapshot.imputed-nonzero.json",
            "GraphSnapshot",
            _imputed_nonzero(base),
            "an imputed entry (mask 0) carries a value that is not 0.0",
        ),
    ]


def _mask_shape_mismatch(base: dict[str, Any]) -> dict[str, Any]:
    """Drop one column from the first node-mask row so it stops matching."""

    payload = copy.deepcopy(base)
    payload["node_mask"][0] = payload["node_mask"][0][:-1]
    return payload


def _imputed_nonzero(base: dict[str, Any]) -> dict[str, Any]:
    """Set the first imputed (mask 0) node feature to a non-zero value."""

    payload = copy.deepcopy(base)
    row, col = _first_imputed_cell(payload["node_mask"])
    payload["node_features"][row][col] = 0.5
    return payload


def _first_imputed_cell(mask: list[list[int]]) -> tuple[int, int]:
    """Return the (row, col) of the first mask-0 entry in a node mask matrix."""

    for row_index, row in enumerate(mask):
        for col_index, flag in enumerate(row):
            if flag == 0:
                return row_index, col_index
    raise AssertionError("snapshot has no imputed cell to corrupt")
