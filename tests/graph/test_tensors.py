"""Model-compatible tensors and batches (stage S3, DC-07e, DC-07h).

Checks the documented shapes and dtypes of :func:`to_tensors`, that the arrays
carry the snapshot values, that ``derived`` is an explicit all-zero / all-False
block when no feature record is given, that :func:`collate` stacks items and
requires identical node and edge order (raising otherwise), and that the lazy
:meth:`to_torch` conversion preserves dtypes. The whole pipeline runs on numpy
and networkx alone; torch is only touched inside ``to_torch``.
"""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pytest

from cyberswarm.contracts import (
    ErrorCode,
    GraphFeatureRecord,
    GraphSnapshot,
    Topology,
)
from cyberswarm.graph import (
    GraphConversionError,
    collate,
    compute_features,
    convert,
    to_tensors,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None


# --------------------------------------------------------------------------- #
# Shapes and dtypes (DC-07e)
# --------------------------------------------------------------------------- #
def test_tensor_shapes_and_dtypes(
    snapshot_step5: GraphSnapshot, features_step5: GraphFeatureRecord
) -> None:
    n = len(snapshot_step5.node_order)
    e = len(snapshot_step5.edge_order)
    tensors = to_tensors(snapshot_step5, features_step5)

    assert tensors.x.shape == (n, 11)
    assert tensors.x.dtype == np.float32
    assert tensors.x_mask.shape == (n, 11)
    assert tensors.x_mask.dtype == np.bool_
    assert tensors.edge_index.shape == (2, e)
    assert tensors.edge_index.dtype == np.int64
    assert tensors.edge_attr.shape == (e, 6)
    assert tensors.edge_attr.dtype == np.float32
    assert tensors.edge_mask.shape == (e, 6)
    assert tensors.edge_mask.dtype == np.bool_
    assert tensors.derived.shape == (n, 8)
    assert tensors.derived.dtype == np.float32
    assert tensors.derived_mask.shape == (n, 8)
    assert tensors.derived_mask.dtype == np.bool_
    assert tensors.node_order == tuple(snapshot_step5.node_order)
    assert tensors.edge_order == tuple(snapshot_step5.edge_order)


def test_tensor_values_mirror_the_snapshot(
    snapshot_step5: GraphSnapshot, features_step5: GraphFeatureRecord
) -> None:
    tensors = to_tensors(snapshot_step5, features_step5)
    assert tensors.x.tolist() == snapshot_step5.node_features
    assert tensors.x_mask.tolist() == [[bool(v) for v in row] for row in snapshot_step5.node_mask]
    assert tensors.edge_index.tolist() == snapshot_step5.edge_index
    assert tensors.edge_attr.tolist() == snapshot_step5.edge_features
    assert tensors.derived.tolist() == features_step5.values


def test_derived_block_is_explicit_zero_without_features(
    snapshot_step5: GraphSnapshot,
) -> None:
    n = len(snapshot_step5.node_order)
    tensors = to_tensors(snapshot_step5)
    assert tensors.derived.shape == (n, 8)
    assert not tensors.derived.any()
    assert tensors.derived_mask.shape == (n, 8)
    assert not tensors.derived_mask.any()


def test_feature_record_node_order_mismatch_raises(
    snapshot_step5: GraphSnapshot, features_step5: GraphFeatureRecord
) -> None:
    shuffled = features_step5.model_copy(
        update={
            "node_order": list(reversed(features_step5.node_order)),
        }
    )
    with pytest.raises(GraphConversionError) as excinfo:
        to_tensors(snapshot_step5, shuffled)
    assert excinfo.value.code is ErrorCode.INVARIANT_VIOLATION


# --------------------------------------------------------------------------- #
# Batching (DC-07h)
# --------------------------------------------------------------------------- #
def test_collate_stacks_items_sharing_order(snapshot_step5: GraphSnapshot) -> None:
    n = len(snapshot_step5.node_order)
    e = len(snapshot_step5.edge_order)
    items = [to_tensors(snapshot_step5) for _ in range(3)]
    batch = collate(items)

    assert batch.batch_size == 3
    assert batch.x.shape == (3, n, 11)
    assert batch.x.dtype == np.float32
    assert batch.edge_index.shape == (3, 2, e)
    assert batch.edge_index.dtype == np.int64
    assert batch.edge_attr.shape == (3, e, 6)
    assert batch.derived.shape == (3, n, 8)
    assert batch.x_mask.dtype == np.bool_
    assert batch.node_order == tuple(snapshot_step5.node_order)
    assert batch.edge_order == tuple(snapshot_step5.edge_order)


def test_collate_rejects_mismatched_node_order(
    snapshot_step5: GraphSnapshot,
    tiny_observation,
    tiny_topology: Topology,
) -> None:
    big = to_tensors(snapshot_step5)
    small = to_tensors(convert(tiny_observation, tiny_topology))
    with pytest.raises(GraphConversionError) as excinfo:
        collate([big, small])
    assert excinfo.value.code is ErrorCode.INVARIANT_VIOLATION


def test_collate_rejects_mismatched_edge_order(snapshot_step5: GraphSnapshot) -> None:
    reference = to_tensors(snapshot_step5)
    # Same node order, but a different edge order must still be rejected.
    other = dataclasses.replace(
        reference, edge_order=tuple(reversed(reference.edge_order))
    )
    with pytest.raises(GraphConversionError) as excinfo:
        collate([reference, other])
    assert excinfo.value.code is ErrorCode.INVARIANT_VIOLATION


def test_collate_rejects_empty_sequence() -> None:
    with pytest.raises(GraphConversionError) as excinfo:
        collate([])
    assert excinfo.value.code is ErrorCode.INVARIANT_VIOLATION


# --------------------------------------------------------------------------- #
# Optional lazy torch conversion (DC-07h)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_TORCH, reason="torch is an optional dependency")
def test_to_torch_preserves_dtypes(
    snapshot_step5: GraphSnapshot, features_step5: GraphFeatureRecord
) -> None:
    import torch

    tensors = to_tensors(snapshot_step5, features_step5).to_torch()
    assert tensors["x"].dtype is torch.float32
    assert tensors["edge_index"].dtype is torch.int64
    assert tensors["x_mask"].dtype is torch.bool
    assert tensors["derived"].dtype is torch.float32
    assert tuple(tensors["x"].shape) == (len(snapshot_step5.node_order), 11)

    batch = collate([to_tensors(snapshot_step5) for _ in range(2)]).to_torch()
    assert tuple(batch["x"].shape) == (2, len(snapshot_step5.node_order), 11)
    assert batch["edge_index"].dtype is torch.int64
