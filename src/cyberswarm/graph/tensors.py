"""Model-compatible tensors and batches (CONTRACTS 10, DESIGN section 4.5).

:class:`GraphTensors` holds one snapshot (and its optional derived features) as
numpy arrays with the documented shapes and dtypes; :class:`GraphBatch` stacks a
list of them. Both expose ``to_torch()``, which imports ``torch`` **lazily** so
the graph package imports and runs the whole conversion and feature pipeline
with numpy and networkx alone - no learned-GNN dependency (Requirement 5.4).

Documented shapes and dtypes (``N`` nodes, ``E`` edges, ``B`` batch):

================  =====================  ==================
field             GraphTensors           GraphBatch
================  =====================  ==================
``x``             ``(N, 11)`` float32    ``(B, N, 11)`` float32
``x_mask``        ``(N, 11)`` bool       ``(B, N, 11)`` bool
``edge_index``    ``(2, E)`` int64       ``(B, 2, E)`` int64
``edge_attr``     ``(E, 6)`` float32     ``(B, E, 6)`` float32
``edge_mask``     ``(E, 6)`` bool        ``(B, E, 6)`` bool
``derived``       ``(N, 8)`` float32     ``(B, N, 8)`` float32
``derived_mask``  ``(N, 8)`` bool        ``(B, N, 8)`` bool
================  =====================  ==================

When :func:`to_tensors` is called without a feature record, ``derived`` is an
all-zero ``(N, 8)`` array and ``derived_mask`` is all ``False`` so "no derived
features" is explicit and the batch still stacks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..contracts import GraphFeatureRecord, GraphSnapshot, ErrorCode
from .errors import GraphConversionError

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch


def _import_torch() -> Any:
    """Import torch lazily; raise a clear error when the optional dep is absent."""

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised without torch
        raise GraphConversionError(
            ErrorCode.INVARIANT_VIOLATION,
            "to_torch() needs the optional 'torch' dependency (install the 'rl' extra)",
        ) from exc
    return torch


@dataclass(frozen=True, eq=False)
class GraphTensors:
    """One snapshot as numpy arrays (shapes and dtypes documented above)."""

    x: np.ndarray
    x_mask: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_mask: np.ndarray
    derived: np.ndarray
    derived_mask: np.ndarray
    node_order: tuple[str, ...]
    edge_order: tuple[str, ...]

    def to_torch(self) -> dict[str, "torch.Tensor"]:
        """Return the arrays as torch tensors (torch imported lazily)."""

        torch = _import_torch()
        return {
            "x": torch.from_numpy(np.ascontiguousarray(self.x)),
            "x_mask": torch.from_numpy(np.ascontiguousarray(self.x_mask)),
            "edge_index": torch.from_numpy(np.ascontiguousarray(self.edge_index)),
            "edge_attr": torch.from_numpy(np.ascontiguousarray(self.edge_attr)),
            "edge_mask": torch.from_numpy(np.ascontiguousarray(self.edge_mask)),
            "derived": torch.from_numpy(np.ascontiguousarray(self.derived)),
            "derived_mask": torch.from_numpy(np.ascontiguousarray(self.derived_mask)),
        }


@dataclass(frozen=True, eq=False)
class GraphBatch:
    """A stack of :class:`GraphTensors` sharing node and edge order."""

    x: np.ndarray
    x_mask: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_mask: np.ndarray
    derived: np.ndarray
    derived_mask: np.ndarray
    node_order: tuple[str, ...]
    edge_order: tuple[str, ...]
    batch_size: int

    def to_torch(self) -> dict[str, "torch.Tensor"]:
        """Return the batched arrays as torch tensors (torch imported lazily)."""

        torch = _import_torch()
        return {
            "x": torch.from_numpy(np.ascontiguousarray(self.x)),
            "x_mask": torch.from_numpy(np.ascontiguousarray(self.x_mask)),
            "edge_index": torch.from_numpy(np.ascontiguousarray(self.edge_index)),
            "edge_attr": torch.from_numpy(np.ascontiguousarray(self.edge_attr)),
            "edge_mask": torch.from_numpy(np.ascontiguousarray(self.edge_mask)),
            "derived": torch.from_numpy(np.ascontiguousarray(self.derived)),
            "derived_mask": torch.from_numpy(np.ascontiguousarray(self.derived_mask)),
        }


def to_tensors(
    snapshot: GraphSnapshot, features: GraphFeatureRecord | None = None
) -> GraphTensors:
    """Build :class:`GraphTensors` from a snapshot and optional derived features."""

    node_count = len(snapshot.node_order)
    x = np.asarray(snapshot.node_features, dtype=np.float32).reshape(node_count, 11)
    x_mask = np.asarray(snapshot.node_mask, dtype=bool).reshape(node_count, 11)
    edge_count = len(snapshot.edge_order)
    edge_index = np.asarray(snapshot.edge_index, dtype=np.int64).reshape(2, edge_count)
    edge_attr = np.asarray(snapshot.edge_features, dtype=np.float32).reshape(
        edge_count, 6
    )
    edge_mask = np.asarray(snapshot.edge_mask, dtype=bool).reshape(edge_count, 6)

    if features is not None:
        if list(features.node_order) != list(snapshot.node_order):
            raise GraphConversionError(
                ErrorCode.INVARIANT_VIOLATION,
                "feature record node_order must match the snapshot node_order",
            )
        derived = np.asarray(features.values, dtype=np.float32).reshape(node_count, 8)
        derived_mask = np.asarray(features.mask, dtype=bool).reshape(node_count, 8)
    else:
        derived = np.zeros((node_count, 8), dtype=np.float32)
        derived_mask = np.zeros((node_count, 8), dtype=bool)

    return GraphTensors(
        x=x,
        x_mask=x_mask,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_mask=edge_mask,
        derived=derived,
        derived_mask=derived_mask,
        node_order=tuple(snapshot.node_order),
        edge_order=tuple(snapshot.edge_order),
    )


def collate(items: Sequence[GraphTensors]) -> GraphBatch:
    """Stack :class:`GraphTensors` into a :class:`GraphBatch`.

    Every item must share the same ``node_order`` and ``edge_order``; otherwise
    the stacked tensors would not line up and a
    :class:`~cyberswarm.graph.errors.GraphConversionError` is raised.
    """

    if not items:
        raise GraphConversionError(
            ErrorCode.INVARIANT_VIOLATION, "cannot collate an empty sequence"
        )
    reference = items[0]
    for item in items[1:]:
        if item.node_order != reference.node_order:
            raise GraphConversionError(
                ErrorCode.INVARIANT_VIOLATION,
                "all batch items must share the same node_order",
            )
        if item.edge_order != reference.edge_order:
            raise GraphConversionError(
                ErrorCode.INVARIANT_VIOLATION,
                "all batch items must share the same edge_order",
            )

    return GraphBatch(
        x=np.stack([item.x for item in items]),
        x_mask=np.stack([item.x_mask for item in items]),
        edge_index=np.stack([item.edge_index for item in items]),
        edge_attr=np.stack([item.edge_attr for item in items]),
        edge_mask=np.stack([item.edge_mask for item in items]),
        derived=np.stack([item.derived for item in items]),
        derived_mask=np.stack([item.derived_mask for item in items]),
        node_order=reference.node_order,
        edge_order=reference.edge_order,
        batch_size=len(items),
    )
