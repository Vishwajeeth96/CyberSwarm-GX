"""Observation-to-graph pipeline (stage S3, DESIGN section 4.5).

Required lead-owned infrastructure that turns agent-visible observations into
deterministic, inspectable graph tensors for the conventional-versus-graph
comparison. It uses no learned message passing: ``torch`` is imported lazily and
only inside :meth:`GraphTensors.to_torch` / :meth:`GraphBatch.to_torch`, so

    import cyberswarm.graph

succeeds with numpy and networkx alone (Requirement 5.4).

Public surface:

* :class:`TopologyGraph` - stable ordering, id resolution, structural queries.
* :class:`GraphConverter` / :func:`convert` - observation -> ``GraphSnapshot``
  (``graph-base-v1`` + ``impute-v1``), taking no ground-truth argument.
* :func:`compute_features` - snapshot + topology -> ``GraphFeatureRecord``
  (``graph-derived-v1``), taking no ground-truth argument.
* :class:`GraphTensors` / :class:`GraphBatch` / :func:`to_tensors` /
  :func:`collate` - the numpy tensor and batch interface.
* :class:`GraphConversionError` - raised for the pipeline's own invariants.
"""

from __future__ import annotations

from .converter import CONVERTER_ID, GraphConverter, convert
from .errors import GraphConversionError
from .features import compute_features
from .tensors import GraphBatch, GraphTensors, collate, to_tensors
from .topology_graph import TopologyGraph

__all__ = [
    "TopologyGraph",
    "GraphConverter",
    "convert",
    "CONVERTER_ID",
    "compute_features",
    "GraphTensors",
    "GraphBatch",
    "to_tensors",
    "collate",
    "GraphConversionError",
]
