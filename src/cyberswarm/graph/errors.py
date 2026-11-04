"""Errors raised by the graph pipeline (DESIGN section 4.5).

The topology helper and the converter raise :class:`GraphConversionError` for a
guard the graph package owns itself: an unknown identifier passed to
:class:`~cyberswarm.graph.topology_graph.TopologyGraph`, an observation that
references a node or edge outside the topology, a batch whose members disagree
on node or edge order, or a missing optional dependency in
:meth:`~cyberswarm.graph.tensors.GraphTensors.to_torch`.

It mirrors :class:`cyberswarm.interfaces.SimulatorError`: a structured
:class:`~cyberswarm.contracts.ErrorCode` plus a human-readable message.
Malformed *records* remain the responsibility of the contracts layer, which
raises its own ``ContractError``/``ValidationError``; this exception is only for
the graph package's own invariants.
"""

from __future__ import annotations

from ..contracts import ErrorCode


class GraphConversionError(Exception):
    """A graph-pipeline invariant was violated (unknown id, batch mismatch...).

    ``code`` is an :class:`~cyberswarm.contracts.ErrorCode`.
    :data:`ErrorCode.UNKNOWN_ID` is raised when a node or edge id cannot be
    resolved against the topology; :data:`ErrorCode.INVARIANT_VIOLATION` is
    raised when a batch's members disagree on ordering or when an optional
    dependency is missing.
    """

    code: ErrorCode

    def __init__(self, code: ErrorCode, message: str | None = None) -> None:
        self.code = code
        super().__init__(message if message is not None else str(code.value))
