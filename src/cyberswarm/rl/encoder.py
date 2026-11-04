"""Observation encoders for the two representations (stage S3, RL_IMPLEMENTATION 3).

:class:`ObservationEncoder` turns one agent's
:class:`~cyberswarm.interfaces.DecisionContext` into a fixed-length ``float32``
vector whose every entry is finite and in ``[0, 1]``. Both modes share the same
node-slot layout so the conventional-versus-graph comparison is matched; the
graph mode only appends eight hand-engineered columns per node.

Layout (canonical node order over all ``N`` topology nodes). Per node, the
``conventional`` block is 17 values:

===== ========================================================= ================
Index Name                                                      Source
===== ========================================================= ================
0     ``criticality``                                           topology (public)
1-5   ``status_up`` ... ``status_restoring`` (one-hot)          observation
6     ``alert_count_norm``                                      observation
7     ``max_alert_severity_norm``                               observation
8     ``evidence_level_norm`` (0 if null)                       observation
9     ``service_health`` (0 if null)                            observation
10    ``steps_since_investigated_norm`` (0 if null)             observation
11    ``suspicion``                                             detection summary
12    ``observed`` (1 if the node is in the observation)        observation
13    ``own_scope`` (1 if ``scope = own``)                      observation
14-16 ``evidence_missing``/``service_health_missing``/          observation
      ``investigated_missing`` (1 if observed and the value is null)
===== ========================================================= ================

Columns 0-10 are exactly the ``graph-base-v1`` node-feature columns and are read
from the :class:`~cyberswarm.contracts.GraphSnapshot`, so there is a single
conversion path. The ``graph_features`` mode appends the eight
``graph-derived-v1`` columns (CONTRACTS 10.2) after the base block, making the
per-node block 25 values. The vector ends with two globals: ``step_norm = t / H``
and ``last_action_rejected``.

====================== ============== =============
Mode                   Dimension      Mock (N = 8)
====================== ============== =============
``conventional``       ``17 N + 2``   138
``graph_features``     ``25 N + 2``   202
====================== ============== =============

No ground truth is read: the encoder sees only the agent-visible snapshot,
observation and detection summary. It builds its own deterministic snapshot and
features when the context does not already carry them (a conventional-only run),
using the same :mod:`cyberswarm.graph` converter the engine uses.

Imports are limited to :mod:`cyberswarm.contracts`, :mod:`cyberswarm.interfaces`,
:mod:`cyberswarm.graph` and :mod:`numpy`; no simulator, environment or ``torch``.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ..contracts import (
    DERIVED_FEATURE_NAMES_V1,
    NODE_FEATURE_NAMES_BASE_V1,
    AgentId,
    NodeScope,
    Representation,
    Topology,
)
from ..graph import GraphConverter, compute_features
from ..interfaces.policy import DecisionContext

__all__ = [
    "EXTRA_NODE_FEATURE_NAMES",
    "GLOBAL_FEATURE_NAMES",
    "CONVENTIONAL_NODE_WIDTH",
    "GRAPH_NODE_WIDTH",
    "GLOBAL_WIDTH",
    "ObservationEncoder",
]


#: The six per-node columns (indices 11-16) that the base ``graph-base-v1`` block
#: does not carry: the detection suspicion and the observed/scope/missing flags.
EXTRA_NODE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "suspicion",
    "observed",
    "own_scope",
    "evidence_missing",
    "service_health_missing",
    "investigated_missing",
)

#: The two trailing global columns.
GLOBAL_FEATURE_NAMES: Final[tuple[str, ...]] = ("step_norm", "last_action_rejected")

#: Per-node block widths and the global suffix width.
CONVENTIONAL_NODE_WIDTH: Final[int] = len(NODE_FEATURE_NAMES_BASE_V1) + len(
    EXTRA_NODE_FEATURE_NAMES
)  # 17
GRAPH_NODE_WIDTH: Final[int] = CONVENTIONAL_NODE_WIDTH + len(DERIVED_FEATURE_NAMES_V1)  # 25
GLOBAL_WIDTH: Final[int] = len(GLOBAL_FEATURE_NAMES)  # 2

#: Number of ``graph-base-v1`` columns taken verbatim from the snapshot.
_BASE_COLS: Final[int] = len(NODE_FEATURE_NAMES_BASE_V1)  # 11


class ObservationEncoder:
    """Encode one agent's :class:`DecisionContext` into a ``[0, 1]`` ``float32`` vector."""

    def __init__(
        self, topology: Topology, agent_id: AgentId, mode: Representation | str
    ) -> None:
        self._topology = topology
        self._agent_id = agent_id
        self._mode = Representation(mode)
        if self._mode not in (
            Representation.conventional,
            Representation.graph_features,
        ):
            raise ValueError(
                f"ObservationEncoder supports only conventional and graph_features, "
                f"not {self._mode.value!r}"
            )
        self._graph = self._mode is Representation.graph_features
        self._converter = GraphConverter(topology)
        self._node_order: tuple[str, ...] = topology.node_ids
        self._n = len(self._node_order)
        self._node_width = GRAPH_NODE_WIDTH if self._graph else CONVENTIONAL_NODE_WIDTH
        self._dim = self._node_width * self._n + GLOBAL_WIDTH

        block = list(NODE_FEATURE_NAMES_BASE_V1) + list(EXTRA_NODE_FEATURE_NAMES)
        if self._graph:
            block += list(DERIVED_FEATURE_NAMES_V1)
        names: list[str] = []
        for node_id in self._node_order:
            names.extend(f"{node_id}:{name}" for name in block)
        names.extend(GLOBAL_FEATURE_NAMES)
        self._feature_names: tuple[str, ...] = tuple(names)

    @property
    def mode(self) -> Representation:
        """The observation representation this encoder produces."""

        return self._mode

    @property
    def dim(self) -> int:
        """The vector dimension ``D`` (``17 N + 2`` or ``25 N + 2``)."""

        return self._dim

    @property
    def feature_names(self) -> tuple[str, ...]:
        """One name per entry of the encoded vector (length ``dim``)."""

        return self._feature_names

    def encode(
        self, ctx: DecisionContext, *, last_action_rejected: bool
    ) -> np.ndarray:
        """Encode ``ctx`` into a ``float32`` vector of shape ``(dim,)`` in ``[0, 1]``.

        ``last_action_rejected`` is the trailing global flag: ``True`` if any of
        this agent's proposals was rejected in the previous step.
        """

        snapshot = ctx.graph_snapshot
        if snapshot is None:
            snapshot = self._converter.convert(ctx.observation)

        base = np.asarray(snapshot.node_features, dtype=np.float32)
        snap_index = {node_id: i for i, node_id in enumerate(snapshot.node_order)}

        if self._graph:
            features = ctx.graph_features
            if features is None:
                features = compute_features(snapshot, self._converter.graph)
            derived = np.asarray(features.values, dtype=np.float32)
            derived_index = {
                node_id: i for i, node_id in enumerate(features.node_order)
            }
        else:
            derived = None
            derived_index = {}

        suspicion = {node.node_id: float(node.suspicion) for node in ctx.detection.nodes}
        observed = {node.node_id: node for node in ctx.observation.nodes}

        grid = np.zeros((self._n, self._node_width), dtype=np.float32)
        for row, node_id in enumerate(self._node_order):
            grid[row, 0:_BASE_COLS] = base[snap_index[node_id]]
            grid[row, 11] = suspicion.get(node_id, 0.0)
            node_obs = observed.get(node_id)
            if node_obs is not None:
                grid[row, 12] = 1.0
                grid[row, 13] = 1.0 if node_obs.scope is NodeScope.own else 0.0
                grid[row, 14] = 1.0 if node_obs.evidence_level is None else 0.0
                grid[row, 15] = 1.0 if node_obs.service_health is None else 0.0
                grid[row, 16] = (
                    1.0 if node_obs.steps_since_investigated is None else 0.0
                )
            if derived is not None:
                grid[row, _BASE_COLS + len(EXTRA_NODE_FEATURE_NAMES) :] = derived[
                    derived_index[node_id]
                ]

        step_norm = ctx.step / ctx.max_steps if ctx.max_steps > 0 else 0.0
        globals_ = np.array(
            [step_norm, 1.0 if last_action_rejected else 0.0], dtype=np.float32
        )
        vector = np.concatenate([grid.reshape(-1), globals_]).astype(np.float32)
        np.clip(vector, 0.0, 1.0, out=vector)
        return vector
