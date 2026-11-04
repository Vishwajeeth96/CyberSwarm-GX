"""Static topology as a NetworkX graph with stable ordering (DESIGN section 4.5).

:class:`TopologyGraph` wraps a validated :class:`~cyberswarm.contracts.Topology`
and exposes the deterministic order and the structural queries the converter and
the hand-engineered features need:

* ``node_order`` / ``edge_order`` - the canonical ascending id order, identical
  to :attr:`Topology.node_ids` / :attr:`Topology.edge_ids`.
* ``index_of`` - the row index of a node in ``node_order``.
* ``link_neighbors`` - the ``network_link`` neighbours of a node, in either
  direction (the set ``L(n)`` of CONTRACTS section 10.2).
* ``dependents`` - every node that depends on a node directly or transitively
  over ``service_dependency`` edges.
* ``diameter`` / ``hops`` - distances over the undirected ``network_link`` graph.

``.nx`` is a directed convenience view (``networkx.DiGraph``) carrying the
``edge_type`` attribute on every edge. A ``Topology`` may hold a ``network_link``
and a ``service_dependency`` edge for the same ordered ``(source, target)`` pair;
a ``DiGraph`` keeps a single edge per ordered pair, so those parallel edges are
collapsed into one ``.nx`` edge whose ``edge_types`` and ``edge_ids`` attributes
retain both. The queries above never read ``.nx``; they use type-exact adjacency
built directly from the topology, so the collapse cannot affect a result.

Only the standard library and NetworkX are imported here (import boundary of
DESIGN section 3.2: the graph package may use networkx and numpy).
"""

from __future__ import annotations

from functools import cached_property

import networkx as nx

from ..contracts import EdgeType, ErrorCode, Topology
from .errors import GraphConversionError


class TopologyGraph:
    """A validated topology with stable ordering and structural queries."""

    def __init__(self, topology: Topology) -> None:
        self.topology = topology
        self.node_order: list[str] = list(topology.node_ids)
        self.edge_order: list[str] = list(topology.edge_ids)
        self._index: dict[str, int] = {
            node_id: position for position, node_id in enumerate(self.node_order)
        }
        self._criticality: dict[str, float] = {
            node.node_id: float(node.criticality) for node in topology.nodes
        }

        # Directed convenience view (collapses parallel typed edges per pair).
        self.nx: nx.DiGraph = nx.DiGraph()
        for node in topology.nodes:
            self.nx.add_node(
                node.node_id,
                criticality=float(node.criticality),
                node_type=node.node_type.value,
                zone_id=node.zone_id,
            )

        # Type-exact adjacency used by every public query.
        self._link_adjacency: dict[str, set[str]] = {
            node_id: set() for node_id in self.node_order
        }
        self._dependency_predecessors: dict[str, set[str]] = {
            node_id: set() for node_id in self.node_order
        }
        self._link_undirected: nx.Graph = nx.Graph()
        self._link_undirected.add_nodes_from(self.node_order)

        for edge in topology.edges:
            source, target, edge_type = edge.source, edge.target, edge.edge_type
            if self.nx.has_edge(source, target):
                data = self.nx.edges[source, target]
                types = tuple(sorted({*data["edge_types"], edge_type.value}))
                data["edge_types"] = types
                data["edge_ids"] = (*data["edge_ids"], edge.edge_id)
                if EdgeType.network_link.value in types:
                    data["edge_type"] = EdgeType.network_link.value
            else:
                self.nx.add_edge(
                    source,
                    target,
                    edge_type=edge_type.value,
                    edge_types=(edge_type.value,),
                    edge_ids=(edge.edge_id,),
                )

            if edge_type is EdgeType.network_link:
                self._link_adjacency[source].add(target)
                self._link_adjacency[target].add(source)
                self._link_undirected.add_edge(source, target)
            else:  # service_dependency: source depends on target
                self._dependency_predecessors[target].add(source)

    # -- id resolution ----------------------------------------------------- #
    def _require_node(self, node_id: str) -> None:
        if node_id not in self._index:
            raise GraphConversionError(
                ErrorCode.UNKNOWN_ID, f"unknown node id {node_id!r}"
            )

    def index_of(self, node_id: str) -> int:
        """Return the row index of ``node_id`` in :attr:`node_order`."""

        self._require_node(node_id)
        return self._index[node_id]

    def criticality_of(self, node_id: str) -> float:
        """Return the static criticality of ``node_id``."""

        self._require_node(node_id)
        return self._criticality[node_id]

    # -- structural queries ------------------------------------------------ #
    def link_neighbors(self, node_id: str) -> list[str]:
        """Return ``L(n)``: the ``network_link`` neighbours, either direction.

        The list is sorted ascending for determinism. The topology validator
        guarantees a connected link graph with ``N >= 2``, so the list is never
        empty (CONTRACTS section 10.2).
        """

        self._require_node(node_id)
        return sorted(self._link_adjacency[node_id])

    def dependents(self, node_id: str) -> list[str]:
        """Return every node depending on ``node_id`` over ``service_dependency``.

        A directed edge ``u -> v`` means "u depends on v"; this returns every
        ``u`` with a directed path to ``node_id`` (its transitive dependants),
        excluding ``node_id`` itself, sorted ascending.
        """

        self._require_node(node_id)
        seen: set[str] = set()
        stack = list(self._dependency_predecessors[node_id])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self._dependency_predecessors[current])
        seen.discard(node_id)
        return sorted(seen)

    @property
    def link_undirected(self) -> nx.Graph:
        """The undirected ``network_link`` graph (used for distances)."""

        return self._link_undirected

    @cached_property
    def diameter(self) -> int:
        """Diameter of the undirected ``network_link`` graph (>= 1 for N >= 2)."""

        return int(nx.diameter(self._link_undirected))

    def hops(self, a: str, b: str) -> int:
        """Shortest hop distance between ``a`` and ``b`` over ``network_link``.

        Distance is measured on the undirected link graph; ``hops(a, a)`` is 0.
        The link graph is connected, so the distance is always finite.
        """

        self._require_node(a)
        self._require_node(b)
        return int(nx.shortest_path_length(self._link_undirected, a, b))
