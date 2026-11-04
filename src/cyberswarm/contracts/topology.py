"""Static network records: :class:`Node`, :class:`Edge`, :class:`Topology`.

The topology validators enforce the structural rules of CONTRACTS section 7.3,
including the two rules this stage must guarantee for the graph pipeline:
``N >= 2`` and a connected ``network_link`` graph. Connectivity and the
service-dependency acyclicity check use only the standard library (the contracts
package never imports networkx).
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from .common import ContractError, ContractModel, ShortText, UnitFloat
from .ids import AgentId, EdgeId, NodeId, TopologyId, ZoneId, agent_id_for_zone


class NodeType(StrEnum):
    """Abstract node role."""

    gateway = "gateway"
    web = "web"
    app = "app"
    database = "database"
    auth = "auth"
    file = "file"
    workstation = "workstation"


class EdgeType(StrEnum):
    """Directed edge kind."""

    network_link = "network_link"
    service_dependency = "service_dependency"


class Node(ContractModel):
    """A network node (CONTRACTS section 7.1)."""

    node_id: NodeId
    node_type: NodeType
    zone_id: ZoneId
    criticality: UnitFloat
    label: ShortText | None = None


class Edge(ContractModel):
    """A directed network edge (CONTRACTS section 7.2)."""

    edge_id: EdgeId
    source: NodeId
    target: NodeId
    edge_type: EdgeType

    @model_validator(mode="after")
    def _no_self_loop(self) -> "Edge":
        if self.source == self.target:
            raise ValueError("edge source and target must differ (no self-loop)")
        return self


def _expected_edge_id(rank: int) -> str:
    """Return the canonical edge id for a 1-based rank (``edge-001`` ...)."""

    width = 3 if rank < 1000 else 4
    return f"edge-{rank:0{width}d}"


class Topology(ContractModel):
    """A network topology document (CONTRACTS section 7.3).

    Stored at ``configs/topologies/<topology_id>.json`` and copied into every run
    output. Zones, agents and authority are derived, never stored.
    """

    schema_version: Literal["1.0"] = "1.0"
    topology_id: TopologyId
    nodes: list[Node]
    edges: list[Edge]

    @model_validator(mode="after")
    def _validate_structure(self) -> "Topology":
        self._validate_nodes()
        self._validate_edges()
        self._validate_connected_link_graph()
        self._validate_dependency_acyclic()
        return self

    # -- node rules -------------------------------------------------------- #
    def _validate_nodes(self) -> None:
        if not (2 <= len(self.nodes) <= 64):
            raise ValueError(
                "topology must have between 2 and 64 nodes "
                "(graph features divide by N - 1)"
            )
        ids = [node.node_id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("node ids must be unique")
        if ids != sorted(ids):
            raise ValueError("nodes must be sorted ascending by node_id")

    # -- edge rules -------------------------------------------------------- #
    def _validate_edges(self) -> None:
        node_ids = {node.node_id for node in self.nodes}
        seen: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(
                    f"edge {edge.edge_id} references an unknown endpoint"
                )
            key = (edge.source, edge.target, edge.edge_type.value)
            if key in seen:
                raise ValueError(
                    "duplicate edge for (source, target, edge_type): "
                    f"{key}"
                )
            seen.add(key)

        edge_ids = [edge.edge_id for edge in self.edges]
        if edge_ids != sorted(edge_ids):
            raise ValueError("edges must be sorted ascending by edge_id")

        # edge_id numbers equal the rank of (source, target, edge_type).
        ranked = sorted(
            self.edges, key=lambda e: (e.source, e.target, e.edge_type.value)
        )
        for rank, edge in enumerate(ranked, start=1):
            expected = _expected_edge_id(rank)
            if edge.edge_id != expected:
                raise ValueError(
                    "edge_id must equal the rank of (source, target, edge_type): "
                    f"expected {expected} for {(edge.source, edge.target, edge.edge_type.value)}, "
                    f"got {edge.edge_id}"
                )

    # -- graph rules ------------------------------------------------------- #
    def _validate_connected_link_graph(self) -> None:
        """The ``network_link`` edges, read as undirected, connect all nodes."""

        adjacency: dict[str, set[str]] = {node.node_id: set() for node in self.nodes}
        for edge in self.edges:
            if edge.edge_type is EdgeType.network_link:
                adjacency[edge.source].add(edge.target)
                adjacency[edge.target].add(edge.source)

        start = self.nodes[0].node_id
        seen = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        if len(seen) != len(self.nodes):
            raise ValueError(
                "network_link edges must connect all nodes (topology is disconnected)"
            )

    def _validate_dependency_acyclic(self) -> None:
        """The ``service_dependency`` edges form a directed acyclic graph."""

        successors: dict[str, list[str]] = {node.node_id: [] for node in self.nodes}
        for edge in self.edges:
            if edge.edge_type is EdgeType.service_dependency:
                successors[edge.source].append(edge.target)

        WHITE, GREY, BLACK = 0, 1, 2
        colour = {node.node_id: WHITE for node in self.nodes}

        def visit(node_id: str) -> None:
            colour[node_id] = GREY
            for nxt in successors[node_id]:
                if colour[nxt] == GREY:
                    raise ValueError(
                        "service_dependency edges must form a DAG (cycle found)"
                    )
                if colour[nxt] == WHITE:
                    visit(nxt)
            colour[node_id] = BLACK

        for node in self.nodes:
            if colour[node.node_id] == WHITE:
                visit(node.node_id)

    # -- derived views ----------------------------------------------------- #
    @property
    def node_ids(self) -> tuple[str, ...]:
        """All node ids in canonical (ascending) order."""

        return tuple(node.node_id for node in self.nodes)

    @property
    def edge_ids(self) -> tuple[str, ...]:
        """All edge ids in canonical (ascending) order."""

        return tuple(edge.edge_id for edge in self.edges)

    @property
    def zone_ids(self) -> tuple[str, ...]:
        """The zones of the topology, ascending and unique."""

        return tuple(sorted({node.zone_id for node in self.nodes}))

    @property
    def agent_ids(self) -> tuple[str, ...]:
        """One decision agent per zone (``agent-<zone slug>``), ascending."""

        return tuple(agent_id_for_zone(zone) for zone in self.zone_ids)

    def nodes_of_zone(self, zone_id: str) -> tuple[str, ...]:
        """Node ids whose zone is ``zone_id``, ascending."""

        return tuple(
            node.node_id for node in self.nodes if node.zone_id == zone_id
        )

    def zone_of(self, node_id: str) -> str:
        """The zone that owns ``node_id``."""

        for node in self.nodes:
            if node.node_id == node_id:
                return node.zone_id
        raise ContractError(
            f"unknown node id {node_id!r}", code="UNKNOWN_ID", reason="unknown_node"
        )


def load_topology(path: Path) -> Topology:
    """Read and validate a topology JSON document, raising ``ContractError``."""

    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(
            f"cannot read topology at {path}",
            code="IO_ERROR",
            reason=str(exc),
        ) from exc
    try:
        return Topology.model_validate(raw)
    except ContractError:
        raise
    except Exception as exc:  # pydantic ValidationError and friends
        raise ContractError(
            f"invalid topology at {path}",
            reason=str(exc),
        ) from exc
