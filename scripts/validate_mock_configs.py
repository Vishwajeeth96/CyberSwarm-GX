"""Validate the task 5.1 mock topologies, scenarios and splits.

Loads every topology and scenario under ``configs/`` through the neutral
contracts (``Topology.model_validate`` / ``ScenarioConfig.model_validate``),
independently re-derives the v2 structural guarantees (connected network-link
graph, acyclic service-dependency graph, canonical edge-id ranks), and checks
``configs/experiments/splits.toml`` for cross-split disjointness and the
AD-24 rule that ``topo-mock-small-v2`` appears only in ``eval_utopo``.

Run from the repository root:

    .venv/bin/python scripts/validate_mock_configs.py

Prints ``PASS`` with counts and exits 0 on success; raises (non-zero exit) on
the first violation. Paths are resolved relative to the repository root so the
content stays repository-relative.
"""

from __future__ import annotations

import json
import sys
import tomllib
from itertools import combinations
from pathlib import Path

from cyberswarm.contracts.common import Split
from cyberswarm.contracts.scenario import ScenarioConfig
from cyberswarm.contracts.topology import EdgeType, Topology

ROOT = Path(__file__).resolve().parents[1]
TOPO_DIR = ROOT / "configs" / "topologies"
SCN_DIR = ROOT / "configs" / "scenarios" / "mock"
SPLITS_PATH = ROOT / "configs" / "experiments" / "splits.toml"

PRIMARY_SPLITS = ("train", "cal", "eval", "eval_utopo")
UNSEEN_TOPOLOGY_ID = "topo-mock-small-v2"
TRAINING_TOPOLOGY_ID = "topo-mock-small-v1"


# --------------------------------------------------------------------------- #
# Independent structural re-derivation (does not trust the contract validator)
# --------------------------------------------------------------------------- #
def expected_edge_ranks_ok(topo: Topology) -> bool:
    """Every ``edge_id`` equals the rank of ``(source, target, edge_type)``."""

    ranked = sorted(topo.edges, key=lambda e: (e.source, e.target, e.edge_type.value))
    for rank, edge in enumerate(ranked, start=1):
        if edge.edge_id != f"edge-{rank:03d}":
            return False
    # edges must also be stored in ascending edge-id order
    return [e.edge_id for e in topo.edges] == sorted(e.edge_id for e in topo.edges)


def link_graph_connected(topo: Topology) -> bool:
    """The ``network_link`` edges, read as undirected, reach every node."""

    adjacency: dict[str, set[str]] = {n.node_id: set() for n in topo.nodes}
    for edge in topo.edges:
        if edge.edge_type is EdgeType.network_link:
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
    start = topo.nodes[0].node_id
    seen = {start}
    stack = [start]
    while stack:
        for nxt in adjacency[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(topo.nodes)


def dependency_graph_is_dag(topo: Topology) -> bool:
    """The ``service_dependency`` edges form a directed acyclic graph."""

    succ: dict[str, list[str]] = {n.node_id: [] for n in topo.nodes}
    for edge in topo.edges:
        if edge.edge_type is EdgeType.service_dependency:
            succ[edge.source].append(edge.target)
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {n.node_id: WHITE for n in topo.nodes}

    def visit(node_id: str) -> bool:
        colour[node_id] = GREY
        for nxt in succ[node_id]:
            if colour[nxt] == GREY:
                return False
            if colour[nxt] == WHITE and not visit(nxt):
                return False
        colour[node_id] = BLACK
        return True

    return all(colour[n.node_id] != WHITE or visit(n.node_id) for n in topo.nodes)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_topologies() -> dict[str, Topology]:
    topologies: dict[str, Topology] = {}
    for path in sorted(TOPO_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        topo = Topology.model_validate(raw)  # raises on any contract violation
        assert topo.topology_id == path.stem, (
            f"{path.name}: topology_id {topo.topology_id!r} != file name"
        )
        topologies[topo.topology_id] = topo
    return topologies


def load_scenarios() -> dict[str, ScenarioConfig]:
    scenarios: dict[str, ScenarioConfig] = {}
    for path in sorted(SCN_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        scn = ScenarioConfig.model_validate(raw)  # raises on any violation
        assert scn.scenario_id == path.stem, (
            f"{path.name}: scenario_id {scn.scenario_id!r} != file name"
        )
        scenarios[scn.scenario_id] = scn
    return scenarios


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    print(f"repo root: {ROOT}")

    # 1. Topologies -------------------------------------------------------- #
    topologies = load_topologies()
    assert set(topologies) == {TRAINING_TOPOLOGY_ID, UNSEEN_TOPOLOGY_ID}, (
        f"unexpected topology set: {sorted(topologies)}"
    )
    print(f"\ntopologies validated: {len(topologies)}")
    for tid in sorted(topologies):
        topo = topologies[tid]
        n_link = sum(e.edge_type is EdgeType.network_link for e in topo.edges)
        n_dep = sum(e.edge_type is EdgeType.service_dependency for e in topo.edges)
        connected = link_graph_connected(topo)
        is_dag = dependency_graph_is_dag(topo)
        ranks_ok = expected_edge_ranks_ok(topo)
        assert connected, f"{tid}: network_link graph is not connected"
        assert is_dag, f"{tid}: service_dependency graph is not a DAG"
        assert ranks_ok, f"{tid}: edge ids do not match the rank rule"
        print(
            f"  {tid}: nodes={len(topo.nodes)} zones={len(topo.zone_ids)} "
            f"edges={len(topo.edges)} (link={n_link}, dep={n_dep}) "
            f"connected={connected} dag={is_dag} edge_ranks_ok={ranks_ok}"
        )

    # v1 and v2 share node ids and zones but differ in edges/criticalities
    v1, v2 = topologies[TRAINING_TOPOLOGY_ID], topologies[UNSEEN_TOPOLOGY_ID]
    assert v1.node_ids == v2.node_ids, "v1 and v2 must share node ids"
    assert v1.zone_ids == v2.zone_ids, "v1 and v2 must share zones"
    v1_edges = {(e.source, e.target, e.edge_type.value) for e in v1.edges}
    v2_edges = {(e.source, e.target, e.edge_type.value) for e in v2.edges}
    assert v1_edges != v2_edges, "v2 must use a different edge set than v1"
    v1_crit = {n.node_id: n.criticality for n in v1.nodes}
    v2_crit = {n.node_id: n.criticality for n in v2.nodes}
    assert v1_crit != v2_crit, "v2 must use different criticalities than v1"
    changed = sorted(k for k in v1_crit if v1_crit[k] != v2_crit[k])
    print(
        f"  v2 is an unseen variant of v1: same {len(v1.node_ids)} node ids and "
        f"{len(v1.zone_ids)} zones, {len(v1_edges ^ v2_edges)} differing edges, "
        f"changed criticalities for {changed}"
    )

    # 2. Scenarios --------------------------------------------------------- #
    scenarios = load_scenarios()
    print(f"\nscenarios validated: {len(scenarios)}")
    for sid in sorted(scenarios):
        scn = scenarios[sid]
        assert scn.topology_id in topologies, (
            f"{sid}: references unknown topology {scn.topology_id!r}"
        )
        topo = topologies[scn.topology_id]
        assert scn.incident.entry_node_id in topo.node_ids, (
            f"{sid}: entry node {scn.incident.entry_node_id!r} not in {scn.topology_id}"
        )
        assert 1 <= scn.incident.start_step < scn.max_steps, (
            f"{sid}: start_step {scn.incident.start_step} not in [1, {scn.max_steps})"
        )
        if scn.objective_node_id is not None:
            assert scn.objective_node_id in topo.node_ids, (
                f"{sid}: objective node {scn.objective_node_id!r} not in {scn.topology_id}"
            )
    for prefix in ("train", "cal", "eval", "utopo"):
        ids = sorted(s for s in scenarios if f"-{prefix}-" in s)
        print(f"  {prefix:<6}: {len(ids)} -> {ids}")

    # 3. Splits ------------------------------------------------------------ #
    splits = tomllib.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    for name in PRIMARY_SPLITS:
        assert name in splits, f"splits.toml missing [{name}]"
        # every primary split name is a valid Split enum member (AD-24 added eval_utopo)
        Split(name)
    smoke = splits.get("smoke", {})

    scn_sets = {name: set(splits[name]["scenarios"]) for name in PRIMARY_SPLITS}
    seed_sets = {name: set(splits[name]["seeds"]) for name in PRIMARY_SPLITS}

    # every referenced scenario exists on disk
    for name in PRIMARY_SPLITS:
        for sid in scn_sets[name]:
            assert sid in scenarios, f"[{name}] references missing scenario {sid!r}"

    # scenario-id and seed disjointness across the four primary splits
    for a, b in combinations(PRIMARY_SPLITS, 2):
        inter_scn = scn_sets[a] & scn_sets[b]
        inter_seed = seed_sets[a] & seed_sets[b]
        assert not inter_scn, f"scenario overlap {a}/{b}: {sorted(inter_scn)}"
        assert not inter_seed, f"seed overlap {a}/{b}: {sorted(inter_seed)}"

    # smoke tables are a strict subset of their parent split
    for child in ("train", "cal", "eval"):
        if child in smoke:
            assert set(smoke[child]["scenarios"]) <= scn_sets[child], (
                f"smoke.{child} scenarios are not a subset of {child}"
            )
            assert set(smoke[child]["seeds"]) <= seed_sets[child], (
                f"smoke.{child} seeds are not a subset of {child}"
            )

    # AD-24: topo-mock-small-v2 appears only via eval_utopo
    topo_of_split = {
        name: {scenarios[sid].topology_id for sid in scn_sets[name]}
        for name in PRIMARY_SPLITS
    }
    assert topo_of_split["eval_utopo"] == {UNSEEN_TOPOLOGY_ID}, (
        f"eval_utopo must be on {UNSEEN_TOPOLOGY_ID}, got {topo_of_split['eval_utopo']}"
    )
    for name in ("train", "cal", "eval"):
        assert UNSEEN_TOPOLOGY_ID not in topo_of_split[name], (
            f"{name} must not use the unseen topology {UNSEEN_TOPOLOGY_ID}"
        )
        assert topo_of_split[name] == {TRAINING_TOPOLOGY_ID}, (
            f"{name} must be on {TRAINING_TOPOLOGY_ID}, got {topo_of_split[name]}"
        )

    print("\nsplits validated:")
    for name in PRIMARY_SPLITS:
        pairs = len(scn_sets[name]) * len(seed_sets[name])
        topo_ids = ", ".join(sorted(topo_of_split[name]))
        print(
            f"  {name:<11}: scenarios={len(scn_sets[name])} seeds={len(seed_sets[name])} "
            f"matched_pairs={pairs} topology={topo_ids}"
        )
    for child in ("train", "cal", "eval"):
        if child in smoke:
            pairs = len(smoke[child]["scenarios"]) * len(smoke[child]["seeds"])
            print(
                f"  smoke.{child:<5}: scenarios={len(smoke[child]['scenarios'])} "
                f"seeds={len(smoke[child]['seeds'])} matched_pairs={pairs}"
            )
    print("  cross-split scenario ids disjoint: True")
    print("  cross-split seeds disjoint: True")
    print(f"  {UNSEEN_TOPOLOGY_ID} only in eval_utopo: True")

    print(
        f"\nPASS: {len(topologies)} topologies, {len(scenarios)} scenarios, "
        f"{len(PRIMARY_SPLITS)} splits (+ {len(smoke)} smoke subsets) validated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
