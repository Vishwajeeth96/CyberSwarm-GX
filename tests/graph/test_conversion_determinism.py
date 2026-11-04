"""Deterministic, stably-ordered conversion (stage S3, DC-07c, DC-07d).

A repeated conversion of the same observation is byte-identical, the result
reproduces the committed golden snapshot exactly, and node/edge order equals the
topology order regardless of which (or how many) nodes the agent observed. Both
the free ``convert`` function and a reused ``GraphConverter`` produce the same
bytes, including through the tolerated ``convert(topology, observation)``
argument order.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from cyberswarm.contracts import (
    AgentObservation,
    NodeObservation,
    NodeStatus,
    Topology,
    dumps_canonical,
)
from cyberswarm.graph import GraphConverter, convert

_VALID_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "records" / "valid"


# --------------------------------------------------------------------------- #
# Determinism (DC-07c)
# --------------------------------------------------------------------------- #
def test_repeated_conversion_is_byte_identical(
    obs_step5: AgentObservation, topology_v1: Topology
) -> None:
    first = convert(obs_step5, topology_v1)
    second = convert(obs_step5, topology_v1)
    assert dumps_canonical(first) == dumps_canonical(second)


def test_free_function_and_reused_converter_agree(
    obs_step5: AgentObservation, topology_v1: Topology
) -> None:
    reusable = GraphConverter(topology_v1)
    from_free = convert(obs_step5, topology_v1)
    from_object_a = reusable.convert(obs_step5)
    from_object_b = reusable.convert(obs_step5)
    assert dumps_canonical(from_object_a) == dumps_canonical(from_free)
    assert dumps_canonical(from_object_b) == dumps_canonical(from_free)


def test_argument_order_is_tolerated(
    obs_step5: AgentObservation, topology_v1: Topology
) -> None:
    # convert(observation, topology) and convert(topology, observation) agree.
    canonical = dumps_canonical(convert(obs_step5, topology_v1))
    assert dumps_canonical(convert(topology_v1, obs_step5)) == canonical


def test_conversion_matches_the_committed_golden_snapshot(
    obs_step5: AgentObservation, topology_v1: Topology
) -> None:
    produced = convert(obs_step5, topology_v1)
    golden = (_VALID_FIXTURES / "GraphSnapshot.dmz-step5.json").read_text().strip()
    assert dumps_canonical(produced) == golden


# --------------------------------------------------------------------------- #
# Stable ordering (DC-07d)
# --------------------------------------------------------------------------- #
def test_node_and_edge_order_always_equal_the_topology(
    make_node: Callable[..., NodeObservation],
    make_obs: Callable[..., AgentObservation],
    topology_v1: Topology,
) -> None:
    node_order = list(topology_v1.node_ids)
    edge_order = list(topology_v1.edge_ids)

    # Empty observation: still the full, ordered topology.
    empty = convert(make_obs([], step=0), topology_v1)
    assert empty.node_order == node_order
    assert empty.edge_order == edge_order

    # A partial, single-node observation: order is unchanged.
    partial = convert(make_obs([make_node("node-web-02", alert_count=3)], step=9), topology_v1)
    assert partial.node_order == node_order
    assert partial.edge_order == edge_order

    # edge_index is a pure function of the topology, so it is stable too.
    assert empty.edge_index == partial.edge_index


def test_order_is_stable_across_steps_with_changing_state(
    make_node: Callable[..., NodeObservation],
    make_obs: Callable[..., AgentObservation],
    topology_v1: Topology,
) -> None:
    orders = set()
    edge_orders = set()
    for step, status in enumerate(
        [NodeStatus.up, NodeStatus.rate_limited, NodeStatus.quarantined, NodeStatus.isolated]
    ):
        snap = convert(
            make_obs([make_node("node-gw-01", status=status, alert_count=step)], step=step),
            topology_v1,
        )
        orders.add(tuple(snap.node_order))
        edge_orders.add(tuple(snap.edge_order))
    # Exactly one order seen across every step.
    assert orders == {tuple(topology_v1.node_ids)}
    assert edge_orders == {tuple(topology_v1.edge_ids)}
