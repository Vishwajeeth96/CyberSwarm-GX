"""Hand-constructed reward cases R-1 .. R-12 (RL_IMPLEMENTATION section 4.3).

Each case builds ground-truth snapshots and an executed-action list directly and
checks the exact reward under the committed default weights. Most cases drive the
pure :func:`compute_breakdown` (so ``d_t`` and the terminal factor are explicit);
the delay accumulation and the terminal charge are also exercised through the
stateful :class:`RewardModel`.

Validates design Property 13 (reward linear in the weights), Property 14 (healthy
disruption measures defender-induced loss only) and Property 15 (reward ordering
of containment decisions).
"""

from __future__ import annotations

import pytest

from cyberswarm.contracts.common import NodeStatus
from cyberswarm.contracts.results import TerminationReason
from cyberswarm.contracts.vocabulary import ActionKind
from cyberswarm.rl.reward import (
    RewardModel,
    compute_breakdown,
    signed_weighted,
    weighted_total,
)


# --------------------------------------------------------------------------- #
# R-1  healthy critical versus low-value
# --------------------------------------------------------------------------- #
def test_r1_isolating_a_healthy_critical_node_costs_more(
    default_config, make_node, make_snapshot, make_execution
):
    def breakdown_for(criticality: float):
        node = make_node(
            "node-web-01", status=NodeStatus.isolated, defense_availability=0.0
        )
        return compute_breakdown(
            default_config,
            criticality={"node-web-01": criticality},
            curr=make_snapshot([node]),
            executions=[make_execution(ActionKind.isolate_node, "node-web-01")],
        )

    critical = breakdown_for(1.0)
    low_value = breakdown_for(0.1)

    # critical: false_isolation = 0.5 + 1.0, healthy_disruption = 1.0 * (1 - 0)
    assert critical.false_isolation == pytest.approx(1.5)
    assert critical.healthy_disruption == pytest.approx(1.0)
    assert critical.total == pytest.approx(-3.5)

    assert low_value.false_isolation == pytest.approx(0.6)
    assert low_value.total == pytest.approx(-1.25)

    # Property 15: the critical node is strictly more expensive to isolate.
    assert critical.total < low_value.total


# --------------------------------------------------------------------------- #
# R-2  justified containment beats inaction
# --------------------------------------------------------------------------- #
def test_r2_justified_containment_scores_higher_than_inaction(
    default_config, make_node, make_snapshot, make_execution
):
    criticality = {"node-db-01": 0.5}

    # (a) isolate the compromised node: it enters containment this step, so no
    # compromise is uncontained and the delay term is 0.
    contained = make_node(
        "node-db-01",
        compromised=True,
        status=NodeStatus.isolated,
        defense_availability=0.0,
        service_availability=0.0,
    )
    justified = compute_breakdown(
        default_config,
        criticality=criticality,
        curr=make_snapshot([contained]),
        executions=[make_execution(ActionKind.isolate_node, "node-db-01")],
        prev_contained_ids=set(),
        delay_steps=0,
    )
    assert justified.containment == pytest.approx(1.0)
    assert justified.attack_impact == pytest.approx(0.5)
    assert justified.response_delay == pytest.approx(0.0)
    assert justified.false_isolation == pytest.approx(0.0)
    assert justified.total == pytest.approx(2.5)

    # (b) no_op on the same compromised, uncontained node, first compromise 3
    # steps ago.
    uncontained = make_node(
        "node-db-01", compromised=True, status=NodeStatus.up, service_availability=0.9
    )
    inaction = compute_breakdown(
        default_config,
        criticality=criticality,
        curr=make_snapshot([uncontained]),
        executions=[],
        delay_steps=3,
    )
    assert inaction.attack_impact == pytest.approx(0.5)
    assert inaction.response_delay == pytest.approx(3 / 20)
    assert inaction.total == pytest.approx(-0.575)

    # Property 15: justified containment scores strictly higher than inaction.
    assert justified.total > inaction.total


# --------------------------------------------------------------------------- #
# R-3  delay increases the penalty monotonically, then caps at d_max
# --------------------------------------------------------------------------- #
def test_r3_delay_penalty_is_monotonic_and_capped(
    default_config, make_node, make_snapshot
):
    def delay_contribution(delay_steps: int) -> float:
        node = make_node("node-db-01", compromised=True, status=NodeStatus.up)
        breakdown = compute_breakdown(
            default_config,
            criticality={"node-db-01": 0.5},
            curr=make_snapshot([node]),
            delay_steps=delay_steps,
        )
        return signed_weighted(breakdown.raw_magnitudes(), default_config)[
            "response_delay"
        ]

    d2, d10, d20, d30 = (delay_contribution(d) for d in (2, 10, 20, 30))

    # The three values named in RL_IMPLEMENTATION 4.3.
    assert d2 == pytest.approx(-0.05)
    assert d10 == pytest.approx(-0.25)
    assert d30 == pytest.approx(-0.5)

    # Strictly increasing penalty up to the cap, then flat.
    assert abs(d2) < abs(d10) < abs(d20)
    assert abs(d20) == pytest.approx(0.5)
    assert d30 == pytest.approx(d20)


# --------------------------------------------------------------------------- #
# R-4  investigation has a small defined cost
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kind, expected",
    [
        (ActionKind.monitor_node, -0.025),
        (ActionKind.investigate_node, -0.05),
        (ActionKind.request_evidence, -0.15),
    ],
)
def test_r4_investigation_cost(
    default_config, make_node, make_snapshot, make_execution, kind, expected
):
    node = make_node("node-web-01")
    breakdown = compute_breakdown(
        default_config,
        criticality={"node-web-01": 0.5},
        curr=make_snapshot([node]),
        executions=[make_execution(kind, "node-web-01")],
    )
    assert breakdown.investigation_cost > 0.0
    assert breakdown.total == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# R-5  invalid-action guard
# --------------------------------------------------------------------------- #
def test_r5_invalid_action_guard(default_config, make_node, make_snapshot):
    node = make_node("node-web-01")
    breakdown = compute_breakdown(
        default_config,
        criticality={"node-web-01": 0.5},
        curr=make_snapshot([node]),
        invalid_count=1,
    )
    assert breakdown.invalid_action == pytest.approx(1.0)
    assert breakdown.total == pytest.approx(-0.1)


# --------------------------------------------------------------------------- #
# R-6  false-isolation floor
# --------------------------------------------------------------------------- #
def test_r6_false_isolation_floor(
    default_config, make_node, make_snapshot, make_execution
):
    node = make_node(
        "node-web-01", status=NodeStatus.isolated, defense_availability=1.0
    )
    breakdown = compute_breakdown(
        default_config,
        criticality={"node-web-01": 0.0},
        curr=make_snapshot([node]),
        executions=[make_execution(ActionKind.isolate_node, "node-web-01")],
    )
    assert breakdown.false_isolation == pytest.approx(0.5)  # base only
    weighted = signed_weighted(breakdown.raw_magnitudes(), default_config)
    assert weighted["false_isolation"] == pytest.approx(-1.0)
    assert breakdown.total == pytest.approx(-1.0)
    assert breakdown.total < 0.0


# --------------------------------------------------------------------------- #
# R-7  terminal charge removes the incentive to fail fast
# --------------------------------------------------------------------------- #
def test_r7_terminal_charge_formula(default_config, make_node, make_snapshot):
    objective = make_node(
        "node-obj", compromised=True, status=NodeStatus.up, service_availability=0.0
    )
    breakdown = compute_breakdown(
        default_config,
        criticality={"node-obj": 1.0},
        curr=make_snapshot([objective], step=10),
        terminal_factor=29,  # H - 1 - t = 40 - 1 - 10
    )
    assert breakdown.attack_impact == pytest.approx(30.0)
    assert breakdown.total == pytest.approx(-30.0)


def test_r7_terminal_charge_through_the_model(
    default_config, make_node, make_snapshot, make_topology
):
    model = RewardModel(default_config)
    model.reset(make_topology({"node-aux": 0.5, "node-obj": 1.0}), max_steps=40)
    objective = make_node(
        "node-obj", compromised=True, status=NodeStatus.up, service_availability=0.0
    )
    score = model.score(
        step=10,
        prev=None,
        curr=make_snapshot([objective], step=10),
        executions=[],
        invalid_count=0,
        termination_reason=TerminationReason.objective_reached,
    )
    assert score.raw["attack_impact"] == pytest.approx(30.0)
    assert score.weighted["attack_impact"] == pytest.approx(-30.0)
    assert score.total == pytest.approx(-30.0)


# --------------------------------------------------------------------------- #
# R-8  configurable weights and the reweight identity
# --------------------------------------------------------------------------- #
def test_r8_zeroing_a_weight_drops_only_that_term(
    default_config, make_node, make_snapshot, make_execution
):
    node = make_node(
        "node-web-01", status=NodeStatus.isolated, defense_availability=0.0
    )
    curr = make_snapshot([node])
    executions = [make_execution(ActionKind.isolate_node, "node-web-01")]
    criticality = {"node-web-01": 1.0}

    base = compute_breakdown(
        default_config, criticality=criticality, curr=curr, executions=executions
    )
    assert base.total == pytest.approx(-3.5)  # R-1 critical case

    zeroed = default_config.model_copy(
        update={
            "weights": default_config.weights.model_copy(
                update={"false_isolation": 0.0}
            )
        }
    )
    off = compute_breakdown(
        zeroed, criticality=criticality, curr=curr, executions=executions
    )

    # The raw magnitudes are unchanged; only the weighted false_isolation drops.
    assert off.raw_magnitudes() == base.raw_magnitudes()
    assert off.false_isolation == pytest.approx(1.5)
    assert off.total == pytest.approx(-0.5)  # only healthy_disruption remains

    # Property 13: the reward recomputed from the stored magnitudes matches.
    assert weighted_total(base.raw_magnitudes(), zeroed) == pytest.approx(off.total)
    assert weighted_total(base.raw_magnitudes(), default_config) == pytest.approx(
        base.total
    )


# --------------------------------------------------------------------------- #
# R-9  quiet network
# --------------------------------------------------------------------------- #
def test_r9_quiet_network_scores_zero(
    default_config, make_node, make_snapshot, make_execution
):
    nodes = [make_node("node-a"), make_node("node-b")]
    breakdown = compute_breakdown(
        default_config,
        criticality={"node-a": 0.5, "node-b": 1.0},
        curr=make_snapshot(nodes),
        executions=[make_execution(ActionKind.no_op)],
    )
    assert breakdown.total == pytest.approx(0.0)
    assert all(value == pytest.approx(0.0) for value in breakdown.raw_magnitudes().values())


# --------------------------------------------------------------------------- #
# R-10  dependency collateral of a wrong isolation
# --------------------------------------------------------------------------- #
def test_r10_dependency_collateral(
    default_config, make_node, make_snapshot, make_execution
):
    # A (crit 0.7) depends on B (crit 0.3); isolating B drops both to davail 0.
    node_a = make_node("node-a", status=NodeStatus.up, defense_availability=0.0)
    node_b = make_node("node-b", status=NodeStatus.isolated, defense_availability=0.0)
    breakdown = compute_breakdown(
        default_config,
        criticality={"node-a": 0.7, "node-b": 0.3},
        curr=make_snapshot([node_a, node_b]),
        executions=[make_execution(ActionKind.isolate_node, "node-b")],
    )
    assert breakdown.healthy_disruption == pytest.approx(1.0)  # 0.7 + 0.3
    assert breakdown.false_isolation == pytest.approx(0.8)  # base + crit(B)


# --------------------------------------------------------------------------- #
# R-11  attack-induced loss is not charged to the defense (Property 14)
# --------------------------------------------------------------------------- #
def test_r11_attack_induced_loss_is_not_charged_to_defense(
    default_config, make_node, make_snapshot
):
    # A is healthy and depends on compromised, uncontained B. A's service drops
    # (service_availability 0.9) but its defense_availability stays 1.0, so the
    # loss is attack-induced and must not appear in healthy_disruption.
    node_a = make_node(
        "node-a", status=NodeStatus.up, defense_availability=1.0, service_availability=0.9
    )
    node_b = make_node(
        "node-b",
        compromised=True,
        status=NodeStatus.up,
        defense_availability=1.0,
        service_availability=0.9,
    )
    breakdown = compute_breakdown(
        default_config,
        criticality={"node-a": 0.7, "node-b": 0.3},
        curr=make_snapshot([node_a, node_b], step=3),
        executions=[],
        delay_steps=0,  # first compromise this step
    )
    assert breakdown.healthy_disruption == pytest.approx(0.0)
    assert breakdown.attack_impact == pytest.approx(0.3)
    assert breakdown.total == pytest.approx(-0.3)


# --------------------------------------------------------------------------- #
# R-12  collateral of a necessary containment is charged
# --------------------------------------------------------------------------- #
def test_r12_collateral_of_necessary_containment_is_charged(
    default_config, make_node, make_snapshot, make_execution
):
    # A (crit 0.7) depends on B (crit 0.3). B is compromised and quarantined this
    # step; quarantine drops davail(B) and davail(A) to 0.5. B is excluded from
    # healthy_disruption (compromised), so only A's collateral is charged.
    node_a = make_node("node-a", status=NodeStatus.up, defense_availability=0.5)
    node_b = make_node(
        "node-b",
        compromised=True,
        status=NodeStatus.quarantined,
        defense_availability=0.5,
        service_availability=0.5,
    )
    breakdown = compute_breakdown(
        default_config,
        criticality={"node-a": 0.7, "node-b": 0.3},
        curr=make_snapshot([node_a, node_b], step=3),
        executions=[make_execution(ActionKind.quarantine_node, "node-b")],
        prev_contained_ids=set(),  # B was uncontained at t-1
        delay_steps=0,
    )
    assert breakdown.containment == pytest.approx(1.0)
    assert breakdown.attack_impact == pytest.approx(0.3)
    assert breakdown.healthy_disruption == pytest.approx(0.35)  # 0.7 * (1 - 0.5)
    assert breakdown.false_isolation == pytest.approx(0.0)  # B is compromised
    assert breakdown.total == pytest.approx(2.525)


# --------------------------------------------------------------------------- #
# Supporting behaviour: containment is earned once, delay tracks uncontainment
# --------------------------------------------------------------------------- #
def test_containment_is_not_recharged_for_an_already_contained_node(
    default_config, make_node, make_snapshot, make_execution
):
    node = make_node(
        "node-db-01",
        compromised=True,
        status=NodeStatus.quarantined,
        defense_availability=0.5,
    )
    breakdown = compute_breakdown(
        default_config,
        criticality={"node-db-01": 0.5},
        curr=make_snapshot([node]),
        executions=[make_execution(ActionKind.quarantine_node, "node-db-01")],
        prev_contained_ids={"node-db-01"},  # already contained at t-1
    )
    assert breakdown.containment == pytest.approx(0.0)


def test_delay_accumulates_only_while_a_compromise_is_uncontained(
    default_config, make_node, make_snapshot, make_topology
):
    model = RewardModel(default_config)
    model.reset(make_topology({"node-aux": 0.5, "node-obj": 1.0}), max_steps=40)

    def up_snapshot(step: int):
        node = make_node("node-obj", compromised=True, status=NodeStatus.up)
        return make_snapshot([node], step=step)

    first = model.score(
        step=0,
        prev=None,
        curr=up_snapshot(0),
        executions=[],
        invalid_count=0,
        termination_reason=None,
    )
    assert first.raw["response_delay"] == pytest.approx(0.0)  # first compromise, d=0

    later = model.score(
        step=3,
        prev=up_snapshot(2),
        curr=up_snapshot(3),
        executions=[],
        invalid_count=0,
        termination_reason=None,
    )
    assert later.raw["response_delay"] == pytest.approx(3 / 20)

    contained_node = make_node(
        "node-obj",
        compromised=True,
        status=NodeStatus.quarantined,
        defense_availability=0.5,
    )
    contained = model.score(
        step=4,
        prev=up_snapshot(3),
        curr=make_snapshot([contained_node], step=4),
        executions=[],
        invalid_count=0,
        termination_reason=None,
    )
    assert contained.raw["response_delay"] == pytest.approx(0.0)  # no uncontained compromise
