"""MOCK ``service_health`` AUROC guard (AD-18, design Property 8).

AD-18 deliberately weakens the ``service_health`` channel (compromised
availability factor ``0.9``, noise ``0.08``, edge offset ``0.05``) so it is
informative about compromise but not an oracle. This guard runs a random-policy
rollout, pairs each own-node ``service_health`` reading with the end-of-step
ground-truth ``compromised`` label, and checks that the best threshold on
``service_health`` separates the two classes with an AUROC below the agreed
bound of ``0.85`` - yet clearly above chance. If this ever rises toward ``1.0``,
a learned policy could read compromise off a single input and the IPPO-vs-rule
comparison would partly measure access to an oracle-like channel.

The rollout is fully deterministic: fixed scenarios, fixed episode seeds and a
per-episode PCG64 action stream, so the measured AUROC is a fixed number.
"""

from __future__ import annotations

import numpy as np

from cyberswarm.contracts import ActionKind

# Fixed rollout configuration -> a deterministic AUROC.
_SCENARIOS = (
    "scn-mock-lateral-eval-01",
    "scn-mock-lateral-eval-02",
    "scn-mock-lateral-train-01",
    "scn-mock-lateral-train-02",
)
_SEEDS = (3001, 3002, 3003, 3004, 3005)

#: AD-18 agreed upper bound: the channel must not be an oracle.
_AUROC_MAX = 0.85
#: The channel must still carry real signal (clearly above chance).
_AUROC_MIN = 0.55


def _auroc(scores: list[float], labels: list[bool]) -> float | None:
    """AUROC via the Mann-Whitney statistic with average ranks for ties.

    ``scores`` are oriented so a higher score means a more likely positive. The
    result is ``P(score(positive) > score(negative))`` with ties counted as a
    half. Returns ``None`` if either class is empty.
    """

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0  # 1-based average over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1

    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    rank_sum_positive = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def _random_rollout(new_simulator, load_scenario, validated_action, episode_id_for):
    """Collect (service_health, compromised) over a random-policy rollout."""

    health: list[float] = []
    compromised: list[bool] = []
    for scenario_id in _SCENARIOS:
        scenario = load_scenario(scenario_id)
        for seed in _SEEDS:
            sim, gt = new_simulator()
            episode_id = episode_id_for(scenario_id, seed, "pol-random")
            sim.reset(scenario, seed, episode_id)
            action_rng = np.random.Generator(np.random.PCG64(seed + 777))

            guard = 0
            while not sim.done and guard < scenario.max_steps + 2:
                guard += 1
                observations = sim.observe()
                # service_health is only visible for a node the agent owns.
                health_by_node: dict[str, float] = {}
                for observation in observations.values():
                    for node in observation.nodes:
                        if node.scope is node.scope.own and node.service_health is not None:
                            health_by_node[node.node_id] = node.service_health

                for counter, agent_id in enumerate(sim.agent_ids, start=1):
                    available = sim.available_actions(agent_id)
                    index = int(action_rng.integers(0, len(available.actions)))
                    choice = available.actions[index]
                    sim.apply_action(
                        validated_action(
                            episode_id,
                            sim.step,
                            agent_id,
                            choice.action_kind,
                            choice.target_node_id,
                            counter=counter,
                        )
                    )
                sim.end_step()
                snapshot = gt.ground_truth()
                compromised_by_node = {n.node_id: n.compromised for n in snapshot.nodes}
                for node_id, value in health_by_node.items():
                    health.append(value)
                    compromised.append(compromised_by_node[node_id])
            sim.close()
    return health, compromised


def test_service_health_is_informative_but_not_an_oracle(
    new_simulator, load_scenario, validated_action, episode_id_for
) -> None:
    health, compromised = _random_rollout(
        new_simulator, load_scenario, validated_action, episode_id_for
    )

    # A meaningful rollout with both classes present.
    assert len(health) >= 1000
    positives = sum(compromised)
    assert 0 < positives < len(compromised), "need both compromised and healthy samples"

    # service_health falls when a node is compromised, so the predictive
    # orientation is low-health -> compromised; the raw orientation is below
    # chance, and the best orientation is its mirror.
    auc_raw = _auroc(health, compromised)
    auc_low_health = _auroc([1.0 - value for value in health], compromised)
    assert auc_raw is not None and auc_low_health is not None
    assert auc_raw < 0.5, "higher service_health should indicate a healthy node"
    best = max(auc_raw, auc_low_health)

    assert best <= _AUROC_MAX, (
        f"service_health AUROC {best:.3f} exceeds the AD-18 bound {_AUROC_MAX}: "
        "the channel is too close to an oracle for compromise"
    )
    assert best >= _AUROC_MIN, (
        f"service_health AUROC {best:.3f} is below {_AUROC_MIN}: the channel "
        "carries too little signal to be the intended weak proxy"
    )


def test_rollout_exercises_the_weak_channel_constants() -> None:
    # Guard the specific AD-18 constants the AUROC bound depends on, so a change
    # to the channel strength is a conscious, reviewed edit.
    from cyberswarm.mock import observation as obs_mod
    from cyberswarm.mock import effects as eff_mod

    assert eff_mod.COMPROMISE_FACTOR_SERVICE == 0.9
    assert eff_mod.COMPROMISE_FACTOR_DEFENSE == 1.0
    assert obs_mod.SERVICE_HEALTH_NOISE == 0.08
    assert obs_mod.EDGE_TRAFFIC_NOISE == 0.08
    assert obs_mod.EDGE_COMPROMISE_OFFSET == 0.05
