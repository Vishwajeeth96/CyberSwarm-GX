"""Configurable team reward model (docs/RL_IMPLEMENTATION.md section 4; stage S3).

The reward is a training and evaluation signal computed after P4 of a step from
the end-of-step ground-truth snapshots of steps ``t-1`` and ``t`` and the actions
executed in step ``t``. It is one scalar per step, shared by all agents
(``sharing = "team"``), and is never an observation.

:class:`RewardModel` implements the
:class:`~cyberswarm.interfaces.ports.StepScorer` protocol. Every term is
``s_k * w_k * m_k`` with a fixed sign ``s_k``, a non-negative configurable weight
``w_k`` and a non-negative raw magnitude ``m_k``. The model returns the seven
*unweighted* magnitudes alongside the weighted total, because the reward is
linear in the weights: ``evaluation.reweight`` can recompute an episode's reward
for any weight vector from the stored magnitudes without re-running the episode
(RL_IMPLEMENTATION 4.4).

Healthy-service disruption is charged on
``GroundTruthSnapshot.defense_availability`` (the compromise factor fixed at
1.0), so availability lost purely to an uncontained attack is scored under
``attack_impact`` only and never billed to the defender (DECISIONS AD-17).
Collateral loss from a containment or restore the defender chose stays in the
term, because ``defense_availability`` reflects it.

Imports are limited to :mod:`cyberswarm.contracts` (including the evaluator-only
ground-truth records), :mod:`cyberswarm.interfaces` and :mod:`cyberswarm.common`;
this module imports no simulator, environment or policy.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import Field

from cyberswarm.common.config import ConfigModel, load_toml
from cyberswarm.contracts.actions import ExecutedAction
from cyberswarm.contracts.common import ExecutionStatus, NodeStatus, ShortText
from cyberswarm.contracts.evaluator_only import GroundTruthSnapshot
from cyberswarm.contracts.results import TerminationReason
from cyberswarm.contracts.topology import Topology
from cyberswarm.contracts.vocabulary import (
    ACTION_DEFS,
    CONTAINMENT_CLASS,
    INFORMATION_GATHERING,
)
from cyberswarm.interfaces.ports import StepScore

__all__ = [
    "TERM_NAMES",
    "TERM_SIGNS",
    "CONTAINED_STATUSES",
    "RewardWeights",
    "RewardConfig",
    "RewardBreakdown",
    "RewardModel",
    "compute_breakdown",
    "signed_weighted",
    "weighted_total",
]


#: The seven reward terms in a fixed order (matches ``RewardTermTotals``).
TERM_NAMES: Final[tuple[str, ...]] = (
    "containment",
    "response_delay",
    "attack_impact",
    "false_isolation",
    "healthy_disruption",
    "investigation_cost",
    "invalid_action",
)

#: The fixed sign of each term: containment is the only positive term; the five
#: penalties and the invalid-action guard are negative (RL_IMPLEMENTATION 4.1).
TERM_SIGNS: Final[Mapping[str, float]] = {
    "containment": 1.0,
    "response_delay": -1.0,
    "attack_impact": -1.0,
    "false_isolation": -1.0,
    "healthy_disruption": -1.0,
    "investigation_cost": -1.0,
    "invalid_action": -1.0,
}

#: A node counts as contained when its status is one of these (the ``K_t`` set).
CONTAINED_STATUSES: Final[frozenset[NodeStatus]] = frozenset(
    {NodeStatus.quarantined, NodeStatus.isolated, NodeStatus.restoring}
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class RewardWeights(ConfigModel):
    """The seven non-negative reward weights (RL_IMPLEMENTATION 4.1).

    Field names match :data:`TERM_NAMES`. A negative weight is rejected, because
    a term's sign is fixed in the model and must not be flipped by a weight.
    """

    attack_impact: float = Field(ge=0.0)
    response_delay: float = Field(ge=0.0)
    containment: float = Field(ge=0.0)
    false_isolation: float = Field(ge=0.0)
    healthy_disruption: float = Field(ge=0.0)
    investigation_cost: float = Field(ge=0.0)
    invalid_action: float = Field(ge=0.0)


class RewardConfig(ConfigModel):
    """Reward weights and the two shape parameters, loaded from TOML.

    ``weights_id`` is recorded in every ``EpisodeResult`` so a reward value is
    always traceable to the weights that produced it. ``d_max`` (the delay cap)
    and ``false_isolation_base`` are not linear weights: changing them needs a
    new run, so they are not part of post-hoc reweighting.
    """

    weights_id: ShortText
    sharing: Literal["team", "zone_local"] = "team"
    d_max: int = Field(gt=0)
    false_isolation_base: float = Field(ge=0.0)
    weights: RewardWeights

    @classmethod
    def from_toml(cls, path: Path | str) -> "RewardConfig":
        """Read and validate a reward-weights TOML file.

        A malformed file or an unknown key raises
        :class:`~cyberswarm.contracts.common.ContractError` with code
        ``CONFIG_INVALID`` (via :func:`cyberswarm.common.config.load_toml`).
        """

        return load_toml(path, cls)


# --------------------------------------------------------------------------- #
# Per-step breakdown
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RewardBreakdown:
    """The seven unweighted per-term magnitudes and the signed weighted total.

    Each magnitude is ``>= 0``; the sign lives in :data:`TERM_SIGNS`. ``total``
    is ``sum_k TERM_SIGNS[k] * w_k * m_k`` under the weights that produced it.
    """

    containment: float
    response_delay: float
    attack_impact: float
    false_isolation: float
    healthy_disruption: float
    investigation_cost: float
    invalid_action: float
    total: float

    def raw_magnitudes(self) -> dict[str, float]:
        """The seven unweighted magnitudes keyed by term name."""

        return {name: float(getattr(self, name)) for name in TERM_NAMES}


def signed_weighted(raw: Mapping[str, float], config: RewardConfig) -> dict[str, float]:
    """Return the signed weighted contribution ``s_k * w_k * m_k`` of each term."""

    weights = config.weights
    return {
        name: TERM_SIGNS[name] * float(getattr(weights, name)) * float(raw[name])
        for name in TERM_NAMES
    }


def weighted_total(raw: Mapping[str, float], config: RewardConfig) -> float:
    """Recompute the scalar reward from stored magnitudes (linear in weights).

    This is the identity ``evaluation.reweight`` relies on: for any weight
    vector, the reward is the signed weighted sum of the stored raw magnitudes.
    """

    return float(sum(signed_weighted(raw, config).values()))


def _criticality_of(node_id: str, criticality: Mapping[str, float]) -> float:
    try:
        return float(criticality[node_id])
    except KeyError:
        raise KeyError(
            f"no criticality for node {node_id!r}; reset() must see every node"
        ) from None


def compute_breakdown(
    config: RewardConfig,
    *,
    criticality: Mapping[str, float],
    curr: GroundTruthSnapshot,
    executions: Sequence[ExecutedAction] = (),
    invalid_count: int = 0,
    prev_contained_ids: Collection[str] = (),
    delay_steps: int = 0,
    terminal_factor: int = 0,
) -> RewardBreakdown:
    """Compute one step's reward from fully specified inputs.

    This is the pure core of the model. ``delay_steps`` is ``d_t`` (0 when no
    compromise is uncontained) and ``terminal_factor`` is ``H - 1 - t`` when the
    episode ends this step with ``objective_reached`` (0 otherwise);
    :class:`RewardModel` derives both from the snapshot stream and the horizon.
    ``prev_contained_ids`` is ``K_{t-1}``, the set of nodes contained at the end
    of the previous step.
    """

    compromised = {node.node_id for node in curr.nodes if node.compromised}
    prev_contained = set(prev_contained_ids)

    # attack_impact: criticality mass of every compromised node, plus the
    # remaining-horizon charge when the objective falls this step.
    base_impact = sum(_criticality_of(nid, criticality) for nid in compromised)
    attack_impact = base_impact * (1.0 + float(max(terminal_factor, 0)))

    # response_delay: fraction of the cap, 0 unless a compromise is uncontained.
    capped = min(max(delay_steps, 0), config.d_max)
    response_delay = capped / config.d_max

    containment_nodes: set[str] = set()
    false_isolation = 0.0
    investigation_cost = 0.0
    for execution in executions:
        if execution.status is not ExecutionStatus.executed:
            continue  # a failed execution changed nothing and costs nothing
        kind = execution.action.action_kind
        target = execution.action.target_node_id
        if kind in INFORMATION_GATHERING:
            investigation_cost += ACTION_DEFS[kind].cost_units
        if kind in CONTAINMENT_CLASS and target is not None:
            if target in compromised:
                if target not in prev_contained:
                    containment_nodes.add(target)  # earned once, on entry
            else:
                false_isolation += config.false_isolation_base + _criticality_of(
                    target, criticality
                )

    # healthy_disruption: defender-induced availability loss on healthy nodes.
    # defense_availability excludes the compromise penalty, so attack-induced
    # loss contributes nothing here (AD-17).
    healthy_disruption = 0.0
    for node in curr.nodes:
        if node.node_id in compromised:
            continue
        healthy_disruption += _criticality_of(node.node_id, criticality) * (
            1.0 - node.defense_availability
        )

    raw: dict[str, float] = {
        "containment": float(len(containment_nodes)),
        "response_delay": response_delay,
        "attack_impact": attack_impact,
        "false_isolation": false_isolation,
        "healthy_disruption": healthy_disruption,
        "investigation_cost": investigation_cost,
        "invalid_action": float(max(invalid_count, 0)),
    }
    return RewardBreakdown(total=weighted_total(raw, config), **raw)


def _contained_ids(snapshot: GroundTruthSnapshot | None) -> set[str]:
    if snapshot is None:
        return set()
    return {
        node.node_id for node in snapshot.nodes if node.status in CONTAINED_STATUSES
    }


# --------------------------------------------------------------------------- #
# The scorer
# --------------------------------------------------------------------------- #
class RewardModel:
    """Configurable team reward model implementing :class:`StepScorer`.

    ``reset`` captures the per-node criticalities and the horizon from the
    topology; ``score`` is called once per closed step with the previous and
    current end-of-step ground-truth snapshots, the step's executions, the count
    of invalid (substituted) actions and the termination reason when the episode
    ended on this step. The model tracks the first-compromise step so that the
    response-delay term grows while an incident stays uncontained.
    """

    def __init__(self, config: RewardConfig) -> None:
        self._config = config
        self._criticality: dict[str, float] = {}
        self._max_steps = 0
        self._first_compromise_step: int | None = None

    @property
    def config(self) -> RewardConfig:
        return self._config

    def reset(self, topology: Topology, max_steps: int) -> None:
        self._criticality = {
            node.node_id: float(node.criticality) for node in topology.nodes
        }
        self._max_steps = int(max_steps)
        self._first_compromise_step = None

    def score(
        self,
        *,
        step: int,
        prev: GroundTruthSnapshot | None,
        curr: GroundTruthSnapshot,
        executions: Sequence[ExecutedAction],
        invalid_count: int,
        termination_reason: TerminationReason | None,
    ) -> StepScore:
        if (
            self._first_compromise_step is None
            and any(node.compromised for node in curr.nodes)
        ):
            self._first_compromise_step = step

        has_uncontained = any(
            node.compromised and node.status not in CONTAINED_STATUSES
            for node in curr.nodes
        )
        if has_uncontained and self._first_compromise_step is not None:
            delay_steps = step - self._first_compromise_step
        else:
            delay_steps = 0

        if termination_reason is TerminationReason.objective_reached:
            terminal_factor = max(self._max_steps - 1 - step, 0)
        else:
            terminal_factor = 0

        breakdown = compute_breakdown(
            self._config,
            criticality=self._criticality,
            curr=curr,
            executions=executions,
            invalid_count=invalid_count,
            prev_contained_ids=_contained_ids(prev),
            delay_steps=delay_steps,
            terminal_factor=terminal_factor,
        )
        raw = breakdown.raw_magnitudes()
        return StepScore(
            total=breakdown.total,
            raw=raw,
            weighted=signed_weighted(raw, self._config),
        )
