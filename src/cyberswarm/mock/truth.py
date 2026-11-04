"""MOCK simulator - evaluator-only ground truth (lead-owned, task 5.2).

**This is MOCK, not a real simulator.** This module builds the hidden
:class:`~cyberswarm.contracts.GroundTruthSnapshot`. It is reached only through
the separate :class:`~cyberswarm.interfaces.GroundTruthAccessor`, never through
an observation, so decision-time code cannot read it.

Timing (AD-16, CONTRACTS section 5 rule 9): the simulator calls
:func:`capture_snapshot` inside ``end_step()`` of step ``t``, after P4 and
*before* P0 of step ``t+1``. The snapshot therefore describes the end-of-step-
``t`` state, with ``step == StepResult.step_closed``; a ``restore_completed`` or
new compromise that happens in P0 of ``t+1`` is not yet reflected.

Each node carries both availabilities of AD-17: ``service_availability`` (with
the ``0.9`` compromise factor, for attack impact) and ``defense_availability``
(status and dependency only, compromise factor ``1.0``, for the defender's
disruption charge).

It imports only :mod:`cyberswarm.contracts`, the sibling :mod:`.effects` and the
standard library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..contracts import (
    AlertTruth,
    CompromiseCause,
    GroundTruthNode,
    GroundTruthSnapshot,
    NewCompromise,
)
from .effects import (
    CONTAINMENT_STATUSES,
    NodeState,
    defense_availabilities,
    service_availabilities,
)

__all__ = ["capture_snapshot"]


def capture_snapshot(
    *,
    episode_id: str,
    step: int,
    node_order: Sequence[str],
    node_states: Mapping[str, NodeState],
    dependencies: Mapping[str, tuple[str, ...]],
    incident_active: bool,
    new_compromises: Sequence[tuple[str, CompromiseCause]],
    alert_truth: Sequence[tuple[str, bool]],
) -> GroundTruthSnapshot:
    """Build the end-of-step-``t`` :class:`GroundTruthSnapshot` (AD-16, AD-17).

    ``new_compromises`` and ``alert_truth`` are the compromises and
    ``alert_raised`` events produced during this step's P0; the simulator
    collects them while P0 runs and hands them over here unchanged.
    """

    service = service_availabilities(node_states, dependencies)
    defense = defense_availabilities(node_states, dependencies)

    nodes = [
        GroundTruthNode(
            node_id=node_id,
            compromised=node_states[node_id].compromised,
            service_availability=service[node_id],
            defense_availability=defense[node_id],
            contained=node_states[node_id].status in CONTAINMENT_STATUSES,
            status=node_states[node_id].status,
        )
        for node_id in node_order
    ]

    return GroundTruthSnapshot(
        episode_id=episode_id,
        step=step,
        incident_active=incident_active,
        nodes=nodes,
        new_compromises=[
            NewCompromise(node_id=node_id, cause=cause)
            for node_id, cause in new_compromises
        ],
        alert_truth=[
            AlertTruth(event_id=event_id, true_positive=true_positive)
            for event_id, true_positive in alert_truth
        ],
    )
