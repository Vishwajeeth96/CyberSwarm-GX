"""MOCK simulator - exogenous P0 dynamics (lead-owned, task 5.2).

**This is MOCK, not a real simulator.** The "attacker" here is a seeded random
process over an in-memory graph: no packets, no hosts, no addresses. There is
one incident type (propagating compromise) and no learning adversary.

Determinism uses **common random numbers** (DESIGN 5.3). At ``reset`` every
random draw an episode will ever need is pre-computed into fixed tables indexed
by ``(step, node)`` or ``(step, edge)``. The tables depend only on
``(seed, H, N, E)`` and never on the agents' actions, so two policies run on the
same ``(scenario, seed)`` face identical draws, which makes matched comparisons
far less noisy and makes the whole simulator reproducible for a fixed seed.

:func:`run_p0` runs phase P0 of one step ``t >= 1`` in the fixed order of DESIGN
5.3: restore completion, incident start, propagation, alert generation, then
event emission (restore-completed events by node, then due alerts by node). It
mutates the node states and the pending-alert queue in place and returns a
:class:`P0Result` the simulator turns into event records.

It imports only :mod:`cyberswarm.contracts`, the siblings :mod:`.effects` and
:mod:`.observation`, NumPy and the standard library.
"""

from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence
from dataclasses import dataclass, field

import numpy as np

from ..contracts import (
    AlertCategory,
    CompromiseCause,
    NodeStatus,
    ScenarioConfig,
    Severity,
)
from .effects import NodeState
from .observation import TopologyIndex

__all__ = ["CommonRandomTables", "PendingAlert", "P0Result", "run_p0"]

#: The three abstract alert categories, indexed by the ``category`` table.
_CATEGORY_ORDER = (
    AlertCategory.traffic_anomaly,
    AlertCategory.access_anomaly,
    AlertCategory.service_anomaly,
)


class CommonRandomTables:
    """Pre-drawn common-random-number tables for one episode (DESIGN 5.3).

    All tables are drawn once, in a fixed order, from a single
    ``numpy.random.Generator(PCG64(seed))``. Because the draw order and shapes
    depend only on ``(seed, H, N, E)``, the tables - and therefore the whole
    episode - are reproducible for a fixed seed and differ for a different seed.
    """

    def __init__(self, seed: int, horizon: int, node_count: int, edge_count: int) -> None:
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        self.spread = rng.random((horizon, edge_count))
        self.alert = rng.random((horizon, node_count))
        self.delay = rng.random((horizon, node_count))
        self.severity = rng.random((horizon, node_count))
        self.category = rng.random((horizon, node_count))
        self.false_alert = rng.random((horizon, node_count))
        # ``noise`` is a standard normal truncated at +/-3 (AD-18).
        self.health_noise = np.clip(rng.standard_normal((horizon, node_count)), -3.0, 3.0)
        self.traffic_noise = np.clip(rng.standard_normal((horizon, edge_count)), -3.0, 3.0)
        self.missing_node = rng.random((horizon, node_count))
        self.missing_edge = rng.random((horizon, edge_count))
        self.evidence = rng.random((horizon, node_count, 2))


@dataclass(frozen=True, slots=True)
class PendingAlert:
    """An alert generated at one step, due to be emitted at ``emit_step``.

    ``true_positive`` is fixed at generation time (DESIGN 5.3): a compromised
    node raises a true alert, a healthy node a false one.
    """

    emit_step: int
    node_id: str
    severity: Severity
    alert_category: AlertCategory
    true_positive: bool


@dataclass(frozen=True, slots=True)
class DueAlert:
    """One alert that becomes visible at the current step."""

    node_id: str
    severity: Severity
    alert_category: AlertCategory
    true_positive: bool


@dataclass
class P0Result:
    """What phase P0 produced, for the simulator to turn into events."""

    restores_completed: list[str] = field(default_factory=list)
    due_alerts: list[DueAlert] = field(default_factory=list)
    new_compromises: list[tuple[str, CompromiseCause]] = field(default_factory=list)


def run_p0(
    node_states: MutableMapping[str, NodeState],
    pending_alerts: MutableSequence[PendingAlert],
    tables: CommonRandomTables,
    scenario: ScenarioConfig,
    index: TopologyIndex,
    step: int,
) -> P0Result:
    """Run phase P0 for step ``t = step`` (``>= 1``) in the DESIGN 5.3 order.

    Mutates ``node_states`` and ``pending_alerts`` in place and returns the
    restore completions, due alerts (ascending ``node_id``) and new compromises
    the simulator records. Event-id assignment and ``visible_to`` are the
    simulator's job; this function is pure dynamics.
    """

    result = P0Result()
    incident = scenario.incident
    observability = scenario.observability
    horizon = scenario.max_steps

    # 1. Restore completion (ascending node id).
    for node_id in index.node_order:
        state = node_states[node_id]
        if state.restore_left > 0:
            state.restore_left -= 1
            if state.restore_left == 0:
                state.status = NodeStatus.up
                state.compromised = False
                state.rate_level = None
                result.restores_completed.append(node_id)

    # 2. Incident start: the entry node is compromised at the start step, unless
    #    it happens to be restoring.
    if step == incident.start_step:
        entry = node_states[incident.entry_node_id]
        if entry.status is not NodeStatus.restoring and not entry.compromised:
            entry.compromised = True
            result.new_compromises.append(
                (incident.entry_node_id, CompromiseCause.initial_entry)
            )
    # 3. Propagation on steps after the start step. A node compromised during
    #    this phase does not spread in the same phase (we iterate a snapshot).
    elif step > incident.start_step:
        compromised_now = [n for n in index.node_order if node_states[n].compromised]
        for source_id in compromised_now:
            source = node_states[source_id]
            if not source.can_send():
                continue
            factor = source.send_factor()
            threshold = incident.spread_probability * factor
            for edge_id, target_id in index.link_out(source_id):
                target = node_states[target_id]
                if target.compromised or not target.can_receive():
                    continue
                if tables.spread[step, index.edge_index[edge_id]] < threshold:
                    target.compromised = True
                    result.new_compromises.append((target_id, CompromiseCause.propagation))

    # 4. Alert generation. Severity/category/delay/truth are fixed here; the
    #    alert is only emitted once its (possibly delayed) due step arrives.
    for node_id in index.node_order:
        state = node_states[node_id]
        node_i = index.node_index[node_id]
        monitored = state.monitored
        if state.compromised:
            probability = min(
                1.0, observability.alert_detect_probability * (1.5 if monitored else 1.0)
            )
            raised = tables.alert[step, node_i] < probability
            true_positive = True
            severity = (
                Severity.high if tables.severity[step, node_i] >= 0.6 else Severity.medium
            )
        else:
            probability = observability.false_alert_rate * (0.5 if monitored else 1.0)
            raised = tables.false_alert[step, node_i] < probability
            true_positive = False
            severity = (
                Severity.medium if tables.severity[step, node_i] >= 0.7 else Severity.low
            )
        if not raised:
            continue
        category = _CATEGORY_ORDER[min(2, int(tables.category[step, node_i] * 3))]
        delay = int(tables.delay[step, node_i] * (observability.alert_delay_max_steps + 1))
        emit_step = step + delay
        if emit_step < horizon:
            pending_alerts.append(
                PendingAlert(
                    emit_step=emit_step,
                    node_id=node_id,
                    severity=severity,
                    alert_category=category,
                    true_positive=true_positive,
                )
            )

    # 5. Collect alerts due this step (ascending node id) and drop them from the
    #    queue. The simulator assigns their event ids after the restore events.
    due = [alert for alert in pending_alerts if alert.emit_step == step]
    due.sort(key=lambda alert: alert.node_id)
    result.due_alerts = [
        DueAlert(
            node_id=alert.node_id,
            severity=alert.severity,
            alert_category=alert.alert_category,
            true_positive=alert.true_positive,
        )
        for alert in due
    ]
    pending_alerts[:] = [alert for alert in pending_alerts if alert.emit_step != step]

    # 6. Monitoring counts down at the end of each P0 (DESIGN 5.3).
    for node_id in index.node_order:
        state = node_states[node_id]
        if state.monitor_left > 0:
            state.monitor_left -= 1

    return result
