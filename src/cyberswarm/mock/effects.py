"""MOCK simulator - action effects and availability (lead-owned, task 5.2).

**This is MOCK, not a real simulator.** Every function here is an abstract
state transition on in-memory :class:`NodeState` objects. Nothing shells out,
opens a socket, touches the network, or writes a file; an "action effect" is a
field assignment and nothing more.

This module owns three things:

* :class:`NodeState` - the mutable per-node state of one episode.
* the two availability views required by AD-17: :func:`service_availabilities`
  (status, dependency *and* compromise effects, compromise factor ``0.9``; feeds
  the attack-impact view and the weakened ``service_health`` channel) and
  :func:`defense_availabilities` (status and dependency effects only, compromise
  factor fixed at ``1.0``; feeds the healthy-disruption reward term and the two
  disruption metrics so attack-induced loss is never charged to the defender).
* :func:`action_precondition` - the applicability rule for all eight allowlisted
  actions, so an out-of-state action becomes a ``failed`` result that changes
  nothing (conformance SC-9), and :func:`apply_action_effect` - the P4 field
  changes of the seven *targeted* effects of the DESIGN section 5.3 table.
  ``no_op`` carries no target and has no state change, so the simulator applies
  it directly (it only emits the ``action_applied`` event).

It imports only :mod:`cyberswarm.contracts` and the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from ..contracts import ActionKind, ErrorCode, NodeStatus, RateLimitLevel

__all__ = [
    "COMPROMISE_FACTOR_SERVICE",
    "COMPROMISE_FACTOR_DEFENSE",
    "NodeState",
    "status_availability",
    "own_availability",
    "compute_availabilities",
    "service_availabilities",
    "defense_availabilities",
    "action_precondition",
    "draw_evidence_level",
    "EffectOutcome",
    "apply_action_effect",
]

# --------------------------------------------------------------------------- #
# Availability constants (DESIGN 5.3, corrected by AD-17 / AD-18)
# --------------------------------------------------------------------------- #
#: Compromise factor applied to ``service_availability`` (AD-18: was 0.7).
COMPROMISE_FACTOR_SERVICE: float = 0.9
#: Compromise factor for ``defense_availability`` - fixed at 1.0 (AD-17), so a
#: node's compromise never lowers the availability charged to the defender.
COMPROMISE_FACTOR_DEFENSE: float = 1.0

#: Base ``status_avail`` for the statuses that do not depend on the rate level.
_STATUS_AVAIL: Mapping[NodeStatus, float] = {
    NodeStatus.up: 1.0,
    NodeStatus.quarantined: 0.5,
    NodeStatus.isolated: 0.0,
    NodeStatus.restoring: 0.0,
}
#: ``status_avail`` for a rate-limited node, by level (DESIGN 5.3).
_RATE_AVAIL: Mapping[RateLimitLevel, float] = {
    RateLimitLevel.light: 0.85,
    RateLimitLevel.heavy: 0.6,
}

#: Spread transmission factor of the *sending* node, by its throttle level.
SEND_FACTOR_UP: float = 1.0
SEND_FACTOR_LIGHT: float = 0.6
SEND_FACTOR_HEAVY: float = 0.3

#: Statuses in which a node counts as contained (CONTRACTS section 9).
CONTAINMENT_STATUSES: frozenset[NodeStatus] = frozenset(
    {NodeStatus.quarantined, NodeStatus.isolated, NodeStatus.restoring}
)


# --------------------------------------------------------------------------- #
# Mutable per-node state
# --------------------------------------------------------------------------- #
@dataclass
class NodeState:
    """The mutable state of one node during one MOCK episode (DESIGN 5.3).

    ``compromised`` is hidden truth and never leaves this process except through
    the evaluator-only :class:`~cyberswarm.contracts.GroundTruthSnapshot`.
    """

    node_id: str
    status: NodeStatus = NodeStatus.up
    compromised: bool = False
    restore_left: int = 0
    rate_level: RateLimitLevel | None = None
    monitor_left: int = 0
    evidence_level: int | None = None
    evidence_step: int | None = None
    last_investigated: int | None = None

    @property
    def monitored(self) -> bool:
        """``True`` while a ``monitor_node`` effect is still counting down."""

        return self.monitor_left > 0

    def can_send(self) -> bool:
        """Whether a compromised node may spread over an outgoing link."""

        return self.status in (NodeStatus.up, NodeStatus.rate_limited)

    def can_receive(self) -> bool:
        """Whether a node may be reached by propagation (DESIGN 5.3 P0.3)."""

        return self.status not in (NodeStatus.isolated, NodeStatus.restoring)

    def send_factor(self) -> float:
        """The spread multiplier of this node as a sender."""

        if self.status is NodeStatus.rate_limited:
            return SEND_FACTOR_LIGHT if self.rate_level is RateLimitLevel.light else SEND_FACTOR_HEAVY
        return SEND_FACTOR_UP


# --------------------------------------------------------------------------- #
# Availability (AD-17)
# --------------------------------------------------------------------------- #
def status_availability(status: NodeStatus, rate_level: RateLimitLevel | None) -> float:
    """Return ``status_avail`` for a status/level pair (DESIGN 5.3)."""

    if status is NodeStatus.rate_limited:
        return _RATE_AVAIL[rate_level] if rate_level is not None else _RATE_AVAIL[RateLimitLevel.light]
    return _STATUS_AVAIL[status]


def own_availability(state: NodeState, *, compromise_factor: float) -> float:
    """Return a node's own availability before dependency effects."""

    base = status_availability(state.status, state.rate_level)
    if state.compromised:
        base *= compromise_factor
    return base


def compute_availabilities(
    node_states: Mapping[str, NodeState],
    dependencies: Mapping[str, tuple[str, ...]],
    *,
    compromise_factor: float,
) -> dict[str, float]:
    """Return ``avail(n)`` for every node with a given compromise factor.

    ``avail(n) = min(own_avail(n), min over dependencies d of avail(d))`` over
    the acyclic ``service_dependency`` graph (DESIGN 5.3). The topology contract
    guarantees the dependency graph is a DAG, so the memoised recursion
    terminates. The result is clamped to ``[0, 1]`` so it is always a valid
    ``UnitFloat``.
    """

    cache: dict[str, float] = {}

    def avail(node_id: str) -> float:
        cached = cache.get(node_id)
        if cached is not None:
            return cached
        value = own_availability(node_states[node_id], compromise_factor=compromise_factor)
        for dependency in dependencies.get(node_id, ()):  # nodes this one depends on
            value = min(value, avail(dependency))
        value = min(1.0, max(0.0, value))
        cache[node_id] = value
        return value

    return {node_id: avail(node_id) for node_id in node_states}


def service_availabilities(
    node_states: Mapping[str, NodeState],
    dependencies: Mapping[str, tuple[str, ...]],
) -> dict[str, float]:
    """Availability including the compromise penalty (factor ``0.9``, AD-18)."""

    return compute_availabilities(
        node_states, dependencies, compromise_factor=COMPROMISE_FACTOR_SERVICE
    )


def defense_availabilities(
    node_states: Mapping[str, NodeState],
    dependencies: Mapping[str, tuple[str, ...]],
) -> dict[str, float]:
    """Availability from status and dependency only (compromise factor ``1.0``).

    This is the quantity the defender is scored on (AD-17): a node's compromise
    never reduces it, so attack-induced loss is charged to attack impact, never
    to the defender.
    """

    return compute_availabilities(
        node_states, dependencies, compromise_factor=COMPROMISE_FACTOR_DEFENSE
    )


# --------------------------------------------------------------------------- #
# Action preconditions and evidence (DESIGN 5.3 P4 table)
# --------------------------------------------------------------------------- #
def action_precondition(kind: ActionKind, status: NodeStatus) -> bool:
    """Return whether ``kind`` is applicable to a node in ``status``.

    A false result means the action is a normal ``failed`` application that
    leaves the state unchanged (conformance SC-9); it is never an exception.
    """

    if kind is ActionKind.no_op:
        return True
    if kind is ActionKind.rate_limit_node:
        return status is NodeStatus.up
    if kind is ActionKind.quarantine_node:
        return status in (NodeStatus.up, NodeStatus.rate_limited)
    if kind is ActionKind.isolate_node:
        return status in (NodeStatus.up, NodeStatus.rate_limited, NodeStatus.quarantined)
    # monitor, investigate, request_evidence and restore: anything but restoring.
    return status is not NodeStatus.restoring


def draw_evidence_level(uniform: float, *, compromised: bool, strong: bool) -> int:
    """Map a uniform draw to an evidence level (DESIGN 5.3 evidence tables).

    ``strong`` selects the ``request_evidence`` distribution; otherwise the
    ``investigate_node`` distribution is used. A compromised node tends to
    higher levels; a healthy one tends to ``0``.
    """

    if strong:  # request_evidence
        if compromised:
            return 3 if uniform < 0.90 else 2
        return 0 if uniform < 0.90 else 1
    # investigate_node
    if compromised:
        if uniform < 0.60:
            return 3
        if uniform < 0.85:
            return 2
        return 1
    if uniform < 0.70:
        return 0
    if uniform < 0.90:
        return 1
    if uniform < 0.98:
        return 2
    return 3


@dataclass(frozen=True)
class EffectOutcome:
    """The result of applying one action effect to a node's state."""

    applied: bool
    error_code: ErrorCode | None
    status_from: NodeStatus | None
    status_to: NodeStatus | None
    status_changed: bool
    investigation_evidence: int | None
    affected: tuple[str, ...]


def apply_action_effect(
    state: NodeState,
    kind: ActionKind,
    *,
    step: int,
    duration_steps: int | None,
    level: RateLimitLevel | None,
    restore_duration_steps: int,
    evidence_uniform: float,
) -> EffectOutcome:
    """Apply one targeted action's effect to ``state`` (DESIGN 5.3 P4 table).

    ``kind`` must be a targeted action (``no_op`` carries no target and is
    handled by the simulator). If the precondition fails, the state is left
    untouched and a ``failed`` outcome is returned. Otherwise the field changes
    of the P4 table are applied in place and an ``executed`` outcome describing
    the visible status change (if any) is returned.
    """

    if not action_precondition(kind, state.status):
        return EffectOutcome(
            applied=False,
            error_code=ErrorCode.ACTION_EXECUTION_FAILED,
            status_from=None,
            status_to=None,
            status_changed=False,
            investigation_evidence=None,
            affected=(),
        )

    status_from = state.status
    investigation_evidence: int | None = None

    if kind is ActionKind.monitor_node:
        state.monitor_left = duration_steps if duration_steps is not None else 3
    elif kind is ActionKind.investigate_node:
        investigation_evidence = draw_evidence_level(
            evidence_uniform, compromised=state.compromised, strong=False
        )
        state.evidence_level = investigation_evidence
        state.evidence_step = step
        state.last_investigated = step
    elif kind is ActionKind.request_evidence:
        investigation_evidence = draw_evidence_level(
            evidence_uniform, compromised=state.compromised, strong=True
        )
        state.evidence_level = investigation_evidence
        state.evidence_step = step
        state.last_investigated = step
    elif kind is ActionKind.rate_limit_node:
        state.status = NodeStatus.rate_limited
        state.rate_level = level if level is not None else RateLimitLevel.light
    elif kind is ActionKind.quarantine_node:
        state.status = NodeStatus.quarantined
        state.rate_level = None
    elif kind is ActionKind.isolate_node:
        state.status = NodeStatus.isolated
        state.rate_level = None
    elif kind is ActionKind.restore_node:
        state.status = NodeStatus.restoring
        state.restore_left = restore_duration_steps
        state.monitor_left = 0
    else:  # pragma: no cover - defensive; no_op is handled by the simulator
        raise ValueError(f"apply_action_effect does not handle {kind!r}")

    status_to = state.status
    return EffectOutcome(
        applied=True,
        error_code=None,
        status_from=status_from,
        status_to=status_to,
        status_changed=status_from != status_to,
        investigation_evidence=investigation_evidence,
        affected=(state.node_id,),
    )
