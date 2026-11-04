"""The single, closed, simulator-only action vocabulary (CONTRACTS section 6).

This is the only action vocabulary. The coordinator allowlist, the RL action
table, the baselines, the dashboard and the simulator all use it. Every action
is an abstract simulator state transition: nothing here can carry a shell
string, a command, an address or a payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Mapping

from .common import ImpactClass

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from .actions import ActionParameters


class ActionKind(StrEnum):
    """The eight abstract defensive actions (CONTRACTS section 6)."""

    no_op = "no_op"
    monitor_node = "monitor_node"
    investigate_node = "investigate_node"
    request_evidence = "request_evidence"
    rate_limit_node = "rate_limit_node"
    quarantine_node = "quarantine_node"
    isolate_node = "isolate_node"
    restore_node = "restore_node"


@dataclass(frozen=True, slots=True)
class ActionDef:
    """Static metadata for one action kind (one row of the section 6 table)."""

    needs_target: bool
    allowed_parameters: tuple[str, ...]
    impact_class: ImpactClass
    high_impact: bool
    cost_units: float


#: Fixed order used by action tables and masks (the row order of the table,
#: then ascending ``node_id`` at the call site).
ACTION_ORDER: Final[tuple[ActionKind, ...]] = (
    ActionKind.no_op,
    ActionKind.monitor_node,
    ActionKind.investigate_node,
    ActionKind.request_evidence,
    ActionKind.rate_limit_node,
    ActionKind.quarantine_node,
    ActionKind.isolate_node,
    ActionKind.restore_node,
)


ACTION_DEFS: Final[Mapping[ActionKind, ActionDef]] = {
    ActionKind.no_op: ActionDef(
        needs_target=False,
        allowed_parameters=(),
        impact_class=ImpactClass.none,
        high_impact=False,
        cost_units=0.0,
    ),
    ActionKind.monitor_node: ActionDef(
        needs_target=True,
        allowed_parameters=("duration_steps",),
        impact_class=ImpactClass.low,
        high_impact=False,
        cost_units=0.5,
    ),
    ActionKind.investigate_node: ActionDef(
        needs_target=True,
        allowed_parameters=(),
        impact_class=ImpactClass.low,
        high_impact=False,
        cost_units=1.0,
    ),
    ActionKind.request_evidence: ActionDef(
        needs_target=True,
        allowed_parameters=(),
        impact_class=ImpactClass.low,
        high_impact=False,
        cost_units=3.0,
    ),
    ActionKind.rate_limit_node: ActionDef(
        needs_target=True,
        allowed_parameters=("level",),
        impact_class=ImpactClass.medium,
        high_impact=False,
        cost_units=0.0,
    ),
    ActionKind.quarantine_node: ActionDef(
        needs_target=True,
        allowed_parameters=(),
        impact_class=ImpactClass.medium,
        high_impact=False,
        cost_units=0.0,
    ),
    ActionKind.isolate_node: ActionDef(
        needs_target=True,
        allowed_parameters=(),
        impact_class=ImpactClass.high,
        high_impact=True,
        cost_units=0.0,
    ),
    ActionKind.restore_node: ActionDef(
        needs_target=True,
        allowed_parameters=(),
        impact_class=ImpactClass.high,
        high_impact=True,
        cost_units=0.0,
    ),
}


#: High-impact actions (``impact_class = high``).
HIGH_IMPACT: Final[frozenset[ActionKind]] = frozenset(
    kind for kind, spec in ACTION_DEFS.items() if spec.high_impact
)

#: Containment-class actions, used by the false-isolation reward term and metric.
CONTAINMENT_CLASS: Final[frozenset[ActionKind]] = frozenset(
    {ActionKind.quarantine_node, ActionKind.isolate_node, ActionKind.restore_node}
)

#: Information-gathering actions, used by the investigation-cost reward term.
INFORMATION_GATHERING: Final[frozenset[ActionKind]] = frozenset(
    {ActionKind.monitor_node, ActionKind.investigate_node, ActionKind.request_evidence}
)


def default_parameters(kind: ActionKind) -> "ActionParameters":
    """Return the default parameters for an action kind (CONTRACTS section 6).

    ``monitor_node`` defaults to ``duration_steps = 3``; ``rate_limit_node``
    defaults to ``level = light``; every other kind has empty parameters. The
    import is local to avoid a cycle with :mod:`cyberswarm.contracts.actions`,
    which imports this module.
    """

    from .actions import ActionParameters, RateLimitLevel

    if kind is ActionKind.monitor_node:
        return ActionParameters(duration_steps=3, level=None)
    if kind is ActionKind.rate_limit_node:
        return ActionParameters(duration_steps=None, level=RateLimitLevel.light)
    return ActionParameters(duration_steps=None, level=None)
