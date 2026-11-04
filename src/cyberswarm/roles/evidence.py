"""Investigation-support evidence role (DESIGN section 4.4, ARCHITECTURE sec. 6).

:func:`evidence_view` is a pure function over an
:class:`~cyberswarm.contracts.AgentObservation`. It reads only agent-visible
fields and turns the per-node evidence measurements of the agent's *own* nodes
into an :class:`~cyberswarm.interfaces.EvidenceView`.

A node is flagged ``insufficient`` (that is, investigation is advisable) when it
is alerted and its evidence is absent (``evidence_level is None``) or below
``e_min``. This is the single signal the rule-based baselines and the
uncertainty rule use to decide whether to gather more evidence, so ``e_min``
lives in one place, :class:`EvidenceConfig`.

Boundary nodes carry no visible private evidence (``evidence_level``,
``service_health`` and ``steps_since_investigated`` are withheld as ``None``)
and lie outside the agent's authority, so they are not part of the view.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import AgentObservation, NodeScope
from ..interfaces import EvidenceView, NodeEvidence

#: Default minimum evidence level that counts as sufficient (DESIGN sections
#: 4.4, 4.7 and 7). Evidence levels range over ``0..3``.
DEFAULT_E_MIN = 2

_EVIDENCE_LEVEL_MAX = 3


def _check_e_min(e_min: int) -> None:
    if not 0 <= e_min <= _EVIDENCE_LEVEL_MAX:
        raise ValueError(f"e_min must be in 0..{_EVIDENCE_LEVEL_MAX}, got {e_min}")


@dataclass(frozen=True)
class EvidenceConfig:
    """The shared evidence threshold.

    A single home for ``e_min`` (the minimum evidence level that counts as
    sufficient) so the rule-based baselines and the uncertainty rule agree on
    what "insufficient evidence" means (DESIGN section 4.4). ``e_min`` ranges
    over the evidence-level domain ``0..3``.
    """

    e_min: int = DEFAULT_E_MIN

    def __post_init__(self) -> None:
        _check_e_min(self.e_min)


def evidence_view(obs: AgentObservation, *, e_min: int = DEFAULT_E_MIN) -> EvidenceView:
    """Build the per-own-node evidence view for one agent (pure function).

    ``e_min`` defaults to :data:`DEFAULT_E_MIN`; pass ``EvidenceConfig.e_min`` to
    share one threshold across components. The returned mapping follows the
    observation's ascending node order and contains only ``own``-scope nodes.
    """

    _check_e_min(e_min)
    nodes: dict[str, NodeEvidence] = {}
    for node in obs.nodes:
        if node.scope is not NodeScope.own:
            continue
        alerted = node.alert_count_recent > 0
        insufficient = alerted and (
            node.evidence_level is None or node.evidence_level < e_min
        )
        nodes[node.node_id] = NodeEvidence(
            evidence_level=node.evidence_level,
            steps_since_investigated=node.steps_since_investigated,
            alerted=alerted,
            insufficient=insufficient,
        )
    return EvidenceView(nodes=nodes)
