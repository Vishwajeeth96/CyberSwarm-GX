"""Per-agent flat action index space (stage S3, DESIGN 4.7, RL_IMPLEMENTATION 2.4).

An :class:`ActionTable` is the bijection between an agent's ``Discrete(A_i)``
action index and a fully specified :class:`~cyberswarm.contracts.ActionSpec`,
built once from the canonical action vocabulary and the agent's own-zone nodes:

* index ``0`` is ``no_op`` (no target);
* then, for each of the seven *targeted* kinds in the fixed vocabulary order
  (``monitor_node``, ``investigate_node``, ``request_evidence``,
  ``rate_limit_node``, ``quarantine_node``, ``isolate_node``, ``restore_node``),
  one index per node of the agent's zone in ascending ``node_id`` order.

Every spec carries the kind's default parameters (``monitor_node`` ->
``duration_steps = 3``; ``rate_limit_node`` -> ``level = light``); the RL action
space never exposes a non-default parameter. On the mock topology this gives
``A_dmz = 1 + 7 x 3 = 22`` and ``A_core = 1 + 7 x 5 = 36``.

:meth:`ActionTable.mask` turns an :class:`~cyberswarm.contracts.AvailableActions`
record into the boolean validity mask the environment and the actor use: an
index is valid exactly when its ``(kind, target)`` pair is currently available,
so the mask equals what the coordinator will accept for state-dependent
availability. ``no_op`` is always unmasked, so a valid action always exists.

:meth:`ActionTable.digest` is the checkpoint-compatibility fingerprint of the
action space (AD-24). It is keyed on the agent's node-id list and the action
set, and is deliberately independent of the topology id, the edges and the
criticalities, so a checkpoint trained on ``topo-mock-small-v1`` loads on the
unseen variant ``topo-mock-small-v2`` (same node ids and zones) without change.

This module imports only :mod:`cyberswarm.contracts`, :mod:`cyberswarm.common`
and :mod:`numpy`; it imports no simulator, environment or policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from ..common.hashing import sha256_json
from ..contracts import (
    ACTION_ORDER,
    ActionKind,
    ActionSpec,
    AgentId,
    AvailableActions,
    Topology,
    agent_id_for_zone,
    default_parameters,
)

__all__ = ["TARGETED_KINDS", "ActionTable"]


#: The seven targeted action kinds, in the fixed vocabulary order (CONTRACTS
#: section 6, ``ACTION_ORDER`` without ``no_op``). Each contributes one index per
#: own-zone node to the action table.
TARGETED_KINDS: Final[tuple[ActionKind, ...]] = tuple(
    kind for kind in ACTION_ORDER if kind is not ActionKind.no_op
)


def _spec_key(spec: ActionSpec) -> tuple[object, ...]:
    """Return the hashable identity of a spec for index lookup.

    The RL action space is keyed on ``(kind, target, duration_steps, level)``;
    two specs with the same kind, target and default parameters map to the same
    index, and a spec carrying a non-default parameter is simply not in the
    table.
    """

    return (
        spec.action_kind,
        spec.target_node_id,
        spec.parameters.duration_steps,
        spec.parameters.level,
    )


def _own_nodes(topology: Topology, agent_id: AgentId) -> tuple[str, ...]:
    """Return the agent's own-zone node ids, ascending (its authority)."""

    for zone_id in topology.zone_ids:
        if agent_id_for_zone(zone_id) == agent_id:
            return topology.nodes_of_zone(zone_id)
    raise ValueError(
        f"agent {agent_id!r} owns no zone in topology {topology.topology_id!r}"
    )


@dataclass(frozen=True)
class ActionTable:
    """The flat ``Discrete`` action space of one agent (DESIGN 4.7).

    ``specs`` is the index-ordered tuple of fully specified actions (index 0 is
    ``no_op``); ``own_nodes`` is the ascending own-zone node-id list the targeted
    actions range over. Build it with :meth:`build`; it is immutable afterwards.
    """

    agent_id: AgentId
    own_nodes: tuple[str, ...]
    specs: tuple[ActionSpec, ...]
    _index_by_key: dict[tuple[object, ...], int] = field(
        init=False, compare=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        index_by_key: dict[tuple[object, ...], int] = {}
        for index, spec in enumerate(self.specs):
            index_by_key[_spec_key(spec)] = index
        object.__setattr__(self, "_index_by_key", index_by_key)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, topology: Topology, agent_id: AgentId) -> "ActionTable":
        """Build the action table for ``agent_id`` over ``topology``.

        Index 0 is ``no_op``; then the seven targeted kinds in order, each over
        the agent's own-zone nodes ascending, every spec with default parameters.
        """

        own = _own_nodes(topology, agent_id)
        specs: list[ActionSpec] = [ActionSpec(action_kind=ActionKind.no_op)]
        for kind in TARGETED_KINDS:
            for node_id in own:
                specs.append(
                    ActionSpec(
                        action_kind=kind,
                        target_node_id=node_id,
                        parameters=default_parameters(kind),
                    )
                )
        return cls(agent_id=agent_id, own_nodes=own, specs=tuple(specs))

    # ------------------------------------------------------------------ #
    # Size and mapping
    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        """The action count ``A_i`` (``1 + 7 * len(own_nodes)``)."""

        return len(self.specs)

    def spec_of(self, index: int) -> ActionSpec:
        """Return the :class:`ActionSpec` at ``index`` (inverse of :meth:`index_of`)."""

        if not (0 <= index < self.n):
            raise IndexError(
                f"action index {index} is out of range [0, {self.n}) for {self.agent_id!r}"
            )
        return self.specs[index]

    def index_of(self, spec: ActionSpec) -> int:
        """Return the index of ``spec`` (inverse of :meth:`spec_of`).

        Raises :class:`KeyError` for a spec that is not in the action space (an
        unknown kind/target, or any non-default parameter).
        """

        try:
            return self._index_by_key[_spec_key(spec)]
        except KeyError:
            raise KeyError(
                f"action {spec.action_kind.value!r} on {spec.target_node_id!r} "
                f"is not in the action table of {self.agent_id!r}"
            ) from None

    # ------------------------------------------------------------------ #
    # Mask and digest
    # ------------------------------------------------------------------ #
    def mask(self, available: AvailableActions) -> np.ndarray:
        """Return the boolean validity mask of shape ``(n,)`` from ``available``.

        An index is valid exactly when its ``(kind, target)`` is listed in
        ``available``; ``no_op`` (index 0) is always valid. ``available`` must be
        this agent's record.
        """

        if available.agent_id != self.agent_id:
            raise ValueError(
                f"available actions are for {available.agent_id!r}, "
                f"not {self.agent_id!r}"
            )
        allowed = {
            (action.action_kind, action.target_node_id)
            for action in available.actions
        }
        mask = np.array(
            [(spec.action_kind, spec.target_node_id) in allowed for spec in self.specs],
            dtype=bool,
        )
        mask[0] = True  # no_op is always unmasked, so a valid action always exists
        return mask

    def digest(self) -> str:
        """Return the checkpoint-compatibility digest of the action space (AD-24).

        Keyed on the agent id, the own-node-id list and the ordered action kinds
        only, so it is identical for the two mock topologies (which share node
        ids and zones) and a ``v1`` checkpoint loads on ``v2``.
        """

        payload = {
            "agent_id": self.agent_id,
            "node_order": list(self.own_nodes),
            "action_kinds": [kind.value for kind in ACTION_ORDER],
        }
        return sha256_json(payload)
