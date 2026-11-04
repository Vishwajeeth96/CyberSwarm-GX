"""Proposal validation: the sole creator of ``ValidatedAction`` (stage S4).

:class:`ProposalValidator` is the single validation boundary of the whole
system (DESIGN section 3.2). It is the only place in production code that
constructs a :class:`~cyberswarm.interfaces.ValidatedAction`, and it assigns the
coordinator reason codes in the fixed dependency order of CONTRACTS section 8.6
and AD-21.

Dependency order (AD-21). ``SCHEMA_INVALID`` short-circuits every other check.
Once the input parses, ``EPISODE_MISMATCH``, ``EPISODE_NOT_ACTIVE``,
``WRONG_STEP``, ``AGENT_UNKNOWN``, ``DUPLICATE_PROPOSAL`` and
``UNKNOWN_ACTION_KIND`` always run. ``ACTION_NOT_ALLOWLISTED``,
``TARGET_REQUIRED`` / ``TARGET_NOT_ALLOWED`` and ``PARAMETER_INVALID`` need a
known action kind; ``UNKNOWN_TARGET_NODE`` needs a target; and
``TARGET_OUT_OF_AUTHORITY`` needs a known target and a known agent.
``ACTION_NOT_AVAILABLE`` runs only if none of those action/target/agent checks
failed, and ``CONFLICTING_PROPOSAL`` / ``HIGH_IMPACT_BUDGET_EXCEEDED`` run only
when every earlier check passed. Every check whose precondition holds is
evaluated and all its failures are listed, in canonical order.

To report the specific action-side codes (rather than a blanket
``SCHEMA_INVALID``) the validator inspects the raw input: a proposal object or a
mapping. A mapping that forms a well-shaped proposal envelope but carries an
unknown kind, a missing/forbidden target or a bad parameter is *not*
``SCHEMA_INVALID``; only an input that cannot be read as a proposal at all is.

This module imports only :mod:`cyberswarm.contracts` and
:mod:`cyberswarm.interfaces`. It performs no input or output and starts no
external program; validation is pure, in-memory computation.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from ..contracts import (
    ACTION_DEFS,
    HIGH_IMPACT,
    ActionKind,
    ActionParameters,
    ActionSpec,
    AgentId,
    AvailableActions,
    Decision,
    EpisodeId,
    PolicyProposal,
    ProposalId,
    REASON_CODE_ORDER,
    ReasonCode,
    Topology,
    ValidationId,
    ValidationResult,
    agent_id_for_zone,
    default_parameters,
)
from ..interfaces import ValidatedAction
from .allowlist import ActionAllowlist
from .config import CoordinatorConfig

__all__ = ["ProposalValidator", "ProposalView"]

#: Adapters used to tell whether a scraped id from an unparseable input is still
#: a well-formed id worth echoing back on the ``SCHEMA_INVALID`` result.
_PROPOSAL_ID_ADAPTER = TypeAdapter(ProposalId)
_AGENT_ID_ADAPTER = TypeAdapter(AgentId)

#: Codes whose presence suppresses ``ACTION_NOT_AVAILABLE`` (CONTRACTS 8.6).
_AVAILABILITY_BLOCKERS: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.AGENT_UNKNOWN,
        ReasonCode.UNKNOWN_ACTION_KIND,
        ReasonCode.TARGET_REQUIRED,
        ReasonCode.TARGET_NOT_ALLOWED,
        ReasonCode.UNKNOWN_TARGET_NODE,
        ReasonCode.TARGET_OUT_OF_AUTHORITY,
        ReasonCode.PARAMETER_INVALID,
    }
)


def _valid_or_none(adapter: TypeAdapter, value: Any) -> str | None:
    """Return ``value`` if it validates against ``adapter``, else ``None``."""

    if not isinstance(value, str):
        return None
    try:
        return adapter.validate_python(value)
    except Exception:  # noqa: BLE001 - a malformed id is simply "no id"
        return None


def _params_ok(kind: ActionKind, raw_parameters: Any) -> bool:
    """Return whether ``raw_parameters`` are valid for ``kind`` (CONTRACTS 6)."""

    try:
        params = ActionParameters.model_validate(raw_parameters or {})
    except Exception:  # noqa: BLE001 - out-of-range/garbage params are "invalid"
        return False
    allowed = ACTION_DEFS[kind].allowed_parameters
    if params.duration_steps is not None and "duration_steps" not in allowed:
        return False
    if params.level is not None and "level" not in allowed:
        return False
    return True


def _resolve_kind(value: Any) -> ActionKind | None:
    """Return the :class:`ActionKind` for ``value`` if it names one, else ``None``."""

    if isinstance(value, ActionKind):
        return value
    if isinstance(value, str):
        try:
            return ActionKind(value)
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class ProposalView:
    """A lenient read of one proposal, enough to assign every reason code.

    ``schema_invalid`` is set when the input cannot be read as a proposal at
    all. Otherwise the envelope fields are populated and the action is described
    structurally (``kind`` is ``None`` for an unknown kind, ``target_present``
    records whether a target was supplied, ``params_ok`` whether the parameters
    are legal for the kind). ``proposal`` is the fully-validated
    :class:`~cyberswarm.contracts.PolicyProposal` when the input was already
    valid, and ``None`` when only the envelope could be validated.
    """

    schema_invalid: bool
    proposal_id: str | None
    agent_id: str | None
    episode_id: str | None
    step: int | None
    kind: ActionKind | None
    kind_present: bool
    target: str | None
    target_present: bool
    params_ok: bool
    proposal: PolicyProposal | None


class ProposalValidator:
    """Validate proposals and build the one and only ``ValidatedAction``.

    Construct it per episode with the episode's
    :class:`~cyberswarm.contracts.Topology` and the
    :class:`~cyberswarm.coordinator.config.CoordinatorConfig`. The coordinator
    then calls :meth:`parse` once per raw proposal (to order them by agent and
    to reuse the read) and :meth:`validate` for each, supplying the running
    per-step context (accepted actions so far, proposal ids seen, the assigned
    ``validation_id`` and the state digest).
    """

    def __init__(self, topology: Topology, config: CoordinatorConfig) -> None:
        self._allowlist = ActionAllowlist(config.allowlist)
        self._allowlist_id = config.allowlist_id
        self._max_high_impact = config.max_high_impact_per_step
        self._node_ids = frozenset(topology.node_ids)
        self._agent_ids = frozenset(topology.agent_ids)
        self._authority: dict[str, frozenset[str]] = {
            agent_id_for_zone(zone): frozenset(topology.nodes_of_zone(zone))
            for zone in topology.zone_ids
        }

    # ------------------------------------------------------------------ #
    # Reading a raw proposal
    # ------------------------------------------------------------------ #
    def parse(self, raw: PolicyProposal | Mapping[str, Any]) -> ProposalView:
        """Read ``raw`` into a :class:`ProposalView` without raising."""

        if isinstance(raw, PolicyProposal):
            return self._view_from_proposal(raw)
        if not isinstance(raw, Mapping):
            return _schema_invalid(None, None)

        data = dict(raw)
        try:
            return self._view_from_proposal(PolicyProposal.model_validate(data))
        except Exception:  # noqa: BLE001 - fall back to a lenient envelope read
            pass

        envelope = dict(data)
        envelope["action"] = {"action_kind": ActionKind.no_op.value}
        try:
            env_prop = PolicyProposal.model_validate(envelope)
        except Exception:  # noqa: BLE001 - the envelope itself is malformed
            return _schema_invalid(
                _valid_or_none(_PROPOSAL_ID_ADAPTER, data.get("proposal_id")),
                _valid_or_none(_AGENT_ID_ADAPTER, data.get("agent_id")),
            )

        action = data.get("action")
        if not isinstance(action, Mapping) or action.get("action_kind") is None:
            return ProposalView(
                schema_invalid=True,
                proposal_id=env_prop.proposal_id,
                agent_id=env_prop.agent_id,
                episode_id=env_prop.episode_id,
                step=env_prop.step,
                kind=None,
                kind_present=False,
                target=None,
                target_present=False,
                params_ok=False,
                proposal=None,
            )

        kind = _resolve_kind(action.get("action_kind"))
        target_raw = action.get("target_node_id")
        target_present = target_raw is not None
        target = target_raw if isinstance(target_raw, str) else None
        params_ok = True if kind is None else _params_ok(kind, action.get("parameters"))
        return ProposalView(
            schema_invalid=False,
            proposal_id=env_prop.proposal_id,
            agent_id=env_prop.agent_id,
            episode_id=env_prop.episode_id,
            step=env_prop.step,
            kind=kind,
            kind_present=True,
            target=target,
            target_present=target_present,
            params_ok=params_ok,
            proposal=None,
        )

    @staticmethod
    def _view_from_proposal(prop: PolicyProposal) -> ProposalView:
        action = prop.action
        return ProposalView(
            schema_invalid=False,
            proposal_id=prop.proposal_id,
            agent_id=prop.agent_id,
            episode_id=prop.episode_id,
            step=prop.step,
            kind=action.action_kind,
            kind_present=True,
            target=action.target_node_id,
            target_present=action.target_node_id is not None,
            params_ok=True,
            proposal=prop,
        )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(
        self,
        raw: PolicyProposal | Mapping[str, Any] | ProposalView,
        *,
        episode_id: EpisodeId,
        step: int,
        available: Mapping[AgentId, AvailableActions],
        accepted_so_far: Sequence[ValidatedAction],
        seen_proposal_ids: Collection[str],
        validation_id: ValidationId,
        state_digest: str | None = None,
        active: bool = True,
    ) -> tuple[ValidationResult, ValidatedAction | None]:
        """Validate one proposal; return its result and (if accepted) its action.

        ``episode_id`` and ``step`` are the coordinator's active episode and
        current step. ``accepted_so_far`` are the actions already accepted this
        step (for the conflict and budget checks); ``seen_proposal_ids`` are the
        proposal ids already processed this episode (for ``DUPLICATE_PROPOSAL``).
        ``active`` is ``False`` when there is no running episode.
        """

        view = raw if isinstance(raw, ProposalView) else self.parse(raw)

        if view.schema_invalid:
            result = self._result(
                validation_id,
                episode_id,
                step,
                proposal_id=view.proposal_id,
                agent_id=view.agent_id,
                decision=Decision.rejected,
                codes=[ReasonCode.SCHEMA_INVALID],
                resolved=None,
                state_digest=state_digest,
            )
            return result, None

        codes = self._collect_codes(
            view,
            episode_id=episode_id,
            step=step,
            available=available,
            accepted_so_far=accepted_so_far,
            seen_proposal_ids=seen_proposal_ids,
            active=active,
        )

        if codes:
            result = self._result(
                validation_id,
                episode_id,
                step,
                proposal_id=view.proposal_id,
                agent_id=view.agent_id,
                decision=Decision.rejected,
                codes=codes,
                resolved=None,
                state_digest=state_digest,
            )
            return result, None

        resolved = self._resolve_action(view)
        validated = ValidatedAction(
            episode_id=episode_id,
            step=step,
            agent_id=view.agent_id,
            proposal_id=view.proposal_id,
            validation_id=validation_id,
            action=resolved,
        )
        result = self._result(
            validation_id,
            episode_id,
            step,
            proposal_id=view.proposal_id,
            agent_id=view.agent_id,
            decision=Decision.accepted,
            codes=[],
            resolved=resolved,
            state_digest=state_digest,
        )
        return result, validated

    # ------------------------------------------------------------------ #
    # Ordered reason-code checks (CONTRACTS 8.6, AD-21)
    # ------------------------------------------------------------------ #
    def _collect_codes(
        self,
        view: ProposalView,
        *,
        episode_id: EpisodeId,
        step: int,
        available: Mapping[AgentId, AvailableActions],
        accepted_so_far: Sequence[ValidatedAction],
        seen_proposal_ids: Collection[str],
        active: bool,
    ) -> list[ReasonCode]:
        codes: list[ReasonCode] = []
        kind_known = view.kind is not None
        agent_known = view.agent_id in self._agent_ids

        # Always run once the input parses.
        if active and view.episode_id != episode_id:
            codes.append(ReasonCode.EPISODE_MISMATCH)
        if not active:
            codes.append(ReasonCode.EPISODE_NOT_ACTIVE)
        if view.step != step:
            codes.append(ReasonCode.WRONG_STEP)
        if not agent_known:
            codes.append(ReasonCode.AGENT_UNKNOWN)
        if view.proposal_id is not None and view.proposal_id in seen_proposal_ids:
            codes.append(ReasonCode.DUPLICATE_PROPOSAL)
        if not kind_known:
            codes.append(ReasonCode.UNKNOWN_ACTION_KIND)

        # Need a known action kind.
        if kind_known:
            if not self._allowlist.permits(view.kind):
                codes.append(ReasonCode.ACTION_NOT_ALLOWLISTED)
            needs_target = ACTION_DEFS[view.kind].needs_target
            if needs_target and not view.target_present:
                codes.append(ReasonCode.TARGET_REQUIRED)
            if (not needs_target) and view.target_present:
                codes.append(ReasonCode.TARGET_NOT_ALLOWED)

        # Need a target.
        target_known_node = view.target_present and view.target in self._node_ids
        if view.target_present and not target_known_node:
            codes.append(ReasonCode.UNKNOWN_TARGET_NODE)

        # Need a known target and a known agent.
        if target_known_node and agent_known:
            if view.target not in self._authority.get(view.agent_id, frozenset()):
                codes.append(ReasonCode.TARGET_OUT_OF_AUTHORITY)

        # Need a known action kind.
        if kind_known and not view.params_ok:
            codes.append(ReasonCode.PARAMETER_INVALID)

        # Runs only if none of the action/target/agent checks failed.
        if not (set(codes) & _AVAILABILITY_BLOCKERS):
            if not self._is_available(view, available):
                codes.append(ReasonCode.ACTION_NOT_AVAILABLE)

        # Run only when every earlier check passed.
        if not codes:
            if self._conflicts(view, accepted_so_far):
                codes.append(ReasonCode.CONFLICTING_PROPOSAL)
            if self._exceeds_high_impact_budget(view, accepted_so_far):
                codes.append(ReasonCode.HIGH_IMPACT_BUDGET_EXCEEDED)

        return _canonical(codes)

    def _is_available(
        self, view: ProposalView, available: Mapping[AgentId, AvailableActions]
    ) -> bool:
        menu = available.get(view.agent_id)
        if menu is None:
            return False
        target = self._effective_target(view)
        for action in menu.actions:
            if action.action_kind == view.kind and action.target_node_id == target:
                return True
        return False

    @staticmethod
    def _conflicts(
        view: ProposalView, accepted_so_far: Sequence[ValidatedAction]
    ) -> bool:
        if view.kind is ActionKind.no_op or view.target is None:
            return False
        for accepted in accepted_so_far:
            if (
                accepted.action.action_kind is not ActionKind.no_op
                and accepted.action.target_node_id == view.target
            ):
                return True
        return False

    def _exceeds_high_impact_budget(
        self, view: ProposalView, accepted_so_far: Sequence[ValidatedAction]
    ) -> bool:
        if view.kind not in HIGH_IMPACT:
            return False
        already = sum(
            1 for a in accepted_so_far if a.action.action_kind in HIGH_IMPACT
        )
        return already + 1 > self._max_high_impact

    # ------------------------------------------------------------------ #
    # Building the resolved action and the result record
    # ------------------------------------------------------------------ #
    def _resolve_action(self, view: ProposalView) -> ActionSpec:
        """Fill defaults into the accepted action (CONTRACTS 8.6 resolved_action)."""

        kind = view.kind
        assert kind is not None  # guaranteed: an accepted proposal has a known kind
        base = default_parameters(kind)
        provided = view.proposal.action.parameters if view.proposal is not None else base
        params = ActionParameters(
            duration_steps=(
                provided.duration_steps
                if provided.duration_steps is not None
                else base.duration_steps
            ),
            level=provided.level if provided.level is not None else base.level,
        )
        return ActionSpec(
            action_kind=kind,
            target_node_id=self._effective_target(view),
            parameters=params,
        )

    @staticmethod
    def _effective_target(view: ProposalView) -> str | None:
        if view.proposal is not None:
            return view.proposal.action.target_node_id
        if view.kind is not None and ACTION_DEFS[view.kind].needs_target:
            return view.target
        return None

    def _result(
        self,
        validation_id: ValidationId,
        episode_id: EpisodeId,
        step: int,
        *,
        proposal_id: str | None,
        agent_id: str | None,
        decision: Decision,
        codes: Sequence[ReasonCode],
        resolved: ActionSpec | None,
        state_digest: str | None,
    ) -> ValidationResult:
        return ValidationResult(
            validation_id=validation_id,
            episode_id=episode_id,
            step=step,
            proposal_id=proposal_id,
            agent_id=agent_id,
            decision=decision,
            reason_codes=list(codes),
            resolved_action=resolved,
            allowlist_id=self._allowlist_id,
            state_digest=state_digest,
        )


def _schema_invalid(proposal_id: str | None, agent_id: str | None) -> ProposalView:
    """A view for an input that cannot be read as a proposal at all."""

    return ProposalView(
        schema_invalid=True,
        proposal_id=proposal_id,
        agent_id=agent_id,
        episode_id=None,
        step=None,
        kind=None,
        kind_present=False,
        target=None,
        target_present=False,
        params_ok=False,
        proposal=None,
    )


def _canonical(codes: Sequence[ReasonCode]) -> list[ReasonCode]:
    """Return ``codes`` in the canonical check order, without repeats."""

    present = set(codes)
    return [code for code in REASON_CODE_ORDER if code in present]
