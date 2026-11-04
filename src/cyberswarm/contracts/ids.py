"""Identifier types, patterns and deterministic builders (CONTRACTS section 4).

Every identifier is lowercase ASCII. Slugs match ``[a-z0-9]+(-[a-z0-9]+)*``.
Nothing in an identifier encodes ground truth. ``NodeId`` additionally rejects
an address look-alike (four numeric groups after ``node-``) so that an IP cannot
masquerade as a node id.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, StringConstraints

from .common import ContractError

# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
_SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"

_NODE_ADDRESS_LOOKALIKE = re.compile(r"^\d{1,3}(?:-\d{1,3}){3}$")


def _reject_node_address_lookalike(value: str) -> str:
    """Reject ``node-10-0-0-1`` and similar four-group numeric ids (CONTRACTS 4)."""

    remainder = value[len("node-") :]
    if _NODE_ADDRESS_LOOKALIKE.match(remainder):
        raise ContractError(
            "NodeId must not look like an IP address",
            reason="node_address_lookalike_rejected",
        )
    return value


def _pattern(regex: str, *, max_length: int | None = None):
    """Return pydantic ``StringConstraints`` metadata for an id pattern."""

    return StringConstraints(pattern=regex, max_length=max_length)


# --------------------------------------------------------------------------- #
# Identifier types (CONTRACTS section 4)
# --------------------------------------------------------------------------- #
NodeId = Annotated[
    str,
    _pattern(rf"^node-{_SLUG}$", max_length=40),
    AfterValidator(_reject_node_address_lookalike),
]
EdgeId = Annotated[str, _pattern(r"^edge-[0-9]{3,4}$")]
ZoneId = Annotated[str, _pattern(rf"^zone-{_SLUG}$")]
AgentId = Annotated[str, _pattern(rf"^agent-{_SLUG}$")]
TopologyId = Annotated[str, _pattern(rf"^topo-{_SLUG}-v[0-9]+$")]
ScenarioId = Annotated[str, _pattern(rf"^scn-{_SLUG}$")]
RunId = Annotated[str, _pattern(rf"^run-{_SLUG}$")]
PolicyId = Annotated[str, _pattern(rf"^pol-{_SLUG}$")]
PolicyFamilyId = Annotated[str, _pattern(rf"^fam-{_SLUG}$")]
EpisodeId = Annotated[
    str, _pattern(rf"^ep-{_SLUG}-s[0-9]+$", max_length=120)
]
EventId = Annotated[str, _pattern(r"^(?:evt|cev)-[0-9]{6}$")]
ProposalId = Annotated[str, _pattern(r"^prop-[0-9]{6}$")]
ValidationId = Annotated[str, _pattern(r"^val-[0-9]{6}$")]
ExecutionId = Annotated[str, _pattern(r"^exe-[0-9]{6}$")]
ErrorId = Annotated[str, _pattern(r"^err-[0-9]{6}$")]

# Derived per-step ids.
ObservationId = Annotated[str, _pattern(rf"^obs-[0-9]{{4}}-agent-{_SLUG}$")]
AvailableActionsId = Annotated[str, _pattern(rf"^avl-[0-9]{{4}}-agent-{_SLUG}$")]
DetectionId = Annotated[str, _pattern(rf"^det-[0-9]{{4}}-agent-{_SLUG}$")]
SnapshotId = Annotated[str, _pattern(rf"^gsn-[0-9]{{4}}-agent-{_SLUG}$")]
FeatureRecordId = Annotated[str, _pattern(rf"^gft-[0-9]{{4}}-agent-{_SLUG}$")]
GraphModelOutputId = Annotated[str, _pattern(rf"^gmo-[0-9]{{4}}-agent-{_SLUG}$")]
NodeScoreId = Annotated[str, _pattern(rf"^nsc-[0-9]{{4}}-agent-{_SLUG}$")]

# Evaluator-assigned summary ids.
MetricsId = Annotated[str, _pattern(rf"^met-{_SLUG}$")]
AggregateId = Annotated[str, _pattern(rf"^agg-{_SLUG}$")]
ComparisonId = Annotated[str, _pattern(rf"^cmp-{_SLUG}$")]
MatchedSetId = Annotated[str, _pattern(rf"^set-{_SLUG}$")]
CalibrationReportId = Annotated[str, _pattern(rf"^cbr-{_SLUG}$")]
DecisionId = Annotated[str, _pattern(rf"^dec-{_SLUG}$")]


# --------------------------------------------------------------------------- #
# Deterministic builders
# --------------------------------------------------------------------------- #
def _slug_after(prefix: str, identifier: str) -> str:
    """Return the slug part of ``identifier`` after a known ``prefix-``."""

    if not identifier.startswith(prefix):
        raise ContractError(
            f"expected an id starting with {prefix!r}, got {identifier!r}",
            reason="id_prefix_mismatch",
        )
    return identifier[len(prefix) :]


def agent_id_for_zone(zone_id: str) -> str:
    """Return the decision agent id for a zone (``zone-dmz`` -> ``agent-dmz``)."""

    return "agent-" + _slug_after("zone-", zone_id)


def make_episode_id(policy_id: str, scenario_id: str, seed: int) -> str:
    """Build the deterministic episode id from policy, scenario and seed.

    ``pol-rule-conv-v1`` + ``scn-mock-lateral-eval-01`` + ``2001`` ->
    ``ep-rule-conv-v1-mock-lateral-eval-01-s2001`` (CONTRACTS section 4).
    """

    if seed < 0:
        raise ContractError("seed must be non-negative", reason="seed_negative")
    policy_slug = _slug_after("pol-", policy_id)
    scenario_slug = _slug_after("scn-", scenario_id)
    return f"ep-{policy_slug}-{scenario_slug}-s{seed}"


def observation_id(step: int, agent_id: str) -> str:
    """``obs-<step:04d>-<agent_id>``."""

    return f"obs-{step:04d}-{agent_id}"


def available_actions_id(step: int, agent_id: str) -> str:
    """``avl-<step:04d>-<agent_id>``."""

    return f"avl-{step:04d}-{agent_id}"


def detection_id(step: int, agent_id: str) -> str:
    """``det-<step:04d>-<agent_id>``."""

    return f"det-{step:04d}-{agent_id}"


def snapshot_id(step: int, agent_id: str) -> str:
    """``gsn-<step:04d>-<agent_id>``."""

    return f"gsn-{step:04d}-{agent_id}"


def feature_record_id(step: int, agent_id: str) -> str:
    """``gft-<step:04d>-<agent_id>``."""

    return f"gft-{step:04d}-{agent_id}"


def graph_model_output_id(step: int, agent_id: str) -> str:
    """``gmo-<step:04d>-<agent_id>``."""

    return f"gmo-{step:04d}-{agent_id}"


def node_score_id(step: int, agent_id: str, scorer_slug: str) -> str:
    """``nsc-<step:04d>-<agent_id>-<scorer slug>``."""

    return f"nsc-{step:04d}-{agent_id}-{scorer_slug}"


def event_id(counter: int, prefix: str = "evt") -> str:
    """``<prefix>-<counter:06d>`` with ``evt`` (simulator) or ``cev`` (coordinator)."""

    if prefix not in ("evt", "cev"):
        raise ContractError(
            "event id prefix must be 'evt' or 'cev'", reason="event_prefix_invalid"
        )
    return f"{prefix}-{counter:06d}"


def proposal_id(counter: int) -> str:
    """``prop-<counter:06d>``."""

    return f"prop-{counter:06d}"


def validation_id(counter: int) -> str:
    """``val-<counter:06d>``."""

    return f"val-{counter:06d}"


def execution_id(counter: int) -> str:
    """``exe-<counter:06d>``."""

    return f"exe-{counter:06d}"


def error_id(counter: int) -> str:
    """``err-<counter:06d>``."""

    return f"err-{counter:06d}"
