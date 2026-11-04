"""Common contract primitives: base model, errors, shared enums and types.

This module is the root of the ``cyberswarm.contracts`` package. It imports
only ``pydantic`` and the standard library (contract rule: the contracts
package has no third-party dependency other than pydantic).

It defines:

* :class:`ContractModel` - the frozen, ``extra="forbid"``, finite-float base
  class shared by every record (CONTRACTS C2, C3, C5).
* :class:`ContractError` - the exception raised when a record fails validation
  on read (CONTRACTS C11) and by the security refinements of :data:`ShortText`.
* :class:`Provenance` and the shared enums listed in CONTRACTS section 3.
* the constrained scalar types :data:`Step`, :data:`UnitFloat`,
  :data:`ShortText`, :data:`RelPath`, :data:`Hex64` (CONTRACTS section 3, C8).
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Final

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)

SCHEMA_VERSION: Final[str] = "1.0"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ContractError(Exception):
    """A record was rejected whole during validation (CONTRACTS C11).

    ``ContractError`` is deliberately *not* a subclass of :class:`ValueError`.
    Pydantic only converts :class:`ValueError`/:class:`AssertionError` raised
    inside validators into an aggregated ``ValidationError``; any other
    exception propagates unchanged. The security refinements on
    :data:`ShortText` and :data:`~cyberswarm.contracts.ids.NodeId` therefore
    raise ``ContractError`` directly so that constructing a record with, for
    example, a dotted-quad text field raises ``ContractError`` (CONTRACTS C8).

    The read helpers in :mod:`cyberswarm.contracts.io` additionally wrap
    pydantic ``ValidationError`` into ``ContractError`` so that every failure on
    read surfaces as ``ContractError`` with a ``code`` of ``SCHEMA_INVALID`` or
    ``SCHEMA_VERSION_MISMATCH``.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "SCHEMA_INVALID",
        field_path: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.code = code
        self.field_path = field_path
        self.reason = reason if reason is not None else message
        super().__init__(message)


# --------------------------------------------------------------------------- #
# Base model
# --------------------------------------------------------------------------- #
class ContractModel(BaseModel):
    """Base class for every contract record.

    * ``extra="forbid"`` - an unknown field is a validation error, never
      ignored. This is how evaluator-only fields are kept out of agent-visible
      schemas by construction (CONTRACTS C3).
    * ``frozen=True`` - records are immutable once validated.
    * ``allow_inf_nan=False`` - ``NaN`` and ``Infinity`` are rejected on every
      float field and never written (CONTRACTS C5).
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
    )


# --------------------------------------------------------------------------- #
# Constrained scalar types (CONTRACTS section 3, C8)
# --------------------------------------------------------------------------- #
Step = Annotated[int, Field(ge=0, le=10000)]
"""Integer simulation step, 0 to 10000."""

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
"""Finite float in the closed interval [0.0, 1.0]."""

Hex64 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
"""Lowercase 64-character hexadecimal digest (for example a SHA-256 hash)."""


_SHORT_TEXT_PATTERN = r"^[A-Za-z0-9 .,()_-]+$"
_DOTTED_QUAD = re.compile(r"\b\d{1,3}(\.\d{1,3}){3}\b")


def _reject_shorttext_lookalikes(value: str) -> str:
    """Reject the dotted quad, ``..`` and a leading ``-`` inside ``ShortText``.

    The base charset (enforced by :class:`StringConstraints`) already forbids
    slashes, colons, quotes, pipes and the other shell/URL characters. These
    three extra refinements stop an IP address, a relative path or an option
    flag from hiding inside an otherwise legal text field (CONTRACTS C8).
    """

    if _DOTTED_QUAD.search(value):
        raise ContractError(
            "ShortText must not contain a dotted-quad IP-like value",
            reason="dotted_quad_rejected",
        )
    if ".." in value:
        raise ContractError(
            "ShortText must not contain the sequence '..'",
            reason="double_dot_rejected",
        )
    if value.startswith("-"):
        raise ContractError(
            "ShortText must not start with '-'",
            reason="leading_dash_rejected",
        )
    return value


ShortText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=160, pattern=_SHORT_TEXT_PATTERN),
    AfterValidator(_reject_shorttext_lookalikes),
]
"""1 to 160 characters from ``[A-Za-z0-9 .,()_-]``; no IP, path, URL or shell.

A dotted quad, the sequence ``..`` and a leading ``-`` are rejected with
:class:`ContractError` (CONTRACTS C8).
"""


_REL_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _check_rel_path(value: str) -> str:
    """Validate a relative POSIX path used only as a reference (CONTRACTS 3).

    Segments are ``[A-Za-z0-9._-]+`` joined by ``/``; no leading ``/``, no empty
    segment, no segment equal to ``..`` or starting with ``..``. A ``RelPath`` is
    never opened as a command.
    """

    if value.startswith("/"):
        raise ValueError("RelPath must not start with '/'")
    segments = value.split("/")
    for segment in segments:
        if segment == "":
            raise ValueError("RelPath must not contain an empty segment")
        if segment.startswith(".."):
            raise ValueError("RelPath segment must not start with '..'")
        if not _REL_PATH_SEGMENT.match(segment):
            raise ValueError(
                "RelPath segment must match [A-Za-z0-9._-]+",
            )
    return value


RelPath = Annotated[
    str,
    StringConstraints(min_length=1, max_length=200),
    AfterValidator(_check_rel_path),
]
"""Relative POSIX path of at most 200 characters, used only as a reference."""


# --------------------------------------------------------------------------- #
# Shared enums (CONTRACTS section 3)
# --------------------------------------------------------------------------- #
class ProvenanceKind(StrEnum):
    """Where a record's values came from."""

    fixture_synthetic = "fixture_synthetic"
    mock_simulator_run = "mock_simulator_run"
    simulator_run = "simulator_run"


class NodeStatus(StrEnum):
    """Agent-visible node status; follows from actions, not a compromise flag."""

    up = "up"
    rate_limited = "rate_limited"
    quarantined = "quarantined"
    isolated = "isolated"
    restoring = "restoring"


class ImpactClass(StrEnum):
    """Abstract impact class of an action."""

    none = "none"
    low = "low"
    medium = "medium"
    high = "high"


class Representation(StrEnum):
    """Observation representation of a policy arm.

    ``learned_embedding`` names the staged learned-graph experiment and is part
    of version 1.0 so that adding that stage is not a contract change; no record
    carries it unless that stage ran.
    """

    conventional = "conventional"
    graph_features = "graph_features"
    learned_embedding = "learned_embedding"


class Split(StrEnum):
    """Dataset split.

    ``eval`` holds held-out scenarios on the training topology; ``eval_utopo``
    holds held-out scenarios on the unseen topology variant. ``cal`` is used
    only for threshold selection, calibration and model selection and is never
    reported as a result.
    """

    train = "train"
    cal = "cal"
    eval = "eval"
    eval_utopo = "eval_utopo"
    dev = "dev"
    fixture = "fixture"


class ComponentStatus(StrEnum):
    """Execution-derived status of a staged component."""

    not_run = "not_run"
    implemented = "implemented"
    no_go = "no_go"
    deferred = "deferred"


class SimulatorKind(StrEnum):
    """Which simulator produced a run. A mock run is never labelled a result."""

    mock = "mock"
    simulator = "simulator"
    fixture = "fixture"


class Decision(StrEnum):
    """Coordinator decision on a proposal. There is no third state."""

    accepted = "accepted"
    rejected = "rejected"


class ExecutionStatus(StrEnum):
    """Whether an accepted action executed or failed (state then unchanged)."""

    executed = "executed"
    failed = "failed"


class ReasonCode(StrEnum):
    """Coordinator rejection codes (CONTRACTS section 8.6).

    The member order is the canonical check order used to validate that a
    :class:`~cyberswarm.contracts.actions.ValidationResult` lists its codes in
    order. ``ReasonCode`` and :class:`ErrorCode` are separate enums that share
    only the literal ``SCHEMA_INVALID``.
    """

    SCHEMA_INVALID = "SCHEMA_INVALID"
    EPISODE_MISMATCH = "EPISODE_MISMATCH"
    EPISODE_NOT_ACTIVE = "EPISODE_NOT_ACTIVE"
    WRONG_STEP = "WRONG_STEP"
    AGENT_UNKNOWN = "AGENT_UNKNOWN"
    DUPLICATE_PROPOSAL = "DUPLICATE_PROPOSAL"
    UNKNOWN_ACTION_KIND = "UNKNOWN_ACTION_KIND"
    ACTION_NOT_ALLOWLISTED = "ACTION_NOT_ALLOWLISTED"
    TARGET_REQUIRED = "TARGET_REQUIRED"
    TARGET_NOT_ALLOWED = "TARGET_NOT_ALLOWED"
    UNKNOWN_TARGET_NODE = "UNKNOWN_TARGET_NODE"
    TARGET_OUT_OF_AUTHORITY = "TARGET_OUT_OF_AUTHORITY"
    PARAMETER_INVALID = "PARAMETER_INVALID"
    ACTION_NOT_AVAILABLE = "ACTION_NOT_AVAILABLE"
    CONFLICTING_PROPOSAL = "CONFLICTING_PROPOSAL"
    HIGH_IMPACT_BUDGET_EXCEEDED = "HIGH_IMPACT_BUDGET_EXCEEDED"


#: Canonical order of the coordinator reason codes (the member order above).
REASON_CODE_ORDER: Final[tuple[ReasonCode, ...]] = tuple(ReasonCode)
_REASON_CODE_INDEX: Final[dict[ReasonCode, int]] = {
    code: index for index, code in enumerate(REASON_CODE_ORDER)
}


class ErrorCode(StrEnum):
    """Structured error codes (CONTRACTS section 11.4)."""

    SCHEMA_INVALID = "SCHEMA_INVALID"
    SCHEMA_VERSION_MISMATCH = "SCHEMA_VERSION_MISMATCH"
    UNKNOWN_ID = "UNKNOWN_ID"
    SCENARIO_UNSUPPORTED = "SCENARIO_UNSUPPORTED"
    SIMULATOR_FAILURE = "SIMULATOR_FAILURE"
    ACTION_EXECUTION_FAILED = "ACTION_EXECUTION_FAILED"
    POLICY_FAILURE = "POLICY_FAILURE"
    UNCERTAINTY_UNAVAILABLE = "UNCERTAINTY_UNAVAILABLE"
    CHECKPOINT_ERROR = "CHECKPOINT_ERROR"
    CONFIG_INVALID = "CONFIG_INVALID"
    IO_ERROR = "IO_ERROR"
    LEAKAGE_DETECTED = "LEAKAGE_DETECTED"
    INVARIANT_VIOLATION = "INVARIANT_VIOLATION"


def reason_codes_in_order(codes: list[ReasonCode]) -> bool:
    """Return ``True`` if ``codes`` are strictly ascending in the check order."""

    indices = [_REASON_CODE_INDEX[code] for code in codes]
    return all(earlier < later for earlier, later in zip(indices, indices[1:]))


# --------------------------------------------------------------------------- #
# Provenance (CONTRACTS section 3, C10)
# --------------------------------------------------------------------------- #
class Provenance(ContractModel):
    """Where a measured or compared record's values came from."""

    kind: ProvenanceKind
    note: ShortText | None = None
