"""Typed TOML configuration loading and canonical snapshots (stage S1).

:func:`load_toml` reads a TOML file and validates it against a pydantic model.
An unknown key is rejected because configuration models forbid extra fields
(subclass :class:`ConfigModel`, or pass any model with ``extra="forbid"``); a
rejected or malformed file raises
:class:`~cyberswarm.contracts.common.ContractError` with code ``CONFIG_INVALID``
so a run stops before it starts (ARCHITECTURE section 8).

:func:`snapshot_config` returns the canonical JSON of a resolved configuration
and :func:`config_digest` its SHA-256, so a run can record exactly which
configuration produced it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from cyberswarm.contracts.common import ContractError

from .hashing import canonical_json, sha256_json

M = TypeVar("M", bound=BaseModel)


class ConfigModel(BaseModel):
    """Base class for configuration models: unknown keys are rejected.

    ``extra="forbid"`` makes an unexpected key a validation error (surfaced as
    ``CONFIG_INVALID`` by :func:`load_toml`). ``frozen=True`` keeps a resolved
    configuration immutable after loading.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


def validate_config(data: Any, model: type[M], *, source: str | None = None) -> M:
    """Validate a decoded mapping against ``model``, raising ``ContractError``.

    Unknown keys (when the model forbids extras), wrong types and missing
    required keys all surface as ``ContractError(code="CONFIG_INVALID")``.
    """

    try:
        return model.model_validate(data)
    except ValidationError as exc:
        where = f" in {source}" if source else ""
        raise ContractError(
            f"invalid {model.__name__} configuration{where}",
            code="CONFIG_INVALID",
            reason=str(exc),
        ) from exc


def loads_toml(text: str, model: type[M], *, source: str | None = None) -> M:
    """Parse TOML ``text`` and validate it against ``model``."""

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        where = f" in {source}" if source else ""
        raise ContractError(
            f"malformed TOML{where}",
            code="CONFIG_INVALID",
            reason=str(exc),
        ) from exc
    return validate_config(data, model, source=source)


def load_toml(path: Path | str, model: type[M]) -> M:
    """Read the TOML file at ``path`` and validate it against ``model``."""

    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(
            f"cannot read configuration at {path}",
            code="IO_ERROR",
            reason=str(exc),
        ) from exc
    return loads_toml(text, model, source=str(path))


def snapshot_config(obj: BaseModel | dict[str, Any]) -> str:
    """Return the canonical JSON of a resolved configuration."""

    return canonical_json(obj)


def config_digest(obj: BaseModel | dict[str, Any]) -> str:
    """Return the SHA-256 digest of the canonical configuration snapshot."""

    return sha256_json(obj)
