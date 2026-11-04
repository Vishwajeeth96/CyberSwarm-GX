"""Coordinator configuration (stage S4, DESIGN section 4.8).

:class:`CoordinatorConfig` is the typed, frozen view of
``configs/experiments/coordinator_default.toml``: the allowlist identifier, the
set of allowlisted action kinds and the per-step high-impact budget. Unknown
keys are rejected (it subclasses :class:`~cyberswarm.common.config.ConfigModel`),
so a malformed configuration stops a run before it starts.

This module imports only :mod:`cyberswarm.contracts` and
:mod:`cyberswarm.common` (DESIGN section 3.2). It reads a configuration file
only through :func:`cyberswarm.common.config.load_toml`; it performs no other
input or output and spawns nothing.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from ..common.config import ConfigModel, load_toml
from ..contracts import ActionKind, ShortText

__all__ = ["CoordinatorConfig", "load_coordinator_config", "DEFAULT_CONFIG_PATH"]


#: Repository-relative default configuration, resolved from this file's location
#: so no machine-specific absolute path is baked into committed code.
DEFAULT_CONFIG_PATH: Path = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "experiments"
    / "coordinator_default.toml"
)


class CoordinatorConfig(ConfigModel):
    """Typed coordinator configuration (DESIGN section 4.8).

    ``allowlist`` is stored as a ``frozenset`` of :class:`~cyberswarm.contracts.ActionKind`
    because membership is the only thing the coordinator asks of it; a list in
    the TOML is coerced to the set on load. ``allowlist_id`` is the
    :class:`~cyberswarm.contracts.ShortText` identifier written to every
    :class:`~cyberswarm.contracts.ValidationResult`.
    """

    allowlist_id: ShortText
    allowlist: frozenset[ActionKind]
    max_high_impact_per_step: int = Field(default=1, ge=0)


def load_coordinator_config(path: Path | str | None = None) -> CoordinatorConfig:
    """Load and validate the coordinator configuration.

    ``path`` defaults to :data:`DEFAULT_CONFIG_PATH`
    (``configs/experiments/coordinator_default.toml``). A malformed or
    unknown-key file raises
    :class:`~cyberswarm.contracts.common.ContractError` (code ``CONFIG_INVALID``).
    """

    return load_toml(path if path is not None else DEFAULT_CONFIG_PATH, CoordinatorConfig)
