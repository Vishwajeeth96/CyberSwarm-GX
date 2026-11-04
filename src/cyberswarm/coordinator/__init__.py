"""The safe defense coordinator (stage S4, DESIGN section 4.8).

This package is the system's single safety boundary. It validates every policy
proposal and is the only caller of the simulator's public ``apply_action``; it
is pure, in-memory validation and bookkeeping with no real-host, remote or
filesystem side effects (DESIGN section 3.3). It imports only
:mod:`cyberswarm.contracts`, :mod:`cyberswarm.interfaces` and
:mod:`cyberswarm.common`.

* :class:`CoordinatorConfig` / :func:`load_coordinator_config` - the allowlist
  and high-impact budget, loaded from
  ``configs/experiments/coordinator_default.toml``.
* :class:`ActionAllowlist` - the explicit, abstract-only action allowlist.
* :class:`ProposalValidator` - the sole creator of
  :class:`~cyberswarm.interfaces.ValidatedAction`, assigning reason codes in the
  fixed dependency order (CONTRACTS section 8.6, AD-21).
* :class:`DefenseCoordinator` - the :class:`~cyberswarm.interfaces.CoordinatorPort`
  implementation driving validation (P3) and execution (P4).
* :class:`RecoveryVerifier` - agent-visible recovery verification (AD-20).
"""

from __future__ import annotations

from .allowlist import ActionAllowlist
from .config import CoordinatorConfig, load_coordinator_config, DEFAULT_CONFIG_PATH
from .coordinator import DefenseCoordinator
from .recovery import RecoveryVerifier
from .validator import ProposalValidator, ProposalView

__all__ = [
    "ActionAllowlist",
    "CoordinatorConfig",
    "load_coordinator_config",
    "DEFAULT_CONFIG_PATH",
    "DefenseCoordinator",
    "RecoveryVerifier",
    "ProposalValidator",
    "ProposalView",
]
