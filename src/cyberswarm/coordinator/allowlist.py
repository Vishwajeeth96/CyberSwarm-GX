"""The explicit action allowlist (stage S4, DESIGN section 4.8).

:class:`ActionAllowlist` is a thin, immutable membership test over a set of
:class:`~cyberswarm.contracts.ActionKind` values. It is the coordinator's
enforcement of the ``ACTION_NOT_ALLOWLISTED`` reason code (CONTRACTS section
8.6): an action kind that is in the vocabulary but not on the allowlist is
rejected.

The allowlist is abstract-only: it ranges over the single eight-action
vocabulary (CONTRACTS section 6), each entry an abstract simulator state
transition. It never names a real address, command or payload.

This module imports only :mod:`cyberswarm.contracts`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..contracts import ACTION_DEFS, ActionKind

__all__ = ["ActionAllowlist"]


class ActionAllowlist:
    """An immutable set of permitted :class:`~cyberswarm.contracts.ActionKind`.

    Construct it from any iterable of action kinds (for example
    ``config.allowlist``). Every kind must be a member of the vocabulary; an
    unknown kind is a configuration error, raised at construction.
    """

    __slots__ = ("_kinds",)

    def __init__(self, kinds: Iterable[ActionKind]) -> None:
        resolved = frozenset(ActionKind(kind) for kind in kinds)
        unknown = resolved - set(ACTION_DEFS)
        if unknown:
            names = ", ".join(sorted(kind.value for kind in unknown))
            raise ValueError(f"allowlist contains unknown action kinds: {names}")
        self._kinds = resolved

    def permits(self, kind: ActionKind) -> bool:
        """Return ``True`` if ``kind`` is on the allowlist."""

        return kind in self._kinds

    @property
    def kinds(self) -> frozenset[ActionKind]:
        """The permitted action kinds (an immutable set)."""

        return self._kinds

    def __contains__(self, kind: object) -> bool:
        return kind in self._kinds

    def __iter__(self) -> Iterator[ActionKind]:
        return iter(self._kinds)

    def __len__(self) -> int:
        return len(self._kinds)
