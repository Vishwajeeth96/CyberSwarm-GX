"""Canonical JSON and SHA-256 digests (stage S1).

The canonical JSON form used here is deterministic and key-order independent:
UTF-8, ``ensure_ascii=True``, compact separators and ``sort_keys=True``. Two
mappings that differ only in key insertion order therefore hash identically, and
a pydantic model hashes the same as its plain-``dict`` equivalent because a model
is first converted with ``model_dump(mode="json")``.

Note: because this form sorts keys, it is deliberately *not* the on-disk record
form of :mod:`cyberswarm.contracts.io` (which preserves pydantic
field-declaration order so JSON Lines stay human-readable). Use
:func:`sha256_file` when the exact bytes of a file must be hashed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

#: Read size for streaming file hashing (64 KiB).
_CHUNK: int = 1 << 16


def _to_jsonable(obj: Any) -> Any:
    """Return a JSON-native view of ``obj`` (pydantic models become dicts)."""

    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return obj


def canonical_json(obj: Any) -> str:
    """Return the canonical JSON string of ``obj`` (sorted keys, compact, ASCII)."""

    return json.dumps(
        _to_jsonable(obj),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(obj: Any) -> bytes:
    """Return :func:`canonical_json` encoded as UTF-8 bytes."""

    return canonical_json(obj).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase 64-character hex SHA-256 digest of ``data``."""

    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    """Return the SHA-256 digest of the canonical JSON bytes of ``obj``."""

    return sha256_bytes(canonical_json_bytes(obj))


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest of the raw bytes of the file at ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
