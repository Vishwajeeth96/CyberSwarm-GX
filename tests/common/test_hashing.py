"""Unit tests for :mod:`cyberswarm.common.hashing`.

Covers digest stability, the canonical (key-order-independent) property, the
hex64 shape, model/dict equivalence, and raw-file hashing.
"""

from __future__ import annotations

import hashlib
import random
import re

from cyberswarm.common.hashing import (
    canonical_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from cyberswarm.contracts.manifest import Component, LogLevel, LogRecord

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def test_sha256_json_is_stable() -> None:
    obj = {"b": 2, "a": 1, "c": [3, 2, 1]}
    assert sha256_json(obj) == sha256_json(obj)


def test_sha256_json_is_canonical_key_order_independent() -> None:
    ascending = {"a": 1, "b": 2, "c": 3}
    descending = {"c": 3, "b": 2, "a": 1}
    assert canonical_json(ascending) == canonical_json(descending)
    assert sha256_json(ascending) == sha256_json(descending)


def test_sha256_json_is_lowercase_hex64() -> None:
    assert _HEX64.match(sha256_json({"x": 1}))


def test_sha256_json_changes_with_content() -> None:
    assert sha256_json({"x": 1}) != sha256_json({"x": 2})


def test_sha256_json_matches_model_dump_equivalent() -> None:
    record = LogRecord(
        ts="2026-10-30T20:00:00Z",
        level=LogLevel.info,
        component=Component.runner,
        message="Run started",
    )
    assert sha256_json(record) == sha256_json(record.model_dump(mode="json"))


def test_canonical_json_bytes_is_utf8_of_canonical_json() -> None:
    obj = {"greeting": "hello", "n": 3}
    assert canonical_json_bytes(obj) == canonical_json(obj).encode("utf-8")


def test_sha256_bytes_matches_hashlib() -> None:
    data = b"cyberswarm"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_sha256_json_canonical_over_many_key_shufflings() -> None:
    base = {f"k{i}": i for i in range(16)}
    expected = sha256_json(base)
    rng = random.Random(0)
    keys = list(base)
    for _ in range(100):
        rng.shuffle(keys)
        shuffled = {key: base[key] for key in keys}
        assert sha256_json(shuffled) == expected


def test_sha256_file_matches_raw_bytes(tmp_path) -> None:
    data = b"cyberswarm-common-hashing\n\x00\x01\x02binary"
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_sha256_file_handles_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert sha256_file(path) == hashlib.sha256(b"").hexdigest()
