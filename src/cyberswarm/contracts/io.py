"""Canonical JSON and JSON Lines read/write helpers (CONTRACTS section 2).

Canonical form (so determinism tests can compare bytes): UTF-8, ``ensure_ascii``,
no spaces after separators, field order equal to the pydantic model declaration
order, Python ``repr`` floats (shortest round-trip), integers as JSON integers,
booleans as ``true``/``false``. Reads reject a bad record whole with
:class:`ContractError` (CONTRACTS C11).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .common import SCHEMA_VERSION, ContractError, ContractModel
from .registry import RECORD_TYPES


def dumps_canonical(model: ContractModel) -> str:
    """Serialize a record to a single canonical JSON string (no trailing newline)."""

    payload = model.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def check_version(raw_version: str) -> None:
    """Raise ``SCHEMA_VERSION_MISMATCH`` unless ``raw_version`` is readable here.

    A reader accepts its own MAJOR and any MINOR less than or equal to its own
    (CONTRACTS section 13). The current version is ``1.0``.
    """

    current_major, current_minor = (int(part) for part in SCHEMA_VERSION.split("."))
    try:
        major_str, minor_str = str(raw_version).split(".")
        major, minor = int(major_str), int(minor_str)
    except (ValueError, AttributeError) as exc:
        raise ContractError(
            f"unreadable schema_version {raw_version!r}",
            code="SCHEMA_VERSION_MISMATCH",
            reason="schema_version_unparseable",
        ) from exc
    if major != current_major or minor > current_minor:
        raise ContractError(
            f"schema_version {raw_version!r} is not readable by {SCHEMA_VERSION}",
            code="SCHEMA_VERSION_MISMATCH",
            reason="schema_version_mismatch",
        )


def _validate_payload(name: str, payload: object) -> ContractModel:
    """Validate a decoded payload against a registered record type."""

    try:
        model_cls = RECORD_TYPES[name]
    except KeyError as exc:
        raise ContractError(
            f"unknown record type {name!r}",
            code="UNKNOWN_ID",
            reason="unknown_record_type",
        ) from exc

    if isinstance(payload, dict) and "schema_version" in payload:
        check_version(payload["schema_version"])

    try:
        return model_cls.model_validate(payload)
    except ContractError:
        raise
    except Exception as exc:  # pydantic ValidationError and friends
        raise ContractError(
            f"invalid {name} record",
            code="SCHEMA_INVALID",
            reason=str(exc),
        ) from exc


def loads_record(name: str, text: str) -> ContractModel:
    """Parse one canonical JSON record of a named type, raising ``ContractError``."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError(
            f"{name} record is not valid JSON",
            code="SCHEMA_INVALID",
            reason=str(exc),
        ) from exc
    return _validate_payload(name, payload)


def read_json(path: Path, model: type[ContractModel]) -> ContractModel:
    """Read one JSON document and validate it as ``model``."""

    name = model.__name__
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(
            f"cannot read {name} at {path}",
            code="IO_ERROR",
            reason=str(exc),
        ) from exc
    return loads_record(name, text)


def read_jsonl(path: Path, model: type[ContractModel]) -> Iterator[ContractModel]:
    """Yield validated records from a JSON Lines file."""

    name = model.__name__
    path = Path(path)
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise ContractError(
            f"cannot read {name} lines at {path}",
            code="IO_ERROR",
            reason=str(exc),
        ) from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield loads_record(name, stripped)
            except ContractError as exc:
                raise ContractError(
                    f"invalid {name} at {path}:{line_number}",
                    code=exc.code,
                    field_path=exc.field_path,
                    reason=exc.reason,
                ) from exc


def write_json(path: Path, model: ContractModel) -> None:
    """Write one record as a single canonical JSON document with a trailing newline."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_canonical(model) + "\n", encoding="utf-8")


class JsonlWriter:
    """Append canonical records to a JSON Lines file, one record per line."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("w", encoding="utf-8")

    def write(self, model: ContractModel) -> None:
        self._handle.write(dumps_canonical(model) + "\n")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
