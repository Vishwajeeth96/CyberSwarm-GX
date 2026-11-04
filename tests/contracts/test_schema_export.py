"""Exported JSON Schemas equal the committed files (DELTA checks DC-01, DC-02).

Re-running the exporter must reproduce every committed
``schemas/<Name>.schema.json`` byte-for-byte, the committed set must be exactly
one file per registered record type, and each file must carry the agreed
``$schema`` dialect and the stable ``$id`` URN.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyberswarm.contracts import SCHEMA_VERSION
from cyberswarm.contracts.export_schemas import (
    JSON_SCHEMA_DIALECT,
    _render,
    export,
    schema_for,
    schema_id,
)
from cyberswarm.contracts.registry import RECORD_TYPES

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = REPO_ROOT / "schemas"

_RECORD_NAMES = sorted(RECORD_TYPES)


def test_committed_schema_set_matches_registry() -> None:
    committed = {p.name for p in SCHEMAS_DIR.glob("*.schema.json")}
    expected = {f"{name}.schema.json" for name in RECORD_TYPES}
    assert committed == expected


@pytest.mark.parametrize("name", _RECORD_NAMES)
def test_committed_schema_is_byte_identical_to_fresh_export(name: str) -> None:
    committed = (SCHEMAS_DIR / f"{name}.schema.json").read_text(encoding="utf-8")
    assert committed == _render(schema_for(name))


@pytest.mark.parametrize("name", _RECORD_NAMES)
def test_schema_declares_dialect_and_id(name: str) -> None:
    schema = json.loads((SCHEMAS_DIR / f"{name}.schema.json").read_text("utf-8"))
    assert schema["$schema"] == JSON_SCHEMA_DIALECT
    assert schema["$id"] == schema_id(name)
    assert schema["$id"] == f"urn:cyberswarm:schema:{name}:{SCHEMA_VERSION}"


def test_reexport_to_temp_dir_matches_committed(tmp_path) -> None:
    written = export(tmp_path)
    assert len(written) == len(RECORD_TYPES)
    for path in written:
        committed = (SCHEMAS_DIR / path.name).read_text(encoding="utf-8")
        assert path.read_text(encoding="utf-8") == committed
