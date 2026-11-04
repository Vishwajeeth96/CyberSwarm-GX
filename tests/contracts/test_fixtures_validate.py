"""Every fixture validates (valid) or fails for its one reason (invalid) (DC-05).

For each committed *valid* fixture both the pydantic model and the exported
JSON Schema (via ``jsonschema``) accept it. For each committed *invalid* fixture
the pydantic model rejects it; the single intended reason is carried alongside
the fixture, and the committed files are checked to match the declared set so
none is silently missing.

The declared set is the S1 storyboard (``fixture_storyboard``) plus the S3 graph
records derived through the real converter (``graph_fixtures``). The storyboard
stays contracts-only, so the graph snapshot/feature fixtures are declared by the
separate graph builder rather than by the storyboard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from cyberswarm.contracts.common import ContractError
from cyberswarm.contracts.registry import RECORD_TYPES

REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_DIR = REPO_ROOT / "fixtures" / "records" / "valid"
INVALID_DIR = REPO_ROOT / "fixtures" / "records" / "invalid"
SCHEMAS_DIR = REPO_ROOT / "schemas"

# The storyboard (single source of the fixture values and the one reason each
# invalid fixture must fail for) lives under scripts/.
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import fixture_storyboard as sb  # noqa: E402
import graph_fixtures as gf  # noqa: E402

_VALID_FILES = sorted(VALID_DIR.glob("*.json"))
# Declared fixtures: S1 storyboard records plus the S3 graph records derived by
# scripts/graph_fixtures.py (the storyboard itself imports only contracts).
_DECLARED_VALID = [*sb.valid_fixtures(), *gf.valid_graph_fixtures()]
_INVALID_FIXTURES = [*sb.invalid_fixtures(), *gf.invalid_graph_fixtures()]


def _record_name(path: Path) -> str:
    return path.name.split(".")[0]


def _load_schema(name: str) -> dict:
    return json.loads((SCHEMAS_DIR / f"{name}.schema.json").read_text("utf-8"))


# --------------------------------------------------------------------------- #
# Valid fixtures: pydantic AND jsonschema both accept
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", _VALID_FILES, ids=lambda p: p.name)
def test_valid_fixture_passes_pydantic_and_jsonschema(path: Path) -> None:
    name = _record_name(path)
    assert name in RECORD_TYPES, f"{path.name} has no registered record type"
    payload = json.loads(path.read_text(encoding="utf-8"))

    RECORD_TYPES[name].model_validate(payload)  # pydantic
    jsonschema.validate(payload, _load_schema(name))  # exported JSON Schema


def test_valid_fixture_files_match_storyboard() -> None:
    on_disk = {p.relative_to(REPO_ROOT).as_posix() for p in _VALID_FILES}
    declared = {fixture.rel_path for fixture in _DECLARED_VALID}
    assert on_disk == declared


# --------------------------------------------------------------------------- #
# Invalid fixtures: pydantic rejects each for its one declared reason
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fixture",
    _INVALID_FIXTURES,
    ids=[f"{f.record_type}-{Path(f.rel_path).stem}" for f in _INVALID_FIXTURES],
)
def test_invalid_fixture_fails_for_its_reason(fixture: "sb.InvalidFixture") -> None:
    path = REPO_ROOT / fixture.rel_path
    assert path.exists(), f"missing committed invalid fixture {fixture.rel_path}"
    payload = json.loads(path.read_text(encoding="utf-8"))

    # The committed file must be rejected for its one intended reason (carried
    # in the test id). Byte-level agreement with the storyboard is covered by
    # test_fixtures_up_to_date.
    with pytest.raises((ContractError, ValidationError)):
        RECORD_TYPES[fixture.record_type].model_validate(payload)


def test_invalid_fixture_files_match_storyboard() -> None:
    on_disk = {p.relative_to(REPO_ROOT).as_posix() for p in INVALID_DIR.glob("*.json")}
    declared = {fixture.rel_path for fixture in _INVALID_FIXTURES}
    assert on_disk == declared
