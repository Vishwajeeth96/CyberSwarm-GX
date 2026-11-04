"""Shared setup for the contract tests (stage S1).

Puts the repository ``scripts/`` directory on ``sys.path`` so test modules may
import the fixture storyboard (the single source of the fixture *values*), and
exposes the repository root and the committed ``schemas/`` and ``fixtures/``
locations as pytest fixtures. Kept deliberately small: the parametrised tests
compute their own paths at import time, so collection never depends on a
fixture being requested first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

#: Repository root: ``tests/contracts/conftest.py`` -> ``<repo>``.
REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = REPO_ROOT / "scripts"

# The fixture storyboard and build script live under scripts/. Make them
# importable for every test module collected in this directory. conftest.py is
# imported before the sibling test modules, so a module-level
# ``import fixture_storyboard`` works.
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def schemas_dir() -> Path:
    return REPO_ROOT / "schemas"


@pytest.fixture(scope="session")
def valid_fixture_dir() -> Path:
    return REPO_ROOT / "fixtures" / "records" / "valid"


@pytest.fixture(scope="session")
def invalid_fixture_dir() -> Path:
    return REPO_ROOT / "fixtures" / "records" / "invalid"


@pytest.fixture(scope="session")
def mini_episode_dir() -> Path:
    return REPO_ROOT / "fixtures" / "mini_episode"


@pytest.fixture(scope="session")
def committed_schemas(schemas_dir: Path) -> dict[str, dict]:
    """Every committed JSON Schema keyed by record name."""

    out: dict[str, dict] = {}
    for path in sorted(schemas_dir.glob("*.schema.json")):
        name = path.name[: -len(".schema.json")]
        out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out
