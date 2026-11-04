"""The committed fixtures match a fresh regeneration (DELTA check DC-05).

Runs ``scripts/build_fixtures.py --check`` with the current interpreter. A zero
exit code means the committed ``fixtures/`` tree is byte-identical to what the
storyboard produces, so there is no drift between the fixtures and the models.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_FIXTURES = REPO_ROOT / "scripts" / "build_fixtures.py"


def test_build_fixtures_check_reports_zero_drift() -> None:
    result = subprocess.run(
        [sys.executable, str(BUILD_FIXTURES), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"build_fixtures --check reported drift (exit {result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "zero drift" in result.stdout
