"""Documented contract examples validate (stage S1).

Every fenced block in ``docs/CONTRACTS.md`` marked ``json contract=<Name>`` is
parsed and validated against its pydantic model; for a registered top-level
record it is also validated against the committed exported JSON Schema. This
keeps the documentation examples honest: a drifted or malformed example (for
instance a text field holding a character the contract forbids) fails here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jsonschema
import pytest

from cyberswarm.contracts import ProposalFeedback, UncertaintyInfo
from cyberswarm.contracts.registry import RECORD_TYPES

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS_MD = REPO_ROOT / "docs" / "CONTRACTS.md"
SCHEMAS_DIR = REPO_ROOT / "schemas"

_FENCE = re.compile(r"^```json contract=(\w+)\s*$")

# Nested models that appear as documented examples but are not top-level
# registered records (so they have no exported schema of their own).
_EXTRA_MODELS = {"ProposalFeedback": ProposalFeedback, "UncertaintyInfo": UncertaintyInfo}
_MODELS = {**RECORD_TYPES, **_EXTRA_MODELS}


def _parse_examples(text: str) -> list[tuple[str, str]]:
    """Return ``(record_name, json_body)`` for every ``json contract=`` block."""

    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        match = _FENCE.match(lines[i])
        if not match:
            i += 1
            continue
        name = match.group(1)
        body: list[str] = []
        i += 1
        while i < len(lines) and not lines[i].startswith("```"):
            body.append(lines[i])
            i += 1
        blocks.append((name, "\n".join(body).strip()))
        i += 1
    return blocks


_EXAMPLES = _parse_examples(CONTRACTS_MD.read_text(encoding="utf-8"))


def test_examples_were_found() -> None:
    # Guard against a parser or doc change that would make this suite vacuous.
    assert len(_EXAMPLES) >= 25
    found = {name for name, _ in _EXAMPLES}
    required = {
        "Node",
        "Edge",
        "Topology",
        "ScenarioConfig",
        "EventRecord",
        "AgentObservation",
        "AvailableActions",
        "PolicyProposal",
        "ValidationResult",
        "ExecutedAction",
        "EpisodeResult",
        "ErrorRecord",
        "RunManifest",
    }
    assert required <= found


@pytest.mark.parametrize(
    ("name", "body"),
    _EXAMPLES,
    ids=[f"{i}-{name}" for i, (name, _) in enumerate(_EXAMPLES)],
)
def test_documented_example_validates(name: str, body: str) -> None:
    assert name in _MODELS, f"unknown documented record type {name!r}"
    payload = json.loads(body)

    # Always validate against the pydantic model.
    _MODELS[name].model_validate(payload)

    # Registered records also validate against their exported JSON Schema.
    if name in RECORD_TYPES:
        schema = json.loads(
            (SCHEMAS_DIR / f"{name}.schema.json").read_text(encoding="utf-8")
        )
        jsonschema.validate(payload, schema)
