"""Unit tests for :mod:`cyberswarm.common.config`.

Covers typed TOML loading, rejection of an unknown key as ``CONFIG_INVALID``,
rejection of malformed TOML, and the canonical resolved-config snapshot and its
round trip.
"""

from __future__ import annotations

import json
import re

import pytest

from cyberswarm.common.config import (
    ConfigModel,
    config_digest,
    load_toml,
    loads_toml,
    snapshot_config,
)
from cyberswarm.common.hashing import canonical_json
from cyberswarm.contracts.common import ContractError

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class _DemoConfig(ConfigModel):
    name: str
    weight: float
    steps: int = 10


def test_load_toml_reads_and_validates(tmp_path) -> None:
    path = tmp_path / "demo.toml"
    path.write_text('name = "demo"\nweight = 0.5\nsteps = 3\n', encoding="utf-8")

    config = load_toml(path, _DemoConfig)
    assert config == _DemoConfig(name="demo", weight=0.5, steps=3)


def test_load_toml_applies_defaults(tmp_path) -> None:
    path = tmp_path / "demo.toml"
    path.write_text('name = "demo"\nweight = 0.25\n', encoding="utf-8")

    config = load_toml(path, _DemoConfig)
    assert config.steps == 10


def test_load_toml_rejects_unknown_key(tmp_path) -> None:
    path = tmp_path / "demo.toml"
    path.write_text(
        'name = "demo"\nweight = 0.5\nmystery = 9\n', encoding="utf-8"
    )

    with pytest.raises(ContractError) as excinfo:
        load_toml(path, _DemoConfig)
    assert excinfo.value.code == "CONFIG_INVALID"


def test_loads_toml_rejects_unknown_key() -> None:
    with pytest.raises(ContractError) as excinfo:
        loads_toml('name = "demo"\nweight = 0.5\nextra = 1\n', _DemoConfig)
    assert excinfo.value.code == "CONFIG_INVALID"


def test_loads_toml_rejects_malformed_toml() -> None:
    with pytest.raises(ContractError) as excinfo:
        loads_toml("name = \n", _DemoConfig)
    assert excinfo.value.code == "CONFIG_INVALID"


def test_loads_toml_rejects_wrong_type() -> None:
    with pytest.raises(ContractError) as excinfo:
        loads_toml('name = "demo"\nweight = "heavy"\n', _DemoConfig)
    assert excinfo.value.code == "CONFIG_INVALID"


def test_snapshot_config_round_trips_through_json() -> None:
    config = _DemoConfig(name="demo", weight=0.25, steps=7)
    snapshot = snapshot_config(config)

    data = json.loads(snapshot)
    assert data == config.model_dump(mode="json")
    assert _DemoConfig.model_validate(data) == config


def test_snapshot_config_is_canonical_and_key_sorted() -> None:
    config = _DemoConfig(name="demo", weight=0.25, steps=7)
    snapshot = snapshot_config(config)

    assert snapshot == canonical_json(config)
    keys = list(json.loads(snapshot).keys())
    assert keys == sorted(keys)


def test_snapshot_config_is_deterministic() -> None:
    config = _DemoConfig(name="demo", weight=0.25, steps=7)
    assert snapshot_config(config) == snapshot_config(config)


def test_config_digest_is_stable_hex64() -> None:
    config = _DemoConfig(name="demo", weight=0.25, steps=7)
    digest = config_digest(config)
    assert _HEX64.match(digest)
    assert config_digest(config) == digest


def test_config_digest_matches_snapshot_hash() -> None:
    from cyberswarm.common.hashing import sha256_json

    config = _DemoConfig(name="demo", weight=0.25, steps=7)
    assert config_digest(config) == sha256_json(config)
