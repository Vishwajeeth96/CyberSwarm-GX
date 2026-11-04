"""Unit tests for :mod:`cyberswarm.common.seeding`.

Covers deterministic derivation of the three sub-stream seeds, their
distinctness, and that ``seed_everything`` runs both without ``torch`` installed
and with ``torch`` present (seeding it deterministically).
"""

from __future__ import annotations

import sys

import pytest

from cyberswarm.common.seeding import SeedStreams, derive_seeds, seed_everything


def test_derive_seeds_is_deterministic_for_a_fixed_seed() -> None:
    assert derive_seeds(12345) == derive_seeds(12345)


def test_derive_seeds_known_values() -> None:
    # Regression guard: SeedSequence is a stable, documented algorithm.
    result = derive_seeds(12345)
    assert result == SeedStreams(
        simulator=2688385916,
        scenario=3048105090,
        policy=4196366895,
    )


def test_derive_seeds_streams_are_distinct() -> None:
    result = derive_seeds(2024)
    assert len({result.simulator, result.scenario, result.policy}) == 3


def test_derive_seeds_varies_with_seed() -> None:
    assert derive_seeds(1) != derive_seeds(2)


def test_derive_seeds_values_are_in_uint32_range() -> None:
    result = derive_seeds(7)
    for value in (result.simulator, result.scenario, result.policy):
        assert 0 <= value < (1 << 32)


def test_derive_seeds_deterministic_and_distinct_over_many_seeds() -> None:
    for seed in range(300):
        first = derive_seeds(seed)
        assert derive_seeds(seed) == first  # deterministic
        assert len({first.simulator, first.scenario, first.policy}) == 3  # distinct


def test_seed_streams_is_frozen() -> None:
    streams = derive_seeds(99)
    with pytest.raises((AttributeError, TypeError)):
        streams.simulator = 0  # type: ignore[misc]


def test_seed_everything_runs_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mapping ``torch`` to None in sys.modules makes ``import torch`` raise
    # ImportError, exercising the lazy-import fallback path.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert seed_everything(123) is None
    assert seed_everything(123, deterministic_torch=True) is None


def test_seed_everything_without_torch_still_seeds_stdlib_rngs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import random

    import numpy as np

    monkeypatch.setitem(sys.modules, "torch", None)

    seed_everything(2024)
    py_a = random.random()
    np_a = np.random.rand()

    seed_everything(2024)
    py_b = random.random()
    np_b = np.random.rand()

    assert py_a == py_b
    assert np_a == np_b


def test_seed_everything_with_torch_is_deterministic() -> None:
    torch = pytest.importorskip("torch")

    seed_everything(2024)
    first = torch.rand(4)
    seed_everything(2024)
    second = torch.rand(4)

    assert torch.equal(first, second)


def test_seed_everything_deterministic_torch_flag_runs() -> None:
    pytest.importorskip("torch")
    # Should configure deterministic algorithms without raising.
    assert seed_everything(7, deterministic_torch=True) is None
