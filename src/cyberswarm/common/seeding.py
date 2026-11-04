"""Deterministic seed derivation and global seeding (stage S1).

:func:`derive_seeds` turns one integer master seed into three independent,
reproducible sub-stream seeds - one each for the simulator, the scenario and the
policy - using :class:`numpy.random.SeedSequence`, which is designed to spawn
statistically independent streams from a single seed.

:func:`seed_everything` seeds the Python :mod:`random` module, NumPy's legacy
global generator and - only if it is installed - PyTorch. ``torch`` is imported
lazily inside the function so that importing this module (and the whole
``cyberswarm.common`` package) never requires the optional ``rl`` dependency
group. This is the only module permitted to import ``torch``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

#: Modulus for NumPy's legacy ``RandomState`` seed, which must be in [0, 2**32).
_UINT32: int = 1 << 32


@dataclass(frozen=True, slots=True)
class SeedStreams:
    """Three independent sub-stream seeds derived from one master seed.

    Each field is a 32-bit integer in ``[0, 2**32)``. The values are
    reproducible for a fixed master seed and independent of one another.
    """

    simulator: int
    scenario: int
    policy: int


def derive_seeds(seed: int) -> SeedStreams:
    """Derive independent simulator, scenario and policy seeds from ``seed``.

    The three values are the first three 32-bit words produced by
    ``numpy.random.SeedSequence(seed).generate_state(3)``. For a fixed ``seed``
    the result is fully reproducible; the three streams are independent by the
    design of :class:`numpy.random.SeedSequence`.
    """

    state = np.random.SeedSequence(int(seed)).generate_state(3)
    return SeedStreams(
        simulator=int(state[0]),
        scenario=int(state[1]),
        policy=int(state[2]),
    )


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> None:
    """Seed every global RNG in use, importing ``torch`` lazily.

    Seeds Python's :mod:`random` module and NumPy's legacy global generator
    unconditionally. If PyTorch is installed it is seeded too; when
    ``deterministic_torch`` is true, deterministic algorithms and (when present)
    the cuDNN deterministic flags are requested as well. When PyTorch is not
    installed the function seeds the standard-library generators and returns
    without error - importing this module must never require ``torch``.
    """

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % _UINT32)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU in this environment
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        cudnn = getattr(torch.backends, "cudnn", None)
        if cudnn is not None:
            cudnn.deterministic = True
            cudnn.benchmark = False
