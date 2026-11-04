"""Reinforcement-learning layer (stage S3).

Exposes the configurable reward model (:mod:`.reward`), the per-agent action
table (:mod:`.action_table`), the observation encoder (:mod:`.encoder`) and the
PettingZoo-style environment (:mod:`.env`).

``import cyberswarm.rl`` needs ``numpy``, ``gymnasium`` and ``pettingzoo`` (the
environment's dependencies) but never ``torch``: the environment and the graph
pipeline it uses are numpy-only, and the learning code that needs ``torch``
lives in :mod:`cyberswarm.marl`, not here.
"""

from __future__ import annotations

from .action_table import ActionTable
from .encoder import ObservationEncoder
from .env import CyberSwarmParallelEnv, EnvStateError
from .reward import (
    RewardBreakdown,
    RewardConfig,
    RewardModel,
    RewardWeights,
    compute_breakdown,
    signed_weighted,
    weighted_total,
)

__all__ = [
    "ActionTable",
    "ObservationEncoder",
    "CyberSwarmParallelEnv",
    "EnvStateError",
    "RewardBreakdown",
    "RewardConfig",
    "RewardModel",
    "RewardWeights",
    "compute_breakdown",
    "signed_weighted",
    "weighted_total",
]
