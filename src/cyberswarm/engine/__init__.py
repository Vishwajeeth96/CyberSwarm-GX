"""Episode engine and record sinks (stage S3, DESIGN section 4.6).

The engine turns a :class:`~cyberswarm.interfaces.Simulator` and a
:class:`~cyberswarm.interfaces.CoordinatorPort` into a recorded, phase-ordered
episode. It is the one place that reads evaluator-only ground truth after a step
closes (AD-16), and it attaches each agent's previous-step proposal feedback to
its next observation. The recorders decide where the produced records go: to the
CONTRACTS section 15 files (:class:`EpisodeRecorder`), to memory
(:class:`MemoryRecorder`), or nowhere (:class:`NullRecorder`, for training).

This package imports only ``contracts``, ``interfaces``, ``common``, ``roles``
and ``graph`` (DESIGN section 3.2); the coordinator and reward model arrive as
Protocols, never as concrete imports.
"""

from __future__ import annotations

from .recorder import (
    EPISODE_JSON_FILES,
    EPISODE_JSONL_FILES,
    EpisodeRecorder,
    MemoryRecorder,
    NullRecorder,
    Recorder,
)
from .step_engine import (
    DETECTION_SCORER_ID,
    EpisodeEngine,
    StepContext,
    StepOutcome,
)

__all__ = [
    "EpisodeEngine",
    "StepContext",
    "StepOutcome",
    "DETECTION_SCORER_ID",
    "Recorder",
    "EpisodeRecorder",
    "MemoryRecorder",
    "NullRecorder",
    "EPISODE_JSONL_FILES",
    "EPISODE_JSON_FILES",
]
