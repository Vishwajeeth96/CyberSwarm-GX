"""Deterministic, agent-visible logical roles (DESIGN section 4.4).

Pure functions over an :class:`~cyberswarm.contracts.AgentObservation`:

* :class:`DetectionRole` turns an observation into a
  :class:`~cyberswarm.contracts.DetectionSummary` (per-node suspicion and an
  ``alerted`` flag) using the CONTRACTS 8.4 formula.
* :func:`evidence_view` turns an observation into an
  :class:`~cyberswarm.interfaces.EvidenceView` (per-own-node evidence state and
  whether investigation is advisable).
* :class:`EvidenceConfig` is the single shared home for the ``e_min`` evidence
  threshold used by the baselines and the uncertainty rule.

The roles read only agent-visible fields; they never read ground truth and
import only :mod:`cyberswarm.contracts` and :mod:`cyberswarm.interfaces` (for
the shared :class:`EvidenceView`/:class:`NodeEvidence` boundary types).
"""

from __future__ import annotations

from .detection import DEFAULT_PRODUCER, DetectionRole, suspicion_score
from .evidence import DEFAULT_E_MIN, EvidenceConfig, evidence_view

__all__ = [
    "DetectionRole",
    "suspicion_score",
    "DEFAULT_PRODUCER",
    "evidence_view",
    "EvidenceConfig",
    "DEFAULT_E_MIN",
]
