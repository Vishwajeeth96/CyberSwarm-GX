"""Public simulator, policy and engine-port interfaces (stage S2).

This package is *neutral*: neither teammate edits it. It declares the Protocols
and the small record/dataclass types that sit on the boundaries between the
simulator, the decision policies, the coordinator and the episode engine, plus
the reusable :class:`SimulatorConformance` test mixin. It imports only
:mod:`cyberswarm.contracts` and the standard library (DESIGN section 3.2).
"""

from __future__ import annotations

from .simulator import (
    ActionApplication,
    GroundTruthAccessor,
    Simulator,
    SimulatorError,
    StepResult,
    ValidatedAction,
)
from .policy import (
    DecisionContext,
    DistributionPolicy,
    EvidenceView,
    NodeEvidence,
    Policy,
    PolicyDecision,
)
from .ports import (
    CoordinatorPort,
    CoordinatorResult,
    StepScore,
    StepScorer,
)
from .conformance import SimulatorConformance

__all__ = [
    # simulator
    "Simulator",
    "GroundTruthAccessor",
    "ValidatedAction",
    "ActionApplication",
    "StepResult",
    "SimulatorError",
    # policy
    "Policy",
    "DistributionPolicy",
    "DecisionContext",
    "PolicyDecision",
    "EvidenceView",
    "NodeEvidence",
    # ports
    "CoordinatorPort",
    "CoordinatorResult",
    "StepScorer",
    "StepScore",
    # conformance
    "SimulatorConformance",
]
