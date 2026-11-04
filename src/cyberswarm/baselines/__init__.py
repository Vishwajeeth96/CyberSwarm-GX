"""Non-learned baseline decision policies (stage S4, tasks 8.1-8.2).

These policies implement the :class:`~cyberswarm.interfaces.Policy` Protocol
with hand-written rules rather than learning, so every learned arm has a
non-learned reference on the same scenarios and seeds (ARCHITECTURE section 6).

* :class:`RuleBasedPolicy` (task 8.1) reads the *conventional* agent-visible
  inputs only and never touches the graph channels.
* :class:`GraphFeaturePolicy` (task 8.2) reuses the exact same escalation ladder
  on the hand-engineered graph features, differing from the conventional
  baseline only in how it derives node suspicion and priority; it raises
  :class:`PolicyError` when the graph features are absent.

The package imports only ``contracts``, ``interfaces``, ``common`` and ``roles``
(DESIGN section 3.2); the graph baseline additionally reads the graph records
already present on the decision context.
"""

from __future__ import annotations

from .graph_rule import GraphFeaturePolicy, PolicyError
from .rule import RuleBasedPolicy

__all__ = [
    "RuleBasedPolicy",
    "GraphFeaturePolicy",
    "PolicyError",
]
