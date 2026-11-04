"""Deterministic detection role (DESIGN section 4.4, CONTRACTS section 8.4).

:class:`DetectionRole` is a pure function over an
:class:`~cyberswarm.contracts.AgentObservation`. It reads only agent-visible
fields (``alert_count_recent`` and ``max_alert_severity`` per node) and emits a
:class:`~cyberswarm.contracts.DetectionSummary` with one entry per observed
node, in the same order. It never reads ground truth and never guesses: an
observation that already validated as an :class:`AgentObservation` always
produces a contract-valid summary, and an invalid ``producer`` raises on
construction of the summary.

The per-node suspicion is the CONTRACTS 8.4 formula::

    alert_count_norm      = min(alert_count_recent, 5) / 5
    max_alert_severity_norm = max_alert_severity / 3
    suspicion = min(1, 0.5 * alert_count_norm + 0.5 * max_alert_severity_norm)

``alerted`` is ``True`` when the node has at least one alert in the observation
window (``alert_count_recent > 0``).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import (
    AgentObservation,
    DetectionNode,
    DetectionSummary,
    detection_id,
)

# Normalisation constants of the CONTRACTS 8.4 suspicion formula.
_ALERT_COUNT_CAP = 5
_SEVERITY_MAX = 3

DEFAULT_PRODUCER = "detection.rule-v1"


def suspicion_score(alert_count_recent: int, max_alert_severity: int) -> float:
    """Return the CONTRACTS 8.4 suspicion in ``[0.0, 1.0]`` (pure function)."""

    alert_count_norm = min(alert_count_recent, _ALERT_COUNT_CAP) / _ALERT_COUNT_CAP
    max_alert_severity_norm = max_alert_severity / _SEVERITY_MAX
    return min(1.0, 0.5 * alert_count_norm + 0.5 * max_alert_severity_norm)


@dataclass(frozen=True)
class DetectionRole:
    """Deterministic detection component (ARCHITECTURE section 6).

    ``producer`` names the component and version recorded on every summary (for
    example ``detection.rule-v1``). The role holds no mutable state, so a single
    instance can summarise any number of observations.
    """

    producer: str = DEFAULT_PRODUCER

    def summarize(self, obs: AgentObservation) -> DetectionSummary:
        """Summarise one agent observation into a :class:`DetectionSummary`.

        One :class:`~cyberswarm.contracts.DetectionNode` is produced per
        observed node, preserving the observation's ascending node order.
        """

        nodes = [
            DetectionNode(
                node_id=node.node_id,
                suspicion=suspicion_score(
                    node.alert_count_recent, node.max_alert_severity
                ),
                alerted=node.alert_count_recent > 0,
            )
            for node in obs.nodes
        ]
        return DetectionSummary(
            detection_id=detection_id(obs.step, obs.agent_id),
            episode_id=obs.episode_id,
            step=obs.step,
            agent_id=obs.agent_id,
            producer=self.producer,
            nodes=nodes,
        )
