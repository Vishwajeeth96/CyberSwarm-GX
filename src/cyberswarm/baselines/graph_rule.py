"""Non-neural graph-feature baseline policy (stage S4, task 8.2).

:class:`GraphFeaturePolicy` is the graph-feature sibling of
:class:`~cyberswarm.baselines.rule.RuleBasedPolicy`. It runs the *same*
deterministic escalation ladder - the investigate-before-isolate discipline,
the containment rungs and rule 2f (AD-22) all live in
:meth:`RuleBasedPolicy._node_candidate` and are inherited verbatim - but reads
its ranking signals from the hand-engineered graph features
(``graph-derived-v1``, CONTRACTS 10.2) that the engine assembles into
``ctx.graph_features``. The two baselines therefore differ *only* in how they
derive which node to act on, never in the safety rules, which is what makes the
conventional-versus-graph comparison (RL_IMPLEMENTATION comparison C-1) a
matched one.

Three things come from the graph features instead of the conventional inputs:

* **Suspicion** (node ranking). The detection suspicion is blended with
  neighbour alert aggregation: ``suspicion' = clamp(suspicion +
  neighbor_alert_weight * neighbor_alert_mean)``. A node whose neighbourhood is
  lighting up is ranked ahead of an equally-suspicious node in a quiet corner.
* **Priority** (suspicion-tie break / "which node first"). Instead of the
  node's own criticality the baseline uses ``dependents_crit_norm`` - how much
  critical service depends on the node - so an incident sitting on a load-bearing
  node is handled before an equally-suspicious leaf.
* **Fallback.** When no node warrants a ladder action, instead of an immediate
  ``no_op`` the baseline may ``monitor_node`` the own node closest to the
  alerted frontier (highest ``neighbor_alert_max`` with
  ``hop_to_alerted_norm`` within ``premonitor_hop_max``), a cheap pre-emptive
  look that the conventional baseline cannot make.

The escalation ladder's rung choice (investigate, rate-limit, quarantine,
isolate, restore) is inherited unchanged; the design's literal "rule 2e uses
``dependents_crit_norm``" wording is realised here as the node-priority signal
rather than as a different containment rung, because the committed ladder keys
the quarantine-versus-isolate rung on alert severity and the two baselines must
share that safety discipline exactly.

Because it cannot run without the features, :meth:`propose` raises
:class:`PolicyError` (``ErrorCode.POLICY_FAILURE``) when ``ctx.graph_features``
is ``None`` (ARCHITECTURE section 16 row 9; DESIGN section 4.9). It also exposes
the optional node-scoring interface: :meth:`score_nodes` emits a contract-valid
:class:`~cyberswarm.contracts.NodeScoreRecord` (scorer kind
``graph_feature_rule``, scorer id ``graph-rule-v1``) from the graph suspicion so
the evaluator's origin-ranking metric (AD-25) can consume it.

Imports stay within the baselines' boundary (DESIGN section 3.2): ``contracts``,
``interfaces``, ``common`` and ``roles`` only. The baseline reads the graph
*records* already present on the context; it never imports
:mod:`cyberswarm.graph` (the conversion that built them), ``rl``, ``marl`` or
``mock``.
"""

from __future__ import annotations

from ..contracts import (
    ActionKind,
    EpisodeId,
    ErrorCode,
    GraphFeatureRecord,
    NodeScore,
    NodeScoreRecord,
    PolicyId,
    PolicyProposal,
    RationaleCode,
    Representation,
    ScorerKind,
    node_score_id,
)
from ..contracts.graph import DERIVED_FEATURE_NAMES_V1
from ..interfaces import DecisionContext
from ..roles import EvidenceConfig
from .rule import (
    DEFAULT_E_CONTAIN,
    DEFAULT_HIGH_SEVERITY,
    DEFAULT_INVESTIGATE_COOLDOWN,
    RuleBasedPolicy,
    _AvailableIndex,
)

__all__ = [
    "DEFAULT_GRAPH_POLICY_ID",
    "DEFAULT_NEIGHBOR_ALERT_WEIGHT",
    "DEFAULT_PREMONITOR_NEIGHBOR_ALERT",
    "DEFAULT_PREMONITOR_HOP_MAX",
    "GRAPH_SCORER_ID",
    "GraphFeaturePolicy",
    "PolicyError",
]

#: Default policy id for the graph-feature baseline (RL_IMPLEMENTATION 9.1;
#: ``^pol-<slug>$``). Family ``fam-rule-graph``.
DEFAULT_GRAPH_POLICY_ID: PolicyId = "pol-rule-graph-v1"

#: Weight on ``neighbor_alert_mean`` added to the detection suspicion when
#: ranking nodes (DESIGN ``RuleConfig.neighbor_alert_weight``).
DEFAULT_NEIGHBOR_ALERT_WEIGHT: float = 0.5

#: Pre-emptive ``monitor_node`` fires only for an own node whose
#: ``neighbor_alert_max`` is at least this (DESIGN ``premonitor_neighbor_alert``).
DEFAULT_PREMONITOR_NEIGHBOR_ALERT: float = 0.2

#: ...and whose ``hop_to_alerted_norm`` is at most this - close to the alerted
#: frontier (DESIGN ``premonitor_hop_max``).
DEFAULT_PREMONITOR_HOP_MAX: float = 0.25

#: Scorer id/slug of the graph-feature suspicion scorer (AD-25; the detection
#: scorer is ``detection-rule-v1``). Used for both the ``scorer_id`` field and
#: the ``node_score_id`` slug, so it must be a valid id slug.
GRAPH_SCORER_ID: str = "graph-rule-v1"

# Column indices into the graph-derived-v1 feature rows, taken from the single
# registry so they cannot drift from the contract's column order.
_COL_DEPENDENTS_CRIT_NORM = DERIVED_FEATURE_NAMES_V1.index("dependents_crit_norm")
_COL_NEIGHBOR_ALERT_MEAN = DERIVED_FEATURE_NAMES_V1.index("neighbor_alert_mean")
_COL_NEIGHBOR_ALERT_MAX = DERIVED_FEATURE_NAMES_V1.index("neighbor_alert_max")
_COL_HOP_TO_ALERTED_NORM = DERIVED_FEATURE_NAMES_V1.index("hop_to_alerted_norm")


class PolicyError(Exception):
    """A decision policy could not run on the context it was given.

    Mirrors :class:`cyberswarm.interfaces.SimulatorError` and
    :class:`cyberswarm.graph.GraphConversionError`: a structured
    :class:`~cyberswarm.contracts.ErrorCode` (default
    :data:`~cyberswarm.contracts.ErrorCode.POLICY_FAILURE`) plus a
    human-readable message. :class:`GraphFeaturePolicy` raises it when a graph
    representation is configured but the context carries no graph features - a
    configuration bug, not a recoverable per-step condition.
    """

    code: ErrorCode

    def __init__(
        self, code: ErrorCode = ErrorCode.POLICY_FAILURE, message: str | None = None
    ) -> None:
        self.code = code
        super().__init__(message if message is not None else str(code.value))


def _clamp_unit(value: float) -> float:
    """Clamp to the ``[0, 1]`` suspicion/score domain."""

    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


class _GraphFeatureIndex:
    """Per-node lookup over a :class:`GraphFeatureRecord`'s derived columns.

    A masked entry (``mask == 0``, an imputed/unobserved value) or an unknown
    node reads as ``0.0`` - no information rather than a spurious signal.
    """

    def __init__(self, features: GraphFeatureRecord) -> None:
        self._row = {node_id: i for i, node_id in enumerate(features.node_order)}
        self._values = features.values
        self._mask = features.mask
        self.node_ids: tuple[str, ...] = tuple(features.node_order)

    def get(self, node_id: str, col: int) -> float:
        """Return feature column ``col`` for ``node_id`` (``0.0`` if masked)."""

        row = self._row.get(node_id)
        if row is None or self._mask[row][col] == 0:
            return 0.0
        return self._values[row][col]


class GraphFeaturePolicy(RuleBasedPolicy):
    """Deterministic graph-feature baseline over the hand-engineered features.

    Reuses :class:`RuleBasedPolicy`'s escalation ladder wholesale and overrides
    only the three feature-derivation hooks plus :meth:`propose` (to require the
    features). Construction mirrors the conventional baseline and adds the three
    graph knobs; the shared evidence thresholds are forwarded to the base.
    """

    representation: Representation = Representation.graph_features

    def __init__(
        self,
        *,
        policy_id: PolicyId = DEFAULT_GRAPH_POLICY_ID,
        evidence: EvidenceConfig | None = None,
        e_contain: int = DEFAULT_E_CONTAIN,
        investigate_cooldown: int = DEFAULT_INVESTIGATE_COOLDOWN,
        high_severity: int = DEFAULT_HIGH_SEVERITY,
        neighbor_alert_weight: float = DEFAULT_NEIGHBOR_ALERT_WEIGHT,
        premonitor_neighbor_alert: float = DEFAULT_PREMONITOR_NEIGHBOR_ALERT,
        premonitor_hop_max: float = DEFAULT_PREMONITOR_HOP_MAX,
        scorer_id: str = GRAPH_SCORER_ID,
    ) -> None:
        super().__init__(
            policy_id=policy_id,
            evidence=evidence,
            e_contain=e_contain,
            investigate_cooldown=investigate_cooldown,
            high_severity=high_severity,
        )
        if neighbor_alert_weight < 0.0:
            raise ValueError("neighbor_alert_weight must be non-negative")
        if not 0.0 <= premonitor_neighbor_alert <= 1.0:
            raise ValueError("premonitor_neighbor_alert must be in [0, 1]")
        if not 0.0 <= premonitor_hop_max <= 1.0:
            raise ValueError("premonitor_hop_max must be in [0, 1]")
        self._neighbor_alert_weight = neighbor_alert_weight
        self._premonitor_neighbor_alert = premonitor_neighbor_alert
        self._premonitor_hop_max = premonitor_hop_max
        self._scorer_id = scorer_id

    # ------------------------------------------------------------------ #
    # Policy Protocol
    # ------------------------------------------------------------------ #
    def propose(self, ctx: DecisionContext) -> PolicyProposal:
        """Require the graph features, then run the shared ladder.

        Raises :class:`PolicyError` (``POLICY_FAILURE``) when
        ``ctx.graph_features`` is ``None``: a graph-representation policy cannot
        act without the features it is defined over.
        """

        self._require_graph_features(ctx)
        return super().propose(ctx)

    # ------------------------------------------------------------------ #
    # Feature-derivation hooks (graph features in place of conventional inputs)
    # ------------------------------------------------------------------ #
    def _suspicion_by_node(self, ctx: DecisionContext) -> dict[str, float]:
        """Detection suspicion blended with neighbour alert aggregation."""

        features = self._require_graph_features(ctx)
        index = _GraphFeatureIndex(features)
        return {
            node.node_id: _clamp_unit(
                node.suspicion
                + self._neighbor_alert_weight
                * index.get(node.node_id, _COL_NEIGHBOR_ALERT_MEAN)
            )
            for node in ctx.detection.nodes
        }

    def _priority_by_node(self, ctx: DecisionContext) -> dict[str, float]:
        """Exposure of critical dependents (``dependents_crit_norm``)."""

        features = self._require_graph_features(ctx)
        index = _GraphFeatureIndex(features)
        return {
            node_id: index.get(node_id, _COL_DEPENDENTS_CRIT_NORM)
            for node_id in index.node_ids
        }

    def _fallback(
        self, ctx: DecisionContext, available: "_AvailableIndex"
    ) -> PolicyProposal:
        """Pre-emptively monitor the own node nearest the alerted frontier.

        Picks the own node with the highest ``neighbor_alert_max`` that clears
        ``premonitor_neighbor_alert`` and sits within ``premonitor_hop_max`` of
        an alerted node and for which ``monitor_node`` is available; ties break
        on the lowest ``node_id``. Falls back to the base ``no_op`` otherwise.
        """

        features = self._require_graph_features(ctx)
        index = _GraphFeatureIndex(features)
        best_node: str | None = None
        best_alert = -1.0
        for node in ctx.observation.nodes:  # ascending node_id (contract order)
            if node.scope.value != "own":
                continue
            node_id = node.node_id
            alert_max = index.get(node_id, _COL_NEIGHBOR_ALERT_MAX)
            hop = index.get(node_id, _COL_HOP_TO_ALERTED_NORM)
            if (
                alert_max >= self._premonitor_neighbor_alert
                and hop <= self._premonitor_hop_max
                and available.has(ActionKind.monitor_node, node_id)
                and alert_max > best_alert
            ):
                best_alert = alert_max
                best_node = node_id

        if best_node is not None:
            return self._build(
                ctx,
                available.spec(ActionKind.monitor_node, best_node),
                (RationaleCode.alerts_present,),
                f"monitor {best_node} near active alerts",
            )
        return super()._fallback(ctx, available)

    # ------------------------------------------------------------------ #
    # Optional node-scoring interface (AD-25; CONTRACTS section 8.8)
    # ------------------------------------------------------------------ #
    def score_nodes(self, ctx: DecisionContext) -> NodeScoreRecord:
        """Emit the graph suspicion as a contract-valid :class:`NodeScoreRecord`.

        Scores exactly the agent's observed nodes, in observation order, with
        the same graph suspicion that ranks candidates in :meth:`propose`. The
        engine (or a runner) may stream this to ``node_scores.jsonl`` for the
        evaluator's origin-ranking metric; the policy itself never reads truth.
        """

        suspicion = self._suspicion_by_node(ctx)
        return NodeScoreRecord(
            node_score_id=node_score_id(ctx.step, ctx.agent_id, self._scorer_id),
            episode_id=ctx.episode_id,
            step=ctx.step,
            agent_id=ctx.agent_id,
            scorer_id=self._scorer_id,
            scorer_kind=ScorerKind.graph_feature_rule,
            nodes=[
                NodeScore(node_id=node.node_id, score=suspicion.get(node.node_id, 0.0))
                for node in ctx.observation.nodes
            ],
        )

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _require_graph_features(self, ctx: DecisionContext) -> GraphFeatureRecord:
        """Return ``ctx.graph_features`` or raise :class:`PolicyError`."""

        features = ctx.graph_features
        if features is None:
            raise PolicyError(
                ErrorCode.POLICY_FAILURE,
                f"{self.policy_id} requires graph features, but ctx.graph_features is None",
            )
        return features
