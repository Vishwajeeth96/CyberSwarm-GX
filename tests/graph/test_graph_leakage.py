"""Leakage prevention (stage S3, DC-07j, DC-04).

The graph pipeline may read only the topology and the agent-visible observation
(whose events already satisfy ``step <= t``). These tests pin that boundary from
four angles: the public signatures expose no ground-truth parameter; the graph
source imports nothing from ``evaluator_only`` and never names
``GroundTruthSnapshot``; the serialized snapshot and feature records contain no
evaluator-only field name; and an event from a future step never contributes to
``max_event_step`` even when a caller passes it in explicitly.
"""

from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Callable
from pathlib import Path

from cyberswarm.contracts import (
    EVALUATOR_ONLY_FIELD_NAMES,
    AgentObservation,
    AlertCategory,
    EventProducer,
    EventRecord,
    EventType,
    GraphFeatureRecord,
    GraphSnapshot,
    NodeObservation,
    Phase,
    Severity,
    Topology,
    check_no_evaluator_fields,
    dumps_canonical,
    event_id,
)
import cyberswarm.graph as graph_pkg
from cyberswarm.graph import GraphConverter, compute_features, convert

# Parameter-name fragments that would betray a ground-truth input.
_FORBIDDEN_PARAM_FRAGMENTS = (
    "truth",
    "ground",
    "compromis",
    "evaluator",
    "availability",
    "oracle",
)


# --------------------------------------------------------------------------- #
# Signatures expose no ground-truth parameter (DC-04)
# --------------------------------------------------------------------------- #
def _assert_clean_signature(func) -> None:
    signature = inspect.signature(func)
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        lowered = name.lower()
        assert not any(fragment in lowered for fragment in _FORBIDDEN_PARAM_FRAGMENTS), (
            f"{func.__qualname__} parameter {name!r} looks like a ground-truth input"
        )
        annotation = str(parameter.annotation)
        assert "GroundTruth" not in annotation
        assert "evaluator_only" not in annotation


def test_public_signatures_have_no_ground_truth_parameter() -> None:
    _assert_clean_signature(convert)
    _assert_clean_signature(compute_features)
    _assert_clean_signature(GraphConverter.__init__)
    _assert_clean_signature(GraphConverter.convert)


def test_convert_parameters_are_only_observation_topology_events() -> None:
    assert list(inspect.signature(convert).parameters) == [
        "observation",
        "topology",
        "events",
    ]


def test_compute_features_parameters_are_only_snapshot_topology() -> None:
    assert list(inspect.signature(compute_features).parameters) == [
        "snapshot",
        "topology",
    ]


# --------------------------------------------------------------------------- #
# The graph source never imports or references evaluator-only data
# --------------------------------------------------------------------------- #
# The evaluator-only record symbols that must never appear in graph code. The
# boundary is checked over the AST (imports and name references), so the
# docstrings that *describe* the boundary ("takes no ground-truth parameter")
# are correctly ignored.
_EVALUATOR_SYMBOLS = frozenset(
    {"GroundTruthSnapshot", "GroundTruthNode", "NewCompromise", "AlertTruth"}
)


def test_graph_source_does_not_import_or_reference_ground_truth() -> None:
    package_dir = Path(graph_pkg.__file__).parent
    modules = sorted(package_dir.glob("*.py"))
    assert modules  # guard against an empty glob
    for module_path in modules:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "evaluator_only" not in (node.module or ""), module_path.name
                for alias in node.names:
                    assert alias.name not in _EVALUATOR_SYMBOLS, module_path.name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "evaluator_only" not in alias.name, module_path.name
            elif isinstance(node, ast.Name):
                assert node.id not in _EVALUATOR_SYMBOLS, module_path.name
            elif isinstance(node, ast.Attribute):
                assert node.attr != "evaluator_only", module_path.name
                assert node.attr not in _EVALUATOR_SYMBOLS, module_path.name


# --------------------------------------------------------------------------- #
# Serialized records carry no evaluator-only field name (X-10)
# --------------------------------------------------------------------------- #
def _keys(payload: object) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            found.add(key)
            found |= _keys(value)
    elif isinstance(payload, list):
        for item in payload:
            found |= _keys(item)
    return found


def test_serialized_records_contain_no_evaluator_fields(
    snapshot_step5: GraphSnapshot, features_step5: GraphFeatureRecord
) -> None:
    for record in (snapshot_step5, features_step5):
        keys = _keys(json.loads(dumps_canonical(record)))
        assert keys.isdisjoint(EVALUATOR_ONLY_FIELD_NAMES)
        # A couple of the most tempting fields, spelled out explicitly.
        assert "compromised" not in keys
        assert "defense_availability" not in keys


def test_written_graph_files_pass_the_leakage_scan(
    snapshot_step5: GraphSnapshot,
    features_step5: GraphFeatureRecord,
    tmp_path: Path,
) -> None:
    snap_path = tmp_path / "graph_snapshots.jsonl"
    feat_path = tmp_path / "graph_features.jsonl"
    snap_path.write_text(dumps_canonical(snapshot_step5) + "\n", encoding="utf-8")
    feat_path.write_text(dumps_canonical(features_step5) + "\n", encoding="utf-8")
    assert check_no_evaluator_fields([snap_path, feat_path]) == []


# --------------------------------------------------------------------------- #
# Future events never contribute to max_event_step (DC-07j)
# --------------------------------------------------------------------------- #
def _alert_event(episode_id: str, step: int, counter: int) -> EventRecord:
    return EventRecord(
        event_id=event_id(counter),
        episode_id=episode_id,
        step=step,
        phase=Phase.dynamics,
        event_type=EventType.alert_raised,
        producer=EventProducer.simulator,
        visible_to=["agent-dmz"],
        node_id="node-web-02",
        severity=Severity.high,
        alert_category=AlertCategory.traffic_anomaly,
    )


def test_future_event_is_excluded_from_max_event_step(
    obs_step5: AgentObservation, topology_v1: Topology
) -> None:
    future = _alert_event(obs_step5.episode_id, step=7, counter=99)
    snapshot = convert(obs_step5, topology_v1, events=[future])
    # The observation already carries a step-5 event; the step-7 event is dropped.
    assert snapshot.max_event_step == 5


def test_past_event_is_counted_but_future_is_not(
    make_node: Callable[..., NodeObservation],
    make_obs: Callable[..., AgentObservation],
    topology_v1: Topology,
) -> None:
    obs = make_obs([make_node("node-web-02", alert_count=1, severity=2)], step=5)
    assert not obs.recent_events  # the observation itself carries no events

    past = _alert_event(obs.episode_id, step=3, counter=3)
    future = _alert_event(obs.episode_id, step=9, counter=9)

    # A past event is consumed...
    assert convert(obs, topology_v1, events=[past]).max_event_step == 3
    # ...a future event is ignored (no events remain -> null)...
    assert convert(obs, topology_v1, events=[future]).max_event_step is None
    # ...and when both are supplied, only the past one counts.
    assert convert(obs, topology_v1, events=[past, future]).max_event_step == 3
