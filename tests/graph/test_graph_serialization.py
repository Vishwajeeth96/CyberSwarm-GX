"""Serialization round-trips (stage S3, DC-07g).

A ``GraphSnapshot`` and a ``GraphFeatureRecord`` survive a round trip through the
contract IO helpers unchanged: canonical JSON -> parse -> canonical JSON is a
fixed point, the parsed record equals the original, and the same holds through
``write_json``/``read_json`` and the JSON Lines writer/reader. This is what lets
the graph records be streamed to ``graph_snapshots.jsonl`` /
``graph_features.jsonl`` and read back for the dashboard and the baselines.
"""

from __future__ import annotations

from pathlib import Path

from cyberswarm.contracts import (
    GraphFeatureRecord,
    GraphSnapshot,
    JsonlWriter,
    dumps_canonical,
    loads_record,
    read_json,
    read_jsonl,
    write_json,
)


def test_snapshot_canonical_round_trip(snapshot_step5: GraphSnapshot) -> None:
    text = dumps_canonical(snapshot_step5)
    parsed = loads_record("GraphSnapshot", text)
    assert isinstance(parsed, GraphSnapshot)
    assert parsed == snapshot_step5
    # Canonical serialization is a fixed point.
    assert dumps_canonical(parsed) == text


def test_feature_record_canonical_round_trip(
    features_step5: GraphFeatureRecord,
) -> None:
    text = dumps_canonical(features_step5)
    parsed = loads_record("GraphFeatureRecord", text)
    assert isinstance(parsed, GraphFeatureRecord)
    assert parsed == features_step5
    assert dumps_canonical(parsed) == text


def test_snapshot_model_json_round_trip(snapshot_step5: GraphSnapshot) -> None:
    reparsed = GraphSnapshot.model_validate_json(snapshot_step5.model_dump_json())
    assert reparsed == snapshot_step5


def test_graph_records_round_trip_through_files(
    snapshot_step5: GraphSnapshot,
    features_step5: GraphFeatureRecord,
    tmp_path: Path,
) -> None:
    snap_path = tmp_path / "snapshot.json"
    feat_path = tmp_path / "features.json"
    write_json(snap_path, snapshot_step5)
    write_json(feat_path, features_step5)

    assert read_json(snap_path, GraphSnapshot) == snapshot_step5
    assert read_json(feat_path, GraphFeatureRecord) == features_step5


def test_snapshots_round_trip_through_jsonl(
    snapshot_step5: GraphSnapshot, tmp_path: Path
) -> None:
    path = tmp_path / "graph_snapshots.jsonl"
    with JsonlWriter(path) as writer:
        writer.write(snapshot_step5)
        writer.write(snapshot_step5)

    records = list(read_jsonl(path, GraphSnapshot))
    assert len(records) == 2
    assert all(record == snapshot_step5 for record in records)
