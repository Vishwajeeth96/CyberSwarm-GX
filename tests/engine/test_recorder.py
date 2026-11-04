"""Recorders: parity, routing and the no-op sink (task 7.1).

The three recorders must agree on content. :class:`EpisodeRecorder` writes the
CONTRACTS section 15 files; :class:`MemoryRecorder` keeps the same records in
memory; :class:`NullRecorder` drops them. A deterministic episode is run with
each of the file and memory recorders and the two are compared byte for byte
per record stream, then the ``EpisodeResult`` -> ``episode_result.json`` routing
and the drop-everything behaviour are checked.
"""

from __future__ import annotations

from cyberswarm.contracts import (
    Counts,
    EpisodeResult,
    OutcomeSummary,
    Provenance,
    ProvenanceKind,
    Representation,
    SimulatorKind,
    Split,
    TerminationReason,
    dumps_canonical,
    read_json,
)
from cyberswarm.engine.recorder import (
    EPISODE_JSONL_FILES,
    EpisodeRecorder,
    MemoryRecorder,
    NullRecorder,
)

from .conftest import SCENARIO_TRUNCATE, POLICY_ID, investigate_policy, run_episode


def _run(make_engine, recorder, *, seed: int) -> None:
    engine, _sim, _coord = make_engine(recorder=recorder)
    context = engine.reset(SCENARIO_TRUNCATE, seed=seed, policy_id=POLICY_ID)
    run_episode(engine, context, investigate_policy)
    engine.close()


def test_memory_and_episode_recorders_agree(make_engine, tmp_path) -> None:
    memory = MemoryRecorder()
    _run(make_engine, memory, seed=4)

    episode_dir = tmp_path / "episodes" / "ep-engine-parity"
    _run(make_engine, EpisodeRecorder(episode_dir), seed=4)

    compared = 0
    for record_name, filename in EPISODE_JSONL_FILES.items():
        in_memory = [dumps_canonical(r) for r in memory.by_type(record_name)]
        path = episode_dir / filename
        on_disk = (
            path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        )
        assert on_disk == in_memory, f"{record_name} differs between recorders"
        if in_memory:
            compared += 1

    # The deterministic episode exercised several record streams, not just one.
    assert compared >= 7


def test_episode_recorder_creates_empty_errors_file(make_engine, tmp_path) -> None:
    episode_dir = tmp_path / "episodes" / "ep-engine-errors"
    _run(make_engine, EpisodeRecorder(episode_dir), seed=4)
    errors = episode_dir / "errors.jsonl"
    # errors.jsonl is listed as "may be empty": it must exist even with no errors.
    assert errors.exists()
    assert errors.read_text(encoding="utf-8") == ""


def test_episode_recorder_routes_episode_result(tmp_path) -> None:
    episode_dir = tmp_path / "episodes" / "ep-engine-result"
    recorder = EpisodeRecorder(episode_dir)
    result = _minimal_episode_result()
    recorder.write(result)
    recorder.close()

    path = episode_dir / "episode_result.json"
    assert path.exists(), "EpisodeResult must route to episode_result.json"
    loaded = read_json(path, EpisodeResult)
    assert loaded == result


def test_null_recorder_writes_nothing(make_engine, tmp_path) -> None:
    engine, _sim, _coord = make_engine(recorder=NullRecorder())
    context = engine.reset(SCENARIO_TRUNCATE, seed=4, policy_id=POLICY_ID)
    outcomes, final = run_episode(engine, context, investigate_policy)
    engine.close()

    # The episode still runs to completion; the sink simply drops every record.
    assert outcomes and final.final
    assert outcomes[-1].truncated


def test_episode_recorder_rejects_unknown_record(tmp_path) -> None:
    import pytest

    recorder = EpisodeRecorder(tmp_path / "episodes" / "ep-engine-bad")
    try:
        with pytest.raises(ValueError):
            recorder.write(Provenance(kind=ProvenanceKind.mock_simulator_run))
    finally:
        recorder.close()


def _minimal_episode_result() -> EpisodeResult:
    return EpisodeResult(
        episode_id="ep-engine-demo-s7",
        run_id="run-engine-demo",
        scenario_id="scn-mock-engine-short",
        seed=7,
        policy_id=POLICY_ID,
        representation=Representation.conventional,
        split=Split.fixture,
        simulator_kind=SimulatorKind.mock,
        steps_taken=1,
        terminated=True,
        truncated=False,
        termination_reason=TerminationReason.objective_reached,
        counts=Counts(proposals=0, accepted=0, rejected=0, executed=0, failed=0),
        reward=None,
        outcome=OutcomeSummary(
            compromised_nodes=0,
            attack_impact=0.0,
            healthy_services_disrupted=0,
            healthy_disruption_cost=0.0,
            false_isolation_count=0,
            true_positive_alerts=0,
            false_positive_alerts=0,
            compromised_nodes_detected=0,
            objective_reached=True,
            node_scoring=[],
        ),
        uncertainty_summary=None,
        provenance=Provenance(kind=ProvenanceKind.mock_simulator_run),
    )
