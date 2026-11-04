"""The ground-truth stream is written only under ``evaluator_only/`` (task 7.1).

AD-16 / CONTRACTS section 9: the engine is the one component that reads the
end-of-step ground-truth snapshot, and it writes that stream only to
``evaluator_only/ground_truth.jsonl``. No evaluator-only field may appear in any
agent-visible file. This test runs an episode through an :class:`EpisodeRecorder`
and checks both: the snapshot file exists with one record per step, and a scan
of every agent-visible file finds no ground-truth field (rule X-10).
"""

from __future__ import annotations

from cyberswarm.contracts import check_no_evaluator_fields, read_jsonl
from cyberswarm.contracts import GroundTruthSnapshot
from cyberswarm.engine.recorder import EpisodeRecorder, MemoryRecorder

from .conftest import SCENARIO_TRUNCATE, POLICY_ID, investigate_policy, run_episode


def _episode_dir(tmp_path):
    return tmp_path / "episodes" / "ep-engine-test"


def test_ground_truth_only_under_evaluator_only(make_engine, tmp_path) -> None:
    episode_dir = _episode_dir(tmp_path)
    recorder = EpisodeRecorder(episode_dir)
    engine, _sim, _coord = make_engine(recorder=recorder)
    context = engine.reset(SCENARIO_TRUNCATE, seed=9, policy_id=POLICY_ID)
    outcomes, _final = run_episode(engine, context, investigate_policy)
    engine.close()

    gt_path = episode_dir / "evaluator_only" / "ground_truth.jsonl"
    assert gt_path.exists(), "the ground-truth stream must be written"

    snapshots = list(read_jsonl(gt_path, GroundTruthSnapshot))
    # One snapshot per closed step, in ascending step order.
    assert [s.step for s in snapshots] == list(range(len(outcomes)))
    assert len(outcomes) == SCENARIO_TRUNCATE.max_steps

    # The only ground-truth file is the one under evaluator_only/.
    gt_files = [p for p in episode_dir.rglob("ground_truth.jsonl")]
    assert gt_files == [gt_path]


def test_no_evaluator_fields_leak_into_agent_visible_files(
    make_engine, tmp_path
) -> None:
    episode_dir = _episode_dir(tmp_path)
    recorder = EpisodeRecorder(episode_dir)
    engine, _sim, _coord = make_engine(recorder=recorder)
    context = engine.reset(SCENARIO_TRUNCATE, seed=9, policy_id=POLICY_ID)
    run_episode(engine, context, investigate_policy)
    engine.close()

    # Scan every produced file; check_no_evaluator_fields skips evaluator_only/.
    agent_visible = [
        p for p in episode_dir.rglob("*.jsonl") if "evaluator_only" not in p.parts
    ]
    assert agent_visible, "expected agent-visible record files"
    violations = check_no_evaluator_fields(agent_visible)
    assert violations == [], f"evaluator-only fields leaked: {violations}"


def test_ground_truth_snapshot_matches_the_closed_step(make_engine) -> None:
    recorder = MemoryRecorder()
    engine, _sim, _coord = make_engine(recorder=recorder)
    context = engine.reset(SCENARIO_TRUNCATE, seed=3, policy_id=POLICY_ID)

    # The snapshot the engine records for step t reports step == t (AD-16).
    while not context.final:
        step_t = context.step
        proposals = {
            agent_id: _proposal(agent_ctx)
            for agent_id, agent_ctx in context.agents.items()
        }
        outcome = engine.step(proposals)
        snapshots = recorder.by_type("GroundTruthSnapshot")
        assert snapshots[-1].step == step_t
        context = outcome.context
    engine.close()


def _proposal(agent_ctx):
    from .conftest import build_proposal

    return build_proposal(agent_ctx, investigate_policy(agent_ctx))
