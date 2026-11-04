"""Episode lifecycle: termination, the final context, and the scorer (task 7.1).

Checks the end-of-episode behaviour the engine owns: an episode that terminates
with ``objective_reached`` reports it through ``termination_reason``; the
post-terminal observation is returned in a ``final`` context but never recorded
(CONTRACTS 5 rule 8); and an attached :class:`~cyberswarm.interfaces.StepScorer`
is driven once per step with the end-of-step snapshot and the step's executions.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from cyberswarm.contracts import AgentObservation, GroundTruthSnapshot, read_jsonl
from cyberswarm.engine import EpisodeEngine
from cyberswarm.engine.recorder import EpisodeRecorder
from cyberswarm.interfaces import StepScore

from .conftest import (
    SCENARIO_OBJECTIVE,
    SCENARIO_TRUNCATE,
    POLICY_ID,
    build_proposal,
    investigate_policy,
    noop_policy,
    run_episode,
)


class RecordingScorer:
    """A :class:`StepScorer` that records its calls and returns a dummy score."""

    def __init__(self) -> None:
        self.reset_args: tuple | None = None
        self.calls: list[dict] = []

    def reset(self, topology, max_steps: int) -> None:
        self.reset_args = (topology.topology_id, max_steps)

    def score(
        self,
        *,
        step: int,
        prev,
        curr,
        executions: Sequence,
        invalid_count: int,
        termination_reason,
    ) -> StepScore:
        self.calls.append(
            {
                "step": step,
                "prev": prev,
                "curr": curr,
                "n_exec": len(executions),
                "invalid_count": invalid_count,
                "termination_reason": termination_reason,
            }
        )
        return StepScore(total=float(step), raw={}, weighted={})


def test_objective_reached_terminates_and_reports_reason(make_engine) -> None:
    engine, _sim, _coord = make_engine()
    context = engine.reset(SCENARIO_OBJECTIVE, seed=1, policy_id=POLICY_ID)
    outcomes, final = run_episode(engine, context, noop_policy)

    last = outcomes[-1]
    assert last.done and last.terminated and not last.truncated
    assert final.final
    assert engine.termination_reason == "objective_reached"
    # The objective (the entry node) is compromised by the end of step 1.
    assert [o.terminated for o in outcomes] == [False, True]
    engine.close()


def test_truncation_sets_truncated_not_terminated(make_engine) -> None:
    engine, _sim, _coord = make_engine()
    context = engine.reset(SCENARIO_TRUNCATE, seed=1, policy_id=POLICY_ID)
    outcomes, final = run_episode(engine, context, noop_policy)

    last = outcomes[-1]
    assert last.done and last.truncated and not last.terminated
    assert final.final
    assert engine.termination_reason == "max_steps"
    engine.close()


def test_final_observation_is_not_recorded(make_engine, tmp_path) -> None:
    episode_dir = tmp_path / "episodes" / "ep-engine-final"
    recorder = EpisodeRecorder(episode_dir)
    engine, _sim, _coord = make_engine(recorder=recorder)
    context = engine.reset(SCENARIO_OBJECTIVE, seed=1, policy_id=POLICY_ID)
    _outcomes, final = run_episode(engine, context, noop_policy)
    engine.close()

    # The episode terminated at step 1, so steps 0 and 1 are recorded.
    observed_steps = {
        obs.step
        for obs in read_jsonl(episode_dir / "observations.jsonl", AgentObservation)
    }
    assert observed_steps == {0, 1}

    # The final context exposes the post-terminal observation (step 2) that was
    # deliberately not written to any file.
    final_steps = {ctx.observation.step for ctx in final.agents.values()}
    assert final_steps == {2}
    gt_steps = {
        s.step
        for s in read_jsonl(
            episode_dir / "evaluator_only" / "ground_truth.jsonl",
            GroundTruthSnapshot,
        )
    }
    assert gt_steps == {0, 1}


def test_scorer_is_called_once_per_step(make_engine) -> None:
    scorer = RecordingScorer()
    engine, _sim, _coord = make_engine(scorer=scorer)
    context = engine.reset(SCENARIO_OBJECTIVE, seed=1, policy_id=POLICY_ID)
    outcomes, _final = run_episode(engine, context, investigate_policy)
    engine.close()

    assert scorer.reset_args == ("topo-mock-small-v1", SCENARIO_OBJECTIVE.max_steps)
    # One score per step, in order, each attached to its outcome.
    assert [c["step"] for c in scorer.calls] == list(range(len(outcomes)))
    assert [o.score.total for o in outcomes] == [float(i) for i in range(len(outcomes))]

    # The scorer receives the end-of-step snapshot for its step and the prior
    # snapshot as prev (None on the first step).
    assert scorer.calls[0]["prev"] is None
    assert scorer.calls[0]["curr"].step == 0
    assert scorer.calls[1]["prev"].step == 0
    assert scorer.calls[1]["curr"].step == 1
    # Both agents investigate every step, so two executions are reported.
    assert all(c["n_exec"] == 2 for c in scorer.calls)
    # The terminating step carries its reason through to the scorer.
    assert scorer.calls[-1]["termination_reason"] is not None


def test_scorer_without_truth_is_rejected(make_engine) -> None:
    with pytest.raises(ValueError):
        make_engine(scorer=RecordingScorer(), truth=False)
