"""Phase ordering and step numbering of the recorded stream (task 7.1).

CONTRACTS section 5 fixes the per-step phases P0 (dynamics) -> P1 (observe) ->
P2 (propose) -> P3 (validate) -> P4 (execute) -> end, and rule 1 requires every
record produced during step ``t`` to carry ``step = t``. These tests drive the
engine and check both properties on the actual recorded order.
"""

from __future__ import annotations

from cyberswarm.contracts import EventType, Phase
from cyberswarm.engine.recorder import MemoryRecorder

from .conftest import SCENARIO_TRUNCATE, POLICY_ID, investigate_policy, run_episode

# Rank of each record within a step, in CONTRACTS section 5 phase order.
_P1_RECORDS = {
    "AgentObservation",
    "AvailableActions",
    "DetectionSummary",
    "NodeScoreRecord",
    "GraphSnapshot",
    "GraphFeatureRecord",
}


def _phase_rank(record) -> int:
    """Return the phase rank (0=P0 .. 5=end) of one recorded record."""

    name = type(record).__name__
    if name == "EventRecord":
        if record.event_type is EventType.episode_started:
            return 0
        if record.event_type is EventType.episode_ended:
            return 5
        if record.phase is Phase.dynamics:
            return 0
        if record.phase is Phase.outcome:
            return 4
        return 0
    if name in _P1_RECORDS:
        return 1
    if name == "ErrorRecord":
        return 2  # a missing-proposal substitution is part of P2
    if name == "PolicyProposal":
        return 2
    if name == "ValidationResult":
        return 3
    if name == "ExecutedAction":
        return 4
    if name == "GroundTruthSnapshot":
        return 5
    raise AssertionError(f"unexpected record type {name!r}")


def test_records_follow_phase_order_within_each_step(make_engine) -> None:
    recorder = MemoryRecorder()
    engine, _sim, _coord = make_engine(recorder=recorder)
    context = engine.reset(SCENARIO_TRUNCATE, seed=7, policy_id=POLICY_ID)
    run_episode(engine, context, investigate_policy)
    engine.close()

    keys = [(record.step, _phase_rank(record)) for record in recorder.log]
    assert keys, "the episode produced no records"
    # Non-decreasing in (step, phase_rank): phases never run out of order and a
    # later step's records never precede an earlier step's (CONTRACTS 5 rule 4).
    assert keys == sorted(keys)

    # The very first record is episode_started in P0 of step 0.
    first = recorder.log[0]
    assert type(first).__name__ == "EventRecord"
    assert first.event_type is EventType.episode_started
    assert first.step == 0


def test_every_step_has_the_full_phase_chain(make_engine) -> None:
    recorder = MemoryRecorder()
    engine, _sim, _coord = make_engine(recorder=recorder)
    context = engine.reset(SCENARIO_TRUNCATE, seed=7, policy_id=POLICY_ID)
    run_episode(engine, context, investigate_policy)
    engine.close()

    # For each acted step, P1..P4 records must all be present with that step.
    steps_with_proposals = {r.step for r in recorder.by_type("PolicyProposal")}
    for step in steps_with_proposals:
        for record_type in (
            "AgentObservation",
            "AvailableActions",
            "DetectionSummary",
            "NodeScoreRecord",
            "PolicyProposal",
            "ValidationResult",
            "ExecutedAction",
        ):
            on_step = [r for r in recorder.by_type(record_type) if r.step == step]
            assert on_step, f"step {step} is missing a {record_type}"


def test_step_numbers_match_the_acted_step(make_engine) -> None:
    recorder = MemoryRecorder()
    engine, _sim, _coord = make_engine(recorder=recorder)
    context = engine.reset(SCENARIO_TRUNCATE, seed=11, policy_id=POLICY_ID)

    acted_steps: list[int] = []
    while not context.final:
        step_t = context.step
        acted_steps.append(step_t)
        proposals = {
            agent_id: _proposal(agent_ctx)
            for agent_id, agent_ctx in context.agents.items()
        }
        outcome = engine.step(proposals)
        # Every validation/execution of this step carries step == step_t.
        assert all(v.step == step_t for v in outcome.validations)
        assert all(e.step == step_t for e in outcome.executions)
        context = outcome.context
    engine.close()

    # Steps are the contiguous range 0..H-1 for the truncating scenario.
    assert acted_steps == list(range(SCENARIO_TRUNCATE.max_steps))

    # The ground-truth stream has exactly one snapshot per step, in step order.
    gt_steps = [s.step for s in recorder.by_type("GroundTruthSnapshot")]
    assert gt_steps == acted_steps


def _proposal(agent_ctx):
    from .conftest import build_proposal

    return build_proposal(agent_ctx, investigate_policy(agent_ctx))
