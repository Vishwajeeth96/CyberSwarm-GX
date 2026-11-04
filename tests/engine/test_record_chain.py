"""The full per-step record chain is internally consistent (task 7.1).

The engine produces the whole decision-loop chain for each step: events,
observations, available actions, detection, node scores, graph snapshot and
features, proposals, validations and executions. This test runs a few steps on
the MOCK and checks the cross-record rules X-1..X-16 with the contracts'
:func:`~cyberswarm.contracts.check_episode`, which is the authoritative check of
id resolution, the proposal/validation/execution chain, outcome-event linkage
and the detection node-score equality.
"""

from __future__ import annotations

from cyberswarm.contracts import Decision, check_episode
from cyberswarm.engine.recorder import MemoryRecorder

from .conftest import (
    SCENARIO_TRUNCATE,
    POLICY_ID,
    investigate_policy,
    make_record_set,
    run_episode,
)


def test_record_chain_is_consistent(make_engine, topology_v1) -> None:
    recorder = MemoryRecorder()
    engine, _sim, _coord = make_engine(recorder=recorder)
    context = engine.reset(SCENARIO_TRUNCATE, seed=5, policy_id=POLICY_ID)
    run_episode(engine, context, investigate_policy)
    engine.close()

    issues = check_episode(make_record_set(recorder), topology_v1)
    assert issues == [], f"cross-record consistency issues: {issues}"


def test_chain_links_proposals_validations_executions(make_engine) -> None:
    recorder = MemoryRecorder()
    engine, _sim, _coord = make_engine(recorder=recorder)
    context = engine.reset(SCENARIO_TRUNCATE, seed=5, policy_id=POLICY_ID)
    run_episode(engine, context, investigate_policy)
    engine.close()

    proposals = recorder.by_type("PolicyProposal")
    validations = recorder.by_type("ValidationResult")
    executions = recorder.by_type("ExecutedAction")

    # Accept-all: one validation and one execution per proposal, same ids.
    assert len(validations) == len(proposals)
    assert len(executions) == len(proposals)

    proposal_ids = {p.proposal_id for p in proposals}
    assert {v.proposal_id for v in validations} == proposal_ids
    assert {e.proposal_id for e in executions} == proposal_ids
    assert all(v.decision is Decision.accepted for v in validations)

    # Each execution's action_applied outcome event is linked back by id (X-8).
    events = {e.event_id: e for e in recorder.by_type("EventRecord")}
    for execution in executions:
        assert execution.outcome_event_ids, "an executed action must emit events"
        for event_id in execution.outcome_event_ids:
            assert event_id in events, f"dangling outcome event {event_id!r}"


def test_proposal_ids_are_contiguous_in_agent_order(make_engine) -> None:
    recorder = MemoryRecorder()
    engine, sim, _coord = make_engine(recorder=recorder)
    context = engine.reset(SCENARIO_TRUNCATE, seed=5, policy_id=POLICY_ID)
    run_episode(engine, context, investigate_policy)
    engine.close()

    proposals = recorder.by_type("PolicyProposal")
    # The engine assigns proposal ids as a per-episode counter in ascending
    # agent order; with two agents the ids run 1, 2, 3, ... contiguously.
    counters = [int(p.proposal_id.split("-")[1]) for p in proposals]
    assert counters == list(range(1, len(proposals) + 1))

    agents = sorted(sim.agent_ids)
    # Within each step the proposals are in ascending agent order.
    by_step: dict[int, list[str]] = {}
    for proposal in proposals:
        by_step.setdefault(proposal.step, []).append(proposal.agent_id)
    for step_agents in by_step.values():
        assert step_agents == agents
