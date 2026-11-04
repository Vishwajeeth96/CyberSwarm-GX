"""A missing proposal is substituted with a recorded no_op (task 7.1).

DESIGN section 4.6: when an agent returns no proposal for a step, the engine
substitutes a ``no_op`` proposal and emits an ``ErrorRecord(POLICY_FAILURE)``,
so the coordinator still sees one proposal per agent and the record chain stays
complete. These tests omit one agent's proposal and check the substitution and
that the resulting chain is still internally consistent.
"""

from __future__ import annotations

from cyberswarm.contracts import (
    ActionKind,
    Component,
    Decision,
    ErrorCode,
    check_episode,
)
from cyberswarm.engine.recorder import MemoryRecorder

from .conftest import (
    SCENARIO_TRUNCATE,
    POLICY_ID,
    build_proposal,
    investigate_policy,
    make_record_set,
    noop_policy,
)


def test_missing_proposal_is_substituted_with_noop(make_engine) -> None:
    recorder = MemoryRecorder()
    engine, _sim, _coord = make_engine(recorder=recorder)
    context = engine.reset(SCENARIO_TRUNCATE, seed=8, policy_id=POLICY_ID)

    agents = sorted(context.agents)
    omitted, kept = agents[0], agents[1]
    omitted_pid = context.agents[omitted].next_proposal_id

    # Provide a proposal only for the kept agent.
    proposals = {
        kept: build_proposal(context.agents[kept], investigate_policy(context.agents[kept]))
    }
    outcome = engine.step(proposals)
    engine.close()

    # An ErrorRecord(POLICY_FAILURE) was recorded for the omitted agent.
    errors = recorder.by_type("ErrorRecord")
    assert len(errors) == 1
    assert errors[0].code is ErrorCode.POLICY_FAILURE
    assert errors[0].component is Component.engine
    assert errors[0].proposal_id == omitted_pid
    assert errors[0].step == 0

    # The substituted proposal is a no_op recorded for the omitted agent.
    step0_proposals = {
        p.agent_id: p
        for p in recorder.by_type("PolicyProposal")
        if p.step == 0
    }
    assert step0_proposals[omitted].proposal_id == omitted_pid
    assert step0_proposals[omitted].action.action_kind is ActionKind.no_op

    # The omitted agent still receives accepted+executed feedback for its no_op.
    assert outcome.feedback[omitted][0].decision is Decision.accepted


def test_substituted_chain_stays_consistent(make_engine, topology_v1) -> None:
    recorder = MemoryRecorder()
    engine, _sim, _coord = make_engine(recorder=recorder)
    context = engine.reset(SCENARIO_TRUNCATE, seed=8, policy_id=POLICY_ID)

    kept = sorted(context.agents)[1]
    while not context.final:
        proposals = {
            kept: build_proposal(context.agents[kept], noop_policy(context.agents[kept]))
        }
        context = engine.step(proposals).context
    engine.close()

    # Every step substituted a no_op for the other agent; the chain still passes
    # the cross-record consistency rules X-1..X-16.
    issues = check_episode(make_record_set(recorder), topology_v1)
    assert issues == [], f"cross-record consistency issues: {issues}"
