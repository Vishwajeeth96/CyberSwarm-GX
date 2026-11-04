"""Previous-step proposal feedback is attached to observations (task 7.1).

The MOCK leaves ``AgentObservation.last_step_feedback`` empty (CONTRACTS 5.2);
the engine fills it from the coordinator's decisions of the previous step. This
test checks that step 0 carries no feedback, that each step's
:class:`~cyberswarm.engine.StepOutcome` reports the agent's own proposal outcome,
and that the feedback reappears on the next step's observation (both in the
returned context and in the recorded observation).
"""

from __future__ import annotations

from cyberswarm.contracts import Decision, ExecutionStatus
from cyberswarm.engine.recorder import MemoryRecorder

from .conftest import (
    SCENARIO_TRUNCATE,
    POLICY_ID,
    build_proposal,
    noop_policy,
)


def test_step_zero_has_no_feedback(make_engine) -> None:
    engine, _sim, _coord = make_engine()
    context = engine.reset(SCENARIO_TRUNCATE, seed=2, policy_id=POLICY_ID)
    for agent_ctx in context.agents.values():
        assert agent_ctx.observation.last_step_feedback == []
    engine.close()


def test_feedback_describes_each_agents_proposal(make_engine) -> None:
    recorder = MemoryRecorder()
    engine, _sim, _coord = make_engine(recorder=recorder)
    context0 = engine.reset(SCENARIO_TRUNCATE, seed=2, policy_id=POLICY_ID)

    # Remember the proposal id the engine assigned each agent for step 0.
    step0_proposal_ids = {
        agent_id: agent_ctx.next_proposal_id
        for agent_id, agent_ctx in context0.agents.items()
    }

    proposals = {
        agent_id: build_proposal(agent_ctx, noop_policy(agent_ctx))
        for agent_id, agent_ctx in context0.agents.items()
    }
    outcome0 = engine.step(proposals)

    # The outcome reports each agent's own accepted+executed proposal.
    assert set(outcome0.feedback) == set(step0_proposal_ids)
    for agent_id, feedback in outcome0.feedback.items():
        assert len(feedback) == 1
        fb = feedback[0]
        assert fb.proposal_id == step0_proposal_ids[agent_id]
        assert fb.decision is Decision.accepted
        assert fb.execution_status is ExecutionStatus.executed

    # The next step's observation carries exactly that feedback.
    context1 = outcome0.context
    assert not context1.final
    for agent_id, agent_ctx in context1.agents.items():
        attached = agent_ctx.observation.last_step_feedback
        assert len(attached) == 1
        assert attached[0].proposal_id == step0_proposal_ids[agent_id]
        assert attached[0].decision is Decision.accepted

    engine.close()

    # The recorded step-1 observations also carry the feedback.
    step1_observations = [
        obs for obs in recorder.by_type("AgentObservation") if obs.step == 1
    ]
    assert step1_observations
    for obs in step1_observations:
        assert len(obs.last_step_feedback) == 1
        assert obs.last_step_feedback[0].proposal_id == step0_proposal_ids[obs.agent_id]
