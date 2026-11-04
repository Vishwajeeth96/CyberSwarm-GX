"""PettingZoo-style parallel environment over the engine (stage S3, DESIGN 4.7).

:class:`CyberSwarmParallelEnv` is a thin adapter that exposes
:class:`~cyberswarm.engine.EpisodeEngine` through the PettingZoo ``ParallelEnv``
API with Gymnasium spaces. The engine owns the step semantics (CONTRACTS section
5), the coordinator and the record stream; the environment only translates
integer action indices into proposals, encodes the agent-visible observations,
and surfaces the per-step team reward.

It never imports a simulator implementation. The simulator (and the coordinator,
ground-truth accessor and reward model) are injected through the
``engine_factory`` callable, which returns a fully wired
:class:`~cyberswarm.engine.EpisodeEngine`; the runner/factory supplies that
closure in later stages. The environment consumes only the public ``Simulator``
interface, and that only indirectly, through the engine.

Key behaviours (RL_IMPLEMENTATION section 2):

* ``reset(seed, options)`` requires an integer ``seed`` (reproducibility is
  mandatory); ``seed=None`` raises :class:`ValueError`. ``options["scenario_id"]``
  picks a scenario, otherwise one is drawn from the seed's scenario stream.
* ``step(actions)`` maps each agent to an integer index. An index outside
  ``[0, A_i)``, a missing agent key or an unknown agent key raises
  :class:`ValueError`. An in-range but masked index is substituted with
  ``no_op``, flagged ``info[agent]["invalid_action"] = True``, and charged by the
  reward model's ``invalid_action`` term; the coordinator never receives the
  masked proposal.
* The mask of a final observation (the one returned with ``terminated`` or
  ``truncated``) allows ``no_op`` only; it exists to bootstrap a truncated
  episode's value, not to drive a further decision.
* ``info[agent]`` carries only ``step``, ``episode_id``, ``proposal_id``,
  ``validation``, ``reason_codes`` and ``invalid_action`` - no ground truth, no
  reward breakdown, no termination reason and no future information.
* ``state()`` returns the ascending-agent-order concatenation of the agents'
  current observation vectors - agent-visible data only - and is the centralized
  critic hook of a later multi-agent algorithm. The per-agent critics never call
  it. It raises before ``reset`` and after the episode ended.

This module imports :mod:`cyberswarm.contracts`, :mod:`cyberswarm.interfaces`,
:mod:`cyberswarm.common`, :mod:`cyberswarm.engine`, the sibling action table and
encoder, :mod:`numpy`, :mod:`gymnasium` and :mod:`pettingzoo`; it imports no
simulator, no ground truth and no ``torch``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from ..common.seeding import derive_seeds
from ..contracts import (
    ActionKind,
    ActionSpec,
    AgentId,
    Decision,
    PolicyId,
    PolicyProposal,
    RationaleCode,
    Representation,
    ScenarioConfig,
)
from ..engine import EpisodeEngine, StepContext, StepOutcome
from ..interfaces import StepScore
from .action_table import ActionTable
from .encoder import ObservationEncoder

__all__ = ["CyberSwarmParallelEnv", "EnvStateError", "DEFAULT_ENV_SEED"]


#: Seed used to bootstrap topology discovery at construction. The first real
#: episode is only ever created by an explicit :meth:`CyberSwarmParallelEnv.reset`.
DEFAULT_ENV_SEED: int = 0

#: Default policy id stamped on the proposals the environment builds.
_DEFAULT_ENV_POLICY_ID: PolicyId = "pol-env-v0"


class EnvStateError(RuntimeError):
    """Raised when ``step`` or ``state`` is called out of the episode lifecycle."""


class CyberSwarmParallelEnv(ParallelEnv):
    """A PettingZoo ``ParallelEnv`` over :class:`EpisodeEngine` (DESIGN 4.7)."""

    metadata = {"name": "cyberswarm_parallel_v0", "render_modes": []}

    def __init__(
        self,
        engine_factory: Callable[[], EpisodeEngine],
        scenarios: Sequence[ScenarioConfig],
        *,
        observation_mode: Representation | str,
        policy_id: PolicyId = _DEFAULT_ENV_POLICY_ID,
    ) -> None:
        if not scenarios:
            raise ValueError("scenarios must be a non-empty sequence")
        self._engine_factory = engine_factory
        self._scenarios: tuple[ScenarioConfig, ...] = tuple(scenarios)
        self._mode = Representation(observation_mode)
        self._policy_id = policy_id
        self.render_mode = None

        # Build the engine once and reuse it: EpisodeEngine.reset fully
        # re-initialises the episode, so a single instance serves every episode.
        self._engine = engine_factory()

        # Discover the topology and agents by a throwaway bootstrap reset, then
        # require an explicit reset() before the first real episode.
        boot = self._engine.reset(
            self._scenarios[0], derive_seeds(DEFAULT_ENV_SEED).simulator, policy_id
        )
        self.possible_agents: list[AgentId] = sorted(boot.agents)
        topology = boot.agents[self.possible_agents[0]].topology
        self._tables: dict[AgentId, ActionTable] = {
            agent: ActionTable.build(topology, agent) for agent in self.possible_agents
        }
        self._encoders: dict[AgentId, ObservationEncoder] = {
            agent: ObservationEncoder(topology, agent, self._mode)
            for agent in self.possible_agents
        }
        self._observation_spaces: dict[AgentId, spaces.Dict] = {
            agent: spaces.Dict(
                {
                    "action_mask": spaces.MultiBinary(self._tables[agent].n),
                    "obs": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self._encoders[agent].dim,),
                        dtype=np.float32,
                    ),
                }
            )
            for agent in self.possible_agents
        }
        self._action_spaces: dict[AgentId, spaces.Discrete] = {
            agent: spaces.Discrete(self._tables[agent].n)
            for agent in self.possible_agents
        }

        self.agents: list[AgentId] = []
        self._started = False
        self._done = False
        self._context: StepContext | None = None
        self._obs_vectors: dict[AgentId, np.ndarray] = {}
        self._last_score: StepScore | None = None

    # ------------------------------------------------------------------ #
    # Spaces
    # ------------------------------------------------------------------ #
    def observation_space(self, agent: AgentId) -> spaces.Dict:
        """The ``Dict({action_mask, obs})`` space of one agent."""

        return self._observation_spaces[agent]

    def action_space(self, agent: AgentId) -> spaces.Discrete:
        """The ``Discrete(A_i)`` space of one agent."""

        return self._action_spaces[agent]

    def action_table(self, agent: AgentId) -> ActionTable:
        """The agent's :class:`ActionTable` (index <-> spec, mask, digest)."""

        return self._tables[agent]

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(
        self, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[dict[AgentId, dict[str, Any]], dict[AgentId, dict[str, Any]]]:
        """Start a new episode and return ``(observations, infos)``.

        ``seed`` must be a non-negative integer; ``None`` raises
        :class:`ValueError`. ``options["scenario_id"]`` selects a scenario,
        otherwise one is drawn deterministically from the seed.
        """

        effective_seed = self._require_seed(seed)
        streams = derive_seeds(effective_seed)
        scenario = self._choose_scenario(streams.scenario, options)

        context = self._engine.reset(scenario, streams.simulator, self._policy_id)
        self._context = context
        self._done = False
        self._started = True
        self.agents = list(self.possible_agents)
        self._last_score = None

        observations, infos = self._build_outputs(context, feedback=None, invalid=None)
        return observations, infos

    def step(
        self, actions: Mapping[AgentId, int]
    ) -> tuple[
        dict[AgentId, dict[str, Any]],
        dict[AgentId, float],
        dict[AgentId, bool],
        dict[AgentId, bool],
        dict[AgentId, dict[str, Any]],
    ]:
        """Advance one step and return the five PettingZoo parallel dicts."""

        if not self._started:
            raise EnvStateError("call reset() before step()")
        if self._done:
            raise EnvStateError("the episode is done; call reset() to start another")
        assert self._context is not None

        self._check_action_keys(actions)

        active_agents = list(self.agents)
        proposals: dict[AgentId, PolicyProposal] = {}
        invalid: dict[AgentId, bool] = {}
        invalid_counts: dict[AgentId, int] = {}
        for agent in active_agents:
            table = self._tables[agent]
            ctx = self._context.agents[agent]
            index = self._coerce_index(agent, actions[agent], table.n)
            mask = table.mask(ctx.available_actions)
            if not bool(mask[index]):
                index = 0  # substitute no_op for a masked-but-in-range index
                invalid[agent] = True
            else:
                invalid[agent] = False
            invalid_counts[agent] = 1 if invalid[agent] else 0
            proposals[agent] = self._build_proposal(ctx, table.spec_of(index))

        outcome: StepOutcome = self._engine.step(
            proposals, invalid_counts=invalid_counts
        )
        self._last_score = outcome.score
        self._context = outcome.context

        reward = float(outcome.score.total) if outcome.score is not None else 0.0
        rewards = {agent: reward for agent in active_agents}
        terminations = {agent: outcome.terminated for agent in active_agents}
        truncations = {agent: outcome.truncated for agent in active_agents}

        observations, infos = self._build_outputs(
            outcome.context, feedback=outcome.feedback, invalid=invalid
        )

        if outcome.done:
            self._done = True
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """Return the ascending-agent-order concatenation of the obs vectors.

        Shape ``(n_agents * D,)``, ``float32``, values in ``[0, 1]``. It holds
        only agent-visible data and raises before ``reset`` or after the episode
        ended.
        """

        if not self._started or self._done or not self._obs_vectors:
            raise EnvStateError(
                "state() is available only between reset() and the episode end"
            )
        ordered = [self._obs_vectors[agent] for agent in sorted(self._obs_vectors)]
        return np.concatenate(ordered).astype(np.float32)

    @property
    def last_reward_breakdown(self) -> StepScore | None:
        """The most recent step's reward score (for the training harness only).

        It is never placed in ``info`` and never read by a policy.
        """

        return self._last_score

    def render(self) -> None:
        """No-op: the environment has no visual rendering."""

        return None

    def close(self) -> None:
        """Close the underlying engine (and simulator). Idempotent."""

        self._engine.close()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _require_seed(seed: int | None) -> int:
        if seed is None:
            raise ValueError(
                "CyberSwarmParallelEnv.reset requires an integer seed "
                "(reproducibility is mandatory)"
            )
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"seed must be an int, got {type(seed).__name__}")
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        return seed

    def _choose_scenario(
        self, scenario_seed: int, options: Mapping[str, Any] | None
    ) -> ScenarioConfig:
        if options is not None and options.get("scenario_id") is not None:
            wanted = options["scenario_id"]
            for scenario in self._scenarios:
                if scenario.scenario_id == wanted:
                    return scenario
            raise ValueError(f"unknown scenario_id {wanted!r}")
        generator = np.random.default_rng(scenario_seed)
        return self._scenarios[int(generator.integers(len(self._scenarios)))]

    def _check_action_keys(self, actions: Mapping[AgentId, int]) -> None:
        provided = set(actions)
        expected = set(self.agents)
        missing = expected - provided
        if missing:
            raise ValueError(f"missing action for agents: {sorted(missing)}")
        unknown = provided - expected
        if unknown:
            raise ValueError(f"unknown agents in actions: {sorted(unknown)}")

    @staticmethod
    def _coerce_index(agent: AgentId, index: Any, n: int) -> int:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise ValueError(
                f"action for {agent!r} must be an integer index, "
                f"got {type(index).__name__}"
            )
        index = int(index)
        if not (0 <= index < n):
            raise ValueError(
                f"action index {index} for {agent!r} is out of range [0, {n})"
            )
        return index

    def _build_proposal(
        self, ctx: Any, spec: ActionSpec
    ) -> PolicyProposal:
        if spec.action_kind is ActionKind.no_op:
            rationale = [RationaleCode.no_action_needed]
        else:
            rationale = [RationaleCode.learned_policy]
        return PolicyProposal(
            proposal_id=ctx.next_proposal_id,
            episode_id=ctx.episode_id,
            step=ctx.step,
            agent_id=ctx.agent_id,
            policy_id=self._policy_id,
            observation_id=ctx.observation.observation_id,
            action=spec,
            rationale_codes=rationale,
        )

    def _build_outputs(
        self,
        context: StepContext,
        *,
        feedback: Mapping[AgentId, Sequence[Any]] | None,
        invalid: Mapping[AgentId, bool] | None,
    ) -> tuple[dict[AgentId, dict[str, Any]], dict[AgentId, dict[str, Any]]]:
        observations: dict[AgentId, dict[str, Any]] = {}
        infos: dict[AgentId, dict[str, Any]] = {}
        vectors: dict[AgentId, np.ndarray] = {}
        for agent in sorted(context.agents):
            ctx = context.agents[agent]
            table = self._tables[agent]
            if context.final:
                mask = np.zeros(table.n, dtype=np.int8)
                mask[0] = 1  # a final observation allows no_op only
            else:
                mask = table.mask(ctx.available_actions).astype(np.int8)
            last_rejected = any(
                item.decision is Decision.rejected
                for item in ctx.observation.last_step_feedback
            )
            vector = self._encoders[agent].encode(
                ctx, last_action_rejected=last_rejected
            )
            observations[agent] = {"action_mask": mask, "obs": vector}
            vectors[agent] = vector
            infos[agent] = self._info_for(agent, context, feedback, invalid)
        self._obs_vectors = vectors
        return observations, infos

    @staticmethod
    def _info_for(
        agent: AgentId,
        context: StepContext,
        feedback: Mapping[AgentId, Sequence[Any]] | None,
        invalid: Mapping[AgentId, bool] | None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "step": context.agents[agent].step,
            "episode_id": context.episode_id,
            "proposal_id": None,
            "validation": None,
            "reason_codes": [],
            "invalid_action": bool(invalid.get(agent, False)) if invalid else False,
        }
        if feedback:
            items = feedback.get(agent, ())
            if items:
                item = items[0]
                info["proposal_id"] = item.proposal_id
                info["validation"] = item.decision.value
                info["reason_codes"] = [code.value for code in item.reason_codes]
        return info
