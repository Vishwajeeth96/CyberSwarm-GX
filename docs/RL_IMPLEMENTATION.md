# CyberSwarm Reinforcement-Learning Design

**Version:** 0.2 (design, revised after the first design review). **Status:** SPECIFIED, NOT IMPLEMENTED. Stages S3 (environment, reward), S5 (IPPO, then the uncertainty mechanism), S6 (evaluation), S7 (learned-graph stage) and S8 (MAPPO stage) of `.agents/tasks/cyberswarm-foundation/DESIGN.md` implement this document.
**Companion documents:** [ARCHITECTURE.md](ARCHITECTURE.md), [CONTRACTS.md](CONTRACTS.md), [DECISIONS.md](DECISIONS.md), [RESEARCH.md](RESEARCH.md).
**Owner:** project lead. Nothing in this document is delegated to Raksha or Mayukh Aryan.

Nothing here is a result. Numbers named as "defaults" are starting points chosen for stated reasons before any experiment. They are not tuned to make any method win. Results will be reported only from evaluator output on matched scenarios and seeds. Mock-trained numbers are never scientific evidence.

## 1. Scope

| Item | Decision |
|------|----------|
| Learning problem | Cooperative MARL. Two zone agents (`agent-dmz`, `agent-core` on the mock topology) defend one network with a shared team reward and partial observability. |
| Algorithms | Rule-based baseline, graph-feature rule baseline (S4), IPPO (S5), uncertainty-aware wrapper with a matched control (S5, after IPPO), MAPPO (planned stage S8, section 6; implemented unless the stage finds an evidenced blocker). |
| Library | Compact self-written PyTorch PPO (section 5.1). |
| Representations | `conventional` and `graph_features` (section 3), plus `learned_embedding` only if the learned-graph stage runs (section 8). Same policy architecture and training budget in every arm. |
| Learned GNN | Staged. Specified now (task, models, inputs, go criteria, matched ablation: section 8); the go or no-go is taken in stage S7, after IPPO, the uncertainty extension and the evaluation path are stable. |
| Where uncertainty acts | Only in the decision rule (section 7). It does not enter the reward. |
| Evaluation splits | `train`, `cal`, `eval` (held-out scenarios on the training topology) and `eval_utopo` (held-out scenarios on an unseen topology variant). Results are reported separately for `eval` and `eval_utopo` (section 9). |

## 2. Environment

### 2.1 Class and construction

`cyberswarm.rl.env.CyberSwarmParallelEnv` implements the PettingZoo `ParallelEnv` API with Gymnasium spaces. It is a thin adapter over `cyberswarm.engine.EpisodeEngine`, which owns step semantics (CONTRACTS section 5), the coordinator, and the record stream. The environment imports only the public `Simulator` Protocol through the engine. It never imports a simulator implementation.

Construction arguments: an engine factory (simulator factory, coordinator configuration), a list of `ScenarioConfig` objects, `observation_mode` in `conventional`, `graph_features`, a `RewardModel` or `None`, and the horizon taken from the scenario.

`possible_agents` is the ascending list of agent ids. All agents act at every step and end together, so `agents` is either the full list or empty.

### 2.2 Lifecycle

| Call | Behavior |
|------|----------|
| `reset(seed, options)` | `seed` is an int >= 0 or `None`. `None` means: if the environment was reset before, draw the seed from the environment's own `numpy` generator (seeded by the first reset; the Gymnasium convention); if it was never reset, use `DEFAULT_ENV_SEED = 0`. So `pettingzoo.test.parallel_api_test`, which calls `reset()` without a seed, works and stays deterministic. `options["scenario_id"]` picks a scenario; when absent the scenario is `scenarios[k]` where `k` comes from a generator seeded by the scenario-choice seed stream of the effective seed. Calls `engine.reset`, which resets the simulator, runs P0 and P1 of step 0, and returns `(observations, infos)`. |
| `step(actions)` | `actions` maps every agent id to an integer index. The environment converts each index into a `PolicyProposal` (default parameters), runs P2 to P4 through the coordinator, calls `end_step`, builds the next observations (P0 and P1 of step `t+1`), and returns `(observations, rewards, terminations, truncations, infos)`. |
| After the end | When `terminations` or `truncations` is true for the agents, `agents` becomes empty and the last observations are the final observations. A further `step` raises `EnvStateError`. |
| `close()` | Closes the simulator. Idempotent. |

`state()` is provided from stage S3: it returns the concatenation, in ascending agent order, of the agents' current `obs` vectors (shape `(n_agents * D,)`, float32, values in [0, 1]). It contains only what the actors receive, which is agent-visible data (no ground truth, no future events, no reward breakdown). It raises `EnvStateError` before `reset` and after the episode ended. It is the hook the MAPPO critic uses (section 6.2); the IPPO critics never call it. It is built in S3 because the environment is lead-owned and the hook is a missing-API risk only if it is left out.

### 2.3 Observation space

For each agent: `Dict({"action_mask": MultiBinary(A_i), "obs": Box(0.0, 1.0, (D,), float32)})`. The vector layout and the dimension `D` depend on the observation mode (section 3). Every value is finite and in [0, 1].

### 2.4 Action space, masks and invalid actions

Each agent has `Discrete(A_i)`. The action table for agent `i` is built once from the topology:

- Index 0 is `no_op`.
- Then, for each of the seven targeted kinds in the fixed order of CONTRACTS section 6 (`monitor_node`, `investigate_node`, `request_evidence`, `rate_limit_node`, `quarantine_node`, `isolate_node`, `restore_node`), one index per node of the agent's zone, ascending `node_id`.
- Parameters are defaults (`duration_steps` = 3, `level` = `light`). The RL action space does not expose non-default parameters.

On the mock topology `A_dmz = 1 + 7 x 3 = 22` and `A_core = 1 + 7 x 5 = 36`. `index_of(ActionSpec)` and `spec_of(index)` are inverse functions, tested.

The mask for step `t` is derived from the simulator's `AvailableActions` for that agent, so it equals what the coordinator will accept for state-dependent availability. `no_op` is always unmasked, so a valid action always exists. The mask of a **final observation** (the one returned with `terminated` or `truncated`) allows `no_op` only, because no further decision is taken from it; it exists to bootstrap the value of a truncated episode.

Invalid-action handling (explicit, tested):

1. A well-behaved policy samples only unmasked actions (masked logits).
2. If an index is in range but masked, the environment substitutes `no_op`, records `info[agent]["invalid_action"] = True`, and the reward model charges the `invalid_action` term. The coordinator never receives the masked proposal.
3. An index outside `[0, A_i)`, a missing agent key or an unknown agent key raises `ValueError`.
4. Independently of the mask, the coordinator still validates every executed proposal. A proposal that passes the mask but fails a coordinator check (for example `CONFLICTING_PROPOSAL`, `HIGH_IMPACT_BUDGET_EXCEEDED`) is rejected there, charged nothing by the environment, and reported to the agent through `last_step_feedback`. The agent's action then has no effect for that step.

### 2.5 What each logical role can see

| Role or component | Sees | Never sees |
|-------------------|------|------------|
| Detection role | The agent's `AgentObservation` | Other agents' observations, ground truth |
| Evidence view | The agent's `AgentObservation` | Same |
| Decision policy of `agent-dmz` | Its own `DecisionContext`: observation, available actions, detection summary, and in graph mode its graph snapshot and features. Own-zone nodes with full per-node signals, one-hop out-of-zone neighbours with alert and status signals only. | The other agent's observation, proposals or actions (there is no message channel in version 1), ground truth, future events, the reward breakdown |
| Decision policy of `agent-core` | Its own context, symmetric | Same |
| Coordinator | All proposals of the step, the topology, `AvailableActions`, simulator state through the public API | Ground truth |
| Reward model, evaluator | `GroundTruthSnapshot` and all records | They produce signals and metrics; no policy reads their output |

The agents therefore coordinate only through the shared reward, the shared environment, and the coordinator's conflict rules.

### 2.6 Termination versus truncation

- `terminated = True` for all agents at the end of step `t` when the objective node is compromised (`objective_reached`) or the incident has started and every node is `up` and none is compromised (`incident_resolved`).
- `truncated = True` for all agents at the end of step `H-1` if not terminated (`max_steps`).
- For PPO, a truncated episode bootstraps the return with `V(final observation)`; a terminated episode uses 0. The environment returns the final observation in both cases.
- The environment never exposes the termination reason.

### 2.7 Info dictionary

`info[agent]` contains only: `step` (int), `episode_id` (str), `proposal_id` (str or None), `validation` (`accepted`, `rejected` or None), `reason_codes` (list of str), `invalid_action` (bool). It contains no ground truth, no reward breakdown, no termination reason and no future information. A test serializes every `info` and observation to JSON and scans for evaluator-only field names.

### 2.8 Seeds and determinism

`common.seeding.derive_seeds(seed)` produces three independent integer streams from `numpy.random.SeedSequence(seed)`: simulator, scenario choice and policy sampling.

- `reset(seed=s)` passes the simulator stream to `Simulator.reset`.
- Policy sampling uses a `torch.Generator` seeded from the policy stream. The uncertainty wrapper draws no random numbers.
- Training takes a master seed. It sets Python, NumPy and PyTorch seeds, uses `torch.use_deterministic_algorithms(True)` and one CPU thread, and derives each agent's initialization seed from the master seed and the agent index.
- Contract: same (scenario, seed, policy, checkpoint) gives identical record files and identical `state_digest` sequences. Two different training seeds give different checkpoints.

## 3. Observation modes (matched comparison)

Both modes encode the same agent-visible information in the same node-slot layout; the graph mode appends eight hand-engineered graph features. Policies in the two modes use the same network shape except for the input width.

Node slots follow the canonical node order over all `N` topology nodes. Per-node block, 17 values (`conventional`):

| Idx | Name | Source |
|-----|------|--------|
| 0 | `criticality` | topology (public) |
| 1 to 5 | `status_up`, `status_rate_limited`, `status_quarantined`, `status_isolated`, `status_restoring` | observation |
| 6 | `alert_count_norm` | observation |
| 7 | `max_alert_severity_norm` | observation |
| 8 | `evidence_level_norm` | observation, 0 if null |
| 9 | `service_health` | observation, 0 if null |
| 10 | `steps_since_investigated_norm` | observation, 0 if null |
| 11 | `suspicion` | `DetectionSummary` |
| 12 | `observed` | 1 if the node is in the observation |
| 13 | `own_scope` | 1 if `scope = own` |
| 14 to 16 | `evidence_missing`, `service_health_missing`, `investigated_missing` | 1 if the node is observed and the corresponding value is null |

Indices 0 to 10 are exactly the `graph-base-v1` node feature columns, taken from the `GraphSnapshot` in both modes so that there is one conversion path.

Graph mode adds a block of 8 values per node (the `graph-derived-v1` columns, CONTRACTS 10.2) after the base block. The vector ends with two global values: `step_norm = t / H` and `last_action_rejected` (1 if any proposal of this agent was rejected in the previous step).

| Mode | Dimension for N nodes | Mock (N = 8) |
|------|-----------------------|--------------|
| `conventional` | `17 N + 2` | 138 |
| `graph_features` | `25 N + 2` | 202 |

Unobserved nodes use the `impute-v1` values (CONTRACTS 10.1): zeros, with `criticality` kept and `observed = 0`.

A third mode, `learned_embedding`, exists only if the learned-graph stage (section 8) runs Task B: the eight derived columns are replaced by eight embedding columns of a frozen model, giving the same dimension `25 N + 2` and the same policy architecture.

What the graph mode can add: structure that the conventional vector lacks, namely neighbour alert aggregation, dependency exposure, hop distance to alerts and blocked-edge fraction. Whether this helps is the hypothesis H-E8 (RESEARCH section 3); the direction is not assumed.

## 4. Reward

### 4.1 Definition

The reward is a training and evaluation signal computed after P4 of step `t` by `cyberswarm.rl.reward.RewardModel` from the end-of-step ground truth of steps `t-1` and `t` (the `t-1` snapshot is empty for `t = 0`) and the actions executed in step `t`. It is a scalar per step, shared by all agents (`sharing = "team"`). It is not an observation. `sharing = "zone_local"` (each agent receives the terms attributed to its zone) is a configuration option for ablations; it is not run by default.

Notation. `C_t`: truly compromised nodes at the end of step `t`. `K_t`: contained nodes (`quarantined`, `isolated`, `restoring`). `crit(n)`: criticality. `avail_t(n)`: true service availability in [0, 1], including the compromise penalty and dependency effects (`GroundTruthSnapshot.service_availability`). `davail_t(n)`: defense availability, the same quantity with the compromise factor fixed at 1.0, so it reflects only the status the defender gave the node and the status of the nodes it depends on (`GroundTruthSnapshot.defense_availability`). `H`: horizon. `d_t`: `t - t_first`, where `t_first` is the step of the first true compromise, counted only while `C_t` contains a node outside `K_t`; otherwise 0.

The reward is `r_t = sum_k s_k * w_k * m_k` with sign `s_k`, weight `w_k >= 0` and raw magnitude `m_k >= 0`:

| Term | Sign | Raw magnitude `m_k` at step `t` | Default weight | Scale and intent |
|------|------|--------------------------------|----------------|------------------|
| `attack_impact` | negative | `sum over n in C_t of crit(n)`. If the episode terminates with `objective_reached` at step `t`, add `(H - 1 - t) * sum over n in C_t of crit(n)`, the impact the remaining steps would have cost, so that ending the episode by failure is never cheaper than continuing. | 1.0 | Unit scale: one fully critical node compromised for one step costs 1.0. |
| `response_delay` | negative | `min(d_t, d_max) / d_max` with `d_max = 20` | 0.5 | At most 0.5 per step. A longer-uncontained incident costs more each step, so the cumulative cost grows faster than linearly with delay. |
| `containment` | positive | Number of nodes `n` such that a containment-class action (`quarantine_node`, `isolate_node`, `restore_node`) was executed on `n` in step `t`, `n in C_t`, and `n not in K_{t-1}` | 3.0 | Reward for justified containment, earned once per node entering containment. About three steps of a mid-value node's impact. |
| `false_isolation` | negative | `sum over executed containment-class actions on a node n not in C_t of (b + crit(n))`, base `b = 0.5` | 2.0 | Wrong containment costs more for a critical node and never less than a floor of `0.5 * 2.0 = 1.0`. |
| `healthy_disruption` | negative | `sum over n not in C_t of crit(n) * (1 - davail_t(n))` | 0.5 | Ongoing cost of service lost on healthy nodes because of what the defense did, including collateral loss through dependencies of a node that was contained or restored. Loss caused by the attack itself is not charged here (it is in `attack_impact`), so faster containment cannot lower a term that is meant to measure harm done by the defender. Half the weight of attack impact. |
| `investigation_cost` | negative | `sum of cost_units` of executed `monitor_node`, `investigate_node`, `request_evidence` (0.5, 1.0, 3.0) | 0.05 | Small: investigating once costs 0.05, requesting deep evidence 0.15. Enough to prevent investigating every node every step, small enough not to deter investigation when it is informative. |
| `invalid_action` (guard, not one of the six named terms) | negative | Number of masked-action substitutions in step `t` | 0.1 | Matches the invalid-action penalty of CyMARL (RESEARCH section 1.1). Should be zero for masked policies. |

Only actions with `status = executed` count. A `failed` execution changes nothing and costs nothing.

### 4.2 Why these defaults, and what they trade off

- The default ratio of `attack_impact` (1.0) to `healthy_disruption` (0.5) makes an uncontained compromised node about twice as costly per step as a healthy node of equal criticality that is down. Containing early is favored when the evidence is good. The `false_isolation` floor adds an up-front price for being wrong, which is what makes investigation worth its small cost.
- `containment` is positive but earned once per node, so repeated actions on one node cannot farm it, and it is zero for healthy nodes. Together with `false_isolation` it makes the sign of a containment depend on the truth that only the reward model sees.
- `response_delay` is partly redundant with `attack_impact` (both grow with time). It is kept as its own term because the brief requires delay to be explicit and because it applies even when the compromised nodes have low criticality.
- The terminal charge in `attack_impact` removes an incentive to fail fast. Without it, ending the episode early by letting the objective node fall would stop all further penalties.
- A shared team reward matches the PPO multi-agent setting of Yu et al. and aligns the zone agents, since an incident that enters one zone threatens the other. The cost is weaker credit assignment: an agent's reward includes the other agent's mistakes. `zone_local` is the documented alternative.
- Reward magnitudes per episode on the mock are expected to range from about 0 (no incident) to a few tens of negative units. This is an expectation, not a measurement: the S3 tests check the formula, and the measured range of random-policy and baseline episodes is recorded in the S6 stage report. The PPO value normalization (section 5.2) removes the need to scale rewards by hand.
- Weights are configuration (`configs/experiments/reward_default.toml`), recorded as `weights_id` in every `EpisodeResult`. They are frozen before any evaluation run.

### 4.3 Hand-constructed cases (tests in `tests/rl/test_reward_terms.py`)

Each case builds two ground-truth snapshots and an executed-action list directly, without a simulator, and checks the exact value with default weights.

| Case | Setup | Expected |
|------|-------|----------|
| R-1 healthy critical versus low-value | One step, nothing compromised, `isolate_node` executed on a healthy node with own availability 0 and no dependents: once with `crit = 1.0`, once with `crit = 0.1` | `r = -(2.0 x (0.5 + 1.0)) - 0.5 x 1.0 x 1.0 = -3.5` and `r = -(2.0 x 0.6) - 0.5 x 0.1 x 1.0 = -1.25`. The critical case is strictly lower. |
| R-2 justified containment | Node with `crit = 0.5`, compromised, not contained at `t-1`; `isolate_node` executed on it at step `t`; first compromise 3 steps ago | `containment = +3.0`, `attack_impact = -0.5` (still compromised until restored), `response_delay = 0`, `false_isolation = 0`; `r = +2.5`. The same state with `no_op` gives `-0.5 - 0.5 x 3/20 = -0.575`. Justified containment scores higher. |
| R-3 delay | Same uncontained compromised node, `d_t = 2` versus `d_t = 10` versus `d_t = 30` | Delay term `-0.05`, `-0.25`, `-0.5` (capped at `d_max`). Strictly decreasing until the cap. |
| R-4 investigation cost | `investigate_node` executed; no other change | `r = -0.05`. `request_evidence`: `-0.15`. `monitor_node`: `-0.025`. |
| R-5 invalid action | One masked substitution, otherwise quiet | `r = -0.1` |
| R-6 false isolation floor | `isolate_node` on a healthy node with `crit = 0` and availability 1 | `false_isolation` term `= -1.0`; reward negative |
| R-7 terminal charge | `objective_reached` at `t = 10`, `H = 40`, objective node `crit = 1.0` compromised, nothing else | `attack_impact` magnitude `1.0 + 29 x 1.0 = 30`; contribution `-30.0` |
| R-8 configurable weights | Same inputs with `w_false_isolation = 0` | The term vanishes; all other terms unchanged. The recomputed reward equals `sum s_k w_k m_k` from the stored raw magnitudes. |
| R-9 quiet network | Healthy network, `no_op` | `r = 0.0` |
| R-10 dependency collateral | Healthy node `A` (`crit = 0.7`) depends on healthy node `B` (`crit = 0.3`); `B` isolated | `davail(A)` drops to 0, so the unweighted magnitude is `0.7 x 1.0 + 0.3 x 1.0 = 1.0` (and `false_isolation` for `B`) |
| R-11 attack-induced loss is not charged to the defense | Healthy node `A` (`crit = 0.7`) depends on compromised, uncontained node `B` (`crit = 0.3`, status `up`); first compromise at this step (`d_t = 0`); `no_op` | `avail(A) = 0.9` but `davail(A) = 1.0`, so `healthy_disruption = 0`; `attack_impact = 0.3`; `r = -0.3` |
| R-12 collateral of necessary containment is charged | `B` as in R-11 and `quarantine_node` executed on it (status `quarantined`, status availability 0.5); same step | `davail(B) = 0.5`, `davail(A) = 0.5`; healthy magnitude `0.7 x 0.5 = 0.35` (B is compromised, so only `A` counts); `containment = +3.0`; `attack_impact = -0.3`; `r = +3.0 - 0.3 - 0.5 x 0.35 = +2.525` |

### 4.4 Stored raw magnitudes and reweighting

`EpisodeResult.reward.term_totals` stores the unweighted sum over the episode of each `m_k`. Because `r` is linear in the weights, `cyberswarm.evaluation.reweight.reweight(results, weights)` recomputes `episode_reward` for any weight vector from the stored totals without re-running episodes. The terms `d_max` and `b` are not linear parameters; changing them needs a new run.

### 4.5 Sensitivity-analysis plan

Purpose: show how much conclusions depend on the reward design, instead of tuning weights until a method wins.

1. **Post-hoc reweighting (cheap, S6 hook).** For every evaluated policy, recompute `episode_reward` with each weight vector below from the stored totals. Report whether the ordering of approaches by reward changes.
2. **Retraining sweep (later experiment, plan only).** One-at-a-time multipliers `{0.5, 1, 2}` on each of the six main weights (12 non-default settings plus the default), plus `d_max` in `{10, 20, 40}` and `b` in `{0, 0.5, 1.0}`. For each setting, train IPPO with at least five training seeds, and evaluate on the held-out `eval` pairs (and report `eval_utopo` separately). No weight, threshold or hyperparameter is ever chosen using `eval` or `eval_utopo`.
3. **Report weight-independent metrics.** Because reward values are not comparable across weight settings, compare approaches on `false_isolation_count`, `compromised_nodes`, `detection_to_containment_steps`, `healthy_disruption_cost` and `attack_impact` with spread. State whether the ordering of approaches on each metric is stable across settings.
4. **Defaults are fixed first.** The default weight file is committed before any training run. A change after seeing results is recorded as a new `weights_id` and the earlier results stay in the report.
5. Configuration file: `configs/experiments/reward_sensitivity.toml` lists the grids. S6 implements the reweighting hook and the grid file; the retraining sweep is not part of the foundation phase.

## 5. IPPO

### 5.1 Library or compact PyTorch: decision (DECISIONS AD-08)

Decision: **a compact self-written PyTorch implementation**, confirming D-06. No multi-agent RL library is a dependency. TorchRL 0.14.1 (`IPPOLoss`, `MAPPOLoss`, `MultiAgentGAE`, `MaskedCategorical`) is the named contingency and an optional parity cross-check.

| Reason | Detail |
|--------|--------|
| Fit | The environment is small and custom. A library route would add TensorDict specs and a wrapper around the PettingZoo environment. |
| Access | The uncertainty mechanism needs the action distribution and a matched control sharing code. That is direct in our own policy class. |
| Dependencies | Only `torch` and `numpy`. TorchRL pins `tensordict<0.15` and renamed spec classes between 0.8.0 and 0.14.1 (RESEARCH section 2.4). BenchMARL needs `torchrl<0.12`, Hydra, torchvision. |
| Disk | The measured free space is small (DECISIONS section 1). |
| Clean ablation | IPPO and a later MAPPO share one code path and differ only in the critic input. |

Cost and risk: a new implementation can be wrong while still producing plausible curves. Mitigation is the test set in 5.5, which includes a toy task with a known optimum. Fallback trigger: if the toy-task or hand-computed GAE tests cannot be made to pass within stage S5, use TorchRL's loss and advantage code and keep everything else. A parity test against `ClipPPOLoss` on one fixed batch is optional.

### 5.2 Algorithm

Clipped-objective PPO with GAE, per agent, following the practical settings of Yu et al. (value normalization, at most 10 to 15 epochs, clip at or below 0.2, large batches, no mini-batch splitting).

| Setting | Default (`configs/experiments/ippo_default.toml`) | Smoke (`ippo_smoke.toml`) |
|---------|--------------------------------------------------|---------------------------|
| Actor, critic | MLP `D -> 64 -> 64 -> out`, tanh, orthogonal init (gain sqrt(2) hidden, 0.01 for actor output, 1.0 for critic output) | `D -> 32 -> 32 -> out` |
| Discount, GAE lambda | 0.99, 0.95 | same |
| Clip epsilon | 0.2 | same |
| Epochs per update | 4, full batch (no mini-batches) | 2 |
| Optimizers | Adam, learning rate 3e-4, separate for actor and critic of each agent | same |
| Entropy coefficient | 0.01 | same |
| Value loss | Clipped MSE on normalized returns; value normalization with running mean and standard deviation | same |
| Gradient clip | 0.5 (global norm per network) | same |
| Advantage normalization | Per agent over the update batch | same |
| Episodes per update | 16 | 4 |
| Updates | 200 | 3 |
| Greedy evaluation | argmax of the masked distribution | same |

The smoke budget runs in seconds on CPU and exists to prove train, save, load, infer and held-out evaluation. It will not produce a good policy and no quality is claimed.

Masking: masked logits are set to the dtype minimum before the softmax, so masked actions have probability exactly 0 and contribute 0 to entropy.

### 5.3 Independence

Each agent has its own actor, its own critic, its own optimizers and its own value normalizer. Nothing is shared: not parameters, not optimizer state, not normalizers, not advantages. Each critic sees only the agent's local `obs` vector. The only coupling between agents is through the environment and the shared reward value in their rollouts. Tests (5.5) prove independence structurally and behaviorally.

Because the two agents have different action-space sizes (22 and 36 on the mock), parameter sharing across agents would need padding and an agent-id input; that would be a different algorithm and is not IPPO as specified here.

### 5.4 Roles and heads

Following the architecture decision (ARCHITECTURE section 6.2), the learned network is the decision role only. Detection and investigation are deterministic components feeding it (`suspicion`, evidence fields) and investigation is a set of ordinary actions with a cost. There are no role-specific heads and no separate role policies.

### 5.5 Rollout, checkpoint, inference

Rollout. `marl.rollout.collect(env, policies, n_episodes, seed)` runs whole episodes and stores per agent: observation, mask, action, log-probability, value, team reward, terminated and truncated flags and the final observation value for bootstrapping. GAE is computed per episode.

Checkpoint (gitignored `runs/<run_id>/checkpoints/<name>/`): `checkpoint.pt` (state dicts of every actor and critic, value-normalizer statistics, optimizer states, update counter) and `checkpoint.json` (a small pydantic model `CheckpointMeta` in `marl/checkpoint.py`, not a registry record: agent ids, observation mode, `obs_dim` and `n_actions` per agent, the list of node ids, `ActionTable.digest()`, training configuration, master seed, `trained_topology_id` for information only, torch version, SHA-256 of `checkpoint.pt`). Loading uses `torch.load(..., weights_only=True)`, verifies the SHA-256, the shapes, the node-id list and the action-table digest, and raises `CheckpointError` on any mismatch. **It does not compare topology ids**: a checkpoint trained on `topo-mock-small-v1` loads and runs on `topo-mock-small-v2`, because both have the same node ids, zones and action tables (this is what the unseen-topology evaluation needs). A checkpoint for a topology with different node ids is refused. There is no silent fallback to random weights. The training log (`training_log.jsonl`, one `TrainingLogRecord` per update: update index, episodes, mean team return, per-agent policy loss, value loss, entropy, approximate KL, clip fraction, learning-check window) is a second small pydantic model in the same module.

Inference. `IPPOPolicy` implements the `Policy` Protocol: it encodes the `DecisionContext`, applies the mask, picks the greedy action (or samples with the policy generator when `stochastic=True`), and returns a `PolicyProposal` with `rationale_codes = ["learned_policy"]`, `action_probability` set, and `uncertainty = null`.

Tests (stage S5):

| Test | What it proves |
|------|----------------|
| `test_gae.py` | GAE equals a hand-computed small example, including termination (bootstrap 0) and truncation (bootstrap with value) |
| `test_ppo_loss.py` | Clipped surrogate equals a closed-form value on both sides of the clip, gradient sign correct |
| `test_masked_sampling.py` | Over many draws a masked action is never selected; entropy ignores masked actions |
| `test_value_norm.py` | Normalize and denormalize round trip |
| `test_ippo_independence.py` | Parameter sets of the two agents are disjoint objects; the gradient of agent A's loss with respect to agent B's parameters is `None`; an update on A leaves B's parameters bitwise unchanged |
| `test_toy_task.py` | On a tiny test-only cooperative environment with a known optimal return (each agent must copy its own observed bit), both agents reach at least 90 percent of the optimum within a fixed seeded budget |
| `test_checkpoint_roundtrip.py` | Save then load gives identical logits; corrupted file, wrong hash and wrong action table are refused |
| `test_inference_proposals.py` | Proposals validate against the contract and pass the coordinator for unmasked actions |
| `test_marl_determinism.py` | Same seed gives the same checkpoint hash and the same episode records |
| Smoke | Train for the smoke budget on the mock, save, load, run inference, run held-out evaluation through the harness |

## 6. MAPPO: planned stage S8, design and acceptance tests

### 6.1 Decision

**MAPPO is implemented in its own stage, S8, after IPPO (S5), the uncertainty mechanism (S5, second part), the evaluation harness (S6) and the learned-graph stage (S7) are stable** (DECISIONS AD-10). The design below is complete: decentralized actors that see only their own observation, one centralized team critic over the concatenation of the agents' agent-visible observation vectors (no ground truth), one shared GAE, and the shared team reward. There are only two agents and the PPO code already exists, so the design is judged feasible. It is **not deferred at design time**. The stage executes work items W1 to W7 and the tests M-1 to M-13. It is deferred only if the stage finds a concrete blocker, and then the deferral is recorded with evidence (section 6.3). Until the stage runs, every manifest says `mappo: not_run`. The repository contains no `marl/mappo.py` and no MAPPO stub before S8, because an empty module would imply an implementation that does not exist; after S8 it contains the real implementation. IPPO is never relabelled as MAPPO (test M-5).

Things that are **not** open questions and need no confirmation: the shared team reward and the concatenated-observation critic input are the design (project requirements, DELTA addendum 3). Equal-budget multi-seed experiments are in scope and run on the mock, sized for this machine's CPU; every result is labelled mock-only.

### 6.2 Exact technical design

| Aspect | Specification |
|--------|---------------|
| Actors | Identical to IPPO: per-agent actor, local `obs` and mask only, decentralized at inference. |
| Parameter sharing | Actors: none (agents have different action-space sizes). Critic: **one** centralized critic shared by all agents, because the reward is a shared team reward and therefore the value function is a single team value. Stated explicitly in the checkpoint metadata as `actor_sharing = "none"`, `critic_sharing = "single_team_critic"`, `critic_input = "global_concat"`. |
| Centralized critic input | `s_t` = `env.state()` = concatenation, in ascending agent-id order, of every agent's `obs` vector at step `t` (for the mock: `138 + 138 = 276` in conventional mode). These are agent-visible vectors only. This is valid training state: it contains no ground truth, no future information and nothing the evaluator alone knows. |
| Alternative rejected for version 1 | A privileged global state that includes ground truth (true compromise flags). It is allowed in some CTDE work but it would add information the actors never have, confound IPPO versus MAPPO with an information advantage, and put ground truth in a training input that must be fenced from decision time. If ever adopted, it needs a new DECISIONS entry and a leakage design that keeps it out of every actor and inference path. |
| Advantage | One GAE computed per episode from the team reward and the central value: `delta_t = r_t + gamma V(s_{t+1}) (1 - terminated) - V(s_t)`, `A_t = sum_l (gamma lambda)^l delta_{t+l}`, truncation bootstrapped with `V(s_T)`. `r_t` is the shared team reward (`StepScore.total`). The same `A_t` updates both actors. Advantages are normalized over the joint batch. |
| Losses | Per-agent clipped surrogate with the shared advantage; centralized critic loss on normalized team returns with value clipping. |
| Reward | Shared team reward (`sharing = "team"`). The `zone_local` option is not valid for MAPPO in version 1. |
| Inference | Actors only. The critic is not needed and must not be loaded by the inference path. |
| Same code path as IPPO | `marl.ppo` and `marl.gae` are shared. The difference is configuration of the critic: `critic_input = "local"` (one critic per agent on its own `obs`, IPPO) or `"global_concat"` (one team critic on `env.state()`, MAPPO). |

### 6.3 Ordering, work items and what counts as a blocker

No new library. The things that used to be listed as blockers are ordering or work items, not blockers:

| Item | Nature | Where it is handled |
|------|--------|---------------------|
| Stable IPPO (toy task, GAE, loss, masking, determinism, smoke evidence) | Ordering: MAPPO reuses that code | Stage S5 acceptance precedes S8 |
| `state()` hook on the environment | Work item, built early because the environment is lead-owned | Stage S3 (section 2.2); W1 only re-verifies it |
| Critic-input and team-reward choice | Settled by the design | Section 6.2 |
| Equal-budget multi-seed comparison with IPPO | In scope; sized for CPU | W7; `configs/experiments/ippo_compare.toml` and `mappo_compare.toml` (same seeds 0 to 4, same `updates` and `episodes_per_update` in every arm; if the measured wall-clock time is too long the budget is reduced for all arms together, never for one arm) |

A **blocker** is a failed check with evidence. Only these justify a `deferred` status: (a) one of M-1 to M-13 cannot be made to pass after debugging inside the stage, with the failing test output recorded; (b) the equal-budget comparison cannot finish within the budget on this machine, with the measured times recorded; (c) the dependency installation is impossible (for example `torch` cannot be installed because of disk space, with the measured `df` output). The stage then writes `reports/stage_decisions/mappo.json` (`StageDecision`, state `deferred`, `reason`, the failing criterion with its measured value, `evidence_paths` to the stage report) and the manifest status shows `deferred` with that reason. "Not enough time" and "not needed" are not accepted.

### 6.4 Work items (stage S8)

| ID | Work item | Files (new or changed) |
|----|-----------|------------------------|
| W1 | Re-verify `state()` and `state_space` on the environment: shape `(n_agents * D,)`, equals the concatenation of the actors' vectors, no ground truth, `EnvStateError` outside an episode | `rl/env.py`, `tests/rl/test_env_state.py` (created in S3) |
| W2 | Joint rollout storage that keeps the per-step concatenated state | `marl/rollout.py` |
| W3 | Central-critic network and loss path reusing `marl/ppo.py` and `marl/gae.py` | `marl/mappo.py` (new), `marl/networks.py` |
| W4 | `MAPPOPolicy` implementing `Policy`, loading actors only | `marl/mappo.py`, `marl/checkpoint.py` |
| W5 | `algorithm = "mappo"` in the training configuration, `mappo_default.toml` and `mappo_compare.toml`; the training CLI `runner/train.py` (created in S5 with `--algorithm` whose only choice is `ippo`) gains the choice `mappo` | `configs/experiments/`, `runner/train.py` |
| W6 | The policy factory knows `fam-mappo-conv`; `runner.export.compute_status` already derives `mappo` from the episodes (no constant to flip); documents updated | `runner/factory.py`, docs |
| W7 | Equal-budget comparison run through the existing harness on `eval` and `eval_utopo` (comparison C-8, section 9); failed or underperforming seeds stay in the output | `runner/evaluate.py` (no change expected) |

### 6.5 Acceptance tests (all required for status `implemented`)

| ID | Test |
|----|------|
| M-1 | Critic input dimension equals the sum of the actor observation dimensions; a scan of the input shows only agent-visible features |
| M-2 | Actor forward functions take no critic input; a gradient test shows actor parameters receive no gradient from the critic loss and the critic receives none from the actor loss |
| M-3 | Inference works with the critic weights removed from the checkpoint |
| M-4 | GAE with the central value equals a hand-computed example; both actors receive the identical advantage vector |
| M-5 | With `critic_input = "local"` the MAPPO code reproduces the IPPO critic inputs and, for the same seed, the same IPPO update on a fixed batch (guards against a relabelled implementation: the default configuration must differ from IPPO in critic input and parameter count) |
| M-6 | A toy task whose reward depends on both agents' observations, where a centralized critic can estimate the value more accurately than a local one: both trainers run, and MAPPO's mean return over 3 seeds is at least 90 percent of IPPO's within the seeded budget (a no-regression check, not a superiority claim) |
| M-7 | Determinism: same seed gives the same checkpoint hash |
| M-8 | Status equals execution: the manifest builder reports `mappo = implemented` if and only if the run's episodes contain MAPPO policies, `deferred` if and only if none did and a `deferred` `StageDecision` exists, and `not_run` otherwise (the generic test `tests/runner/test_manifest_status.py`) |
| M-9 | `test_actor_input_is_own_obs_only`: at execution each actor receives only its own local observation. The critic module is patched to raise; `MAPPOPolicy.propose` still works, and the tensor given to the actor equals that agent's own `obs`, shape `(D,)` |
| M-10 | `test_critic_input_global_vs_local`: on the same environment the MAPPO critic input dimension is `n_agents * D` and the IPPO critic input dimension is `D`; the MAPPO critic input contains only agent-visible fields and no evaluator-only field |
| M-11 | `test_advantage_uses_team_reward`: the per-agent reward arrays used in the rollout equal the step's `StepScore.total`; both actors receive the same advantage vector |
| M-12 | `test_mappo_checkpoint_roundtrip`: save then load gives identical logits; an actors-only load works; a corrupted file or wrong hash is refused |
| M-13 | `test_mappo_inference_proposals`: inference returns proposals that validate against `PolicyProposal` and pass the real coordinator for unmasked actions |

The status-honesty rule is general and is not specific to MAPPO: no manifest, status table or document may say a component is `implemented` unless the run or the code actually executed it (`tests/runner/test_manifest_status.py`, and the documentation validator at S9).

## 7. Uncertainty-aware mechanism

The mechanism is a decision rule wrapped around an IPPO policy. It reads the policy's own action distribution, the evidence view and the available actions. It changes the final action only in the cases below. It has no reward term and no extra network. Hypothesis H-E4 (RESEARCH section 3): at equal training and evaluation conditions, the rule reduces false isolations without worsening detection-to-containment time compared with the matched control. This is a hypothesis; nothing here claims it holds.

### 7.1 Signal

For the masked action distribution `pi` over the `K` currently valid actions:

`u = H(pi) / ln K` where `H(pi) = - sum_a pi(a) ln pi(a)`, computed in float64. For `K = 1`, `u = 0.0`. So `u` is in [0, 1]: 0 means the policy puts all probability on one action, 1 means uniform over the valid actions. The associated confidence is `c = 1 - u`.

If the logits are not finite or `K = 0`, no value is computed: `u = null` (CONTRACTS `UncertaintyInfo.value`), an `ErrorRecord(UNCERTAINTY_UNAVAILABLE)` is written, and the rule passes the proposal through with outcome `unavailable_pass_through`.

Why normalized policy entropy: it needs no extra network, no extra training budget and no sampling; it is deterministic given the weights; it is available for any stochastic actor; and it lets the matched control share every weight.

Limits (stated so that nobody over-reads it): entropy measures how spread the policy's action probabilities are. It is not epistemic uncertainty about the world. The entropy bonus in PPO shapes it. It can be high early in training and low for a confidently wrong policy. Whether it predicts correctness is therefore tested on held-out data (7.4) and not assumed. Ensemble disagreement and MC-dropout were considered and not chosen for version 1 because they need several networks or sampling passes and a larger training budget; they are the named alternatives if the entropy signal fails the informativeness gate.

### 7.2 Decision rule

Inputs: the base policy's chosen action `a*` (greedy in evaluation mode), `u`, the target node `n` of `a*`, the node's `evidence_level` `e(n)` from the observation (`null` counts as 0 for the rule), `steps_since_investigated(n)`, the available actions, and the per-(agent, node) count of consecutive substitutions in this episode.

Parameters (`configs/experiments/uncertainty_default.toml`): `tau_evidence = 0.50`, `tau_defer = 0.30` (required `tau_defer <= tau_evidence`), `e_contain = 3`, `max_consecutive_substitutions = 2`. The weak-evidence level `e_min = 2` is not a parameter of this file: it is defined once, in `roles.EvidenceConfig` (DESIGN section 4.4), and read by the rule baselines, the evidence view and this rule, so the three cannot disagree. The thresholds are defaults chosen before any run; the calibration procedure in 7.4 may replace them with values selected on the `cal` split only.

Rules, evaluated in this order. The first matching rule decides.

| # | Condition | Final action | `rule_outcome` | Rationale codes |
|---|-----------|--------------|----------------|-----------------|
| U0 | `u` is null | `a*` | `unavailable_pass_through` | none added |
| U1 | `a*` is `no_op`, or its impact class is `none` or `low` (`monitor_node`, `investigate_node`, `request_evidence`) | `a*` | `pass_through` | none |
| U2 | Consecutive substitutions for (agent, `n`) already equal `max_consecutive_substitutions` | `a*` | `pass_through` | none |
| U3 | `a*` is high-impact (`isolate_node`, `restore_node`), `e(n) >= e_contain`, and `u >= tau_defer` | `a*` | `containment_allowed_strong_evidence` | `strong_evidence_override` |
| U4 | `a*` is high-impact and `u >= tau_defer` (here `e(n) < e_contain`) | If `e(n) < e_min`: evidence request (7.2.1). If `e(n) == e_min`: downgrade to `quarantine_node(n)`, else `rate_limit_node(n)`, else the evidence request. | `deferred_high_impact` | `uncertainty_deferral` (plus `uncertainty_evidence_request` when the substitute is an evidence action) |
| U5 | `a*` has impact class `medium` (`rate_limit_node`, `quarantine_node`), `e(n) < e_min`, and `u >= tau_evidence` | Evidence request | `evidence_request` | `uncertainty_evidence_request` |
| U6 | otherwise | `a*` | `pass_through` | none |

Notes:

- Thresholds are inclusive: the rule fires at `u == tau`. Boundary tests use `tau - 1e-9`, `tau` and `tau + 1e-9`.
- 7.2.1 Evidence request: choose `investigate_node(n)` if `steps_since_investigated(n)` is null or at least 2; otherwise `request_evidence(n)`. If the first choice is not in `AvailableActions`, try the other; if neither is available, use `no_op`. The proposal's `uncertainty.original_action` records `a*`.
- U3 is reported only when the rule would otherwise have deferred (`u >= tau_defer`). When the policy is confident and the evidence is strong, the outcome is `pass_through` and nothing needs to be explained. This is the case "containment is allowed despite uncertainty": strong evidence (`e >= 3`) permits a high-impact action even at `u = 1.0`.
- U2 prevents paralysis: after two consecutive substitutions on the same node, the original action goes through.
- A medium-impact action with good evidence (`e >= e_min`) is never blocked by uncertainty.
- The wrapper never draws random numbers. It keeps one integer counter per (agent, node) per episode; the counters reset at `reset`.

### 7.3 Matched control

`MatchedControlPolicy` is the same wrapper class with `rules_enabled = False`, built over the same checkpoint and the same policy-sampling generator. It computes and records `u` for diagnostics, always reports `rule_outcome = not_applied`, and returns the base action unchanged. The only difference between the uncertainty-aware policy and its control is the decision rule.

Policy ids: base `pol-ippo-conv-s0`, uncertainty-aware `pol-ippo-conv-s0-ua`, control `pol-ippo-conv-s0-ctl` (the same pattern for the graph mode).

Equivalence tests: (a) the control's proposal equals the base policy's proposal in every field except `policy_id` and `uncertainty`; (b) the aware policy with thresholds that cannot fire (`tau_defer = tau_evidence = 1.01`) equals the control in `action`, `rationale_codes`, and every other field except `policy_id` and `uncertainty.rule_outcome`; (c) with the rule enabled, proposals differ from the control only at steps where a rule fired, and then only in `action`, `rationale_codes` and `uncertainty`; (d) both consume the same random numbers.

### 7.4 Calibration and validation on held-out episodes

The question: does a high `u` indicate that the base policy's committing proposals are more likely to be wrong?

- **Splits.** IPPO is trained on `train` scenarios. Thresholds and the calibrator are fitted on the `cal` split. Everything is reported on `eval` and, separately, on `eval_utopo`. Scenario ids, seeds and topology ids are disjoint between `train`, `cal` and the two evaluation splits, and the unseen topology never appears in `train` or `cal` (tested).
- **Committing proposals.** Proposals of the matched control (rule disabled) whose action has impact class `medium` or `high`.
- **Correctness label (evaluator-only, assigned after the episode).** A committing proposal is correct if its target node was truly compromised at execution time, incorrect otherwise. Computed by `evaluation.calibration_labels(proposals, truth) -> (u, correct)` from `PolicyProposal` records and the ground-truth stream; never available at decision time. `uncertainty.calibration` takes the resulting arrays (`u`, `correct`) and never imports ground truth, so the evaluator-only boundary of DESIGN section 3.2 holds.
- **Reliability.** Confidence `c = 1 - u`; 10 equal-width bins; per bin the count, mean confidence and empirical accuracy. `ece_raw = sum_b (n_b / n) |acc_b - mean_c_b|`. Maximum calibration error, Brier score and AUROC of `u` for predicting incorrect proposals (with a percentile bootstrap interval) are also reported.
- **Calibrator.** Histogram binning fitted on `cal`: confidence bin to empirical accuracy, bins with fewer than 20 samples merged with the neighbour. `ece_calibrated` is the ECE on `eval` after applying the `cal`-fitted mapping. Rules still act on raw `u` so that thresholds remain inspectable. `uncertainty_ece` in `EvaluationMetrics` is `ece_calibrated`.
- **Informativeness gate.** Status `ok` requires at least 100 committing proposals in `cal` and `eval` each and an AUROC of at least 0.60 with a bootstrap lower bound above 0.5. Fewer proposals: `insufficient_data`. Below the gate: `not_informative`. Reachability is checked, not assumed: the S6 stage report records the measured number of committing proposals per episode for the random and rule policies, and the seed lists of `cal` and `eval` are raised before any evaluation result is read if the expected count is below 150 (open item 6 in section 10). In either case `uncertainty_ece` is `null` when it cannot be computed, and every report of the uncertainty-aware policy must state the status next to its results.
- **Threshold selection (optional).** `select_thresholds` picks `tau_defer` as the lowest `u` such that, among `cal` high-impact committing proposals with `u >= tau`, at most 50 percent were correct; `tau_evidence` likewise for medium-impact proposals with `e < e_min`. If no such value exists the threshold is set to 1.01 (the rule cannot fire) and the report says so. The selection uses `cal` only.
- **Output.** One `CalibrationReport` record per reporting split (CONTRACTS 11.7) in `evaluation/calibration.jsonl`: bins, `ece_raw`, `ece_calibrated`, `mce`, `brier`, `auroc` and interval, `status`, the thresholds and the split that selected them. The dashboard can draw the reliability diagram from it.

### 7.5 Decision-boundary and behavior tests (stage S5, second part)

| ID | Test |
|----|------|
| U-1 | Entropy of a uniform, a one-hot, a masked and a single-valid-action distribution equals the closed-form values `1.0`, `0.0`, correct masked value, `0.0` |
| U-2 | Boundary: for each threshold, `tau - 1e-9` does not fire, `tau` and `tau + 1e-9` fire (U4, U5) |
| U-3 | Evidence request (U5): medium-impact action, `e = null`, `u >= tau_evidence` gives `investigate_node` or `request_evidence` as specified, with `original_action` recorded |
| U-4 | High-impact deferral (U4): `isolate_node`, `e = 1`, `u >= tau_defer` gives an evidence action; `e = 2` gives `quarantine_node` |
| U-5 | Justified containment (U3): `isolate_node`, `e = 3`, `u = 1.0` passes with `containment_allowed_strong_evidence` |
| U-6 | Unavailable uncertainty: non-finite logits give `value = null`, `unavailable_pass_through`, an `ErrorRecord`, and an unchanged action; the `null` survives JSON round trip |
| U-7 | Low-impact and `no_op` actions are never substituted |
| U-8 | Substitution cap: the third consecutive substitution on one node passes through |
| U-9 | Unavailable substitute action: falls back as specified and ends at `no_op` |
| U-10 | Matched-control equivalence (7.3 a to d) |
| U-11 | `evaluation.calibration_labels` on hand-built proposals and truth (known correct and incorrect labels, committing proposals only), then calibration arithmetic on a hand-built array pair (known bin counts and accuracies give the known ECE, AUROC and status) and the `insufficient_data` and `not_informative` branches |
| U-12 | `train`, `cal`, `eval` and `eval_utopo` scenario ids, seeds and topology ids are disjoint as required (7.4) |
| U-13 | The aware policy never produces an action outside `AvailableActions` and always passes the coordinator for the cases above |

## 8. Learned graph models: staged, decided in stage S7

### 8.1 Status and timing

The graph pipeline of section 3 is required infrastructure and is implemented in S2. A **learned** graph model (GraphSAGE-style or GAT-style) is a secondary, staged hypothesis. Its go or no-go is **taken in stage S7**, after IPPO, the uncertainty mechanism and the evaluation path are stable and after the conventional-versus-graph-feature ablation has run, and not at design time (DECISIONS D-19, AD-11). Dependencies and compute are not blockers (D-10: a plain-PyTorch dense-adjacency implementation avoids PyTorch Geometric entirely; the topology has 8 nodes). A no-go therefore needs a specific, evidenced reason (section 8.4). The result "no measurable improvement over the graph-feature baseline" is a valid outcome and is reported as a result, never hidden or turned into a reason to skip reporting. Scores, rankings, attention weights and embeddings are predictive outputs, never causal explanations, and the hypothesis is not assumed to hold.

### 8.2 Candidate tasks

| Task | Definition | Supervision | Output |
|------|------------|-------------|--------|
| A. Node suspicion scoring | For one agent snapshot (a `GraphBatch` item), predict for each observed node whether it is compromised at that step. Binary cross-entropy with `pos_weight = negatives / positives` of the training set, computed over observed nodes only (the label mask is the snapshot's observed-node set). | Offline dataset of pairs (`GraphBatch` item, label vector) built from `train`-split episodes only; the label is `GroundTruthSnapshot.nodes[].compromised` at the same step. | A score in [0, 1] per node (sigmoid), written as `GraphModelOutput` (`output_kind = anomaly_score`) and as `NodeScoreRecord` with `scorer_kind = learned_graph_model`. |
| B. Embeddings as IPPO encoder (run only if Task A completed) | The Task A model's hidden layer, `d_emb = 8` dimensions per node, frozen after Task A training, is concatenated to the base node block in place of the eight hand-engineered features. | None for the embedding beyond Task A; the IPPO policy is trained by reinforcement learning only. | Observation mode `learned_embedding`: per node 17 base values plus 8 embedding values, dimension `25 N + 2`, the same as `graph_features`, so the policy architecture is identical in all three arms. |

Labels are used only to train Task A, in the dataset builder `gnn/dataset.py` (the only `gnn` module allowed to import `evaluator_only`, listed in the single import list of DESIGN section 3.2). They never appear in a decision-time input: the scorer and the embedding encoder take `GraphBatch` tensors only. The `cal` split is used to select the model and stop training (not for reporting). `eval` and `eval_utopo` are never used for training, selection, early stopping or tuning, and nothing on the unseen topology is used before the final evaluation.

### 8.3 Architectures compared (plain PyTorch, no PyTorch Geometric)

Both models take the existing `GraphBatch` (CONTRACTS 10.1, DESIGN `graph.tensors`): `node_x` (B, N, 11) concatenated with `node_mask` (B, N, 11) and, as an option fixed before training, the eight derived features; `edge_index` (2, E); `edge_x`. A dense adjacency is built once per topology from `edge_index`: `A_link` and `A_dep` (N x N, 0 or 1, row = target, column = source for message flow). Two layers, hidden width 32, ReLU, dropout 0.0, one linear output head.

- `SageDense` (GraphSAGE-style mean aggregation): `h_i' = ReLU(W_0 h_i + sum over r in {link, dep} of W_r mean_{j in N_r(i)} h_j)`, with the mean over the relation's neighbours (zero vector if none).
- `GatDense` (graph attention): for each relation `r` and head (2 heads, concatenated): `e_ij = LeakyReLU(a_r^T [W h_i || W h_j])` for `j` in `N_r(i)` plus a self term, masked by the adjacency, softmax over `j`; `h_i' = ReLU(W_0 h_i + sum_r sum_j alpha_ij^r W_r h_j)`.

Attention weights are stored only for debugging and are never shown as explanations. Feasibility versus topology size: with 8 nodes and diameter 4 (3 on the unseen variant) two layers reach the whole graph, both models train in seconds on CPU, and the hand-engineered features already contain neighbour alert aggregates. That last fact is the hypothesis being tested, not a reason to skip the test.

### 8.4 Go and no-go criteria (evaluated in S7 by `scripts/gnn_go_nogo.py`)

All of G1 to G5 must hold to implement. A failing criterion is the evidenced reason for `no_go`, recorded with its measured value in the `StageDecision`.

| ID | Criterion | Measured by |
|----|-----------|-------------|
| G1 | Data: every `train` scenario contributes at least 20 positive and 20 negative labelled node-steps to the Task A dataset | Dataset builder counts |
| G2 | Compute: a forward and backward pass of each model on a batch of 64 snapshots takes at most 0.5 s on CPU, and a full training run of at most 30 epochs takes at most 10 minutes of CPU time per model per seed | Timed in the script |
| G3 | Split: scenario ids, seeds and topology ids of `train` and `cal` are disjoint from `eval` and `eval_utopo`, and the unseen topology never appears in training or calibration | `evaluation.splits.assert_disjoint` |
| G4 | Plain-PyTorch models pass the shape, mask-handling, forward and backward tests on the real `GraphBatch` | `tests/gnn/` |
| G5 | The matched reference exists: the S6 outputs for `ippo-conv` and `ippo-graph` and the rule baselines on `eval` and `eval_utopo` are present | File check |

A decision to implement is recorded as `StageDecision(state = implemented)` after the work is done. The stage report states, with spread, whether Task A and the embedding arm improved on the graph-feature arm; "no measurable improvement" (the bootstrap interval of the paired difference includes zero or is on the unfavourable side) is stated as such. `no_go` is not used for that outcome.

Task B is run only if Task A met a non-inferiority check on `cal`: the better model's `origin_mrr` on `cal` is at least that of `graph-rule-v1` minus 0.02. If not, Task B is not run, the stage report records the two numbers as the reason, and `learned_embedding` appears in no record. Task A results are still reported.

### 8.5 Matched ablation

On identical `eval` and `eval_utopo` (scenario, seed) pairs, with otherwise identical policy architecture, budget and training seeds 0 to 4:

| Arm | Observation | Status |
|-----|-------------|--------|
| `ippo-conv` | conventional (`17 N + 2`) | S5 |
| `ippo-graph` | hand-engineered graph features (`25 N + 2`) | S5 |
| `ippo-emb` | learned embedding (`25 N + 2`) | S7, if Task B ran |

Comparisons C-2 (conventional versus graph features) is **required whatever the GNN decision** and is produced in S6. C-6 and C-7 (embedding versus graph features and versus conventional) exist only in S7. Node scoring is compared through the ranking metrics `origin_rank_top1` and `origin_mrr` (CONTRACTS 11.2) for the scorers `detection.rule-v1`, `graph-rule-v1`, `gnn-sage-v1` and `gnn-gat-v1`, plus the AUROC of each score against the evaluator-only compromise labels on `eval` and `eval_utopo`. The AUROC is computed by the evaluator, never by the scorer.

Optional (only if the stage goes ahead and time allows): a topology with a different node count, `topo-mock-large-v1` (the 8 nodes of v1 plus `node-app-02` in the core zone and `node-web-03` in the dmz zone), evaluated on the Task A scorers and `graph-rule-v1` only. The IPPO policy has a fixed input width and is not evaluated on it. It is never used in training or selection.

### 8.6 Tests (stage S7)

| ID | Test |
|----|------|
| GN-1 | Dataset builder: positives and negatives per scenario; only `train` episodes; labels aligned with observed nodes |
| GN-2 | Leakage: the scorer and encoder signatures accept `GraphBatch` only; the decision-time modules do not import `evaluator_only`; perturbing ground truth does not change scores |
| GN-3 | `SageDense` and `GatDense`: output shapes, mask handling, permutation consistency over batch, forward and backward gradients finite |
| GN-4 | Checkpoint save and load round trip (identical outputs); corrupted file refused |
| GN-5 | Contract-valid `GraphModelOutput` and `NodeScoreRecord` from inference; unavailable model gives `not_computed` |
| GN-6 | Split guard: training on `eval`, `eval_utopo` or the unseen topology raises |
| GN-7 | `learned_embedding` observation has the same dimension as `graph_features` and contains no label-derived column |
| GN-8 | Seeded determinism: same seed gives the same checkpoint hash |

## 9. Comparison matrix (identical scenarios and seeds)

### 9.1 Arms

| Arm (family id) | Policy id pattern | Representation | Stage | Trained |
|-----------------|-------------------|----------------|-------|---------|
| `fam-rule-conv` | `pol-rule-conv-v1` | conventional | S4 | no |
| `fam-rule-graph` | `pol-rule-graph-v1` | graph_features | S4 | no |
| `fam-ippo-conv` | `pol-ippo-conv-s<k>` | conventional | S5 | yes, training seeds `k` = 0 to 4 |
| `fam-ippo-graph` | `pol-ippo-graph-s<k>` | graph_features | S5 | yes |
| `fam-ippo-conv-ua` | `pol-ippo-conv-s<k>-ua` | conventional | S5 | no extra training (wraps `ippo-conv`) |
| `fam-ippo-conv-ctl` | `pol-ippo-conv-s<k>-ctl` | conventional | S5 | no extra training (matched control of the previous arm) |
| `fam-ippo-emb` | `pol-ippo-emb-s<k>` | learned_embedding | S7, only if Task B ran | yes |
| `fam-mappo-conv` | `pol-mappo-conv-s<k>` | conventional | S8, unless a deferral with evidence is recorded | yes |

### 9.2 Comparisons produced (each separately for `eval` and `eval_utopo`)

| ID | Baseline arm | Candidate arm | Unit | Question |
|----|--------------|---------------|------|----------|
| C-1 | `fam-rule-conv` | `fam-rule-graph` | matched pair | Do hand-engineered graph features change the rule baseline's outcomes? |
| C-2 | `fam-ippo-conv` | `fam-ippo-graph` | training seed | Conventional versus hand-engineered graph-feature observations for IPPO (**required whatever the GNN decision**) |
| C-3 | `fam-ippo-conv-ctl` | `fam-ippo-conv-ua` | training seed | Does the uncertainty rule change outcomes against its matched control? |
| C-4 | `fam-rule-conv` | `fam-ippo-conv` | training seed | Rule-based versus learned, conventional observations |
| C-5 | `fam-rule-graph` | `fam-ippo-graph` | training seed | Rule-based versus learned, graph features |
| C-6 | `fam-ippo-graph` | `fam-ippo-emb` | training seed | Learned embedding versus hand-engineered features (S7 only) |
| C-7 | `fam-ippo-conv` | `fam-ippo-emb` | training seed | Learned embedding versus conventional (S7 only) |
| C-8 | `fam-ippo-conv` | `fam-mappo-conv` | training seed | IPPO versus MAPPO at equal budget (S8 only) |

Each comparison is a test of a difference; none is assumed to favour the candidate.

### 9.3 Rules

- The harness evaluates every arm on the same sorted list of (scenario, seed) pairs per split (`matched_set_id`) and produces a `ComparisonRecord` only between arms with equal matched sets.
- **Aggregation.** For trained arms the unit is the training seed: `AggregateMetrics` (CONTRACTS 11.8) holds the per-seed means and their mean, sample standard deviation and percentile bootstrap interval across training seeds (interval null below 5 seeds). Episode-level bootstrap intervals in `EvaluationMetrics` describe variation over scenarios and seeds for one trained policy only. Rule-based arms use the episode-level bootstrap directly.
- **Failed runs stay.** A training seed that aborted, did not learn (`learning_check = not_improved`) or underperforms is kept in `AggregateMetrics.training_outcomes` and in every comparison. No run is discarded or repeated until a favourable result appears.
- **No tuning on the evaluation splits.** Thresholds, weights, hyperparameters and model selection use `train` and `cal` only. `eval` and `eval_utopo` are read once per final configuration.
- **Unseen topology.** `eval_utopo` uses `topo-mock-small-v2` (same node ids and zones, different links, dependencies and criticalities). No file with that topology id is used for training, threshold selection or calibration (tested). A checkpoint trained on v1 runs on v2 unchanged (section 5.5). Metrics for `eval` and `eval_utopo` are never pooled.
- **Ranking metric.** Where a component produces per-node suspicion scores (the detection role `detection.rule-v1`, the graph-feature baseline `graph-rule-v1`, and the learned scorers if built) the evaluator reports `origin_rank_top1` and `origin_mrr` per scorer (CONTRACTS 11.2), labelled as a ranking of predicted suspicion and never as a root-cause finding.
- Train, calibration and evaluation scenario sets are disjoint. Results in this phase come only from the mock and are labelled `mock_simulator_run`; none of them is a research finding.

## 10. Assumptions and open items

| # | Item |
|---|------|
| 1 | The shared team reward is a design choice (the setting of Yu et al.) fixed by the project requirements; it needs no confirmation. Its credit-assignment cost is stated in 4.2 and `zone_local` is the documented alternative. |
| 2 | The default weights and thresholds are untested starting points. The calibration and sensitivity plans exist so that they are examined, not trusted. |
| 3 | The entropy signal may fail the informativeness gate. The design reports that outcome instead of hiding it. |
| 4 | The mock has one incident type and 8 nodes (plus an unseen 8-node variant). Conclusions about real or richer simulators do not follow from it. |
| 5 | The mock's `service_health` channel is deliberately weak (DESIGN 5.3: compromise factor 0.9, noise standard deviation 0.08) and a test bounds the AUROC of a `service_health` threshold. If a learned policy still reads compromise from that channel alone, the finding is reported and the mock channel is weakened before any claim. |
| 6 | The calibration gate needs at least 100 committing proposals in `cal` and in each reported evaluation split. The S6 stage report records the measured count per episode for the random and rule policies, and the number of `cal` and `eval` seeds is raised before any evaluation result is read if the expected count is below 150 (a 50 percent margin). Seeds are chosen from committing-proposal counts only, never from outcomes. |
| 7 | MAPPO is planned for S8; the only accepted grounds for deferral are the evidenced blockers of 6.3. Learned graph models are planned for S7 with the go criteria of 8.4. |
