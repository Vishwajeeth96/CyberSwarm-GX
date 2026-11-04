# CyberSwarm Architecture

**Version:** 0.2 (design). Supersedes the untracked draft v0.1 of 2026-10-29.
**Status:** SPECIFIED, NOT IMPLEMENTED. No source code exists yet. Every component below is a design for the implementation stages S1 to S9 in `.agents/tasks/cyberswarm-foundation/DESIGN.md`. Nothing here claims a result.
**Direction:** CyberSwarm is self-learning autonomous cyber defense with multi-agent reinforcement learning (MARL), inside a controlled simulator only. Authority: `.agents/tasks/cyberswarm-foundation/DELTA.md` and [DECISIONS.md](DECISIONS.md) AD-01.
**Companion documents:** [CONTRACTS.md](CONTRACTS.md) (every record and the step semantics), [RL_IMPLEMENTATION.md](RL_IMPLEMENTATION.md) (spaces, reward, IPPO, MAPPO status, uncertainty), [RESEARCH.md](RESEARCH.md) (verified background), [DECISIONS.md](DECISIONS.md).

## Contents

1. [Summary](#1-summary)
2. [Research basis](#2-research-basis)
3. [Scope and direction](#3-scope-and-direction)
4. [System layout](#4-system-layout)
5. [Component catalogue](#5-component-catalogue)
6. [Logical roles, agents and policies](#6-logical-roles-agents-and-policies)
7. [Data flow](#7-data-flow)
8. [Error handling and failure behavior](#8-error-handling-and-failure-behavior)
9. [Ground-truth separation](#9-ground-truth-separation)
10. [Ownership and integration sequence](#10-ownership-and-integration-sequence)
11. [Design principles](#11-design-principles)
12. [Constraints and safety](#12-constraints-and-safety)
13. [Technology stack (locked on approval)](#13-technology-stack-locked-on-approval)
14. [Testability](#14-testability)
15. [Implementation status](#15-implementation-status)
16. [Reconciliation with draft v0.1](#16-reconciliation-with-draft-v01)

---

## 1. Summary

CyberSwarm investigates whether cooperating reinforcement-learning agents, with an uncertainty-aware decision rule, can contain simulated cyber incidents while disrupting fewer healthy services than a rule-based defense and a standard learned defense. The question is stated as a hypothesis. It is not assumed to be true, and the project makes no novelty or superiority claim (RESEARCH.md section 1.6 lists overlapping prior work).

The first version is a modular monolith: one Python package, in-process calls, JSON and JSON Lines files for records. There are no services, databases, message brokers or LLM-chat agents.

Approaches compared on identical scenarios and seeds: a rule-based baseline on conventional observations, a non-neural graph-feature baseline, independent PPO (IPPO), an uncertainty-aware variant with a matched non-uncertainty control, and MAPPO, which is implemented in its own stage after IPPO, the uncertainty variant and the evaluation path are stable, unless that stage finds a concrete, evidenced blocker (section 6.5 and RL_IMPLEMENTATION section 6).

Everything that touches the network is abstract simulator state. No component scans, exploits, isolates or changes anything real.

## 2. Research basis

Full record with source tags: [RESEARCH.md](RESEARCH.md). The points that matter for the design:

| Point | Status |
|-------|--------|
| CyMARL (arXiv:2310.05939, Wiebe, Al Mallah, Li; workshop paper, 2023): cooperative defenders on CybORG, IQL and QMIX, importance-weighted team reward, a heuristic baseline. | VERIFIED. Value-based, so not a PPO reference. Code exists at `Poly-AIvsAI/CyMARL` (no licence); the identity with the paper is an inference. |
| "CMARL-ACD": a GitHub repository (`cyb3rlab/CMARL-ACD`) and a JAIST master's thesis. Competitive attacker and defender, IPPO and MAPPO in TorchRL, noisy IDS alerts. The journal paper was not located. | VERIFIED as a repository and thesis name. Which source the project brief means is unconfirmed (RESEARCH.md section 4, item 1). |
| Yu et al. (arXiv:2103.01955): MAPPO is PPO with a centralized value input, IPPO uses local inputs. | VERIFIED. Source of the practical settings in RL_IMPLEMENTATION. |
| CAGE Challenge 4 / CybORG: closest maintained analogue for "contain while keeping legitimate work running". | VERIFIED. Design reference only, never a dependency. |
| Related work already combines MAPPO with graph encoders and risk constraints (arXiv:2606.13832), and compares MAPPO with IPPO (arXiv:2401.10149). | REPORTED (abstracts). Therefore no novelty claim. |

Corrections to draft v0.1: it said the CyMARL repository is not public (it is), that CybORG is not open source (it is MIT-licensed), and that CyMARL was "published October 2023" (the arXiv v1 is dated 25 Aug 2023 and it is a workshop paper). Those statements are removed.

## 3. Scope and direction

| Topic | Decision |
|-------|----------|
| Primary line | MARL for autonomous defense: rule-based baseline, IPPO, uncertainty-aware variant with matched control, MAPPO only if correct. Central question: simulated containment versus disruption of healthy services. |
| Graph infrastructure | Required, lead-owned, built before the RL environment, and not the central claim. It provides stable topology and dynamic graph state, deterministic observation-to-graph conversion, missing-data masks with a non-leaking imputation policy, hand-engineered graph features, and a graph observation mode for a matched comparison with conventional observations. NetworkX is its default foundation. No learned message passing. (DELTA.md current text; DECISIONS D-09, D-18.) |
| Learned GNN (GraphSAGE, GAT) and the "Faultline" root-cause concept | Not central. A staged, secondary hypothesis. The go/no-go is taken in its own learned-graph stage after IPPO, the uncertainty extension and the evaluation path are stable, from the candidate task, the two plain-PyTorch architectures, the go criteria and the matched ablation specified in RL_IMPLEMENTATION section 8 (DECISIONS D-19, AD-11); it is not decided at design time. "No measurable improvement over the graph-feature baseline" is a valid reported result, not a no-go. Root-cause localization is not a headline task, but a ranking of predicted suspicion is evaluated: origin top-1 accuracy and the mean reciprocal rank of the incident-origin node, per scorer, labelled as predicted suspicion and never a causal finding. |
| Simulator | Raksha's real simulator is specified, not implemented. A small lead-owned MOCK simulator (`cyberswarm.mock`) implements the same public `Simulator` Protocol so that everything else can run. The mock is never described as the real simulator. |
| Dashboard | Mayukh's module is specified, not implemented. The lead provides contracts, fixtures and dashboard-consumable output files only. |

## 4. System layout

### 4.1 Layers and dependency rule

```
 Scenario configuration (ScenarioConfig JSON, seed)
            |
            v
 +----------------------------+        +------------------------------+
 | Simulator (Protocol)       |        | Ground-truth accessor        |
 |  Raksha's real simulator   |------->| (evaluator-only, separate)   |
 |  or lead-owned MOCK        |        +------------------------------+
 +----------------------------+                    |  (reward, evaluator only)
            | AgentObservation, AvailableActions,  |
            | EventRecord (agent-visible)          |
            v                                      |
 +----------------------------+                    |
 | Roles and graph pipeline   |                    |
 |  detection (deterministic) |                    |
 |  evidence view             |                    |
 |  graph snapshot + features |                    |
 +----------------------------+                    |
            | DecisionContext                       |
            v                                      |
 +----------------------------+                    |
 | Decision role: policy per  |   PolicyProposal   |
 | zone agent                 |------------+       |
 |  rule | graph-rule | IPPO  |            |       |
 |  (+ uncertainty wrapper)   |            v       |
 +----------------------------+   +-----------------------+
                                  | Defense coordinator   |
                                  | validate, allowlist,  |
                                  | execute via public    |
                                  | simulator API         |
                                  +-----------------------+
                                            | ValidationResult, ExecutedAction
                                            v
                         Updated simulator state, outcome events
                                            |
                                            v
                         Reward model (RL training only) and Evaluator
                                            |
                                            v
                         Dashboard-consumable records (JSON / JSONL)
```

Dependencies point down the following order only. A higher package never imports a lower one's opposite direction, and a test (`tests/integration/test_import_boundaries.py`) enforces it.

```
contracts  <-  interfaces  <-  common
    ^              ^
    |              +---- mock            (implements interfaces.Simulator)
    +---- roles, graph, coordinator, evaluation   (evaluation is pure: records in, records out)
                  ^
                  +---- engine            (step semantics; uses roles, graph, coordinator)
                           ^
                           +---- rl      (env, action table, encoder, reward)
                                   ^
              baselines (needs interfaces.Policy, roles, graph records only)
                                   +---- marl (IPPO)  <-  uncertainty
                           ^
                           +---- runner  (config, factories, evaluation harness, smoke, export)
```

### 4.2 Repository layout (decision: `src/` layout, one package `cyberswarm`)

`src/` layout prevents tests from importing uninstalled code and matches the editable install used in the virtual environment. Each directory has one owner. The full tree with every file is in DESIGN.md section 2.

| Path | Owner | Purpose |
|------|-------|---------|
| `src/cyberswarm/contracts/` | Project lead (neutral) | Pydantic models, ids, vocabulary, feature registries, schema export |
| `src/cyberswarm/interfaces/` | Project lead (neutral) | `Simulator`, `GroundTruthAccessor` and `Policy` Protocols; simulator conformance suite |
| `src/cyberswarm/common/` | Project lead | Seeding, hashing, JSON/JSONL I/O, logging, TOML configuration |
| `src/cyberswarm/mock/` | Project lead | MOCK simulator (labelled mock) |
| `src/cyberswarm/roles/` | Project lead | Detection role, evidence view |
| `src/cyberswarm/graph/` | Project lead | Graph conversion, features, tensors |
| `src/cyberswarm/coordinator/` | Project lead | Defense coordinator, allowlist, recovery verification |
| `src/cyberswarm/engine/` | Project lead | Episode step engine, record recorder |
| `src/cyberswarm/rl/` | Project lead | Action table, observation encoder, reward, PettingZoo environment |
| `src/cyberswarm/baselines/` | Project lead | Rule-based and graph-feature baselines |
| `src/cyberswarm/marl/` | Project lead | IPPO; MAPPO implemented in its own later stage (no `marl/mappo.py` until that stage runs) |
| `src/cyberswarm/uncertainty/` | Project lead | Uncertainty signal, rules, matched control, calibration |
| `src/cyberswarm/evaluation/` | Project lead | Outcome, metrics, statistics, comparison, splits, reweighting |
| `src/cyberswarm/runner/` | Project lead | Configuration, policy factory, evaluation harness, smoke command, export |
| `src/cyberswarm/simulator/` | Raksha | Real simulator. Not created yet (no placeholder) |
| `tests/simulator/`, `configs/scenarios/simulator/` | Raksha | Her tests and scenarios. Not created yet |
| `dashboard/` | Mayukh Aryan | Dashboard application. Not created yet |
| `schemas/`, `fixtures/`, `configs/` (other than the line above), `scripts/` | Project lead (neutral) | Exported JSON Schema, synthetic fixtures, configuration, utilities |
| `tests/` (other than `tests/simulator/`) | Project lead | All other tests |
| `docs/` | Project lead; teammates edit only their `MODULE_*` document after the lead authorizes it | Documentation |
| `runs/`, `.venv/`, `checkpoints/` | Nobody (generated, gitignored) | Outputs and environment |

## 5. Component catalogue

Notation: **I** inputs, **O** outputs, **Dep** dependencies, **API** public interface (signatures in DESIGN.md section 4), **Err** error handling and failure behavior (details in section 8).

| # | Component | Owner | I | O | Dep | API (summary) | Err |
|---|-----------|-------|---|---|-----|---------------|-----|
| 1 | Network simulator | Raksha (real, not implemented); lead (MOCK) | `ScenarioConfig`, seed, validated actions | agent observations, available actions, events, step results, state digest | contracts, interfaces; NetworkX for topology logic | `Simulator` Protocol: `reset`, `observe`, `available_actions`, `apply_action`, `end_step`, `state_digest`, `close` | `SimulatorError` with `ErrorCode`; episode aborts |
| 2 | Attack and scenario engine | Raksha (inside 1); lead (inside MOCK) | scenario incident spec, seed | incident state transitions (compromise, spread) | internal to the simulator | Not public. Visible only through events and the ground-truth accessor | Unsupported scenario refused at `reset` |
| 3 | Event pipeline | Simulator side (producer); lead (consumers) | state transitions | ordered `EventRecord` stream | contracts | `Simulator` (events returned from `apply_action`, `end_step`; collected by the engine) | Order or id violations are `INVARIANT_VIOLATION` |
| 4 | Observation pipeline | Simulator side | state | `AgentObservation`, `AvailableActions` (partial, per zone agent) | contracts | `Simulator.observe`, `Simulator.available_actions` | Conformance suite checks content |
| 5 | Detection role | Lead | `AgentObservation` | `DetectionSummary` | contracts | `DetectionRole.summarize(obs)` | Pure function; invalid input raises, never guesses |
| 6 | Investigation support | Lead | `AgentObservation` | `EvidenceView` (per-node evidence level, age, whether investigation is advisable) | contracts | `evidence_view(obs)` | Pure function |
| 7 | Graph pipeline | Lead | `Topology`, `AgentObservation` | `GraphSnapshot`, `GraphFeatureRecord`, tensors and batches | contracts, NetworkX, numpy | `GraphConverter.convert`, `compute_features`, `to_tensors`, `collate` | `GraphConversionError` (`UNKNOWN_ID`, `INVARIANT_VIOLATION`) |
| 8 | Rule-based baseline | Lead | `DecisionContext` (conventional) | `PolicyProposal` | roles, contracts | `RuleBasedPolicy` implements `Policy` | Never raises on valid input; unknown state yields `no_op` |
| 9 | Graph-feature baseline | Lead | `DecisionContext` with graph features | `PolicyProposal` | roles, graph records | `GraphFeaturePolicy` implements `Policy` | As above; refuses a context without graph features (`POLICY_FAILURE`) |
| 10 | RL environment adapter | Lead | proposals as action indices; simulator via engine | per-agent observation dicts (vector plus mask), rewards, terminated, truncated, info | PettingZoo, Gymnasium, engine, graph, roles | `CyberSwarmParallelEnv` (`reset`, `step`, and `state` returning the ascending-agent-order concatenation of agent-visible observation vectors, the hook the later MAPPO critic uses) | `EnvStateError`, `ValueError` for caller bugs; masked action substituted by `no_op` |
| 11 | Reward model | Lead | `GroundTruthSnapshot` pair, executed actions, config | `RewardBreakdown` | contracts (evaluator-only module) | `RewardModel.compute(...)` | Missing ground-truth accessor is a configuration error at construction |
| 12 | IPPO | Lead | local observation vectors, masks, rewards | trained checkpoints; proposals through the `Policy` interface | PyTorch, rl | `train(config)`, `IPPOPolicy` implements `Policy`, `save`/`load` | `CHECKPOINT_ERROR`; non-finite loss aborts training |
| 13 | MAPPO | Lead | local observation vectors, masks, team reward, concatenated agent-visible state via `env.state()` | trained checkpoints; proposals through the `Policy` interface | PyTorch, rl, engine | Planned stage after IPPO, uncertainty and evaluation: per-agent decentralized actors and one centralized team critic over `env.state()`; `MAPPOPolicy` implements `Policy` and shares the PPO code path with IPPO. Design and acceptance tests M-1 to M-13 in RL_IMPLEMENTATION section 6 | `CHECKPOINT_ERROR`; non-finite loss aborts training |
| 14 | Uncertainty extension | Lead | policy action distribution, observation | final `PolicyProposal` with `UncertaintyInfo` | marl | `UncertaintyAwarePolicy`, `MatchedControlPolicy`, `normalized_entropy`, `calibrate` | Non-finite distribution gives `null` uncertainty and pass-through |
| 15 | Defense coordinator | Lead | raw proposals, simulator state via the public API | `ValidationResult`, `ExecutedAction`, recovery events | contracts, interfaces | `DefenseCoordinator.process(...)` | Rejection is a normal result; execution failure is a `failed` `ExecutedAction` |
| 16 | Simulated containment engine | Raksha (real); lead (MOCK) | `ValidatedAction` | state change, outcome events | inside the simulator | `Simulator.apply_action` | `ACTION_NOT_AVAILABLE` or failure leaves state unchanged |
| 17 | Recovery verification | Lead | executed restore/containment, later observations | `recovery_verified` or `recovery_failed` events | coordinator, contracts | `RecoveryVerifier.check(...)` | Uses simulator-reported (agent-visible) data only |
| 18 | Episode engine | Lead | simulator, policies, coordinator, optional reward model | all per-step records | all of the above | `EpisodeEngine.reset`, `EpisodeEngine.step` | Records `ErrorRecord`s; aborts the episode on fatal errors |
| 19 | Evaluation and experiment tracking | Lead | episode records, ground-truth stream | `EpisodeResult.outcome`, `EvaluationMetrics`, `ComparisonRecord` | contracts, numpy | pure functions in `cyberswarm.evaluation`; harness in `cyberswarm.runner.evaluate` | Undefined metric is `null` with `n_undefined` |
| 20 | Dashboard | Mayukh Aryan (not implemented) | run directory files | UI | JSON Schema files | `DataSource` interface (MODULE_MAYUKH_DASHBOARD.md) | Shows malformed or missing data honestly |
| 21 | Logging and configuration | Lead | TOML files, JSON scenario and topology files | `LogRecord` lines, resolved configuration snapshot | standard library | `common.logging.get_logger`, `common.config.load_*` | `CONFIG_INVALID` before any run starts |

The graph pipeline (7) is required infrastructure. It is not a learned model and does not change the primary line of the project.

## 6. Logical roles, agents and policies

### 6.1 The four logical roles

Detection, investigation, decision and coordination are software roles with typed inputs and outputs. They are not LLM-chat agents, they do not exchange free text, and none of them calls a language model.

| Role | Realization | Input | Output | Learns? |
|------|-------------|-------|--------|---------|
| Detection | Deterministic component `roles.detection`. | `AgentObservation` | `DetectionSummary` (per-node suspicion, alerted flag). `suspicion` is also an input feature of the policy. | No |
| Investigation | Two parts. (a) The act of investigating is an **action** chosen by the decision role: `monitor_node`, `investigate_node`, `request_evidence`, with a cost. (b) The result comes back from the simulator as `investigation_result` events and the `evidence_level` observation field; `roles.evidence` turns that into an `EvidenceView`. | `AgentObservation` | `EvidenceView` | No (the choice of when to investigate is learned or rule-based inside the decision role) |
| Decision | One policy per zone agent: rule-based, graph-feature rule-based, or an IPPO actor. | `DecisionContext` | `PolicyProposal` | Yes for IPPO |
| Coordination | The defense coordinator, outside every policy. | `PolicyProposal` | `ValidationResult`, `ExecutedAction` | No |

### 6.2 Decision: deterministic components around one policy per agent

The architecture decision (DECISIONS AD-06):

- **Decision role = one policy per zone agent.** Detection and investigation are deterministic components around it. Coordination is an independent safety boundary.
- **Agents = zone defenders.** There is one decision agent per zone (`agent-<zone>`), so the mock topology has `agent-dmz` and `agent-core`. Each agent can act only on nodes of its zone and sees its zone plus one-hop neighbours across the boundary. This gives the MARL problem real partial observability and real cooperation (an incident that enters in one zone can spread into the other).
- **IPPO parameterization:** each agent has its own actor and its own critic, with no shared parameters, trained from local observations (RL_IMPLEMENTATION section 5).

Alternatives considered and rejected:

| Alternative | Why rejected |
|-------------|--------------|
| Separate learned policies for detection, investigation and decision | Detection has no action space of its own. Its training signal would be a ground-truth label (which node is compromised), which must never reach decision-time data. Investigation is naturally an action with a cost, not a separate agent. Three policies would need three reward definitions and credit assignment between them. |
| One shared trunk with role-specific heads | The detection head would need supervised labels from evaluator-only ground truth, or an invented intermediate objective. The investigation head would duplicate the decision head's action choice. More complexity without a signal to train it. |
| One flat policy with no explicit detection component | Possible, but then rule-based and learned policies would not share the detection summary and the dashboard would have no detection feed. The deterministic `suspicion` feature is cheap, inspectable and identical for all compared policies. |

Consequence: the compared policies receive the same detection summary, evidence view, observation and action masks, so differences come mainly from the decision role. One qualification: the `service_health` observation channel is a weakened compromise proxy (on the mock, compromised factor 0.9, `service_health` noise 0.08, edge offset 0.05, with a mock AUROC guard so the channel is informative but not an oracle; RL_IMPLEMENTATION section 3 and DESIGN section 5.3). A learned policy may read that channel while the rule baseline does not, so the policies are not strictly identical in their use of inputs; the weakening limits, but does not fully remove, that gap.

### 6.3 Coordination is not a policy

The coordinator validates every proposal against schema, identifiers, an explicit allowlist, the agent's authority, the simulator's available actions, conflicts and a per-step high-impact budget. It never learns, never modifies a proposal, and is the only caller of `Simulator.apply_action`. Policies cannot reach the simulator's action API, so a faulty policy cannot bypass it.

### 6.4 Uncertainty-aware variant

The uncertainty mechanism wraps an IPPO policy in the decision role. It reads the policy's action distribution and the evidence view, and either passes the proposal through, replaces it with an evidence request, or replaces a high-impact action with a less disruptive one. The matched control is the same wrapper with the rule disabled. Details: RL_IMPLEMENTATION section 7.

### 6.5 MAPPO status

Planned as its own implementation stage after IPPO, the uncertainty mechanism and the evaluation path are stable (DECISIONS AD-10). The design is complete in RL_IMPLEMENTATION section 6: per-agent decentralized actors on local observations, one centralized team critic over the concatenation of the agents' agent-visible observation vectors (the `env.state()` hook, no ground truth), one shared GAE and the shared team reward. IPPO is never relabelled as MAPPO. The stage runs work items W1-W7 and acceptance tests M-1 to M-13, and defers MAPPO only if it finds a concrete, evidenced blocker recorded with command output. Until that stage runs there is no `marl/mappo.py` and the run manifest reports `mappo: not_run`; the manifest status is computed from what actually executed, never hard-coded.

## 7. Data flow

Required flow: Scenario configuration, simulator, observations and events, detection and investigation, RL policy, defense coordinator, validated simulated action, updated simulator state, outcome evaluation, dashboard. The graph pipeline sits inside the observation step. The numbered phases are the simulation-step phases of CONTRACTS section 5.

| Phase | Flow | Records |
|-------|------|---------|
| Setup | `ScenarioConfig` and seed enter `Simulator.reset`. The run directory receives the topology, scenario and manifest. | `ScenarioConfig`, `Topology` |
| P0 dynamics | The simulator advances exogenous dynamics: normal activity, incident progress, alert generation, restore completion. | `EventRecord` (phase `dynamics`) |
| P1 observe | `Simulator.observe()` returns one `AgentObservation` and one `AvailableActions` per agent. The detection role produces `DetectionSummary`. The graph converter produces `GraphSnapshot` and `GraphFeatureRecord` from the observation and topology only. The engine assembles a `DecisionContext` per agent. | `AgentObservation`, `AvailableActions`, `DetectionSummary`, `GraphSnapshot`, `GraphFeatureRecord` |
| P2 propose | Each agent's policy returns one `PolicyProposal` using only its `DecisionContext`. The uncertainty wrapper, if present, runs here. | `PolicyProposal` |
| P3 validate | The coordinator validates proposals in ascending agent order. Rejections change nothing. | `ValidationResult` |
| P4 execute | The coordinator sends accepted actions to `Simulator.apply_action`. Effects apply at once. The recovery verifier may add events. | `ExecutedAction`, `EventRecord` (phase `outcome`) |
| Reward (training and evaluation) | After P4 the reward model compares the end-of-step ground truth with the previous step's and scores the executed actions. The environment returns the scalar reward with the next observation. | `GroundTruthSnapshot` (evaluator-only file), reward breakdown (training log only) |
| End of step | `Simulator.end_step()` decides termination and truncation and advances to the next step. | `episode_ended` event on the last step |
| Evaluation | After the episode, `evaluation.outcome` computes the outcome block from agent-visible records plus ground truth. The harness aggregates `EvaluationMetrics` and `ComparisonRecord` over matched (scenario, seed) pairs. | `EpisodeResult`, `EvaluationMetrics`, `ComparisonRecord` |
| Dashboard | The dashboard loads the run directory. It reads records as written. | all of CONTRACTS section 15 |

Observation representations (matched comparison): `conventional` uses the per-node observation features and the detection `suspicion`; `graph_features` uses the same plus the eight hand-engineered graph features. The policy architecture is otherwise identical (RL_IMPLEMENTATION section 3).

## 8. Error handling and failure behavior

Every operation that can fail has one defined behavior. "Level" is the log level of the `LogRecord`; `fatal` errors are logged at `error` and stop the run. `ErrorRecord` is written to the episode's `errors.jsonl` (or the run log when no episode is active). Training runs fail fast: where a row says "evaluation falls back", training raises instead, because a silent fallback would corrupt learning.

| Component and operation | Failure condition | Class | Caller receives | Logged |
|-------------------------|-------------------|-------|-----------------|--------|
| Contracts: load a record or file | Parse error, unknown field, bad enum, bad id, non-finite number | Fatal for that file | `ContractError` with `ErrorCode.SCHEMA_INVALID`, field path and message | error |
| Contracts: version check | Record MAJOR differs, or MINOR higher than the reader | Fatal for that file | `ContractError(SCHEMA_VERSION_MISMATCH)` | error |
| Config: load TOML or scenario | Missing file, invalid value, unknown key | Fatal before the run starts | `ConfigError(CONFIG_INVALID)` | error |
| Simulator `reset` | Scenario refers to an unknown topology or node, or uses a field the simulator cannot honour | Episode fatal, run continues | `SimulatorError(SCENARIO_UNSUPPORTED)`; runner writes `ErrorRecord`, skips the episode, counts it as aborted | error |
| Simulator `observe`, `available_actions`, `end_step` | Internal exception or invariant violation | Episode fatal | `SimulatorError(SIMULATOR_FAILURE)`; runner aborts the episode, keeps partial records, writes `ErrorRecord` | error |
| Simulator `apply_action` | Action not applicable in the current state | Recoverable | Returns a failed `ActionApplication`; the state is unchanged (digest equal); coordinator writes `ExecutedAction(status=failed, ACTION_EXECUTION_FAILED)` | warning |
| Coordinator `process` | Proposal does not parse or fails any check | Recoverable (normal outcome) | `ValidationResult(decision=rejected, reason_codes)`; no state change; the agent gets the rejection in `last_step_feedback` | info for expected rejections; warning plus `ErrorRecord(SCHEMA_INVALID)` for unparseable input |
| Coordinator `process` | Called with the wrong step or no active episode | Caller bug | Every proposal rejected (`WRONG_STEP`, `EPISODE_NOT_ACTIVE`) | warning |
| Roles: detection, evidence | Observation fails validation | Fatal for the step | `ContractError` | error |
| Graph converter | A node or edge id in the observation is not in the topology; shape or mask mismatch; non-finite value | Episode fatal | `GraphConversionError(UNKNOWN_ID or INVARIANT_VIOLATION)`; nothing is imputed silently | error |
| Policy `propose` (rule or learned) | Exception or invalid proposal in evaluation | Recoverable in evaluation | Engine substitutes a `no_op` proposal with `rationale_text` "Policy failure fallback" and writes `ErrorRecord(POLICY_FAILURE)`. In training the exception propagates. | error |
| Graph-feature policy | Context has no graph features | Configuration bug | `PolicyError(POLICY_FAILURE)` at the first step | error |
| RL environment `step` | Called before `reset` or after the episode ended | Caller bug | `EnvStateError` | none (raises) |
| RL environment `step` | Action index outside the action space, or wrong agent keys | Caller bug | `ValueError` | none (raises) |
| RL environment `step` | Action index inside the space but masked (unavailable) | Recoverable | The environment replaces it with `no_op`, adds the `invalid_action` penalty, and sets `info[agent]["invalid_action"] = True`. The coordinator never sees the masked proposal. | debug |
| Reward model | Ground-truth accessor missing when a reward is requested | Configuration bug | `ConfigError` at environment construction | error |
| IPPO training | Non-finite loss, gradient or advantage | Fatal for the training run | `TrainingError(INVARIANT_VIOLATION)`; no checkpoint is written from that update | error |
| IPPO checkpoint save | Disk error | Fatal | `CheckpointError(IO_ERROR)`; a partial file is removed | error |
| IPPO checkpoint load | Missing file, hash mismatch, shape or configuration mismatch | Fatal for the load | `CheckpointError(CHECKPOINT_ERROR)`; there is no fallback to random weights | error |
| Uncertainty signal | Non-finite logits or an empty valid-action set | Recoverable | `uncertainty.value = null`, `rule_outcome = unavailable_pass_through`; proposal unchanged; `ErrorRecord(UNCERTAINTY_UNAVAILABLE)` | warning |
| Calibration | Fewer than the minimum episodes on the calibration split, or the signal fails the informativeness gate | Reported | `CalibrationReport` with `status = insufficient_data` or `not_informative`; `uncertainty_ece` stays `null` when it cannot be computed | warning |
| Evaluation | A record fails validation or an episode is aborted | Aborted episodes are counted | `n_aborted` in `EvaluationMetrics`; aborted episodes do not enter the means | warning |
| Evaluation | Two approaches have different matched sets | Comparison not produced | No `ComparisonRecord`; `ErrorRecord(INVARIANT_VIOLATION)` with severity `error` | error |
| Export | Disk write error | Fatal | `IO_ERROR`; the manifest is written last through a temporary file, so a manifest on disk means the run completed | error |
| Smoke command | Any stage fails | Fatal | Non-zero exit status and a one-line summary of the failed stage; nothing is reported as passed | error |

## 9. Ground-truth separation

Ground truth means the hidden state of the simulation: which nodes are really compromised, which alerts are false, true service availability. The same mechanisms are used everywhere.

| Mechanism | What it guarantees | Test |
|-----------|--------------------|------|
| Separate accessor: `GroundTruthAccessor.ground_truth()` is a different Protocol from `Simulator` | Agent-facing code is typed against `Simulator` only | `tests/interfaces/test_conformance_mock.py` |
| Closed schemas (`extra="forbid"`) | An observation, event, graph record or `info` payload cannot carry a ground-truth field | `tests/contracts/test_models_validation.py`, `tests/rl/test_env_visibility_leakage.py` |
| Import boundary: only `rl.reward`, `engine` (to write the ground-truth stream), `evaluation` and tests import `cyberswarm.contracts.evaluator_only` or call `ground_truth()` | Roles, graph, baselines, marl, uncertainty, coordinator cannot read it | `tests/integration/test_import_boundaries.py` (static scan) |
| Separate directory: ground truth is written only to `evaluator_only/` | Agent-visible files can be scanned for evaluator-only field names | `tests/integration/test_leakage_scan_outputs.py` |
| No future events: observations contain events with `step <= t` only | Decision-time data never contains the future | `tests/contracts/test_fixture_consistency.py` (X-5), conformance suite |
| `info` dictionaries: contain only agent-visible values (step, proposal ids, validation decisions, masks, invalid-action flag); the reward breakdown and the termination reason are not included | Serialized `info` is leak-free | `tests/rl/test_env_visibility_leakage.py` |
| Reward is privileged but not an input: it is computed from ground truth and returned as a scalar. It is a training signal, not an observation. | The training signal exists without exposing state | `tests/rl/test_reward_terms.py` |
| Episode end: `terminated` and `truncated` are visible after the last decision. The termination reason is evaluator data. | One unavoidable bit of end-of-episode information; the reason stays hidden | `tests/rl/test_env_lifecycle.py` |

The graph pipeline and the detection role consume only `Topology` and `AgentObservation`. Their functions have no parameter through which ground truth could be passed, and the static scan above fails the build if they import it.

## 10. Ownership and integration sequence

### 10.1 Raksha: network simulator and attack engine

Owns `src/cyberswarm/simulator/` and her tests and scenarios (not created yet). She delivers topology, node and edge definitions, synthetic normal activity, one abstract incident scenario (state transitions only), event and observation generation, evaluator-only ground truth, simulated containment effects, seeded reproducibility, tests and documentation. Her simulator implements `interfaces.Simulator` and the `GroundTruthAccessor`, and must pass `cyberswarm.interfaces.conformance`. She does not own the observation-to-graph conversion, graph features, RL, policies, the coordinator or the evaluator. She does not import lead-owned internals; she depends on `contracts` and `interfaces` only. Specified, not implemented.

### 10.2 Mayukh Aryan: dashboard and visualization

Owns `dashboard/` (not created yet). He delivers the topology view, node status, alert and incident timeline, agent activity feed, proposed versus validated versus executed action display, attack status and recovery, episode summary, baseline-versus-learned comparison, fixture mode and later integration mode behind one `DataSource` interface, and tests. He consumes `schemas/` and `fixtures/bundle_demo/` and later `runs/<run_id>/`. He never invents values. Recommended stack: React, Vite, TypeScript, Cytoscape.js (DECISIONS D-11, unconfirmed). Specified, not implemented.

### 10.3 Project lead

Owns everything else: architecture, contracts, fixtures, research review, rule-based and graph-feature baselines, graph pipeline, RL environment, reward, IPPO, MAPPO decision, uncertainty extension, coordinator and safety, integration, end-to-end tests, evaluation and reporting. None of the RL or graph-learning responsibilities is reassigned.

### 10.4 Integration sequence

| Step | What is integrated | Gate |
|------|--------------------|------|
| 1 | Contracts, schemas, fixtures validate | Contract tests pass |
| 2 | Mock simulator passes the conformance suite | Conformance tests pass on the mock |
| 3 | Graph pipeline over mock observations | Graph tests pass |
| 4 | RL environment over the engine | Environment tests pass in both observation modes |
| 5 | Baselines and coordinator through the engine | Safety tests and baseline tests pass |
| 6 | Evaluator and dashboard-consumable output | Smoke command passes; every emitted file validates against `schemas/` |
| 7 | IPPO and uncertainty on the mock | Train, save, load, infer, evaluate on held-out scenarios |
| 8 | Raksha's simulator replaces the mock | The same conformance suite and the same smoke command with `simulator_kind = simulator`; no change to lead-owned code except configuration |
| 9 | Mayukh's dashboard loads real runs | The same loader, pointed at `runs/<run_id>/`; no architecture change |

## 11. Design principles

Preserved from draft v0.1 and made testable:

1. **Modularity with clear contracts.** One owner per directory, a versioned public interface, no importing another module's private internals, tests and fixtures with each module.
2. **Honesty.** No hard-coded success, no fabricated metrics, limitations documented. Metrics come from evaluator code over real run records. Mock and fixture evidence is labelled and is not a research result.
3. **Reproducibility.** Fixed seed, fixed configuration and the same checkpoint give identical record files, compared by state digest and file bytes (wall-clock log lines excluded).
4. **Ground truth and observation are separate** (section 9).
5. **Incremental complexity.** Small topology, one incident type first. Advanced items (MAPPO, learned graph models) only when the earlier stages are stable and justified.
6. **Safety by construction.** Only the coordinator executes; only allowlisted abstract actions exist; no component can run a command.

## 12. Constraints and safety

In scope: simulated topologies, synthetic incidents, abstract simulated defense actions, RL training and evaluation, visualization.

Out of scope and forbidden anywhere in the repository: attacks or scans on real systems, exploit code, malware, credential handling, real firewall or device changes, real containment, arbitrary command execution, sockets or any network call at runtime, writing files as an action effect. The coordinator package contains no `subprocess`, `os.system`, `socket`, `requests`, `urllib`, `shutil` or `os.remove`/`open(..., "w")` usage; a source-scan test enforces this.

Action allowlist: the eight abstract actions of CONTRACTS section 6 (`no_op`, `monitor_node`, `investigate_node`, `request_evidence`, `rate_limit_node`, `quarantine_node`, `isolate_node`, `restore_node`). Anything else is rejected. Draft v0.1 listed `investigate_host`, `isolate_host`, `restore_host` and `monitor_network`; those names are replaced by the single vocabulary.

Third-party code (CyMARL, CMARL-ACD, CybORG, CAGE-4) is a design reference only. No code is copied. CMARL-ACD's real-testbed tooling is never run, adapted or vendored.

## 13. Technology stack (locked on approval)

| Area | Choice | Reference |
|------|--------|-----------|
| Language | Python, `requires-python >= 3.12`, developed on 3.13.2 | D-01 |
| Packaging | `pyproject.toml`, `src/` layout, repo-local `.venv` (gitignored), `pip` | D-02, D-17 |
| Contracts | pydantic v2 (`>=2.12,<3`), JSON Schema Draft 2020-12 export, JSON and JSONL | D-03, D-04 |
| Numerics | numpy, PyTorch (CPU) | D-07, D-15 |
| Environment API | PettingZoo `ParallelEnv` and Gymnasium spaces | D-05 |
| Graph | NetworkX 3.x for the pipeline and the mock; not imported by `contracts` | D-09, D-18 |
| Multi-agent PPO | Self-written compact PyTorch IPPO; TorchRL's losses are a named contingency and optional parity check | D-06, AD-08 |
| Configuration | TOML read with `tomllib`; scenarios and topologies are validated JSON | D-12 |
| Logging | standard `logging` with JSON lines | D-13 |
| Tests | pytest; `jsonschema` (development only) | D-14 |
| Dashboard (Mayukh) | React, Vite, TypeScript, Cytoscape.js, Vitest, Ajv (recommended, unconfirmed) | D-11, AD-09 |
| Not used | Stable-Baselines3, SuperSuit, RLlib/Ray, BenchMARL, EPyMARL, PyTorch Geometric (staged, optional, not installed), TensorBoard, databases, brokers, HTTP servers | D-08, D-10, D-13 |

## 14. Testability

| Kind | Examples | Where |
|------|----------|-------|
| Unit | id patterns, vocabulary table, reward terms on hand-built states, GAE and PPO loss, entropy and rule boundaries, graph feature formulas, metric formulas, coordinator checks | `tests/<package>/` |
| Contract | every fixture against pydantic models and exported schemas; cross-record rules X-1 to X-14; schema export up to date | `tests/contracts/` |
| Conformance | the same suite against the mock now and Raksha's simulator later | `tests/interfaces/` and `cyberswarm.interfaces.conformance` |
| Integration | smoke command, bundle schema consumption, leakage scan of outputs, import boundaries, determinism | `tests/integration/` |
| Experiment (not tests) | reward sensitivity, representation comparison, uncertainty versus matched control, IPPO versus MAPPO | harness runs; results only from evaluator output |

The design avoids hard-to-test parts: the coordinator and detection are pure and deterministic; policies are invoked through a one-method interface; the environment is a thin adapter over the engine; and the mock gives fast deterministic episodes for every test.

## 15. Implementation status

| Item | Status |
|------|--------|
| Architecture, contracts, RL design (this document set) | Specified |
| Contracts package, schemas, fixtures | Not implemented (stage S1) |
| Simulator Protocol, mock simulator, conformance suite | Not implemented (stage S2) |
| Graph pipeline, RL environment, reward | Not implemented (stage S3) |
| Rule-based and graph-feature baselines, coordinator | Not implemented (stage S4) |
| Evaluator, smoke command, dashboard-consumable output | Not implemented (stage S5) |
| IPPO | Not implemented (stage S6) |
| MAPPO | Not implemented (planned stage after IPPO, uncertainty and evaluation; RL_IMPLEMENTATION section 6) |
| Uncertainty extension and matched control | Not implemented (stage S7) |
| Learned GNN | Not implemented (staged go/no-go taken in its own stage after IPPO, uncertainty and evaluation; RL_IMPLEMENTATION section 8, AD-11) |
| Raksha's real simulator | Not implemented (specification only) |
| Mayukh's dashboard | Not implemented (specification only) |

## 16. Reconciliation with draft v0.1

Kept: executive intent, episode idea, owners (Raksha, Mayukh Aryan, project lead), module responsibilities, design principles, safety stance, JSON and pytest choices, React plus TypeScript recommendation for the dashboard.

Changed:

| v0.1 | Now |
|------|-----|
| Simulator implements `gymnasium.Env` and returns `obs, event, reward, done, info` | Simulator is a separate Protocol with an evaluator-only accessor; the RL environment is an adapter (section 5, DESIGN section 4). The simulator returns no reward. |
| Event record has a `ground_truth` field "only populated by evaluator" | No ground-truth field exists in any agent-visible record (CONTRACTS 8.1, section 9 here). |
| Observation has `process_list`; actions `investigate_host`, `isolate_host`, `restore_host`, `monitor_network` | Abstract per-node observation fields and the eight-action vocabulary. |
| `timestamp` holds a step number | `step` with exact semantics (CONTRACTS section 5). |
| Ray/RLlib listed | Not used. |
| "Root-cause rank accuracy" metric and "explicit root-cause investigation" enhancement | Causal root-cause localization is not a headline task, but a ranking of predicted suspicion is an evaluated metric: origin top-1 accuracy and the mean reciprocal rank of the incident-origin node, per scorer (detection role, graph-feature baseline, and learned models if built), labelled as predicted suspicion and never a causal finding. |
| GNNs optional and unspecified | Graph pipeline required but not central; learned GNN staged, with the go/no-go taken in its own stage after IPPO, uncertainty and evaluation (section 3). |
| CyMARL as the only reference; wrong repository, CybORG and date claims | Corrected (section 2). |
| Python 3.10+, "Pydantic dataclasses" | Python 3.12+, pydantic v2 models with exported JSON Schema (section 13). |

Removed: the claims listed under section 2 as corrected, and the Ray/RLlib technology row.
