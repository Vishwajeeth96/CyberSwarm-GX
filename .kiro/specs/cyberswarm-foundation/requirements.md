# Requirements Document

## Introduction

CyberSwarm is a multi-agent reinforcement-learning (MARL) system for autonomous cyber defense inside a controlled simulator. This spec covers the **project-lead-owned foundation only**. It does not implement Raksha's real network simulator or Mayukh Aryan's dashboard UI; those remain specifications plus shared fixtures.

The detailed, authoritative design already exists and is the source of truth. This document and [design.md](design.md) stay short and link to those documents rather than copying them. Read, in order:

- [`.agents/tasks/cyberswarm-foundation/DELTA.md`](../../../.agents/tasks/cyberswarm-foundation/DELTA.md) (latest authority, including Addenda 1-3) and [`BRIEF.md`](../../../.agents/tasks/cyberswarm-foundation/BRIEF.md)
- [`.agents/tasks/cyberswarm-foundation/DESIGN.md`](../../../.agents/tasks/cyberswarm-foundation/DESIGN.md) (file-by-file inventory and stages)
- [`docs/ARCHITECTURE.md`](../../../docs/ARCHITECTURE.md), [`docs/CONTRACTS.md`](../../../docs/CONTRACTS.md), [`docs/RL_IMPLEMENTATION.md`](../../../docs/RL_IMPLEMENTATION.md) (v0.2, revised), [`docs/DECISIONS.md`](../../../docs/DECISIONS.md)
- [`.agents/tasks/cyberswarm-foundation/gates/design-review.md`](../../../.agents/tasks/cyberswarm-foundation/gates/design-review.md) (blocking review: 4 HIGH, 17 MEDIUM, 11 NIT)

Where RL_IMPLEMENTATION.md v0.2, CONTRACTS.md, and the DELTA addenda disagree with the older DESIGN.md or ARCHITECTURE.md, the corrected documents win. The corrected scope (learned GNN as its own staged decision, MAPPO implemented in its own stage unless an evidenced blocker is found, an unseen-topology evaluation split `eval_utopo`, and an origin-ranking metric) is encoded in [tasks.md](tasks.md).

## Glossary

- **CyberSwarm_Foundation**: the lead-owned package tree under `src/cyberswarm/` plus shared `fixtures/`, `schemas/`, `configs/`, `scripts/`, `tests/`, and `docs/`.
- **Contracts_Package**: `src/cyberswarm/contracts/`, the neutral pydantic v2 record models, identifiers, action vocabulary, and JSON Schema export.
- **Simulator_Interface**: the public `Simulator` and `GroundTruthAccessor` Protocols and the reusable conformance suite in `src/cyberswarm/interfaces/`.
- **Mock_Simulator**: the lead-owned, clearly labelled deterministic simulator in `src/cyberswarm/mock/`, outside any `simulator/` module, implementing only the public interface. It is never described as the real simulator.
- **Graph_Pipeline**: `src/cyberswarm/graph/`, the observation-to-graph conversion, hand-engineered features, and tensor/batch interface.
- **RL_Environment**: `src/cyberswarm/rl/env.py`, the PettingZoo-style parallel environment over the public interface, with `conventional` and `graph_features` observation modes.
- **Reward_Model**: `src/cyberswarm/rl/reward.py`, the configurable reward scorer.
- **Baselines**: the rule-based conventional policy and the non-neural graph-feature policy in `src/cyberswarm/baselines/`.
- **Defense_Coordinator**: `src/cyberswarm/coordinator/`, the sole action-validation and execution boundary.
- **IPPO_Trainer**: the self-written PyTorch Independent-PPO training, checkpoint, and inference code in `src/cyberswarm/marl/`.
- **Uncertainty_Wrapper**: `src/cyberswarm/uncertainty/`, the entropy-based decision rule and its matched control.
- **Evaluation_Harness**: `src/cyberswarm/evaluation/` plus `src/cyberswarm/runner/`, which compute metrics from execution records and emit dashboard-consumable bundles.
- **Learned_Graph_Stage**: the staged go/no-go experiment for learned graph models (`SageDense`, `GatDense`) described in RL_IMPLEMENTATION.md section 8.
- **MAPPO_Stage**: the staged multi-agent PPO implementation described in RL_IMPLEMENTATION.md section 6.
- **Ground_Truth**: evaluator-only state (`GroundTruthSnapshot`), never present in agent-visible observations, graph records, policy inputs, serialized info, or future events.
- **Contract_Valid**: validates against both the pydantic model and the exported JSON Schema for its record type.

## Requirements

### Requirement 1: Neutral versioned contracts and schema export

**User Story:** As the project lead, I want a neutral, versioned contracts package with exported JSON Schemas, so that all components and both teammates share one validated data vocabulary.

#### Acceptance Criteria

1. THE Contracts_Package SHALL define the record types enumerated in [`docs/CONTRACTS.md`](../../../docs/CONTRACTS.md) as frozen pydantic v2 models with `extra="forbid"`, a `SCHEMA_VERSION` constant, stable identifier types, and a single abstract action vocabulary.
2. WHEN `python -m cyberswarm.contracts.export_schemas --out schemas` is run, THE Contracts_Package SHALL write one `schemas/<Name>.schema.json` per registered record type, byte-identical to the committed schema files.
3. IF a record carries a non-finite number, an unknown enum value, an extra field, or an identifier that resembles an IP address or path, THEN THE Contracts_Package SHALL raise a `ContractError`.
4. WHERE a producer did not compute an uncertainty, graph-model, reward, or comparison value, THE Contracts_Package SHALL represent that value as `null` and preserve the `null` across a JSON round trip.

### Requirement 2: Synthetic fixtures and contract tests

**User Story:** As the project lead, I want synthetic fixtures and contract tests, so that every record shape and cross-record rule is machine-validated before dependent code exists.

#### Acceptance Criteria

1. THE Contracts_Package SHALL provide synthetic fixtures, labelled as synthetic and illustrative rather than results, covering healthy and changing state, masks and imputation, conventional and graph observations, available actions, accepted and rejected coordinator paths, and episode outcomes.
2. WHEN the contract test suite runs, THE Contracts_Package SHALL validate every valid fixture against the pydantic model and the JSON Schema, and SHALL confirm that each invalid fixture fails for its one declared reason.
3. THE Contracts_Package SHALL enforce the cross-record consistency rules of [`docs/CONTRACTS.md`](../../../docs/CONTRACTS.md), including identifier resolution, topology resolution, and that no agent-visible record contains a Ground_Truth field name or a future event.
4. WHEN `scripts/build_fixtures.py --check` is run, THE Contracts_Package SHALL report zero drift between committed fixtures and regenerated fixtures.

### Requirement 3: Public simulator interface and conformance suite

**User Story:** As the project lead, I want a public simulator interface and a reusable conformance suite, so that the mock and Raksha's future simulator are interchangeable and verifiable.

#### Acceptance Criteria

1. THE Simulator_Interface SHALL declare the `Simulator` and `GroundTruthAccessor` Protocols and a conformance suite covering the checks SC-1 through SC-15 defined in the reconciled design.
2. THE Simulator_Interface SHALL expose agent-visible data only through the agent-facing methods and SHALL keep Ground_Truth behind the separate `GroundTruthAccessor`.
3. THE Simulator_Interface SHALL define exact step semantics, including that `GroundTruthAccessor.ground_truth()` returns the end-of-step-`t` snapshot captured before `end_step()` runs phase P0 of step `t+1`.

### Requirement 4: Lead-owned mock simulator

**User Story:** As the project lead, I want a deterministic lead-owned mock simulator, so that the RL environment, coordinator, uncertainty, and smoke paths run before Raksha delivers.

#### Acceptance Criteria

1. THE Mock_Simulator SHALL implement only the Simulator_Interface, SHALL reside in `src/cyberswarm/mock/` outside any `simulator/` module, and SHALL declare itself a mock in its module docstring and manifest field.
2. WHEN `Mock_Simulator.reset` is called twice with the same scenario and seed, THE Mock_Simulator SHALL produce identical observation, event, and `state_digest` sequences; and a different seed SHALL change at least the digest sequence.
3. IF an action is out of authority or inapplicable, THEN THE Mock_Simulator SHALL return a `failed` application and leave `state_digest()` unchanged.
4. THE Mock_Simulator SHALL pass the Simulator_Interface conformance suite and SHALL provide one held-out unseen-topology variant (`topo-mock-small-v2`) that is never used in training or calibration.

### Requirement 5: Observation-to-graph pipeline

**User Story:** As the project lead, I want a deterministic observation-to-graph pipeline before the RL environment, so that graph intelligence is required infrastructure with inspectable features.

#### Acceptance Criteria

1. WHEN the Graph_Pipeline converts an `AgentObservation`, THE Graph_Pipeline SHALL produce a Contract_Valid `GraphSnapshot` with stable node identifiers, deterministic node and tensor ordering, aligned missing-data masks, and the documented `impute-v1` values.
2. THE Graph_Pipeline SHALL compute hand-engineered `graph-derived-v1` features from the snapshot and topology only, consuming agent-visible observations and current or past events exclusively.
3. IF Ground_Truth or a future event is supplied, THEN THE Graph_Pipeline SHALL exclude it, and the converter and feature functions SHALL expose no Ground_Truth parameter.
4. THE Graph_Pipeline SHALL expose a model-compatible tensor and batch interface with the dtypes and shapes defined in [`docs/CONTRACTS.md`](../../../docs/CONTRACTS.md) without requiring a learned-GNN dependency.

### Requirement 6: Multi-agent RL environment

**User Story:** As the project lead, I want a multi-agent RL environment over the public interface with two observation modes, so that conventional and graph representations are compared under matched conditions.

#### Acceptance Criteria

1. THE RL_Environment SHALL adapt the public Simulator_Interface through the episode engine and SHALL NOT import any simulator implementation.
2. WHERE the observation mode is `conventional` or `graph_features`, THE RL_Environment SHALL expose the documented observation dimensions (138 and 202 on the mock) with values in [0, 1].
3. WHEN `reset` is called with an integer seed, THE RL_Environment SHALL produce a deterministic trajectory; and WHEN a step supplies an in-range but masked action index, THE RL_Environment SHALL substitute `no_op`, flag `invalid_action`, and keep `no_op` always available.
4. THE RL_Environment SHALL report `terminated` versus `truncated` per the step semantics and SHALL exclude Ground_Truth, reward breakdown, future events, and termination reason from observations and the info dictionary.
5. THE RL_Environment SHALL provide a `state()` hook returning the ascending-agent-order concatenation of agent-visible observation vectors, for later use by the MAPPO_Stage centralized critic.

### Requirement 7: Configurable reward

**User Story:** As the project lead, I want a configurable reward with hand-constructed tests, so that containment, disruption, and investigation trade-offs are explicit and reproducible.

#### Acceptance Criteria

1. THE Reward_Model SHALL compute the terms containment, response delay, criticality-weighted attack impact, false isolation, healthy-service disruption, and investigation cost with configurable non-negative weights and a recorded `weights_id`.
2. THE Reward_Model SHALL charge healthy-service disruption using defense availability (`GroundTruthSnapshot.defense_availability`, compromise factor fixed at 1.0) so that attack-induced loss is scored only under attack impact and not under healthy disruption.
3. WHEN the hand-constructed reward cases R-1 through R-12 of [`docs/RL_IMPLEMENTATION.md`](../../../docs/RL_IMPLEMENTATION.md) section 4.3 run, THE Reward_Model SHALL return the stated values, including that isolating a healthy critical node costs strictly more than a low-value node and that justified containment scores higher than inaction.
4. THE Reward_Model SHALL store unweighted per-term magnitudes so that `evaluation.reweight` recomputes episode reward for any weight vector without re-running episodes.

### Requirement 8: Conventional and graph-feature baselines

**User Story:** As the project lead, I want conventional and graph-feature baselines, so that learned policies have matched non-learned references that emit per-node suspicion scores.

#### Acceptance Criteria

1. THE Baselines SHALL each emit Contract_Valid proposals, SHALL select only actions present in `AvailableActions`, and SHALL investigate when evidence is insufficient rather than isolating.
2. THE graph-feature baseline SHALL consume the hand-engineered graph features and SHALL raise a policy error when those features are absent; and the conventional baseline SHALL ignore perturbed graph features.
3. THE Baselines SHALL record per-node suspicion scores as `NodeScoreRecord` so that the Evaluation_Harness can compute origin-ranking metrics.
4. WHILE a contained node remains compromised with null, weak, or moderate evidence, THE rule-based baseline SHALL re-investigate it so that the episode can reach `incident_resolved`.

### Requirement 9: Safe defense coordinator

**User Story:** As the project lead, I want a safe defense coordinator as the sole action boundary, so that only validated, allowlisted actions reach the simulator and rejected proposals change nothing.

#### Acceptance Criteria

1. THE Defense_Coordinator SHALL validate proposal schemas, resolve node identifiers, enforce an explicit action allowlist, and emit structured reason codes evaluated in the documented dependency order.
2. IF a proposal is malformed, unknown, disallowed, or out of authority, THEN THE Defense_Coordinator SHALL reject it with reason codes and SHALL leave `state_digest()` unchanged.
3. THE Defense_Coordinator SHALL execute accepted actions only through the public simulator API and SHALL contain no shell, subprocess, socket, network, or file-mutation call as an action effect.
4. THE Defense_Coordinator SHALL emit Contract_Valid proposal, validation, execution, and outcome records and SHALL verify recovery from simulator-reported data using the documented `recovery_verified` and `recovery_failed` rules.

### Requirement 10: Self-written IPPO

**User Story:** As the project lead, I want a compact self-written PyTorch IPPO, so that independent learned policies can train, checkpoint, and infer with contract-valid proposals.

#### Acceptance Criteria

1. THE IPPO_Trainer SHALL give each agent its own actor, critic, optimizer, and value normalizer with no shared parameters or gradients.
2. WHEN training runs with a fixed seed, THE IPPO_Trainer SHALL write training logs, save a checkpoint to a gitignored directory, load it with hash and action-table verification, and run greedy evaluation.
3. THE IPPO_Trainer SHALL return Contract_Valid proposals that the Defense_Coordinator accepts for unmasked actions.
4. THE IPPO_Trainer SHALL label all mock-trained numbers as mock-only and SHALL make no claim of learned-policy quality.

### Requirement 11: Uncertainty-aware wrapper and matched control

**User Story:** As the project lead, I want an uncertainty-aware decision wrapper with a matched control, so that entropy-based deferral can be tested and calibrated against a policy that differs only in the rule.

#### Acceptance Criteria

1. THE Uncertainty_Wrapper SHALL compute normalized policy entropy as its signal and SHALL represent an unavailable signal as `null` with an `ErrorRecord` and a pass-through outcome.
2. WHEN the decision rules U0 through U6 of [`docs/RL_IMPLEMENTATION.md`](../../../docs/RL_IMPLEMENTATION.md) section 7.2 apply, THE Uncertainty_Wrapper SHALL request evidence under low evidence, defer high-impact actions under uncertainty, and allow containment under strong evidence.
3. THE Uncertainty_Wrapper SHALL provide a matched control that differs from the aware policy only in the decision rule and SHALL produce identical proposals in every other field.
4. THE Uncertainty_Wrapper SHALL support calibration on the held-out calibration split using evaluator-supplied correctness arrays, without importing Ground_Truth.

### Requirement 12: Evaluation harness and smoke path

**User Story:** As the project lead, I want a reproducible evaluation harness and an end-to-end smoke path, so that metrics come from real records and feed dashboard-consumable bundles.

#### Acceptance Criteria

1. THE Evaluation_Harness SHALL compute metrics from actual execution records, including precision and recall where defined, false-positive rate, detection-to-containment time, attack impact, compromised nodes, healthy-service disruption separated from attack-induced loss, false-isolation count, episode reward, and origin-ranking top-1 accuracy and mean reciprocal rank.
2. THE Evaluation_Harness SHALL evaluate matched scenarios and seeds across approaches on disjoint `train`, `cal`, `eval`, and held-out `eval_utopo` splits, SHALL report `eval` and `eval_utopo` separately, and SHALL keep failed or underperforming runs in the output.
3. IF a split overlap or a tuning use of `eval` or `eval_utopo` is attempted, THEN THE Evaluation_Harness SHALL fail the disjointness assertion.
4. WHEN `python -m cyberswarm.runner.smoke --config configs/experiments/smoke.toml` is run, THE Evaluation_Harness SHALL exit 0, validate every emitted file against `schemas/`, label output as a mock run that is not a research result, and emit dashboard-consumable bundles that preserve unavailable values as `null`.

### Requirement 13: Staged learned-graph model

**User Story:** As the project lead, I want the learned-graph model as its own staged go/no-go, so that GraphSAGE-style and GAT-style models are decided on evidence after IPPO is stable.

#### Acceptance Criteria

1. THE Learned_Graph_Stage SHALL run only after the Graph_Pipeline, RL_Environment, Baselines, Defense_Coordinator, IPPO_Trainer, Uncertainty_Wrapper, and the conventional-versus-graph-feature ablation are stable.
2. THE Learned_Graph_Stage SHALL implement plain-PyTorch `SageDense` and `GatDense` models over the existing graph batch for node-suspicion scoring, and optionally learned embeddings as an IPPO observation mode.
3. WHERE the documented go criteria G1 through G5 hold, THE Learned_Graph_Stage SHALL implement, train, checkpoint, and evaluate the models and SHALL run the conventional-versus-graph-feature-versus-learned-embedding ablation on identical `eval` and `eval_utopo` pairs.
4. IF a go criterion fails, THEN THE Learned_Graph_Stage SHALL record an evidenced no-go `StageDecision` with the measured value, and no-measurable-improvement SHALL be reported as a result rather than a no-go.

### Requirement 14: Staged MAPPO

**User Story:** As the project lead, I want MAPPO implemented in its own stage unless an evidenced blocker is found, so that a genuine centralized-critic comparison with IPPO exists.

#### Acceptance Criteria

1. THE MAPPO_Stage SHALL run after the IPPO_Trainer, Uncertainty_Wrapper, Evaluation_Harness, and Learned_Graph_Stage are stable.
2. THE MAPPO_Stage SHALL use decentralized actors on local observations and one centralized critic over the concatenation of agent-visible observations with no evaluator-only data, driven by the shared team reward, and SHALL NOT relabel IPPO as MAPPO.
3. WHEN the MAPPO_Stage runs, THE MAPPO_Stage SHALL pass tests M-1 through M-13, round-trip a checkpoint, return Contract_Valid inference proposals, and run an equal-budget multi-seed comparison with IPPO on `eval` and `eval_utopo`.
4. IF a concrete blocker is found, THEN THE MAPPO_Stage SHALL record a `deferred` `StageDecision` with command-output evidence, and the manifest status SHALL equal what actually executed.

### Requirement 15: Design-document reconciliation

**User Story:** As the project lead, I want the design documents reconciled to the corrected scope early, so that implementation follows one consistent, non-contradictory specification.

#### Acceptance Criteria

1. THE CyberSwarm_Foundation SHALL reconcile [`docs/ARCHITECTURE.md`](../../../docs/ARCHITECTURE.md) and [`docs/DECISIONS.md`](../../../docs/DECISIONS.md) with [`docs/RL_IMPLEMENTATION.md`](../../../docs/RL_IMPLEMENTATION.md) v0.2 and the DELTA addenda, so that the learned GNN is staged, MAPPO is planned, and the graph pipeline is required rather than optional.
2. THE CyberSwarm_Foundation SHALL record corrected decisions resolving the review's HIGH and MEDIUM findings, including Ground_Truth timing relative to `end_step()`, healthy-disruption separation via defense availability, weakening of the `service_health` compromise proxy, the event-visibility table, the unseen-topology split, the origin-ranking metric, and execution-derived status fields.
3. THE CyberSwarm_Foundation SHALL make every repository configuration file and documented command repository-relative, with no machine-specific absolute path in shared content.

### Requirement 16: Teammate specs, AI prompts, and documentation

**User Story:** As the project lead, I want teammate specifications, personalized AI prompts, and reconciled project documentation, so that Raksha and Mayukh Aryan can build their modules against stable boundaries.

#### Acceptance Criteria

1. THE CyberSwarm_Foundation SHALL produce `docs/MODULE_RAKSHA_SIMULATOR.md` and `docs/MODULE_MAYUKH_DASHBOARD.md` as specifications plus fixtures, without implementing either teammate module.
2. THE CyberSwarm_Foundation SHALL produce `docs/AI_PROMPT_RAKSHA.md`, `docs/AI_PROMPT_MAYUKH.md`, `docs/AI_PROMPT_PROJECT_LEAD.md`, and `AGENTS.md`.
3. WHEN `scripts/validate_docs.py` runs, THE CyberSwarm_Foundation SHALL confirm that every relative link and referenced path resolves, every `json contract=` example validates, and documented commands exist, across `README.md`, `docs/INTEGRATION.md`, `docs/TESTING.md`, `docs/EVALUATION.md`, `docs/DEMO.md`, `docs/TEAM_WORKFLOW.md`, and the module and prompt documents.
4. THE CyberSwarm_Foundation SHALL make documentation status words agree with the manifest status derived from actual execution.

### Requirement 17: Safety, git, and environment guardrails

**User Story:** As the project lead, I want strict safety, git, and environment guardrails, so that the work stays defensive-only, reversible, and within disk limits.

#### Acceptance Criteria

1. THE CyberSwarm_Foundation SHALL represent every attack, containment, isolation, and recovery as an abstract simulator state transition, with no real-network, real-device, shell, subprocess, socket, network, or filesystem action effect anywhere.
2. THE CyberSwarm_Foundation SHALL keep `.venv/`, `checkpoints/`, `runs/`, logs, `__pycache__/`, `node_modules/`, and `.env` out of git, and SHALL add no secrets.
3. THE CyberSwarm_Foundation SHALL NOT commit, push, change remotes, branches, collaborators, or git identity, and the previously guessed git email SHALL be re-verified by the user before any commit.
4. WHEN preparing to install PyTorch, THE CyberSwarm_Foundation SHALL run `df -h /Users/guru`, SHALL stop and report a blocker if free space is under 1.5 GB, and SHALL otherwise install with `pip install --no-cache-dir`.
5. THE CyberSwarm_Foundation SHALL report only actually-executed test results with observed pass, fail, and skip counts, and SHALL never report fabricated results.
