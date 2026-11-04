# CyberSwarm Implementation Blueprint (DESIGN)

**Version:** 1.0 (design step output). **Audience:** the coder and reviewer steps that implement stages S1 to S9. **Status:** SPECIFIED, NOT IMPLEMENTED. Nothing in this blueprint has been built or run.
**Repository:** `/Users/guru/CyberSwarm-GX`, branch `main`. Work in place. Use absolute paths. Do not commit, push, or change git configuration.
**Governing documents (read in this order):** `.agents/tasks/cyberswarm-foundation/DELTA.md`, `BRIEF.md`, then `docs/ARCHITECTURE.md`, `docs/CONTRACTS.md`, `docs/RL_IMPLEMENTATION.md`, `docs/DECISIONS.md` (AD-04 to AD-15), `docs/RESEARCH.md`. Where this blueprint and those documents differ, the documents above win and this file is corrected.

## Contents

1. [Rules for implementers](#1-rules-for-implementers)
2. [Directory tree and owners](#2-directory-tree-and-owners)
3. [Tooling, dependencies and import rules](#3-tooling-dependencies-and-import-rules)
4. [Module APIs](#4-module-apis)
5. [MOCK simulator scope](#5-mock-simulator-scope)
6. [Configuration files](#6-configuration-files)
7. [Smoke command and dashboard-consumable outputs](#7-smoke-command-and-dashboard-consumable-outputs)
8. [Fixture list](#8-fixture-list)
9. [DELTA check register and test files](#9-delta-check-register-and-test-files)
10. [Cross-check matrix](#10-cross-check-matrix)
11. [Implementation stages S1 to S9](#11-implementation-stages-s1-to-s9)
12. [Verification commands, risks and open items](#12-verification-commands-risks-and-open-items)

---

## 1. Rules for implementers

1. **Direction.** CyberSwarm is multi-agent RL for autonomous defense inside a simulator. The lead-owned graph pipeline is required infrastructure that precedes the RL environment, and it is not the central claim. A learned GNN is NO-GO in this phase (DECISIONS AD-11). MAPPO is deferred (AD-10) and must not appear as code, as a stub, or as an implemented claim.
2. **Source conflict.** The step instructions for this design described the graph pipeline as optional. The current `DELTA.md` (19:49) makes it required. This blueprint follows `DELTA.md` (AD-15). If the user reverses this, remove the files named in AD-15 and the graph rows of sections 9 and 10.
3. **Mandatory order inside the stages** follows `DELTA.md`: contracts and fixtures, Simulator Protocol and MOCK; graph conversion and graph tests; RL environment and reward; baselines; coordinator; IPPO; uncertainty; evaluation and smoke; GNN decision; MAPPO decision; documentation reconciliation; final review. The stage names S1 to S9 below keep this order, with the graph pipeline as the first part of S3 and the coordinator as the second part of S4. Because the environment (S3) precedes the coordinator (S4), S3 tests use a test double for the coordinator port; S4 adds the real coordinator and a test that runs the environment through it.
4. **Safety.** No shell, subprocess, socket, network call, file mutation as an action effect, real address, command or payload anywhere. All actions are abstract simulator state transitions. The coordinator is the sole caller of `Simulator.apply_action`.
5. **Honesty.** The MOCK is labelled `mock` everywhere and is never called the real simulator. Mock and fixture outputs are never reported as research results. No novelty, superiority or causal claim. Report exact commands and observed pass, fail and skip counts.
6. **Preservation.** `README.md`, `docs/RESEARCH.md` and `docs/DECISIONS.md` exist and are changed only as stated (S8 for the README and reconciliation edits). Do not delete or overwrite unrelated work.
7. **Ownership.** Do not create `src/cyberswarm/simulator/` or `dashboard/` (no placeholders). Do not assign graph or RL work to Raksha or Mayukh Aryan.
8. **Disk.** About 650 MiB was free when last measured. PyTorch needs about 0.7 to 1 GB during installation. S1 must check free space (`df -h /Users/guru/CyberSwarm-GX`) before creating `.venv` and must report a blocker instead of installing if there is less than about 1.5 GB. Use `pip install --no-cache-dir`.
9. **Git identity.** The local identity was set with a guessed email. Do not reuse it, do not commit. The user verifies it first (AD-14).
10. **License.** Do not add a `LICENSE` file or license metadata (AD-13).

## 2. Directory tree and owners

Single source of truth for paths. Section 11 lists the same files by stage; section 9 lists the test files with their checks. `.gitignore` ignores `.venv/`, `runs/`, `checkpoints/`, `__pycache__/`, `.pytest_cache/`, `*.pt`, `node_modules/`, `.env`, `*.log`, `datasets/`.

<!--GEN:TREE-->
| Path | Owner | Stage | Purpose |
|------|-------|-------|---------|
| `pyproject.toml` | Project lead | S1 | Package metadata, `requires-python >= 3.12`, dependency groups, pytest configuration, `cyberswarm` console entry points |
| `.gitignore` | Project lead | S1 | Ignores `.venv/`, `runs/`, `checkpoints/`, `__pycache__/`, `.pytest_cache/`, `*.pt`, `node_modules/`, `.env`, logs, datasets |
| `requirements-tested.txt` | Project lead | S1 | `pip freeze` output after the first successful install (records tested versions) |
| `scripts/fixture_storyboard.py` | Project lead (neutral) | S1 | Literal storyboard data (steps, alerts, proposals, decisions) from which fixtures are built; independent of mock, graph and coordinator code |
| `scripts/build_fixtures.py` | Project lead (neutral) | S1 | Builds and `--check`s all fixtures deterministically; S1 builds record fixtures, S3 adds graph records, S5 adds metrics, comparison and bundles |
| `scripts/graph_evidence.py` | Project lead | S3 | Prints fixture to validated snapshot to features to tensor shapes, dtypes and masks (focused graph evidence command) |
| `scripts/validate_docs.py` | Project lead | S9 | Documentation validator: relative links, referenced paths, `json contract=` examples against schemas, documented commands |
| `README.md` | Project lead | S8 | Rewritten around the confirmed MARL direction (exists; reconciled, not recreated) |
| `docs/INTEGRATION.md` | Project lead | S8 | Integration sequence and interface boundaries among simulator, graph pipeline, env, roles, coordinator, evaluator, dashboard; fixture-to-real swap; checklist |
| `docs/TESTING.md` | Project lead | S8 | Test strategy, safeguards, acceptance criteria per module, commands actually run |
| `docs/EVALUATION.md` | Project lead | S8 | Experimental design, metric definitions, statistics, ablations, threats to validity |
| `docs/DEMO.md` | Project lead | S8 | Ten-step expo demo; live versus prerecorded labelled |
| `docs/TEAM_WORKFLOW.md` | Project lead | S8 | Branching, review, change-request and handoff process for the three contributors |
| `docs/MODULE_RAKSHA_SIMULATOR.md` | Project lead | S9 | Specification of Raksha's simulator module (specified, not implemented) |
| `docs/MODULE_MAYUKH_DASHBOARD.md` | Project lead | S9 | Specification of Mayukh Aryan's dashboard module (specified, not implemented) |
| `docs/AI_PROMPT_RAKSHA.md` | Project lead | S9 | Copy-paste prompt for Raksha's coding agent |
| `docs/AI_PROMPT_MAYUKH.md` | Project lead | S9 | Copy-paste prompt for Mayukh Aryan's coding agent |
| `docs/AI_PROMPT_PROJECT_LEAD.md` | Project lead | S9 | Copy-paste prompt for the project lead's coding agent |
| `docs/AI_INSTRUCTIONS.md` | Project lead | S9 | Persistent repository-level guidance for any coding agent |
| `AGENTS.md` | Project lead | S9 | Generic agent instructions with directory ownership |
| `CLAUDE.md` | Project lead | S9 | Optional one-line pointer to `AGENTS.md` (created only if it does not conflict) |
| `configs/topologies/topo-mock-small-v1.json` | Project lead (neutral) | S1 | Reference topology for the MOCK: 8 nodes, 2 zones, 19 edges |
| `configs/scenarios/mock/scn-mock-lateral-train-01.json` | Project lead (MOCK) | S1 | Training scenario 1 |
| `configs/scenarios/mock/scn-mock-lateral-train-02.json` | Project lead (MOCK) | S1 | Training scenario 2 |
| `configs/scenarios/mock/scn-mock-lateral-train-03.json` | Project lead (MOCK) | S1 | Training scenario 3 |
| `configs/scenarios/mock/scn-mock-lateral-train-04.json` | Project lead (MOCK) | S1 | Training scenario 4 |
| `configs/scenarios/mock/scn-mock-lateral-train-05.json` | Project lead (MOCK) | S1 | Training scenario 5 |
| `configs/scenarios/mock/scn-mock-lateral-train-06.json` | Project lead (MOCK) | S1 | Training scenario 6 |
| `configs/scenarios/mock/scn-mock-lateral-cal-01.json` | Project lead (MOCK) | S1 | Calibration scenario 1 (uncertainty module only) |
| `configs/scenarios/mock/scn-mock-lateral-cal-02.json` | Project lead (MOCK) | S1 | Calibration scenario 2 (uncertainty module only) |
| `configs/scenarios/mock/scn-mock-lateral-eval-01.json` | Project lead (MOCK) | S1 | Held-out evaluation scenario 1 |
| `configs/scenarios/mock/scn-mock-lateral-eval-02.json` | Project lead (MOCK) | S1 | Held-out evaluation scenario 2 |
| `configs/scenarios/mock/scn-mock-lateral-eval-03.json` | Project lead (MOCK) | S1 | Held-out evaluation scenario 3 |
| `configs/scenarios/mock/scn-mock-lateral-eval-04.json` | Project lead (MOCK) | S1 | Held-out evaluation scenario 4 |
| `configs/experiments/splits.toml` | Project lead | S1 | Scenario ids and seeds of the train, cal, eval and smoke splits |
| `configs/experiments/reward_default.toml` | Project lead | S3 | Default reward weights, `d_max`, false-isolation base, sharing mode, `weights_id` |
| `configs/experiments/coordinator_default.toml` | Project lead | S4 | Allowlist, `allowlist_id`, `max_high_impact_per_step` |
| `configs/experiments/reward_sensitivity.toml` | Project lead | S5 | Weight grids for post-hoc reweighting and the later retraining sweep |
| `configs/experiments/eval_default.toml` | Project lead | S5 | Bootstrap resamples, interval level, repeated-seed lists, output options |
| `configs/experiments/smoke.toml` | Project lead | S5 | The single smoke configuration (stages for baselines, IPPO, uncertainty; flags enabled by S6 and S7) |
| `configs/experiments/ippo_default.toml` | Project lead | S6 | IPPO policy and training settings (RL_IMPLEMENTATION 5.2) |
| `configs/experiments/ippo_smoke.toml` | Project lead | S6 | Tiny IPPO budget for train, save, load, infer |
| `configs/experiments/uncertainty_default.toml` | Project lead | S7 | Thresholds and rule parameters (RL_IMPLEMENTATION 7.2) |
| `src/cyberswarm/__init__.py` | Project lead | S1 | Package version constant |
| `src/cyberswarm/contracts/__init__.py` | Project lead (neutral) | S1 | Re-exports record types and `SCHEMA_VERSION` |
| `src/cyberswarm/contracts/common.py` | Project lead (neutral) | S1 | Base model (`extra=forbid`, frozen, no NaN), `Provenance`, enums, `ShortText`, `UnitFloat`, `ContractError` |
| `src/cyberswarm/contracts/ids.py` | Project lead (neutral) | S1 | Identifier types, regexes, deterministic builders (`make_episode_id`, `observation_id`, ...), address look-alike rejection |
| `src/cyberswarm/contracts/vocabulary.py` | Project lead (neutral) | S1 | `ActionKind`, action table (target, parameters, impact, high-impact, cost), derived sets and fixed order |
| `src/cyberswarm/contracts/topology.py` | Project lead (neutral) | S1 | `Node`, `Edge`, `Topology`, zone and agent derivation, canonical order, loader |
| `src/cyberswarm/contracts/scenario.py` | Project lead (neutral) | S1 | `ScenarioConfig`, `IncidentSpec`, `ObservabilitySpec` |
| `src/cyberswarm/contracts/events.py` | Project lead (neutral) | S1 | `EventRecord`, event types, per-type field rules |
| `src/cyberswarm/contracts/observation.py` | Project lead (neutral) | S1 | `NodeObservation`, `EdgeObservation`, `ProposalFeedback`, `AgentObservation`, `DetectionSummary` |
| `src/cyberswarm/contracts/actions.py` | Project lead (neutral) | S1 | `ActionParameters`, `ActionSpec`, `AvailableActions`, `PolicyProposal`, `UncertaintyInfo`, `ValidationResult`, `ExecutedAction`, reason codes |
| `src/cyberswarm/contracts/graph.py` | Project lead (neutral) | S1 | `GraphSnapshot`, `GraphFeatureRecord`, `GraphModelOutput`, feature registries |
| `src/cyberswarm/contracts/results.py` | Project lead (neutral) | S1 | `EpisodeResult`, `EvaluationMetrics`, `ComparisonRecord`, `CalibrationReport` |
| `src/cyberswarm/contracts/manifest.py` | Project lead (neutral) | S1 | `ErrorCode`, `ErrorRecord`, `LogRecord`, `RunManifest` |
| `src/cyberswarm/contracts/evaluator_only.py` | Project lead (neutral) | S1 | `GroundTruthSnapshot` (import-restricted) |
| `src/cyberswarm/contracts/io.py` | Project lead (neutral) | S1 | Canonical JSON and JSONL read and write, version check |
| `src/cyberswarm/contracts/registry.py` | Project lead (neutral) | S1 | Record-name to model mapping used by export, validators and tests |
| `src/cyberswarm/contracts/export_schemas.py` | Project lead (neutral) | S1 | `python -m cyberswarm.contracts.export_schemas` writes `schemas/*.schema.json` |
| `src/cyberswarm/contracts/consistency.py` | Project lead (neutral) | S1 | Cross-record rules X-1 to X-14 as functions |
| `src/cyberswarm/common/__init__.py` | Project lead | S1 | Package marker |
| `src/cyberswarm/common/seeding.py` | Project lead | S1 | `derive_seeds`, `seed_everything` |
| `src/cyberswarm/common/hashing.py` | Project lead | S1 | SHA-256 of canonical JSON and of files, state-digest helper |
| `src/cyberswarm/common/logging.py` | Project lead | S1 | JSON-line `LogRecord` logging setup |
| `src/cyberswarm/common/config.py` | Project lead | S1 | Typed TOML loading (`tomllib`), unknown-key rejection, resolved-configuration snapshot |
| `src/cyberswarm/interfaces/__init__.py` | Project lead (neutral) | S2 | Exports |
| `src/cyberswarm/interfaces/simulator.py` | Project lead (neutral) | S2 | `Simulator` and `GroundTruthAccessor` Protocols, `ValidatedAction`, `ActionApplication`, `StepResult`, `SimulatorError` |
| `src/cyberswarm/interfaces/policy.py` | Project lead (neutral) | S2 | `Policy`, `DistributionPolicy` Protocols, `DecisionContext`, `EvidenceView` |
| `src/cyberswarm/interfaces/ports.py` | Project lead (neutral) | S2 | `CoordinatorPort` and `StepScorer` Protocols used by the engine |
| `src/cyberswarm/interfaces/conformance.py` | Project lead (neutral) | S2 | Reusable simulator conformance checks (`SimulatorConformance` test mixin) for the mock and for Raksha's simulator |
| `src/cyberswarm/mock/__init__.py` | Project lead (MOCK) | S2 | Exports `MockSimulator`; module docstring states it is a MOCK |
| `src/cyberswarm/mock/simulator.py` | Project lead (MOCK) | S2 | `MockSimulator` implementing both Protocols |
| `src/cyberswarm/mock/dynamics.py` | Project lead (MOCK) | S2 | Seeded incident spread, alert generation, normal activity noise, restore completion |
| `src/cyberswarm/mock/effects.py` | Project lead (MOCK) | S2 | Action effects on node state and availability |
| `src/cyberswarm/mock/observation.py` | Project lead (MOCK) | S2 | Builds agent-visible observations, available actions and events per zone |
| `src/cyberswarm/mock/truth.py` | Project lead (MOCK) | S2 | Builds `GroundTruthSnapshot` and the state digest |
| `src/cyberswarm/roles/__init__.py` | Project lead | S3 | Exports |
| `src/cyberswarm/roles/detection.py` | Project lead | S3 | `DetectionRole.summarize` (deterministic suspicion) |
| `src/cyberswarm/roles/evidence.py` | Project lead | S3 | `evidence_view` (per-node evidence state, insufficiency) |
| `src/cyberswarm/graph/__init__.py` | Project lead | S3 | Exports |
| `src/cyberswarm/graph/topology_graph.py` | Project lead | S3 | NetworkX graph from `Topology` with stable ordering, ID resolution |
| `src/cyberswarm/graph/converter.py` | Project lead | S3 | `GraphConverter.convert` observation to `GraphSnapshot` with `impute-v1` |
| `src/cyberswarm/graph/features.py` | Project lead | S3 | `compute_features` for `graph-derived-v1` |
| `src/cyberswarm/graph/tensors.py` | Project lead | S3 | `GraphTensors`, `GraphBatch`, `to_tensors`, `collate`, optional `to_torch` |
| `src/cyberswarm/engine/__init__.py` | Project lead | S3 | Exports |
| `src/cyberswarm/engine/step_engine.py` | Project lead | S3 | `EpisodeEngine`: phases P0 to P4, coordinator port, scorer port, ground-truth stream |
| `src/cyberswarm/engine/recorder.py` | Project lead | S3 | `EpisodeRecorder` (JSONL files), `MemoryRecorder`, `NullRecorder` |
| `src/cyberswarm/rl/__init__.py` | Project lead | S3 | Exports |
| `src/cyberswarm/rl/action_table.py` | Project lead | S3 | `ActionTable` index and `ActionSpec` mapping, mask from `AvailableActions` |
| `src/cyberswarm/rl/encoder.py` | Project lead | S3 | `ObservationEncoder` for the two modes (layouts of RL_IMPLEMENTATION section 3) |
| `src/cyberswarm/rl/reward.py` | Project lead | S3 | `RewardConfig`, `RewardBreakdown`, `RewardModel` (implements `StepScorer`) |
| `src/cyberswarm/rl/env.py` | Project lead | S3 | `CyberSwarmParallelEnv` (PettingZoo ParallelEnv) |
| `src/cyberswarm/baselines/__init__.py` | Project lead | S4 | Exports |
| `src/cyberswarm/baselines/rule.py` | Project lead | S4 | `RuleBasedPolicy` (conventional inputs only) |
| `src/cyberswarm/baselines/graph_rule.py` | Project lead | S4 | `GraphFeaturePolicy` (non-neural, graph features) |
| `src/cyberswarm/coordinator/__init__.py` | Project lead | S4 | Exports |
| `src/cyberswarm/coordinator/config.py` | Project lead | S4 | `CoordinatorConfig` (allowlist, budget) |
| `src/cyberswarm/coordinator/allowlist.py` | Project lead | S4 | `ActionAllowlist` |
| `src/cyberswarm/coordinator/validator.py` | Project lead | S4 | `ProposalValidator` (ordered checks and reason codes), sole creator of `ValidatedAction` |
| `src/cyberswarm/coordinator/coordinator.py` | Project lead | S4 | `DefenseCoordinator` implementing `CoordinatorPort` |
| `src/cyberswarm/coordinator/recovery.py` | Project lead | S4 | `RecoveryVerifier` |
| `src/cyberswarm/evaluation/__init__.py` | Project lead | S5 | Exports (pure functions only; no harness import) |
| `src/cyberswarm/evaluation/outcome.py` | Project lead | S5 | `compute_outcome` from records and ground-truth stream |
| `src/cyberswarm/evaluation/metrics.py` | Project lead | S5 | `summarize` to `EvaluationMetrics` |
| `src/cyberswarm/evaluation/stats.py` | Project lead | S5 | mean, sample standard deviation, percentile bootstrap |
| `src/cyberswarm/evaluation/compare.py` | Project lead | S5 | `compare` to `ComparisonRecord` for matched sets |
| `src/cyberswarm/evaluation/splits.py` | Project lead | S5 | Load splits, disjointness assertions, `matched_set_id` |
| `src/cyberswarm/evaluation/reweight.py` | Project lead | S5 | `reweight` episode rewards from stored raw magnitudes |
| `src/cyberswarm/runner/__init__.py` | Project lead | S5 | Exports |
| `src/cyberswarm/runner/config.py` | Project lead | S5 | Experiment configuration models |
| `src/cyberswarm/runner/factory.py` | Project lead | S5 | Simulator and policy factories |
| `src/cyberswarm/runner/evaluate.py` | Project lead | S5 | `run_episode`, `evaluate_policies` harness over matched pairs |
| `src/cyberswarm/runner/export.py` | Project lead | S5 | Run directory writer and `RunManifest` builder |
| `src/cyberswarm/runner/smoke.py` | Project lead | S5 | `python -m cyberswarm.runner.smoke` entry point |
| `src/cyberswarm/marl/__init__.py` | Project lead | S6 | Exports; no MAPPO symbol |
| `src/cyberswarm/marl/networks.py` | Project lead | S6 | MLP actor and critic builders, `masked_categorical` |
| `src/cyberswarm/marl/gae.py` | Project lead | S6 | `compute_gae` |
| `src/cyberswarm/marl/value_norm.py` | Project lead | S6 | `ValueNorm` |
| `src/cyberswarm/marl/ppo.py` | Project lead | S6 | Clipped PPO update for one agent |
| `src/cyberswarm/marl/rollout.py` | Project lead | S6 | Episode collection and trajectory storage |
| `src/cyberswarm/marl/ippo.py` | Project lead | S6 | `IPPOConfig`, `IPPOTrainer` (independent agents) |
| `src/cyberswarm/marl/checkpoint.py` | Project lead | S6 | `save_checkpoint`, `load_checkpoint`, `CheckpointError` |
| `src/cyberswarm/marl/policy.py` | Project lead | S6 | `IPPOPolicy` (implements `DistributionPolicy`) |
| `src/cyberswarm/marl/train.py` | Project lead | S6 | Training entry point and training log |
| `src/cyberswarm/uncertainty/__init__.py` | Project lead | S7 | Exports |
| `src/cyberswarm/uncertainty/signal.py` | Project lead | S7 | `normalized_entropy` |
| `src/cyberswarm/uncertainty/rules.py` | Project lead | S7 | `UncertaintyRuleConfig`, `apply_rules` |
| `src/cyberswarm/uncertainty/policy.py` | Project lead | S7 | `UncertaintyAwarePolicy`, `MatchedControlPolicy` |
| `src/cyberswarm/uncertainty/calibration.py` | Project lead | S7 | Reliability bins, ECE, AUROC, calibrator, threshold selection, `CalibrationReport` |
| `schemas/*.schema.json` | Project lead (neutral) | S1 | One exported JSON Schema per record in the registry (see section 4.1); regenerated, never edited by hand |
| `fixtures/` | Project lead (neutral) | S1, S3, S5 | Synthetic fixtures, listed in section 8 |
| `tests/` | Project lead (`tests/simulator/`: Raksha, not created) | S1 to S9 | Listed in section 9 |
| `src/cyberswarm/simulator/` | Raksha | not in scope | Real simulator. Not created. No placeholder |
| `dashboard/` | Mayukh Aryan | not in scope | Dashboard application. Not created. No placeholder |
| `runs/`, `checkpoints/`, `.venv/` | none | generated | Gitignored outputs and environment |
<!--/GEN:TREE-->

Directory owners (one owner per directory; teammates edit only their own):

| Directory | Owner |
|-----------|-------|
| `src/cyberswarm/contracts/`, `src/cyberswarm/interfaces/`, `schemas/`, `fixtures/`, `configs/topologies/` | Project lead (neutral: neither teammate edits) |
| `src/cyberswarm/mock/`, `configs/scenarios/mock/` | Project lead (MOCK) |
| `src/cyberswarm/{common,roles,graph,coordinator,engine,rl,baselines,marl,uncertainty,evaluation,runner}/`, `configs/experiments/`, `scripts/`, `docs/`, `tests/` (except `tests/simulator/`) | Project lead |
| `src/cyberswarm/simulator/`, `tests/simulator/`, `configs/scenarios/simulator/` | Raksha (not created) |
| `dashboard/` | Mayukh Aryan (not created) |

## 3. Tooling, dependencies and import rules

### 3.1 `pyproject.toml` (S1)

- `[build-system]`: `setuptools>=68`.
- `[project]`: `name = "cyberswarm"`, `version = "0.1.0"`, `requires-python = ">=3.12"`. No `license` key.
- Core dependencies: `pydantic>=2.12,<3`, `numpy>=2.0`, `networkx>=3.4`.
- Optional dependency groups: `rl = ["torch>=2.5", "gymnasium>=1.0,<2", "pettingzoo>=1.24"]`, `dev = ["pytest>=8", "jsonschema>=4.20"]`. Install with `pip install --no-cache-dir -e ".[rl,dev]"`.
- `[tool.setuptools.packages.find] where = ["src"]`.
- `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, `pythonpath = ["."]`, `addopts = "-q --import-mode=importlib"`, markers `slow`.
- `[project.scripts]`: none required; the smoke command uses `python -m`.
- The contracts package imports only `pydantic` and the standard library. The graph package imports `numpy` and `networkx`; `torch` is imported lazily and only inside `to_torch`. The mock imports `numpy` and `networkx`.

### 3.2 Import rules (enforced by `tests/integration/test_import_boundaries.py`)

Allowed imports, by package (a package may import itself, the standard library and the third-party libraries named; anything not listed is forbidden):

| Package | May import |
|---------|------------|
| `contracts` | `pydantic`, stdlib |
| `common` | `contracts`, `numpy` |
| `interfaces` | `contracts`, `common`, `numpy` |
| `mock` | `contracts`, `interfaces`, `common`, `numpy`, `networkx` |
| `roles` | `contracts`, `interfaces`, `common` |
| `graph` | `contracts`, `common`, `numpy`, `networkx` (`torch` only lazily in `tensors.to_torch`) |
| `coordinator` | `contracts`, `interfaces`, `common` |
| `evaluation` | `contracts`, `common`, `numpy` |
| `engine` | `contracts`, `interfaces`, `common`, `roles`, `graph` (not `evaluation`; the runner computes outcomes) |
| `rl` | `contracts`, `interfaces`, `common`, `roles`, `graph`, `engine`, `numpy`, `gymnasium`, `pettingzoo` |
| `baselines` | `contracts`, `interfaces`, `common`, `roles` |
| `marl` | `contracts`, `interfaces`, `common`, `rl`, `engine`, `numpy`, `torch` |
| `uncertainty` | `contracts`, `interfaces`, `common`, `marl`, `numpy` (`torch` only through `marl`) |
| `runner` | every lead-owned package |

Additional boundary rules:

- Only `interfaces.simulator` and `interfaces.ports` (type annotations only), `mock.*`, `rl.reward`, `engine.step_engine`, `evaluation.*`, `runner.*` and `tests/` may import `cyberswarm.contracts.evaluator_only`. Only `engine.step_engine` and `tests/` may call `ground_truth()` or `termination_reason()`.
- Only `coordinator.validator` constructs `ValidatedAction` (the mock may construct it only in tests; the test double lives under `tests/`).
- No package under `src/` imports `cyberswarm.mock` except `runner.factory`.
- No file under `src/` contains the token `mappo` (case-insensitive) except `contracts/manifest.py` (the `status.mappo` field, constant `deferred`) and `runner/export.py` (the status text).
- `coordinator/` contains none of: `subprocess`, `os.system`, `os.popen`, `socket`, `requests`, `urllib`, `http`, `shutil`, `pathlib.Path.write`, `open(` with a write mode, `eval(`, `exec(`.

## 4. Module APIs

Signatures are Python 3.12. `ContractModel` is the frozen, `extra="forbid"` pydantic base class in `contracts/common.py`. Types named in CONTRACTS are not repeated.

### 4.1 `cyberswarm.contracts` (S1)

- `SCHEMA_VERSION: Final[str] = "1.0"`.
- Registry `contracts.registry.RECORD_TYPES: dict[str, type[ContractModel]]` with exactly these 23 entries, each exported as `schemas/<Name>.schema.json`: `Node`, `Edge`, `Topology`, `ScenarioConfig`, `EventRecord`, `AgentObservation`, `AvailableActions`, `DetectionSummary`, `ActionSpec`, `PolicyProposal`, `ValidationResult`, `ExecutedAction`, `GroundTruthSnapshot`, `GraphSnapshot`, `GraphFeatureRecord`, `GraphModelOutput`, `EpisodeResult`, `EvaluationMetrics`, `ComparisonRecord`, `CalibrationReport`, `ErrorRecord`, `LogRecord`, `RunManifest`.
- `ids.py`: annotated string types (`NodeId`, `EdgeId`, `ZoneId`, `AgentId`, `TopologyId`, `ScenarioId`, `RunId`, `PolicyId`, `EpisodeId`, `EventId`, `ProposalId`, `ValidationId`, `ExecutionId`, `ErrorId`, ...) and builders: `make_episode_id(policy_id, scenario_id, seed) -> EpisodeId`, `observation_id(step, agent_id)`, `available_actions_id(...)`, `detection_id(...)`, `snapshot_id(...)`, `feature_record_id(...)`, `agent_id_for_zone(zone_id) -> AgentId`, `event_id(n, prefix="evt")`, `proposal_id(n)`, `validation_id(n)`, `execution_id(n)`, `error_id(n)`.
- `vocabulary.py`: `ActionKind` (StrEnum), `ActionDef` (frozen dataclass: `needs_target`, `allowed_parameters`, `impact_class`, `high_impact`, `cost_units`), `ACTION_DEFS: Mapping[ActionKind, ActionDef]`, `ACTION_ORDER: tuple[ActionKind, ...]`, `HIGH_IMPACT`, `CONTAINMENT_CLASS`, `INFORMATION_GATHERING`, `default_parameters(kind) -> ActionParameters`.
- `topology.py`: `Topology.node_ids`, `.zone_ids`, `.agent_ids`, `.nodes_of_zone(zone_id)`, `.zone_of(node_id)`, `.edge_ids`, `load_topology(path: Path) -> Topology`.
- `io.py`: `dumps_canonical(model) -> str`, `loads_record(name: str, text: str) -> ContractModel`, `read_json(path, model)`, `read_jsonl(path, model) -> Iterator[ContractModel]`, `write_json(path, model)`, `class JsonlWriter(path)` (`write(model)`, `close()`), `check_version(raw_version: str)` raising `ContractError(SCHEMA_VERSION_MISMATCH)`.
- `consistency.py`: `check_episode(records: EpisodeRecordSet, topology: Topology) -> list[ConsistencyIssue]` implementing X-1 to X-13; `check_no_evaluator_fields(paths) -> list[str]` implementing X-10.
- `export_schemas.py`: `export(out_dir: Path) -> list[Path]`; `python -m cyberswarm.contracts.export_schemas --out /Users/guru/CyberSwarm-GX/schemas`.

### 4.2 `cyberswarm.interfaces` (S2)

```python
# interfaces/simulator.py
class SimulatorError(Exception):
    code: ErrorCode
    def __init__(self, code: ErrorCode, message: str) -> None: ...

class ValidatedAction(ContractModel):          # created only by coordinator.validator
    episode_id: EpisodeId
    step: Step
    agent_id: AgentId
    proposal_id: ProposalId
    validation_id: ValidationId
    action: ActionSpec                          # resolved, defaults filled

class StatusChange(ContractModel):
    node_id: NodeId
    status_from: NodeStatus
    status_to: NodeStatus

class ActionApplication(ContractModel):
    status: Literal["executed", "failed"]
    error_code: ErrorCode | None
    affected_node_ids: tuple[NodeId, ...]
    status_changes: tuple[StatusChange, ...]
    outcome_event_ids: tuple[EventId, ...]

class StepResult(ContractModel):
    step_closed: Step
    terminated: bool
    truncated: bool

@runtime_checkable
class Simulator(Protocol):
    kind: Literal["mock", "simulator"]
    @property
    def topology(self) -> Topology: ...
    @property
    def agent_ids(self) -> tuple[AgentId, ...]: ...           # ascending
    def authority(self, agent_id: AgentId) -> tuple[NodeId, ...]: ...   # nodes of the agent's zone, ascending
    def reset(self, scenario: ScenarioConfig, seed: int, episode_id: EpisodeId) -> None: ...
    @property
    def step(self) -> int: ...                                 # current step t; 0 after reset
    @property
    def done(self) -> bool: ...
    def observe(self) -> Mapping[AgentId, AgentObservation]: ...     # P1; idempotent within a step; after done returns the final observation (step = last+1)
    def available_actions(self, agent_id: AgentId) -> AvailableActions: ...
    def apply_action(self, action: ValidatedAction) -> ActionApplication: ...  # P4; never raises for an inapplicable action: returns status "failed" with state unchanged
    def end_step(self) -> StepResult: ...                      # closes step t, evaluates termination, runs P0 of t+1 if not done
    def drain_events(self) -> list[EventRecord]: ...           # events emitted since the last drain, in emission order
    def state_digest(self) -> str: ...                         # SHA-256 hex of the canonical full internal state
    def close(self) -> None: ...

@runtime_checkable
class GroundTruthAccessor(Protocol):                           # evaluator-only
    def ground_truth(self) -> GroundTruthSnapshot: ...         # end-of-step truth for the current step
    def termination_reason(self) -> Literal["objective_reached", "incident_resolved", "max_steps"] | None: ...
```

`reset` raises `SimulatorError(SCENARIO_UNSUPPORTED)` for an unknown topology, node, or unsupported scenario field. `apply_action`, `end_step` and `observe` raise `SimulatorError(SIMULATOR_FAILURE)` only for internal faults or misuse (for example `end_step` after the episode is done). A rejected or inapplicable action is a normal `failed` result.

```python
# interfaces/policy.py
@dataclass(frozen=True)
class NodeEvidence:
    evidence_level: int | None
    steps_since_investigated: int | None
    alerted: bool
    insufficient: bool          # alerted and (evidence_level is None or evidence_level < 2)

@dataclass(frozen=True)
class EvidenceView:
    nodes: Mapping[NodeId, NodeEvidence]

@dataclass(frozen=True)
class DecisionContext:
    episode_id: EpisodeId
    step: int
    max_steps: int
    agent_id: AgentId
    topology: Topology
    observation: AgentObservation
    available_actions: AvailableActions
    detection: DetectionSummary
    evidence: EvidenceView
    graph_snapshot: GraphSnapshot | None
    graph_features: GraphFeatureRecord | None
    next_proposal_id: ProposalId

class Policy(Protocol):
    policy_id: PolicyId
    representation: Representation
    def reset(self, episode_id: EpisodeId, seed: int) -> None: ...
    def propose(self, ctx: DecisionContext) -> PolicyProposal: ...

@dataclass(frozen=True)
class PolicyDecision:
    chosen: ActionSpec
    valid_actions: tuple[ActionSpec, ...]
    probs: tuple[float, ...]            # aligned with valid_actions, sums to 1.0 (float64)

class DistributionPolicy(Policy, Protocol):
    def decide(self, ctx: DecisionContext) -> PolicyDecision: ...
    def build_proposal(self, ctx: DecisionContext, decision: PolicyDecision, *, action: ActionSpec,
                       rationale_codes: Sequence[str], uncertainty: UncertaintyInfo | None) -> PolicyProposal: ...
```

```python
# interfaces/ports.py
@dataclass(frozen=True)
class CoordinatorResult:
    validations: tuple[ValidationResult, ...]
    executions: tuple[ExecutedAction, ...]
    errors: tuple[ErrorRecord, ...]

class CoordinatorPort(Protocol):
    def start_episode(self, episode_id: EpisodeId, topology: Topology) -> None: ...
    def process(self, proposals: Sequence[PolicyProposal | Mapping[str, Any]], *, step: int,
                available: Mapping[AgentId, AvailableActions]) -> CoordinatorResult: ...
    def verify_recovery(self, step: int, observations: Mapping[AgentId, AgentObservation]) -> list[EventRecord]: ...

@dataclass(frozen=True)
class StepScore:
    total: float
    raw: Mapping[str, float]            # unweighted magnitudes of the seven terms
    weighted: Mapping[str, float]       # signed contributions

class StepScorer(Protocol):
    def reset(self, topology: Topology, max_steps: int) -> None: ...
    def score(self, *, step: int, prev: GroundTruthSnapshot | None, curr: GroundTruthSnapshot,
              executions: Sequence[ExecutedAction], invalid_count: int,
              termination_reason: str | None) -> StepScore: ...
```

Conformance checks (class `SimulatorConformance`; each is a test method; they are the checks that every simulator must pass):

| ID | Check |
|----|-------|
| SC-1 | Node ids equal the topology's ids, are identical across resets and episodes, and are in canonical order |
| SC-2 | Every edge endpoint resolves to a node; edge ids are unique and ordered |
| SC-3 | `reset` with an unsupported scenario raises `SimulatorError(SCENARIO_UNSUPPORTED)` and leaves the simulator unusable until a valid `reset` |
| SC-4 | Step semantics: `step == 0` after `reset`; `observe()` returns one observation per agent with `step` equal to the current step; `end_step()` increments the step; calling `end_step()` after `done` raises |
| SC-5 | Determinism: two instances with the same scenario, seed and scripted actions produce identical observations, events, state digests; a different seed changes at least the digest sequence |
| SC-6 | Observations and available actions validate against the contracts, and a JSON scan finds no evaluator-only field name |
| SC-7 | No future events: every event in `recent_events` has `step <=` the observation step; event ids are contiguous per prefix |
| SC-8 | Missing-value semantics: with `missing_value_rate = 1.0` every measurement that may be missing is `null` and none is `0.0`; with `0.0` none is `null`; boundary-scope private fields are always `null` |
| SC-9 | An inapplicable or out-of-authority action returns `status = failed` and leaves `state_digest()` unchanged |
| SC-10 | The object returned by `observe()` exposes no ground-truth attribute; `ground_truth()` covers every node in canonical order and its `contained` flags agree with the visible statuses |
| SC-11 | Termination: `terminated` or `truncated` is reported at the right step; after the last step `observe()` returns the final observation with `step = last + 1`; `apply_action` after `done` returns `failed` or raises `SimulatorError` |
| SC-12 | `available_actions` agree with the vocabulary table (impact class, high-impact flag, cost), start with `no_op`, and list only targets in the agent's zone |
| SC-13 | `kind` is `mock` for the mock and `simulator` for a real implementation; `state_digest()` is a 64-character hex string and changes when the state changes |

`interfaces/conformance.py`: `class SimulatorConformance` is a pytest mixin. A subclass implements `make_simulator(self) -> tuple[Simulator, GroundTruthAccessor]` and `scenarios(self) -> dict[str, ScenarioConfig]` (keys `base`, `no_missing`, `all_missing`, `unsupported`). The checks SC-1 to SC-13 above are inherited as test methods (DELTA check DC-06, with DC-04 for SC-6, SC-7 and SC-10). `tests/interfaces/test_conformance_mock.py` instantiates it for the mock; Raksha subclasses it in `tests/simulator/`.

### 4.3 `cyberswarm.common` (S1)

- `seeding.derive_seeds(seed: int) -> SeedStreams` (fields `simulator`, `scenario`, `policy`, each an int from `numpy.random.SeedSequence(seed).generate_state(3)`), `seed_everything(seed: int, *, deterministic_torch: bool = False) -> None` (imports `torch` lazily).
- `hashing.sha256_json(obj) -> str` (canonical JSON), `sha256_file(path) -> str`.
- `logging.get_logger(component, *, run_id=None, sink: Path | None = None)` writing `LogRecord` lines; the only place that reads the wall clock.
- `config.load_toml(path: Path, model: type[BaseModel]) -> BaseModel` (rejects unknown keys), `snapshot_config(obj) -> str` (canonical JSON of the resolved configuration).

### 4.4 `cyberswarm.roles` (S3)

- `DetectionRole(producer: str = "detection.rule-v1")`: `summarize(obs: AgentObservation) -> DetectionSummary`; suspicion formula in CONTRACTS 8.4.
- `evidence_view(obs: AgentObservation, *, e_min: int = 2) -> EvidenceView`.

### 4.5 `cyberswarm.graph` (S3, first part)

- `TopologyGraph(topology)`: `.nx` (NetworkX `DiGraph` with edge attribute `edge_type`), `.node_order`, `.edge_order`, `.index_of(node_id)`, `.link_neighbors(node_id)`, `.dependents(node_id)` (transitive), `.diameter`, `.hops(a, b)`. Unknown id raises `GraphConversionError(UNKNOWN_ID)`.
- `GraphConverter(topology)`: `convert(obs: AgentObservation) -> GraphSnapshot` implementing `graph-base-v1` and `impute-v1` (CONTRACTS 10.1). `max_event_step` is computed from `obs.recent_events`.
- `compute_features(topology: Topology, snapshot: GraphSnapshot) -> GraphFeatureRecord` implementing `graph-derived-v1`. It receives no observation and no truth object: it works from the snapshot (which carries the masks) and the topology.
- `GraphTensors` (numpy): `x (N, 11) float32`, `x_mask (N, 11) bool`, `edge_index (2, E) int64`, `edge_attr (E, 6) float32`, `edge_mask (E, 6) bool`, `derived (N, 8) float32`, `derived_mask (N, 8) bool`, `node_order`, `edge_order`. `to_tensors(snapshot, features=None) -> GraphTensors`. `collate(items) -> GraphBatch` stacks to `(B, N, F)` and `(B, 2, E)` and requires identical `node_order` and `edge_order`. `GraphTensors.to_torch()` and `GraphBatch.to_torch()` import `torch` lazily and return tensors with the same dtypes (float32, int64, bool).

### 4.6 `cyberswarm.engine` (S3)

```python
@dataclass(frozen=True)
class StepContext:
    episode_id: EpisodeId
    step: int
    max_steps: int
    agents: Mapping[AgentId, DecisionContext]
    final: bool                               # True for the unrecorded final observation after the last step

@dataclass(frozen=True)
class StepOutcome:
    context: StepContext                      # next step, or the final observation when done
    done: bool
    terminated: bool
    truncated: bool
    feedback: Mapping[AgentId, tuple[ProposalFeedback, ...]]
    validations: tuple[ValidationResult, ...]
    executions: tuple[ExecutedAction, ...]
    score: StepScore | None                   # None when no scorer is attached

class EpisodeEngine:
    def __init__(self, simulator: Simulator, coordinator: CoordinatorPort, *,
                 truth: GroundTruthAccessor | None = None, scorer: StepScorer | None = None,
                 recorder: Recorder | None = None, detection: DetectionRole | None = None,
                 compute_graph: bool = True) -> None: ...
    def reset(self, scenario: ScenarioConfig, seed: int, policy_id: PolicyId) -> StepContext: ...
    def step(self, proposals: Mapping[AgentId, PolicyProposal | Mapping[str, Any]], *,
             invalid_counts: Mapping[AgentId, int] | None = None) -> StepOutcome: ...
    @property
    def termination_reason(self) -> str | None: ...        # evaluator data; None without a truth accessor
    def close(self) -> None: ...
```

`reset` builds `episode_id = make_episode_id(policy_id, scenario.scenario_id, seed)`, resets the simulator and coordinator, scorer and recorder, drains P0 events, and runs P1 (observations, available actions, detection, graph snapshot and features per agent). `step` runs P2 recording, P3 and P4 through the coordinator, `verify_recovery`, `end_step`, drains events, queries the truth accessor for the end-of-step snapshot (if present), calls the scorer, records everything in the phase order of CONTRACTS section 5, then builds the next context. A proposal missing for an agent is replaced by a `no_op` proposal with an `ErrorRecord(POLICY_FAILURE)`.

Recorders: `Recorder` Protocol with `write(record)` and `close()`. `EpisodeRecorder(dir)` writes the episode files of CONTRACTS section 15 (and `evaluator_only/ground_truth.jsonl`). `MemoryRecorder` keeps lists by record type. `NullRecorder` drops everything (used for training).

### 4.7 `cyberswarm.rl` (S3)

```python
@dataclass(frozen=True)
class ActionTable:
    agent_id: AgentId
    specs: tuple[ActionSpec, ...]                           # index 0 is no_op
    @classmethod
    def build(cls, topology: Topology, agent_id: AgentId) -> "ActionTable": ...
    @property
    def n(self) -> int: ...
    def index_of(self, spec: ActionSpec) -> int: ...
    def spec_of(self, index: int) -> ActionSpec: ...
    def mask(self, available: AvailableActions) -> np.ndarray: ...   # bool, shape (n,)
    def digest(self) -> str: ...                                     # hash used in checkpoints

class ObservationEncoder:
    def __init__(self, topology: Topology, agent_id: AgentId, mode: Representation) -> None: ...
    @property
    def dim(self) -> int: ...                                # 17N+2 or 25N+2
    @property
    def feature_names(self) -> tuple[str, ...]: ...
    def encode(self, ctx: DecisionContext, *, last_action_rejected: bool) -> np.ndarray: ...   # float32 in [0, 1]

class RewardConfig(BaseModel):   # frozen, extra=forbid
    weights_id: str; sharing: Literal["team", "zone_local"]; d_max: int; false_isolation_base: float
    weights: RewardWeights        # seven non-negative floats named as the terms
    @classmethod
    def from_toml(cls, path: Path) -> "RewardConfig": ...

class RewardModel:                # implements StepScorer
    def __init__(self, config: RewardConfig) -> None: ...

class CyberSwarmParallelEnv(ParallelEnv):
    metadata = {"name": "cyberswarm_parallel_v0"}
    def __init__(self, engine_factory: Callable[[], EpisodeEngine], scenarios: Sequence[ScenarioConfig], *,
                 observation_mode: Representation, policy_id: PolicyId = "pol-env-v0") -> None: ...
    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]: ...
    def step(self, actions: dict[AgentId, int]) -> tuple[dict, dict, dict, dict, dict]: ...
    def observation_space(self, agent): ...
    def action_space(self, agent): ...
    def close(self) -> None: ...
    @property
    def last_reward_breakdown(self) -> StepScore | None: ...       # for the training harness; never placed in info
```

`reset` requires an integer `seed`; `seed=None` raises `ValueError` (reproducibility is mandatory).

### 4.8 `cyberswarm.coordinator` (S4)

```python
class CoordinatorConfig(BaseModel):          # configs/experiments/coordinator_default.toml
    allowlist_id: str
    allowlist: frozenset[ActionKind]
    max_high_impact_per_step: int = 1

class ActionAllowlist:
    def __init__(self, kinds: Iterable[ActionKind]) -> None: ...
    def permits(self, kind: ActionKind) -> bool: ...

class ProposalValidator:
    def __init__(self, topology: Topology, config: CoordinatorConfig) -> None: ...
    def validate(self, raw: PolicyProposal | Mapping[str, Any], *, episode_id: EpisodeId, step: int,
                 available: Mapping[AgentId, AvailableActions], accepted_so_far: Sequence[ValidatedAction],
                 seen_proposal_ids: set[str], validation_id: ValidationId, state_digest: str | None
                 ) -> tuple[ValidationResult, ValidatedAction | None]: ...      # checks in the order of CONTRACTS 8.6

class DefenseCoordinator:                    # implements CoordinatorPort
    def __init__(self, simulator: Simulator, config: CoordinatorConfig) -> None: ...

class RecoveryVerifier:
    def __init__(self, *, health_min: float = 0.8, window: int = 2) -> None: ...
    def note_execution(self, execution: ExecutedAction) -> None: ...
    def check(self, step: int, observations: Mapping[AgentId, AgentObservation], event_counter: Counter) -> list[EventRecord]: ...
```

`process` orders raw proposals by `agent_id` (unparseable ones last, in input order), validates each, executes accepted ones in order through `simulator.apply_action`, and returns the records. It never mutates a proposal. It performs no file or network operation.

### 4.9 `cyberswarm.baselines` (S4)

```python
class RuleConfig(BaseModel):
    e_min: int = 2; e_contain: int = 3; crit_quarantine_min: float = 0.7
    investigate_cooldown: int = 2; dependents_quarantine_min: float = 0.10
    neighbor_alert_weight: float = 0.5; premonitor_neighbor_alert: float = 0.2; premonitor_hop_max: float = 0.25

class RuleBasedPolicy:                       # policy_id "pol-rule-conv-v1", representation "conventional"
    def __init__(self, config: RuleConfig = RuleConfig(), policy_id: PolicyId = "pol-rule-conv-v1") -> None: ...
class GraphFeaturePolicy:                    # policy_id "pol-rule-graph-v1", representation "graph_features"
    def __init__(self, config: RuleConfig = RuleConfig(), policy_id: PolicyId = "pol-rule-graph-v1") -> None: ...
```

Rule-based decision (one action per agent per step, deterministic, uses only `ctx.observation`, `ctx.detection`, `ctx.evidence`, `ctx.available_actions`, `ctx.topology` criticality):

1. Candidate nodes are own-scope nodes that are alerted, or have `evidence_level >= e_min`, or have status `quarantined`/`isolated`. Order by descending `suspicion`, then ascending `node_id`.
2. For the first candidate `n` for which at least one rule below yields an available action:
   a. Contained (`quarantined` or `isolated`) and evidence `0` (clean): `restore_node(n)` with codes `recovery_after_containment`.
   b. Contained and `evidence_level >= e_contain`: `restore_node(n)` with `recovery_after_containment`.
   c. Not contained and (`evidence_level` is null or `< e_min`) and alerted: if `steps_since_investigated` is null or `>= investigate_cooldown` then `investigate_node(n)` (`alerts_present`, `evidence_insufficient`); else if alert count `>= 2` then `rate_limit_node(n)` (`alerts_present`, `evidence_insufficient`); else continue to the next candidate.
   d. `evidence_level == e_min`: `rate_limit_node(n)` (`evidence_moderate`), or `quarantine_node(n)` if rate limiting is unavailable.
   e. `evidence_level >= e_contain`: if `criticality(n) >= crit_quarantine_min` then `quarantine_node(n)` else `isolate_node(n)` (`evidence_strong`, `containment_ladder`); if that action is unavailable use the next less disruptive available one.
3. If no candidate yields an action: `no_op` (`no_evidence` or `no_action_needed`).

Graph-feature variant: identical ladder with three differences, all from `ctx.graph_features`: candidate ordering by `suspicion + neighbor_alert_weight * neighbor_alert_mean`; in rule 2e the choice between `quarantine_node` and `isolate_node` uses `dependents_crit_norm >= dependents_quarantine_min` instead of the node's own criticality; and when no candidate exists, `monitor_node(n)` on the own node with the highest `neighbor_alert_max` if `neighbor_alert_max >= premonitor_neighbor_alert` and `hop_to_alerted_norm <= premonitor_hop_max`. It raises `PolicyError(POLICY_FAILURE)` if `ctx.graph_features` is `None`.

Both set `rationale_codes` from the CONTRACTS enum, `action_probability = None`, `uncertainty = None`, and a `rationale_text` built from fixed templates (no interpolated free text other than node ids).

### 4.10 `cyberswarm.marl` (S6)

```python
def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical: ...
def compute_gae(rewards: Sequence[float], values: Sequence[float], last_value: float, terminated: bool,
                gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]: ...        # advantages, returns
class ValueNorm: update(x), normalize(x), denormalize(x), state_dict()/load_state_dict()
class IPPOConfig(BaseModel): hidden: int; gamma: float; lam: float; clip: float; epochs: int; lr: float; entropy_coef: float; max_grad_norm: float; episodes_per_update: int; updates: int; observation_mode: Representation; seed: int
class IPPOTrainer:
    def __init__(self, env: CyberSwarmParallelEnv, config: IPPOConfig, out_dir: Path) -> None: ...
    def train(self) -> TrainingSummary: ...                  # writes training_log.jsonl (one JSON object per update) under out_dir
def save_checkpoint(trainer_or_state, out_dir: Path) -> Path
def load_checkpoint(path: Path, *, expect_topology: Topology) -> IPPOState          # raises CheckpointError
class IPPOPolicy:                                            # implements DistributionPolicy
    @classmethod
    def from_checkpoint(cls, path: Path, topology: Topology, *, stochastic: bool = False, policy_id: PolicyId | None = None) -> "IPPOPolicy": ...
```

`training_log.jsonl` lines are plain JSON objects with the fields `update`, `episodes`, `mean_team_return` (training signal on the mock, not evidence), `policy_loss`, `value_loss`, `entropy`, one record per agent. They are not contract records and never feed the evaluator.

### 4.11 `cyberswarm.uncertainty` (S7)

```python
def normalized_entropy(probs: Sequence[float]) -> float | None: ...      # None if non-finite or empty; 0.0 for one valid action
class UncertaintyRuleConfig(BaseModel): tau_evidence: float = 0.50; tau_defer: float = 0.30; e_min: int = 2; e_contain: int = 3; max_consecutive_substitutions: int = 2
@dataclass(frozen=True)
class RuleDecision: action: ActionSpec; rule_outcome: str; rationale_codes: tuple[str, ...]; original_action: ActionSpec | None
def apply_rules(*, chosen: ActionSpec, u: float | None, evidence_level: int | None, steps_since_investigated: int | None,
                available: AvailableActions, streak: int, config: UncertaintyRuleConfig) -> RuleDecision: ...
class UncertaintyWrapperPolicy:                              # base class; holds the wrapped DistributionPolicy
    def __init__(self, base: DistributionPolicy, *, rules_enabled: bool, config: UncertaintyRuleConfig, suffix: str) -> None: ...
class UncertaintyAwarePolicy(UncertaintyWrapperPolicy):     # rules_enabled=True, policy id suffix "-ua"
class MatchedControlPolicy(UncertaintyWrapperPolicy):       # rules_enabled=False, policy id suffix "-ctl"
# calibration.py
def reliability_bins(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[Bin]: ...
def ece(bins: list[Bin]) -> float | None: ...
def auroc(uncertainty: np.ndarray, incorrect: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, float, float] | None: ...
def fit_histogram_calibrator(confidence, correct, min_per_bin: int = 20) -> Calibrator
def select_thresholds(cal_samples, config) -> tuple[float, float, str]       # tau_evidence, tau_defer, "selected_on_cal"|"default"
def build_report(*, cal_samples, eval_samples, ..., provenance) -> CalibrationReport
```

### 4.12 `cyberswarm.evaluation` (S5)

```python
def compute_outcome(topology: Topology, trace: EpisodeTraceView, truth: Sequence[GroundTruthSnapshot]) -> EpisodeOutcome: ...
def summarize(results: Sequence[EpisodeResult], *, metrics_id: str, run_id: str, matched_set_id: str,
              n_aborted: int, bootstrap: BootstrapConfig | None, provenance: Provenance) -> EvaluationMetrics: ...
def compare(baseline: Sequence[EpisodeResult], candidate: Sequence[EpisodeResult], *, comparison_id: str, run_id: str,
            bootstrap: BootstrapConfig | None, provenance: Provenance) -> ComparisonRecord: ...
def load_splits(path: Path) -> Splits; def assert_disjoint(splits: Splits) -> None
def matched_set_id(pairs: Sequence[tuple[str, int]]) -> str      # "set-" + first 12 hex of SHA-256 of the sorted pair list
def reweight(results: Sequence[EpisodeResult], weights: RewardWeights) -> list[float]
```

### 4.13 `cyberswarm.runner` (S5; extended in S6 and S7)

- `config.ExperimentConfig` / `SmokeConfig` models for `smoke.toml`.
- `factory.make_simulator(kind: str) -> tuple[Simulator, GroundTruthAccessor]` (`"mock"` only in this phase), `factory.make_coordinator(...)`, `factory.make_policy(spec: str, ...)` where `spec` is `rule-conv`, `rule-graph`, `ippo-conv:<ckpt>`, `ippo-graph:<ckpt>`, `ippo-conv-ua:<ckpt>`, `ippo-conv-ctl:<ckpt>`.
- `evaluate.run_episode(engine, policies, scenario, seed, out_dir | None, split) -> EpisodeResult` runs the loop `reset`, `propose` per agent, `step`, writes `episode_result.json` after `evaluation.outcome.compute_outcome`.
- `evaluate.evaluate_policies(policy_specs, pairs, run_dir, ...) -> list[EvaluationMetrics]` evaluates every policy on the identical sorted `pairs` and writes `evaluation/metrics.jsonl` and `evaluation/comparison.jsonl`.
- `export.RunWriter(run_dir)` creates the directory layout, copies topology and scenario files, writes `run_manifest.json` last (temporary file then rename), computes `files` counts, sets `status` (`ippo`: `implemented` if an IPPO policy ran else `not_run`; `mappo`: always `deferred`; `learned_gnn`: always `not_implemented`).
- `smoke.main(argv)`: the single smoke command (section 7).

## 5. MOCK simulator scope

The mock is `cyberswarm.mock.MockSimulator`. It implements `Simulator` and `GroundTruthAccessor`, `kind = "mock"`, and nothing else. It is lead-owned, lives outside Raksha's directory, imports no private internals of any other simulator, and is never described as the real simulator. Its module docstring and `run_manifest.simulator_kind` say MOCK.

### 5.1 Topology `topo-mock-small-v1`

Nodes (ascending id), zone, type, criticality:

| Node | Zone | Type | Criticality |
|------|------|------|-------------|
| `node-app-01` | `zone-core` | app | 0.7 |
| `node-auth-01` | `zone-core` | auth | 0.8 |
| `node-db-01` | `zone-core` | database | 1.0 |
| `node-file-01` | `zone-core` | file | 0.3 |
| `node-gw-01` | `zone-dmz` | gateway | 0.6 |
| `node-web-01` | `zone-dmz` | web | 0.5 |
| `node-web-02` | `zone-dmz` | web | 0.2 |
| `node-ws-01` | `zone-core` | workstation | 0.1 |

Agents: `agent-core` (5 nodes), `agent-dmz` (3 nodes). Total criticality 4.2. Edges (19, numbered in ascending order of source, target, type): `edge-001` app-01 to auth-01 link; `edge-002` app-01 to auth-01 dependency; `edge-003` app-01 to db-01 link; `edge-004` app-01 to db-01 dependency; `edge-005` app-01 to file-01 link; `edge-006` app-01 to web-01 link; `edge-007` app-01 to web-02 link; `edge-008` auth-01 to app-01 link; `edge-009` db-01 to app-01 link; `edge-010` file-01 to app-01 link; `edge-011` file-01 to ws-01 link; `edge-012` gw-01 to web-01 link; `edge-013` gw-01 to web-02 link; `edge-014` web-01 to app-01 link; `edge-015` web-01 to app-01 dependency; `edge-016` web-01 to gw-01 link; `edge-017` web-02 to app-01 link; `edge-018` web-02 to gw-01 link; `edge-019` ws-01 to file-01 link. (Dependencies: web-01 depends on app-01; app-01 depends on db-01 and on auth-01.) Link graph diameter: 4 (ws-01 to gw-01).

### 5.2 Scenarios

Twelve files in `configs/scenarios/mock/`, all with `max_steps = 40`, `observation_window_steps = 5`, `restore_duration_steps = 3`, `objective_node_id = node-db-01`, `alert_detect_probability = 0.35`, `alert_delay_max_steps = 1`, `missing_value_rate = 0.05`:

| Scenario id | Entry node | Start | Spread | False alert rate |
|-------------|-----------|-------|--------|------------------|
| `scn-mock-lateral-train-01` | `node-gw-01` | 3 | 0.20 | 0.02 |
| `scn-mock-lateral-train-02` | `node-web-01` | 3 | 0.20 | 0.02 |
| `scn-mock-lateral-train-03` | `node-web-02` | 3 | 0.20 | 0.02 |
| `scn-mock-lateral-train-04` | `node-gw-01` | 4 | 0.35 | 0.02 |
| `scn-mock-lateral-train-05` | `node-web-01` | 4 | 0.35 | 0.02 |
| `scn-mock-lateral-train-06` | `node-web-02` | 4 | 0.35 | 0.02 |
| `scn-mock-lateral-cal-01` | `node-web-02` | 4 | 0.25 | 0.03 |
| `scn-mock-lateral-cal-02` | `node-gw-01` | 3 | 0.30 | 0.03 |
| `scn-mock-lateral-eval-01` | `node-web-02` | 3 | 0.25 | 0.02 |
| `scn-mock-lateral-eval-02` | `node-file-01` | 5 | 0.25 | 0.02 |
| `scn-mock-lateral-eval-03` | `node-ws-01` | 4 | 0.30 | 0.02 |
| `scn-mock-lateral-eval-04` | `node-gw-01` | 3 | 0.45 | 0.05 |

Seeds (`configs/experiments/splits.toml`): train 1000 to 1019, calibration 1500 to 1509, evaluation 2000 to 2009. Smoke subset: train scenarios `train-01`, `train-04` with seeds 1000 to 1003; calibration `cal-01`, `cal-02` with seeds 1500 to 1503; evaluation scenarios `eval-01`, `eval-02` with seeds 2000 and 2001 (4 matched pairs).

### 5.3 State and dynamics

Per-node mutable state: `status`, `compromised`, `restore_left`, `rate_level` (`light`, `heavy` or none), `monitor_left`, `evidence` (level and step, or none), `last_investigated`, plus the pending alert queue. Random numbers use **common random numbers**: at `reset`, tables are drawn from `numpy.random.Generator(PCG64(seed))` for every (step, node) or (step, edge) cell, for the purposes spread, alert, delay, severity, category, false alert, health noise, traffic noise, missing, and evidence (two tables). The tables depend only on `(seed, H, N, E)`, never on actions. Two policies run on the same (scenario, seed) therefore face the same random draws, which makes matched comparisons less noisy.

Availability: `status_avail` is 1.0 for `up`, 0.85 (`light`) or 0.6 (`heavy`) for `rate_limited`, 0.5 for `quarantined`, 0.0 for `isolated` and `restoring`. `own_avail = status_avail * (0.7 if compromised else 1.0)`. `avail(n) = min(own_avail(n), min over dependencies d of avail(d))` (the dependency graph is acyclic). Containment status (`contained`): `quarantined`, `isolated`, `restoring`.

Phase P0 for step `t >= 1`, in this order:

1. Restore completion: for each node with `restore_left > 0` (ascending id), decrement; at 0 set status `up`, `compromised = False`, clear `rate_level`; emit `restore_completed`.
2. Incident start: at `t == incident.start_step` the entry node becomes compromised (`initial_entry`), unless its status is `restoring`.
3. Propagation (steps after the start step): for each node `c` compromised at the beginning of this phase, ascending id, and each outgoing `network_link` edge `c -> m`, ascending edge id: spread happens if `c` can send (`status` in `up`, `rate_limited`), `m` can receive (`status` not `isolated`, not `restoring`), `m` is not compromised, and `spread_u[t, edge] < spread_probability * f`, with `f = 1.0` for `up`, `0.6` for `light`, `0.3` for `heavy`. New compromises are recorded as `propagation`; a node compromised in this phase does not spread in the same phase.
4. Alerts: for each node ascending id: a compromised node raises an alert if `alert_u[t, n] < alert_detect_probability * (1.5 if monitored else 1.0)` (capped at 1); a non-compromised node raises a false alert if `false_u[t, n] < false_alert_rate * (0.5 if monitored else 1.0)`. The alert is emitted at step `t + floor(delay_u[t, n] * (alert_delay_max_steps + 1))` if that step is below `H`. Severity: true alert `medium` (60 percent) or `high` (40 percent); false alert `low` (70 percent) or `medium` (30 percent). Category: uniform over the three abstract categories. `visible_to` is the agent of the node's zone plus agents of other zones that have a node adjacent over a `network_link` edge. Truth (`true_positive`) is fixed at generation time.
5. Events emitted at P0 get ids in order: `restore_completed` events by node, then due alerts by node.

Phase P4 effects (action applied to a node, immediately, in ascending agent order):

| Action | Effect | Availability condition | Events (outcome phase) |
|--------|--------|------------------------|------------------------|
| `no_op` | none | always | `action_applied` |
| `monitor_node` | `monitor_left = duration_steps` (counts down at each P0) | status not `restoring` | `action_applied` |
| `investigate_node` | Evidence level drawn from table `evidence_u[t, n, 0]`; stored with step; `last_investigated = t` | status not `restoring` | `action_applied`, `investigation_result` |
| `request_evidence` | Same with table 1 | status not `restoring` | `action_applied`, `investigation_result` |
| `rate_limit_node` | status `rate_limited`, `rate_level` set | status `up` | `action_applied`, `node_status_changed` |
| `quarantine_node` | status `quarantined` | status `up` or `rate_limited` | `action_applied`, `node_status_changed` |
| `isolate_node` | status `isolated` | status `up`, `rate_limited` or `quarantined` | `action_applied`, `node_status_changed` |
| `restore_node` | status `restoring`, `restore_left = restore_duration_steps`, monitoring cleared | status not `restoring` | `action_applied`, `node_status_changed` |

Evidence tables (level 0 to 3): `investigate_node`: compromised node gives 3 with probability 0.60, 2 with 0.25, 1 with 0.15; healthy node gives 0 with 0.70, 1 with 0.20, 2 with 0.08, 3 with 0.02. `request_evidence`: compromised gives 3 with 0.90, 2 with 0.10; healthy gives 0 with 0.90, 1 with 0.10. Evidence stays visible for 5 steps, then reverts to null.

Observation measurements: `service_health = clip(avail(n) + 0.03 * noise, 0, 1)` (noise halved while monitored), null if `missing_u[t, n] < missing_value_rate`. Edge `traffic_level = clip(0.5 * state_factor + 0.15 * (source compromised and edge active) + 0.03 * noise, 0, 1)` with `state_factor` 1.0 active, 0.6 limited, 0.0 blocked; null with the missing rate. Edge state: `blocked` if source or target is `isolated` or `restoring`, or the source is `quarantined`; `limited` if source or target is `rate_limited`; else `active`.

End of step: `terminated` with reason `objective_reached` if the objective node is compromised; else `incident_resolved` if the incident has started, no node is compromised and every node is `up`; `truncated` (reason `max_steps`) if `t == H - 1` and not terminated.

State digest: SHA-256 of canonical JSON of `(episode_id, seed, step, node states, pending alerts, event counters, done flags)`.

### 5.4 What the mock deliberately does not do

No real networking, no packets, no hosts, no addresses, no files written by actions. One incident type only. No learning attacker. It is small so that tests run in milliseconds. It never imports `cyberswarm.simulator`, and nothing in `src/` outside `runner.factory` imports it.

## 6. Configuration files

All TOML is read with `tomllib` through `common.config.load_toml`, which rejects unknown keys. Scenario and topology files are JSON validated by the contracts.

| File | Keys |
|------|------|
| `configs/experiments/splits.toml` | `[train] scenarios, seeds`; `[cal] scenarios, seeds`; `[eval] scenarios, seeds`; `[smoke.train]`, `[smoke.cal]`, `[smoke.eval]` each with `scenarios, seeds` |
| `configs/experiments/reward_default.toml` | `weights_id = "reward-default-v1"`, `sharing = "team"`, `d_max = 20`, `false_isolation_base = 0.5`, `[weights]` with `attack_impact = 1.0`, `response_delay = 0.5`, `containment = 3.0`, `false_isolation = 2.0`, `healthy_disruption = 0.5`, `investigation_cost = 0.05`, `invalid_action = 0.1` |
| `configs/experiments/coordinator_default.toml` | `allowlist_id = "allowlist-default-v1"`, `allowlist = [the eight action kinds]`, `max_high_impact_per_step = 1` |
| `configs/experiments/ippo_default.toml` | `observation_mode = "conventional"`, `hidden = 64`, `gamma = 0.99`, `lam = 0.95`, `clip = 0.2`, `epochs = 4`, `lr = 3e-4`, `entropy_coef = 0.01`, `max_grad_norm = 0.5`, `episodes_per_update = 16`, `updates = 200`, `seed = 0` |
| `configs/experiments/ippo_smoke.toml` | as above with `hidden = 32`, `epochs = 2`, `episodes_per_update = 4`, `updates = 3` |
| `configs/experiments/uncertainty_default.toml` | `tau_evidence = 0.50`, `tau_defer = 0.30`, `e_min = 2`, `e_contain = 3`, `max_consecutive_substitutions = 2`, `calibration_bins = 10`, `min_samples = 100`, `auroc_min = 0.60`, `bootstrap_resamples = 1000` |
| `configs/experiments/eval_default.toml` | `[bootstrap] resamples = 1000, level = 0.95, seed = 0`; `[training] seeds = [0, 1, 2, 3, 4]`; `[policies] list = ["rule-conv", "rule-graph", "ippo-conv", "ippo-graph", "ippo-conv-ua", "ippo-conv-ctl"]` |
| `configs/experiments/reward_sensitivity.toml` | `[grid] multipliers = [0.5, 1.0, 2.0]`, `terms = [six main terms]`, `d_max = [10, 20, 40]`, `false_isolation_base = [0.0, 0.5, 1.0]` |
| `configs/experiments/smoke.toml` | `run_id = "run-smoke"`, `simulator = "mock"`, `topology_id`, `output_dir = "/Users/guru/CyberSwarm-GX/runs"`, `[stages] baselines = true, ippo = false, uncertainty = false` (S6 and S7 set them to true), `[paths]` to the other TOML files, `[data]` taken from `splits.toml` smoke subset |

## 7. Smoke command and dashboard-consumable outputs

### 7.1 The single smoke command

```
cd /Users/guru/CyberSwarm-GX && .venv/bin/python -m cyberswarm.runner.smoke --config /Users/guru/CyberSwarm-GX/configs/experiments/smoke.toml
```

It is idempotent (it deletes and recreates `runs/run-smoke/` only; refuses to delete any other path) and takes well under two minutes on CPU with the smoke budgets.

Stages (each can be disabled in `smoke.toml`; the exit status is 0 only if every enabled stage passes):

1. Load and validate configuration, topology and scenarios; check split disjointness.
2. Baselines (S5): run `rule-conv` and `rule-graph` on the smoke evaluation pairs: mock scenario, observation, graph conversion, proposal, coordinator validation, simulated action, outcome, reward/ground-truth stream, evaluator.
3. IPPO (S6): train with `ippo_smoke.toml` on the smoke training pairs, save a checkpoint under `runs/run-smoke/checkpoints/`, load it, run inference on the evaluation pairs, evaluate.
4. Uncertainty (S7): collect committing proposals of the matched control on the calibration pairs, fit the calibrator, run the uncertainty-aware policy and its control on the evaluation pairs, write `CalibrationReport`.
5. Evaluate and compare: `evaluation/metrics.jsonl`, `evaluation/comparison.jsonl` (only between equal matched sets).
6. Write `run_manifest.json` last.
7. Validate every emitted file against the pydantic models and the exported JSON Schemas, run the cross-record checks X-1 to X-13 and the leakage scan on agent-visible files.
8. Print a table of evaluator-computed metrics and the line `MOCK SIMULATOR RUN - NOT A RESEARCH RESULT`, the status block (`mappo: deferred`, `learned_gnn: not_implemented`), and the run directory path.

### 7.2 Dashboard-consumable output files

The run directory layout of CONTRACTS section 15. Mayukh's dashboard loads `runs/run-smoke/` or `fixtures/bundle_demo/` through the same loader. Files per episode: `events.jsonl`, `observations.jsonl`, `available_actions.jsonl`, `detection.jsonl`, `graph_snapshots.jsonl`, `graph_features.jsonl`, `proposals.jsonl`, `validations.jsonl`, `executions.jsonl`, `errors.jsonl`, `episode_result.json`, `evaluator_only/ground_truth.jsonl`. Files per run: `run_manifest.json`, `topologies/`, `scenarios/`, `evaluation/metrics.jsonl`, `evaluation/comparison.jsonl`, `evaluation/calibration.jsonl` (only with an uncertainty signal), `logs/run.log.jsonl`. No file is transformed for the dashboard; values that were not computed are `null`; `graph_model_outputs` is `not_computed` and no `graph_model_outputs` file exists.

## 8. Fixture list

All fixtures are synthetic and illustrative. They are labelled `fixture_synthetic` (records with provenance) and described as synthetic in `fixtures/README.md`. They are not results and not mock-run output. Three kinds:

- **Primary fixtures** are written from a literal storyboard (`scripts/fixture_storyboard.py`) that does not import the mock, the graph package, the coordinator or the evaluator. They are valid by construction against the contracts only.
- **Derived fixtures** (graph records in S3; episode results, metrics, comparison and manifest in S5) are produced by running the real converter and the real evaluator on the primary fixtures, then reviewed by hand against the formula tables of CONTRACTS section 10 and the metric definitions of section 11.2. They are golden files: a test fails if the committed file differs from regeneration (`scripts/build_fixtures.py --check`).
- **Invalid fixtures** each break exactly one rule. The test table in `tests/contracts/test_fixtures_validate.py` maps each file to the expected failure; a fixture that unexpectedly validates fails the test.

`scripts/build_fixtures.py` accepts `--write` and `--check`, and builds only what the installed stage supports: S1 builds the primary record fixtures, S3 adds the graph records, S5 adds results, metrics, comparison and the bundles.

### 8.1 Demo storyboard (scenario `scn-mock-lateral-eval-01`, seed 2001, 13 steps)

Hidden truth (written to `evaluator_only/ground_truth.jsonl` only): `node-web-02` is compromised at step 3; the compromise spreads to `node-app-01` at P0 of step 6 (before the isolation of `node-web-02` is executed in P4 of step 6); both are cleaned by restoration. Visible story for the rule-based conventional episode (`ep-rule-conv-v1-mock-lateral-eval-01-s2001`):

| Step | Events and observation | `agent-dmz` proposal and result | `agent-core` proposal and result |
|------|------------------------|---------------------------------|----------------------------------|
| 0 | `episode_started`; healthy network | `no_op`, accepted | `no_op`, accepted |
| 1 | False alert (low) on `node-gw-01`, visible to `agent-dmz` | `no_op` | `no_op` |
| 2 | Alert still in window | `investigate_node(node-gw-01)`, accepted | `no_op` |
| 3 | `investigation_result` evidence 0 (clean) on `node-gw-01` visible; hidden: incident starts | `no_op` | `no_op` |
| 4 | No visible event (the true alert on `node-web-02` is generated at step 4 with a one-step delay) | `no_op` | `no_op` |
| 5 | True alert (medium) on `node-web-02` emitted at P0 of step 5, visible to both agents | `investigate_node(node-web-02)`, accepted | `no_op` (the alert is on a boundary node) |
| 6 | Evidence 3 on `node-web-02`; hidden: spread to `node-app-01` | `isolate_node(node-web-02)`, accepted, executed | Injected faulty proposal `isolate_node(node-web-02)`: **rejected** `TARGET_OUT_OF_AUTHORITY` (no execution record). Real baselines do not do this; the storyboard injects it to exercise the rejected path |
| 7 | True alert on `node-app-01` visible to `agent-core`; `node-web-02` isolated | `restore_node(node-web-02)`, accepted | `investigate_node(node-app-01)`, accepted |
| 8 | Evidence 3 on `node-app-01` | `no_op` | `quarantine_node(node-app-01)` (criticality 0.7), accepted |
| 9 | | `no_op` | `restore_node(node-app-01)`, accepted |
| 10 | `restore_completed` for `node-web-02` | `no_op` | `no_op` |
| 11 | | `no_op` | `no_op` |
| 12 | `restore_completed` for `node-app-01`; `recovery_verified` (coordinator) for `node-web-02`; end of step: incident resolved | `no_op` | `no_op`; `episode_ended` (terminated) |

Counts: 26 proposals, 25 accepted and executed, 1 rejected. The graph-feature episode (`ep-rule-graph-v1-mock-lateral-eval-01-s2001`) uses the same hidden truth and differs by one decision: at step 5 `agent-core` proposes `monitor_node(node-app-01)` (premonitoring from the neighbour alert), which in the storyboard makes the alert on `node-app-01` appear one step earlier (step 6 instead of 7) and shifts the core agent's later actions one step earlier. The comparison fixture therefore shows a difference. This is illustrative storyboard data; it says nothing about either policy.

Graph snapshots are produced for both agents at every step: the `agent-dmz` view observes `node-app-01` (boundary), `node-gw-01`, `node-web-01`, `node-web-02`; the other four nodes are unobserved and imputed with mask 0. Step 5 and step 6 snapshots show changed dynamic state (alert counts, evidence, status after step 6) with identical `node_order` and `edge_order`. Some `service_health` values are `null` to exercise masks.

### 8.2 Files

<!--GEN:FIXTURES-->
| Fixture | Stage | Consumers | Content |
|---------|-------|-----------|---------|
| `fixtures/README.md` | S1 | Everyone | States that all fixtures are synthetic and illustrative, not results; explains valid, invalid, bundle directories |
| `fixtures/records/valid/Node.web-02.json` | S1 | contracts tests, Raksha, Mayukh | Low-value public web node |
| `fixtures/records/valid/Edge.link.json` | S1 | contracts tests, Raksha, Mayukh | `network_link` edge |
| `fixtures/records/valid/ScenarioConfig.eval-01.json` | S1 | contracts tests, Raksha, Mayukh | Scenario example (equals the configured eval-01) |
| `fixtures/records/valid/EventRecord.alert.json` | S1 | contracts tests, Raksha, Mayukh | `alert_raised` |
| `fixtures/records/valid/EventRecord.action_applied.json` | S1 | contracts tests, Raksha, Mayukh | `action_applied` outcome |
| `fixtures/records/valid/AgentObservation.dmz-step5.json` | S1 | contracts tests, Raksha, Mayukh | Own and boundary nodes, edges, window events |
| `fixtures/records/valid/AvailableActions.dmz-step5.json` | S1 | contracts tests, Raksha, Mayukh | State-dependent availability |
| `fixtures/records/valid/DetectionSummary.dmz-step5.json` | S1 | contracts tests, Raksha, Mayukh | Suspicion values |
| `fixtures/records/valid/PolicyProposal.investigate.json` | S1 | contracts tests, Raksha, Mayukh | Rule-based proposal without uncertainty (uncertainty null) |
| `fixtures/records/valid/PolicyProposal.uncertainty-evidence-request.json` | S1 | contracts tests, Raksha, Mayukh | Substituted proposal with `UncertaintyInfo` and `original_action` |
| `fixtures/records/valid/ValidationResult.accepted.json` | S1 | contracts tests, Raksha, Mayukh | Accepted with resolved action |
| `fixtures/records/valid/ValidationResult.rejected.json` | S1 | contracts tests, Raksha, Mayukh | Rejected with reason code |
| `fixtures/records/valid/ExecutedAction.isolate.json` | S1 | contracts tests, Raksha, Mayukh | Executed with status change and outcome events |
| `fixtures/records/valid/ExecutedAction.failed.json` | S1 | contracts tests, Raksha, Mayukh | Failed execution with `ACTION_EXECUTION_FAILED` |
| `fixtures/records/valid/GroundTruthSnapshot.step5.json` | S1 | contracts tests, Raksha, Mayukh | Evaluator-only truth |
| `fixtures/records/valid/GraphModelOutput.not_computed.json` | S1 | contracts tests, Raksha, Mayukh | Explicit not-computed record |
| `fixtures/records/valid/ErrorRecord.coordinator.json` | S1 | contracts tests, Raksha, Mayukh | `SCHEMA_INVALID` warning |
| `fixtures/records/valid/LogRecord.info.json` | S1 | contracts tests, Raksha, Mayukh | Log line |
| `fixtures/records/valid/CalibrationReport.insufficient.json` | S1 | contracts tests, Raksha, Mayukh | `insufficient_data` report with nulls |
| `fixtures/records/valid/GraphSnapshot.dmz-step5.json` | S3 | graph tests, Mayukh | Whole-topology snapshot, masks and imputation (8 nodes, 19 edges), produced by the converter from the observation fixture and reviewed against the formula tables |
| `fixtures/records/valid/GraphSnapshot.dmz-step6.json` | S3 | graph tests, Mayukh | Next step with changed dynamic state and identical node and edge order |
| `fixtures/records/valid/GraphFeatureRecord.dmz-step5.json` | S3 | graph tests, Mayukh | Hand-engineered features for step 5 |
| `fixtures/records/valid/EpisodeResult.demo.json` | S5 | evaluation tests, Mayukh | Episode result of the demo bundle episode |
| `fixtures/records/valid/EvaluationMetrics.demo.json` | S5 | evaluation tests, Mayukh | Metrics computed by the evaluator from the demo episode results |
| `fixtures/records/valid/ComparisonRecord.demo.json` | S5 | evaluation tests, Mayukh | Paired comparison of the two demo episodes |
| `fixtures/records/valid/RunManifest.demo.json` | S5 | evaluation tests, Mayukh | Manifest of `bundle_demo` |
| `fixtures/records/invalid/ActionSpec.unknown-kind.json` | S1 | contracts tests | Unknown `action_kind` |
| `fixtures/records/invalid/ActionSpec.raw-ip-target.json` | S1 | contracts tests | Target `10.0.0.5` |
| `fixtures/records/invalid/ActionSpec.ip-lookalike-target.json` | S1 | contracts tests | Target `node-10-0-0-1` |
| `fixtures/records/invalid/ActionSpec.extra-command-field.json` | S1 | contracts tests | Extra field `command` |
| `fixtures/records/invalid/ActionSpec.wrong-parameter.json` | S1 | contracts tests | `level` on `monitor_node` |
| `fixtures/records/invalid/AgentObservation.extra-ground-truth-field.json` | S1 | contracts tests | Extra field `compromised` |
| `fixtures/records/invalid/AgentObservation.future-event.json` | S1 | contracts tests | Event with step greater than the observation step |
| `fixtures/records/invalid/AgentObservation.boundary-private-value.json` | S1 | contracts tests | Boundary node with non-null `service_health` |
| `fixtures/records/invalid/PolicyProposal.unsafe-rationale-text.json` | S1 | contracts tests | Characters outside `ShortText` |
| `fixtures/records/invalid/EventRecord.alert-missing-node.json` | S1 | contracts tests | `alert_raised` without `node_id` |
| `fixtures/records/invalid/ValidationResult.rejected-without-reason.json` | S1 | contracts tests | Empty `reason_codes` while rejected |
| `fixtures/records/invalid/ValidationResult.accepted-without-action.json` | S1 | contracts tests | `resolved_action` null while accepted |
| `fixtures/records/invalid/ExecutedAction.failed-without-error-code.json` | S1 | contracts tests | `failed` with `error_code` null |
| `fixtures/records/invalid/GraphSnapshot.mask-shape-mismatch.json` | S3 | contracts tests | Mask shape differs from features |
| `fixtures/records/invalid/GraphSnapshot.imputed-nonzero.json` | S3 | contracts tests | Mask 0 with value not 0.0 |
| `fixtures/records/invalid/AgentObservation.nan-service-health.json` | S1 | contracts tests | JSON `NaN` |
| `fixtures/records/invalid/EpisodeResult.version-2.json` | S5 | contracts tests | `schema_version` 2.0 |
| `fixtures/records/invalid/ScenarioConfig.start-after-end.json` | S1 | contracts tests | `start_step` not less than `max_steps` |
| `fixtures/records/invalid/Topology.dangling-edge.json` | S1 | contracts tests | Edge to an unknown node |
| `fixtures/records/invalid/RunManifest.unknown-provenance.json` | S5 | contracts tests | Unknown `provenance.kind` |
| `fixtures/bundle_demo/run_manifest.json` | S5 | Mayukh, bundle tests | Run manifest with provenance `fixture_synthetic`, status block, `graph_model_outputs = not_computed` |
| `fixtures/bundle_demo/topologies/topo-mock-small-v1.json` | S5 | Mayukh | Copy of the topology |
| `fixtures/bundle_demo/scenarios/scn-mock-lateral-eval-01.json` | S5 | Mayukh | Copy of the scenario |
| `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/events.jsonl` | S5 | Mayukh, bundle tests | events.jsonl, rule-based conventional episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/observations.jsonl` | S5 | Mayukh, bundle tests | observations.jsonl, rule-based conventional episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/available_actions.jsonl` | S5 | Mayukh, bundle tests | available_actions.jsonl, rule-based conventional episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/detection.jsonl` | S5 | Mayukh, bundle tests | detection.jsonl, rule-based conventional episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/graph_snapshots.jsonl` | S5 | Mayukh, bundle tests | graph_snapshots.jsonl, rule-based conventional episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/graph_features.jsonl` | S5 | Mayukh, bundle tests | graph_features.jsonl, rule-based conventional episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/proposals.jsonl` | S5 | Mayukh, bundle tests | proposals.jsonl, rule-based conventional episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/validations.jsonl` | S5 | Mayukh, bundle tests | validations.jsonl, rule-based conventional episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/executions.jsonl` | S5 | Mayukh, bundle tests | executions.jsonl, rule-based conventional episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/errors.jsonl` | S5 | Mayukh, bundle tests | errors.jsonl, rule-based conventional episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/episode_result.json` | S5 | Mayukh, bundle tests | episode_result.json, rule-based conventional episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/evaluator_only/ground_truth.jsonl` | S5 | Mayukh, bundle tests | evaluator_only/ground_truth.jsonl, rule-based conventional episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/events.jsonl` | S5 | Mayukh, bundle tests | events.jsonl, graph-feature rule-based episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/observations.jsonl` | S5 | Mayukh, bundle tests | observations.jsonl, graph-feature rule-based episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/available_actions.jsonl` | S5 | Mayukh, bundle tests | available_actions.jsonl, graph-feature rule-based episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/detection.jsonl` | S5 | Mayukh, bundle tests | detection.jsonl, graph-feature rule-based episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/graph_snapshots.jsonl` | S5 | Mayukh, bundle tests | graph_snapshots.jsonl, graph-feature rule-based episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/graph_features.jsonl` | S5 | Mayukh, bundle tests | graph_features.jsonl, graph-feature rule-based episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/proposals.jsonl` | S5 | Mayukh, bundle tests | proposals.jsonl, graph-feature rule-based episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/validations.jsonl` | S5 | Mayukh, bundle tests | validations.jsonl, graph-feature rule-based episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/executions.jsonl` | S5 | Mayukh, bundle tests | executions.jsonl, graph-feature rule-based episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/errors.jsonl` | S5 | Mayukh, bundle tests | errors.jsonl, graph-feature rule-based episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/episode_result.json` | S5 | Mayukh, bundle tests | episode_result.json, graph-feature rule-based episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/evaluator_only/ground_truth.jsonl` | S5 | Mayukh, bundle tests | evaluator_only/ground_truth.jsonl, graph-feature rule-based episode (healthy start, incident, alerts, investigation, one rejected and one accepted containment, recovery check) |
| `fixtures/bundle_demo/evaluation/metrics.jsonl` | S5 | Mayukh | `EvaluationMetrics` for both policies, computed by the evaluator |
| `fixtures/bundle_demo/evaluation/comparison.jsonl` | S5 | Mayukh | `ComparisonRecord` |
| `fixtures/bundle_empty/run_manifest.json` | S5 | Mayukh (empty state) | Manifest with no episodes |
| `fixtures/bundle_partial/run_manifest.json` | S5 | Mayukh (partial state) | Manifest listing one episode that has only `events.jsonl` |
| `fixtures/bundle_partial/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/events.jsonl` | S5 | Mayukh (partial state) | Events only; no result file |
| `fixtures/bundle_malformed/run_manifest.json` | S5 | Mayukh (malformed state) | Manifest listing one episode with a malformed line (expected-invalid, excluded from the valid-fixture sweep by name) |
| `fixtures/bundle_malformed/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/events.jsonl` | S5 | Mayukh (malformed state) | Contains one line that is not valid JSON and one that violates the schema |
<!--/GEN:FIXTURES-->

## 9. DELTA check register and test files

Every requirement of `DELTA.md` (and the matching requirement of `BRIEF.md`) has an identifier. The register shows which tests verify it and which files implement it. A check with "(documentation and process check)" has no automated test of its own; it is verified by the documentation validator or by the final review.

<!--GEN:DCREG-->
| ID | Requirement (from DELTA.md and BRIEF.md) | Verified by (files under `tests/`) | Implemented in |
|----|-------------------------------------------|--------------------------|----------------|
| DC-01 | Typed, versioned contracts in a neutral package covering topology nodes and edges, graph snapshots and features, scenarios, events, observations, available actions, proposals, coordinator decisions, executed actions, episode outcomes, evaluator metrics, optional graph-model outputs, structured errors and log fields; machine-validated | `contracts/test_models_validation.py`, `contracts/test_schema_export.py` | `src/cyberswarm/contracts/__init__.py`, `src/cyberswarm/contracts/common.py`, `src/cyberswarm/contracts/topology.py`, `src/cyberswarm/contracts/scenario.py`, `src/cyberswarm/contracts/events.py`, `src/cyberswarm/contracts/observation.py`, `src/cyberswarm/contracts/actions.py`, `src/cyberswarm/contracts/graph.py`, `src/cyberswarm/contracts/results.py`, `src/cyberswarm/contracts/manifest.py`, `src/cyberswarm/contracts/registry.py`, `src/cyberswarm/contracts/export_schemas.py` |
| DC-02 | One serialization approach, stable identifiers, deterministic node ordering, exact step semantics, explicit compatibility policy, single abstract action vocabulary | `common/test_seeding_hashing.py`, `contracts/test_ids.py`, `contracts/test_action_vocabulary.py`, `contracts/test_serialization.py`, `contracts/test_schema_export.py`, `contracts/test_step_semantics.py`, `engine/test_step_engine.py` | `src/cyberswarm/contracts/ids.py`, `src/cyberswarm/contracts/vocabulary.py`, `src/cyberswarm/contracts/topology.py`, `src/cyberswarm/contracts/io.py`, `src/cyberswarm/engine/step_engine.py` |
| DC-03 | Optional uncertainty and learned-model values are null or absent when no producer computed them | `contracts/test_null_semantics.py`, `uncertainty/test_unavailable_uncertainty.py` | `src/cyberswarm/contracts/common.py`, `src/cyberswarm/contracts/actions.py` |
| DC-04 | Evaluator-only ground truth and future events never enter observations, graph conversion, graph features, model outputs, actor inputs, serialized info dictionaries or decision-time data | `contracts/test_models_validation.py`, `contracts/test_fixture_consistency.py`, `interfaces/test_conformance_mock.py`, `mock/test_mock_ground_truth_separation.py`, `integration/test_import_boundaries.py`, `graph/test_graph_leakage.py`, `rl/test_env_visibility_leakage.py`, `baselines/test_baseline_inputs.py`, `integration/test_leakage_scan_outputs.py` | `src/cyberswarm/contracts/events.py`, `src/cyberswarm/contracts/observation.py`, `src/cyberswarm/contracts/evaluator_only.py` |
| DC-05 | Neutral synthetic labelled fixtures (healthy and changing graph state, masks and imputation, conventional and graph observations, available actions, accepted and rejected coordinator paths, episode outcomes, evaluator metrics, dashboard consumption) and contract tests with cross-record consistency | `contracts/test_fixtures_validate.py`, `contracts/test_fixture_consistency.py`, `contracts/test_fixtures_up_to_date.py` | `scripts/fixture_storyboard.py`, `scripts/build_fixtures.py`, `src/cyberswarm/contracts/consistency.py` |
| DC-06 | Public Simulator Protocol and a lead-owned MOCK: stable ids, seeded determinism, agent-visible data only through the agent interface, separate ground-truth accessor, enough transitions and actions, no Raksha internals, passes conformance checks | `interfaces/test_simulator_protocol.py`, `interfaces/test_conformance_mock.py`, `mock/test_mock_dynamics.py`, `mock/test_mock_actions.py`, `mock/test_mock_determinism.py`, `mock/test_mock_ground_truth_separation.py` | `docs/MODULE_RAKSHA_SIMULATOR.md`, `configs/topologies/topo-mock-small-v1.json`, `src/cyberswarm/interfaces/__init__.py`, `src/cyberswarm/interfaces/simulator.py`, `src/cyberswarm/interfaces/conformance.py`, `src/cyberswarm/mock/__init__.py`, `src/cyberswarm/mock/simulator.py`, `src/cyberswarm/mock/dynamics.py`, `src/cyberswarm/mock/effects.py`, `src/cyberswarm/mock/observation.py`, `src/cyberswarm/mock/truth.py` |
| DC-07a | Graph: static topology | `graph/test_topology_graph.py` | `src/cyberswarm/graph/topology_graph.py` |
| DC-07b | Graph: dynamic updates | `graph/test_dynamic_updates.py` | `src/cyberswarm/graph/converter.py` |
| DC-07c | Graph: deterministic conversion | `graph/test_conversion_determinism.py` | `src/cyberswarm/graph/converter.py` |
| DC-07d | Graph: stable ordering across steps | `graph/test_conversion_determinism.py` | `src/cyberswarm/graph/converter.py` |
| DC-07e | Graph: tensor shapes and dtypes | `graph/test_tensors.py` | `src/cyberswarm/graph/tensors.py` |
| DC-07f | Graph: masks and imputation (deterministic, non-leaking) | `graph/test_masks_imputation.py` | `src/cyberswarm/graph/converter.py` |
| DC-07g | Graph: serialization | `graph/test_graph_serialization.py` | `src/cyberswarm/graph/converter.py` |
| DC-07h | Graph: batch processing (model-compatible interface without a learned GNN dependency) | `graph/test_tensors.py` | `src/cyberswarm/graph/tensors.py` |
| DC-07i | Graph: ID resolution | `graph/test_topology_graph.py` | `src/cyberswarm/graph/topology_graph.py` |
| DC-07j | Graph: leakage prevention (current and past events only) | `graph/test_graph_leakage.py` | `src/cyberswarm/graph/converter.py`, `src/cyberswarm/graph/features.py` |
| DC-07 | Graph pipeline as a whole: hand-engineered, inspectable features, provenance, built before the RL environment | `graph/test_topology_graph.py`, `graph/test_dynamic_updates.py`, `graph/test_conversion_determinism.py`, `graph/test_tensors.py`, `graph/test_masks_imputation.py`, `graph/test_graph_features.py`, `graph/test_graph_serialization.py`, `graph/test_graph_leakage.py` | `scripts/graph_evidence.py`, `src/cyberswarm/contracts/graph.py`, `src/cyberswarm/graph/__init__.py`, `src/cyberswarm/graph/topology_graph.py`, `src/cyberswarm/graph/converter.py`, `src/cyberswarm/graph/features.py`, `src/cyberswarm/graph/tensors.py` |
| DC-08 | RL environment over the public interface only: conventional and graph modes, spaces, action masks or invalid-action handling, reset and step lifecycle, seed behavior, termination versus truncation, role-specific visibility | `rl/test_spaces_and_masks.py`, `rl/test_env_lifecycle.py`, `rl/test_env_seeding.py`, `rl/test_env_obs_modes.py`, `rl/test_env_visibility_leakage.py`, `coordinator/test_env_with_coordinator.py` | `src/cyberswarm/rl/__init__.py`, `src/cyberswarm/rl/action_table.py`, `src/cyberswarm/rl/encoder.py`, `src/cyberswarm/rl/env.py` |
| DC-09 | Configurable reward terms (containment, response delay, criticality-weighted impact, false isolation, healthy-service disruption, investigation cost), documented sign and scale, sensitivity plan, hand-constructed tests | `rl/test_reward_terms.py`, `rl/test_reward_config.py`, `evaluation/test_reweight_hook.py` | `configs/experiments/reward_default.toml`, `configs/experiments/reward_sensitivity.toml`, `src/cyberswarm/rl/reward.py`, `src/cyberswarm/evaluation/reweight.py`, docs/RL_IMPLEMENTATION.md section 4 |
| DC-10 | Detection, investigation, decision and coordination as logical software roles; explicit policy-architecture choice; coordination independently testable | `integration/test_import_boundaries.py`, `roles/test_detection.py`, `roles/test_evidence.py` | `src/cyberswarm/interfaces/policy.py`, `src/cyberswarm/interfaces/ports.py`, `src/cyberswarm/roles/__init__.py`, `src/cyberswarm/roles/detection.py`, `src/cyberswarm/roles/evidence.py`, docs/ARCHITECTURE.md section 6, docs/DECISIONS.md AD-06 |
| DC-11 | Executable, tested rule-based baseline (investigates when evidence is insufficient) and non-neural graph-feature baseline; explicit inputs; matched scenarios and seeds | `baselines/test_rule_baseline.py`, `baselines/test_graph_feature_baseline.py`, `baselines/test_baseline_inputs.py` | `src/cyberswarm/baselines/__init__.py`, `src/cyberswarm/baselines/rule.py`, `src/cyberswarm/baselines/graph_rule.py` |
| DC-12 | Safe coordinator: schema validation, id resolution, allowlist, structured reason codes, unchanged state on rejection, execution only via the public API, contract-valid records, recovery verification, no shell, subprocess, socket or file mutation | `interfaces/test_simulator_protocol.py`, `coordinator/test_allowlist_validation.py`, `coordinator/test_rejection_state_unchanged.py`, `coordinator/test_execution_path.py`, `coordinator/test_recovery_verification.py`, `coordinator/test_source_guard.py`, `coordinator/test_env_with_coordinator.py`, `marl/test_inference_proposals.py` | `configs/experiments/coordinator_default.toml`, `src/cyberswarm/contracts/actions.py`, `src/cyberswarm/interfaces/ports.py`, `src/cyberswarm/coordinator/__init__.py`, `src/cyberswarm/coordinator/config.py`, `src/cyberswarm/coordinator/allowlist.py`, `src/cyberswarm/coordinator/validator.py`, `src/cyberswarm/coordinator/coordinator.py`, `src/cyberswarm/coordinator/recovery.py` |
| DC-13 | IPPO: independent actors, seeded training, configuration, logs, checkpoint save and load, evaluation mode, contract-valid inference, short mock budget, held-out evaluation, no quality claim | `marl/test_gae.py`, `marl/test_ppo_loss.py`, `marl/test_masked_sampling.py`, `marl/test_value_norm.py`, `marl/test_ippo_independence.py`, `marl/test_toy_task.py`, `marl/test_checkpoint_roundtrip.py`, `marl/test_inference_proposals.py`, `marl/test_marl_determinism.py`, `marl/test_ippo_smoke.py` | `configs/experiments/ippo_default.toml`, `configs/experiments/ippo_smoke.toml`, `src/cyberswarm/marl/__init__.py`, `src/cyberswarm/marl/networks.py`, `src/cyberswarm/marl/gae.py`, `src/cyberswarm/marl/value_norm.py`, `src/cyberswarm/marl/ppo.py`, `src/cyberswarm/marl/rollout.py`, `src/cyberswarm/marl/ippo.py`, `src/cyberswarm/marl/checkpoint.py`, `src/cyberswarm/marl/policy.py`, `src/cyberswarm/marl/train.py` |
| DC-14 | Uncertainty: defined signal, calibration, evidence request, high-impact deferral, containment despite uncertainty, unavailable as null, matched control | `uncertainty/test_entropy_signal.py`, `uncertainty/test_rule_boundaries.py`, `uncertainty/test_unavailable_uncertainty.py`, `uncertainty/test_matched_control.py`, `uncertainty/test_calibration.py` | `configs/scenarios/mock/scn-mock-lateral-cal-01.json`, `configs/scenarios/mock/scn-mock-lateral-cal-02.json`, `configs/experiments/uncertainty_default.toml`, `src/cyberswarm/uncertainty/__init__.py`, `src/cyberswarm/uncertainty/signal.py`, `src/cyberswarm/uncertainty/rules.py`, `src/cyberswarm/uncertainty/policy.py`, `src/cyberswarm/uncertainty/calibration.py`, docs/RL_IMPLEMENTATION.md section 7 |
| DC-15 | Evaluator: metrics from execution records, train and evaluation separation, matched scenarios and seeds, repeated-seed spread, dashboard records unchanged with nulls preserved | `evaluation/test_metrics_from_records.py`, `evaluation/test_outcome.py`, `evaluation/test_splits_disjoint.py`, `evaluation/test_matched_sets_and_stats.py`, `integration/test_bundle_schema_consumption.py` | `configs/scenarios/mock/scn-mock-lateral-train-01.json`, `configs/scenarios/mock/scn-mock-lateral-train-02.json`, `configs/scenarios/mock/scn-mock-lateral-train-03.json`, `configs/scenarios/mock/scn-mock-lateral-train-04.json`, `configs/scenarios/mock/scn-mock-lateral-train-05.json`, `configs/scenarios/mock/scn-mock-lateral-train-06.json`, `configs/scenarios/mock/scn-mock-lateral-cal-01.json`, `configs/scenarios/mock/scn-mock-lateral-cal-02.json`, `configs/scenarios/mock/scn-mock-lateral-eval-01.json`, `configs/scenarios/mock/scn-mock-lateral-eval-02.json`, `configs/scenarios/mock/scn-mock-lateral-eval-03.json`, `configs/scenarios/mock/scn-mock-lateral-eval-04.json`, `configs/experiments/splits.toml`, `configs/experiments/eval_default.toml`, `src/cyberswarm/contracts/results.py`, `src/cyberswarm/evaluation/__init__.py`, `src/cyberswarm/evaluation/outcome.py`, `src/cyberswarm/evaluation/metrics.py`, `src/cyberswarm/evaluation/stats.py`, `src/cyberswarm/evaluation/compare.py`, `src/cyberswarm/evaluation/splits.py`, `src/cyberswarm/runner/evaluate.py` |
| DC-16 | End-to-end smoke path from mock scenario to dashboard-consumable records | `engine/test_step_engine.py`, `integration/test_smoke_end_to_end.py`, `integration/test_bundle_schema_consumption.py`, `marl/test_ippo_smoke.py` | `configs/experiments/smoke.toml`, `src/cyberswarm/engine/__init__.py`, `src/cyberswarm/engine/step_engine.py`, `src/cyberswarm/engine/recorder.py`, `src/cyberswarm/runner/__init__.py`, `src/cyberswarm/runner/factory.py`, `src/cyberswarm/runner/export.py`, `src/cyberswarm/runner/smoke.py` |
| DC-17 | Learned GNN go or no-go documented (task, supervision, tensors, dependencies, compute, evaluation, GraphSAGE versus GAT) | `status/test_status_honesty.py` | docs/RL_IMPLEMENTATION.md section 8, docs/DECISIONS.md AD-11 |
| DC-18 | MAPPO honest deferral: design, blockers, work, API points, acceptance tests; code, tests, status and output consistent | `status/test_status_honesty.py` | `src/cyberswarm/marl/__init__.py`, docs/RL_IMPLEMENTATION.md section 6, docs/DECISIONS.md AD-10 |
| DC-19 | Mandatory implementation order | (documentation and process check) | `docs/AI_PROMPT_PROJECT_LEAD.md`, DESIGN.md section 11 stage order |
| DC-20 | Verification evidence: repo-local venv, minimal dependencies, full suite, determinism, documentation checks, exact commands and counts | `common/test_config.py`, `common/test_seeding_hashing.py`, `integration/test_determinism.py`, `docs/test_docs_validator.py` | `pyproject.toml`, `requirements-tested.txt`, `scripts/graph_evidence.py`, `scripts/validate_docs.py`, `src/cyberswarm/__init__.py`, `src/cyberswarm/common/__init__.py`, `src/cyberswarm/common/seeding.py`, `src/cyberswarm/common/hashing.py`, `src/cyberswarm/common/logging.py`, `src/cyberswarm/common/config.py`, `src/cyberswarm/runner/config.py`, DESIGN.md section 12 commands |
| DC-21 | Git and safety rules: gitignore, no commit or push, no secrets, no runtime network, no real offensive or containment capability | `coordinator/test_source_guard.py` | `.gitignore`, docs/DECISIONS.md AD-13, AD-14 |
| DC-22 | Documentation reconciliation: README, architecture, contracts, RL, testing, evaluation, integration, decisions, demo, ownership, prompts match code and status | `docs/test_docs_validator.py` | `scripts/validate_docs.py`, `README.md`, `docs/INTEGRATION.md`, `docs/TESTING.md`, `docs/EVALUATION.md`, `docs/DEMO.md`, `docs/TEAM_WORKFLOW.md`, `docs/MODULE_RAKSHA_SIMULATOR.md`, `docs/MODULE_MAYUKH_DASHBOARD.md`, `docs/AI_PROMPT_RAKSHA.md`, `docs/AI_PROMPT_MAYUKH.md`, `docs/AI_PROMPT_PROJECT_LEAD.md`, `docs/AI_INSTRUCTIONS.md`, `AGENTS.md`, `CLAUDE.md`, stage S8 reconciliation |
| DC-23 | Final reporting distinguishes implemented-and-tested work, mock and fixture evidence, unimplemented teammate work, GNN and MAPPO status | (documentation and process check) | final report of the last workflow step |
<!--/GEN:DCREG-->

Test files (every file runs under `.venv/bin/python -m pytest`):

<!--GEN:TESTS-->
| Test file | Stage | DELTA checks | What it asserts |
|-----------|-------|--------------|-----------------|
| `tests/conftest.py` | S1 | support | Shared fixtures: topology, scenarios, fixture paths, temporary run directory |
| `tests/common/test_config.py` | S1 | DC-20 | TOML loading, unknown keys rejected, resolved-configuration snapshot is canonical |
| `tests/common/test_seeding_hashing.py` | S1 | DC-20,DC-02 | `derive_seeds` streams are independent and stable; canonical SHA-256 stable |
| `tests/contracts/test_ids.py` | S1 | DC-02 | Id patterns, builders, address look-alike and shell-string rejection |
| `tests/contracts/test_models_validation.py` | S1 | DC-01,DC-04 | Valid and invalid cases per schema (`extra=forbid`, enums, ranges, per-type event fields, ground-truth fields rejected) |
| `tests/contracts/test_action_vocabulary.py` | S1 | DC-02 | Single vocabulary: kinds, targets, parameters, impact, high-impact sets, order; matches CONTRACTS section 6 |
| `tests/contracts/test_serialization.py` | S1 | DC-02 | Canonical JSON and JSONL bytes, float32 round trip, NaN rejection, version check |
| `tests/contracts/test_null_semantics.py` | S1 | DC-03 | Uncomputed uncertainty, model output, reward, comparison values stay null through round trip |
| `tests/contracts/test_schema_export.py` | S1 | DC-01,DC-02 | Exported schemas equal committed files; `$schema` and `$id`; compatibility policy cases |
| `tests/contracts/test_fixtures_validate.py` | S1 | DC-05 | Every fixture validates with pydantic and `jsonschema`; invalid fixtures fail as listed |
| `tests/contracts/test_fixture_consistency.py` | S1 | DC-05,DC-04 | Cross-record rules X-1 to X-14 on the demo bundle |
| `tests/contracts/test_fixtures_up_to_date.py` | S1 | DC-05 | `build_fixtures.py --check` finds no drift |
| `tests/contracts/test_step_semantics.py` | S1 | DC-02 | Phase order, step numbers per record kind, no future events helper |
| `tests/interfaces/test_simulator_protocol.py` | S2 | DC-06,DC-12 | Protocol surface; `ValidatedAction` only constructed by allowed modules |
| `tests/interfaces/test_conformance_mock.py` | S2 | DC-06,DC-04 | Runs the conformance suite on the mock: stable ids, resolvable edges, determinism, missing-value semantics, ground-truth separation |
| `tests/mock/test_mock_dynamics.py` | S2 | DC-06 | Incident start and spread, alert behavior, termination reasons, truncation |
| `tests/mock/test_mock_actions.py` | S2 | DC-06 | Effects and availability of every action; rejected application leaves digest unchanged |
| `tests/mock/test_mock_determinism.py` | S2 | DC-06 | Same scenario and seed gives identical events and digests; different seed differs |
| `tests/mock/test_mock_ground_truth_separation.py` | S2 | DC-04,DC-06 | No ground-truth field or future event in any agent-facing output |
| `tests/integration/test_import_boundaries.py` | S2 | DC-04,DC-10 | Package dependency order; only allowed modules import `evaluator_only` or call `ground_truth()`; no `mappo` module |
| `tests/graph/test_topology_graph.py` | S3 | DC-07a,DC-07i | Static topology, canonical order, id resolution, unknown id errors |
| `tests/graph/test_dynamic_updates.py` | S3 | DC-07b | Dynamic state changes across steps with stable ids and order |
| `tests/graph/test_conversion_determinism.py` | S3 | DC-07c,DC-07d | Same input gives identical snapshot; golden fixture match; order stable across steps |
| `tests/graph/test_tensors.py` | S3 | DC-07e,DC-07h | Shapes and dtypes; batch collation; optional torch conversion |
| `tests/graph/test_masks_imputation.py` | S3 | DC-07f | Unobserved nodes, null measurements, `impute-v1` values and masks |
| `tests/graph/test_graph_features.py` | S3 | DC-07 | Hand-computed values for each `graph-derived-v1` feature |
| `tests/graph/test_graph_serialization.py` | S3 | DC-07g | Snapshot and feature records round trip through JSONL and schemas |
| `tests/graph/test_graph_leakage.py` | S3 | DC-07j,DC-04 | Converter and features reject or ignore ground truth and future events; signatures have no truth input |
| `tests/roles/test_detection.py` | S3 | DC-10 | Suspicion formula and alerted flag on hand-built observations |
| `tests/roles/test_evidence.py` | S3 | DC-10 | Evidence view and insufficiency flag |
| `tests/engine/test_step_engine.py` | S3 | DC-02,DC-16 | Phase order, step numbers, record chain, ground-truth stream written |
| `tests/rl/test_spaces_and_masks.py` | S3 | DC-08 | Spaces, action tables (22 and 36 on the mock), masks equal available actions, invalid-action substitution |
| `tests/rl/test_env_lifecycle.py` | S3 | DC-08 | reset and step, final observation, error on step after end, termination versus truncation |
| `tests/rl/test_env_seeding.py` | S3 | DC-08 | Same seed same trajectory, different seed differs, seed streams independent |
| `tests/rl/test_env_obs_modes.py` | S3 | DC-08 | Both modes, dimensions 138 and 202, shared base block, matched policies |
| `tests/rl/test_env_visibility_leakage.py` | S3 | DC-04,DC-08 | Role visibility; observation, info and serialized payloads contain no ground-truth names or future events |
| `tests/rl/test_reward_terms.py` | S3 | DC-09 | Hand-constructed cases R-1 to R-10 |
| `tests/rl/test_reward_config.py` | S3 | DC-09 | Configurable weights, default file, sign and scale, team sharing |
| `tests/baselines/test_rule_baseline.py` | S4 | DC-11 | Investigates when evidence is insufficient; escalation ladder; valid proposals; only available actions |
| `tests/baselines/test_graph_feature_baseline.py` | S4 | DC-11 | Uses graph features; differs from rule baseline only where features matter; refuses missing features |
| `tests/baselines/test_baseline_inputs.py` | S4 | DC-11,DC-04 | Conventional baseline ignores perturbed graph features; both are deterministic |
| `tests/coordinator/test_allowlist_validation.py` | S4 | DC-12 | Every reason code, check order, allowlist restriction |
| `tests/coordinator/test_rejection_state_unchanged.py` | S4 | DC-12 | Rejected proposals leave the state digest unchanged |
| `tests/coordinator/test_execution_path.py` | S4 | DC-12 | Accepted actions go through the public API; conflicts; high-impact budget; record chain |
| `tests/coordinator/test_recovery_verification.py` | S4 | DC-12 | `recovery_verified` and `recovery_failed` from agent-visible data only |
| `tests/coordinator/test_source_guard.py` | S4 | DC-12,DC-21 | Source scan: no subprocess, os.system, socket, requests, urllib, shutil, file writes in `coordinator/` |
| `tests/coordinator/test_env_with_coordinator.py` | S4 | DC-12,DC-08 | Environment and baselines run through the real coordinator |
| `tests/evaluation/test_metrics_from_records.py` | S5 | DC-15 | Metrics computed from hand-built records equal known values; changing records changes metrics |
| `tests/evaluation/test_outcome.py` | S5 | DC-15 | Detection, containment time, impact, disruption, false isolation from hand-built truth |
| `tests/evaluation/test_splits_disjoint.py` | S5 | DC-15 | Train, cal and eval scenario ids and seeds are disjoint |
| `tests/evaluation/test_matched_sets_and_stats.py` | S5 | DC-15 | Matched set ids, paired comparison, mean, std, bootstrap, undefined handling |
| `tests/evaluation/test_reweight_hook.py` | S5 | DC-09 | Reweighting equals recomputation; identity at default weights |
| `tests/integration/test_smoke_end_to_end.py` | S5 | DC-16 | Runs the smoke command and checks stage results and exit status |
| `tests/integration/test_bundle_schema_consumption.py` | S5 | DC-15,DC-16 | Every emitted file validates against `schemas/`; dashboard loader contract (fixture and run bundles) |
| `tests/integration/test_leakage_scan_outputs.py` | S5 | DC-04 | Agent-visible files contain no evaluator-only field names and no future events |
| `tests/integration/test_determinism.py` | S5 | DC-20 | Two runs with the same seeds give byte-identical record files (log excluded) |
| `tests/marl/test_gae.py` | S6 | DC-13 | Hand-computed GAE with termination and truncation |
| `tests/marl/test_ppo_loss.py` | S6 | DC-13 | Closed-form clipped objective and gradient sign |
| `tests/marl/test_masked_sampling.py` | S6 | DC-13 | Masked actions never sampled; entropy ignores them |
| `tests/marl/test_value_norm.py` | S6 | DC-13 | Normalize and denormalize round trip |
| `tests/marl/test_ippo_independence.py` | S6 | DC-13 | Disjoint parameters, no cross gradients, update isolation |
| `tests/marl/test_toy_task.py` | S6 | DC-13 | Known-optimum toy task reaches 90 percent of optimum |
| `tests/marl/test_checkpoint_roundtrip.py` | S6 | DC-13 | Save and load equality; corrupted, wrong hash and wrong action table refused |
| `tests/marl/test_inference_proposals.py` | S6 | DC-13,DC-12 | Contract-valid proposals accepted by the coordinator |
| `tests/marl/test_marl_determinism.py` | S6 | DC-13 | Same seed same checkpoint hash and records |
| `tests/marl/test_ippo_smoke.py` | S6 | DC-13,DC-16 | Train, save, load, infer, held-out evaluation on the mock with the smoke budget |
| `tests/status/test_status_honesty.py` | S6 | DC-17,DC-18 | MAPPO deferred and learned GNN not implemented in code, manifests, status tables and docs |
| `tests/uncertainty/test_entropy_signal.py` | S7 | DC-14 | Test U-1 |
| `tests/uncertainty/test_rule_boundaries.py` | S7 | DC-14 | Tests U-2 to U-5, U-7 to U-9, U-13 |
| `tests/uncertainty/test_unavailable_uncertainty.py` | S7 | DC-14,DC-03 | Test U-6 |
| `tests/uncertainty/test_matched_control.py` | S7 | DC-14 | Test U-10 |
| `tests/uncertainty/test_calibration.py` | S7 | DC-14 | Tests U-11 and U-12 |
| `tests/docs/test_docs_validator.py` | S9 | DC-20,DC-22 | Runs `scripts/validate_docs.py` and checks that status claims match the code |
| `tests/rl/doubles.py` | S3 | support | Test-only `AcceptAllCoordinator` implementing `CoordinatorPort` (S3 precedes the real coordinator) |
| `tests/marl/toy_env.py` | S6 | support | Test-only cooperative toy environment (helper, not a test) |
<!--/GEN:TESTS-->

Test conventions: each test file names the DELTA check ids in its module docstring; tests use the mock only through `tests/conftest.py` factories; no test touches the network; tests that run training use the smoke budget; slow tests carry the `slow` marker and still run in the full suite.

## 10. Cross-check matrix

Every interface: producer, consumer, schema, valid and invalid fixture, test, stage. "—" means none applies (for example an internal dataclass has no JSON fixture).

| ID | Interface | Producer | Consumer | Schema or type | Valid fixture | Invalid fixture | Tests | Stage |
|----|-----------|----------|----------|----------------|---------------|-----------------|-------|-------|
| I-01 | Topology file (nodes, edges) | topology author | simulator, graph, rl, coordinator, evaluation, dashboard | `Node`, `Edge`, `Topology` | `Node.web-02`, `Edge.link`, `configs/topologies/topo-mock-small-v1.json` | `Topology.dangling-edge` | `contracts/test_models_validation`, `contracts/test_fixtures_validate`, `graph/test_topology_graph` | S1 |
| I-02 | Scenario configuration | scenario author | `Simulator.reset`, runner | `ScenarioConfig` | `ScenarioConfig.eval-01`, 12 scenario files | `ScenarioConfig.start-after-end` | `contracts/test_models_validation`, `evaluation/test_splits_disjoint` | S1 |
| I-03 | Simulator lifecycle (`reset`, `observe`, `available_actions`, `apply_action`, `end_step`, `drain_events`, `state_digest`) | mock; Raksha later | engine | `Simulator` Protocol | — | — | `interfaces/test_simulator_protocol`, `interfaces/test_conformance_mock`, `mock/test_mock_dynamics`, `mock/test_mock_actions`, `mock/test_mock_determinism` | S2 |
| I-04 | Ground-truth accessor | mock; Raksha later | engine (reward scorer, ground-truth stream) | `GroundTruthAccessor`, `GroundTruthSnapshot` | `GroundTruthSnapshot.step5` | — | `mock/test_mock_ground_truth_separation`, `integration/test_import_boundaries`, `integration/test_leakage_scan_outputs` | S2 |
| I-05 | Event stream | simulator; coordinator (recovery) | observation builder, graph, evaluation, dashboard | `EventRecord` | `EventRecord.alert`, `EventRecord.action_applied` | `EventRecord.alert-missing-node` | `contracts/test_models_validation`, `contracts/test_step_semantics`, `mock/test_mock_dynamics`, `coordinator/test_recovery_verification` | S1 |
| I-06 | Agent observation | simulator | roles, graph, rl encoder, baselines, dashboard | `AgentObservation` | `AgentObservation.dmz-step5` | `AgentObservation.extra-ground-truth-field`, `.future-event`, `.boundary-private-value`, `.nan-service-health` | `contracts/test_models_validation`, `mock/test_mock_ground_truth_separation`, `rl/test_env_visibility_leakage` | S1 |
| I-07 | Available actions | simulator | rl mask, baselines, coordinator, dashboard | `AvailableActions` | `AvailableActions.dmz-step5` | — | `contracts/test_models_validation`, `rl/test_spaces_and_masks`, `coordinator/test_allowlist_validation` | S1 |
| I-08 | Detection summary | `roles.detection` | rl encoder, baselines, dashboard | `DetectionSummary` | `DetectionSummary.dmz-step5` | — | `roles/test_detection` | S1 (schema), S3 (producer) |
| I-09 | Evidence view | `roles.evidence` | baselines, uncertainty | `EvidenceView` (internal dataclass) | — | — | `roles/test_evidence` | S3 |
| I-10 | Graph snapshot | `graph.converter` | `graph.features`, rl encoder, dashboard | `GraphSnapshot` | `GraphSnapshot.dmz-step5`, `GraphSnapshot.dmz-step6` | `GraphSnapshot.mask-shape-mismatch`, `.imputed-nonzero` | `graph/test_conversion_determinism`, `graph/test_masks_imputation`, `graph/test_dynamic_updates`, `graph/test_graph_serialization`, `graph/test_graph_leakage` | S3 |
| I-11 | Graph-derived features | `graph.features` | rl encoder (graph mode), graph baseline, dashboard | `GraphFeatureRecord` | `GraphFeatureRecord.dmz-step5` | — | `graph/test_graph_features` | S3 |
| I-12 | Tensor and batch interface | `graph.tensors` | future models, tests | `GraphTensors`, `GraphBatch` | derived from snapshot fixtures | — | `graph/test_tensors` | S3 |
| I-13 | Optional graph-model output | none in this phase | dashboard, manifest | `GraphModelOutput` | `GraphModelOutput.not_computed` | — | `contracts/test_null_semantics`, `status/test_status_honesty` | S1 |
| I-14 | Decision context and policy call | engine | policies | `DecisionContext`, `Policy` | — | — | `engine/test_step_engine`, `baselines/test_baseline_inputs`, `marl/test_inference_proposals` | S2 (types), S3 (producer) |
| I-15 | Policy proposal | policies | coordinator, evaluation, dashboard | `PolicyProposal` | `PolicyProposal.investigate`, `PolicyProposal.uncertainty-evidence-request` | `PolicyProposal.unsafe-rationale-text` | `baselines/test_rule_baseline`, `marl/test_inference_proposals`, `uncertainty/test_rule_boundaries`, `contracts/test_models_validation` | S1 |
| I-16 | Action vocabulary and `ActionSpec` | contracts | all | `ActionSpec`, `ActionKind` | embedded in proposals | `ActionSpec.unknown-kind`, `.raw-ip-target`, `.ip-lookalike-target`, `.extra-command-field`, `.wrong-parameter` | `contracts/test_action_vocabulary`, `contracts/test_ids`, `contracts/test_models_validation` | S1 |
| I-17 | Coordinator validation | coordinator | engine, evaluation, dashboard | `ValidationResult` | `ValidationResult.accepted`, `.rejected` | `.rejected-without-reason`, `.accepted-without-action` | `coordinator/test_allowlist_validation`, `coordinator/test_rejection_state_unchanged` | S1 (schema), S4 (producer) |
| I-18 | Validated action to simulator; execution record | coordinator; simulator | engine, evaluation, dashboard | `ValidatedAction`, `ActionApplication`, `ExecutedAction` | `ExecutedAction.isolate`, `ExecutedAction.failed` | `ExecutedAction.failed-without-error-code` | `coordinator/test_execution_path`, `mock/test_mock_actions`, `interfaces/test_simulator_protocol` | S1 (schema), S2 (Protocol), S4 (producer) |
| I-19 | Recovery verification | coordinator | engine, dashboard | `EventRecord` (`recovery_verified`, `recovery_failed`) | demo bundle events | — | `coordinator/test_recovery_verification` | S4 |
| I-20 | Reward scoring | `rl.reward` | engine, env | `StepScorer`, `StepScore`, `RewardConfig` | `reward_default.toml` | — | `rl/test_reward_terms`, `rl/test_reward_config`, `evaluation/test_reweight_hook` | S3 |
| I-21 | Environment spaces, masks, info | `rl.env` | marl, tests | Gymnasium spaces, info dict | — | — | `rl/test_spaces_and_masks`, `rl/test_env_lifecycle`, `rl/test_env_seeding`, `rl/test_env_obs_modes`, `rl/test_env_visibility_leakage` | S3 |
| I-22 | Error record | any component | runner, dashboard | `ErrorRecord` | `ErrorRecord.coordinator` | — | `contracts/test_models_validation`, `coordinator/test_allowlist_validation` | S1 |
| I-23 | Log record | `common.logging` | run log | `LogRecord` | `LogRecord.info` | — | `contracts/test_models_validation`, `integration/test_smoke_end_to_end` | S1 |
| I-24 | Episode result | runner with `evaluation.outcome` | evaluation, dashboard | `EpisodeResult` | `EpisodeResult.demo` | `EpisodeResult.version-2` | `evaluation/test_outcome`, `integration/test_bundle_schema_consumption`, `contracts/test_fixtures_validate` | S1 (schema), S5 (producer) |
| I-25 | Evaluation metrics | `evaluation.metrics` | dashboard | `EvaluationMetrics` | `EvaluationMetrics.demo` | — | `evaluation/test_metrics_from_records`, `evaluation/test_matched_sets_and_stats` | S1 (schema), S5 (producer) |
| I-26 | Paired comparison | `evaluation.compare` | dashboard | `ComparisonRecord` | `ComparisonRecord.demo` | — | `evaluation/test_matched_sets_and_stats` | S1 (schema), S5 (producer) |
| I-27 | Calibration report | `uncertainty.calibration` | evaluation (`uncertainty_ece`), dashboard | `CalibrationReport` | `CalibrationReport.insufficient` | — | `uncertainty/test_calibration`, `contracts/test_null_semantics` | S1 (schema), S7 (producer) |
| I-28 | Run manifest | `runner.export` | dashboard, validators | `RunManifest` | `RunManifest.demo` | `RunManifest.unknown-provenance` | `integration/test_bundle_schema_consumption`, `status/test_status_honesty` | S1 (schema), S5 (producer) |
| I-29 | Run directory layout | `runner.export`, `engine.recorder` | dashboard | files of CONTRACTS section 15 | `bundle_demo`, `bundle_empty`, `bundle_partial` | `bundle_malformed` | `integration/test_bundle_schema_consumption`, `integration/test_leakage_scan_outputs`, `integration/test_determinism` | S5 |
| I-30 | IPPO checkpoint | `marl.checkpoint` | `marl.policy`, runner | `checkpoint.pt`, `checkpoint.json` | generated by tests | corrupted, wrong hash, wrong action table (generated by tests) | `marl/test_checkpoint_roundtrip` | S6 |
| I-31 | Action distribution of a policy | `marl.policy` | `uncertainty.policy` | `PolicyDecision`, `DistributionPolicy` | — | — | `marl/test_inference_proposals`, `uncertainty/test_matched_control` | S2 (types), S6 |
| I-32 | Uncertainty information in proposals | `uncertainty.policy` | evaluation, dashboard | `UncertaintyInfo` (inside `PolicyProposal`) | `PolicyProposal.uncertainty-evidence-request` | — | `uncertainty/test_rule_boundaries`, `uncertainty/test_unavailable_uncertainty`, `contracts/test_null_semantics` | S1 (schema), S7 (producer) |
| I-33 | Splits and matched sets | `configs/experiments/splits.toml`; `evaluation.splits` | evaluation, runner, uncertainty | `Splits`, `MatchedSetId` | `splits.toml` | overlapping split (generated by tests) | `evaluation/test_splits_disjoint`, `evaluation/test_matched_sets_and_stats`, `uncertainty/test_calibration` | S1 (file), S5 (code) |
| I-34 | Configuration files | lead | runner, rl, coordinator, marl, uncertainty | pydantic config models | the TOML files | unknown key (generated by tests) | `common/test_config`, `rl/test_reward_config` | S1 |
| I-35 | Smoke command | `runner.smoke` | user, reviewers | command line | — | — | `integration/test_smoke_end_to_end`, `marl/test_ippo_smoke` | S5 (extended S6, S7) |
| I-36 | Reward reweighting | `evaluation.reweight` | evaluation | function | — | — | `evaluation/test_reweight_hook` | S5 |

### 10.1 Completeness proof (checked mechanically at design time and again at the end of S9)

1. **Every record in the registry (23) appears in the matrix**: `Node`, `Edge`, `Topology` (I-01), `ScenarioConfig` (I-02), `EventRecord` (I-05), `AgentObservation` (I-06), `AvailableActions` (I-07), `DetectionSummary` (I-08), `ActionSpec` (I-16), `PolicyProposal` (I-15), `ValidationResult` (I-17), `ExecutedAction` (I-18), `GroundTruthSnapshot` (I-04), `GraphSnapshot` (I-10), `GraphFeatureRecord` (I-11), `GraphModelOutput` (I-13), `EpisodeResult` (I-24), `EvaluationMetrics` (I-25), `ComparisonRecord` (I-26), `CalibrationReport` (I-27), `ErrorRecord` (I-22), `LogRecord` (I-23), `RunManifest` (I-28).
2. **No file is created by two stages**: section 11 lists each path exactly once (generated from the same inventory as section 2; the generator aborts on a duplicate path).
3. **Every test file named in this matrix exists in section 9**, and every test file in section 9 has at least one DELTA check id (or is marked support).
4. **Every valid fixture is consumed** by `contracts/test_fixtures_validate` and, where listed, by a component test; every invalid fixture appears in the invalid-case table of that test.
5. **Single owner of each interface**: each producer column names one component; the only two-producer record is `EventRecord`, split by the `producer` field and separate id prefixes (`evt-`, `cev-`).
6. **No interface is duplicated**: the action vocabulary exists once (`contracts/vocabulary.py`); the mask is derived from `AvailableActions` once (`rl/action_table.py`); the base node features are converted once (`graph/converter.py`) and reused by both observation modes.

## 11. Implementation stages S1 to S9

### 11.1 Overview

Each stage is one coder-then-reviewer loop. The reviewer step is always last in the loop and is approval-gated. A stage is complete only when its acceptance commands (11.3) were run and their output reported. Mandatory order: S1 to S9, no skipping. Stage numbering follows the user's split; the graph pipeline is the first part of S3 and the coordinator the second part of S4.

| Stage | Scope | DELTA order step |
|-------|-------|------------------|
| S1 | Scaffolding, virtual environment, dependencies, contracts, JSON Schema, fixtures, contract tests | 1 |
| S2 | Simulator Protocol, MOCK simulator, conformance suite | 1 |
| S3 | Graph pipeline and graph tests, roles, engine, RL environment and reward | 2, 3 |
| S4 | Rule-based baseline, graph-feature baseline, defense coordinator, safety tests | 4, 5 |
| S5 | Evaluation harness, reward sensitivity hook, smoke milestone, dashboard-consumable outputs | 8 |
| S6 | IPPO, MAPPO decision (deferral) and GNN no-go confirmation | 6, 9, 10 |
| S7 | Uncertainty-aware mechanism and matched control | 7 |
| S8 | Documentation reconciliation: README and all non-handoff documents | 11 |
| S9 | Raksha and Mayukh module specifications, the three AI prompts, `AGENTS.md`, documentation validator | 11 |
| Final | Independent whole-repository consistency and safety review (outside S1 to S9) | 12 |

### 11.2 Files created by each stage

A file belongs to exactly one stage (the stage that creates it). Later stages may extend a file; the extensions are named in 11.3 (for example the smoke configuration is created in S5 and its flags are set in S6 and S7; the fixture builder script is created in S1 and extended in S3 and S5).

<!--GEN:STAGES-->
#### S1. Scaffolding, virtual environment and dependencies, contracts, JSON Schema, fixtures, contract tests

Files created by this stage (91):

- `pyproject.toml`, `.gitignore`, `requirements-tested.txt`, `scripts/fixture_storyboard.py`, `scripts/build_fixtures.py`, `tests/conftest.py`, `fixtures/README.md`
- `configs/topologies/topo-mock-small-v1.json`
- `configs/scenarios/mock/scn-mock-lateral-train-01.json`, `configs/scenarios/mock/scn-mock-lateral-train-02.json`, `configs/scenarios/mock/scn-mock-lateral-train-03.json`, `configs/scenarios/mock/scn-mock-lateral-train-04.json`, `configs/scenarios/mock/scn-mock-lateral-train-05.json`, `configs/scenarios/mock/scn-mock-lateral-train-06.json`, `configs/scenarios/mock/scn-mock-lateral-cal-01.json`, `configs/scenarios/mock/scn-mock-lateral-cal-02.json`, `configs/scenarios/mock/scn-mock-lateral-eval-01.json`, `configs/scenarios/mock/scn-mock-lateral-eval-02.json`, `configs/scenarios/mock/scn-mock-lateral-eval-03.json`, `configs/scenarios/mock/scn-mock-lateral-eval-04.json`
- `configs/experiments/splits.toml`
- `src/cyberswarm/__init__.py`
- `src/cyberswarm/contracts/__init__.py`, `src/cyberswarm/contracts/common.py`, `src/cyberswarm/contracts/ids.py`, `src/cyberswarm/contracts/vocabulary.py`, `src/cyberswarm/contracts/topology.py`, `src/cyberswarm/contracts/scenario.py`, `src/cyberswarm/contracts/events.py`, `src/cyberswarm/contracts/observation.py`, `src/cyberswarm/contracts/actions.py`, `src/cyberswarm/contracts/graph.py`, `src/cyberswarm/contracts/results.py`, `src/cyberswarm/contracts/manifest.py`, `src/cyberswarm/contracts/evaluator_only.py`, `src/cyberswarm/contracts/io.py`, `src/cyberswarm/contracts/registry.py`, `src/cyberswarm/contracts/export_schemas.py`, `src/cyberswarm/contracts/consistency.py`
- `src/cyberswarm/common/__init__.py`, `src/cyberswarm/common/seeding.py`, `src/cyberswarm/common/hashing.py`, `src/cyberswarm/common/logging.py`, `src/cyberswarm/common/config.py`
- `tests/common/test_config.py`, `tests/common/test_seeding_hashing.py`
- `tests/contracts/test_ids.py`, `tests/contracts/test_models_validation.py`, `tests/contracts/test_action_vocabulary.py`, `tests/contracts/test_serialization.py`, `tests/contracts/test_null_semantics.py`, `tests/contracts/test_schema_export.py`, `tests/contracts/test_fixtures_validate.py`, `tests/contracts/test_fixture_consistency.py`, `tests/contracts/test_fixtures_up_to_date.py`, `tests/contracts/test_step_semantics.py`
- `fixtures/records/valid/Node.web-02.json`, `fixtures/records/valid/Edge.link.json`, `fixtures/records/valid/ScenarioConfig.eval-01.json`, `fixtures/records/valid/EventRecord.alert.json`, `fixtures/records/valid/EventRecord.action_applied.json`, `fixtures/records/valid/AgentObservation.dmz-step5.json`, `fixtures/records/valid/AvailableActions.dmz-step5.json`, `fixtures/records/valid/DetectionSummary.dmz-step5.json`, `fixtures/records/valid/PolicyProposal.investigate.json`, `fixtures/records/valid/PolicyProposal.uncertainty-evidence-request.json`, `fixtures/records/valid/ValidationResult.accepted.json`, `fixtures/records/valid/ValidationResult.rejected.json`, `fixtures/records/valid/ExecutedAction.isolate.json`, `fixtures/records/valid/ExecutedAction.failed.json`, `fixtures/records/valid/GroundTruthSnapshot.step5.json`, `fixtures/records/valid/GraphModelOutput.not_computed.json`, `fixtures/records/valid/ErrorRecord.coordinator.json`, `fixtures/records/valid/LogRecord.info.json`, `fixtures/records/valid/CalibrationReport.insufficient.json`, `fixtures/records/invalid/ActionSpec.unknown-kind.json`, `fixtures/records/invalid/ActionSpec.raw-ip-target.json`, `fixtures/records/invalid/ActionSpec.ip-lookalike-target.json`, `fixtures/records/invalid/ActionSpec.extra-command-field.json`, `fixtures/records/invalid/ActionSpec.wrong-parameter.json`, `fixtures/records/invalid/AgentObservation.extra-ground-truth-field.json`, `fixtures/records/invalid/AgentObservation.future-event.json`, `fixtures/records/invalid/AgentObservation.boundary-private-value.json`, `fixtures/records/invalid/PolicyProposal.unsafe-rationale-text.json`, `fixtures/records/invalid/EventRecord.alert-missing-node.json`, `fixtures/records/invalid/ValidationResult.rejected-without-reason.json`, `fixtures/records/invalid/ValidationResult.accepted-without-action.json`, `fixtures/records/invalid/ExecutedAction.failed-without-error-code.json`, `fixtures/records/invalid/AgentObservation.nan-service-health.json`, `fixtures/records/invalid/ScenarioConfig.start-after-end.json`, `fixtures/records/invalid/Topology.dangling-edge.json`

#### S2. Simulator Protocol, lead-owned MOCK simulator, conformance suite

Files created by this stage (18):

- `src/cyberswarm/interfaces/__init__.py`, `src/cyberswarm/interfaces/simulator.py`, `src/cyberswarm/interfaces/policy.py`, `src/cyberswarm/interfaces/ports.py`, `src/cyberswarm/interfaces/conformance.py`
- `src/cyberswarm/mock/__init__.py`, `src/cyberswarm/mock/simulator.py`, `src/cyberswarm/mock/dynamics.py`, `src/cyberswarm/mock/effects.py`, `src/cyberswarm/mock/observation.py`, `src/cyberswarm/mock/truth.py`
- `tests/interfaces/test_simulator_protocol.py`, `tests/interfaces/test_conformance_mock.py`
- `tests/mock/test_mock_dynamics.py`, `tests/mock/test_mock_actions.py`, `tests/mock/test_mock_determinism.py`, `tests/mock/test_mock_ground_truth_separation.py`
- `tests/integration/test_import_boundaries.py`

#### S3. Graph pipeline (first), roles, engine, RL environment and reward

Files created by this stage (42):

- `scripts/graph_evidence.py`
- `configs/experiments/reward_default.toml`
- `src/cyberswarm/roles/__init__.py`, `src/cyberswarm/roles/detection.py`, `src/cyberswarm/roles/evidence.py`
- `src/cyberswarm/graph/__init__.py`, `src/cyberswarm/graph/topology_graph.py`, `src/cyberswarm/graph/converter.py`, `src/cyberswarm/graph/features.py`, `src/cyberswarm/graph/tensors.py`
- `src/cyberswarm/engine/__init__.py`, `src/cyberswarm/engine/step_engine.py`, `src/cyberswarm/engine/recorder.py`
- `src/cyberswarm/rl/__init__.py`, `src/cyberswarm/rl/action_table.py`, `src/cyberswarm/rl/encoder.py`, `src/cyberswarm/rl/reward.py`, `src/cyberswarm/rl/env.py`
- `tests/graph/test_topology_graph.py`, `tests/graph/test_dynamic_updates.py`, `tests/graph/test_conversion_determinism.py`, `tests/graph/test_tensors.py`, `tests/graph/test_masks_imputation.py`, `tests/graph/test_graph_features.py`, `tests/graph/test_graph_serialization.py`, `tests/graph/test_graph_leakage.py`
- `tests/roles/test_detection.py`, `tests/roles/test_evidence.py`
- `tests/engine/test_step_engine.py`
- `tests/rl/test_spaces_and_masks.py`, `tests/rl/test_env_lifecycle.py`, `tests/rl/test_env_seeding.py`, `tests/rl/test_env_obs_modes.py`, `tests/rl/test_env_visibility_leakage.py`, `tests/rl/test_reward_terms.py`, `tests/rl/doubles.py`, `tests/rl/test_reward_config.py`
- `fixtures/records/valid/GraphSnapshot.dmz-step5.json`, `fixtures/records/valid/GraphSnapshot.dmz-step6.json`, `fixtures/records/valid/GraphFeatureRecord.dmz-step5.json`, `fixtures/records/invalid/GraphSnapshot.mask-shape-mismatch.json`, `fixtures/records/invalid/GraphSnapshot.imputed-nonzero.json`

#### S4. Rule-based and graph-feature baselines, defense coordinator, safety tests

Files created by this stage (19):

- `configs/experiments/coordinator_default.toml`
- `src/cyberswarm/baselines/__init__.py`, `src/cyberswarm/baselines/rule.py`, `src/cyberswarm/baselines/graph_rule.py`
- `src/cyberswarm/coordinator/__init__.py`, `src/cyberswarm/coordinator/config.py`, `src/cyberswarm/coordinator/allowlist.py`, `src/cyberswarm/coordinator/validator.py`, `src/cyberswarm/coordinator/coordinator.py`, `src/cyberswarm/coordinator/recovery.py`
- `tests/baselines/test_rule_baseline.py`, `tests/baselines/test_graph_feature_baseline.py`, `tests/baselines/test_baseline_inputs.py`
- `tests/coordinator/test_allowlist_validation.py`, `tests/coordinator/test_rejection_state_unchanged.py`, `tests/coordinator/test_execution_path.py`, `tests/coordinator/test_recovery_verification.py`, `tests/coordinator/test_source_guard.py`, `tests/coordinator/test_env_with_coordinator.py`

#### S5. Evaluation harness, reward sensitivity hook, smoke milestone, dashboard-consumable outputs

Files created by this stage (65):

- `configs/experiments/reward_sensitivity.toml`, `configs/experiments/eval_default.toml`, `configs/experiments/smoke.toml`
- `src/cyberswarm/evaluation/__init__.py`, `src/cyberswarm/evaluation/outcome.py`, `src/cyberswarm/evaluation/metrics.py`, `src/cyberswarm/evaluation/stats.py`, `src/cyberswarm/evaluation/compare.py`, `src/cyberswarm/evaluation/splits.py`, `src/cyberswarm/evaluation/reweight.py`
- `src/cyberswarm/runner/__init__.py`, `src/cyberswarm/runner/config.py`, `src/cyberswarm/runner/factory.py`, `src/cyberswarm/runner/evaluate.py`, `src/cyberswarm/runner/export.py`, `src/cyberswarm/runner/smoke.py`
- `tests/evaluation/test_metrics_from_records.py`, `tests/evaluation/test_outcome.py`, `tests/evaluation/test_splits_disjoint.py`, `tests/evaluation/test_matched_sets_and_stats.py`, `tests/evaluation/test_reweight_hook.py`
- `tests/integration/test_smoke_end_to_end.py`, `tests/integration/test_bundle_schema_consumption.py`, `tests/integration/test_leakage_scan_outputs.py`, `tests/integration/test_determinism.py`
- `fixtures/records/valid/EpisodeResult.demo.json`, `fixtures/records/valid/EvaluationMetrics.demo.json`, `fixtures/records/valid/ComparisonRecord.demo.json`, `fixtures/records/valid/RunManifest.demo.json`, `fixtures/records/invalid/EpisodeResult.version-2.json`, `fixtures/records/invalid/RunManifest.unknown-provenance.json`
- `fixtures/bundle_demo/run_manifest.json`, `fixtures/bundle_demo/topologies/topo-mock-small-v1.json`, `fixtures/bundle_demo/scenarios/scn-mock-lateral-eval-01.json`, `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/events.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/observations.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/available_actions.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/detection.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/graph_snapshots.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/graph_features.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/proposals.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/validations.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/executions.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/errors.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/episode_result.json`, `fixtures/bundle_demo/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/evaluator_only/ground_truth.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/events.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/observations.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/available_actions.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/detection.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/graph_snapshots.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/graph_features.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/proposals.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/validations.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/executions.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/errors.jsonl`, `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/episode_result.json`, `fixtures/bundle_demo/episodes/ep-rule-graph-v1-mock-lateral-eval-01-s2001/evaluator_only/ground_truth.jsonl`, `fixtures/bundle_demo/evaluation/metrics.jsonl`, `fixtures/bundle_demo/evaluation/comparison.jsonl`
- `fixtures/bundle_empty/run_manifest.json`
- `fixtures/bundle_partial/run_manifest.json`, `fixtures/bundle_partial/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/events.jsonl`
- `fixtures/bundle_malformed/run_manifest.json`, `fixtures/bundle_malformed/episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/events.jsonl`

#### S6. IPPO and the MAPPO decision (honest deferral)

Files created by this stage (24):

- `configs/experiments/ippo_default.toml`, `configs/experiments/ippo_smoke.toml`
- `src/cyberswarm/marl/__init__.py`, `src/cyberswarm/marl/networks.py`, `src/cyberswarm/marl/gae.py`, `src/cyberswarm/marl/value_norm.py`, `src/cyberswarm/marl/ppo.py`, `src/cyberswarm/marl/rollout.py`, `src/cyberswarm/marl/ippo.py`, `src/cyberswarm/marl/checkpoint.py`, `src/cyberswarm/marl/policy.py`, `src/cyberswarm/marl/train.py`
- `tests/marl/test_gae.py`, `tests/marl/test_ppo_loss.py`, `tests/marl/test_masked_sampling.py`, `tests/marl/test_value_norm.py`, `tests/marl/test_ippo_independence.py`, `tests/marl/test_toy_task.py`, `tests/marl/toy_env.py`, `tests/marl/test_checkpoint_roundtrip.py`, `tests/marl/test_inference_proposals.py`, `tests/marl/test_marl_determinism.py`, `tests/marl/test_ippo_smoke.py`
- `tests/status/test_status_honesty.py`

#### S7. Uncertainty-aware mechanism and matched control

Files created by this stage (11):

- `configs/experiments/uncertainty_default.toml`
- `src/cyberswarm/uncertainty/__init__.py`, `src/cyberswarm/uncertainty/signal.py`, `src/cyberswarm/uncertainty/rules.py`, `src/cyberswarm/uncertainty/policy.py`, `src/cyberswarm/uncertainty/calibration.py`
- `tests/uncertainty/test_entropy_signal.py`, `tests/uncertainty/test_rule_boundaries.py`, `tests/uncertainty/test_unavailable_uncertainty.py`, `tests/uncertainty/test_matched_control.py`, `tests/uncertainty/test_calibration.py`

#### S8. Documentation reconciliation (README and all non-handoff documents)

Files created by this stage (6):

- `README.md`, `docs/INTEGRATION.md`, `docs/TESTING.md`, `docs/EVALUATION.md`, `docs/DEMO.md`, `docs/TEAM_WORKFLOW.md`

#### S9. Raksha and Mayukh module specifications, the three AI prompts, AGENTS.md, documentation validator

Files created by this stage (10):

- `scripts/validate_docs.py`, `docs/MODULE_RAKSHA_SIMULATOR.md`, `docs/MODULE_MAYUKH_DASHBOARD.md`, `docs/AI_PROMPT_RAKSHA.md`, `docs/AI_PROMPT_MAYUKH.md`, `docs/AI_PROMPT_PROJECT_LEAD.md`, `docs/AI_INSTRUCTIONS.md`, `AGENTS.md`, `CLAUDE.md`
- `tests/docs/test_docs_validator.py`

<!--/GEN:STAGES-->

Schemas: `schemas/*.schema.json` (23 files) are created and regenerated by S1 and again whenever a contract changes. `fixtures/records/valid/` and `fixtures/records/invalid/` entries listed under S3 and S5 are added by those stages.

### 11.3 Order of work and acceptance per stage

| Stage | Order of work inside the stage | Acceptance commands (run from `/Users/guru/CyberSwarm-GX`) | Done when |
|-------|-------------------------------|-----------------------------------------------------------|-----------|
| S1 | (1) check free disk space; (2) `pyproject.toml`, `.gitignore`; (3) create `.venv` and install; (4) contracts package; (5) schema export; (6) storyboard and primary fixtures; (7) tests | `df -h /Users/guru/CyberSwarm-GX`; `python3 -m venv .venv`; `.venv/bin/pip install --no-cache-dir -e ".[rl,dev]"` (if disk is short: install without the `rl` group and record the blocker); `.venv/bin/pip freeze > requirements-tested.txt`; `.venv/bin/python -m cyberswarm.contracts.export_schemas --out /Users/guru/CyberSwarm-GX/schemas`; `.venv/bin/python scripts/build_fixtures.py --check`; `.venv/bin/python -m pytest tests/contracts tests/common -q` | 23 schema files match the models; every valid fixture validates by pydantic and `jsonschema`; every invalid fixture fails for the listed reason; contract tests pass with counts reported; no `LICENSE` file; `.venv` ignored |
| S2 | (1) `interfaces/simulator.py`, `policy.py`, `ports.py`; (2) mock dynamics, effects, observation, truth; (3) conformance mixin; (4) tests | `.venv/bin/python -m pytest tests/interfaces tests/mock tests/integration/test_import_boundaries.py -q` | Conformance suite passes on the mock; determinism and ground-truth-separation tests pass; import-boundary test passes |
| S3 | (1) `roles`; (2) `graph` (topology graph, converter, features, tensors); (3) graph fixtures and `scripts/graph_evidence.py`; (4) graph tests; (5) `engine`; (6) `rl.reward`; (7) `rl.action_table`, `rl.encoder`, `rl.env`; (8) tests with the coordinator test double | `.venv/bin/python scripts/build_fixtures.py --check`; `.venv/bin/python scripts/graph_evidence.py`; `.venv/bin/python -m pytest tests/graph tests/roles tests/engine tests/rl -q` | Graph tests (DC-07a to DC-07j) pass before any `rl` file exists in the commit order; environment lifecycle, masks, both modes, seeds, leakage and reward cases R-1 to R-10 pass; graph evidence output shows shapes, dtypes and masks without hidden labels |
| S4 | (1) `baselines` (rule, then graph-feature); (2) `coordinator` (config, allowlist, validator, coordinator, recovery); (3) tests | `.venv/bin/python -m pytest tests/baselines tests/coordinator -q` | Both baselines emit contract-valid proposals and investigate when evidence is insufficient; every reason code is produced by a test; rejection leaves the state digest unchanged; source guard passes; environment runs through the real coordinator |
| S5 | (1) `evaluation`; (2) `runner` (config, factory, evaluate, export, smoke); (3) bundles and derived fixtures; (4) integration tests; (5) smoke config | `.venv/bin/python -m cyberswarm.runner.smoke --config /Users/guru/CyberSwarm-GX/configs/experiments/smoke.toml`; `.venv/bin/python scripts/build_fixtures.py --check`; `.venv/bin/python -m pytest -q` | Smoke command exits 0 and prints evaluator-computed metrics labelled MOCK; every emitted file validates against `schemas/`; leakage scan and determinism tests pass; metrics come from records (changing a record changes the metric) |
| S6 | (1) `marl` units (GAE, loss, masking, value norm); (2) independence and toy-task tests; (3) trainer, checkpoint, policy; (4) train entry point and smoke integration; (5) status test; (6) confirm GNN no-go and MAPPO deferral in code, manifests and docs | `.venv/bin/python -m cyberswarm.marl.train --config /Users/guru/CyberSwarm-GX/configs/experiments/ippo_smoke.toml --out /Users/guru/CyberSwarm-GX/runs/ippo-smoke`; set `ippo = true` in `smoke.toml` and rerun the smoke command; `.venv/bin/python -m pytest tests/marl tests/status -q` | Train, save, load, infer and held-out evaluation run on the mock with the smoke budget; no claim of learned-policy quality; `mappo: deferred` and `learned_gnn: not_implemented` everywhere; no `mappo` module exists |
| S7 | (1) signal; (2) rules; (3) wrappers; (4) calibration; (5) tests; (6) enable `uncertainty = true` in `smoke.toml` | `.venv/bin/python -m pytest tests/uncertainty -q`; the smoke command with the uncertainty stage enabled | Tests U-1 to U-13 pass; matched control equivalence holds; the smoke run writes a `CalibrationReport` whose status is reported as computed (`insufficient_data` is acceptable at the smoke budget) |
| S8 | Reconcile `README.md` (rewrite around MARL, plain-text status, tree that matches reality, quickstart that works), `ARCHITECTURE.md`, `CONTRACTS.md`, `RL_IMPLEMENTATION.md`, `DECISIONS.md` (update statuses only), `docs/INTEGRATION.md`, `TESTING.md`, `EVALUATION.md`, `DEMO.md`, `TEAM_WORKFLOW.md` with the actual code, paths and commands | Run every quickstart command in the README exactly as written; `.venv/bin/python -m pytest -q` | Every path and command in these documents exists or was run; the "implemented, specified, not started" table is true; status words agree with `run_manifest` status |
| S9 | (1) `scripts/validate_docs.py`; (2) module specifications; (3) three prompts and `AI_INSTRUCTIONS.md`; (4) `AGENTS.md` (and `CLAUDE.md` only if non-conflicting) ; (5) `tests/docs` | `.venv/bin/python scripts/validate_docs.py`; `.venv/bin/python -m pytest -q` | All relative links and referenced paths resolve; every `json contract=` example validates; documented commands exist; names and ownership are consistent in all 16 required documents and the prompts; the full suite passes |

Extensions by later stages (the only files modified after their creation stage, besides S8 and S9 documentation edits): `scripts/build_fixtures.py` (S3, S5), `configs/experiments/smoke.toml` (S6, S7), `runner/factory.py` and `runner/smoke.py` (S6, S7: register IPPO and uncertainty policies and stages), `runner/export.py` (S6: `status.ippo`). Any other cross-stage edit must be recorded in the stage report.

### 11.4 Stage reports

Each coder step ends with a report containing: files created or changed, exact commands run, observed pass, fail and skip counts, anything not done, and the working-tree status (`git status --short`). Each reviewer step checks the stage against this blueprint, the DELTA register (section 9) and the cross-check matrix (section 10), and is last in the loop.

## 12. Verification commands, risks and open items

### 12.1 Final verification (after S9)

| Evidence | Command |
|----------|---------|
| Full suite | `.venv/bin/python -m pytest -q` (report passed, failed, skipped) |
| Focused graph evidence | `.venv/bin/python scripts/graph_evidence.py` |
| Smoke end to end | `.venv/bin/python -m cyberswarm.runner.smoke --config /Users/guru/CyberSwarm-GX/configs/experiments/smoke.toml` |
| IPPO train, save, load, infer | `.venv/bin/python -m cyberswarm.marl.train --config /Users/guru/CyberSwarm-GX/configs/experiments/ippo_smoke.toml --out /Users/guru/CyberSwarm-GX/runs/ippo-smoke` |
| Uncertainty and coordinator checks | `.venv/bin/python -m pytest tests/uncertainty tests/coordinator -q` |
| Determinism | `.venv/bin/python -m pytest tests/integration/test_determinism.py tests/mock/test_mock_determinism.py tests/marl/test_marl_determinism.py -q` |
| Schemas and fixtures | `.venv/bin/python -m cyberswarm.contracts.export_schemas --out /Users/guru/CyberSwarm-GX/schemas` then `git status --short schemas`; `.venv/bin/python scripts/build_fixtures.py --check` |
| Documentation | `.venv/bin/python scripts/validate_docs.py` |
| Working tree | `git status --short` (report only; never change it) |

### 12.2 Risks

| Risk | Mitigation |
|------|------------|
| Free disk space is below the PyTorch install size | S1 checks and stops with a blocker report; install `--no-cache-dir`; the user frees space |
| A self-written PPO is wrong but plausible | Hand-computed GAE and loss tests, toy task with known optimum, independence tests; TorchRL fallback trigger in RL_IMPLEMENTATION 5.1 |
| The environment (S3) precedes the coordinator (S4) | `CoordinatorPort` and a test double in S3; the real coordinator is wired in S4 with an integration test |
| Entropy is not informative | The calibration gate reports `not_informative`; results are labelled accordingly |
| The mock is too easy or too artificial | All outputs are labelled mock and not evidence; Raksha's simulator replaces it through the conformance suite |
| Contract drift between teammates | Only the lead edits contracts; exported schemas are tested against the models; fixtures are regenerated and checked |
| The graph pipeline is judged unnecessary | Isolated by AD-15; removal list is explicit |

### 12.3 Open items for the user

1. Confirm that the graph pipeline is required (AD-15).
2. Confirm the shared team reward and, before any MAPPO work, the centralized critic input.
3. Confirm the dashboard framework and Mayukh's skills (AD-09).
4. Decide the license (AD-13) and verify the git identity before any commit (AD-14).
5. Free disk space before S1 creates `.venv`.
