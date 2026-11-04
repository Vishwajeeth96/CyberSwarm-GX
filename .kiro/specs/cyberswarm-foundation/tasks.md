# Implementation Plan: CyberSwarm Foundation

## Overview

Build the project-lead-owned CyberSwarm foundation as real, tested Python 3.12 code under `/Users/guru/CyberSwarm-GX`, in the mandatory order fixed by the DELTA. Each task is sized to be independently testable and ends with the exact command(s) to run and the pass condition. Detail lives in the authoritative documents linked from [requirements.md](requirements.md) and [design.md](design.md); the per-stage "done when" criteria come from [`DESIGN.md` section 11.3](../../../.agents/tasks/cyberswarm-foundation/DESIGN.md) and the acceptance evidence from [`BRIEF.md`](../../../.agents/tasks/cyberswarm-foundation/BRIEF.md) and [`DELTA.md`](../../../.agents/tasks/cyberswarm-foundation/DELTA.md).

All commands run from the repository root `/Users/guru/CyberSwarm-GX` and use the repo-local interpreter `.venv/bin/python`. Every repository path in configs, docs, and commands is repository-relative; machine-specific absolute paths appear only in step prompts, never in committed content.

Each top-level task is tagged **[Implementation]** or **[Documentation]**.

## Guardrails (apply to every task)

- Work directly in `/Users/guru/CyberSwarm-GX`. Do NOT create a git worktree.
- Do NOT `git commit`, push, or change remotes, branches, tracking, collaborators, permissions, or git identity. The previously guessed git email must be re-verified by the user before any commit. Report `git status --short` without changing it.
- No secrets. No network-enabled runtime behavior.
- No real offensive or containment capability. All attacks, containment, isolation, restoration, and recovery are abstract simulator state transitions. No shell, subprocess, socket, network call, real address/command/payload, or filesystem mutation as an action effect anywhere in `src/`.
- Keep `.venv/`, `runs/`, `checkpoints/`, `logs/`, `__pycache__/`, `.pytest_cache/`, `*.pt`, `node_modules/`, `.env`, and datasets out of git.
- Report only actually-executed test results with observed pass/fail/skip counts. Never fabricate results. Label every mock-trained or fixture value as synthetic/mock-only, never as a research result.
- Before installing `torch`: run `df -h /Users/guru`; if free space is under 1.5 GB, STOP and report the measured number as a blocker instead of installing. Install with `pip install --no-cache-dir`. Measure again before the torch install.

## Tasks

- [x] 1. Project scaffold and environment **[Implementation]**
  - Establish the package layout, ignore rules, and a repo-local virtual environment with pinned, recorded dependencies.
  - _Requirements: 17.2, 17.4_

  - [x] 1.1 Create `pyproject.toml` and the `src/` layout
    - `pyproject.toml`: `name = "cyberswarm"`, `requires-python = ">=3.12"`, no `license` key; core deps `pydantic>=2.12,<3`, `numpy>=2.0`, `networkx>=3.4`; optional groups `rl = ["torch>=2.5","gymnasium>=1.0,<2","pettingzoo>=1.24"]`, `dev = ["pytest>=8","jsonschema>=4.20"]`; `[tool.setuptools.packages.find] where=["src"]`; pytest config with `testpaths=["tests"]`, `--import-mode=importlib`, a `slow` marker.
    - Create `src/cyberswarm/__init__.py` (version constant) and empty-but-intentional package dirs only as each stage needs them (no placeholder modules).
    - Acceptance: file parses; `python3 -c "import tomllib,pathlib; tomllib.loads(pathlib.Path('pyproject.toml').read_text())"` succeeds. No `LICENSE` file is added.
    - _Requirements: 17.2_

  - [x] 1.2 Create `.gitignore`
    - Ignore `.venv/`, `runs/`, `checkpoints/`, `__pycache__/`, `.pytest_cache/`, `*.pt`, `node_modules/`, `.env`, `*.log`, `datasets/`.
    - Acceptance: `git status --short` after creating `.venv/` later shows `.venv/` is not tracked. No secrets present.
    - _Requirements: 17.2, 17.3_

  - [x] 1.3 Create the virtual environment and record tested versions
    - Run `df -h /Users/guru`; if free < 1.5 GB, stop and report a blocker. Otherwise `python3 -m venv .venv`, re-measure disk, then `.venv/bin/pip install --no-cache-dir -e ".[rl,dev]"` (if disk is short, install without the `rl` group and record the blocker). Then `.venv/bin/pip freeze > requirements-tested.txt`.
    - Acceptance: `.venv/bin/python -c "import pydantic, numpy, networkx; print(pydantic.__version__)"` prints a 2.x version; `requirements-tested.txt` exists and records the actually-installed versions; torch import is reported as installed or the disk blocker is reported with the measured number.
    - _Requirements: 17.4, 17.5_

- [x] 2. Reconcile design docs and record corrected decisions **[Documentation]**
  - Bring `ARCHITECTURE.md` and `DECISIONS.md` into agreement with `RL_IMPLEMENTATION.md` v0.2 and the DELTA addenda, and record the corrected decisions that resolve the review findings, before implementation encodes them.
  - _Requirements: 15.1, 15.2, 15.3_

  - [x] 2.1 Reconcile `docs/ARCHITECTURE.md` and `docs/DECISIONS.md` to the corrected scope
    - Update stale entries so the learned GNN is a staged decision (not a design-time no-go), MAPPO is a planned stage (not a design-time deferral), and the graph pipeline is required infrastructure (not optional): AD-01, D-09, D-17, D-18, D-19, AD-10, AD-11, AD-15 and the architecture rows the review names.
    - Correct the architecture statements that the review flags as wrong: root-cause ranking IS an evaluated metric (origin top-1/MRR); the "policies differ only in the decision role" claim is qualified by the weakened `service_health` channel.
    - Acceptance: a grep over `docs/ARCHITECTURE.md` and `docs/DECISIONS.md` finds no remaining "no-go for this phase" for the GNN, no "deferred at design time" for MAPPO, and no "optional"/"future" status for the graph pipeline; the four HIGH findings' doc locations are updated. Report a diff summary.
    - _Requirements: 15.1_

  - [x] 2.2 Record corrected decisions for the MEDIUM findings and make shared content relative
    - In `docs/DECISIONS.md`, record decisions for: `ground_truth()` returns the end-of-step-`t` snapshot captured before P0 of `t+1` (SC-14); healthy-disruption uses `GroundTruthSnapshot.defense_availability` so attack-induced loss is not charged to the defender; the `service_health` compromise proxy is weakened (compromised factor 0.9, `service_health` noise 0.08, edge offset 0.05) with a mock AUROC guard; the event-visibility table and SC-15; the `eval_utopo` split and `topo-mock-small-v2`; the `NodeScoreRecord` and origin-ranking metric; execution-derived manifest status fields; a single evaluator-only import list with `evaluation.calibration_labels`; the IPPO/MAPPO training CLI under `runner/`; `ShortText` tightening; and keeping failed runs with no tuning on `eval`/`eval_utopo`.
    - Change every committed config and documented command to repository-relative paths (for example `output_dir = "runs"`, `--out schemas`, `--config configs/experiments/smoke.toml`).
    - Acceptance: `docs/DECISIONS.md` contains dated entries for each item above; a grep for `"/Users/guru"` in `configs/` and in documented quickstart commands returns nothing. Report the decision list.
    - _Requirements: 15.2, 15.3_

- [x] 3. Contracts, schema export, fixtures, and contract tests **[Implementation]**
  - Implement the neutral versioned contracts (including the corrected record additions), export JSON Schemas, build synthetic primary fixtures, and validate everything.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4_

  - [x] 3.1 Implement the contracts package
    - `src/cyberswarm/contracts/`: `common.py` (frozen `extra="forbid"` base, `Provenance`, enums, `ShortText` with the tightened charset rejecting dotted-quads, `..`, and leading `-`, `UnitFloat`, `ContractError`), `ids.py`, `vocabulary.py` (the eight-action table), `topology.py` (require `N >= 2` and a connected link graph), `scenario.py`, `events.py`, `observation.py`, `actions.py`, `graph.py` (`GraphSnapshot` with nullable `max_event_step`, mask rules), `results.py` (add `origin_rank_top1`/`origin_mrr` nullable fields keyed by `scorer_id`, `AggregateMetrics`, `policy_family_id`, `training_seed`), `manifest.py` (status fields `ippo`/`learned_gnn`/`mappo` in {`not_run`,`implemented`,`no_go`,`deferred`} with an optional required-when-set `reason`; `ErrorRecord.component` includes `engine`/`roles`/`mock`), `evaluator_only.py` (`GroundTruthSnapshot` with `service_availability` and `defense_availability`), `io.py`, `registry.py`, `consistency.py`. Add `NodeScoreRecord` and the `eval_utopo` value to the `Split` enum.
    - Acceptance: `.venv/bin/python -c "from cyberswarm.contracts.registry import RECORD_TYPES; print(len(RECORD_TYPES))"` imports cleanly and lists `NodeScoreRecord`; constructing a record with a dotted-quad `ShortText` raises `ContractError`.
    - _Requirements: 1.1, 1.3, 1.4_

  - [x] 3.2 Implement the common package
    - `src/cyberswarm/common/`: `seeding.py` (`derive_seeds`, `seed_everything` with lazy torch), `hashing.py` (canonical SHA-256), `logging.py` (JSON-line `LogRecord`, the only wall-clock reader), `config.py` (typed `tomllib` load rejecting unknown keys, resolved-config snapshot).
    - Acceptance: `.venv/bin/python -m pytest tests/common -q` passes with reported counts.
    - _Requirements: 1.1_

  - [x] 3.3 Export JSON Schemas
    - Implement `contracts/export_schemas.py`; run `.venv/bin/python -m cyberswarm.contracts.export_schemas --out schemas`.
    - Acceptance: one `schemas/<Name>.schema.json` per registered record; `git status --short schemas` after a re-export shows no diff; each schema carries `$schema` and `$id`.
    - _Requirements: 1.2_

  - [x] 3.4 Build the storyboard and primary fixtures
    - `scripts/fixture_storyboard.py` (literal data, imports no mock/graph/coordinator/evaluator), `scripts/build_fixtures.py` (`--write`/`--check`), `fixtures/README.md` (states all fixtures are synthetic and illustrative), and the S1 valid/invalid primary record fixtures plus a primary mini-episode for cross-record checks.
    - Acceptance: `.venv/bin/python scripts/build_fixtures.py --check` reports zero drift.
    - _Requirements: 2.1_

  - [x] 3.5 Write and run the contract tests
    - `tests/contracts/` (models validation, ids, action vocabulary, serialization/NaN/version, null semantics, schema export equality, fixtures validate with `jsonschema`, fixture consistency X-1..X-10 and X-14 on the S1 mini-episode, fixtures up-to-date, step semantics) and `tests/common/`.
    - Acceptance: `.venv/bin/python -m pytest tests/contracts tests/common -q` passes with 0 skipped; every valid fixture validates by pydantic and `jsonschema`; each invalid fixture fails for its one listed reason. Validates design Properties 1-5.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.2, 2.3, 2.4_

- [x] 4. Public simulator interface and conformance suite **[Implementation]**
  - Declare the simulator Protocols and the reusable conformance checks (SC-1..SC-15), with the evaluator-only boundary and import rules.
  - _Requirements: 3.1, 3.2, 3.3_

  - [x] 4.1 Implement interfaces and the conformance mixin
    - `src/cyberswarm/interfaces/`: `simulator.py` (`Simulator`, `GroundTruthAccessor`, `ValidatedAction` constructed only by the coordinator, `ActionApplication`, `StepResult`, `SimulatorError`; document that `ground_truth()` returns the end-of-step-`t` snapshot captured before P0 of `t+1`), `policy.py`, `ports.py` (`CoordinatorPort`, `StepScorer`), `conformance.py` (`SimulatorConformance` mixin with SC-1 through SC-15, including SC-14 ground-truth timing and SC-15 event visibility).
    - Acceptance: `.venv/bin/python -m pytest tests/interfaces/test_simulator_protocol.py -q` passes.
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 4.2 Enforce import boundaries
    - `tests/integration/test_import_boundaries.py`: the single evaluator-only import allowlist, the package dependency order, and only `engine.step_engine` and tests may call `ground_truth()`/`termination_reason()`.
    - Acceptance: `.venv/bin/python -m pytest tests/interfaces tests/integration/test_import_boundaries.py -q` passes.
    - _Requirements: 3.2_

- [x] 5. Lead-owned mock simulator **[Implementation]**
  - Implement the deterministic mock (incl. corrected dynamics) plus the training topology, the unseen-topology variant, scenarios, and splits; prove conformance.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 3.3_

  - [x] 5.1 Author topologies, scenarios, and splits
    - `configs/topologies/topo-mock-small-v1.json` (8 nodes, 2 zones, 19 edges) and `topo-mock-small-v2.json` (same node ids and zones, different edges and criticalities — the unseen variant); the 12 mock scenarios plus `scn-mock-lateral-utopo-01/-02.json` on v2; `configs/experiments/splits.toml` with `train`, `cal`, `eval`, `eval_utopo`, and the smoke subset.
    - Acceptance: all files validate via the contracts (`.venv/bin/python -m pytest tests/contracts/test_models_validation.py -q` extended to load them); v2 passes the topology validator.
    - _Requirements: 4.4_

  - [x] 5.2 Implement the mock dynamics, effects, observation, and truth
    - `src/cyberswarm/mock/`: `dynamics.py` (seeded common-random-number spread/alert/evidence; P0 ordering), `effects.py` (the eight action effects and availability), `observation.py` (agent-visible observations, available actions, per-event `visible_to` from the event-visibility table; `action_applied`/`investigation_result` visible only to the acting agent), `truth.py` (`GroundTruthSnapshot` with both `service_availability` and `defense_availability`; end-of-step-`t` capture before P0 of `t+1`), `simulator.py` (`kind="mock"`). Weaken the `service_health` compromise proxy (compromised factor 0.9, noise 0.08, edge offset 0.05).
    - Acceptance: `.venv/bin/python -c "from cyberswarm.mock import MockSimulator; print(MockSimulator)"` imports; module docstring and `simulator_kind` say MOCK; the mock imports no `cyberswarm.simulator`.
    - _Requirements: 4.1, 3.3_

  - [x] 5.3 Write and run the mock and conformance tests
    - `tests/interfaces/test_conformance_mock.py` (SC-1..SC-15, incl. SC-14 and SC-15), `tests/mock/` (dynamics, actions, determinism, ground-truth separation, and a `service_health` AUROC guard showing the channel is informative but below the agreed bound on a random-policy rollout).
    - Acceptance: `.venv/bin/python -m pytest tests/mock tests/interfaces/test_conformance_mock.py -q` passes with reported counts. Validates design Properties 6, 7, 8.
    - _Requirements: 4.2, 4.3, 4.4, 3.3_

- [x] 6. Observation-to-graph pipeline and graph tests **[Implementation]**
  - Implement roles and the deterministic graph pipeline (snapshot, hand-engineered features, tensors/batch), with leakage prevention, before the RL environment.
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 6.1 Implement the detection and evidence roles
    - `src/cyberswarm/roles/`: `detection.py` (`DetectionRole.summarize` → `DetectionSummary` with per-node suspicion), `evidence.py` (`evidence_view`), and a single `EvidenceConfig` (`e_min`) referenced by baselines and the uncertainty rule.
    - Acceptance: `.venv/bin/python -m pytest tests/roles -q` passes.
    - _Requirements: 5.2_

  - [x] 6.2 Implement the graph conversion, features, and tensors
    - `src/cyberswarm/graph/`: `topology_graph.py` (stable order, id resolution), `converter.py` (`convert` → `GraphSnapshot`, `impute-v1`, agent-visible + current/past events only, no truth parameter), `features.py` (`graph-derived-v1` from snapshot and topology only), `tensors.py` (`GraphTensors`/`GraphBatch`, documented shapes/dtypes, lazy `to_torch`).
    - Acceptance: converter and feature function signatures accept no ground-truth argument; `.venv/bin/python -c "import cyberswarm.graph"` imports without torch.
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 6.3 Build graph fixtures and the focused graph-evidence command
    - Graph record fixtures (`GraphSnapshot.dmz-step5/6`, `GraphFeatureRecord.dmz-step5`, invalid mask/imputation fixtures) via `scripts/build_fixtures.py`; `scripts/graph_evidence.py` printing fixture → validated snapshot → features → tensor shapes/dtypes/masks.
    - Acceptance: `.venv/bin/python scripts/build_fixtures.py --check` reports zero drift; `.venv/bin/python scripts/graph_evidence.py` prints shapes, dtypes, and masks with no hidden labels.
    - _Requirements: 5.1, 5.4_

  - [x] 6.4 Write and run the graph and role tests
    - `tests/graph/` (topology, dynamic updates, conversion determinism and stable order, tensors/batch, masks/imputation, hand-computed features, serialization round-trip, leakage) and `tests/roles/`; include a test that v1 and v2 produce different features for the same observation.
    - Acceptance: `.venv/bin/python -m pytest tests/graph tests/roles -q` passes with reported counts. Validates design Properties 5, 10.
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 7. RL environment and configurable reward **[Implementation]**
  - Implement the episode engine, the reward model (with defender-induced disruption separation), and the PettingZoo-style environment with both observation modes and the `state()` hook.
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4_

  - [x] 7.1 Implement the episode engine and recorders
    - `src/cyberswarm/engine/`: `step_engine.py` (phases P0-P4, `CoordinatorPort`, `StepScorer`, ground-truth stream queried as the end-of-step snapshot after `end_step()`), `recorder.py` (`EpisodeRecorder`, `MemoryRecorder`, `NullRecorder`).
    - Acceptance: `.venv/bin/python -m pytest tests/engine -q` passes (phase order, step numbers, record chain, ground-truth stream written).
    - _Requirements: 6.1_

  - [x] 7.2 Implement the configurable reward model
    - `configs/experiments/reward_default.toml` and `src/cyberswarm/rl/reward.py` (`RewardConfig`, `RewardBreakdown`, `RewardModel` implementing `StepScorer`); healthy-disruption uses `defense_availability`; store unweighted per-term magnitudes.
    - Acceptance: `.venv/bin/python -m pytest tests/rl/test_reward_terms.py tests/rl/test_reward_config.py -q` passes cases R-1..R-12, including critical-vs-low-value ordering and justified-containment-vs-inaction. Validates design Properties 13, 14, 15.
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 7.3 Implement the RL action table, encoder, and environment
    - `src/cyberswarm/rl/`: `action_table.py` (sizes 22 and 36 on the mock, mask from `AvailableActions`), `encoder.py` (dims 138 and 202), `env.py` (`CyberSwarmParallelEnv` over the engine, `conventional`/`graph_features` modes, invalid-action substitution, `reset` requires a seed, final-observation `no_op`-only mask, and the `state()` hook returning the ascending concatenation of agent-visible vectors).
    - Acceptance: `.venv/bin/python -c "from cyberswarm.rl.env import CyberSwarmParallelEnv"` imports; env never imports a simulator implementation.
    - _Requirements: 6.1, 6.2, 6.3, 6.5_

  - [x] 7.4 Write and run the environment and reward tests
    - `tests/rl/` (spaces and masks, lifecycle, seeding, both observation modes, visibility/leakage, `test_env_state` for the `state()` hook) using the `tests/rl/doubles.py` accept-all coordinator port.
    - Acceptance: `.venv/bin/python -m pytest tests/rl tests/engine -q` passes with reported counts. Validates design Properties 5, 11, 12.
    - _Requirements: 6.2, 6.3, 6.4, 6.5_

- [ ] 8. Conventional and graph-feature baselines **[Implementation]**
  - Implement both non-learned policies, emitting contract-valid proposals and per-node suspicion scores, with the contained-node re-investigation rule.
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [x] 8.1 Implement the rule-based conventional baseline
    - `src/cyberswarm/baselines/rule.py` (`RuleBasedPolicy`, conventional inputs only, the escalation ladder plus the rule that re-investigates a contained compromised node with null/weak/moderate evidence); emit `NodeScoreRecord` from the detection suspicion.
    - Acceptance: builds a contract-valid proposal; selects only actions in `AvailableActions`.
    - _Requirements: 8.1, 8.3, 8.4_

  - [x] 8.2 Implement the non-neural graph-feature baseline
    - `src/cyberswarm/baselines/graph_rule.py` (`GraphFeaturePolicy`, same ladder using graph features; raises `PolicyError` when features are absent); emit `NodeScoreRecord` from the graph suspicion score.
    - Acceptance: proposal differs from the conventional baseline only where graph features matter; raises when `ctx.graph_features is None`.
    - _Requirements: 8.1, 8.2, 8.3_

  - [-] 8.3 Write and run the baseline tests
    - `tests/baselines/` (rule investigates when evidence is insufficient; graph baseline uses features and refuses when absent; conventional baseline ignores perturbed graph features; both deterministic; a quarantined compromised node whose evidence expired is re-investigated and the episode can reach `incident_resolved`).
    - Acceptance: `.venv/bin/python -m pytest tests/baselines -q` passes with reported counts. Validates design Property 16.
    - _Requirements: 8.1, 8.2, 8.4_

- [ ] 9. Safe defense coordinator and safety tests **[Implementation]**
  - Implement the sole action-validation and execution boundary with ordered reason codes, unchanged state on rejection, recovery verification, and a source guard.
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [ ] 9.1 Implement the coordinator
    - `configs/experiments/coordinator_default.toml` and `src/cyberswarm/coordinator/`: `config.py`, `allowlist.py`, `validator.py` (sole creator of `ValidatedAction`; reason codes in the defined dependency order with the defined id-assignment order), `coordinator.py` (`DefenseCoordinator` implementing `CoordinatorPort`; executes only via `simulator.apply_action`), `recovery.py` (`recovery_verified`/`recovery_failed` from agent-visible data, tracking executed restores only). No shell/subprocess/socket/network/file-write anywhere in `coordinator/`.
    - Acceptance: `.venv/bin/python -c "from cyberswarm.coordinator import DefenseCoordinator"` imports.
    - _Requirements: 9.1, 9.3, 9.4_

  - [~] 9.2 Write and run the coordinator and safety tests
    - `tests/coordinator/` (every reason code and the check order; rejection leaves `state_digest()` unchanged; accepted actions execute via the public API with conflict and high-impact-budget handling; recovery verified/failed/null-breaks-streak; source guard scans `coordinator/` for forbidden tokens; environment and baselines run through the real coordinator).
    - Acceptance: `.venv/bin/python -m pytest tests/coordinator -q` passes with reported counts. Validates design Property 8.
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ] 10. Self-written IPPO with a thin evaluation slice **[Implementation]**
  - Implement compact PyTorch IPPO (independent actors), train/save/load/infer on the mock with the smoke budget, and a thin evaluation slice so held-out evaluation executes.
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [~] 10.1 Implement the PPO units
    - `src/cyberswarm/marl/`: `networks.py` (`masked_categorical`, MLP builders), `gae.py` (`compute_gae`), `value_norm.py` (`ValueNorm`), `ppo.py` (clipped update).
    - Acceptance: `.venv/bin/python -m pytest tests/marl/test_gae.py tests/marl/test_ppo_loss.py tests/marl/test_masked_sampling.py tests/marl/test_value_norm.py -q` passes hand-computed values.
    - _Requirements: 10.1_

  - [~] 10.2 Implement the trainer, checkpoint, and policy
    - `src/cyberswarm/marl/`: `rollout.py` (joint rollout storing the per-step concatenated state for later MAPPO use), `ippo.py` (`IPPOConfig`, `IPPOTrainer`, per-agent actor/critic/optimizer/value-norm), `checkpoint.py` (`save_checkpoint`/`load_checkpoint` verifying hash, node-id list, and action-table digest, NOT topology id, so a v1 checkpoint loads on v2), `policy.py` (`IPPOPolicy` implementing `DistributionPolicy`). Configs `ippo_default.toml`, `ippo_smoke.toml`.
    - Acceptance: `.venv/bin/python -m pytest tests/marl/test_ippo_independence.py tests/marl/test_toy_task.py tests/marl/test_checkpoint_roundtrip.py tests/marl/test_marl_determinism.py -q` passes.
    - _Requirements: 10.1, 10.2_

  - [~] 10.3 Implement the training CLI and the thin evaluation slice
    - `src/cyberswarm/runner/train_ippo.py` (`python -m cyberswarm.runner.train_ippo`, in `runner/` to respect import boundaries), plus a thin `runner/evaluate.run_episode` and `evaluation/outcome.compute_outcome` so held-out evaluation runs for one slice.
    - Acceptance: `.venv/bin/python -m cyberswarm.runner.train_ippo --config configs/experiments/ippo_smoke.toml --out runs/ippo-smoke` writes `training_log.jsonl` and a checkpoint under a gitignored dir, then loads it and runs greedy evaluation.
    - _Requirements: 10.2, 10.3_

  - [~] 10.4 Write and run the IPPO and status tests
    - `tests/marl/test_inference_proposals.py` (contract-valid proposals accepted by the real coordinator), `tests/marl/test_ippo_smoke.py` (train/save/load/infer/held-out evaluation at the smoke budget), and `tests/runner/test_manifest_status.py` (manifest status equals what executed; no hard-coded constants).
    - Acceptance: `.venv/bin/python -m pytest tests/marl tests/runner -q` passes with reported counts; no learned-policy quality is claimed; mock numbers labelled mock-only. Validates design Properties 17, 18, 19.
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 11. Uncertainty-aware wrapper and matched control **[Implementation]**
  - Implement the normalized-entropy signal, the decision rules, the matched control, and calibration fed by evaluator-supplied correctness arrays.
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [~] 11.1 Implement the signal, rules, and wrappers
    - `src/cyberswarm/uncertainty/`: `signal.py` (`normalized_entropy`), `rules.py` (`UncertaintyRuleConfig`, `apply_rules` for U0-U6), `policy.py` (`UncertaintyAwarePolicy`, `MatchedControlPolicy`); `configs/experiments/uncertainty_default.toml`.
    - Acceptance: `.venv/bin/python -m pytest tests/uncertainty/test_entropy_signal.py tests/uncertainty/test_rule_boundaries.py tests/uncertainty/test_unavailable_uncertainty.py -q` passes (U-1..U-9, U-13).
    - _Requirements: 11.1, 11.2_

  - [~] 11.2 Implement calibration and the evaluator-only label bridge
    - `src/cyberswarm/uncertainty/calibration.py` (reliability bins, ECE/MCE/Brier/AUROC, histogram calibrator, threshold selection on `cal`, `CalibrationReport`) and `evaluation/calibration_labels.py` (`calibration_labels(proposals, truth) -> (u, correct)`) so `uncertainty.calibration` never imports ground truth.
    - Acceptance: `.venv/bin/python -m pytest tests/uncertainty/test_calibration.py -q` passes (U-11, U-12), including the `insufficient_data` and `not_informative` branches.
    - _Requirements: 11.4_

  - [~] 11.3 Verify matched-control equivalence
    - `tests/uncertainty/test_matched_control.py` (U-10): the control equals the base proposal except `policy_id`/`uncertainty`; non-firing thresholds equal the control; identical random numbers.
    - Acceptance: `.venv/bin/python -m pytest tests/uncertainty -q` passes with reported counts. Validates design Properties 20, 21.
    - _Requirements: 11.3_

- [ ] 12. Evaluation harness, dashboard bundle, and end-to-end smoke **[Implementation]**
  - Implement metrics from records (incl. origin ranking and seed aggregation), the matched comparisons, the dashboard-consumable bundles, and the single smoke command.
  - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [~] 12.1 Implement the evaluation package
    - `src/cyberswarm/evaluation/`: `outcome.py` (extend the thin slice with precision/recall where defined, false-positive rate, detection-to-containment, attack impact, compromised nodes, healthy disruption via `defense_availability`, false isolation, episode reward, `origin_rank_top1`/`origin_mrr` from `NodeScoreRecord` and evaluator-only truth), `metrics.py`, `stats.py` (mean, sample std, percentile bootstrap), `compare.py` (`AggregateMetrics` over training seeds; enumerated comparison pairs), `splits.py` (`assert_disjoint` incl. topology-id disjointness for `eval_utopo`), `reweight.py`.
    - Acceptance: `.venv/bin/python -m pytest tests/evaluation -q` passes (metrics-from-records, outcome, splits disjoint, matched sets and stats, reweight identity). Validates design Properties 9, 13, 22.
    - _Requirements: 12.1, 12.2, 12.3_

  - [~] 12.2 Implement the runner and the smoke command
    - `src/cyberswarm/runner/`: `config.py`, `factory.py` (simulator/coordinator/policy factories; only `runner.factory` imports the mock), extend `evaluate.py` (`evaluate_policies` over matched sorted pairs, `eval` and `eval_utopo` reported separately, failed runs kept), `export.py` (`RunWriter`, `run_manifest.json` with execution-derived status), `smoke.py`. `configs/experiments/eval_default.toml`, `reward_sensitivity.toml`, `smoke.toml` (relative `output_dir = "runs"`).
    - Acceptance: `.venv/bin/python -m cyberswarm.runner.smoke --config configs/experiments/smoke.toml` exits 0, prints evaluator-computed metrics and `MOCK SIMULATOR RUN - NOT A RESEARCH RESULT`, and the status block.
    - _Requirements: 12.1, 12.2, 12.4_

  - [~] 12.3 Build the derived fixtures and dashboard bundles
    - Derived record fixtures (`EpisodeResult.demo`, `EvaluationMetrics.demo`, `ComparisonRecord.demo`, `RunManifest.demo`) and `fixtures/bundle_demo/`, `bundle_empty/`, `bundle_partial/`, `bundle_malformed/` via `scripts/build_fixtures.py`; values are null where not computed; `graph_model_outputs` is `not_computed`.
    - Acceptance: `.venv/bin/python scripts/build_fixtures.py --check` reports zero drift.
    - _Requirements: 12.4_

  - [~] 12.4 Write and run the integration tests and the full suite
    - `tests/integration/` (smoke end-to-end, bundle schema consumption for all bundles, leakage scan of agent-visible outputs, two-run byte-identical determinism).
    - Acceptance: `.venv/bin/python -m pytest -q` passes the full suite with reported pass/fail/skip counts; every emitted file validates against `schemas/`. Validates design Properties 5, 22, 23.
    - _Requirements: 12.1, 12.2, 12.4_

- [ ] 13. Staged learned-graph go/no-go **[Implementation]**
  - Only after tasks 6-12 are stable, decide and (if the criteria hold) implement plain-PyTorch graph models for node-suspicion scoring and the representation ablation; otherwise record an evidenced no-go.
  - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [~] 13.1 Build the dataset and the go/no-go gate
    - `src/cyberswarm/gnn/dataset.py` (offline `(GraphBatch, compromised)` pairs from `train`-split episodes only; the only `gnn` module permitted to import `evaluator_only`) and `scripts/gnn_go_nogo.py` evaluating criteria G1-G5 (per-scenario label balance, CPU timing, split disjointness, model smoke, matched-reference presence).
    - Acceptance: `.venv/bin/python scripts/gnn_go_nogo.py` runs and prints each criterion with its measured value and the resulting decision.
    - _Requirements: 13.1, 13.4_

  - [~] 13.2 Implement the plain-PyTorch graph models
    - `src/cyberswarm/gnn/`: `SageDense` (mean aggregation) and `GatDense` (masked dense attention) over the existing `GraphBatch` with dense adjacency, a node-suspicion head, checkpoint save/load to a gitignored dir, inference emitting `GraphModelOutput` and `NodeScoreRecord`; optional frozen embeddings as the `learned_embedding` observation mode. No PyTorch Geometric.
    - Acceptance: forward and backward passes run on CPU on a real `GraphBatch`; output shapes match the node set.
    - _Requirements: 13.2_

  - [~] 13.3 Run the graph-model tests, the ablation, and record the decision
    - `tests/gnn/` (shapes, mask handling, forward/backward, checkpoint round-trip, inference, leakage); run the conventional-vs-graph-feature-vs-learned-embedding ablation on identical `eval` and `eval_utopo` pairs; write a `StageDecision` (implemented, or an evidenced no-go citing the failed criterion). The conventional-vs-graph-feature comparison is required regardless.
    - Acceptance: `.venv/bin/python -m pytest tests/gnn -q` passes (if go); the `StageDecision` records implemented or an evidenced no-go; "no measurable improvement" is reported as a result, not a no-go. Validates design Property 25.
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

- [ ] 14. Staged MAPPO **[Implementation]**
  - After tasks 10-13 are stable, implement MAPPO (centralized critic over concatenated agent-visible observations, decentralized actors, shared team reward) with an equal-budget comparison; defer only with an evidenced blocker.
  - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [~] 14.1 Implement MAPPO and its training path
    - `src/cyberswarm/marl/mappo.py` (central team critic on `env.state()`, `MAPPOPolicy` loading actors only, shared team-reward GAE; reuses `marl/ppo.py` and `marl/gae.py`), `configs/experiments/mappo_default.toml`, `ippo_compare.toml`, `mappo_compare.toml`, and `runner/train.py` with `--algorithm {ippo,mappo}`. Never relabel IPPO as MAPPO.
    - Acceptance: `.venv/bin/python -m cyberswarm.runner.train --algorithm mappo --config configs/experiments/mappo_compare.toml --out runs/mappo-smoke` trains at the smoke budget, saves, and loads.
    - _Requirements: 14.1, 14.2_

  - [~] 14.2 Run the MAPPO tests, the comparison, and record the decision
    - MAPPO tests M-1..M-13 (critic input dimension and content, actor-own-observation input, team-reward advantage, IPPO-vs-MAPPO critic difference, checkpoint round-trip, contract-valid inference accepted by the coordinator, status-equals-execution); the equal-budget multi-seed comparison with IPPO on `eval` and `eval_utopo` with failed seeds retained; write a `StageDecision` (implemented, or `deferred` with command-output evidence).
    - Acceptance: `.venv/bin/python -m pytest tests/marl -q` passes M-1..M-13 (if implemented); the manifest shows `mappo: implemented` iff MAPPO episodes ran, else `deferred` with an evidenced reason. Validates design Properties 18, 19, 24.
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

- [ ] 15. Documentation reconciliation, teammate specs, prompts, and final validation **[Documentation]**
  - Reconcile all remaining documentation to the actual code and status, write the teammate specs and AI prompts, and run the documentation validator and final verification.
  - _Requirements: 16.1, 16.2, 16.3, 16.4_

  - [~] 15.1 Reconcile the project documentation
    - Rewrite `README.md` around the MARL direction with honest status; reconcile `docs/INTEGRATION.md`, `docs/TESTING.md`, `docs/EVALUATION.md`, `docs/DEMO.md`, `docs/TEAM_WORKFLOW.md` to the actual code, paths, and commands; update `ARCHITECTURE.md`, `CONTRACTS.md`, `RL_IMPLEMENTATION.md`, `DECISIONS.md` statuses to match what ran.
    - Acceptance: every quickstart command in `README.md` runs as written from the repo root; status words agree with the manifest status derived from execution.
    - _Requirements: 16.3, 16.4_

  - [~] 15.2 Write the teammate specs and AI prompts
    - `docs/MODULE_RAKSHA_SIMULATOR.md`, `docs/MODULE_MAYUKH_DASHBOARD.md` (specs plus fixtures only, no implementation), `docs/AI_PROMPT_RAKSHA.md`, `docs/AI_PROMPT_MAYUKH.md`, `docs/AI_PROMPT_PROJECT_LEAD.md`, `docs/AI_INSTRUCTIONS.md`, `AGENTS.md` (and `CLAUDE.md` only if non-conflicting). The project-lead prompt preserves the DELTA mandatory order.
    - Acceptance: the two `MODULE_*` docs describe only specifications and fixtures; no `src/cyberswarm/simulator/` or `dashboard/` is created.
    - _Requirements: 16.1, 16.2_

  - [~] 15.3 Run the documentation validator and final verification
    - `scripts/validate_docs.py` and `tests/docs/test_docs_validator.py`; then the final verification set.
    - Acceptance: `.venv/bin/python scripts/validate_docs.py` reports all relative links/paths resolve, every `json contract=` example validates, and documented commands exist; `.venv/bin/python -m pytest -q` passes the full suite; `.venv/bin/python -m cyberswarm.runner.smoke --config configs/experiments/smoke.toml` exits 0; `git status --short` is reported without being changed.
    - _Requirements: 16.3, 16.4_

## Notes

- Tasks are ordered and dependency-aware and mirror the DELTA mandatory order. Task 2 is the early reconciliation the review requires; Tasks 13 and 14 are the staged GNN and MAPPO slots with their own go/no-go and acceptance evidence.
- **[Implementation]** tasks: 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14. **[Documentation]** tasks: 2, 15.
- Per [`BRIEF.md`](../../../.agents/tasks/cyberswarm-foundation/BRIEF.md), tests are mandatory acceptance evidence, so test sub-tasks are first-class (not marked optional); each lists the exact command and the design Property or check ids it covers.
- Every property test runs at least 100 iterations and is tagged **Feature: cyberswarm-foundation, Property N: ...** referencing [design.md](design.md).
- Each task's acceptance is drawn from [`DESIGN.md` section 11.3](../../../.agents/tasks/cyberswarm-foundation/DESIGN.md) "done when" criteria and section 12.1 verification commands; report observed pass/fail/skip counts for every command run.
- GNN and MAPPO stages may end in an evidenced no-go/deferral recorded as a `StageDecision`; "no measurable improvement" and an evidenced blocker are honest outcomes, not failures to hide.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "2.1"] },
    { "id": 2, "tasks": ["2.2", "3.1"] },
    { "id": 3, "tasks": ["3.2", "3.3", "3.4"] },
    { "id": 4, "tasks": ["3.5", "4.1", "5.1"] },
    { "id": 5, "tasks": ["4.2", "5.2", "6.1", "6.2"] },
    { "id": 6, "tasks": ["5.3", "6.3"] },
    { "id": 7, "tasks": ["6.4"] },
    { "id": 8, "tasks": ["7.1", "7.2"] },
    { "id": 9, "tasks": ["7.3"] },
    { "id": 10, "tasks": ["7.4", "8.1", "8.2"] },
    { "id": 11, "tasks": ["8.3", "9.1"] },
    { "id": 12, "tasks": ["9.2"] },
    { "id": 13, "tasks": ["10.1"] },
    { "id": 14, "tasks": ["10.2"] },
    { "id": 15, "tasks": ["10.3"] },
    { "id": 16, "tasks": ["10.4", "11.1"] },
    { "id": 17, "tasks": ["11.2"] },
    { "id": 18, "tasks": ["11.3", "12.1"] },
    { "id": 19, "tasks": ["12.2"] },
    { "id": 20, "tasks": ["12.3"] },
    { "id": 21, "tasks": ["12.4"] },
    { "id": 22, "tasks": ["13.1"] },
    { "id": 23, "tasks": ["13.2"] },
    { "id": 24, "tasks": ["13.3"] },
    { "id": 25, "tasks": ["14.1"] },
    { "id": 26, "tasks": ["14.2"] },
    { "id": 27, "tasks": ["15.1", "15.2"] },
    { "id": 28, "tasks": ["15.3"] }
  ]
}
```
