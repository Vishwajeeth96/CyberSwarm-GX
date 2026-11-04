# Design Document

## Overview

This design is deliberately short. The complete, authoritative design already exists in the repository and is the source of truth; this document links to it and records only the corrected scope and the properties that implementation must satisfy. Do not re-derive or duplicate the linked documents.

The feature is the **project-lead-owned foundation** of CyberSwarm, a multi-agent RL system for autonomous cyber defense in a controlled simulator. In scope: neutral contracts, a public simulator interface, a clearly-labelled lead-owned mock simulator, the observation-to-graph pipeline, the RL environment and reward, non-learned baselines, the safe defense coordinator, self-written IPPO, an uncertainty-aware wrapper, the evaluation harness and smoke path, a staged learned-graph go/no-go, a staged MAPPO implementation, and documentation. Out of scope: Raksha's real network simulator and Mayukh Aryan's dashboard UI (specifications and fixtures only).

## Authoritative design sources

All real detail (module APIs, schemas, mock dynamics, reward arithmetic, metric definitions, stage acceptance) lives here:

- [`.agents/tasks/cyberswarm-foundation/DELTA.md`](../../../.agents/tasks/cyberswarm-foundation/DELTA.md) — latest authority, Addenda 1-3.
- [`.agents/tasks/cyberswarm-foundation/BRIEF.md`](../../../.agents/tasks/cyberswarm-foundation/BRIEF.md) — canonical task.
- [`.agents/tasks/cyberswarm-foundation/DESIGN.md`](../../../.agents/tasks/cyberswarm-foundation/DESIGN.md) — file-by-file inventory (section 2), module APIs (section 4), mock scope (section 5), configs (section 6), smoke and bundles (section 7), fixtures (section 8), DELTA check register and test files (section 9), cross-check matrix (section 10), stages and acceptance (section 11), verification commands (section 12).
- [`docs/ARCHITECTURE.md`](../../../docs/ARCHITECTURE.md), [`docs/CONTRACTS.md`](../../../docs/CONTRACTS.md), [`docs/RL_IMPLEMENTATION.md`](../../../docs/RL_IMPLEMENTATION.md) (v0.2), [`docs/DECISIONS.md`](../../../docs/DECISIONS.md), [`docs/RESEARCH.md`](../../../docs/RESEARCH.md).
- [`.agents/tasks/cyberswarm-foundation/gates/design-review.md`](../../../.agents/tasks/cyberswarm-foundation/gates/design-review.md) — blocking review (4 HIGH, 17 MEDIUM, 11 NIT).

## Architecture

A modular monolith in Python 3.12, `src/` layout, pydantic v2 contracts, JSON/JSON-Lines on disk, NetworkX for topology and graph processing, Gymnasium/PettingZoo for the RL adapter, and self-written PyTorch for PPO. No microservices, databases, brokers, or chat agents. Data flow (see [`docs/ARCHITECTURE.md`](../../../docs/ARCHITECTURE.md)):

`scenario -> simulator -> agent-visible observations/events -> validated graph snapshot + features + masks -> conventional/graph observation views -> detection/evidence -> policy proposal -> defense coordinator (validate, allowlist) -> validated simulated action -> updated simulator state -> outcome events -> evaluator -> dashboard-consumable records`.

Detection, investigation, and decision are logical software roles: deterministic components (`roles/`) feed one decision policy; coordination is an independently testable safety boundary. Ground truth is evaluator-only and never enters any decision-time or agent-visible path.

## Components and Interfaces

The single source of truth for paths and owners is [`DESIGN.md` section 2](../../../.agents/tasks/cyberswarm-foundation/DESIGN.md). Summary of lead-owned packages under `src/cyberswarm/`: `contracts`, `common`, `interfaces`, `mock`, `roles`, `graph`, `engine`, `rl`, `baselines`, `coordinator`, `evaluation`, `runner`, `marl`, `uncertainty`, and the staged learned-graph package. Neutral (neither teammate edits): `contracts/`, `interfaces/`, `schemas/`, `fixtures/`, `configs/topologies/`. Not created in this phase: `src/cyberswarm/simulator/` (Raksha) and `dashboard/` (Mayukh Aryan) — no placeholders.

Import boundaries, the evaluator-only import allowlist, and the coordinator source ban are specified in [`DESIGN.md` section 3.2](../../../.agents/tasks/cyberswarm-foundation/DESIGN.md) and enforced by `tests/integration/test_import_boundaries.py` and `tests/coordinator/test_source_guard.py`.

## Data Models

Record models, identifiers, action vocabulary, step semantics, and the run-directory layout are defined in [`docs/CONTRACTS.md`](../../../docs/CONTRACTS.md) and the API signatures in [`DESIGN.md` section 4](../../../.agents/tasks/cyberswarm-foundation/DESIGN.md). Schemas are exported, never hand-edited. Fixtures are synthetic and labelled as such.

## Mandatory build order

The order is fixed by the DELTA and mirrored by [tasks.md](tasks.md): scaffold; design reconciliation; contracts and fixtures; interface and conformance; mock; graph pipeline; RL environment and reward; baselines; coordinator; IPPO (with a thin evaluation slice); uncertainty; evaluation harness, bundles, and smoke; staged learned-graph go/no-go; staged MAPPO; documentation and teammate handoff. Each stage is independently testable with the acceptance commands drawn from [`DESIGN.md` section 11.3](../../../.agents/tasks/cyberswarm-foundation/DESIGN.md).

## Corrected scope (resolved review findings)

The older [`DESIGN.md`](../../../.agents/tasks/cyberswarm-foundation/DESIGN.md) and [`docs/ARCHITECTURE.md`](../../../docs/ARCHITECTURE.md) pre-date the blocking review. Where they disagree with [`docs/RL_IMPLEMENTATION.md`](../../../docs/RL_IMPLEMENTATION.md) v0.2 and the DELTA addenda, the corrected scope below wins and is encoded in [tasks.md](tasks.md). An early task reconciles `ARCHITECTURE.md` and `DECISIONS.md` to this scope before implementation proceeds.

HIGH findings:

- **Learned GNN is a staged decision, not a design-time no-go.** The go/no-go is taken in its own stage after IPPO, uncertainty, and the evaluation path are stable, from criteria G1-G5. Plain-PyTorch `SageDense` and `GatDense` over the existing graph batch; no PyTorch Geometric. "No measurable improvement over the graph-feature baseline" is a valid reported result, not a no-go.
- **MAPPO is implemented in its own stage** after IPPO/uncertainty/evaluation/learned-graph are stable, with decentralized actors and one centralized critic over concatenated agent-visible observations (the `env.state()` hook), shared team reward, and tests M-1 through M-13. Deferral requires a concrete, evidenced blocker. IPPO is never relabelled as MAPPO. Manifest status is computed from execution, not hard-coded; the `src/` `mappo` token ban and the "no mappo.py" assertion are removed.
- **Unseen-topology evaluation exists.** Add `configs/topologies/topo-mock-small-v2.json` (same node ids and zones, different edges and criticalities) and an `eval_utopo` split; report `eval` and `eval_utopo` separately; checkpoints load across topologies by node-id list and action-table digest, not topology id.
- **Origin-ranking metric exists.** Add a `NodeScoreRecord` and compute `origin_rank_top1` and `origin_mrr` per scorer (detection role, graph-feature baseline, learned models). Label it a ranking of predicted suspicion, never a causal root-cause finding.

Key MEDIUM findings encoded in tasks:

- Stage order follows the DELTA mandatory order (IPPO before the full evaluation harness, with a thin evaluation slice inside the IPPO stage); the graph pipeline and the coordinator are separated from the RL environment stage.
- `ground_truth()` timing is fixed: it returns the end-of-step-`t` snapshot captured before `end_step()` runs P0 of `t+1` (conformance check SC-14).
- Healthy-disruption separates defense-induced from attack-induced loss via `GroundTruthSnapshot.defense_availability` (compromise factor fixed at 1.0); attack-induced loss is charged only to attack impact.
- The `service_health` compromise proxy is weakened (compromised factor 0.9, noise 0.08, edge offset 0.05) with a mock AUROC guard test, so the channel is informative but not an oracle.
- An event-visibility table defines who sees each event type; `action_applied` and `investigation_result` reach only the acting agent (conformance check SC-15).
- Recovery verification defines `recovery_verified` and `recovery_failed` from agent-visible data, tracking executed restores only.
- Coordinator reason codes are evaluated in a defined dependency order with a defined id-assignment order.
- The rule baseline adds a rule so a contained compromised node with expired or middling evidence is re-investigated (so episodes can reach `incident_resolved`).
- `ShortText` drops `/ = + :`, rejects dotted-quads, `..`, and a leading `-`; invalid fixtures for IP-in-rationale and path-in-rationale are added.
- All repository configs and documented commands are repository-relative; absolute paths stay only in step prompts.
- A single evaluator-only import list; `evaluation.calibration_labels` supplies correctness arrays so `uncertainty.calibration` never imports ground truth.
- The IPPO training CLI lives in `runner/`, not `marl/`, to respect import boundaries.
- Aggregation over training seeds (an `AggregateMetrics` record with `policy_family_id`/`training_seed`), an enumerated comparison list, failed runs retained, and no tuning on `eval`/`eval_utopo`.

## Error handling

Contract violations raise `ContractError`; simulator misuse raises `SimulatorError` with an `ErrorCode`; rejected or inapplicable actions are normal `failed` results with structured reason codes, never exceptions; checkpoint mismatches raise `CheckpointError`; missing policy proposals become a `no_op` with an `ErrorRecord`. Unavailable optional values (uncertainty, graph-model output) are `null`, never fabricated. See [`docs/CONTRACTS.md`](../../../docs/CONTRACTS.md) and [`DESIGN.md` section 4](../../../.agents/tasks/cyberswarm-foundation/DESIGN.md).

## Testing strategy

Dual approach. **Property-based tests** verify universal properties across generated inputs (minimum 100 iterations each): graph conversion determinism and round-trip, mask/imputation non-leaking, reward linearity, action-mask invariants, coordinator rejection invariance, checkpoint round-trip, IPPO independence, matched-control equivalence, and byte-identical determinism for fixed seeds. **Example and edge-case unit tests** cover the hand-constructed reward cases, specific reason codes, the demo storyboard, and boundary conditions. Integration and smoke tests exercise the end-to-end path and schema consumption. The test-file-to-check mapping is [`DESIGN.md` section 9](../../../.agents/tasks/cyberswarm-foundation/DESIGN.md); each test names its DELTA check ids, uses the mock only through fixtures, and never touches the network. Every property test is tagged **Feature: cyberswarm-foundation, Property N: ...**.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of the system — a formal statement about what the system should do. Properties bridge the human-readable requirements and machine-verifiable tests. Each property below is universally quantified and annotated with the requirements it validates. Property tests run a minimum of 100 iterations and are tagged* **Feature: cyberswarm-foundation, Property N: ...**. *Redundant candidates were merged during property reflection (for example, the three leakage checks and the two state-invariance checks).*

### Property 1: Schema export round-trip

For all registered record types, exporting the JSON Schema produces bytes identical to the committed `schemas/<Name>.schema.json`, and every committed schema re-imports and validates its own valid fixtures.

**Validates: Requirements 1.1, 1.2**

### Property 2: Malformed records are rejected

For any record constructed with a non-finite number, an unknown enum value, an extra field, or an identifier resembling an IP address or filesystem path, construction raises `ContractError`.

**Validates: Requirements 1.3**

### Property 3: Unavailable optional values stay null

For any record whose optional uncertainty, graph-model, reward, or comparison fields are unset, serializing then deserializing preserves `null` (never `0.0`, and never a silently filled default).

**Validates: Requirements 1.4**

### Property 4: Fixture integrity

For all valid fixtures, both the pydantic model and the exported JSON Schema accept the fixture; and for all derived fixtures, regeneration via `scripts/build_fixtures.py --check` produces bytes identical to the committed files.

**Validates: Requirements 2.1, 2.2, 2.4**

### Property 5: No evaluator-only leakage

For any agent-visible record, graph snapshot, graph-feature record, RL observation, or info dictionary, a scan of its serialized form finds no `GroundTruthSnapshot` field name and no event whose step exceeds the record's step.

**Validates: Requirements 2.3, 5.3, 6.4**

### Property 6: Ground-truth snapshot timing

For any episode and step `t`, after `end_step()` returns, `ground_truth()` returns the end-of-step-`t` snapshot captured before phase P0 of step `t+1` runs, with `snapshot.step == StepResult.step_closed`, until the next `end_step()`.

**Validates: Requirements 3.3**

### Property 7: Mock determinism

For any scenario, seed, and scripted action sequence, two `Mock_Simulator` instances produce identical observation, event, and `state_digest` sequences; and a different seed changes at least the `state_digest` sequence.

**Validates: Requirements 4.2**

### Property 8: Rejected and inapplicable actions preserve state

For any simulator state and any action that is out of authority, inapplicable, or rejected by the Defense_Coordinator, the action produces a `failed`/rejected result and `state_digest()` is unchanged before and after processing.

**Validates: Requirements 4.3, 9.2**

### Property 9: Split disjointness including the unseen topology

For the configured `train`, `cal`, `eval`, and `eval_utopo` splits, the scenario identifiers, seeds, and topology identifiers are pairwise disjoint, and the unseen topology never appears in `train` or `cal`.

**Validates: Requirements 4.4, 12.2, 12.3**

### Property 10: Graph conversion is deterministic and well-formed

For any agent-visible observation, repeated conversion yields an identical `GraphSnapshot`; node and edge ordering is stable across steps of an episode; every mask aligns with its feature array; imputed entries equal `0.0` while `criticality` is preserved and `observed = 0`; and the derived tensors and batches have the documented shapes and dtypes.

**Validates: Requirements 5.1, 5.4**

### Property 11: Observation well-formedness

For any step and any observation mode, the observation vector has the documented dimension (`17N+2` for conventional, `25N+2` for graph features) and every value is finite and within [0, 1].

**Validates: Requirements 6.2**

### Property 12: Action-mask invariants and seed reproducibility

For any step, `no_op` is always unmasked; any in-range but masked action index is substituted with `no_op` and flagged `invalid_action`, and the masked proposal never reaches the coordinator; and for a fixed integer seed the environment trajectory is reproducible.

**Validates: Requirements 6.3**

### Property 13: Reward is linear in the weights

For any stored per-term magnitudes and any weight vector, `evaluation.reweight` recomputes the episode reward equal to the direct sum of signed weighted magnitudes, and equals the stored reward at the default weights.

**Validates: Requirements 7.1, 7.4**

### Property 14: Healthy disruption measures defender-induced loss only

For any ground-truth state, the `healthy_disruption` term and the `healthy_disruption_cost` metric are computed from defense availability (compromise factor fixed at 1.0), so availability loss caused solely by an uncontained attack contributes zero to them.

**Validates: Requirements 7.2**

### Property 15: Reward ordering of containment decisions

For any pair of healthy nodes differing only in criticality, executing a containment-class action on the higher-criticality node yields a strictly lower (more negative) reward; and for any compromised uncontained node, justified containment yields a strictly higher reward than inaction on the same state.

**Validates: Requirements 7.3**

### Property 16: Baselines are valid and respect their inputs

For any decision context, each baseline emits a contract-valid proposal whose action is present in `AvailableActions`; the conventional baseline's proposal is invariant to perturbation of graph features; and the graph-feature baseline raises a policy error when graph features are absent.

**Validates: Requirements 8.1, 8.2**

### Property 17: IPPO agents are independent

For the two IPPO agents, the actor, critic, optimizer, and value-normalizer parameter sets are disjoint objects; the gradient of one agent's loss with respect to the other agent's parameters is `None`; and an update on one agent leaves the other agent's parameters bitwise unchanged.

**Validates: Requirements 10.1**

### Property 18: Checkpoint round-trip identity

For any trained policy state (IPPO or MAPPO), saving then loading yields identical action logits; and a corrupted file, a wrong content hash, or a mismatched action table is refused with `CheckpointError`.

**Validates: Requirements 10.2, 14.3**

### Property 19: Learned-policy inference is contract-valid and accepted

For any decision context, learned-policy inference (IPPO or MAPPO) returns a Contract_Valid `PolicyProposal` that the Defense_Coordinator accepts for unmasked actions.

**Validates: Requirements 10.3, 14.3**

### Property 20: Entropy signal is bounded with explicit unavailability

For any masked action distribution over `K >= 1` valid actions, the normalized entropy `u` lies in [0, 1], equals `0.0` for a one-hot or single valid action and `1.0` for a uniform distribution; and for non-finite logits the signal is `null` with an `ErrorRecord` and a pass-through outcome.

**Validates: Requirements 11.1**

### Property 21: Matched-control equivalence

For any decision context, the matched control's proposal equals the base policy's proposal in every field except `policy_id` and `uncertainty`; the uncertainty-aware policy with non-firing thresholds equals the control outside those fields; and both consume identical random numbers.

**Validates: Requirements 11.3**

### Property 22: Metrics are derived from records

For any set of execution records, the evaluator's metrics equal the values derived from those records and change when a contributing record changes; and origin-ranking top-1 accuracy and mean reciprocal rank are computed per scorer from evaluator-only ground truth only.

**Validates: Requirements 12.1**

### Property 23: Full-run determinism

For any fixed set of seeds, two complete runs produce byte-identical record files (the wall-clock log excluded).

**Validates: Requirements 4.2, 12.4**

### Property 24: MAPPO uses a global critic, local actors, and the team reward

For the MAPPO configuration, the centralized critic input dimension equals the sum of the agents' observation dimensions and contains only agent-visible fields; at execution each actor receives only its own observation vector; and both actors receive the identical advantage vector derived from the shared team reward.

**Validates: Requirements 14.1, 14.2**

### Property 25: Learned-graph models operate on the real graph batch

For any `GraphBatch`, the plain-PyTorch `SageDense` and `GatDense` models produce the documented node-score output shape, handle the node masks, and complete a backward pass on CPU.

**Validates: Requirements 13.2**
