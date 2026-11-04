# LATEST DELTA — PROJECT-LEAD SCOPE APPROVED

> This latest delta supersedes the earlier sentence that declared graph representation optional. It removes that conflict with `BRIEF.md`: multi-agent reinforcement learning (MARL) remains CyberSwarm's primary direction, and the project-lead-owned graph representation, observation-to-graph conversion, and hand-engineered graph-feature pipeline are REQUIRED foundations that must be implemented before the RL environment. Learned GNN models remain a staged experiment after reliable IPPO and may proceed only when the task, supervision, dependencies, compute budget, and evaluation plan justify implementation.

## Authority and approved execution scope

The canonical full task remains `/Users/guru/CyberSwarm-GX/.agents/tasks/cyberswarm-foundation/BRIEF.md`. Every remaining agent must read `BRIEF.md` and this delta. This delta records the user's latest approval and clarifies the immediate execution boundary without weakening the canonical requirements.

The user approved immediate execution of the **project lead's scope only**. “Only my part” means:

- Implement the lead-owned architecture, neutral shared foundations, graph intelligence, RL, coordination, uncertainty, evaluation, integration smoke path, and related reconciliation described below.
- Do **not** implement Raksha's real network simulator or attack/scenario engine.
- Do **not** implement Mayukh Aryan's dashboard UI.
- Retain only the neutral shared contracts, public interfaces, validating fixtures, conformance expectations, and dashboard-consumable records needed for the project lead's implementation to run independently and integrate later.
- A small, deterministic, clearly labeled **lead-owned MOCK simulator** is required. It must remain outside Raksha's module, implement only the public simulator interface, and must never be presented as the real simulator.
- Specs and integration guidance may describe the teammate boundaries required for compatibility, but no teammate-owned production implementation is in the approved immediate coding scope.

## Confirmed project direction

CyberSwarm is primarily **Self-Learning Autonomous Cyber Defense using multi-agent reinforcement learning** inside a controlled simulator. The principal comparison remains rule-based defense, IPPO, uncertainty-aware defense, and MAPPO only if correctly feasible. The central question concerns simulated incident containment while limiting disruption to healthy services.

Graph-based network intelligence is required lead-owned infrastructure, not the central research claim. The implementation must provide stable topology and dynamic graph state, deterministic graph conversion, missing-data handling, non-leaking hand-engineered graph features, and graph observations for controlled comparison with conventional observations. NetworkX is the default topology and graph-processing foundation unless the design justifies another minimal dependency.

A learned GNN is a secondary, staged hypothesis. GraphSAGE, GAT, or another learned graph model may be implemented only after the conventional and graph-feature paths and IPPO are stable and tested. A documented no-go decision satisfies the GNN stage when supervision, a concrete evaluable task, dependency compatibility, compute budget, or expected experimental value is inadequate. Do not claim novelty, causality, superiority, or results that were not produced by matched experiments.

All attacks, incidents, actions, containment, isolation, restoration, and recovery remain abstract simulator state transitions. No real-network scanning, exploitation, malware, credential access, firewall changes, device isolation, arbitrary commands, executable payloads, sockets, or real containment behavior may be implemented.

## Required project-lead implementation

The immediate work must remain real implementation with automated verification, not documentation-only planning.

### Neutral shared contracts and fixtures

Implement lead-owned, versioned, typed contracts in a neutral package and machine-validate them. Preserve the schema coverage required by `BRIEF.md`, including topology nodes and edges, dynamic graph snapshots and features, scenarios, events, agent observations, available actions, policy proposals, coordinator decisions, executed simulated actions, episode outcomes, evaluator metrics, optional graph-model outputs, and structured errors/log fields.

Use one documented serialization approach, stable identifiers, deterministic node ordering, exact simulation-step semantics, an explicit compatibility policy, and a single abstract simulator-only action vocabulary. Optional uncertainty and learned-model values must remain null or absent when no producer computed them. Evaluator-only ground truth and future events must never enter observations, graph conversion, graph features, model outputs, actor inputs, serialized information dictionaries, or decision-time data.

Provide neutral synthetic fixtures for the interfaces used by the lead-owned implementation. Fixtures must include enough valid records to exercise healthy and changing graph state, masks and imputation, conventional and graph observations, available actions, accepted and rejected coordinator paths, episode outcomes, evaluator metrics, and dashboard consumption. Clearly label fixture values as synthetic and illustrative rather than research results. Contract tests must validate all fixtures and cross-record consistency.

### Deterministic lead-owned mock simulator

Implement the public simulator Protocol/ABC and a small lead-owned MOCK simulator outside Raksha's future simulator directory. The mock must:

- use stable node IDs and seeded deterministic behavior;
- expose only agent-visible observations through the agent-facing interface;
- keep evaluator-only ground truth behind a separate accessor;
- support enough abstract state transitions and allowlisted simulated actions for RL, coordinator, uncertainty, evaluation, and end-to-end smoke tests;
- avoid importing or anticipating private internals of Raksha's future implementation; and
- pass the neutral simulator conformance checks needed by the project lead.

Do not implement or describe the mock as Raksha's real simulator.

### Required graph pipeline

Implement and test the observation-to-graph pipeline before the RL environment. The graph foundation must:

- represent static topology plus current dynamic node and edge state;
- preserve stable node IDs and deterministic node/tensor ordering across steps;
- produce contract-valid graph snapshots, feature-name metadata, node and edge feature tensors, provenance, and aligned missing-data masks;
- apply only the documented deterministic, non-leaking imputation policy;
- consume agent-visible observations and permitted current/past events only;
- exclude evaluator-only labels and future events;
- expose a model-compatible tensor/batch interface without requiring a learned GNN dependency; and
- compute inspectable, hand-engineered graph features such as degree/criticality context and neighbor-alert aggregation without hidden labels.

Tests must cover static topology, dynamic updates, deterministic conversion, stable ordering, tensor shapes/dtypes, masks and imputation, serialization, batch processing, ID resolution, and leakage prevention.

### RL environment and reward

Implement a stable multi-agent RL adapter over only the public simulator interface. Preserve conventional and graph observation modes for matched comparisons. Define and test observation/action spaces, action masks or explicit invalid-action handling, reset/step lifecycle, seed behavior, termination versus truncation, and role-specific visibility.

Implement configurable reward terms for containment, response delay, criticality-weighted attack impact, false isolation, healthy-service disruption, and investigation cost. Document signs, scales, default rationale, and a sensitivity-analysis plan. Tests must include hand-constructed cases proving that disruption of a healthy critical service costs more than disruption of a low-value service, justified containment receives the specified reward, delay increases penalty, and investigation has a small defined cost.

Detection, investigation, decision, and coordination remain logical software roles, not LLM-chat agents. The architecture must explicitly choose and implement whether detection/investigation/decision use deterministic components around one policy, separate policies, or shared parameters with role-specific heads. Coordination remains an independently testable safety boundary.

### Conventional and graph-feature baselines

Implement both:

- a functional rule-based baseline over conventional observations that investigates when evidence is insufficient, emits contract-valid proposals, and selects only permitted simulator actions; and
- a functional non-neural graph-feature baseline that consumes the required hand-engineered graph features without learned message passing.

Both baselines must be executable, tested, explicit about their observation inputs, and compatible with matched scenario/seed evaluation.

### Safe defense coordinator

Implement the coordinator as the sole action-validation and execution boundary. It must validate proposal schemas, resolve target IDs, enforce an explicit allowlist, reject malformed/unknown/disallowed actions with structured reason codes, preserve simulator state when a proposal is rejected, and execute accepted actions only through the public simulator API. It must produce contract-valid proposal, validation, execution, and outcome records and support recovery verification from simulator-reported data.

The coordinator must contain no shell commands, subprocess execution, arbitrary command fields, sockets/network calls, firewall operations, real-host operations, or action-triggered filesystem mutation. Add automated safety and unchanged-state-on-rejection tests.

### IPPO foundation

After the environment and both observation paths are stable, implement functional Independent PPO with correctly independent actors/policies or another clearly justified IPPO parameterization. Preserve seeded training, configurable policy/training settings, training logs, checkpoint save/load to gitignored storage, evaluation mode, and an inference interface that returns contract-valid proposals.

Run only a short mock-based training budget to prove train → save → load → inference and held-out evaluation execution. Do not present mock-trained metrics as scientific evidence or claim learned-policy quality.

### Meaningful uncertainty-aware extension

Implement a defined uncertainty signal, such as normalized policy entropy or a justified disagreement measure, rather than an unexplained confidence field. Specify and test:

- how uncertainty is calculated;
- how calibration or reliability is evaluated on held-out episodes;
- when low-evidence uncertainty requests investigation;
- when uncertainty defers a high-impact action;
- when evidence permits containment despite uncertainty;
- how unavailable uncertainty is represented; and
- a matched control policy that differs only in uncertainty-aware decision rules.

Tests must cover threshold boundaries, evidence requests, high-impact deferral, justified containment, null/unavailable uncertainty, and matched-control equivalence outside the uncertainty rule.

### Evaluator and dashboard-consumable output

Implement a reproducible evaluator and end-to-end smoke path covering mock scenario → observation → graph conversion → baseline or policy proposal → coordinator validation → simulated action → outcome → evaluator → dashboard-consumable records.

The evaluator must compute metrics from actual execution records rather than fixture constants. Preserve train/evaluation scenario separation, matched scenarios and seeds across approaches and representations, repeated-seed summaries with spread where practical, and the metric definitions in `BRIEF.md`. Dashboard-facing output must use the same neutral contracts without transformation or invented values and must preserve unavailable uncertainty, graph-model, and comparison values as null.

The future dashboard must be able to consume graph snapshots, hand-engineered graph features, optional model outputs, proposals, validation results, executions, outcomes, and evaluator metrics. This requirement covers interfaces, fixtures, validation, and generated records only; it does not authorize implementing Mayukh Aryan's dashboard UI.

### Staged learned-GNN decision

Consider learned-GNN implementation only after graph infrastructure, RL environment, baselines, coordinator, reliable IPPO execution, uncertainty behavior, and evaluation smoke tests are stable. Before implementation, document:

- the concrete node- or graph-level task;
- the supervision source and strict train/evaluation separation;
- the input/output tensor contracts;
- dependency and local-version compatibility;
- compute/resource budget;
- the matched evaluation and ablation plan; and
- GraphSAGE-versus-GAT feasibility against the topology size, dynamics, and baseline strength.

If justified, implement real model/training/checkpoint/inference code and tests as specified in `BRIEF.md`. If not justified, record a precise no-go decision. In either case, evaluator-only labels may be used only in explicitly separated training/evaluation data and must never leak into decision-time features. Learned anomaly scores, rankings, attention, and embeddings are predictive outputs, not causal explanations.

### Honest MAPPO implementation or deferral

Assess MAPPO only after the earlier stages. Implement MAPPO only if the result has decentralized actors, a genuine centralized critic over valid training state, correct advantage estimation, explicit parameter-sharing semantics, and no evaluator-only/future information in actor inputs. Never relabel IPPO as MAPPO.

If correct implementation is not feasible in this phase, provide the precise design, blockers, remaining work, API integration points, and acceptance tests required for later implementation. Code, tests, status tables, and output must consistently identify MAPPO as deferred.

## Mandatory implementation order

The execution order must match canonical `BRIEF.md`:

1. Neutral contracts and fixtures, public simulator interface, and lead-owned MOCK simulator.
2. Required graph conversion/feature pipeline and graph tests.
3. RL environment and configurable reward implementation/tests.
4. Conventional rule-based and non-neural graph-feature baselines.
5. Safe defense coordinator and safety tests.
6. IPPO train/save/load/infer foundation and mock smoke evidence.
7. Meaningful uncertainty-aware extension and matched control.
8. Evaluator, held-out comparison hooks, dashboard-consumable records, and end-to-end smoke path.
9. Learned GNN experiment only if justified after stable IPPO; otherwise document the no-go decision.
10. MAPPO only if correct; otherwise document honest deferral.
11. Documentation reconciliation so README, architecture, contracts, RL, testing, evaluation, integration, decisions, demo, ownership guidance, and prompts match actual code and status.
12. Independent final whole-repository consistency and safety review, followed by fixes and re-review when required.

Preserve the existing workflow discipline: complete and cross-check design, obtain an independent blocking design review, create a detailed plan, use coder-then-reviewer implementation loops with the reviewer last and approval-gated, run final verification/fix loops, and finish with an independent semantic review. Work directly in `/Users/guru/CyberSwarm-GX`; do not create a git worktree.

## Verification and evidence

Perform the actual verification required by `BRIEF.md` for the approved lead-owned scope. This includes a repo-local gitignored virtual environment, minimal compatible dependencies, the full automated test suite, focused graph evidence, end-to-end smoke output, brief IPPO train/save/load/infer and held-out evaluation, uncertainty/coordinator checks, evaluator-derived metrics, deterministic fixed-seed behavior, leakage prevention, fixture/schema validation, documentation path/link/command checks, and honest GNN/MAPPO status.

Report exact commands and observed pass/fail/skip counts. Label mock and fixture evidence as such. Do not claim completion, quality, research findings, comparisons, or passing checks that were not actually implemented and run.

## Git, preservation, and safety

- Preserve existing and unrelated work. Read and reconcile pre-existing files before editing them.
- Do not implement Raksha's real simulator or Mayukh Aryan's dashboard.
- Do not commit or push.
- Do not change remotes, branches, remote tracking, collaborators, repository permissions, or git identity.
- The previously guessed git email must not be reused; the user must verify identity before any future commit.
- Keep virtual environments, checkpoints, datasets, logs, generated runs, large artifacts, `node_modules`, `.env`, and secrets out of Git.
- Do not add secrets or network-enabled runtime behavior.
- Do not add real offensive or containment capability. All scenarios and actions must remain abstract and simulator-only.
- Final reporting must include the working-tree status without changing it.

## Final reporting emphasis

The final report must distinguish clearly among implemented-and-tested lead-owned work, synthetic mock/fixture evidence, teammate-owned work that remains unimplemented, staged or rejected GNN work, and implemented or deferred MAPPO work. It must confirm that MARL is primary, required project-lead-owned graph infrastructure was implemented before RL, and learned-GNN performance remains a secondary hypothesis. It must not imply that Raksha's real simulator or Mayukh Aryan's dashboard exists merely because shared interfaces, fixtures, records, or specifications exist.

## Addendum: stale graph-scope entries in docs/DECISIONS.md v0.2

`docs/DECISIONS.md` v0.2 was written by the workflow's research-delta step while an earlier, superseded version of this DELTA said graph representation was optional. Where v0.2 follows that earlier wording, it contradicts this DELTA and `BRIEF.md`. This DELTA and `BRIEF.md` govern. This addendum does not edit `docs/DECISIONS.md`, because a running workflow step may be editing it.

### Stale content, superseded by this DELTA

- AD-01 items 3 to 5 and its "Effect on the technology table" row, where they say the graph-snapshot and graph-feature pipeline, the graph-feature baseline, graph schemas beyond node and edge topology, and the staged GNN experiment are "no longer required".
- AD-01 status text, wherever it implies the user confirmed removing the graph pipeline. The user confirmed the MARL pivot, then explicitly REQUIRED project-lead-owned graph representation.
- D-09: NetworkX as "optional support only".
- D-17: the exclusion of `networkx` from the core dependency budget.
- D-18: the graph-feature pipeline and baseline as "optional and future".
- D-19: GraphSAGE and GAT as "no-go for this phase".

### Required corrections, made by wf-design

The design step (wf-design) makes these corrections during its normal `docs/DECISIONS.md` update:

- AD-01 keeps MARL primary and the GNN / Faultline root-cause idea non-central. It states that the observation-to-graph pipeline, the graph schemas, and the non-neural graph-feature baseline are REQUIRED and are built before the RL environment. The graph schemas cover the snapshot, node and edge features, missing-data masks, graph-derived features, and optional model outputs that are null when no model ran. The "Effect on the technology table" row says the same.
- D-09 and D-17: NetworkX becomes a core dependency, unless the design justifies a smaller alternative.
- D-18 becomes Required.
- D-19 becomes "Pending: decided at stage 9, after IPPO, uncertainty and evaluation are stable".
- The verified research facts are recorded as inputs to the stage-9 decision: PyG 2.8.1 resolves on Python 3.13.2 and pulls in `aiohttp` and `requests`.
- A documented no-go remains an acceptable outcome at stage 9.
- PyTorch Geometric stays out of the core dependencies until stage 9 justifies it.

### Review gate

The design-review step must not return APPROVED while the design omits the required graph pipeline, or while any of the `docs/DECISIONS.md` entries listed under "Stale content" above remains stale or contradictory.

### Environment guard for the venv step

At 19:58 IST on 2026-10-30, free disk space was 633 MiB. That is below the roughly 0.8 to 1 GB peak that a PyTorch install needs. The venv step must follow these rules:

- Measure free space with `df -h /Users/guru` before creating `.venv` and again before installing torch.
- Install with `pip install --no-cache-dir`.
- If free space is below 1.5 GB, do not attempt the torch install. Stop the venv step and report the blocker with the measured number.
- Never delete files outside the repository to make space.

## Addendum 2: design-time deferrals and missing evaluation requirements

The design working notes (`design-notes.md`, written 20:16 IST on 2026-10-30) pre-decide two items that the user asked to be decided by evidence at a later stage, and they omit evaluation requirements the user stated explicitly. The following are the user's requirements and govern the design.

1. **Learned GNN: no design-time no-go.** The go/no-go is taken at the GNN stage (stage 9 of the mandatory implementation order), after IPPO, the uncertainty extension and the evaluation smoke path are stable. The design must specify now:

   - the candidate task: node-level suspicion scoring trained only on training-split ground-truth labels, and/or node embeddings used as an IPPO observation encoder;
   - the architectures compared: GraphSAGE-style mean aggregation versus GAT (a plain-PyTorch dense-adjacency implementation is acceptable, and PyTorch Geometric is not required, consistent with D-10);
   - the inputs: the existing graph tensor/batch interface;
   - explicit go/no-go criteria; and
   - the matched ablation: conventional observations versus hand-engineered graph features versus learned embeddings (if built), on identical held-out scenarios and seeds.

   D-10 and the research already establish that dependencies and compute are not blockers, so a no-go must cite a specific, evidenced reason. "No measurable improvement over the graph-feature baseline" is a valid result and must be reported as such. The conventional-versus-hand-engineered-graph-feature ablation is required whatever the GNN decision.

2. **MAPPO: no design-time deferral.** The design already specifies the centralized critic (concatenated agent-visible observations, no ground truth, shared team reward), IPPO is self-written PyTorch, and there are only two agents, so MAPPO appears feasible. Plan MAPPO as an implementation stage after IPPO is stable (stage 10 of the mandatory implementation order). Defer MAPPO only if that stage finds a concrete blocker, documented with evidence. If MAPPO is implemented, these tests are required:

   - at execution, each actor receives only its own local observation;
   - the critic input contains only agent-visible data and no evaluator-only fields;
   - the MAPPO critic input is global, while the IPPO critic input is local;
   - advantages use the shared team reward;
   - checkpoint save/load round trip; and
   - inference returns contract-valid proposals.

   Never relabel IPPO as MAPPO.

3. **Generalization to unseen topologies.** The user requires evaluating generalization to unseen scenarios AND unseen topologies. Add at least one held-out topology variant to the lead-owned mock that is never used in training or calibration. The variant keeps the same node IDs and zones, so fixed-size observation vectors still apply, but has different link and dependency edges and different criticality values. Report metrics separately for held-out scenarios on the training topology and for the unseen topology. Optional, only if the GNN goes ahead: one topology with a different node count, evaluated on the GNN path only.

4. **Ranking metric where applicable.** Where a component produces per-node suspicion scores (the detection role, the graph-feature baseline, the GNN if built), the evaluator reports the rank of the true incident-origin node: top-1 accuracy and mean reciprocal rank, computed after each episode from evaluator-only ground truth. Label the metric as a ranking of predicted suspicion, never as a causal root-cause finding.

5. **Failed experiments.** Keep runs that fail to learn or that underperform in the evaluation outputs, and report them. Do not discard them or rerun until a favourable result appears. Do not tune on the evaluation split.

6. **D-19 wording.** The first addendum asks for D-19 to read "Pending: decided at stage 9, after IPPO, uncertainty and evaluation are stable". "Staged" is acceptable in place of "Pending", because the decision-log status table in `docs/DECISIONS.md` has no "Pending" value. What matters is that the GNN decision is taken at the GNN stage, not at design time.

### Review gate (addendum 2)

This gate adds to the review gate in the first addendum. The design-review step must not return APPROVED while the design:

- pre-decides a learned-GNN no-go;
- pre-decides a MAPPO deferral;
- lacks the unseen-topology evaluation;
- lacks the conventional-versus-hand-engineered-graph-feature ablation; or
- lacks the ranking metric for any component that produces per-node suspicion scores.

## Addendum 3: stage budget and responses to AD-10 and AD-11

The design written at 20:48 IST on 2026-10-30 (`DESIGN.md`, `docs/DECISIONS.md` AD-10 and AD-11) was based on the 19:49 version of this DELTA, which did not yet contain addenda 1 and 2. Its stage plan has no implementation slot for the learned-GNN stage or the MAPPO stage. This addendum states the workflow's stage budget and answers each reason given for deferral. It adds to the review gates of addenda 1 and 2.

### 1. Stage budget

- The workflow runs exactly nine implementation stages, S1 to S9. Each is a coder step followed by a reviewer step. After S9 come one final verify-and-fix loop and the report. A stage numbered S10 or later is never executed.
- The current plan uses S8 and S9 only for documentation and AI prompts, so the GNN and MAPPO stages would never run.
- The revised plan must fit within S1 to S9. It must give the learned-GNN stage and the MAPPO stage each its own slot, after IPPO, the uncertainty extension and the evaluation smoke path are stable, and must not implement the two in the same stage.
- Rebalance as needed: for example, merge documentation reconciliation, the teammate module specifications and the AI prompts into one stage, or have each earlier stage update the documents for the code it adds. The final verify-and-fix loop re-checks documentation consistency.
- S1 to S7 of the current plan, plus separate GNN and MAPPO stages, already make nine stages. A separate documentation stage therefore also requires merging two of the earlier stages into one.
- "Stage 9" and "stage 10" in addenda 1 and 2 are item numbers in the mandatory implementation order, not workflow stage numbers. Several order items may share one workflow stage, but the GNN item and the MAPPO item each need their own.

### 2. Responses to AD-10 (MAPPO deferral)

- "Stable IPPO evidence first": agreed. That is why the MAPPO stage comes after the IPPO stage.
- "User confirmation of the critic input and team reward": not required. The user told the workflow to make reasonable engineering decisions where the approved specifications cover the details and to avoid repeated confirmations. The design's critic input (the concatenation of agent-visible observation vectors, with no evaluator-only data) and the shared team reward stand.
- "An equal-budget multi-seed experiment is outside the foundation scope": it is in scope. The user explicitly asked for reproducible multi-seed experiments with ablations. Run a small equal-budget comparison on the mock, sized to finish on this machine's CPU, and label every result mock-only.
- "`state()` hook": the engine and RL environment are lead-owned, so the hook is designed and built in the stage that builds the environment. A missing hook cannot be a blocker.
- Status fields: the manifest and status values for `mappo` and `learned_gnn` must be computed from what actually ran (`implemented`, `not_run`, or `deferred` with an evidenced reason), not hard-coded constants. Rules that forbid the token `mappo` in `src/` apply only while MAPPO is not implemented.

### 3. Responses to AD-11 (learned-GNN no-go)

- "The only supervision source is the mock's own dynamics": this applies equally to IPPO trained on the mock. Handle it the same way, by labelling every mock-trained result as mock-only, not by skipping the model.
- "8 nodes and one incident type leave little for message passing": this is the hypothesis the experiment tests. The held-out topology variant (addendum 2, item 3) adds structural variation. If the GNN shows no measurable gain over the graph-feature baseline, report that result.
- "PyTorch Geometric adds network libraries": a plain-PyTorch dense-adjacency implementation (D-10) avoids PyG entirely.
- "No matched evidence yet that the graph-feature arm differs from the conventional arm": that evidence comes from the conventional-versus-graph-feature ablation, which runs before the GNN stage. It may inform the go/no-go criteria. The decision itself is taken at the GNN stage, once that evidence exists.
- AD-11 already says the decision is re-checked after IPPO. Keep that, and give the decision an implementation slot so that a go can actually be built and tested.

### Review gate (addendum 3)

This gate adds to the review gates of addenda 1 and 2. The design-review step must not return APPROVED while the design:

- has a stage plan that needs any stage after S9;
- lacks separate slots for the learned-GNN stage and the MAPPO stage; or
- hard-codes the `mappo` or `learned_gnn` status.
