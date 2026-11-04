# CyberSwarm Design Review (independent, blocking)

**Verdict: CHANGES_REQUESTED.** 4 HIGH, 17 MEDIUM, 11 NIT. Any HIGH or MEDIUM blocks approval.

**Reviewed:** `docs/ARCHITECTURE.md`, `docs/CONTRACTS.md`, `docs/RL_IMPLEMENTATION.md`, `docs/RESEARCH.md`, `docs/DECISIONS.md`, `.agents/tasks/cyberswarm-foundation/DESIGN.md`, `research-delta.md`. Authority read first: `DELTA.md` (including both addenda) and `BRIEF.md`. I also read `design-notes.md` because Addendum 2 names it.
**Review type:** first review (no prior review exists).
**Method:** read every document in full; recomputed the reward cases, the mock-topology numbers and the demo-episode metrics by hand; parsed the 22 JSON examples in CONTRACTS; fetched cited URLs and two arXiv abstracts and the CMARL-ACD repository metadata. Nothing in the design is implemented, so no code or test was run.

What passes: the MARL-primary framing, the closed 8-action vocabulary, ownership boundaries, the reward arithmetic, the uncertainty rule table, the IPPO independence design, the CMARL-ACD / CyMARL separation, and the git/license/identity handling. What blocks: Addendum 2's four review-gate conditions are all unmet (GNN no-go pre-decided, MAPPO deferral pre-decided, no unseen-topology evaluation, no ranking metric), plus stage-order, step-semantics, metric-validity and interface gaps below.

---

## HIGH findings

### 1. [HIGH] Learned GNN is decided as a design-time NO-GO; the stage-9 go/no-go is not planned
**Where:** ARCHITECTURE §3 (Learned GNN row), §15, §16; DECISIONS D-19, AD-11; RL_IMPLEMENTATION §1, §8, §9; CONTRACTS §10 intro, §10.3 ("In this phase no producer exists"), §11.6 (`status.learned_gnn` const `not_implemented`); DESIGN §1 rule 1, §4.13 (`learned_gnn: always not_implemented`), S6 row, DC-17, `tests/status/test_status_honesty.py`; `design-notes.md` ("Learned GNN: design-time NO-GO").
**Problem:** Addendum 2 §1 and its review gate say the go/no-go is taken at stage 9, after IPPO, uncertainty and the evaluation smoke path are stable, and the design must specify the candidate task, the two architectures, the inputs, explicit criteria and the matched ablation now. The documents do the opposite. The "reopening conditions" in RL_IMPLEMENTATION §8 ("a richer topology exists", "the user approves the compute plan") are not the user's criteria, and §8 itself says dependencies and compute are not blockers (DECISIONS D-19, D-10). The stage plan (DESIGN §11) has no GNN stage; S6 "confirms" the no-go before the uncertainty stage runs. The manifest schema hard-codes the outcome, so a later go would need a contract change.
**Fix:**
1. Rewrite D-19 as `Staged; decided at stage S8 (GNN) after IPPO, uncertainty and evaluation are stable`. Replace AD-11 with a plan entry. Delete the NO-GO wording in the other locations above.
2. Replace RL_IMPLEMENTATION §8 with a specification:
   - Task A: node-level suspicion scoring, trained on offline `(GraphBatch, nodes.compromised)` pairs built from `train`-split episodes only. Labels never appear in `cal`, `eval` or decision-time data. Task B (optional): node embeddings as the IPPO observation encoder.
   - Models, plain PyTorch on the existing `GraphBatch` (dense adjacency from `edge_index`): `SageDense` (mean aggregation) and `GatDense` (masked attention). PyTorch Geometric is not used.
   - Go criteria (all must hold to proceed to implementation): the training dataset has positive and negative nodes in every training scenario; a forward/backward pass on `GraphBatch` runs on CPU within a stated time budget; the train/eval split is disjoint by scenario, seed and topology.
   - No-go is allowed only with an evidenced reason recorded in the stage report (for example "no measurable improvement over the graph-feature baseline on the matched eval set, with spread"). Such a result is reported, not hidden.
   - Matched ablation: IPPO with `conventional`, `graph_features` and `learned_embedding` observations on identical `eval` and unseen-topology pairs, plus the node-score ranking metric of finding 4.
3. In CONTRACTS §11.6 change `status` to `ippo`/`learned_gnn`/`mappo` ∈ {`not_run`, `implemented`, `no_go`} with an optional `ShortText` `reason` that is required when `no_go`. Remove the `const` values and the `src/` token ban. Replace `test_status_honesty.py` with a test that the manifest status equals what the run executed.
4. Add a GNN stage after the evaluation stage in DESIGN §11 (see finding 6).

### 2. [HIGH] MAPPO is decided as deferred; the blockers are not concrete and the required tests are incomplete
**Where:** ARCHITECTURE §1, §6.5, §15; DECISIONS AD-10, AD-07 ("needs the user's confirmation before any MAPPO work"), D-06 ("otherwise it is reported as not implemented"); RL_IMPLEMENTATION §1, §6.1 to §6.5, §9, §10 items 1 and 2; CONTRACTS §11.6 (`mappo` const `deferred`); DESIGN §1 rule 1, §3.2 ("No file under `src/` contains the token `mappo`"), §4.13, S6, DC-18, `test_status_honesty.py`; `tests/status` asserts that `marl/mappo.py` does not exist.
**Problem:** Addendum 2 §2 says the design already specifies the centralized critic (concatenated agent-visible observations, no ground truth, shared team reward), IPPO is self-written, there are two agents, and MAPPO is a stage after IPPO (stage 10). Deferral is allowed only if that stage finds a concrete, evidenced blocker. RL_IMPLEMENTATION §6.3 lists B1 (IPPO must be stable: a sequencing step), B2 (`state()` hook: a work item), B3 (user confirmation of a critic input and reward that the addendum already fixed) and B4 (an experiment plan: not a code blocker). None is a blocker. The token ban, the `const` manifest value and the "no mappo.py" test make an implementation fail the repository's own tests. D-06 (conditional implement) contradicts AD-10 (deferred). The M-1 to M-8 test list lacks four tests that Addendum 2 requires: an explicit check that each actor receives only its own local observation at execution; an explicit check that the MAPPO critic input is global while the IPPO critic input is local; a checkpoint save/load round trip; and contract-valid proposals from inference. "Advantages use the shared team reward" is only implied by M-4.
**Fix:**
1. Add a MAPPO stage after the GNN stage that executes work items W1 to W7. Replace "deferred" by `not_run` until the stage runs. Defer only if the stage finds a blocker, and record it with evidence (command output) in the stage report.
2. Delete B3 and AD-07's confirmation clause; the shared team reward and the concatenated-observation critic are the design. Delete B1, B2 and B4 as "blockers" and keep them as ordering and work items.
3. Make D-06 and AD-10 consistent with this plan.
4. Replace the token ban and the "no mappo.py" assertion with the status-equals-execution test of finding 1.
5. Add tests: M-9 `test_actor_input_is_own_obs_only` (patch the critic module to raise; `MAPPOPolicy.propose` still works and the actor receives a tensor equal to its own `obs`, shape `(D,)`); M-10 `test_critic_input_global_vs_local` (MAPPO critic input dimension `n_agents * D`, IPPO critic input `D`, on the same environment); M-11 `test_advantage_uses_team_reward` (per-agent reward arrays equal `StepScore.total`; both actors get the same advantage vector); M-12 `test_mappo_checkpoint_roundtrip` (identical logits after load; actors-only load works); M-13 `test_mappo_inference_proposals` (proposals validate against `PolicyProposal` and pass the real coordinator).

### 3. [HIGH] No unseen-topology evaluation anywhere
**Where:** absent from DESIGN §5 (one topology `topo-mock-small-v1`), §5.2 (12 scenarios, all on v1), `splits.toml`, CONTRACTS `Split` enum (`train`, `cal`, `eval`, `dev`, `fixture`), RL_IMPLEMENTATION §9. A search for "unseen" and "topology variant" in all seven documents finds nothing. Also `load_checkpoint(path, *, expect_topology)` (DESIGN §4.10) and the stored "topology id" in `checkpoint.json` (RL_IMPLEMENTATION §5.5) suggest a topology-id check; if the loader compares ids, a v1-trained checkpoint is refused on v2, and the text does not say which comparison it makes.
**Problem:** Addendum 2 §3 requires at least one held-out topology variant that is never used in training or calibration, with the same node ids and zones, different link and dependency edges and different criticality, and metrics reported separately for held-out scenarios on the training topology and for the unseen topology. The addendum's review gate lists this explicitly.
**Fix:**
1. Add `configs/topologies/topo-mock-small-v2.json`: the same 8 `node_id`s and zones, a different `network_link` set, a different `service_dependency` DAG, different criticalities. Example (the lead may choose others): replace the `ws-01`↔`file-01` link by `ws-01`↔`app-01`; replace the dependency `app-01 → auth-01` by `db-01 → auth-01`; set criticality `web-02` 0.6, `file-01` 0.8, `ws-01` 0.4.
2. Add scenarios `scn-mock-lateral-utopo-01` and `-02` (`topology_id = topo-mock-small-v2`) and a split `eval_utopo` in `splits.toml`. Add `eval_utopo` to the `Split` enum before schema 1.0 ships (a later addition is a MINOR change that forces the dashboard to regenerate validators).
3. `assert_disjoint` must also assert that the topology ids of `train` and `cal` scenarios never appear in `eval_utopo`, and a test must enforce it. Nothing on v2 may be used for training, threshold selection or calibration.
4. Change checkpoint loading to compare the node-id list and `ActionTable.digest()` (both identical on v1 and v2) and record the topology id of training and evaluation separately in the report; do not refuse on topology id.
5. The harness reports every metric separately for `eval` and `eval_utopo`; `EvaluationMetrics.split` and `ComparisonRecord.split` already carry it. Add the split to the smoke subset (one pair) and to the demo bundle.
6. Add tests: v2 passes the topology validator and conformance checks SC-1, SC-2; a v1 checkpoint loads and runs on v2; graph features differ between v1 and v2 for the same observation; disjointness. The optional different-node-count topology belongs to the GNN stage only.

### 4. [HIGH] The ranking metric (top-1 accuracy and mean reciprocal rank of the incident-origin node) is missing, and the graph-feature baseline emits no per-node scores
**Where:** CONTRACTS §11.2 metric table (no such metric); RL_IMPLEMENTATION and DESIGN (no definition); ARCHITECTURE §3 ("Root-cause ranking ... is not an evaluated metric") and §16; DECISIONS AD-01 decision 3.
**Problem:** Addendum 2 §4 requires, wherever a component produces per-node suspicion scores (the detection role, the graph-feature baseline, the GNN if built), top-1 accuracy and MRR of the true incident-origin node, computed per episode from evaluator-only ground truth and labelled as a ranking of predicted suspicion. The detection role produces `DetectionSummary.nodes[].suspicion`. The graph-feature baseline computes `suspicion + neighbor_alert_weight * neighbor_alert_mean` inside the policy (DESIGN §4.9) and records it nowhere, so the evaluator cannot rank it. ARCHITECTURE states the opposite of the requirement.
**Fix:**
1. Add a registry record `NodeScoreRecord` (`schema_version`, `score_id`, `episode_id`, `step`, `agent_id`, `scorer_id` ShortText such as `detection.rule-v1` or `graph-rule-v1`, `scorer_kind` ∈ {`detection_rule`, `graph_feature_rule`, `learned_graph_model`}, `nodes` list of `{node_id, score: UnitFloat}` over the nodes the agent observes). File `node_scores.jsonl` in the episode directory. Produced by the detection role, by `GraphFeaturePolicy` (or `graph.features`), and by the GNN if built.
2. Add metrics `origin_rank_top1` and `origin_mrr` per scorer to the §11.2 table and matching nullable fields in `EpisodeResult.outcome` keyed by `scorer_id`. Definition: the origin node `o` is `incident.entry_node_id`. For each step `t` from `first_true_alert_step` to `containment_step` (to the last step if containment is never reached) and each agent whose observation contains `o`, take the rank of `o` among that agent's scored nodes by descending score. Ties get the worst rank of the tie group. `origin_rank_top1` is the fraction of (step, agent) pairs with rank 1; `origin_mrr` is the mean of `1/rank`. Null when there is no pair.
3. State in CONTRACTS, EVALUATION and the dashboard spec: "ranking of predicted suspicion, not a root-cause or causal finding".
4. Correct ARCHITECTURE §3 and §16 and AD-01 decision 3. Add unit tests with hand-built scores (known rank, known tie, null case).

---

## MEDIUM findings

### 5. [MEDIUM] The graph pipeline is still treated as an open question the user must confirm
**Where:** DECISIONS AD-01 "Supersession note" ("If the earlier wording was intended, the rows to change are ..."), AD-15 (status "Needs the user's confirmation"; "Reversal ... drop the files listed above"), §5 item 9, §7 item 10; DESIGN §1 rule 2, §12.2 last row ("The graph pipeline is judged unnecessary"), §12.3 item 1; RESEARCH §4 item 18; `research-delta.md` §6b.
**Problem:** DELTA states the graph pipeline is required and the user confirmed it; the first addendum's gate forbids approval while the DECISIONS entries stay "stale or contradictory". AD-01 says "Confirmed by the user" while AD-15 asks the user to confirm the same fact. The "removal list" frames required infrastructure as removable.
**Fix:** Close AD-15 as `Decided (DELTA.md and BRIEF.md; user-confirmed)`. Delete the supersession hedge in AD-01, open items 9 and 10, DESIGN §1 rule 2, the §12.2 risk row, §12.3 item 1, RESEARCH item 18 and `research-delta.md` §6b. Keep the note that `graph/converter.py` is shared by both observation modes as an architecture fact, not as a removal option.

### 6. [MEDIUM] Stage order violates the mandatory order, and DESIGN §1 rule 3 says it does not
**Where:** DESIGN §11.1 (S5 = DELTA step 8 runs before S6 = steps 6, 9, 10 and S7 = step 7); §1 rule 3 ("The stage names S1 to S9 below keep this order"); ARCHITECTURE §10.4 (IPPO and uncertainty after the evaluator).
**Problem:** DELTA's mandatory order is: contracts/mock, graph, RL env/reward, baselines, coordinator, IPPO, uncertainty, evaluator/smoke, GNN, MAPPO, docs, final review. The design runs the evaluator before IPPO and uncertainty, and takes the GNN and MAPPO decisions inside S6, before uncertainty. The reason is probably real (IPPO's held-out evaluation needs an evaluator), but the document claims compliance and gives no recorded exception. `AI_PROMPT_PROJECT_LEAD` must preserve the DELTA order (BRIEF phase 10), so the prompt and the stages would disagree.
**Fix:** Use this order: S5 IPPO (with a thin `runner.evaluate.run_episode` and `evaluation.outcome.compute_outcome` slice so held-out evaluation executes), S6 uncertainty, S7 evaluation harness, comparison, smoke and dashboard outputs, S8 GNN stage, S9 MAPPO stage, S10 documentation reconciliation, S11 module specs, prompts and docs validator, then the final review. If the lead prefers the evaluator first, record that as a DECISIONS entry with the reason and the user's approval, and correct rule 3 and the prompt.

### 7. [MEDIUM] S3 combines two subsystems and its ordering gate cannot be checked
**Where:** DESIGN §11.2 S3 (42 files: roles, graph, engine, reward, action table, encoder, env) and §11.3 S3 "Done when" ("Graph tests pass before any `rl` file exists in the commit order").
**Problem:** DELTA requires the graph pipeline to be implemented and tested before the RL environment. One coder-then-reviewer loop covers both, and the gate refers to "commit order" although no commits are allowed. The reviewer cannot enforce the order.
**Fix:** Split S3 into S3a (`roles`, `graph`, graph fixtures, `scripts/graph_evidence.py`, graph and role tests; reviewer approval is the gate) and S3b (`engine`, `rl`, reward, env tests). Gate: `ls src/cyberswarm/rl src/cyberswarm/engine` must show neither directory when S3a is approved. Renumber the later stages.

### 8. [MEDIUM] An S1 test depends on S5 artifacts
**Where:** DESIGN §9 `tests/contracts/test_fixture_consistency.py` (stage S1, "X-1 to X-14 on the demo bundle"); §11.3 S1 acceptance (`pytest tests/contracts tests/common`); `fixtures/bundle_demo/` is created in S5.
**Problem:** At S1 the bundle does not exist, so the test fails or skips. S1's "done when" requires reported pass counts, and a skip hides a missing check.
**Fix:** In S1 build a primary mini-episode (events, observations, available actions, proposals, validations, executions, ground truth) directly from `scripts/fixture_storyboard.py`, and run rules X-1 to X-10 and X-14 on it. Move X-11 to X-13 and the full bundle to an S5 test. State "0 skipped" as an S1 acceptance condition.

### 9. [MEDIUM] When `ground_truth()` is read relative to `end_step()` is ambiguous
**Where:** DESIGN §4.2 (`end_step`: "closes step t, evaluates termination, runs P0 of t+1"; `ground_truth()`: "end-of-step truth for the current step"; `step`: "current step t"); §4.6 (engine order: `end_step`, drain events, then query the truth accessor); CONTRACTS §9 ("written after P4 of that step"); RL_IMPLEMENTATION §4.1.
**Problem:** After `end_step()` returns, the simulator has already advanced to t+1 and applied P0 of t+1 (restore completion, new compromise, new alerts). A call to `ground_truth()` then returns either the t+1 state or a snapshot of t, and the text does not say which. The reward (`C_t`, `K_t`), `false_isolation`, `containment`, the `GroundTruthSnapshot.step` field and the outcome metrics all depend on the answer. A simulator written by Raksha could pick the other reading and both would pass the 13 conformance checks.
**Fix:** Specify that `end_step()` captures the end-of-step-t snapshot before running P0 of t+1, `ground_truth()` returns that captured snapshot (with `step == StepResult.step_closed`) until the next `end_step()`, and `termination_reason()` refers to the same step. Add conformance check SC-14: with a scenario where `restore_completed` fires at P0 of t+1, the snapshot for t still shows the node as `restoring` and still compromised.

### 10. [MEDIUM] The healthy-disruption reward term and metrics mix attack-induced degradation with defense-induced disruption
**Where:** DESIGN §5.3 (`own_avail = status_avail * (0.7 if compromised else 1.0)`; `avail(n) = min(own_avail(n), min over dependencies avail(d))`); RL_IMPLEMENTATION §4.1 `healthy_disruption`; CONTRACTS §11.2 `healthy_services_disrupted`, `healthy_disruption_cost`.
**Problem:** A healthy node that depends on a compromised node loses availability through the attack (factor 0.7), and that loss is charged to `healthy_disruption` and counted as a "healthy service disrupted". The central question is about unnecessary disruption caused by the defense. A policy that contains faster has less attack-induced loss and scores better on a metric meant to measure defender harm. By my hand computation on the demo storyboard, `healthy_disruption_cost = 2.125` includes 0.30 of attack-induced loss on `node-web-01` (steps 6 and 7).
**Fix:** Add `defense_availability` per node to `GroundTruthSnapshot`: availability computed with the compromise factor fixed at 1.0 (status and dependency effects only). `healthy_disruption`, `healthy_services_disrupted` and `healthy_disruption_cost` use `defense_availability`; `attack_impact` keeps using compromise. Update R-10 and the demo fixture (by my hand computation the demo `healthy_disruption_cost` becomes 1.75). Keep collateral from necessary containment in the metric, and say so in the definition.

### 11. [MEDIUM] In the mock, `service_health` is a near-deterministic proxy for compromise, and the rule baseline ignores it
**Where:** DESIGN §5.3 (`own_avail` ×0.7 when compromised; `service_health = clip(avail + 0.03 * noise, 0, 1)`; edge `traffic_level` +0.15 when the source is compromised, noise 0.03; the distribution of `noise` is not defined); RL_IMPLEMENTATION §3 (index 9 is in the policy input); DESIGN §4.9 (the rule baseline does not read `service_health`); CONTRACTS §8.2 example shows 0.71 for the alerted node.
**Problem:** A compromised node without compromised dependencies reads about 0.70; a healthy one reads about 1.00; the noise is 0.03. The gap is roughly ten noise units. A learned policy can read compromise off one input, which defeats the alert/evidence/uncertainty mechanism the project studies, and the rule baseline cannot, so IPPO-versus-rule differences would partly measure access to an oracle-like channel. ARCHITECTURE §6.2 claims the compared policies differ only in the decision role; this makes that claim false.
**Fix:** Pick one and record it in DESIGN §5.3: (a) reduce the effect so the channel is informative but not decisive, for example compromised factor 0.9, `service_health` noise 0.08 (d′ ≈ 1.25), edge offset 0.05 with noise 0.08; or (b) keep the channel and add a `service_health`-based candidate rule to both rule baselines. Define `noise` (standard normal truncated at ±3, drawn from the `health_noise` table). Add a mock test that a threshold on `service_health` predicts compromise with AUROC below an agreed bound on a random-policy rollout.

### 12. [MEDIUM] `visible_to` is specified only for alerts
**Where:** DESIGN §5.3 step 4 (alerts only); CONTRACTS §8.1 (`visible_to`: "at least one agent"); RL_IMPLEMENTATION §2.5 ("never sees the other agent's ... proposals or actions").
**Problem:** The mock and Raksha's simulator must decide who sees `action_applied`, `investigation_result`, `node_status_changed`, `restore_completed`, `episode_started`, `episode_ended` and the coordinator's `recovery_*` events. If `action_applied` (which carries `agent_id`) reaches the other zone's agent, that agent sees the other agent's actions, which §2.5 forbids. Conformance cannot test an unspecified rule.
**Fix:** Add a table to CONTRACTS §8.1: `episode_started` and `episode_ended` visible to all agents; `alert_raised`, `node_status_changed`, `restore_completed`: the node's zone agent plus agents owning a node adjacent over a `network_link`; `action_applied` and `investigation_result`: only the acting `agent_id`; `recovery_verified`/`recovery_failed`: the node's zone agent. Add conformance check SC-15: no observation contains an `action_applied` or `investigation_result` caused by another agent.

### 13. [MEDIUM] The recovery-verification rule is not defined
**Where:** ARCHITECTURE §5 row 17 ("executed restore/containment"); DESIGN §4.8 (`RecoveryVerifier(health_min=0.8, window=2)`, `note_execution`, `check`); CONTRACTS §8.1 event table; DESIGN §8.1 storyboard (step 12 `recovery_verified`).
**Problem:** DELTA requires recovery verification from simulator-reported data. No document says what satisfies `recovery_verified`, what produces `recovery_failed`, which executions are tracked, how a null `service_health` counts, who sees the event, or when to give up. `tests/coordinator/test_recovery_verification.py` has nothing to assert.
**Fix:** Specify: track only executed `restore_node` actions. Arm at the `restore_completed` event for the node, at step `s`. Emit `recovery_verified` (producer `coordinator`, id `cev-`, `node_id`, `execution_id`, `visible_to` = node's zone agent) at the first step `t ≥ s` such that, for the last `window` consecutive steps, the zone agent's observation shows `visible_status == up` and a non-null `service_health ≥ health_min`; a null value resets the streak. Emit `recovery_failed` if that does not happen within `verify_timeout_steps` (default 6) after `s`, or if no `restore_completed` arrives within `restore_duration_steps + 2` after the execution. Quarantine and isolate executions are not verified (their recovery is the later restore). Add tests for verified, failed, null-breaks-streak and unfinished-at-episode-end.

### 14. [MEDIUM] Coordinator reason-code evaluation and id assignment order are ambiguous, and the storyboard contradicts the stated rule
**Where:** CONTRACTS §8.6 ("all failures are listed ... a check that depends on an earlier failed check ... is skipped"; ids assigned by the coordinator; proposals processed in ascending `agent_id`); DESIGN §8.1 storyboard step 6; CONTRACTS examples (`prop-000009`/`val-000009` for `agent-dmz`, `val-000010` for `agent-core`).
**Problem:** Which checks depend on which is not given. The storyboard's rejected proposal (core agent targeting `node-web-02`) lists only `TARGET_OUT_OF_AUTHORITY`, but `ACTION_NOT_AVAILABLE` also fails because that action is not in the core agent's `AvailableActions`, so the stated rule would list two codes. Counter assignment is also unclear: ascending `agent_id` puts `agent-core` before `agent-dmz`, but the examples give the lower id to `agent-dmz`.
**Fix:** Add a dependency table: `SCHEMA_INVALID` short-circuits everything; `EPISODE_MISMATCH`, `EPISODE_NOT_ACTIVE`, `WRONG_STEP`, `AGENT_UNKNOWN`, `DUPLICATE_PROPOSAL` and `UNKNOWN_ACTION_KIND` always run once the input parses; `ACTION_NOT_ALLOWLISTED`, `TARGET_REQUIRED`/`TARGET_NOT_ALLOWED` and `PARAMETER_INVALID` need a known kind; `UNKNOWN_TARGET_NODE` needs a target; `TARGET_OUT_OF_AUTHORITY` needs a known target and a known agent; `ACTION_NOT_AVAILABLE` runs only if none of `UNKNOWN_ACTION_KIND`, `TARGET_*`, `UNKNOWN_TARGET_NODE`, `TARGET_OUT_OF_AUTHORITY`, `PARAMETER_INVALID`, `AGENT_UNKNOWN` failed; `CONFLICTING_PROPOSAL` and `HIGH_IMPACT_BUDGET_EXCEEDED` run only when every earlier check passed. State that `prop-`, `val-` and `exe-` counters are assigned in processing order (ascending `agent_id`). Regenerate the examples accordingly.

### 15. [MEDIUM] Three different lists say who may import evaluator-only data, and the calibration module is on none of them
**Where:** CONTRACTS §9 ("Packages other than `rl.reward`, `evaluation`, `runner` and the test suite must not import"); ARCHITECTURE §9 ("only `rl.reward`, `engine`, `evaluation` and tests"); DESIGN §3.2 (adds `interfaces.simulator`, `interfaces.ports`, `mock.*`, `engine.step_engine`); RL_IMPLEMENTATION §7.4 and DESIGN §4.11 (`uncertainty.calibration` labels each proposal correct or incorrect from the ground-truth stream).
**Problem:** The import-boundary test enforces one list. The documents disagree, and `uncertainty.calibration` needs the truth to compute its correctness label but may not import it under any list.
**Fix:** Make DESIGN §3.2 the single list and have CONTRACTS §9 and ARCHITECTURE §9 point to it. Add `evaluation.calibration_labels(proposals, truth) -> (u, correct)` and have `uncertainty.calibration` take those arrays (or move calibration into `evaluation`, which is pure numpy). The runner passes the arrays. Extend the boundary test with the new function.

### 16. [MEDIUM] The IPPO training entry point violates the package import rules
**Where:** DESIGN §3.2 (`marl` may import `contracts, interfaces, common, rl, engine, numpy, torch`; only `runner.factory` imports `mock`); §4.10 (`marl/train.py`); §11.3 S6 and §12.1 (`python -m cyberswarm.marl.train`).
**Problem:** A training command must build the mock simulator, the real coordinator and the reward model, none of which `marl` may import. `test_import_boundaries` would fail, or the module would need an unwritten exception.
**Fix:** Move the CLI to `runner/train_ippo.py` (`python -m cyberswarm.runner.train_ippo --config ... --out ...`). Keep `marl.ippo.IPPOTrainer(env, config, out_dir)` unchanged. Update the tree, S6 acceptance and the verification table.

### 17. [MEDIUM] Aggregation over training seeds and the list of comparisons are not defined
**Where:** RL_IMPLEMENTATION §9 ("repeats over training seeds k, reports mean and spread"); DESIGN §6 `eval_default.toml` (`[training] seeds = [0,1,2,3,4]`); CONTRACTS §11.2 and §11.3 (`EvaluationMetrics` and `ComparisonRecord` have a single `policy_id`).
**Problem:** Episodes of five differently seeded IPPO networks are not independent samples of one policy. The episode-level bootstrap in §11.2 understates the spread that matters for the IPPO comparisons, and no record holds a training-seed aggregate. The document also never lists which arms are compared, so the required conventional-versus-graph-feature and uncertainty-versus-control comparisons are implied, not specified.
**Fix:** Add `policy_family_id` and `training_seed` (nullable) to `EvaluationMetrics`. Add an `AggregateMetrics` record: per `(policy_family_id, representation, split)`, the per-training-seed means and the mean, sample standard deviation and percentile bootstrap interval across training seeds (unit = training seed; interval null below 5). Enumerate the `ComparisonRecord` pairs in RL_IMPLEMENTATION §9: `rule-conv` vs `rule-graph`; `ippo-conv` vs `ippo-graph`; `ippo-conv-ua` vs `ippo-conv-ctl`; `rule-conv` vs `ippo-conv`; `rule-graph` vs `ippo-graph`; each on `eval` and `eval_utopo`. State that failed or underperforming runs stay in the outputs (Addendum 2 §5), and that no tuning uses `eval` or `eval_utopo`.

### 18. [MEDIUM] The rule-based baseline can leave a contained node uncleared indefinitely
**Where:** DESIGN §4.9 rules 2a to 2e.
**Problem:** Rules 2a and 2b act on a contained node only when its evidence is 0 or at least `e_contain`. Rule 2c applies to "not contained" nodes. Evidence reverts to null after 5 steps (§5.3). A quarantined, compromised node with expired or middling evidence gets no rule, so the baseline ends with `no_op` and the node stays compromised until truncation. That handicaps the baseline in every comparison and keeps the episode from reaching `incident_resolved`.
**Fix:** Add rule 2f: a contained node whose evidence is null, 1 or 2 receives `investigate_node(n)` if `steps_since_investigated` is null or at least `investigate_cooldown`, otherwise `request_evidence(n)`, with codes `evidence_insufficient`, `recovery_after_containment`. Apply the same rule in the graph variant. Test: a quarantined compromised node whose evidence expired is re-investigated within `investigate_cooldown + 1` steps and the episode can reach `incident_resolved`.

### 19. [MEDIUM] `ShortText` can carry IP literals and path-like strings
**Where:** CONTRACTS C8 (allowed characters `[A-Za-z0-9 .,:()/%=_+-]`), fields `rationale_text`, `label`, `description`, `note`, `message`.
**Problem:** The design promises no real commands, no IPs of real hosts and no free-form payloads anywhere in the schema. The address-look-alike rejection applies only to `NodeId`. `ShortText` accepts `10.0.0.5`, `192.168.1.1`, `rm -rf /` and `../etc/passwd`. No component executes the text, but the stated rule and the invalid fixtures (`ActionSpec.raw-ip-target`) show the intent.
**Fix:** Remove `/`, `=`, `+` and `:` from the allowed set (keep `[A-Za-z0-9 .,()_-]`). Add validators that reject a dotted-quad `\b\d{1,3}(\.\d{1,3}){3}\b`, the sequence `..`, and a leading `-`. Add invalid fixtures `PolicyProposal.ip-in-rationale.json` and `PolicyProposal.path-in-rationale.json`.

### 20. [MEDIUM] The cross-check matrix omits several interfaces, and its completeness proof checks only the 23 records
**Where:** DESIGN §10 and §10.1.
**Problem:** The matrix has no row for the engine API (`EpisodeEngine`, `StepContext`, `StepOutcome`), `CoordinatorPort`, `Recorder`, `ObservationEncoder` and `ActionTable` (I-21 covers spaces and info only), `EpisodeTraceView`/`EpisodeRecordSet` (inputs to `compute_outcome` and `check_episode`), `ProposalFeedback`, `checkpoint.json` and `training_log.jsonl` (files with a producer and a consumer but no schema), or any item added by findings 1 to 4 (unseen-topology split, `NodeScoreRecord`, GNN dataset, MAPPO `state()` hook). §10.1 point 6 states "no interface is duplicated", but the `e_min` threshold is defined three times (`evidence_view(e_min=2)`, `RuleConfig.e_min`, `UncertaintyRuleConfig.e_min`).
**Fix:** Add rows I-37 and following for each missing interface with producer, consumer, schema or type, fixture, test and stage. Give `checkpoint.json` and `training_log.jsonl` small pydantic models (not registry records) with a round-trip test. Define `e_min` once (`EvidenceConfig` in `roles`) and reference it from the other two configs. Extend §10.1 to check every Protocol and every on-disk file name, not only the registry.

### 21. [MEDIUM] Shared configs and docs contain machine-specific absolute paths
**Where:** CONTRACTS C1 ("exported ... into `/Users/guru/CyberSwarm-GX/schemas/`") and §2 export command; DESIGN §6 `smoke.toml` (`output_dir = "/Users/guru/CyberSwarm-GX/runs"`), §7.1, §11.3 and §12.1 commands (`--config /Users/guru/CyberSwarm-GX/...`).
**Problem:** Raksha and Mayukh Aryan work on other machines. Committed configs and quickstart commands that name `/Users/guru/...` fail for them, which breaks the "starts independently from fixtures" requirement. The absolute-path rule is for workflow step agents, not for repository content.
**Fix:** Make every repository config and documented command repo-relative (`output_dir = "runs"`, `--config configs/experiments/smoke.toml`, run from the repository root). Resolve relative paths against the repository root found from `pyproject.toml`. Keep absolute paths only in step prompts.

---

## NIT findings

### 22. [NIT] Documentation examples are missing for four schemas
`Topology`, `ActionSpec`, `ProposalFeedback` and `UncertaintyInfo` have no `json contract=` example in CONTRACTS (`last_step_feedback` is `[]` in the only observation example). BRIEF asks for an example per schema. **Fix:** add four marked examples; the documentation validator then checks them.

### 23. [NIT] Name mismatches between documents
ARCHITECTURE §5 row 11 uses `RewardModel.compute(...)`, DESIGN uses `StepScorer.score(...)`; row 14 lists `calibrate`, DESIGN has `build_report` and `fit_histogram_calibrator`; the §4.1 diagram orders `contracts <- interfaces <- common` while DESIGN §3.2 has `interfaces` importing `common`; `ErrorRecord.component` has no value for `engine`, `roles` or `mock`. **Fix:** align the names; add `engine`, `roles`, `mock` to the enum.

### 24. [NIT] Graph-record edge cases are unspecified
`GraphSnapshot.max_event_step` is a required `Step` but an observation window can contain no events; `GraphFeatureRecord` features 1 to 3 do not state their mask value in the default case; `degree_norm` divides by `N - 1` (zero when N = 1, which `Topology` allows); the diameter is undefined on a disconnected link graph. **Fix:** make `max_event_step` nullable, give a mask rule for columns 1 to 3, require `N ≥ 2` and a connected link graph in `Topology` validation.

### 25. [NIT] `state_digest()` of the full hidden state sits on the agent-facing `Simulator` Protocol
It is a hash and cannot be inverted, but its changes reveal that hidden state changed. **Fix:** move it to a separate `IntegrityAccessor` Protocol used by the coordinator and engine only, and keep it out of anything an agent receives.

### 26. [NIT] Example record ids do not follow the storyboard counters
With two proposals per step, step 5 proposals are `prop-000011` and `prop-000012`, not `prop-000009`; the rejected example sits at step 5 while the storyboard rejects at step 6. **Fix:** regenerate the examples from the storyboard.

### 27. [NIT] `reset(seed=None)` raises `ValueError`
DESIGN §4.7 makes this deliberate. PettingZoo's `parallel_api_test` calls `reset()` without a seed. **Fix:** accept `None` and derive a documented default seed, or state that the API test is not used.

### 28. [NIT] Disk-space guard is partial
DESIGN S1 measures free space once; DELTA requires a second measurement before the torch install. The fallback "install without the `rl` group" also removes `gymnasium` and `pettingzoo`, which S3b needs and which are small. DECISIONS, RESEARCH and `research-delta.md` quote 650 MiB; I measured 2.6 GiB free during this review. **Fix:** split a `torch` dependency group from an `env` group (`gymnasium`, `pettingzoo`), measure before venv creation and again before the torch install, and date-stamp the quoted figures.

### 29. [NIT] The original ARCHITECTURE draft v0.1 is not preserved
The file was untracked, 546 lines per RESEARCH Appendix A; it is now rewritten in place (409 lines) and no copy exists in the repository. §16 lists what was kept and changed. **Fix:** tell the user in the final report that v0.1 is no longer recoverable from the repository and ask whether they hold a copy.

### 30. [NIT] Teammate-facing artifacts arrive late
`fixtures/bundle_demo/` is created in S5 and the `MODULE_*` specs in S9, although Raksha needs only S1 and S2 (contracts, conformance suite) to start and Mayukh needs only the bundle. **Fix:** publish `MODULE_RAKSHA_SIMULATOR.md` after S2 and `MODULE_MAYUKH_DASHBOARD.md` with the bundle.

### 31. [NIT] Stale cross-references
RESEARCH §4 item 1 says `docs/ARCHITECTURE.md` "never mentions CMARL-ACD" (v0.2 does); DECISIONS AD-01 "Effect on documents" says ARCHITECTURE was left unmodified; DECISIONS D-17 lists `torch_geometric` as a staged extra while AD-04 says it is not declared; `final observation` mask after `done` is unspecified (RL_IMPLEMENTATION §2.2). **Fix:** update the three statements; define the final-observation mask as `no_op` only.

### 32. [NIT] Calibration sample sizes are unverified
The informativeness gate needs at least 100 committing proposals in `cal` and in `eval` (RL_IMPLEMENTATION §7.4); `cal` has 2 scenarios × 10 seeds, and the smoke subset has 8 episodes. The design accepts `insufficient_data`, but nothing estimates whether the full configuration can reach the gate. **Fix:** after S5, record the measured committing-proposal count per episode for the random and rule policies and state the minimum `cal` seed count needed; raise `cal` seeds if the count is below 100.

---

## Verified Assumptions

| # | Assumption | How verified |
|---|-----------|--------------|
| V1 | Repository state: one commit (`dc08399`), branch `main`, working tree has only untracked `.agents/` and `docs/`, no `LICENSE`, no new commits. | `git log`, `git status`, `ls` |
| V2 | The reward cases R-1 to R-9 are arithmetically correct with the default weights (-3.5, -1.25, +2.5, -0.575, -0.05/-0.25/-0.5, -0.05/-0.15/-0.025, -1.0, -30.0). | Recomputed by hand |
| V3 | Mock topology numbers: 19 edges and their numbering, link diameter 4, action-table sizes 22 and 36, observation dimensions 138 and 202, DAG dependencies. The `GraphFeatureRecord` example (5/7, 0.56, 1.0, 0.119, 0.25) and the `DetectionSummary` suspicion 0.4333 follow from the formulas. | Recomputed from DESIGN §5.1 and CONTRACTS §10 |
| V4 | The `GraphSnapshot` example satisfies the shape and "imputed means value 0.0" rules. All 22 `json contract=` examples in CONTRACTS parse as JSON. | Script |
| V5 | Demo `EpisodeResult` values (`attack_impact` 5.6, `containment_step` 8, `detection_to_containment_steps` 3, 26 proposals, `healthy_disruption_cost` 2.125 under the current definition) follow from the storyboard. | Recomputed by hand |
| V6 | The CMARL-ACD repository exists (`cyb3rlab/CMARL-ACD`), is titled "Competitive Multi-Agent Reinforcement Learning for Autonomous Cyber Defense", was created 2026-06-03, pushed 2026-06-23, reports licence `Other/NOASSERTION`, describes IPPO versus MAPPO, IDS uncertainty and Python 3.9 to 3.12. This matches RESEARCH §1.2.1. | GitHub API and README fetch |
| V7 | arXiv:2606.13832 (Silva) and arXiv:2401.10149 (Wilson et al.) abstracts support RESEARCH §1.6 (100 percent downtime-budget violation for unconstrained methods, MAPPO 800K steps versus IPPO 0.966 at 1M steps, alert success 0.75 and 0.9). arXiv:2310.05939 resolves to the CyMARL title. | arXiv API and abstract page |
| V8 | 36 cited URLs returned HTTP 200 (ten arXiv abstract pages, the GitHub repositories, the PyTorch/PettingZoo/Gymnasium/NetworkX/pydantic/PyG/BenchMARL/AAAI pages, npm `cytoscape`, PyPI `torch_geometric`), and the arXiv API answered. The Wiley page returned 403, as RESEARCH records. | `curl` |
| V9 | CMARL-ACD and CyMARL are kept as separate works in ARCHITECTURE §2, RESEARCH §0, DECISIONS AD-02. No novelty, first or superiority claim by this project appears; "novel" and "first" occur only as quoted claims of third-party authors. | Text search |
| V10 | Ownership matches BRIEF: all RL, graph, coordinator, uncertainty and evaluation work is lead-owned; the mock lives in `src/cyberswarm/mock/` and `configs/scenarios/mock/`, outside Raksha's `src/cyberswarm/simulator/`, `tests/simulator/`, `configs/scenarios/simulator/`; `dashboard/` is Mayukh's; no placeholder directories for either. | DESIGN §2, DECISIONS AD-04 |
| V11 | DELTA Addendum 1 corrections are in DECISIONS for AD-01 items 3 to 5, D-09 (NetworkX core), D-17, D-18 (Required); PyG 2.8.1 facts are recorded as stage-9 inputs. | Text comparison |
| V12 | Conventional versus hand-engineered-graph-feature arms exist for the rule baseline and IPPO (`rule-conv`/`rule-graph`, `ippo-conv`/`ippo-graph`) with the same architecture, and the uncertainty-aware policy has a matched control sharing the checkpoint. | RL_IMPLEMENTATION §3, §7.3, §9 |
| V13 | IPPO independence is structurally specified: per-agent actor, critic, optimizer and value normalizer, local-observation critics, tests for disjoint parameters, no cross-gradient and update isolation. | RL_IMPLEMENTATION §5.3, §5.5 |
| V14 | Free disk space is 2.6 GiB at review time, above the 1.5 GB guard in the DELTA addendum. | `df -h` |
| V15 | Optional uncertainty and graph-model values are null or absent when not computed (`UncertaintyInfo.value`, `uncertainty` on baselines, `GraphModelOutput.not_computed`, `uncertainty_summary`, `reward`, metric `n_undefined`). | CONTRACTS §8.5, §10.3, §11 |

## Unverified or Wrong Assumptions

| # | Assumption in the design | Status |
|---|--------------------------|--------|
| W1 | "A documented no-go satisfies the stage" lets the GNN be decided at design time (DESIGN §1 rule 1, AD-11). | **Wrong.** Addendum 2 requires the decision at stage 9, from evidence (finding 1). |
| W2 | MAPPO "deferral" is justified by the four listed blockers (RL_IMPLEMENTATION §6.3). | **Wrong.** None is a technical blocker (finding 2). |
| W3 | "The stage names S1 to S9 keep this [DELTA] order" (DESIGN §1 rule 3). | **Wrong** (finding 6). |
| W4 | "No interface is duplicated" and the completeness proof (DESIGN §10.1). | **Wrong** for the `e_min` threshold, and incomplete for the interfaces in finding 20. |
| W5 | "Any difference between the rule-based baseline and IPPO comes from the decision role alone" (ARCHITECTURE §6.2). | **Wrong** while `service_health` is a near-deterministic proxy the baseline ignores (finding 11). |
| W6 | `healthy_disruption` measures harm done by the defense. | **Wrong** as defined (finding 10). |
| W7 | The demo `healthy_disruption_cost` of 2.125 and the CONTRACTS example ids are final. | **Unverified.** They change under findings 10 and 14. |
| W8 | The calibration gate (at least 100 committing proposals in `cal` and `eval`) is reachable with 20 `cal` episodes. | **Unverified** (finding 32). |
| W9 | Entropy-based uncertainty will pass the AUROC gate. | **Unverified**, and the design says so; reported as a risk in RL_IMPLEMENTATION §10 item 4. |
| W10 | Reward magnitudes "range from about 0 to a few tens of negative units" on the mock (RL_IMPLEMENTATION §4.2). | **Unverified**; no run exists. |
| W11 | A self-written IPPO passes the toy-task and GAE tests; `torch.load(weights_only=True)` accepts the optimizer states; `torch.use_deterministic_algorithms(True)` raises no error on the used ops. | **Unverified**; nothing is implemented. Dry-run dependency resolution proves only that a solution exists. |
| W12 | RESEARCH Table 1 scores for CyMARL and the thesis figures. | **Not re-checked by me.** I had no PDF text extraction tool; RESEARCH tags them REPORTED. |
| W13 | Cited URLs I could not check in this session (network timeouts): the two `raw.githubusercontent.com` files, the SB3 and sb3-contrib documentation pages, the JAIST lab publications page, the dspace PDF, `hdl.handle.net` (RESEARCH records 503), ScienceDirect (RESEARCH records 403), the Crossref DOI endpoints, the PyPI `torch`/`torchrl` JSON, npm `react`/`vite`, `github.com/cyb3rlab`, the Cytoscape.js and dash-cytoscape repositories, and the PyTorch LTS index. | **Unverified by me.** |
| W14 | The 650 MiB free-space figure repeated in DECISIONS, RESEARCH, `research-delta.md` and DESIGN §1 rule 8. | **Stale** (2.6 GiB now). It is time-dependent, so S1 must measure (finding 28). |
