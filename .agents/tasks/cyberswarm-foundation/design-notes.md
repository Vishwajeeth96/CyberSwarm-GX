# Design working notes (step: architecture and contracts design)

Status: in progress. These notes let the design be written incrementally. Final deliverables are the five documents named in the step prompt.

## Source conflict (resolved)
- Step prompt says: graph pipeline NOT mandatory, NetworkX optional.
- Current on-disk DELTA.md (19:49, 18,005 bytes) says: graph pipeline REQUIRED, lead-owned, built before RL env; learned GNN staged; MARL primary.
- research-delta.md section 6b and DECISIONS AD-01 already follow the on-disk DELTA.md.
- Decision: follow the on-disk DELTA.md (it is the stated authority). Keep graph code isolated (own package, own contracts module, own tests, own stage part S3a) so it is cheap to drop. Learned GNN: design-time NO-GO for S1..S9. Flag in final message.

## Layout decision
src layout, package `cyberswarm` under /Users/guru/CyberSwarm-GX/src/cyberswarm/.
Subpackages (owner): contracts (lead, neutral), interfaces (lead, neutral: Simulator Protocol + conformance), common (lead), mock (lead, MOCK), roles (lead: detection, evidence), graph (lead), coordinator (lead), runner (lead: engine, smoke, export), rl (lead: spaces, observation, reward, env), baselines (lead), marl (lead: IPPO; MAPPO deferred, no code), uncertainty (lead), evaluation (lead).
Raksha: src/cyberswarm/simulator/ + tests/simulator/ + configs/scenarios/simulator/ (not created).
Mayukh: dashboard/ (not created).
Neutral lead-owned: schemas/, fixtures/, configs/, scripts/, tests/ (per component).

## Roles
Detection = deterministic component (DetectionRole). Investigation = actions chosen by policy + deterministic evidence view. Decision = one policy per zone agent (rule or IPPO actor). Coordination = deterministic coordinator (safety boundary). Agents: agent-dmz, agent-core (one per zone).

## Step semantics
Step t phases: P0 dynamics -> P1 observe (+graph conversion) -> P2 propose -> P3 validate -> P4 execute (+outcome events). All records of step t carry step=t. Reward for step t computed after P4. end_step() then advances to t+1 and runs P0. Max steps H; steps 0..H-1; truncated when t=H-1 ends without termination. Terminated: objective_reached (node-db-01 compromised at end of step) or incident_resolved (all nodes up and none compromised after incident started).

## Contracts
pydantic v2, extra=forbid, schema_version "1.0", JSON Schema 2020-12 exported to /schemas, JSON/JSONL on disk.
IDs: node-*, edge-NNN, agent-*, zone-*, topo-*-vN, scn-*, run-*, ep-<policy slug>-<scenario slug>-s<seed>, pol-*, evt-/prop-/val-/exe-/err- + 6-digit seq (unique within episode), obs-/avl-/det-/gsn-/gft- + step:04d + agent id.
Action vocabulary (8): no_op, monitor_node(duration_steps 1..5), investigate_node, request_evidence, rate_limit_node(level light|heavy), quarantine_node, isolate_node, restore_node. High-impact: isolate_node, restore_node. Containment family (false isolation): quarantine, isolate, restore.
Reason codes: SCHEMA_INVALID, UNKNOWN_ACTION_KIND, ACTION_NOT_ALLOWLISTED, UNKNOWN_TARGET_NODE, TARGET_REQUIRED, TARGET_NOT_ALLOWED, PARAMETER_INVALID, AGENT_UNKNOWN, TARGET_OUT_OF_AUTHORITY, ACTION_NOT_AVAILABLE, WRONG_STEP, EPISODE_MISMATCH, DUPLICATE_PROPOSAL, CONFLICTING_PROPOSAL, HIGH_IMPACT_BUDGET_EXCEEDED, EPISODE_NOT_ACTIVE.
Records: Node, Edge, Topology, ScenarioConfig, EventRecord, AgentObservation, AvailableActions, DetectionSummary, PolicyProposal, ValidationResult, ExecutedAction, EpisodeResult, EvaluationMetrics, ComparisonRecord, ErrorRecord, LogRecord, GroundTruthSnapshot (evaluator-only), GraphSnapshot, GraphFeatureRecord, GraphModelOutput (optional, not computed), RunManifest.
Graph: snapshot base node features = conventional per-node features (11): criticality, 5 status one-hots (up, rate_limited, quarantined, isolated, restoring), alert_count_norm, max_alert_severity_norm, evidence_level_norm, service_health, steps_since_investigated_norm. Edge features (6). Derived 8: degree_norm, neighbor_crit_mean, neighbor_crit_max, dependents_crit_norm, neighbor_alert_mean, neighbor_alert_max, hop_to_alerted_norm, blocked_edge_frac.
RL vector per node: 11 base + suspicion + observed + 3 missing flags = 16; graph mode +8 derived. dims for N=8: conv 130, graph 194.

## Mock world
8 nodes, 2 zones. dmz: node-gw-01 (0.6), node-web-01 (0.5), node-web-02 (0.2). core: node-app-01 (0.7), node-db-01 (1.0), node-auth-01 (0.8), node-file-01 (0.3), node-ws-01 (0.1). 8 link pairs (16 directed) + 3 dependency edges (web01->app01, app01->db01, app01->auth01) = 19 edges.
Splits: train 6 scenarios, cal 2, eval 4 (disjoint ids and seeds).

## Reward (shared team reward)
Terms: attack_impact (-w*sum crit of compromised), response_delay (-w*min(d,dmax)/dmax while uncontained compromise), containment (+w per newly contained truly compromised node), false_isolation (-w*(0.5+crit) on quarantine/isolate/restore of healthy node), healthy_disruption (-w*sum crit*lost availability on healthy nodes), investigation_cost (-w*cost units: monitor 0.5, investigate 1, request_evidence 3, w=0.05), invalid_action guard (-0.1). Terminal: unrealized-impact charge on objective_reached. Defaults: impact 1.0, delay 0.5 (dmax 20), contain 3.0, false 2.0, disrupt 0.5, invest 0.05.

## IPPO
Self-written compact PyTorch, per-agent actor and critic MLP 64x64 tanh, no sharing, local obs only. MAPPO: DEFERRED with full design (centralized critic over concatenated agent-visible obs vectors, no ground truth, shared team reward).

## Uncertainty
Signal: normalized entropy over masked valid actions. Rules R3 (strong evidence allows high-impact), R2 (defer high-impact when u>=tau_defer and evidence<3), R1 (evidence request when u>=tau_evidence and evidence<2 for targeted impact>=medium). Matched control = same weights, rule disabled. Calibration on separate cal split; ECE + reliability + AUROC gate.

## Smoke command
.venv/bin/python -m cyberswarm.runner.smoke --config configs/experiments/smoke.toml

## Output
runs/<run_id>/ run_manifest.json, topologies/, scenarios/, episodes/<episode_id>/{events,observations,available_actions,detection,graph_snapshots,graph_features,proposals,validations,executions,errors}.jsonl + episode_result.json + evaluator_only/ground_truth.jsonl, evaluation/{metrics.jsonl,comparison.json}, logs/run.log.jsonl.

## Writing order
1. CONTRACTS.md  2. ARCHITECTURE.md  3. RL_IMPLEMENTATION.md  4. DECISIONS.md append  5. DESIGN.md  6. cross-check pass.

## Progress log
- [done] docs/CONTRACTS.md written (778 lines, 21 JSON examples parse). Sections: 1 rules, 2 serialization, 3 types, 4 ids, 5 step semantics, 6 action vocabulary, 7 topology/scenario, 8 decision-loop records (8.6 reason codes), 9 ground truth, 10 graph, 11 episode/eval/error/log/manifest (11.4 ErrorCode), 12 cross-record rules X-1..X-14, 13 compat, 14 fixtures, 15 output layout, 16 producer/consumer.
- References promised by CONTRACTS.md: ARCHITECTURE section 6 = roles/agents, section 8 = per-component error behavior; RL_IMPLEMENTATION section 4 = reward, section 7 = uncertainty (7.4 ECE); DESIGN section 8 = fixtures list; DECISIONS AD-04..AD-11 (AD-11 = GNN no-go).
- Next: ARCHITECTURE.md (keep v0.1 useful content: principles, ownership, safety), then RL_IMPLEMENTATION.md, DECISIONS append, DESIGN.md.
- [done] ARCHITECTURE.md v0.2 (sections 1-16; refs AD-06 roles, AD-08 IPPO, AD-09 dashboard, AD-10 MAPPO, AD-11 GNN)
- [done] RL_IMPLEMENTATION.md (sections 1-10; 4 reward, 5 IPPO, 6 MAPPO deferred, 7 uncertainty, 8 GNN no-go). Dims: conv 17N+2=138, graph 25N+2=202; actions 22/36. Added CalibrationReport (CONTRACTS 11.7, cbr- ids).
- [done] DECISIONS.md v0.3 AD-04..AD-15 appended + section 7 open items; D-06/D-17/D-19 status updated in place.
- Packages decided late: engine/ (step engine, recorder), runner/ (config, factory, evaluate, smoke, export), evaluation is pure (outcome, metrics, stats, compare, splits, reweight), uncertainty has calibration.
- Next: DESIGN.md (sections: 1 overview/stages ref, 2 tree, 3 module deps, 4 Protocol + module APIs, 5 mock scope, 6 configs, 7 smoke + outputs, 8 fixtures list, 9 DELTA check register + tests, 10 cross-check matrix, 11 stage split S1..S9, 12 acceptance). Then cross-check pass over all docs.
- [done] DESIGN.md written (sections 1-12) with generated tables via gen_design_tables.py; check_design.py passes (286 paths, 0 problems). Inventory source of truth: gen_design_tables.py. Re-run both after any inventory edit.
- Remaining: cross-document consistency pass (grep stale names), final message, send_message success.
