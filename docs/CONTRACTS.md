# CyberSwarm Shared Contracts

**Status:** Version 0.2 of the contract design (schema version `1.0`; revised after the first design review). SPECIFIED, NOT IMPLEMENTED. No Python contract code exists yet; stage S1 in `.agents/tasks/cyberswarm-foundation/DESIGN.md` implements everything below.
**Owner:** project lead. Raksha (simulator) and Mayukh Aryan (dashboard) are consumers and, for the simulator side, producers. Neither edits this file, `src/cyberswarm/contracts/`, `schemas/` or `fixtures/`. Change requests go to the project lead (see "Compatibility policy").
**Audience:** Raksha, Mayukh Aryan, and the lead-owned components (graph pipeline, RL environment, baselines, coordinator, IPPO, uncertainty, evaluator).
**Related:** [ARCHITECTURE.md](ARCHITECTURE.md) (components and data flow), [RL_IMPLEMENTATION.md](RL_IMPLEMENTATION.md) (spaces, reward, IPPO, uncertainty), [DECISIONS.md](DECISIONS.md) (D-03, D-04, AD-04 to AD-19).

All records in this file are simulator-only, abstract data. No record can carry a shell string, a command, a real IP address, a host name, a file path to execute, or a free-form payload.

## 1. Rules that apply to every schema

| # | Rule |
|---|------|
| C1 | One serialization format: pydantic v2 models in the neutral package `cyberswarm.contracts` are the single source of truth. JSON Schema (Draft 2020-12) is exported from them into `schemas/` at the repository root (every path in this repository's configs and documents is relative to the repository root, so the files work on any contributor's machine). Records are stored on disk as UTF-8 JSON (one document) or JSON Lines (one record per line). Section 2 gives the justification. |
| C2 | Every top-level record has `schema_version` (string, `"1.0"`). Nested objects (for example `Node`, `Edge`, `ActionSpec`) inherit the version of the record that contains them. |
| C3 | `extra="forbid"` on every model. An unknown field is a validation error, not ignored. This is how ground-truth fields are kept out of agent-visible records: the field simply does not exist in those schemas. |
| C4 | Optional values that no producer computed are `null` (or absent on input). They are never replaced by a default number such as `0.0`. Writers always write every declared field, using `null` for absent optional ones. Readers accept absent or `null`. |
| C5 | Floats must be finite (`allow_inf_nan=False`). NaN and Infinity are rejected on read and never written. |
| C6 | Agent-visible records contain no evaluator-only ground truth and no event with `step` greater than the record's own `step`. Evaluator-only data lives in `GroundTruthSnapshot` (section 9) and in the `evaluator_only/` directory only. |
| C7 | Identifiers follow section 4. Unknown or dangling references are validation errors at the point where the reference can be resolved (section 12). |
| C8 | Text fields are `ShortText`: 1 to 160 characters from `[A-Za-z0-9 .,()_-]` only. In addition a validator rejects a dotted quad (regex `\b\d{1,3}(\.\d{1,3}){3}\b`), the sequence `..`, and a value whose first character is `-`. Slashes, colons, equals signs, plus signs, quotes, backticks, pipes, ampersands, semicolons, dollar signs, angle brackets and backslashes cannot occur, so an IP address, a path (`../etc/passwd`), a URL or a shell line (`rm -rf /`) cannot be stored in any text field. No component interprets or executes text. |
| C9 | Wall-clock time appears only in `LogRecord.ts` and is excluded from every determinism comparison. All other timestamps are simulation steps. |
| C10 | Every record that reports a measurement or comparison carries `provenance` (section 3). Fixture values are labelled synthetic and are never results. |
| C11 | Validation behavior on read: a record that fails validation is rejected whole with `ContractError` (`SCHEMA_INVALID`, field path and reason). Nothing is partially accepted, repaired or defaulted. A producer that emits an invalid agent-visible record aborts its episode (ARCHITECTURE section 8). A consumer that finds an invalid file (the dashboard, the evaluator) reports the file and line as malformed and does not guess a value. |

## 2. Serialization decision (pydantic v2, JSON Schema export, JSON/JSONL)

**Decision (D-03, confirmed by this design):** typed pydantic v2 models in `src/cyberswarm/contracts/`, exported JSON Schema files in `schemas/`, JSON and JSON Lines on disk.

Reasons:

1. One definition produces Python types, runtime validation and the language-neutral schema. A hand-written schema next to dataclasses would drift.
2. `extra="forbid"`, closed `Literal` enums, regex-constrained identifiers and discriminated validators turn "no free-form payloads, no shell strings, no raw IPs, no ground truth in observations" into validation rules that tests can exercise.
3. Mayukh's dashboard and Raksha's tests consume the exported schema files (Ajv for JavaScript, `jsonschema` for Python). Neither has to import the lead's Python package.
4. JSON and JSON Lines need no server or database, are append-only and replayable, and can be opened in a browser.
5. Rejected alternatives: dataclasses plus hand-written JSON Schema (drift), Protocol Buffers (code generation, binary files), msgspec and attrs (no schema export evaluated), Parquet and SQLite (no databases or binary stores in the first version).

Canonical form (required so that determinism tests can compare bytes):

| Item | Rule |
|------|------|
| Encoding | UTF-8, `ensure_ascii=True`, no BOM, `\n` line endings. |
| JSONL | One record per line, no blank lines, trailing `\n` after the last record. Separators `,` and `:` with no spaces. |
| Field order | The declaration order of the pydantic model (`model_dump(mode="json")` order). |
| Floats | Python `repr` (shortest round-trip). Tensor values are produced as `float(numpy.float32(x))`, so a float32 value round-trips exactly. |
| Integers | JSON integers. Booleans are JSON `true` and `false`, never `0` and `1`. |
| Enums | Lowercase snake_case strings, listed in this file. |
| Matrices | Nested arrays, row-major. `node_features[i][j]` is node `i` (in `node_order`), feature `j` (in `node_feature_names`). |
| Export | `python -m cyberswarm.contracts.export_schemas` writes `schemas/<RecordName>.schema.json`, adds `"$schema": "https://json-schema.org/draft/2020-12/schema"` and `"$id": "urn:cyberswarm:schema:<RecordName>:1.0"`. A test fails if the committed files differ from a fresh export. |

Documentation examples in this file are marked with the fence info string `json contract=<RecordName>`. They omit null-valued optional fields for readability; on-disk records include them as `null`. `tests/docs/test_contract_examples.py` (stage S1) validates every marked example in this file against the exported schema, and `scripts/validate_docs.py` (stage S9) additionally checks links, paths and commands.

## 3. Common types

| Type | Definition |
|------|------------|
| `Step` | Integer, 0 to 10000. |
| `UnitFloat` | Finite float in [0.0, 1.0]. |
| `ShortText` | See C8. |
| `RelPath` | Relative POSIX path of at most 200 characters, segments from `[A-Za-z0-9._-]+` joined by `/`, no segment equal to `..` or starting with `.` followed by `.`, no leading `/`. Used only for `RunManifest.files[].path` and `RunManifest.episodes[].directory`. It is a reference, never opened as a command. |
| `NodeStatus` | `up`, `rate_limited`, `quarantined`, `isolated`, `restoring`. The status an agent can see because it follows from actions taken. It is not a compromise indicator. |
| `ActionKind` | `no_op`, `monitor_node`, `investigate_node`, `request_evidence`, `rate_limit_node`, `quarantine_node`, `isolate_node`, `restore_node` (section 6). |
| `ImpactClass` | `none`, `low`, `medium`, `high`. |
| `Representation` | `conventional`, `graph_features`, `learned_embedding`. The third value names the observation mode of the staged learned-graph experiment (RL_IMPLEMENTATION section 8). It is part of version 1.0 so that adding that experiment is not a contract change; no record carries it unless that stage ran. |
| `Split` | `train`, `cal`, `eval`, `eval_utopo`, `dev`, `fixture`. `cal` is the held-out split used only for threshold selection, calibration and (GNN stage) model selection; it is never reported as a result. `eval` holds held-out scenarios on the training topology. `eval_utopo` holds held-out scenarios on the unseen topology variant (same node ids and zones, different links, dependencies and criticalities); nothing on that topology is used for training, selection or calibration. |
| `ComponentStatus` | `not_run`, `implemented`, `no_go`, `deferred`. Used by `RunManifest.status` and `StageDecision` (sections 11.6 and 11.9). The allowed subset depends on the component. |
| `Provenance` | Object: `kind` (required: `fixture_synthetic`, `mock_simulator_run` or `simulator_run`) and `note` (optional `ShortText`). `fixture_synthetic` means the values are illustrative; `mock_simulator_run` means produced by the lead-owned MOCK simulator and is not a research result; `simulator_run` is reserved for Raksha's real simulator. |
| `ReasonCode` | Coordinator rejection codes (section 8.6). |
| `ErrorCode` | Section 11.4. |

## 4. Identifiers

All identifiers are lowercase ASCII. Slugs match `[a-z0-9]+(-[a-z0-9]+)*`.

| ID | Pattern | Example | Stability and producer |
|----|---------|---------|------------------------|
| `NodeId` | `^node-<slug>$`, at most 40 characters. A value whose remainder after `node-` is four numeric groups (for example `node-10-0-0-1`) is rejected as an address look-alike. | `node-web-02` | Stable for the life of a topology and across episodes. Assigned in the topology file. |
| `EdgeId` | `^edge-[0-9]{3,4}$` | `edge-014` | Assigned in the topology file in ascending order of `(source, target, edge_type)`. Unique within a topology. |
| `ZoneId` | `^zone-<slug>$` | `zone-dmz` | Part of the topology. |
| `AgentId` | `^agent-<slug>$`. Derived: `agent-` plus the zone slug. | `agent-dmz` | One decision agent per zone (ARCHITECTURE section 6). |
| `TopologyId` | `^topo-<slug>-v[0-9]+$` | `topo-mock-small-v1` | A changed topology gets a new id. |
| `ScenarioId` | `^scn-<slug>$` | `scn-mock-lateral-eval-01` | Assigned in the scenario file. |
| `RunId` | `^run-<slug>$` | `run-smoke` | Given by the run configuration. No timestamps, so runs are reproducible. |
| `PolicyId` | `^pol-<slug>$` | `pol-rule-conv-v1`, `pol-ippo-conv-s0` | Assigned by the policy factory. One per trained policy (the training seed is part of the slug). |
| `PolicyFamilyId` | `^fam-<slug>$` | `fam-rule-conv`, `fam-ippo-conv` | The comparison arm: a policy type plus its observation representation, independent of training seed. Several `PolicyId` values (one per training seed) share one family. |
| `EpisodeId` | `^ep-<policy slug>-<scenario slug>-s<seed>$`, at most 120 characters. | `ep-rule-conv-v1-mock-lateral-eval-01-s2001` | Deterministic from policy, scenario and seed. Scenario and seed are the matched unit for comparisons. |
| `EventId` | `^(evt\|cev)-[0-9]{6}$` | `evt-000007`, `cev-000001` | `evt-` for events produced by the simulator, `cev-` for events produced by the coordinator (recovery verification). Each prefix has its own per-episode counter starting at 1, assigned by its producer in emission order. |
| `ProposalId` | `^prop-[0-9]{6}$` | `prop-000012` | Per-episode counter, assigned by the engine when it collects the proposals of a step, in ascending `agent_id` order. |
| `ValidationId` | `^val-[0-9]{6}$` | `val-000012` | Per-episode counter, assigned by the coordinator in processing order (ascending `agent_id` within a step). The n-th proposal gets the n-th validation id. |
| `ExecutionId` | `^exe-[0-9]{6}$` | `exe-000009` | Per-episode counter, assigned by the coordinator, in execution order (ascending `agent_id` within a step), only for accepted proposals. |
| `ErrorId` | `^err-[0-9]{6}$` | `err-000001` | Per-episode (or per-run) counter. |
| Derived per-step ids | `obs-<step:04d>-<AgentId>`, `avl-...`, `det-...`, `gsn-...` (graph snapshot), `gft-...` (graph features), `gmo-...` (graph model output) | `obs-0005-agent-dmz` | Deterministic from step and agent. |
| `NodeScoreId` | `nsc-<step:04d>-<AgentId>-<scorer slug>` | `nsc-0005-agent-dmz-detection-rule-v1` | Deterministic from step, agent and `scorer_id` slug. |
| `MetricsId`, `AggregateId`, `ComparisonId`, `MatchedSetId`, `CalibrationReportId`, `DecisionId` | `met-<slug>`, `agg-<slug>`, `cmp-<slug>`, `set-<slug>`, `cbr-<slug>`, `dec-<slug>` | `set-eval-v1` | Assigned by the evaluator (`dec-` by the stage report author for `StageDecision`). |

Counters of one kind (and, for events, one prefix) are contiguous (`000001`, `000002`, ...) within one episode, so a gap is detectable. The pair (`episode_id`, id) is globally unique. Nothing in an identifier encodes ground truth.

Node order. The canonical node order is ascending lexicographic `node_id`. The canonical edge order is ascending `edge_id`. Both are stable by construction and are used by every tensor.

## 5. Simulation-step semantics

A simulation step is a non-negative integer `t`. An episode of `max_steps = H` has steps `0 .. H-1`.

**Step t starts** when the simulator has completed the transition from step `t-1` (or `reset` for `t = 0`) and ends when `end_step()` returns. Inside step `t` there are exactly six phases, always in this order:

| Phase | Name | Who | What happens | Records produced (all carry `step = t`) |
|-------|------|-----|--------------|-----------------------------------------|
| P0 | `dynamics` | Simulator | Exogenous dynamics for step `t`: normal activity, attacker progress, alert generation, restore completion. Skipped for `t = 0` except for `episode_started`. | `EventRecord` with `phase = dynamics` or `lifecycle` |
| P1 | `observe` | Simulator, then graph pipeline and detection role | Build agent observations from the state after P0, with events whose `step <= t` that are visible to the agent. Then deterministic conversion to graph records and detection summary. | `AgentObservation`, `AvailableActions`, `DetectionSummary`, `GraphSnapshot`, `GraphFeatureRecord` |
| P2 | `propose` | Policies (decision role) | Each agent proposes exactly one action using only its P1 records. | `PolicyProposal` |
| P3 | `validate` | Coordinator | Validate each proposal. Rejected proposals change nothing. | `ValidationResult` |
| P4 | `execute` | Coordinator, through the simulator's public API | Apply accepted actions in ascending `agent_id` order. Effects apply immediately. Outcome events are produced. Recovery verification may add events. | `ExecutedAction`, `EventRecord` with `phase = outcome` |
| end | `end_step()` | Simulator | (1) Capture the end-of-step-`t` evaluator snapshot. (2) Evaluate termination and truncation on that state. (3) If not done, advance to `t+1` and run its P0. The order is fixed: the snapshot is taken before P0 of `t+1` (rule 9). | `episode_ended` event (only on the last step, `phase = lifecycle`) |

Rules:

1. Every record produced during step `t` has `step = t`, including outcome events caused by actions executed in step `t`. An action never changes the observation of the step in which it was proposed. Its effect is first visible in the observation of step `t+1` (through the outcome events and changed state).
2. An observation at step `t` contains only events with `step <= t`. Events with `step > t` do not exist yet. This is the "no future events" rule and it is tested.
3. A proposal, validation or execution whose `step` differs from the current step is rejected (`WRONG_STEP`).
4. Total order of all records of an episode: ascending `(step, phase_order, sequence)` where `phase_order` is P0 = 0, P1 = 1, P2 = 2, P3 = 3, P4 = 4 and `sequence` is the id counter or agent order.
5. Reward for step `t` is computed after P4 from the end-of-step state, and is delivered with the observation of step `t+1`. It is a training signal, not a record in any agent-visible file (see RL_IMPLEMENTATION section 4).
6. **Termination** is evaluated at the end of step `t`: `terminated` when the objective node is compromised (`objective_reached`) or when the incident has started, no node is compromised and every node is `up` (`incident_resolved`). **Truncation** is `t == H-1` with no termination. Both flags are visible to agents after the last decision. The termination reason is not (it is evaluator data in `EpisodeResult`).
7. The incident cannot start before step 1, so step 0 always shows a healthy network.
8. After the last step has ended (`terminated` or `truncated`), `observe()` may still be called once to obtain a **final observation** of the end-of-step state, with `step = last step + 1`. It exists only so that a learner can bootstrap a value estimate for truncated episodes. It is never written to a record file, it has no matching proposals, and it is excluded from rule X-3. Its availability mask allows `no_op` only (RL_IMPLEMENTATION section 2.2).
9. **Ground-truth timing.** `end_step()` captures the evaluator snapshot of the closed step `t` before it runs P0 of `t+1`. `GroundTruthAccessor.ground_truth()` returns that captured snapshot, whose `step` equals `StepResult.step_closed`, until the next `end_step()` call. `termination_reason()` refers to the same closed step. So an event that fires at P0 of `t+1` (for example `restore_completed`) is not yet reflected in the snapshot of `t`. The reward of step `t`, the false-isolation flag, the containment step and the `GroundTruthSnapshot.step` all use this snapshot. Conformance check SC-14 tests it.

## 6. Action vocabulary (single, closed, simulator-only)

This is the only action vocabulary. The coordinator allowlist, the RL action table, the baselines, the dashboard and Raksha's simulator all use it. Adding an action is a MINOR contract change (section 13).

| `action_kind` | Target | Parameters | Impact class | High-impact | `cost_units` | Simulated effect (abstract) |
|---------------|--------|------------|--------------|-------------|--------------|-----------------------------|
| `no_op` | none | none | `none` | no | 0 | Nothing. |
| `monitor_node` | node | `duration_steps` int 1 to 5, default 3 | `low` | no | 0.5 | Sustained better observation of the node for the duration (higher alert detection, fewer false alerts, less noisy health). |
| `investigate_node` | node | none | `low` | no | 1.0 | One-shot evidence reading for the node (`investigation_result` event, `evidence_level` 0 to 3, noisy). |
| `request_evidence` | node | none | `low` | no | 3.0 | One-shot deeper evidence reading with higher accuracy than `investigate_node`. This is the action the uncertainty mechanism uses to ask for more evidence. |
| `rate_limit_node` | node | `level` in `light`, `heavy`, default `light` | `medium` | no | 0 | Node stays reachable at reduced capacity and spreads an incident less. |
| `quarantine_node` | node | none | `medium` | no | 0 | Node's outbound edges are blocked. Inbound still works. Service partly available. |
| `isolate_node` | node | none | `high` | **yes** | 0 | All of the node's edges are blocked. Its service is unavailable. Lasts until restored. |
| `restore_node` | node | none | `high` | **yes** | 0 | Node returns to a clean state from simulated backup after `restore_duration_steps` steps of downtime. Clears isolation and quarantine. |

Definitions used elsewhere:

- High-impact actions: `isolate_node`, `restore_node` (the actions with `impact_class = high`).
- Containment-class actions (used by the false-isolation reward term and metric): `quarantine_node`, `isolate_node`, `restore_node`.
- Information-gathering actions (the investigation cost term): `monitor_node`, `investigate_node`, `request_evidence`.
- Fixed action order for tables and masks: the order of the rows above, then ascending `node_id`.

`ActionSpec` (nested in proposals, validation results and executions):

| Field | Type | Required | Rule |
|-------|------|----------|------|
| `action_kind` | `ActionKind` | yes | Unknown value: `UNKNOWN_ACTION_KIND`. |
| `target_node_id` | `NodeId` or null | yes | Required (non-null) for every kind except `no_op`; must be null for `no_op` (`TARGET_REQUIRED`, `TARGET_NOT_ALLOWED`). |
| `parameters` | `ActionParameters` | yes (may be `{}`) | `duration_steps` (int 1 to 5 or null) is allowed only for `monitor_node`; `level` (`light`, `heavy` or null) only for `rate_limit_node`. Anything else: `PARAMETER_INVALID`. The coordinator fills defaults in `resolved_action`. |

There is no field that can hold a command, address, path or script. The RL environment and the baselines use default parameters only.

The mapping of these abstract actions to Raksha's simulator internals is Raksha's to implement; the effects column is the behaviour her simulator must provide (see MODULE_RAKSHA_SIMULATOR.md, published after stage S2).

Example `ActionSpec` (a `monitor_node` action with an explicit parameter):

```json contract=ActionSpec
{"action_kind":"monitor_node","target_node_id":"node-web-01","parameters":{"duration_steps":3,"level":null}}
```

## 7. Static records: topology and scenario

### 7.1 `Node` (network node)

Producer: topology file author (lead for the mock topology, Raksha for simulator topologies). Consumers: simulator, graph pipeline, RL environment, baselines, coordinator, evaluator, dashboard.

| Field | Type | Req | Units, range, allowed values |
|-------|------|-----|------------------------------|
| `node_id` | `NodeId` | yes | Section 4. |
| `node_type` | enum | yes | `gateway`, `web`, `app`, `database`, `auth`, `file`, `workstation`. |
| `zone_id` | `ZoneId` | yes | The zone whose decision agent has authority over this node. |
| `criticality` | `UnitFloat` | yes | Business value of the service on the node. 0.0 = negligible, 1.0 = highest. Dimensionless. Public information (agents may use it). |
| `label` | `ShortText` or null | no | Human-readable name for the dashboard. |

Validation: unknown fields, unknown `node_type`, a `criticality` outside [0, 1] or a malformed id are rejected. Example:

```json contract=Node
{"node_id":"node-web-02","node_type":"web","zone_id":"zone-dmz","criticality":0.2,"label":"Public web server B"}
```

### 7.2 `Edge` (network edge)

| Field | Type | Req | Meaning |
|-------|------|-----|---------|
| `edge_id` | `EdgeId` | yes | Section 4. |
| `source` | `NodeId` | yes | Edge start. |
| `target` | `NodeId` | yes | Edge end. Must differ from `source`. |
| `edge_type` | enum | yes | `network_link`: traffic can flow from `source` to `target`. A physical link is stored as two edges, one per direction. `service_dependency`: the service on `source` depends on the service on `target`. |

Edges are always directed. Dynamic edge state is carried in observations, not in the topology. Validation: `source` and `target` must exist in the enclosing `Topology`; self-loops and duplicates of `(source, target, edge_type)` are rejected. Example:

```json contract=Edge
{"edge_id":"edge-014","source":"node-web-01","target":"node-app-01","edge_type":"network_link"}
```

### 7.3 `Topology`

Document stored at `configs/topologies/<topology_id>.json`; copied into every run output.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version` | const `"1.0"` | yes | |
| `topology_id` | `TopologyId` | yes | |
| `nodes` | list of `Node` | yes | 1 to 64 items, sorted ascending by `node_id`, ids unique. |
| `edges` | list of `Edge` | yes | Sorted ascending by `edge_id`. `edge_id` numbers equal the rank of `(source, target, edge_type)` in ascending order, starting at `edge-001`. Every endpoint exists. The `service_dependency` edges form a directed acyclic graph. |

Additional validation: `nodes` has at least 2 items (graph features divide by `N - 1`), and the `network_link` edges, read as undirected, connect all nodes (a hop distance and a diameter exist for every pair). A disconnected or single-node topology is rejected with `SCHEMA_INVALID`.

Derived, not stored: the zones (set of `zone_id`), the agents (`agent-<zone slug>`, one per zone, each zone has at least one node), and the authority of an agent (the nodes of its zone). The reference mock topology `topo-mock-small-v1` (8 nodes, 2 zones, 19 edges) and its unseen variant `topo-mock-small-v2` (same node ids and zones, different links, dependencies and criticalities) are specified in DESIGN.md section 5.1.

Example `Topology` (a three-node excerpt, valid on its own):

```json contract=Topology
{"schema_version":"1.0","topology_id":"topo-example-v1","nodes":[{"node_id":"node-app-01","node_type":"app","zone_id":"zone-core","criticality":0.7,"label":"Application server"},{"node_id":"node-db-01","node_type":"database","zone_id":"zone-core","criticality":1.0,"label":null},{"node_id":"node-web-01","node_type":"web","zone_id":"zone-dmz","criticality":0.5,"label":null}],"edges":[{"edge_id":"edge-001","source":"node-app-01","target":"node-db-01","edge_type":"network_link"},{"edge_id":"edge-002","source":"node-app-01","target":"node-db-01","edge_type":"service_dependency"},{"edge_id":"edge-003","source":"node-app-01","target":"node-web-01","edge_type":"network_link"},{"edge_id":"edge-004","source":"node-db-01","target":"node-app-01","edge_type":"network_link"},{"edge_id":"edge-005","source":"node-web-01","target":"node-app-01","edge_type":"network_link"},{"edge_id":"edge-006","source":"node-web-01","target":"node-app-01","edge_type":"service_dependency"}]}
```

### 7.4 `ScenarioConfig` (scenario configuration)

Producer: scenario author. Consumers: simulator (`reset`), runner, evaluator, dashboard. Every field is a requirement on any simulator: a simulator that cannot honour a field must refuse the scenario with `SCENARIO_UNSUPPORTED` instead of ignoring it.

| Field | Type | Req | Units, range |
|-------|------|-----|--------------|
| `schema_version` | const | yes | |
| `scenario_id` | `ScenarioId` | yes | |
| `topology_id` | `TopologyId` | yes | Must resolve to a loaded topology. |
| `description` | `ShortText` or null | no | |
| `max_steps` | int | yes | 5 to 500. Horizon `H` (steps). |
| `observation_window_steps` | int | yes | 1 to 20. How many past steps of events an observation shows. |
| `incident.incident_type` | const `propagating_compromise` | yes | The single abstract incident type of version 1.0: a node enters a compromised state and the state may spread along `network_link` edges. State transitions only. |
| `incident.entry_node_id` | `NodeId` | yes | Must exist. |
| `incident.start_step` | int | yes | 1 to `max_steps - 1`. |
| `incident.spread_probability` | `UnitFloat` | yes | Per compromised node, per outgoing `network_link` edge, per step. |
| `observability.alert_detect_probability` | `UnitFloat` | yes | Chance per step that a compromised node raises an alert. |
| `observability.false_alert_rate` | `UnitFloat` | yes | Chance per step that a healthy node raises a false alert. |
| `observability.alert_delay_max_steps` | int | yes | 0 to 5. Alerts are delayed by 0 to this many steps. |
| `observability.missing_value_rate` | `UnitFloat` | yes | Chance that an observable measurement (`service_health`, `traffic_level`) is reported as null. |
| `restore_duration_steps` | int | yes | 1 to 10. Downtime of `restore_node`. |
| `objective_node_id` | `NodeId` or null | yes | If set and compromised at the end of a step, the episode terminates with `objective_reached`. |

Validation: unknown fields, an unresolved node, `start_step >= max_steps`, or an unsupported `incident_type` are rejected. The seed is not part of the scenario; it is passed to `reset(scenario, seed)`. Example:

```json contract=ScenarioConfig
{"schema_version":"1.0","scenario_id":"scn-mock-lateral-eval-01","topology_id":"topo-mock-small-v1","description":"Synthetic propagating incident entering at node-web-02","max_steps":40,"observation_window_steps":5,"incident":{"incident_type":"propagating_compromise","entry_node_id":"node-web-02","start_step":3,"spread_probability":0.25},"observability":{"alert_detect_probability":0.35,"false_alert_rate":0.02,"alert_delay_max_steps":1,"missing_value_rate":0.05},"restore_duration_steps":3,"objective_node_id":"node-db-01"}
```

## 8. Decision-loop records

### 8.1 `EventRecord`

Agent-visible events only. Producer: the simulator (`producer = simulator`) and, for recovery verification only, the coordinator (`producer = coordinator`). Consumers: observation builder, graph pipeline, evaluator, dashboard. Stored in `events.jsonl`.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version` | const | yes | |
| `event_id` | `EventId` | yes | |
| `episode_id` | `EpisodeId` | yes | |
| `step` | `Step` | yes | The step in which the event was produced (section 5). |
| `phase` | enum | yes | `dynamics` (P0), `outcome` (P4) or `lifecycle`. |
| `event_type` | enum | yes | See the table below. |
| `producer` | enum | yes | `simulator` or `coordinator`. |
| `visible_to` | list of `AgentId` | yes | At least one agent, ascending. Set by the producer from the visibility table below. Only these agents receive the event in observations. |
| `node_id` | `NodeId` or null | per type | |
| `agent_id` | `AgentId` or null | per type | The agent whose action caused the event. |
| `severity` | `low`, `medium`, `high` or null | per type | Abstract alert severity. |
| `alert_category` | `traffic_anomaly`, `access_anomaly`, `service_anomaly` or null | per type | Abstract alert class. No technique names. |
| `evidence_level` | int 0 to 3 or null | per type | 0 = clean, 1 = weak, 2 = moderate, 3 = strong. |
| `status_from`, `status_to` | `NodeStatus` or null | per type | |
| `action_kind` | `ActionKind` or null | per type | |
| `proposal_id`, `execution_id` | ids or null | per type | |
| `terminated`, `truncated` | bool or null | per type | |

Required fields by `event_type` (all other optional fields must be null):

| `event_type` | Phase | Producer | Required fields |
|--------------|-------|----------|-----------------|
| `episode_started` | lifecycle | simulator | none beyond the common fields |
| `alert_raised` | dynamics | simulator | `node_id`, `severity`, `alert_category` |
| `restore_completed` | dynamics | simulator | `node_id`, `status_from`, `status_to` |
| `node_status_changed` | outcome | simulator | `node_id`, `status_from`, `status_to`, `execution_id` |
| `action_applied` | outcome | simulator | `action_kind`, `agent_id`, `proposal_id`, `execution_id`; `node_id` unless `no_op` |
| `investigation_result` | outcome | simulator | `node_id`, `evidence_level`, `agent_id`, `execution_id` |
| `recovery_verified` | outcome | coordinator | `node_id`, `execution_id` (the `restore_node` execution); id prefix `cev-` |
| `recovery_failed` | outcome | coordinator | `node_id`, `execution_id` (the `restore_node` execution); id prefix `cev-` |
| `episode_ended` | lifecycle | simulator | `terminated`, `truncated` |

**Visibility rule (who may receive each event).** `visible_to` is a function of the event type, so every simulator produces the same set and the conformance suite (check SC-15) can test it. "Adjacent agent" means an agent that owns a node joined to the event's node by a `network_link` edge in either direction.

| `event_type` | `visible_to` |
|--------------|--------------|
| `episode_started`, `episode_ended` | All agents. |
| `alert_raised`, `node_status_changed`, `restore_completed` | The agent of the node's zone, plus every adjacent agent. |
| `action_applied`, `investigation_result` | Only the acting agent (`agent_id`). An agent never sees another agent's actions or investigation results. |
| `recovery_verified`, `recovery_failed` | The agent of the node's zone. |

**Recovery-verification rule (producer: the coordinator's `RecoveryVerifier`; the simulator knows nothing about it).** Parameters: `health_min = 0.8`, `window = 2`, `verify_timeout_steps = 6`, `restore_wait_steps = restore_duration_steps + 2`.

1. Only executed (`status = executed`) `restore_node` actions are tracked. Quarantine, isolation and rate limiting are not verified (their recovery is the later restore).
2. The verifier is armed at step `s`, the step of the `restore_completed` event for that node. If no `restore_completed` event for the node has arrived `restore_wait_steps` steps after the execution, it emits `recovery_failed` and stops.
3. Once armed, at P4 of every step `t >= s` it reads the node's entry in the observation of step `t` of the node's zone agent. The step counts toward the streak if `visible_status == up` and `service_health` is non-null and `>= health_min`. A null `service_health` or any other status resets the streak to 0 (a missing value is never treated as healthy).
4. When the streak reaches `window` consecutive steps, it emits `recovery_verified` at step `t` (`phase = outcome`, `producer = coordinator`, id `cev-NNNNNN` from the coordinator's own counter, `execution_id` = the restore execution) and stops.
5. If no `recovery_verified` has been emitted by step `s + verify_timeout_steps`, it emits `recovery_failed` at that step and stops.
6. A verifier still armed when the episode ends emits nothing. Both events are visible to the node's zone agent only and use agent-visible data only.

There is deliberately no field for "was this alert a true positive" or "is this node compromised". That information is in `GroundTruthSnapshot.alert_truth`. Example:

```json contract=EventRecord
{"schema_version":"1.0","event_id":"evt-000014","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":5,"phase":"dynamics","event_type":"alert_raised","producer":"simulator","visible_to":["agent-core","agent-dmz"],"node_id":"node-web-02","severity":"medium","alert_category":"traffic_anomaly"}
```

### 8.2 `AgentObservation`

What one decision agent may see at step `t`. Producer: the simulator (`observe()`), built from the state after P0. Consumers: detection role, graph pipeline, observation encoder, baselines, policies, dashboard. Stored in `observations.jsonl`.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `observation_id`, `episode_id`, `step`, `agent_id` | | yes | `observation_id` = `obs-<step:04d>-<agent_id>`. |
| `zone_id` | `ZoneId` | yes | The agent's zone. |
| `max_steps` | int | yes | The known horizon `H`. |
| `observation_window_steps` | int | yes | From the scenario. |
| `nodes` | list of `NodeObservation` | yes | Own-zone nodes (`scope = own`) and out-of-zone nodes joined to an own-zone node by one `network_link` edge in either direction (`scope = boundary`), ascending `node_id`. Other nodes are absent (unobserved), never zero-filled. |
| `edges` | list of `EdgeObservation` | yes | Edges with at least one own-zone endpoint, ascending `edge_id`. |
| `recent_events` | list of `EventRecord` | yes | Events with `agent_id` in `visible_to` and `step - observation_window_steps < event.step <= step`. Ascending `(step, event_id)`. |
| `last_step_feedback` | list of `ProposalFeedback` | yes | The agent's proposals of step `t-1` and what happened to them. Empty at step 0. |

`NodeObservation`: `node_id`; `scope` (`own` or `boundary`); `visible_status` (`NodeStatus`); `alert_count_recent` (int 0 to 50, alerts in the window); `max_alert_severity` (int 0 to 3: none, low, medium, high); `evidence_level` (int 0 to 3 or null; the latest investigation result still valid, null if none); `service_health` (`UnitFloat` or null; measured fraction of nominal service throughput, noisy, null when the measurement is missing); `steps_since_investigated` (int >= 0 or null). For `scope = boundary`, `evidence_level`, `service_health` and `steps_since_investigated` must be null.

`EdgeObservation`: `edge_id`; `edge_state` (`active`, `limited`, `blocked`); `traffic_level` (`UnitFloat` or null).

`ProposalFeedback`: `proposal_id`; `decision` (`accepted` or `rejected`); `reason_codes` (list of `ReasonCode`); `execution_status` (`executed`, `failed` or null). Example (the core agent's injected proposal of step 6, as reported in its observation of step 7):

```json contract=ProposalFeedback
{"proposal_id":"prop-000013","decision":"rejected","reason_codes":["TARGET_OUT_OF_AUTHORITY"],"execution_status":null}
```

Validation: no field outside this list is allowed, so a ground-truth field (for example `compromised`) fails validation by construction. Events after `step`, nodes not in the topology, or boundary nodes with non-null private measurements are rejected. Example (abbreviated to the dmz agent: real records list every edge and every event visible in the window, including the agent's own `action_applied` events):

```json contract=AgentObservation
{"schema_version":"1.0","observation_id":"obs-0005-agent-dmz","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":5,"agent_id":"agent-dmz","zone_id":"zone-dmz","max_steps":40,"observation_window_steps":5,"nodes":[{"node_id":"node-app-01","scope":"boundary","visible_status":"up","alert_count_recent":0,"max_alert_severity":0},{"node_id":"node-gw-01","scope":"own","visible_status":"up","alert_count_recent":1,"max_alert_severity":1,"evidence_level":0,"service_health":0.97,"steps_since_investigated":3},{"node_id":"node-web-01","scope":"own","visible_status":"up","alert_count_recent":0,"max_alert_severity":0,"service_health":0.99},{"node_id":"node-web-02","scope":"own","visible_status":"up","alert_count_recent":1,"max_alert_severity":2,"service_health":0.88}],"edges":[{"edge_id":"edge-014","edge_state":"active","traffic_level":0.52}],"recent_events":[{"schema_version":"1.0","event_id":"evt-000014","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":5,"phase":"dynamics","event_type":"alert_raised","producer":"simulator","visible_to":["agent-core","agent-dmz"],"node_id":"node-web-02","severity":"medium","alert_category":"traffic_anomaly"}],"last_step_feedback":[]}
```

### 8.3 `AvailableActions`

The actions that may be proposed by one agent at step `t`. Producer: the simulator (`available_actions(agent_id)`), because availability depends on simulator state (for example a node that is already isolated cannot be isolated again). Consumers: observation encoder (action mask), baselines, coordinator (state-dependent check), dashboard. Stored in `available_actions.jsonl`.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `available_actions_id` (`avl-<step:04d>-<agent_id>`), `episode_id`, `step`, `agent_id` | | yes | |
| `actions` | list of `AvailableAction` | yes | Fixed action order (section 6). `no_op` is always present and first. Only nodes in the agent's zone appear as targets. |

`AvailableAction`: `action_kind`; `target_node_id` (or null for `no_op`); `parameters` (default parameters only); `impact_class`; `high_impact` (bool); `cost_units` (float >= 0). `impact_class`, `high_impact` and `cost_units` must equal the section 6 table. Example (abbreviated to three entries; real records list every available action):

```json contract=AvailableActions
{"schema_version":"1.0","available_actions_id":"avl-0005-agent-dmz","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":5,"agent_id":"agent-dmz","actions":[{"action_kind":"no_op","target_node_id":null,"parameters":{},"impact_class":"none","high_impact":false,"cost_units":0.0},{"action_kind":"investigate_node","target_node_id":"node-web-02","parameters":{},"impact_class":"low","high_impact":false,"cost_units":1.0},{"action_kind":"isolate_node","target_node_id":"node-web-02","parameters":{},"impact_class":"high","high_impact":true,"cost_units":0.0}]}
```

### 8.4 `DetectionSummary`

Output of the deterministic detection role (ARCHITECTURE section 6). Producer: `cyberswarm.roles.detection`. Consumers: baselines, observation encoder (the `suspicion` feature), dashboard activity feed. Stored in `detection.jsonl`.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `detection_id` (`det-<step:04d>-<agent_id>`), `episode_id`, `step`, `agent_id` | | yes | |
| `producer` | `ShortText` | yes | Component and version, for example `detection.rule-v1`. |
| `nodes` | list of objects | yes | One per `AgentObservation.nodes` entry, same order: `node_id`, `suspicion` (`UnitFloat`), `alerted` (bool: at least one alert in the window). |

`suspicion` = `min(1, 0.5 * alert_count_norm + 0.5 * max_alert_severity_norm)` where `alert_count_norm = min(alert_count_recent, 5) / 5` and `max_alert_severity_norm = max_alert_severity / 3`. It uses only the observation. Example:

```json contract=DetectionSummary
{"schema_version":"1.0","detection_id":"det-0005-agent-dmz","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":5,"agent_id":"agent-dmz","producer":"detection.rule-v1","nodes":[{"node_id":"node-app-01","suspicion":0.0,"alerted":false},{"node_id":"node-web-02","suspicion":0.43333333333333335,"alerted":true}]}
```

### 8.5 `PolicyProposal`

One per agent per step. Producer: the policy runner for a rule-based, graph-feature or learned policy (decision role). Consumers: coordinator, evaluator, dashboard. Stored in `proposals.jsonl`. A proposal is a request, not an action.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `proposal_id`, `episode_id`, `step`, `agent_id` | | yes | |
| `role` | const `decision` | yes | Detection and investigation are not proposers (ARCHITECTURE section 6). |
| `policy_id` | `PolicyId` | yes | |
| `observation_id` | id | yes | The observation this proposal was based on. Must equal `obs-<step>-<agent_id>`. |
| `action` | `ActionSpec` | yes | The final action after any uncertainty rule. |
| `rationale_codes` | list of enum | yes | From: `no_evidence`, `alerts_present`, `evidence_insufficient`, `evidence_moderate`, `evidence_strong`, `containment_ladder`, `recovery_after_containment`, `learned_policy`, `uncertainty_evidence_request`, `uncertainty_deferral`, `strong_evidence_override`, `no_action_needed`. May be empty. |
| `rationale_text` | `ShortText` or null | no | For people. Never parsed. |
| `action_probability` | `UnitFloat` or null | no | Probability the policy assigned to the final action before any uncertainty rule. Null for rule-based policies. |
| `uncertainty` | `UncertaintyInfo` or null | no | Null whenever no uncertainty was computed. |

`UncertaintyInfo`: `method` (const `normalized_entropy`); `value` (`UnitFloat` or null; null if it could not be computed); `rule_id` (`ShortText` or null); `rule_outcome` (`not_applied`, `unavailable_pass_through`, `pass_through`, `evidence_request`, `deferred_high_impact`, `containment_allowed_strong_evidence`); `original_action` (`ActionSpec` or null, set only when the rule replaced the policy's action). Rules and outcomes are defined in RL_IMPLEMENTATION section 7. Example (a high-impact isolation replaced by an evidence request; the numbers are illustrative):

```json contract=UncertaintyInfo
{"method":"normalized_entropy","value":0.62,"rule_id":"uncertainty-rule-v1","rule_outcome":"evidence_request","original_action":{"action_kind":"isolate_node","target_node_id":"node-web-02","parameters":{"duration_steps":null,"level":null}}}
```

Validation: the coordinator re-validates everything; a proposal that does not parse is rejected with `SCHEMA_INVALID` (section 8.6) and never executes. Example:

```json contract=PolicyProposal
{"schema_version":"1.0","proposal_id":"prop-000012","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":5,"agent_id":"agent-dmz","role":"decision","policy_id":"pol-rule-conv-v1","observation_id":"obs-0005-agent-dmz","action":{"action_kind":"investigate_node","target_node_id":"node-web-02","parameters":{}},"rationale_codes":["alerts_present","evidence_insufficient"],"rationale_text":"Alert on node-web-02 with no evidence yet"}
```

### 8.6 `ValidationResult` (coordinator validation result)

One per proposal. Producer: the defense coordinator only. Consumers: policy runner (`last_step_feedback`), evaluator, dashboard. Stored in `validations.jsonl`.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `validation_id`, `episode_id`, `step` | | yes | |
| `proposal_id` | `ProposalId` or null | yes | Null only when the input could not be parsed enough to find a valid id (`SCHEMA_INVALID`). |
| `agent_id` | `AgentId` or null | yes | Null under the same condition. |
| `decision` | `accepted` or `rejected` | yes | There is no third state. Deferral is a policy-side decision and is recorded on the proposal. |
| `reason_codes` | list of `ReasonCode` | yes | Empty if and only if `decision = accepted`. At least one code if rejected. Every check whose precondition holds (dependency table below) is evaluated and all failures are listed, in the table order below; a check whose precondition does not hold is skipped and adds no code. |
| `resolved_action` | `ActionSpec` or null | yes | Non-null if and only if accepted. Defaults are filled in. This is the only action the simulator will receive. |
| `allowlist_id` | `ShortText` | yes | Identifier of the allowlist configuration used (for example `allowlist-default-v1`). |
| `state_digest` | 64 hex characters or null | no | Opaque digest of the simulator state at validation time. For rejected proposals the digest before and after the batch must be equal (tested). |

Reason codes (closed set, checked in this order):

| Code | Meaning |
|------|---------|
| `SCHEMA_INVALID` | Input does not parse as a `PolicyProposal`. |
| `EPISODE_MISMATCH` | `episode_id` differs from the active episode. |
| `EPISODE_NOT_ACTIVE` | No active episode, or the episode already ended. |
| `WRONG_STEP` | `step` differs from the current step. |
| `AGENT_UNKNOWN` | `agent_id` is not an agent of the topology. |
| `DUPLICATE_PROPOSAL` | `proposal_id` already processed in this episode. |
| `UNKNOWN_ACTION_KIND` | `action_kind` is not in the vocabulary. |
| `ACTION_NOT_ALLOWLISTED` | In the vocabulary but not on the coordinator allowlist. |
| `TARGET_REQUIRED` / `TARGET_NOT_ALLOWED` | Target missing, or present for `no_op`. |
| `UNKNOWN_TARGET_NODE` | Target is not a node of the topology. |
| `TARGET_OUT_OF_AUTHORITY` | Target is outside the proposing agent's zone. |
| `PARAMETER_INVALID` | Parameter not allowed for the kind, or out of range. |
| `ACTION_NOT_AVAILABLE` | Not in the simulator's `AvailableActions` for this agent at this step. |
| `CONFLICTING_PROPOSAL` | Another proposal accepted earlier in the same step targets the same node with a non-`no_op` action. |
| `HIGH_IMPACT_BUDGET_EXCEEDED` | More than `max_high_impact_per_step` (default 1) high-impact actions would run in this step. |

**Check dependencies (which checks run).** The engine submits the proposals of a step keyed by agent. The coordinator processes the keys in ascending `agent_id` order and assigns `validation_id` (and, for accepted proposals, `execution_id`) by processing position, so the n-th processed proposal gets the n-th `val-` counter value, including a proposal that fails to parse. Conflicts and budget exhaustion therefore resolve deterministically.

| Check | Runs only when |
|-------|----------------|
| `SCHEMA_INVALID` | Always. If it fails, every other check is skipped. |
| `EPISODE_MISMATCH`, `EPISODE_NOT_ACTIVE`, `WRONG_STEP`, `AGENT_UNKNOWN`, `DUPLICATE_PROPOSAL`, `UNKNOWN_ACTION_KIND` | The input parsed. |
| `ACTION_NOT_ALLOWLISTED`, `TARGET_REQUIRED` / `TARGET_NOT_ALLOWED`, `PARAMETER_INVALID` | The action kind is known (`UNKNOWN_ACTION_KIND` did not fail). |
| `UNKNOWN_TARGET_NODE` | A target is present. |
| `TARGET_OUT_OF_AUTHORITY` | The target is a known node and the agent is known. |
| `ACTION_NOT_AVAILABLE` | None of `UNKNOWN_ACTION_KIND`, `TARGET_REQUIRED`, `TARGET_NOT_ALLOWED`, `UNKNOWN_TARGET_NODE`, `TARGET_OUT_OF_AUTHORITY`, `PARAMETER_INVALID`, `AGENT_UNKNOWN` failed. |
| `CONFLICTING_PROPOSAL`, `HIGH_IMPACT_BUDGET_EXCEEDED` | Every earlier check passed. |

So a core-agent proposal to isolate a dmz node fails only `TARGET_OUT_OF_AUTHORITY` (that action is also absent from the core agent's `AvailableActions`, but `ACTION_NOT_AVAILABLE` is skipped because the authority check failed). Examples (steps 5 and 6 of the demo storyboard, DESIGN.md section 8.1; at step 5 `agent-core` is processed first and gets `val-000011`, `agent-dmz` gets `val-000012`):

```json contract=ValidationResult
{"schema_version":"1.0","validation_id":"val-000012","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":5,"proposal_id":"prop-000012","agent_id":"agent-dmz","decision":"accepted","reason_codes":[],"resolved_action":{"action_kind":"investigate_node","target_node_id":"node-web-02","parameters":{}},"allowlist_id":"allowlist-default-v1"}
```

```json contract=ValidationResult
{"schema_version":"1.0","validation_id":"val-000013","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":6,"proposal_id":"prop-000013","agent_id":"agent-core","decision":"rejected","reason_codes":["TARGET_OUT_OF_AUTHORITY"],"resolved_action":null,"allowlist_id":"allowlist-default-v1"}
```

### 8.7 `ExecutedAction` (executed simulated action)

One per accepted proposal. There is no execution record for a rejected proposal. Producer: the coordinator, from the simulator's reply to `apply_action`. Consumers: evaluator, recovery verifier, dashboard. Stored in `executions.jsonl`.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `execution_id`, `episode_id`, `step` | | yes | |
| `proposal_id`, `validation_id`, `agent_id` | ids | yes | Must resolve to an accepted validation of the same step. |
| `action` | `ActionSpec` | yes | Equal to the validation's `resolved_action`. |
| `status` | `executed` or `failed` | yes | `failed` when the simulator refused or errored; state then unchanged. |
| `error_code` | `ErrorCode` or null | yes | Non-null if and only if `status = failed`. |
| `affected_node_ids` | list of `NodeId` | yes | Nodes whose visible state changed. |
| `status_changes` | list of `{node_id, status_from, status_to}` | yes | Visible status changes only. It never says whether a node was truly compromised. |
| `outcome_event_ids` | list of `EventId` | yes | Outcome events of this execution. |
| `state_digest_before`, `state_digest_after` | 64 hex or null | no | Opaque. Equal when `status = failed`. |

Example:

```json contract=ExecutedAction
{"schema_version":"1.0","execution_id":"exe-000013","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":6,"proposal_id":"prop-000014","validation_id":"val-000014","agent_id":"agent-dmz","action":{"action_kind":"isolate_node","target_node_id":"node-web-02","parameters":{}},"status":"executed","error_code":null,"affected_node_ids":["node-web-02"],"status_changes":[{"node_id":"node-web-02","status_from":"up","status_to":"isolated"}],"outcome_event_ids":["evt-000018","evt-000019"],"state_digest_before":"a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90","state_digest_after":"0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0"}
```

Proposed, validated and executed are three different records with three different ids. The dashboard and the evaluator never infer one from another.

### 8.8 `NodeScoreRecord` (per-node predicted suspicion)

One record per agent per step per scorer, for every component that ranks nodes by suspicion: the detection role (`detection_rule`), the graph-feature rule baseline (`graph_feature_rule`) and, only if the learned-graph stage ran, a learned graph model (`learned_graph_model`). Producers: the engine writes the detection scorer's record from the `DetectionSummary` of the same step (identical numbers); `GraphFeaturePolicy` exposes its score through the optional `NodeScoringPolicy` interface and the engine writes it; the learned scorer writes it through the same interface. Consumers: the evaluator (ranking metrics `origin_rank_top1`, `origin_mrr`, section 11.2) and the dashboard (suspicion ranking view, labelled as predicted suspicion). Stored in `node_scores.jsonl`.

A score is a **predicted suspicion** computed from agent-visible inputs. A ranking of scores is not a root-cause finding and is not a causal claim. The evaluator compares the ranking with the evaluator-only origin node after the episode; the scorer never sees it.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `node_score_id` (`nsc-<step:04d>-<agent_id>-<scorer slug>`), `episode_id`, `step`, `agent_id` | | yes | |
| `scorer_id` | `ShortText` | yes | Component and version, for example `detection.rule-v1`, `graph-rule-v1`. Unique per scorer within a run. |
| `scorer_kind` | enum | yes | `detection_rule`, `graph_feature_rule`, `learned_graph_model`. |
| `nodes` | list of `{node_id, score}` | yes | Exactly the nodes of the agent's `AgentObservation.nodes`, same order (unobserved nodes are not scored). `score` is a `UnitFloat`. |

```json contract=NodeScoreRecord
{"schema_version":"1.0","node_score_id":"nsc-0005-agent-dmz-detection-rule-v1","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":5,"agent_id":"agent-dmz","scorer_id":"detection.rule-v1","scorer_kind":"detection_rule","nodes":[{"node_id":"node-app-01","score":0.0},{"node_id":"node-gw-01","score":0.26666666666666666},{"node_id":"node-web-01","score":0.0},{"node_id":"node-web-02","score":0.43333333333333335}]}
```

## 9. Evaluator-only record: `GroundTruthSnapshot`

Hidden truth about the simulated network. It exists for the evaluator and for simulator-side reward computation. It is never part of an observation, a graph record, a model input, a policy input, an `info` dictionary or any decision-time data. Producer: the simulator's separate ground-truth accessor (`ground_truth()`). Consumers: reward model, evaluator, and the dashboard only as a clearly labelled evaluator overlay. Stored only under `evaluator_only/ground_truth.jsonl`. Which packages may import `cyberswarm.contracts.evaluator_only` or call `ground_truth()` is defined once, in the single list of DESIGN.md section 3.2; ARCHITECTURE section 9 and this file only point to it (enforced by `tests/integration/test_import_boundaries.py`).

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `episode_id`, `step` | | yes | One record per step, captured by `end_step()` of that step, after P4 and before P0 of the next step (section 5, rule 9). `step` equals `StepResult.step_closed`. |
| `visibility` | const `evaluator_only` | yes | |
| `incident_active` | bool | yes | The incident has started. |
| `nodes` | list of objects | yes | One per node, ascending `node_id`: `node_id`, `compromised` (bool), `service_availability` (`UnitFloat`: availability including the compromise penalty and dependency effects; used by the attack-impact view only), `defense_availability` (`UnitFloat`: availability from status and dependency effects only, with the compromise factor fixed at 1.0; this is the quantity the healthy-disruption reward term and the two disruption metrics use, so that damage done by the attack is not charged to the defender), `contained` (bool: status is `quarantined`, `isolated` or `restoring`), `status` (`NodeStatus`). |
| `new_compromises` | list of `{node_id, cause}` | yes | Nodes that became compromised during this step. `cause` is `initial_entry` or `propagation`. |
| `alert_truth` | list of `{event_id, true_positive}` | yes | One entry for every `alert_raised` event of this step. |

```json contract=GroundTruthSnapshot
{"schema_version":"1.0","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":5,"visibility":"evaluator_only","incident_active":true,"nodes":[{"node_id":"node-web-02","compromised":true,"service_availability":0.9,"defense_availability":1.0,"contained":false,"status":"up"}],"new_compromises":[],"alert_truth":[{"event_id":"evt-000014","true_positive":true}]}
```

(The example lists one node for readability; a real record lists every node.)

## 10. Graph records (required lead-owned infrastructure; not the central research claim)

The graph pipeline turns agent-visible records into deterministic tensors. It is infrastructure for a controlled comparison of conventional and graph-feature observations. It uses no learned message passing. Learned graph models are a staged, secondary hypothesis whose go or no-go is decided at stage S7, after IPPO, the uncertainty extension and the evaluation path are stable (DECISIONS D-19, AD-11; RL_IMPLEMENTATION section 8). `GraphModelOutput` exists so that "not computed" is explicit until a producer ran.

**Inputs allowed:** the `Topology`, the `AgentObservation` for the same step and agent (which contains only events with `step <= t`). **Inputs forbidden:** anything from `evaluator_only`, any event with `step > t`.

### 10.1 `GraphSnapshot`

One per agent per step. Producer: `cyberswarm.graph.converter`. Consumers: graph-feature computation, observation encoder, dashboard graph view. Stored in `graph_snapshots.jsonl`.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `snapshot_id` (`gsn-<step:04d>-<agent_id>`), `episode_id`, `step`, `agent_id`, `topology_id` | | yes | |
| `feature_set_id` | const `graph-base-v1` | yes | Registry of feature names and order (below). |
| `node_order` | list of `NodeId` | yes | Ascending `node_id`; equals all nodes of the topology (cross-record rule X-11). |
| `edge_order` | list of `EdgeId` | yes | Ascending `edge_id`; equals all edges of the topology. |
| `edge_index` | two lists of int | yes | `[sources, targets]`, each of length E, values are indices into `node_order`, in `edge_order` order. Direction is source to target. |
| `node_feature_names` | list of str | yes | Must equal the registry for `feature_set_id`. |
| `node_features` | float matrix N x 11 | yes | float32-representable, finite. |
| `node_mask` | 0/1 matrix N x 11 | yes | 1 = value came from the topology or the observation; 0 = imputed. |
| `edge_feature_names` | list of str | yes | Registry. |
| `edge_features` | float matrix E x 6 | yes | |
| `edge_mask` | 0/1 matrix E x 6 | yes | |
| `dtype` | const `float32` | yes | |
| `imputation_policy_id` | const `impute-v1` | yes | Policy below. |
| `source_observation_id` | id | yes | The observation converted. |
| `converter` | `ShortText` | yes | For example `graph.converter-v1`. |
| `max_event_step` | `Step` or null | yes | Highest `step` among the events consumed; null when the observation window contains no events. Otherwise must be `<= step`. |

Node feature registry `graph-base-v1`, in column order (the same quantities as the conventional per-node features, so the two views differ only by the derived features of 10.2):

| Col | Name | Source | Range |
|-----|------|--------|-------|
| 0 | `criticality` | `Node.criticality` (static, always observed) | [0, 1] |
| 1 to 5 | `status_up`, `status_rate_limited`, `status_quarantined`, `status_isolated`, `status_restoring` | one-hot of `visible_status` | {0, 1} |
| 6 | `alert_count_norm` | `min(alert_count_recent, 5) / 5` | [0, 1] |
| 7 | `max_alert_severity_norm` | `max_alert_severity / 3` | [0, 1] |
| 8 | `evidence_level_norm` | `evidence_level / 3`; null maps to imputed | [0, 1] |
| 9 | `service_health` | observation value; null maps to imputed | [0, 1] |
| 10 | `steps_since_investigated_norm` | `min(steps_since_investigated, 10) / 10`; null maps to imputed | [0, 1] |

Edge feature registry `graph-base-v1`: 0 `type_network_link`, 1 `type_service_dependency` (one-hot of `edge_type`, static, always observed); 2 `state_active`, 3 `state_limited`, 4 `state_blocked` (one-hot of `edge_state`); 5 `traffic_level` (null maps to imputed).

**Mask rule.** For an observed node, columns 1 to 7 (status one-hots, `alert_count_norm`, `max_alert_severity_norm`) always have mask `1`, because they exist in every `NodeObservation` (a count of 0 is an observed zero). Columns 8 to 10 have mask `1` only when the observation value is non-null. For a node with `scope = boundary` the observation withholds `evidence_level`, `service_health` and `steps_since_investigated`, so columns 8 to 10 have mask `0`.

**Imputation policy `impute-v1`** (deterministic, uses no learned or hidden values): every imputed value is exactly `0.0` and its mask entry is `0`. A node absent from the observation (unobserved) has all dynamic columns 1 to 10 imputed; column 0 is public topology information and stays observed. A null measurement imputes only that column. An edge absent from the observation imputes columns 2 to 5; columns 0 and 1 stay observed. Observed columns have mask `1` even when their value is `0.0`. Consumers must read the mask to tell "observed zero" from "imputed zero".

```json contract=GraphSnapshot
{"schema_version":"1.0","snapshot_id":"gsn-0005-agent-dmz","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":5,"agent_id":"agent-dmz","topology_id":"topo-mock-small-v1","feature_set_id":"graph-base-v1","node_order":["node-app-01","node-web-02"],"edge_order":["edge-017"],"edge_index":[[1],[0]],"node_feature_names":["criticality","status_up","status_rate_limited","status_quarantined","status_isolated","status_restoring","alert_count_norm","max_alert_severity_norm","evidence_level_norm","service_health","steps_since_investigated_norm"],"node_features":[[0.699999988079071,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.20000000298023224,1.0,0.0,0.0,0.0,0.0,0.20000000298023224,0.6666666865348816,0.0,0.7099999785423279,0.0]],"node_mask":[[1,1,1,1,1,1,1,1,0,0,0],[1,1,1,1,1,1,1,1,0,1,0]],"edge_feature_names":["type_network_link","type_service_dependency","state_active","state_limited","state_blocked","traffic_level"],"edge_features":[[1.0,0.0,1.0,0.0,0.0,0.5199999809265137]],"edge_mask":[[1,1,1,1,1,1]],"dtype":"float32","imputation_policy_id":"impute-v1","source_observation_id":"obs-0005-agent-dmz","converter":"graph.converter-v1","max_event_step":5}
```

(The example is reduced to two nodes and one edge for readability. A real snapshot covers the whole topology; the cross-record rule X-11 enforces that.)

### 10.2 `GraphFeatureRecord` (graph-derived features)

Hand-engineered, inspectable features computed from the topology and one `GraphSnapshot`. Producer: `cyberswarm.graph.features`. Consumers: observation encoder (graph mode), graph-feature baseline, dashboard. Stored in `graph_features.jsonl`.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `feature_record_id` (`gft-<step:04d>-<agent_id>`), `episode_id`, `step`, `agent_id`, `topology_id`, `snapshot_id` | | yes | |
| `feature_set_id` | const `graph-derived-v1` | yes | |
| `node_order` | list of `NodeId` | yes | Same as the snapshot. |
| `feature_names` | list of 8 str | yes | Registry below. |
| `values` | float matrix N x 8 | yes | float32-representable, in [0, 1]. |
| `mask` | 0/1 matrix N x 8 | yes | 1 = computed from at least one real input; 0 = documented default used. |

Registry `graph-derived-v1`. Let `L(n)` be the neighbours of `n` over `network_link` edges in either direction, `N` the node count, and `obs` the set of observed nodes of the agent.

| Col | Name | Definition | Default (mask 0) |
|-----|------|------------|------------------|
| 0 | `degree_norm` | `|L(n)| / (N - 1)` | never missing |
| 1 | `neighbor_crit_mean` | mean `criticality` over `L(n)` | never missing: `Topology` validation guarantees `N >= 2` and a connected link graph, so `L(n)` is never empty; mask always 1 |
| 2 | `neighbor_crit_max` | max `criticality` over `L(n)` | never missing; mask always 1 |
| 3 | `dependents_crit_norm` | sum of `criticality` of all nodes that depend on `n` directly or transitively over `service_dependency` edges, divided by the sum of all criticalities | never missing: 0.0 means "no dependents" and is a computed value; mask always 1 |
| 4 | `neighbor_alert_mean` | mean `alert_count_norm` over `L(n) ∩ obs` | 0.0, mask 0, if `L(n) ∩ obs` empty |
| 5 | `neighbor_alert_max` | max `alert_count_norm` over `L(n) ∩ obs` | 0.0, mask 0, if empty |
| 6 | `hop_to_alerted_norm` | shortest hop distance over `network_link` (undirected) from `n` to the nearest node in `obs` with `alert_count_norm > 0`, divided by the topology diameter; 0.0 if `n` itself is alerted | 1.0, mask 0, if no observed node is alerted |
| 7 | `blocked_edge_frac` | fraction of observed `network_link` edges incident to `n` whose `edge_state` is `blocked` | 0.0, mask 0, if no incident edge is observed |

Features 0 to 3 depend on the public topology only and always have mask 1; features 4 to 7 depend on the agent's observation. `Topology` validation (`N >= 2`, connected link graph) makes the denominators `N - 1` and the diameter positive. None uses ground truth, evaluator data or future events. NetworkX is used for distances and the diameter.

```json contract=GraphFeatureRecord
{"schema_version":"1.0","feature_record_id":"gft-0005-agent-dmz","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":5,"agent_id":"agent-dmz","topology_id":"topo-mock-small-v1","snapshot_id":"gsn-0005-agent-dmz","feature_set_id":"graph-derived-v1","node_order":["node-app-01"],"feature_names":["degree_norm","neighbor_crit_mean","neighbor_crit_max","dependents_crit_norm","neighbor_alert_mean","neighbor_alert_max","hop_to_alerted_norm","blocked_edge_frac"],"values":[[0.7142857313156128,0.5600000023841858,1.0,0.1190476194024086,0.10000000149011612,0.20000000298023224,0.25,0.0]],"mask":[[1,1,1,1,1,1,1,1]]}
```

(One node shown for readability. Values are illustrative of the formulas on `topo-mock-small-v1` as seen by `agent-dmz`.)

### 10.3 `GraphModelOutput` (optional; produced only if the learned-graph stage ran)

Produced only by the staged learned-graph models of stage S7 (RL_IMPLEMENTATION section 8): node suspicion scores (`anomaly_score`) or node embeddings. Scores, rankings, attention weights and embeddings are predictive outputs only, never causal explanations. In a run where no learned graph model executed, the run manifest states `graph_model_outputs = not_computed` and no file is written; when one executed, `graph_model_outputs = present`, the records are written to `graph_model_outputs.jsonl`, and its scores are also written as `NodeScoreRecord` (section 8.8, `scorer_kind = learned_graph_model`). The `not_computed` shape exists so that the dashboard and tests have one agreed form for "no producer ran".

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `graph_model_output_id` (`gmo-<step:04d>-<agent_id>`), `episode_id`, `step`, `agent_id`, `snapshot_id` | | yes | |
| `status` | `computed` or `not_computed` | yes | |
| `model_id` | `ShortText` or null | yes | Null if and only if `not_computed`. |
| `output_kind` | `anomaly_score`, `embedding` or null | yes | Null if and only if `not_computed`. |
| `node_order` | list of `NodeId` or null | yes | Null if and only if `not_computed`. |
| `values` | float matrix or null | yes | Null if and only if `not_computed`. Anomaly score: N x 1 in [0, 1]. Embedding: N x D. |

```json contract=GraphModelOutput
{"schema_version":"1.0","graph_model_output_id":"gmo-0005-agent-dmz","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":5,"agent_id":"agent-dmz","snapshot_id":"gsn-0005-agent-dmz","status":"not_computed","model_id":null,"output_kind":null,"node_order":null,"values":null}
```

## 11. Episode, evaluation, error, log and run records

### 11.1 `EpisodeResult` (episode result)

One per episode. Producer: the runner, with the `outcome` block computed by `cyberswarm.evaluation.outcome` from the episode's records and its ground-truth stream. Consumers: evaluator, dashboard episode summary. Stored in `episode_result.json`. Because it contains ground-truth-derived outcome values it is `post_hoc_evaluation` data: it is written after the episode ends and no decision-time component reads it.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `episode_id`, `run_id`, `scenario_id`, `seed`, `policy_id` | | yes | `seed` is an int >= 0. |
| `representation` | `Representation` | yes | |
| `split` | `Split` | yes | |
| `simulator_kind` | `mock`, `simulator` or `fixture` | yes | A mock episode is never labelled `simulator`. |
| `visibility` | const `post_hoc_evaluation` | yes | |
| `steps_taken`, `terminated`, `truncated`, `termination_reason` | | yes | `termination_reason` in `objective_reached`, `incident_resolved`, `max_steps`. Exactly one of `terminated` and `truncated` is true. |
| `counts` | object | yes | `proposals`, `accepted`, `rejected`, `executed`, `failed` (ints). `accepted = executed + failed`; `proposals = accepted + rejected`. |
| `reward` | object or null | yes | Null when no reward model was attached. Else `sharing` (`team` or `zone_local`), `weights_id` (`ShortText`), `total` (float), `term_totals` with the seven raw, unweighted magnitudes (RL_IMPLEMENTATION section 4): `containment`, `response_delay`, `attack_impact`, `false_isolation`, `healthy_disruption`, `investigation_cost`, `invalid_action`. |
| `outcome` | object | yes | `compromised_nodes` (int: distinct nodes ever compromised), `attack_impact` (float), `healthy_services_disrupted` (int), `healthy_disruption_cost` (float), `false_isolation_count` (int), `true_positive_alerts` (int), `false_positive_alerts` (int), `compromised_nodes_detected` (int), `first_compromise_step`, `first_true_alert_step`, `containment_step`, `detection_to_containment_steps` (each int or null when undefined), `objective_reached` (bool), `origin_node_id` (`NodeId` or null: the node where the incident started, from the ground-truth stream; null if no incident started), `node_scoring` (list, one entry per scorer that produced `NodeScoreRecord`s in this episode: `scorer_id`, `scorer_kind`, `origin_rank_top1` and `origin_mrr` (floats or null), `n_ranked_pairs` (int)). Definitions in 11.2. The ranking values rank predicted suspicion; they are not a root-cause or causal finding. |
| `uncertainty_summary` | object or null | yes | Null unless an uncertainty-aware policy or its matched control ran. Else `deferral_count`, `evidence_request_count`, `containment_allowed_count` (ints), `mean_uncertainty` (`UnitFloat` or null). |
| `provenance` | `Provenance` | yes | |

```json contract=EpisodeResult
{"schema_version":"1.0","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","run_id":"run-fixture-demo","scenario_id":"scn-mock-lateral-eval-01","seed":2001,"policy_id":"pol-rule-conv-v1","representation":"conventional","split":"fixture","simulator_kind":"fixture","visibility":"post_hoc_evaluation","steps_taken":13,"terminated":true,"truncated":false,"termination_reason":"incident_resolved","counts":{"proposals":26,"accepted":25,"rejected":1,"executed":25,"failed":0},"reward":null,"outcome":{"compromised_nodes":2,"attack_impact":5.6,"healthy_services_disrupted":1,"healthy_disruption_cost":1.75,"false_isolation_count":0,"true_positive_alerts":2,"false_positive_alerts":1,"compromised_nodes_detected":2,"first_compromise_step":3,"first_true_alert_step":5,"containment_step":8,"detection_to_containment_steps":3,"objective_reached":false,"origin_node_id":"node-web-02","node_scoring":[{"scorer_id":"detection.rule-v1","scorer_kind":"detection_rule","origin_rank_top1":0.75,"origin_mrr":0.875,"n_ranked_pairs":8}]},"uncertainty_summary":null,"provenance":{"kind":"fixture_synthetic","note":"Illustrative values, not a result"}}
```

### 11.2 `EvaluationMetrics` (evaluation metrics)

One per (policy, representation, split) over a matched set of (scenario, seed) pairs. Producer: `cyberswarm.evaluation.metrics`, from `EpisodeResult` records. Never from constants. Consumers: dashboard comparison view, reports. Stored in `evaluation/metrics.jsonl`.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `metrics_id`, `run_id`, `policy_id`, `representation`, `split` | | yes | `split` is `eval` or `eval_utopo` for reported results; metrics for the two are always separate records. |
| `policy_family_id` | `PolicyFamilyId` | yes | The comparison arm (for example `fam-ippo-conv`). |
| `training_seed` | int >= 0 or null | yes | The training seed of this policy; null for policies that are not trained (rule-based). |
| `matched_set_id` | `MatchedSetId` | yes | Identifies the exact sorted list of (scenario, seed) pairs. Two records are comparable only if equal. |
| `n_episodes` | int >= 1 | yes | Completed episodes that entered the summaries. |
| `n_aborted` | int >= 0 | yes | Episodes of the matched set that aborted (ARCHITECTURE section 8). They are excluded from the means and reported here. |
| `scenario_ids`, `seeds` | lists | yes | Sorted, unique. |
| `ci_method` | `percentile_bootstrap` or null | yes | |
| `bootstrap_resamples` | int or null | yes | Default 1000 when used. |
| `metrics` | list of `MetricSummary` | yes | One per metric name below. |
| `provenance` | `Provenance` | yes | |

`MetricSummary`: `name`, `scorer_id` (`ShortText` or null: set only for the ranking metrics, one summary per scorer), `mean`, `std`, `ci_low`, `ci_high` (floats or null), `n` (episodes where the metric is defined), `n_undefined` (episodes where it is undefined). The summaries of one `EvaluationMetrics` record cover the episodes of ONE policy (one training seed); the episode-level interval describes variation over scenarios and seeds for that trained policy only. Variation across training seeds is reported by `AggregateMetrics` (section 11.8). `std` is the sample standard deviation (ddof = 1), null if `n < 2`. The interval is null if `n < 5`. A metric undefined in every episode has `mean = null`, never 0.

| Metric `name` | Definition (per episode, then summarized) | Undefined when |
|---------------|-------------------------------------------|----------------|
| `alert_precision` | true-positive alerts / all alerts | no alerts |
| `detection_recall` | compromised nodes with at least one true-positive alert before they were contained or the episode ended / nodes ever compromised | no node compromised |
| `false_positive_rate` | false-positive alerts / healthy node-steps | never |
| `detection_to_containment_steps` | `containment_step - first_true_alert_step`. `containment_step` is the first step at or after the first true alert such that no compromised node is uncontained at the end of the step and no new compromise occurs afterwards | no true alert, or containment never reached |
| `attack_impact` | sum over steps of the summed criticality of compromised nodes | never |
| `compromised_nodes` | distinct nodes ever compromised | never |
| `healthy_services_disrupted` | distinct truly healthy nodes (not compromised at that step) with `defense_availability < 1` at any step. Collateral of necessary containment (for example a healthy dependent of a quarantined compromised node) is included; loss caused by the attack itself is not | never |
| `healthy_disruption_cost` | sum over steps of criticality x (1 - `defense_availability`) over healthy nodes. Same convention | never |
| `false_isolation_count` | executed containment-class actions (`quarantine_node`, `isolate_node`, `restore_node`) on a node that was not compromised at execution | never |
| `episode_reward` | `EpisodeResult.reward.total` with the named weights | no reward model attached |
| `origin_rank_top1`, `origin_mrr` (per `scorer_id`) | Ranking of predicted suspicion, not a root-cause or causal finding. Let `o` be `origin_node_id`. Take every pair (step `t`, agent `a`) with `first_true_alert_step <= t <= containment_step` (up to the last step if containment was never reached) such that `o` is among the nodes scored for `a` at `t` in that scorer's `NodeScoreRecord`. The rank of `o` is its position by descending score among those nodes, and a tie takes the worst rank of the tie group. `origin_rank_top1` is the fraction of pairs with rank 1, `origin_mrr` is the mean of `1/rank`. Per episode, then summarized over episodes | no incident, no true alert, or no pair; scorers that produce no `NodeScoreRecord` get no entry |
| `rejected_proposal_count`, `investigation_action_count` | counts from the records | never |
| `deferral_count`, `evidence_request_count` | from `uncertainty_summary` | no uncertainty-aware run |
| `uncertainty_ece` | single value per policy, not per episode: `CalibrationReport.ece_calibrated` (RL_IMPLEMENTATION section 7.4). In the summary `mean` holds it, `std`, `ci_low` and `ci_high` are null, `n` is the number of evaluation committing proposals | no uncertainty computed, or `status` not `ok` |

```json contract=EvaluationMetrics
{"schema_version":"1.0","metrics_id":"met-fixture-rule-conv","run_id":"run-fixture-demo","policy_id":"pol-rule-conv-v1","representation":"conventional","split":"fixture","policy_family_id":"fam-rule-conv","training_seed":null,"matched_set_id":"set-fixture-demo","n_episodes":1,"n_aborted":0,"scenario_ids":["scn-mock-lateral-eval-01"],"seeds":[2001],"ci_method":null,"bootstrap_resamples":null,"metrics":[{"name":"attack_impact","scorer_id":null,"mean":5.6,"std":null,"ci_low":null,"ci_high":null,"n":1,"n_undefined":0},{"name":"origin_mrr","scorer_id":"detection.rule-v1","mean":0.875,"std":null,"ci_low":null,"ci_high":null,"n":1,"n_undefined":0},{"name":"uncertainty_ece","scorer_id":null,"mean":null,"std":null,"ci_low":null,"ci_high":null,"n":0,"n_undefined":1}],"provenance":{"kind":"fixture_synthetic","note":"Illustrative values, not a result"}}
```

(Three of the metrics are shown. A real record lists every metric name in the table.)

### 11.3 `ComparisonRecord`

Paired comparison of two comparison arms (policy families) on the same matched set. Producer: `cyberswarm.evaluation.compare`. Consumers: dashboard baseline-versus-policy view. Stored in `evaluation/comparison.jsonl`. `mean_difference = candidate - baseline`. The pairing unit depends on the arms: when neither arm is trained (rule against rule), the unit is the (scenario, seed) pair; when an arm is trained, the unit is the training seed (for each training seed the difference of the per-seed means over the matched set is computed, and the interval is a percentile bootstrap across training seeds, null below 5). Two learned arms are paired by equal training seed. If the two arms have different matched sets the record is not produced. The comparisons that are produced are enumerated in RL_IMPLEMENTATION section 9, each separately for `eval` and `eval_utopo`.

| Field | Type | Req |
|-------|------|-----|
| `schema_version`, `comparison_id`, `run_id`, `matched_set_id`, `split` | | yes |
| `baseline_policy_family_id`, `candidate_policy_family_id` | `PolicyFamilyId` | yes |
| `baseline_representation`, `candidate_representation` | `Representation` | yes |
| `unit` | `matched_pair` or `training_seed` | yes |
| `n_pairs` | int >= 1 | yes (number of units) |
| `differences` | list of `{name, scorer_id, mean_difference, ci_low, ci_high, n_pairs}` (`scorer_id` `ShortText` or null; the rest floats or null) | yes |
| `provenance` | `Provenance` | yes |

```json contract=ComparisonRecord
{"schema_version":"1.0","comparison_id":"cmp-fixture-demo","run_id":"run-fixture-demo","matched_set_id":"set-fixture-demo","split":"fixture","baseline_policy_family_id":"fam-rule-conv","candidate_policy_family_id":"fam-rule-graph","baseline_representation":"conventional","candidate_representation":"graph_features","unit":"matched_pair","n_pairs":1,"differences":[{"name":"attack_impact","scorer_id":null,"mean_difference":-0.7,"ci_low":null,"ci_high":null,"n_pairs":1}],"provenance":{"kind":"fixture_synthetic","note":"Illustrative values, not a result"}}
```

### 11.4 `ErrorRecord` (error record)

Structured error. Producer: any lead-owned component or Raksha's simulator. Consumers: runner (log), dashboard (error panel), tests. Stored in `errors.jsonl` (per episode) or the run log.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `error_id` | | yes | |
| `run_id`, `episode_id`, `step` | ids or null | no | As much context as is known. |
| `component` | enum | yes | `contracts`, `simulator`, `mock`, `roles`, `graph`, `engine`, `coordinator`, `rl_env`, `baseline`, `marl`, `uncertainty`, `gnn`, `evaluation`, `runner`, `config`. `simulator` is a real simulator; `mock` is the lead-owned MOCK. |
| `code` | `ErrorCode` | yes | See below. |
| `severity` | enum | yes | `warning` (continue, the result is unaffected), `error` (the item failed, the run continues), `fatal` (the run stops). |
| `recoverable` | bool | yes | |
| `message` | `ShortText` | yes | No stack traces, no paths, no secrets. |
| `proposal_id`, `validation_id`, `execution_id`, `node_id` | ids or null | no | |

`ErrorCode`: `SCHEMA_INVALID`, `SCHEMA_VERSION_MISMATCH`, `UNKNOWN_ID`, `SCENARIO_UNSUPPORTED`, `SIMULATOR_FAILURE`, `ACTION_EXECUTION_FAILED`, `POLICY_FAILURE`, `UNCERTAINTY_UNAVAILABLE`, `CHECKPOINT_ERROR`, `CONFIG_INVALID`, `IO_ERROR`, `LEAKAGE_DETECTED`, `INVARIANT_VIOLATION`. (`ReasonCode` and `ErrorCode` are separate enums that share only the literal `SCHEMA_INVALID`.) The per-component behaviour for each code is in ARCHITECTURE section 8.

```json contract=ErrorRecord
{"schema_version":"1.0","error_id":"err-000001","run_id":"run-smoke","episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","step":5,"component":"coordinator","code":"SCHEMA_INVALID","severity":"warning","recoverable":true,"message":"Proposal rejected, not parseable as PolicyProposal"}
```

### 11.5 `LogRecord` (log fields)

One JSON line in `logs/run.log.jsonl`. Logging uses the standard library `logging` module with a JSON formatter. Logs are not part of determinism comparisons and are never read by decision-time components.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version` | const | yes | |
| `ts` | string | yes | UTC, `YYYY-MM-DDTHH:MM:SSZ`. The only wall-clock value in the system. |
| `level` | enum | yes | `debug`, `info`, `warning`, `error`. |
| `component` | enum | yes | As in `ErrorRecord`. |
| `run_id`, `episode_id`, `step` | ids or null | no | |
| `code` | `ErrorCode` or null | no | |
| `message` | `ShortText` | yes | |

```json contract=LogRecord
{"schema_version":"1.0","ts":"2026-10-30T20:00:00Z","level":"info","component":"runner","run_id":"run-smoke","episode_id":null,"step":null,"code":null,"message":"Run started"}
```

### 11.6 `RunManifest`

Index of one run directory. Producer: the runner. Consumers: dashboard loader (fixture and integration modes), documentation validator. Stored at `run_manifest.json`.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `run_id` | | yes | |
| `simulator_kind` | `mock`, `simulator` or `fixture` | yes | |
| `topology_ids`, `scenario_ids`, `policy_ids` | lists | yes | |
| `seeds` | list of int | yes | |
| `config_digest` | 64 hex | yes | SHA-256 of the resolved configuration snapshot. |
| `package_version` | `ShortText` | yes | |
| `episodes` | list of `{episode_id, directory, status}` | yes | `directory` is a `RelPath` relative to the run directory. `status` is `completed` or `aborted`; an aborted episode has no `episode_result.json`. |
| `files` | list of `{path, record_type, count}` | yes | `path` is a `RelPath` relative to the run directory. Every record file in the run, with its contract name and line count. |
| `graph_model_outputs` | `not_computed` or `present` | yes | `present` only if a learned graph model produced `GraphModelOutput` records in this run. |
| `status` | object | yes | Three entries `ippo`, `mappo`, `learned_gnn`, each `{state, reason, decision_id}`. `state` is a `ComponentStatus`: `ippo` is `not_run` or `implemented`; `learned_gnn` is `not_run`, `implemented` or `no_go`; `mappo` is `not_run`, `implemented` or `deferred`. `reason` (`ShortText`) is non-null if and only if `state` is `no_go` or `deferred`. `decision_id` references the `StageDecision` (section 11.9) that justifies `no_go` or `deferred`, else null. **The runner computes every state from what this run executed** (policy families present in its episodes) and from the committed `StageDecision` files; no state is a constant. `implemented` means a policy of that family produced proposals in this run; `no_go` and `deferred` mean none did and a decision record with evidence exists; otherwise `not_run`. The dashboard shows these values as given. |
| `provenance` | `Provenance` | yes | |

```json contract=RunManifest
{"schema_version":"1.0","run_id":"run-fixture-demo","simulator_kind":"fixture","topology_ids":["topo-mock-small-v1"],"scenario_ids":["scn-mock-lateral-eval-01"],"policy_ids":["pol-rule-conv-v1"],"seeds":[2001],"config_digest":"0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a69788796a5b4c3d2e1f0","package_version":"0.1.0","episodes":[{"episode_id":"ep-rule-conv-v1-mock-lateral-eval-01-s2001","directory":"episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001","status":"completed"}],"files":[{"path":"episodes/ep-rule-conv-v1-mock-lateral-eval-01-s2001/events.jsonl","record_type":"EventRecord","count":18}],"graph_model_outputs":"not_computed","status":{"ippo":{"state":"not_run","reason":null,"decision_id":null},"mappo":{"state":"not_run","reason":null,"decision_id":null},"learned_gnn":{"state":"not_run","reason":null,"decision_id":null}},"provenance":{"kind":"fixture_synthetic","note":"Illustrative values, not a result"}}
```

### 11.7 `CalibrationReport` (uncertainty calibration)

Result of the held-out calibration check of the uncertainty signal (RL_IMPLEMENTATION section 7.4). Producer: `cyberswarm.uncertainty.calibration`. Consumers: evaluator (`uncertainty_ece`), dashboard reliability view. Stored in `evaluation/calibration.jsonl`. It exists only for runs that computed an uncertainty signal; otherwise the file is absent and `uncertainty_ece` is undefined.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `report_id` (`cbr-<slug>`), `run_id`, `policy_id` | | yes | `policy_id` is the matched control whose committing proposals were scored. |
| `cal_split`, `eval_split` | `cal` const; `eval` or `eval_utopo` | yes | The fitting split and the reporting split. They must be disjoint (tested). One report per reporting split. |
| `n_cal`, `n_eval` | int >= 0 | yes | Committing proposals (impact class `medium` or `high`) scored in each split. |
| `bins` | list of `{lower, upper, n, mean_confidence, accuracy}` | yes | 10 equal-width confidence bins on the evaluation split; `mean_confidence` and `accuracy` are `UnitFloat` or null for an empty bin. Empty list when no data. |
| `ece_raw`, `ece_calibrated`, `mce`, `brier` | float or null | yes | Null when not computable. |
| `auroc`, `auroc_ci_low`, `auroc_ci_high` | float or null | yes | AUROC of the uncertainty value for predicting incorrect proposals, with a percentile bootstrap interval. |
| `status` | `ok`, `insufficient_data`, `not_informative` | yes | `ok` requires enough data and AUROC >= 0.60 with the interval's lower bound above 0.5. |
| `tau_evidence`, `tau_defer` | `UnitFloat` or float up to 1.01 | yes | Thresholds in force. 1.01 means the rule cannot fire. |
| `threshold_selection` | `default` or `selected_on_cal` | yes | |
| `provenance` | `Provenance` | yes | |

```json contract=CalibrationReport
{"schema_version":"1.0","report_id":"cbr-fixture-demo","run_id":"run-fixture-demo","policy_id":"pol-ippo-conv-s0-ctl","cal_split":"cal","eval_split":"eval","n_cal":0,"n_eval":0,"bins":[],"ece_raw":null,"ece_calibrated":null,"mce":null,"brier":null,"auroc":null,"auroc_ci_low":null,"auroc_ci_high":null,"status":"insufficient_data","tau_evidence":0.5,"tau_defer":0.3,"threshold_selection":"default","provenance":{"kind":"fixture_synthetic","note":"Illustrative values, not a result"}}
```

### 11.8 `AggregateMetrics` (statistics over training seeds)

Episodes of IPPO networks trained with different seeds are not independent samples of one policy, so the spread that matters for learned arms is the spread across training seeds. `AggregateMetrics` holds it. Producer: `cyberswarm.evaluation.metrics.aggregate_over_training_seeds`. Consumers: dashboard comparison view, reports. Stored in `evaluation/aggregate.jsonl`. Rule-based arms have no training seed; their `EvaluationMetrics` (episode-level bootstrap over scenarios and seeds) is their aggregate and no `AggregateMetrics` is written for them.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `aggregate_id` (`agg-<slug>`), `run_id`, `policy_family_id`, `representation`, `split`, `matched_set_id` | | yes | `split` is `eval` or `eval_utopo`. |
| `training_seeds` | list of int >= 0 | yes | Sorted, unique, at least 1. Every trained seed of the family appears, including seeds whose training failed or did not learn. |
| `training_outcomes` | list of `{training_seed, training_status, learning_check}` | yes | `training_status`: `completed` or `aborted`; `learning_check`: `improved`, `not_improved` or `not_assessed`. `improved` means the mean training return over the last 10 percent of updates exceeds the mean over the first 10 percent. It is descriptive and says nothing about policy quality. A seed with `not_improved` or `aborted` stays in the record and is reported. |
| `n_units` | int >= 1 | yes | Number of training seeds with at least one completed evaluation episode. |
| `ci_method`, `bootstrap_resamples` | `percentile_bootstrap` or null; int or null | yes | The interval is a percentile bootstrap across training seeds; null when fewer than 5 seeds are defined. |
| `metrics` | list of `AggregateMetricSummary` | yes | `name`, `scorer_id` (as in `MetricSummary`), `per_seed_means` (floats or null, aligned with `training_seeds`), `mean`, `std` (sample, ddof = 1, null below 2 seeds), `ci_low`, `ci_high`, `n_units_defined`. |
| `provenance` | `Provenance` | yes | |

```json contract=AggregateMetrics
{"schema_version":"1.0","aggregate_id":"agg-fixture-ippo-conv","run_id":"run-fixture-demo","policy_family_id":"fam-ippo-conv","representation":"conventional","split":"fixture","matched_set_id":"set-fixture-demo","training_seeds":[0,1],"training_outcomes":[{"training_seed":0,"training_status":"completed","learning_check":"improved"},{"training_seed":1,"training_status":"completed","learning_check":"not_improved"}],"n_units":2,"ci_method":null,"bootstrap_resamples":null,"metrics":[{"name":"attack_impact","scorer_id":null,"per_seed_means":[6.1,7.3],"mean":6.7,"std":0.848528137423857,"ci_low":null,"ci_high":null,"n_units_defined":2}],"provenance":{"kind":"fixture_synthetic","note":"Illustrative values, not a result"}}
```

### 11.9 `StageDecision` (go, no-go or deferral of a staged component)

The recorded outcome of the learned-GNN stage (S7) or the MAPPO stage (S8). Producer: the author of the stage report, validated by the contract tests. Consumers: the runner (`RunManifest.status`), documentation validator, dashboard (status panel). Stored at `reports/stage_decisions/<component>.json` (committed; outside the run directories). A file is absent until its stage has run.

| Field | Type | Req | Rule |
|-------|------|-----|------|
| `schema_version`, `decision_id` (`dec-<slug>`) | | yes | |
| `component` | `learned_gnn` or `mappo` | yes | |
| `stage` | `S7` or `S8` | yes | The workflow stage that decided. |
| `state` | `ComponentStatus` | yes | `learned_gnn`: `implemented` or `no_go`. `mappo`: `implemented` or `deferred`. |
| `reason` | `ShortText` or null | yes | Non-null if and only if `state` is `no_go` or `deferred`. A no-go or deferral needs a specific, evidenced reason (a failed criterion with its measured value). "Not enough time", "dependencies" without a failing check, and "never tried" are not accepted. |
| `criteria` | list of `{criterion_id, passed, measured, threshold, note}` | yes | `criterion_id` `ShortText`; `passed` bool or null; `measured` and `threshold` floats or null; `note` `ShortText` or null. The criteria are the go criteria of RL_IMPLEMENTATION section 8.4 (GNN) or the blocker checks of section 6.4 (MAPPO). |
| `evidence_paths` | list of `RelPath` | yes | Committed repository files that hold the evidence (for example the stage report). Non-empty when `state` is `no_go` or `deferred`. |
| `provenance` | `Provenance` | yes | |

```json contract=StageDecision
{"schema_version":"1.0","decision_id":"dec-fixture-mappo","component":"mappo","stage":"S8","state":"deferred","reason":"Fixture example only. Centralized critic test failed in the fixture","criteria":[{"criterion_id":"M-10-critic-input-global","passed":false,"measured":null,"threshold":null,"note":"Illustrative"}],"evidence_paths":["reports/stage_reports/fixture-example.md"],"provenance":{"kind":"fixture_synthetic","note":"Illustrative values, not a result"}}
```

## 12. Cross-record consistency rules (tested by `tests/contracts/test_fixture_consistency.py`, `tests/integration/test_bundle_consistency.py` and the conformance suite)

| # | Rule |
|---|------|
| X-1 | Every `node_id`, `edge_id` and `zone_id` in any record resolves to the run's `Topology`. |
| X-2 | Every `agent_id` equals `agent-` plus the zone slug of an existing zone. |
| X-3 | Every record's `episode_id` is the episode's id; `step` is in `[0, steps_taken - 1]`. |
| X-4 | Ordering: within one episode the records sort by `(step, phase_order, sequence)` exactly as produced. Event ids are contiguous and ascending in emission order within each prefix (`evt-`, `cev-`). |
| X-5 | No `AgentObservation` contains an event with `step` greater than the observation `step` (no future events). `last_step_feedback` refers only to proposals of `step - 1`. |
| X-6 | Chain: every `ValidationResult.proposal_id` resolves to a proposal of the same step and agent (except `SCHEMA_INVALID`). Every accepted validation has exactly one `ExecutedAction` with the same `proposal_id`, `validation_id` and `action`. A rejected validation has none. |
| X-7 | `PolicyProposal.observation_id` equals `obs-<step>-<agent_id>` and that observation exists. |
| X-8 | Every `outcome_event_ids` entry resolves to an event with `phase = outcome` in the same step and, for `action_applied`, `execution_id` equal to the execution. |
| X-9 | `AvailableActions` agree with the vocabulary table (impact class, high-impact flag, cost) and every target is in the agent's zone. A proposal accepted by the coordinator appears in the step's `AvailableActions` of its agent. |
| X-10 | Every `GroundTruthSnapshot` exists only under `evaluator_only/`. No agent-visible file contains a field named in the evaluator-only schema (`compromised`, `true_positive`, `service_availability`, `defense_availability`, `new_compromises`, `alert_truth`, `incident_active`). |
| X-11 | A `GraphSnapshot` or `GraphFeatureRecord` has `node_order` equal to all topology nodes in canonical order, `edge_order` equal to all topology edges, `edge_index` consistent with edge endpoints, matrix shapes `(N, F)` and `(E, F)` equal to the registry, masks aligned with features (same shape, values 0 or 1, imputed entries equal `0.0`), `max_event_step <= step` (or null), and `source_observation_id` resolving. |
| X-12 | `EpisodeResult.counts` equal the number of proposal, validation and execution records of the episode. |
| X-13 | `EvaluationMetrics` values equal a recomputation from the episode results (tolerance 1e-9) unless `provenance.kind = fixture_synthetic` and the record is explicitly an isolated schema example. |
| X-14 | Optional values that were not computed are `null`: `uncertainty` on proposals of policies that compute none; `reward` when no reward model ran; `graph_model_outputs = not_computed` when no model ran; `uncertainty_ece` undefined when no uncertainty signal exists. |
| X-15 | Every `NodeScoreRecord` lists exactly the node ids of the `AgentObservation` of the same step and agent, in the same order. For `scorer_kind = detection_rule` each score equals the `suspicion` of the same node in the `DetectionSummary`. |
| X-16 | Event visibility: every event's `visible_to` equals the set given by the visibility rule of section 8.1, and no `AgentObservation` contains an `action_applied` or `investigation_result` whose `agent_id` differs from the observation's agent. |
| X-17 | `RunManifest.status` equals the status recomputed from the run's episodes (policy families that produced proposals) and the committed `StageDecision` files; `graph_model_outputs = present` if and only if `graph_model_outputs.jsonl` exists and is non-empty. |

## 13. Compatibility policy

| Item | Policy |
|------|--------|
| Enum values fixed before 1.0 ships | `Split.eval_utopo` and `Representation.learned_embedding` are part of version 1.0 from the start, because adding an enum value later is a MINOR change that forces the dashboard to regenerate its validators. No record uses `learned_embedding` unless the learned-graph stage ran. |
| Version format | `MAJOR.MINOR` string in `schema_version`. Current: `1.0`. The Python constant is `cyberswarm.contracts.SCHEMA_VERSION`. |
| MINOR change (1.0 to 1.1) | Backward compatible for readers of newer software: new record types, new optional fields (nullable, default null), new enum values. A new enum value (for example a new `ActionKind`) additionally requires updated allowlist, mask, vocabulary tables and tests in the same change. |
| MAJOR change (1.x to 2.0) | Removing or renaming a field, changing a type or unit, changing step semantics or the meaning of a value, removing an enum value. Requires a DECISIONS.md entry, regenerated fixtures and schemas, and an update of every consumer before merge. |
| Readers | A reader accepts records of its own MAJOR and any MINOR less than or equal to its own. A record with a higher MINOR or another MAJOR is rejected with `SCHEMA_VERSION_MISMATCH`. Because every model forbids unknown fields, a reader must be upgraded to read a newer MINOR. The dashboard regenerates its validators from `schemas/` when the version changes. |
| Writers | Always write the current version. Mixed versions inside one run directory are an error. |
| Who may change | Only the project lead edits `src/cyberswarm/contracts/`, `schemas/`, `fixtures/` and this file. Raksha and Mayukh send a change request (issue or message) that names the schema, the field, the reason and the affected consumers. Silent local changes to shared schemas are not allowed, and a pull request that edits these paths without the lead's approval must be rejected. |
| Changelog | A "Contract changelog" section is appended to this file for every version bump. Version 1.0 is the initial design; there are no earlier versions. |

## 14. Fixtures (synthetic, neutral, lead-owned)

All fixtures live under `fixtures/`, are labelled `fixture_synthetic` or documented as synthetic, and are illustrative only. They are not research results and not mock-run outputs. The complete list, the stage that creates each file and the tests that consume it are in DESIGN.md section 8. In short:

- `fixtures/records/valid/<RecordName>.<case>.json` and `fixtures/records/invalid/<RecordName>.<case>.json` give at least one valid and one invalid example per schema, with the expected failure in the file name or a sidecar `expected` field in the test table. Invalid examples include text fields that try to carry an IP address (`PolicyProposal.ip-in-rationale`) or a path (`PolicyProposal.path-in-rationale`), which `ShortText` rejects.
- `fixtures/bundle_demo/` is a complete run directory (section 15) that the dashboard can load in fixture mode: healthy network, an incident, alerts, a proposal that is accepted, a proposal that is rejected, a containment, a recovery check, an episode result and evaluator metrics. Graph snapshots span at least two steps with stable node ids and include missing values with masks. Optional model outputs and uncertainty are `null`. The bundle also holds two short episodes on the unseen topology variant `topo-mock-small-v2`, so that metrics for `eval` and `eval_utopo` appear as separate records and the graph features of the two topologies can be compared.

## 15. Output files the dashboard can consume

A run directory has the following layout. Every file is a list of contract records in JSON Lines (`.jsonl`) or one JSON document (`.json`). The same layout is used by the fixture bundle, the mock-run output and, later, Raksha's simulator through the same runner. The dashboard reads these files without transformation and without any value invented by the loader.

```
runs/<run_id>/                          (gitignored; fixtures/bundle_demo/ has the same layout)
  run_manifest.json                     RunManifest
  topologies/<topology_id>.json         Topology
  scenarios/<scenario_id>.json          ScenarioConfig
  episodes/<episode_id>/
    events.jsonl                        EventRecord
    observations.jsonl                  AgentObservation
    available_actions.jsonl             AvailableActions
    detection.jsonl                     DetectionSummary
    node_scores.jsonl                   NodeScoreRecord (detection scorer always; graph-feature scorer for graph policies; learned scorer if the GNN stage ran)
    graph_snapshots.jsonl               GraphSnapshot
    graph_features.jsonl                GraphFeatureRecord
    graph_model_outputs.jsonl           GraphModelOutput (only if a learned graph model ran; otherwise absent)
    proposals.jsonl                     PolicyProposal
    validations.jsonl                   ValidationResult
    executions.jsonl                    ExecutedAction
    errors.jsonl                        ErrorRecord (may be empty)
    episode_result.json                 EpisodeResult
    evaluator_only/ground_truth.jsonl   GroundTruthSnapshot (evaluator overlay; never agent input)
  evaluation/metrics.jsonl              EvaluationMetrics
  evaluation/aggregate.jsonl            AggregateMetrics (only for trained policy families)
  evaluation/comparison.jsonl           ComparisonRecord
  evaluation/calibration.jsonl          CalibrationReport (only when an uncertainty signal was computed)
  logs/run.log.jsonl                    LogRecord (not consumed by the dashboard)
```

Dashboard honesty rules that follow from the contracts: show proposed, accepted/rejected and executed as three different things; show `null` as "not available"; label `fixture_synthetic` and `mock_simulator_run` provenance on every metric view; show the evaluator overlay only in a panel labelled "evaluator view, not visible to agents"; show the manifest `status` block exactly as given (each of IPPO, MAPPO and the learned GNN is `not_run`, `implemented`, `no_go` or `deferred`, with its reason); show node-score rankings only as "predicted suspicion", never as a root cause; show metrics of `eval` and `eval_utopo` separately. The repository-level `reports/stage_decisions/*.json` files (`StageDecision`) are outside run directories and may be loaded by the status panel.

## 16. Producer and consumer summary

| Record | Producer | Consumers |
|--------|----------|-----------|
| `Node`, `Edge`, `Topology` | topology author (lead for mock, Raksha for simulator) | simulator, graph, rl, baselines, coordinator, evaluation, dashboard |
| `ScenarioConfig` | scenario author | simulator, runner, evaluation, dashboard |
| `EventRecord` | simulator; coordinator (recovery events only) | observation builder, graph, evaluation, dashboard |
| `AgentObservation` | simulator | roles, graph, rl, baselines, marl, dashboard |
| `AvailableActions` | simulator | rl (mask), baselines, coordinator, dashboard |
| `DetectionSummary` | roles.detection | rl (feature), baselines, dashboard |
| `GraphSnapshot`, `GraphFeatureRecord` | graph | rl, baselines, dashboard |
| `GraphModelOutput` | `cyberswarm.gnn` (stage S7, only if it ran); otherwise none | dashboard (null/absent) |
| `NodeScoreRecord` | engine (detection scorer), `baselines.graph_rule`, `gnn` scorer | evaluation (ranking metrics), dashboard |
| `PolicyProposal` | policy runner (baselines, marl, uncertainty) | coordinator, evaluation, dashboard |
| `ValidationResult`, `ExecutedAction` | coordinator | runner, rl (feedback), evaluation, dashboard |
| `GroundTruthSnapshot` | simulator ground-truth accessor | rl.reward, evaluation, dashboard overlay |
| `EpisodeResult` | runner (+ evaluation.outcome) | evaluation, dashboard |
| `EvaluationMetrics`, `AggregateMetrics`, `ComparisonRecord` | evaluation | dashboard |
| `CalibrationReport` | evaluation (`calibration_labels`, report builder) with `uncertainty.calibration` (fitting) | dashboard |
| `StageDecision` | stage report author (S7, S8) | runner (`RunManifest.status`), docs validator, dashboard |
| `ErrorRecord`, `LogRecord`, `RunManifest` | any component; runner | runner, dashboard (errors, manifest) |

## Contract changelog

- 1.0 (design, revision 2): initial version. Revision 2 (after the first design review) adds `NodeScoreRecord`, `AggregateMetrics`, `StageDecision`, the visibility and recovery-verification rules, the `eval_utopo` split, the `learned_embedding` representation, `defense_availability` and the stricter `ShortText`, all before 1.0 ships, so they are not version changes.
