# CyberSwarm Decision Log

**Status:** Version 0.4. Version 0.4 (review reconciliation, 2026-10-30) appends section 8 (AD-16 to AD-29): the corrected decisions that resolve the blocking design review's MEDIUM findings, together with the HIGH findings' new design artifacts (the unseen-topology `eval_utopo` split, the `NodeScoreRecord` origin-ranking metric and the execution-derived manifest status). Version 0.3 (design step, 2026-10-30) appends section 6 (AD-04 to AD-15) and section 7, and updates the status of D-06, D-11 (pointer), D-17 and D-19. Version 0.2 (research delta, 2026-10-30) added the dated architecture entries in section 0 (including the pivot entry AD-01), corrected some technology rows and aligned the technology table with the current `DELTA.md`. Version 0.1 held the technology decisions from the Phase 1-2 research step.
**Facts checked:** 2026-10-30 (all key facts re-checked on that date). Package versions move, so the design step pins ranges and re-checks them at install time.
**Evidence:** Sources and tags (VERIFIED / REPORTED / UNVERIFIED / INFERENCE) are defined in [RESEARCH.md](RESEARCH.md). In this file, "verified" means I read a registry record, a source file or a documentation page, or ran a small check, on the date above.

Entries are appended as decisions are made. Section 0 holds dated architecture-level entries. Sections 1 to 5 hold the technology decisions. The design step adds the remaining architecture entries.

Decision status values:

| Status | Meaning |
|--------|---------|
| Decided | Follows from the brief, from the user's confirmed requirements or from verified constraints. Changing it needs a reason recorded here. |
| Confirmed | The user has explicitly confirmed it. Changing it needs the user's decision. |
| Recommended | My choice. It depends on an assumption the user should confirm (named in the row), or the design step confirms it. |
| Staged | Not built yet, by design. Adoption is gated by an explicit go/no-go decision after the earlier stages are stable. A documented no-go satisfies the stage. Until then it is not a core dependency. |
| Optional | May be used but is never required. Declared as an optional extra. |
| Deferred | Belongs to the design step. Listed so it is not forgotten. |

---

## 0. Architecture decision log (dated)

### AD-01 | 2026-10-30 | Primary direction: self-learning autonomous cyber defense with multi-agent RL. The GNN / Faultline concept is no longer central.

| Field | Entry |
|-------|-------|
| Status | **Confirmed by the user.** Recorded in `.agents/tasks/cyberswarm-foundation/DELTA.md`, the user's latest confirmed requirements, which override `BRIEF.md` where they conflict. The user's original request states the same research direction: rule-based defense, IPPO, MAPPO, uncertainty-aware defense decisions, and comparative evaluation of detection-to-containment performance. |
| Context | The repository README (last modified 2026-10-25, "Concept / Initial Setup") framed CyberSwarm-GX as a GNN-powered root-cause-localization project ("Faultline Mode"). `BRIEF.md` required lead-owned graph infrastructure ahead of the RL environment and a staged GraphSAGE or GAT experiment. The untracked draft `docs/ARCHITECTURE.md` already described a multi-agent RL project with GNNs as optional. |
| Decision | 1. CyberSwarm is primarily **self-learning autonomous cyber defense using multi-agent reinforcement learning**, inside a controlled simulator only. 2. In scope: a rule-based baseline; IPPO; MAPPO only if correctly feasible (decentralized actors with a centralized critic or global training state, correct advantage estimation, a compatible environment; never IPPO relabelled); an uncertainty-aware defense mechanism with a matched non-uncertainty control; and detection-to-containment evaluation on identical scenarios and seeds with train and evaluation sets disjoint. 3. The GNN / Faultline concept is **not central**. Root-cause localization and ranking are no longer a headline task, and the central claim does not depend on any graph model. 4. **Graph-based network intelligence is required lead-owned infrastructure, not the central research claim** (current `DELTA.md`): stable topology and dynamic graph state, deterministic observation-to-graph conversion, missing-data masks with a deterministic non-leaking imputation policy, inspectable hand-engineered graph features, and graph observations for controlled comparison with conventional observations. It is built before the RL environment. NetworkX is the default topology and graph-processing foundation. 5. **A learned GNN (GraphSAGE, GAT or another) is a staged, secondary hypothesis.** It is considered only after the conventional and graph-feature paths and IPPO are stable, and only if a concrete task, a supervision source, dependency compatibility, a compute budget and a matched evaluation plan justify it. A documented no-go satisfies the stage. |
| Supersession note (resolved) | During the research delta two versions of `DELTA.md` existed; the earlier one called the graph pipeline and graph-feature baseline optional and NetworkX optional support. The current `DELTA.md` and `BRIEF.md` govern and make the lead-owned graph pipeline, graph schemas and the non-neural graph-feature baseline REQUIRED, built before the RL environment, with NetworkX a core dependency. The user confirmed this (AD-15). This is settled: there is no pending removal of the graph rows, and the earlier wording is not reopened. |
| Effect on the technology table | D-09 NetworkX: default topology and graph-processing foundation (not imported by the contracts package). D-10 PyTorch Geometric: staged, optional extra, never core. D-18 graph-feature pipeline and baseline: required, lead-owned, no learned message passing. D-19 GraphSAGE and GAT: staged; go/no-go after IPPO is stable. D-03 and D-17: contracts and dependency budget include the graph records and NetworkX. |
| Effect on documents | README, ARCHITECTURE and all later documents must follow this entry. The README (still the GNN-centred concept) and `docs/ARCHITECTURE.md` were deliberately left unmodified in the research delta; later steps align them. RESEARCH.md treats the learned-GNN idea (E6) as a staged, secondary hypothesis and adds the controlled graph-feature comparison (E8). |
| What it does not claim | That graph methods are worse or better, that multi-agent RL is better, or that either is novel. GNN performance is neither claimed nor ruled out; learned scores, rankings, attention weights and embeddings would be predictive outputs, not causal explanations. Related work already combines MAPPO with a graph attention encoder (arXiv:2606.13832), so a graph encoder would not distinguish the project anyway (RESEARCH.md section 1.6). |
| Reversal | Needs an explicit user decision recorded in a new entry. |

### AD-02 | 2026-10-30 | Research correction: CMARL-ACD and CyMARL are different works

| Field | Entry |
|-------|-------|
| Status | Recorded finding. The user's intended referent is unconfirmed (RESEARCH.md section 4, item 1). |
| Finding | arXiv:2310.05939 introduces CyMARL (cooperative defenders; IQL and QMIX; CybORG). "CMARL-ACD" is a GitHub repository (`cyb3rlab/CMARL-ACD`) and the name of a framework in a JAIST master's thesis (competitive attacker and defender; IPPO and MAPPO; a NASim fork). The journal paper that the repository names was not located. |
| Effect | Neither is a dependency (D-16 stands). Earlier statements that treated them as one work, or that called CMARL-ACD unverified as a name, are superseded by RESEARCH.md section 0. CMARL-ACD remains the closest IPPO and MAPPO prior, with the caveats in RESEARCH.md section 1.2. |

### AD-03 | 2026-10-30 | D-06 re-evaluated: TorchRL has IPPO and MAPPO losses

| Field | Entry |
|-------|-------|
| Status | Evidence corrected. Choice unchanged. D-06 moves from "Decided" to "Recommended". |
| Finding | TorchRL 0.14.1 contains `IPPOLoss`, `MAPPOLoss` and `MultiAgentGAE` (RESEARCH.md section 2.4). Version 0.1 had not found them. BenchMARL's README lists IPPO as well as MAPPO. |
| Effect | The self-written IPPO stays the recommendation, for the reasons in section 3. TorchRL's losses become the named contingency and an optional parity cross-check. The design step confirms or overrides D-06. |

---

## 1. Local environment (measured on this machine)

| Item | Value |
|------|-------|
| OS and CPU | macOS 15.7.4, arm64 (Apple Silicon) |
| Default `python3` | 3.13.2 (`/opt/homebrew/bin/python3`) |
| Other interpreters | 3.12.12 (`/opt/homebrew/bin/python3.12`); an Anaconda environment with 3.14.6; system 3.9.6. None of them has `torch` installed. |
| System site-packages | `pydantic` 2.12.5 and `numpy` 2.4.1 are installed in the Homebrew 3.13 environment. Nothing is installed into it by this work. |
| `uv` | Not installed. Plan with `python3 -m venv` and `pip`. |
| Node | v26.8.1, npm 11.19.0, pnpm present |
| **Free disk space** | **272 MiB** on the data volume (`/System/Volumes/Data`, 100% used) when first measured. About 911 MiB after I deleted my own ~210 MB of temporary research files. During the 0.2 delta it read 919 MiB at the start and about 650 MiB near the end, partly from my own temporary files (about 33 MB, removed at the end of the delta). The figure fluctuates. `~/Library/Caches/pip` still holds 2.0 GB, which I did not touch. |

**Likely blocker for the implementation steps.** The `torch` 2.15.0 wheel for Python 3.13 on macOS arm64 is 138.6 MB to download and 564.6 MB installed (the install size was read from the wheel's zip index over HTTP in 0.1; the wheel size was re-read from PyPI in 0.2). With numpy, pydantic and gymnasium the in-repo `.venv` needs roughly 0.7 GB, and the download needs working space on top of that, so peak use is about 0.8 to 1 GB. When free space was 272 MiB, an attempt to install the core stack into a temporary environment failed with `OSError: [Errno 28] No space left on device`. At the roughly 650 MiB measured at the end of the delta the install will very likely fail. The user should free space (about 2 GB of headroom is a safe target) before the step that creates `.venv` and runs PyTorch. Options for the user, none of which I performed: delete unneeded files, run `pip cache purge` (frees the 2.0 GB pip cache; it is regenerable), or point the venv at another volume (which conflicts with the brief's "venv inside the repo"). Use `pip install --no-cache-dir` to avoid adding a 138 MB cache copy.

**How compatibility was checked without touching the system Python.** Throwaway virtual environments under `/tmp` (Python 3.13.2 and 3.12.12) ran `pip install --dry-run` (resolution only; nothing was installed or imported). The system Python's packages are unchanged (`numpy` 2.4.1, `pydantic` 2.12.5, `pydantic_core` 2.41.5 only). A resolver solution is not a runtime test.

---

## 2. Technology decision table

| ID | Area | Choice | Status | Alternatives considered | Rationale | Verified version and compatibility facts |
|----|------|--------|--------|-------------------------|-----------|-------------------------------------------|
| D-01 | Python version | `requires-python >= 3.12`; develop and test on 3.13.2 | Recommended (teammates' Python versions unknown) | 3.11; 3.14 | The latest numpy, networkx and scipy require 3.12 or newer, so a 3.12 floor lets us use current releases. 3.13.2 is the default interpreter here and 3.12.12 is also installed. | numpy 2.5.4 `>=3.12`; networkx 3.7 `>=3.12, !=3.14.1`; scipy 1.18.1 `>=3.12`; torch 2.15.0 `>=3.11`; pydantic 2.14.2 `>=3.10`; pettingzoo 1.27.0 `<3.15,>=3.10`. A dry-run resolve on 3.13.2 succeeded, and the same core set (torch, gymnasium, pettingzoo, pydantic, numpy, networkx, pytest, jsonschema) also resolves on 3.12.12 (both re-run 2026-10-30). Third-party code in `docs/RESEARCH.md` that pins older Python (CMARL-ACD `<3.13`, CyMARL 3.9, CC4 pins) is not reused. |
| D-02 | Environment tooling | `python3 -m venv .venv` and `pip`; `.venv` gitignored | Decided (brief) | `uv`; conda | The brief asks for a venv in the repo. `uv` is not installed. | See the disk blocker in section 1. |
| D-03 | Shared contracts | pydantic v2 models in a neutral, lead-owned package; export JSON Schema files; `extra="forbid"`; discriminated unions for actions | Decided (brief recommends; verified fit) | Dataclasses plus hand-written JSON Schema; JSON Schema only; attrs/cattrs; msgspec; Protocol Buffers | One source of truth for Python types and for the language-neutral schema that Raksha's tests and Mayukh's dashboard consume. `extra="forbid"` and a closed set of action kinds make "no free-form payloads, shell strings or raw IPs" a validation rule. Optional confidence fields are `null` when not computed. Under AD-01 the contracts also carry the graph records (nodes, edges, dynamic graph snapshots, graph features) and an optional graph-model-output record that stays `null` when no model ran. Hand-written schemas would drift from the types. Protocol Buffers add code generation and a binary format that is harder to inspect. attrs/cattrs and msgspec were not evaluated beyond their names. | pydantic 2.14.2 (released 2026-10-29; MIT; needs `pydantic-core==2.50.2`, `typing-extensions>=4.16.0`). Ran a check on the installed 2.12.5: JSON Schema export with `additionalProperties: false`; discriminated union present in the schema; optional field exported as `anyOf [number, null]`; extra field, unknown action kind (`shell`), negative step and a raw IP as `node_id` were all rejected; JSON round trip equal. `model_json_schema()` emits no `$schema` key, so the export step adds `"$schema": "https://json-schema.org/draft/2020-12/schema"` (pydantic documents Draft 2020-12). Declared range: `pydantic>=2.12,<3`. |
| D-04 | On-disk records | UTF-8 JSON for single documents and JSON Lines (one record per line) for streams | Decided (brief) | SQLite; Parquet; MessagePack; Protocol Buffers | No database is allowed. Line-oriented files are append-only, human-readable, replayable and loadable in a browser without a server. | Standard library `json`; no extra dependency. |
| D-05 | Multi-agent environment interface | PettingZoo `ParallelEnv` API, with Gymnasium spaces and the Gymnasium terminated/truncated convention | Decided | One Gymnasium environment with a joint action; a private dict-based API; TorchRL `EnvBase`; RLlib `MultiAgentEnv` | All agents act in the same simulation step, which is what the parallel API models. It has a documented `state()` hook for centralized critics. Standard signatures keep the door open to other trainers. The dependency is small. Only core PettingZoo is used (no `[all]` extra). An AEC (turn-by-turn) API is not needed. | gymnasium 1.4.0: `reset(*, seed=None, options=None) -> (obs, info)`, `step(a) -> (obs, reward, terminated, truncated, info)`. pettingzoo 1.27.0 `ParallelEnv`: `reset(seed, options) -> (obs, infos)`, `step(actions) -> (obs, rewards, terminations, truncations, infos)`, `possible_agents`, `state()`. Its requirements are numpy, `gymnasium>=1.0.0`, `typing-extensions>=4.4.0`. Both resolve on 3.13.2. |
| D-06 | IPPO and MAPPO implementation | **Small self-written PyTorch implementation.** IPPO and MAPPO share one code path and differ only in the critic's input, which is settled (MAPPO uses the concatenation of agent-visible observation vectors; AD-10). MAPPO is implemented in its own stage after IPPO is stable, and is reported as `deferred` only if that stage finds a concrete, evidenced blocker. See section 3 and AD-10. | Decided in the design step (AD-08); was Recommended (re-evaluated in AD-03) | TorchRL directly (it does ship `IPPOLoss` and `MAPPOLoss`); BenchMARL; EPyMARL; Stable-Baselines3 (+ SuperSuit); official `on-policy` code; CMARL-ACD code; RLlib | See section 3. | See section 3. |
| D-07 | Deep-learning framework | PyTorch, CPU only | Decided (brief) | JAX; TensorFlow | The brief names PyTorch. The networks are small, so no GPU or MPS is planned (assumption: observation vectors of well under a thousand features). | torch 2.15.0 (released 2026-10-28). cp313 macOS arm64 wheel, 138.6 MB (564.6 MB installed). No source distribution. `requires-python >= 3.11`. The declared lower bound will be the version we actually test, recorded after the first successful install, since nothing has been installed yet. |
| D-08 | Stable-Baselines3 | Not a dependency | Decided | Use for IPPO via parameter sharing; use for a single-agent sanity baseline | It is single-agent and has no MAPPO. PettingZoo's own SB3 tutorial says: "SB3 is designed for single-agent RL and does not directly support multi-agent algorithms or environments", and that its examples use PPO with parameter sharing so that one model controls all agents. Parameter sharing conflicts with independent per-agent policies. Multi-agent use would need SuperSuit wrappers, which its own README says will be deprecated. `sb3-contrib` has `ppo_mask`, but a masked single-agent PPO does not give a multi-agent comparison. | stable-baselines3 2.9.2 requires `gymnasium>=0.29.1,<2.0`, `torch>=2.8`. sb3-contrib 2.9.2. SuperSuit (PyPI 3.11.0) README on its main branch: "this project will be deprecated and rewritten as part of a new wrappers package in PettingZoo". SuperSuit 3.11.0 needs `tinyscaler`, which has wheels for cp310 to cp312 only, so on 3.13.2 a wheels-only resolve falls back to SuperSuit 0.1.10. Revisit only if a specific single-agent experiment needs it. |
| D-09 | Topology and graph-processing library | NetworkX, used by the simulator and the lead's mock for topology logic and by the lead-owned graph pipeline; **not** imported by the contracts package | Decided (current `DELTA.md`): **required core dependency** and the default topology and graph-processing foundation, unless the design justifies a smaller minimal dependency | Hand-written adjacency dicts; igraph; rustworkx | Topologies are small. Reachability after quarantine, propagation along edges, degree, neighbours and hop distance are tested, standard algorithms, so we do not rewrite them. Keeping NetworkX out of the contracts package keeps the dashboard and the schema independent of it. Edges and nodes cross module boundaries as contract records, not as graph objects. | networkx 3.7 (released 2026-09-21; BSD-3-Clause; `>=3.12, !=3.14.1`); pure wheel 2.1 MB; about 18 MB installed. |
| D-10 | Graph neural networks (library) | PyTorch Geometric is **not** adopted now. It is a staged, optional extra, never core. The required graph pipeline exposes a model-compatible tensor and batch interface that does not need it. | Staged (gated by D-19; was "Decided (brief)" in 0.1) | Adopt PyG at the start; plain-PyTorch message passing | A learned GNN is a secondary hypothesis, considered only after the conventional and graph-feature paths and IPPO are stable. It is justified only for a specific experiment: graph encoder versus non-graph encoder on the same data, with a stated hypothesis. The node count is small enough that a dense-adjacency layer in plain PyTorch would suffice for a first ablation. Adopt PyG only if the go decision is taken and plain PyTorch proves inadequate. A GNN is an observation encoder; it is a separate choice from the RL algorithm. Its dependencies include network libraries (aiohttp, requests), which is a further reason to keep it out of the core install. | torch_geometric 2.8.1 (MIT, `>=3.10`; pure wheel 1.3 MB) declares aiohttp, fsspec, jinja2, numpy, psutil, pyparsing, requests, tqdm and xxhash as dependencies and does not declare torch. PyG's documentation states support for Python 3.10 to 3.14. A dry-run resolve on 3.13.2 succeeded (re-run 2026-10-30). |
| D-11 | Dashboard (Mayukh) | **React + Vite + TypeScript, with Cytoscape.js for the graph, Vitest + Testing Library for tests.** See section 4. | Recommended (assumes Mayukh can learn basic React; user to confirm) | Dash + dash-cytoscape; Streamlit; React Flow | See section 4. | See section 4. |
| D-12 | Configuration files | TOML for lead-owned training and evaluation configs, read with the standard library; scenario configurations are JSON records validated by the contract; each run saves a resolved JSON snapshot | Recommended (reversible) | YAML (PyYAML); JSON only | The brief allows YAML or TOML. TOML needs no dependency on Python 3.11 or newer, and only reading is needed. Schema validation uses the same pydantic models. | `tomllib` imports on 3.13.2 and 3.12.12. PyYAML 6.0.3 has cp313 arm64 wheels, so YAML would also work. |
| D-13 | Logging and experiment tracking | Standard `logging` plus JSON Lines and CSV files under a gitignored run directory | Decided | TensorBoard; Weights & Biases; MLflow | The brief wants no services or databases. TensorBoard would add a protobuf/grpc dependency tree to the core install. Results must come from evaluator output, which is a record file anyway. | Dry run: tensorboard 2.21.0 pulls absl-py, grpcio, protobuf, Markdown, Werkzeug and tensorboard-data-server. It is optional and not in the core set. |
| D-14 | Tests | pytest. `jsonschema` as a development-only dependency to validate fixtures against the exported schema files | Decided | unittest; Hypothesis | pytest is named in the brief. Validating fixtures with `jsonschema` as well as with pydantic checks that the exported schema files, which teammates use, agree with the models. | pytest 9.2.0 (`>=3.10`); jsonschema 4.26.0 (`>=3.10`). For JavaScript, Ajv 8.20.0 ships `dist/2020.js` for Draft 2020-12. |
| D-15 | Statistics | NumPy only (percentile bootstrap, mean and standard deviation) | Decided | SciPy; pandas; statsmodels | The planned statistics are means, standard deviations and bootstrap intervals; NumPy covers them without larger dependencies. | numpy 2.5.4 (`>=3.12`). scipy 1.18.1 and pandas 3.1.0 exist but are not needed. |
| D-16 | Existing simulators and environments | Design references only, never dependencies; no third-party code is copied into the repository | Decided | Build on CybORG / CAGE-4, Yawning Titan, CyberBattleSim, or the CMARL-ACD NASim fork | The brief requires our own small simulator behind our own contracts, owned by Raksha. CC4 pins `numpy==1.26.4`, `torch==2.2.0`, `gymnasium==0.28.1` (the numpy and torch pins have no cp313 wheels; the file does resolve on 3.12.12) and pulls `paramiko`, `sshtunnel` and Ray. Yawning Titan is archived. CyberBattleSim's defender is scripted. CMARL-ACD (repository and thesis) is a competitive attacker and defender setup that includes attacker tooling and real-network components. CyMARL has no license. | See [RESEARCH.md](RESEARCH.md), sections 1.2, 1.3, 1.5, 2.1 and 2.5. |
| D-17 | Package layout, `pyproject.toml` structure, dependency groups, lock file | `src/` layout, package `cyberswarm`; see AD-04 for every directory and owner and for the dependency groups | Decided in the design step (AD-04); was Deferred | `src/` layout; flat layout | Belongs to the design step. For its use, the dependency budget is: core (`pydantic`, `numpy`, `networkx`, required for topology and the graph pipeline), RL (`torch`, `gymnasium`, `pettingzoo`), dev (`pytest`, `jsonschema`), and optional extras (`tensorboard`; `torch_geometric`, staged under D-19). The contracts package imports only `pydantic` and the standard library (it does not import `networkx`). | - |
| D-18 | Graph representation, observation-to-graph conversion and hand-engineered graph features; non-neural graph-feature baseline | **Required, lead-owned, built before the RL environment** (mandatory order step 2). Deterministic conversion with stable node IDs and ordering, aligned missing-data masks, a documented non-leaking imputation policy, inspectable features (for example degree and criticality context, neighbour-alert aggregation), and a model-compatible tensor and batch interface. No learned message passing and no PyG. | Decided (current `DELTA.md` and `BRIEF.md`) | Treat the graph as optional support only (the earlier `DELTA.md` wording); learned encoder from the start | The principal comparison stays rule-based, IPPO, uncertainty-aware and MAPPO if feasible. The graph pipeline makes a controlled comparison of conventional versus graph-feature observations possible on identical scenarios and seeds (RESEARCH.md E8). It must use agent-visible data only, never evaluator-only labels or future events. Its cost is small: numpy, NetworkX and contract records. | No new dependency beyond NetworkX and numpy. |
| D-19 | GraphSAGE and GAT (learned GNN) | **Staged. The go/no-go is decided in its own learned-graph stage, after IPPO, the uncertainty extension and the evaluation path are stable (AD-11; RL_IMPLEMENTATION section 8), not at design time.** Candidate task: node-suspicion scoring trained on `train`-split ground-truth labels only, with optional learned node embeddings as an IPPO observation encoder. Models: plain-PyTorch `SageDense` (mean aggregation) and `GatDense` (masked attention) over the existing `GraphBatch`; PyTorch Geometric is not used. Go criteria G1-G5 and a conventional-versus-graph-feature-versus-learned-embedding ablation on identical `eval` and `eval_utopo` pairs are specified in RL_IMPLEMENTATION section 8. A no-go is valid only with a specific, evidenced reason; "no measurable improvement over the graph-feature baseline" is a valid reported result, not a no-go. | Staged (current `DELTA.md`, addenda 2 and 3; AD-11): decided in its own stage after IPPO, uncertainty and evaluation | GraphSAGE; GAT; another encoder; none | Dependencies and compute are not blockers: a plain-PyTorch dense-adjacency implementation avoids PyTorch Geometric, and the 8-node topology trains in seconds on CPU (D-10). The task, supervision source (with strict train/eval separation that keeps evaluator-only labels out of decision-time features), tensor contracts, resource budget and matched ablation plan are specified now in RL_IMPLEMENTATION section 8; the decision itself is taken in the stage, once the conventional-versus-graph-feature evidence exists. A risk the experiment tests, not a pre-decided finding: on a small topology the hand-engineered features may already capture what message passing would add. If a go is taken, compare against the conventional and graph-feature observations and choose between GraphSAGE and GAT then. Attention weights, scores and embeddings are predictive outputs, not explanations or causes. Related work already pairs a GAT encoder with MAPPO (arXiv:2606.13832), so no novelty is assumed. | torch_geometric 2.8.1 (see D-10); not required, since the models are plain PyTorch. |

---

## 3. IPPO and MAPPO: self-written implementation versus a library (D-06)

**Decision.** Write a small PyTorch implementation of clipped PPO with separate actor and critic networks, GAE, advantage normalization, value normalization, value clipping and action masking. IPPO and MAPPO use the same code. The only difference is what the critic receives: the agent's own observation (IPPO) or the concatenation of the agents' agent-visible observation vectors (MAPPO; AD-10). This centralized input is settled and contains no evaluator-only data. The implementation follows the practical suggestions in Yu et al. (value normalization, local plus global features in the value input, at most 10 to 15 epochs, clip ratio at or below 0.2, large batches).

### Options compared

| Option | Multi-agent PPO available? | Fit with our environment and contracts | Compatibility on this machine (Python 3.13.2) | Verdict |
|--------|---------------------------|----------------------------------------|-----------------------------------------------|---------|
| **Self-written PyTorch** | IPPO and MAPPO, one code path | Exact fit. Direct access to logits for uncertainty, direct control of masks, and contract-valid proposals as output. | Needs only `torch` and `numpy`. | **Chosen** |
| TorchRL 0.14.1 used directly | **Yes.** The 0.14.1 wheel contains `torchrl/objectives/multiagent/mappo.py` (classes `MAPPO`, `IPPO`; exported `IPPOLoss`, `MAPPOLoss`), `MultiAgentGAE`, `MultiAgentMLP`, `MaskedCategorical`, a `PettingZooWrapper`, and an example `sota-implementations/multiagent/mappo_ippo.py`. Its tutorial says the MAPPO critic takes "the global state of the system ... a global observation or simply the concatenation of the agents' observation". CMARL-ACD builds IPPO/MAPPO on TorchRL 0.8.0. (Version 0.1 of this log listed only `ppo.py` and `multiagent.py`; see AD-03.) | Our environment would need TensorDict specs and nested keys (a wrapper around the PettingZoo environment). The uncertainty hooks and contract-valid proposals must be added around TorchRL modules. | Resolves on 3.13.2 to torchrl 0.14.1, tensordict 0.14.3, torch 2.15.0 (dry run, re-run 2026-10-30). TorchRL 0.14.1 requires `tensordict>=0.14.3,<0.15.0` (the latest tensordict is 0.15.3) and new dependencies `pyvers` and `hoptorch`. The spec classes CMARL-ACD imports from TorchRL 0.8.0 (`CompositeSpec` and others) are named `Composite`, `Unbounded`, `Categorical` and `Binary` in 0.14.1 (no import was run), which shows API churn. | The strongest library alternative. Not chosen for the core. **Named contingency and optional parity cross-check.** |
| BenchMARL 1.5.2 | **Yes.** Its README table lists MAPPO and IPPO (on-policy, discrete and continuous actions). | Needs our environment as a TorchRL task and Hydra configs. | Source distribution only; requires `torchrl>=0.10,<0.12`, `hydra-core`, `torchvision` and `av<14`. A dry run on 3.13.2 resolves (torchrl 0.11.1, tensordict 0.11.0, torch 2.15.0, torchvision 0.29.2, av 13.1.0); runtime was not tested. README badge lists Python 3.8 to 3.11. | Rejected: video and vision dependencies and a benchmark-runner structure we do not need. |
| EPyMARL | IPPO, MAPPO, IA2C and others (README); Gymnasium and PettingZoo supported | Run as `python src/main.py` from a clone (not on PyPI); the README has a section on registering custom environments. Support for our per-agent action masks was not checked. | `requirements.txt` pins `PyYAML==5.3.1`, which has wheels only up to cp39. Not installed. | Rejected as a dependency; useful as a reference |
| Stable-Baselines3 + SuperSuit | Single-agent PPO only (documented by PettingZoo's SB3 tutorial) | Parameter-sharing wrapper only; no MAPPO | SB3 resolves; SuperSuit 3.11.0 does not resolve wheels-only on 3.13.2 (see D-08). | Rejected |
| Official `marlbenchmark/on-policy` | MAPPO and IPPO | Environment wrappers for SMAC, Hanabi, MPE and Google Research Football only | README targets Python 3.6.1 and torch 1.5.1. | Reference for the algorithm and its tricks only |
| CMARL-ACD training script | IPPO and MAPPO by a flag | Tied to the NASim fork, a learning attacker and TorchRL 0.8.0 | `python_requires <3.13`; numpy 2.0.2 has no cp313 wheel. | Reference only; no code copied |
| RLlib (Ray 2.59.0) | Not evaluated in depth | General-purpose framework listed in the existing architecture draft | cp313 arm64 wheel exists | Not needed; not used |

### Why self-written

1. The environment is small and custom, and every library route needs an adapter layer around it.
2. The uncertainty extension needs policy internals (logits, entropy, ensembles or dropout) and a matched control with identical code. That is easier in code we own.
3. A shared code path makes IPPO vs MAPPO a clean ablation: nothing else differs.
4. The policy must emit contract-valid proposals through an inference interface that does not depend on a framework's tensor containers.
5. The version pins above show that a library route chases API changes: TorchRL renamed its spec classes between 0.8.0 and 0.14.1, pins `tensordict<0.15`, and BenchMARL requires `torchrl<0.12` while the latest is 0.14.1.
6. A short, readable trainer suits a student project better than a large framework.
7. The disk budget is tight (section 1). TorchRL adds little next to torch itself, but BenchMARL adds torchvision and `av`.

### Costs and risks

- A new implementation can be wrong in ways that still produce plausible learning curves.
- It has no external benchmark validation. TorchRL now offers tested `IPPOLoss` and `MAPPOLoss`, so this is a real cost of the choice, not a formality.
- It creates a maintenance burden for the project lead.

### What would change the recommendation

Adopt TorchRL's `IPPOLoss`, `MAPPOLoss` and `MultiAgentGAE` (keeping our environment, policy heads, uncertainty mechanism and checkpointing) if the toy-task test below cannot be made to pass, or if a MAPPO centralized critic proves harder to get right than the library version. A parity test against TorchRL's loss on the same batch is a cheap cross-check and is optional. This is a design-step decision (AD-03).

### Mitigations (required tests, to be executed in the implementation step)

1. GAE against a hand-computed small example.
2. The clipped surrogate loss against a closed-form value, including gradient sign on both sides of the clip.
3. Masked sampling never selects a masked action, over many draws.
4. A toy cooperative task with a known optimal return. Both IPPO and MAPPO must reach it, or the failure must be understood before any claim is made.
5. Same seed gives the same result (determinism on the mock).
6. Value-normalization round trip.

**Fallback trigger.** If the toy-task test cannot be made to pass within the implementation step, take the loss and advantage computation from TorchRL (`IPPOLoss` or `MAPPOLoss` with `MultiAgentGAE`, or `ClipPPOLoss`) and keep everything else ours. That is a recorded contingency and does not leave the choice open before then.

**MAPPO status.** This decision covers the feasibility of the code. The centralized critic input is settled — the concatenation of agent-visible observation vectors, with no evaluator-only data (AD-10) — and the two-agent design is fixed, so MAPPO is implemented in its own stage after IPPO is stable (RL_IMPLEMENTATION section 6). It is reported as `deferred` only if that stage finds a concrete, evidenced blocker. No MAPPO quality claim is made from mock training; mock-trained numbers are labelled mock-only.

---

## 4. Dashboard framework for Mayukh (D-11)

**Recommendation.** React + Vite + TypeScript, Cytoscape.js for the network graph, Vitest + Testing Library for component tests, schema validation at the data boundary with Ajv against the exported JSON Schema files.

**Assumption to confirm.** Mayukh can pick up basic React and TypeScript. I do not know his experience. If he is Python-only and the user prefers that, the fallback is Dash + dash-cytoscape (not Streamlit). The data-source abstraction below is the same either way.

### Requirements taken from the brief

| ID | Requirement |
|----|-------------|
| R1 | Starts and is useful from fixture JSON before any simulator exists. |
| R2 | Replay with a step scrubber and an incident timeline. |
| R3 | Interactive topology graph with node status and connections. |
| R4 | Proposed, validated and executed actions displayed distinctly. |
| R5 | Loading, empty, malformed and partial-data states shown honestly; no invented numbers. |
| R6 | Fixture mode and integration mode through one data-source interface, without an architecture change. |
| R7 | Unit and component tests. |
| R8 | A student can iterate quickly. |

### Comparison

| Criterion | React + Vite + TS + Cytoscape.js | Dash + dash-cytoscape | Streamlit |
|-----------|----------------------------------|------------------------|-----------|
| R1 fixture-first | Static JSON or JSONL files served by the dev server or bundled as assets | Fixtures read in Python | Fixtures read in Python |
| R2 replay and scrubbing | Step index is client state; changing it re-renders the affected views | Possible with callbacks and a store; each interaction is a server round trip (reasoned, not tested) | Streamlit reruns the script top to bottom on interaction by default (documented); fragments reduce this but add complexity |
| R3 graph interaction | Cytoscape.js, described on npm as a "graph theory (a.k.a. network) library for analysis and visualisation" | dash-cytoscape wraps Cytoscape.js | `streamlit-agraph` last released 2023-01-28; Plotly network charts are less interactive (not tested) |
| R4, R5 distinct states and honest gaps | Typed discriminated unions force a branch for each state, including "not available" | Python-side branching | Python-side branching |
| R6 same code for fixtures and live | A `DataSource` interface in TypeScript with a file/JSONL implementation first, then a local-HTTP or file-watch one | Same idea in Python | Same idea in Python |
| R7 tests | Vitest and Testing Library run headless in Node | Browser-driven tests (not verified here) | `AppTest` exists in Streamlit (source file present) |
| Contract validation | Types generated from, or validators loaded from, the exported schemas. The dashboard depends only on schema files, not on Python. | Can import the Python models directly (an advantage), but couples the dashboard to the lead's package and Python version | Same as Dash |
| R8 student iteration | Hot reload; requires learning JS/TS and React | Pure Python; callback model has a learning curve | Fastest start in pure Python |

### Reasons for the recommendation

1. Replay, scrubbing and a timeline are naturally client-side state; Streamlit's rerun model fits them poorly.
2. Using the exported JSON Schema as the only boundary lets Mayukh work without importing the lead's Python package, which supports independent development and a later swap from fixtures to real data.
3. Cytoscape.js is built for network graphs, so node states and edges need no custom layout code.
4. Types plus schema validation make "never fabricate a value" enforceable: a missing field is an explicit `unavailable` case.

### Costs

- Mayukh needs JavaScript/TypeScript. This is the main risk and the reason the status is "Recommended".
- Two languages in one repository, and a Node toolchain in addition to Python.
- On this machine `node_modules` could be a problem while disk space is short; its size was not measured.

### Verified toolchain facts (npm registry, 2026-10-30)

| Package | Version | License | Node requirement |
|---------|---------|---------|------------------|
| react | 19.3.1 | MIT | any |
| vite | 8.3.5 | MIT | `^20.19.0 \|\| >=22.12.0` |
| typescript | 7.0.2 | Apache-2.0 | `>=16.20.0` |
| cytoscape | 3.34.4 | MIT | any |
| vitest | 5.0.5 | MIT | `^22.12.0 \|\| ^24.0.0 \|\| >=26.0.0` |
| @testing-library/react | 16.3.3 | MIT | `>=18` |
| ajv | 8.20.0 | MIT | - (the tarball contains `dist/2020.js`, the Draft 2020-12 validator) |
| @xyflow/react (React Flow) | 12.12.0 | MIT | - |
| playwright (optional, end-to-end) | 1.64.1 | Apache-2.0 | `>=20` |
| json-schema-to-typescript (optional, not tried) | 16.0.0 | MIT | - |
| dash-cytoscape (Python, fallback) | 1.0.2, released 2024-07-15 | MIT | - |

The local Node v26.8.1 satisfies Vite and Vitest. Versions above come from the registry. I did not install or run any of these packages. Mayukh's scaffold should use whatever `npm create vite@latest` pins and be re-checked then.

**Re-check on 2026-10-30 (research delta).** The registry versions above are unchanged. Additions: the React wrapper `react-cytoscapejs` 2.0.0 was last published on 2022-09-02, so the recommendation is to use `cytoscape` directly from a small React component rather than the wrapper. On PyPI, `dash-cytoscape` 1.0.2 has a source distribution only. Other options were checked by metadata only: Streamlit 1.66.1, Dash 4.5.0, NiceGUI 3.18.0 (Python), sigma 4.0.4 and vis-network 10.1.4 (JavaScript graph renderers). None changes the recommendation. Under AD-01 the dashboard consumes graph snapshots, hand-engineered graph features and optional model outputs through the contracts. A value that no producer computed (for example a learned-model score while D-19 is undecided) stays `null` and is shown as unavailable, never filled in.

### Why not React Flow

`@xyflow/react` describes itself as a library "for building node-based editors and interactive flow charts". CyberSwarm needs to display a network graph with state, not edit one. React Flow is not rejected on quality; it is a worse fit for this view.

### Data-source interface (design intent, specified for the module document)

A single `DataSource` boundary supplies validated contract records to the views. The first implementation reads a fixture file or JSON Lines file and supports replay. A later implementation reads from a local file watch or a local HTTP endpoint served by the lead's integration code. The views never know which one is in use. Details belong in `docs/MODULE_MAYUKH_DASHBOARD.md` and `docs/INTEGRATION.md`, which are not written yet.

---

## 5. Open items that affect these decisions

| # | Item | Needed from |
|---|------|-------------|
| 1 | Free disk space before the `.venv` and PyTorch step (section 1). About 650 MiB was free at the end of the delta; about 2 GB of headroom is a safe target. | User |
| 2 | Confirm the dashboard recommendation and Mayukh's JS/TS experience (section 4). | User |
| 3 | Confirm Python 3.12 as the floor, or give teammates' interpreter versions (D-01). | User |
| 4 | Resolved: the MAPPO centralized-critic input (concatenation of agent-visible observation vectors, no evaluator-only data) and the two-zone-agent design are settled (AD-10; DELTA addendum 3). | — (closed) |
| 5 | Package layout, dependency groups and lock-file policy (D-17). | Design step |
| 6 | Project license (undecided; no LICENSE file is to be added). | User |
| 7 | Confirm that "CMARL-ACD" in the brief means the `cyb3rlab/CMARL-ACD` repository and the JAIST thesis (AD-02; [RESEARCH.md](RESEARCH.md), Section 4, item 1). | User |
| 8 | Confirm or override D-06 (self-written IPPO versus TorchRL's `IPPOLoss` and `MAPPOLoss`; AD-03). | Design step |
| 9 | Resolved: the current `DELTA.md` (graph pipeline required, learned GNN staged) is the intended wording, user-confirmed (AD-01, AD-15). | — (closed) |

---

## 6. Design-step decisions (version 0.3)

Entries are dated 2026-10-30 and refer to documents that are SPECIFIED, NOT IMPLEMENTED: [ARCHITECTURE.md](ARCHITECTURE.md), [CONTRACTS.md](CONTRACTS.md), [RL_IMPLEMENTATION.md](RL_IMPLEMENTATION.md), and `.agents/tasks/cyberswarm-foundation/DESIGN.md`. The pivot entry AD-01 above is unchanged.

### AD-04 | 2026-10-30 | Repository layout: `src/` layout, one package, one owner per directory

| Field | Entry |
|-------|-------|
| Status | Decided (design step). Closes D-17. |
| Decision | `src/` layout with the single package `cyberswarm`. Python `>= 3.12`. Repo-local `.venv` (gitignored). Dependencies in `pyproject.toml`: core `pydantic>=2.12,<3`, `numpy`, `networkx>=3.4`; optional group `rl` with `torch`, `gymnasium>=1.0,<2`, `pettingzoo`; optional group `dev` with `pytest`, `jsonschema`. Optional extras are declared but not installed: none in this phase (`torch_geometric` is not declared, see AD-11). No lock file; after the first successful install a `requirements-tested.txt` (output of `pip freeze`) is committed so that the tested versions are recorded. |
| Alternatives | Flat layout (modules at the repository root): simpler to start, but tests can import uninstalled code and the packages would collide with the top-level `docs`, `fixtures`, `schemas`, `configs` directories. A separate package per owner: more packaging overhead than a three-person modular monolith needs. |
| Directories and owners | `src/cyberswarm/contracts/` lead (neutral); `interfaces/` lead (neutral); `common/` lead; `mock/` lead (MOCK); `roles/` lead; `graph/` lead; `coordinator/` lead; `engine/` lead; `rl/` lead; `baselines/` lead; `marl/` lead; `uncertainty/` lead; `evaluation/` lead; `runner/` lead; `src/cyberswarm/simulator/` **Raksha** (not created); `tests/simulator/` and `configs/scenarios/simulator/` **Raksha** (not created); `dashboard/` **Mayukh Aryan** (not created); `schemas/`, `fixtures/`, `configs/` (rest), `scripts/`, `tests/` (rest), `docs/` lead (neutral); `runs/`, `.venv/`, `checkpoints/` generated and gitignored. Full file-level tree: DESIGN.md section 2. |
| Rationale | One import root, strict directories with one owner, no placeholder packages for work that belongs to teammates. |
| Reversal | A layout change needs a new entry. |

### AD-05 | 2026-10-30 | Shared contracts, identifiers, step semantics and the single action vocabulary

| Field | Entry |
|-------|-------|
| Status | Decided (design step). Confirms D-03 and D-04. |
| Decision | Pydantic v2 models with `extra="forbid"` in `cyberswarm.contracts`, JSON Schema Draft 2020-12 exported to `schemas/`, JSON and JSON Lines on disk, canonical byte form (CONTRACTS section 2). Schema version `1.0` with the MAJOR.MINOR compatibility policy of CONTRACTS section 13. Stable identifiers per CONTRACTS section 4. Step phases P0 dynamics, P1 observe, P2 propose, P3 validate, P4 execute, then `end_step`; every record of step `t` carries `t` (CONTRACTS section 5). One closed vocabulary of eight abstract actions (`no_op`, `monitor_node`, `investigate_node`, `request_evidence`, `rate_limit_node`, `quarantine_node`, `isolate_node`, `restore_node`); high-impact: `isolate_node`, `restore_node`. Optional values that were not computed are `null`. Proposed, validated and executed are three record types. |
| Alternatives | Hand-written schemas (drift); a third `deferred` validation state (deferral is a policy decision and is recorded on the proposal); more actions (for example an explicit release action) were left out to keep the action space small: a node leaves isolation only through `restore_node`. |
| Consequence | Any simulator must provide the effects in CONTRACTS section 6. A new action is a MINOR change that also updates the allowlist, masks and tests. |

### AD-06 | 2026-10-30 | Logical roles: deterministic components around one policy per zone agent

| Field | Entry |
|-------|-------|
| Status | Decided (design step). |
| Decision | Detection is a deterministic component. Investigation is a set of costed actions chosen by the decision role plus a deterministic evidence view. Decision is one policy per zone agent (rule-based, graph-feature rule-based, or IPPO actor; no parameter sharing). Coordination is the defense coordinator, an independently testable safety boundary and the only caller of the simulator action API. Agents are zone defenders (`agent-dmz`, `agent-core` on the mock). None of the roles is an LLM-chat agent. |
| Alternatives | Separate learned policies per role; role-specific heads on a shared policy; one flat policy with no detection component (ARCHITECTURE section 6.2 gives the reasons each was rejected: detection would need ground-truth labels, investigation is naturally an action, and the deterministic summary makes all compared policies share the same detection input). |
| Consequence | A difference between the rule-based baseline and IPPO comes from the decision role alone. |

### AD-07 | 2026-10-30 | Environment, observation modes and reward design

| Field | Entry |
|-------|-------|
| Status | Decided (design step). The shared team reward is part of the design and needs no separate confirmation before MAPPO work (DELTA addendum 3; this supersedes the earlier clause that required user confirmation). |
| Decision | PettingZoo `ParallelEnv` over the engine; per-agent `Dict` observation (`obs` vector plus `action_mask`) and `Discrete` action space; masked-action substitution by `no_op` with a penalty; termination (objective reached or incident resolved) separate from truncation (horizon); `info` contains only agent-visible values; two observation modes `conventional` (`17N+2` values) and `graph_features` (`25N+2`) sharing one conversion path; seven reward terms with a shared team reward, defaults and rationale in RL_IMPLEMENTATION section 4; stored raw magnitudes so that weights can be changed after the fact. |
| Alternatives | One Gymnasium environment with a joint action (hides per-agent masks); zone-local rewards (kept as an option); reward shaping with uncertainty (rejected: it would confound the decision-rule ablation). |
| Consequence | Reward values are privileged training signals computed from evaluator-only ground truth; they are never observations. |

### AD-08 | 2026-10-30 | IPPO implementation: compact self-written PyTorch

| Field | Entry |
|-------|-------|
| Status | Decided (design step). Confirms D-06. |
| Decision | Self-written clipped PPO with GAE, masked categorical policies, value normalization, per-agent actor, critic, optimizers and normalizers. TorchRL's `IPPOLoss`, `MAPPOLoss` and `MultiAgentGAE` are the named contingency (used only if the toy-task or hand-computed GAE tests cannot be made to pass in stage S6) and an optional parity check. SB3, SuperSuit, RLlib, BenchMARL and EPyMARL are not dependencies. |
| Rationale | Small custom environment, direct access to the action distribution for the uncertainty rule and the matched control, one code path for a later MAPPO, minimal dependencies (RL_IMPLEMENTATION section 5.1). |
| Risk | Correctness is ours to prove; the mitigation is the test list in RL_IMPLEMENTATION section 5.5. |

### AD-09 | 2026-10-30 | Dashboard framework for Mayukh Aryan

| Field | Entry |
|-------|-------|
| Status | Recommended (unconfirmed: depends on Mayukh's JavaScript and TypeScript experience). Carries D-11 forward. |
| Decision | React, Vite and TypeScript, Cytoscape.js used directly for the topology graph, Vitest and Testing Library for tests, Ajv for validation against `schemas/`. One `DataSource` interface with a file and JSON Lines loader first (reading `fixtures/bundle_demo/` or `runs/<run_id>/`), a local HTTP or file-watch source later. The dashboard depends on the exported schema files and the run-directory layout of CONTRACTS section 15, not on the lead's Python package. |
| Fallback | Dash with `dash-cytoscape` (not Streamlit) if Mayukh prefers Python. |
| Consequence | The framework is not locked until the user confirms it. `dashboard/` is not created in this phase. |

### AD-10 | 2026-10-30 (revised) | MAPPO: implemented in its own stage

| Field | Entry |
|-------|-------|
| Status | Decided (design step, revised after review addenda 2 and 3): MAPPO is implemented in its own stage after IPPO, the uncertainty mechanism and the evaluation path are stable. It is not deferred at design time. |
| Decision | MAPPO is a planned implementation stage, not a design-time deferral. The design is complete in RL_IMPLEMENTATION section 6: per-agent decentralized actors on local observations; one centralized team critic over the concatenation of the agents' agent-visible observation vectors (the `env.state()` hook), containing no evaluator-only data; one shared GAE; the shared team reward. IPPO is never relabelled as MAPPO. The stage executes work items W1-W7 and acceptance tests M-1 to M-13. The shared team reward and the concatenated-observation critic input are the design and need no further confirmation (DELTA addendum 3). Until the stage runs there is no `marl/mappo.py` and every manifest reports `mappo: not_run`; the manifest status is computed from what actually executed, never hard-coded. |
| Ordering (not blockers) | MAPPO comes after IPPO because it reuses the self-written PPO code path (shared `marl.ppo` and `marl.gae`); the `state()` hook is built with the environment; the equal-budget multi-seed comparison runs on the mock, sized for this machine's CPU, with every result labelled mock-only. These are ordering and work items, not blockers. |
| Blocker | Deferral is allowed only if the stage finds a concrete blocker — a failing M-1 to M-13 that cannot be fixed within the stage, the comparison failing to finish within the CPU budget, or `torch` being uninstallable — recorded as a `deferred` `StageDecision` with command-output evidence. "Not needed" and "no time" are not accepted. |
| Reversal | The status is computed from execution; a `deferred` status requires the evidenced `StageDecision` above. |

### AD-11 | 2026-10-30 (revised) | Learned GNN (GraphSAGE or GAT): staged go/no-go in its own stage

| Field | Entry |
|-------|-------|
| Status | Decided (design step, revised after review addenda 2 and 3): the learned-graph go/no-go is taken in its own stage, not at design time. Carries D-19. |
| Decision | The required hand-engineered graph pipeline is built in its own early stage (AD-01, D-18). A learned graph model is a staged, secondary hypothesis whose go/no-go is decided in its own learned-graph stage, after IPPO, the uncertainty extension and the evaluation path are stable and after the conventional-versus-graph-feature ablation has run. The stage is specified now (RL_IMPLEMENTATION section 8): (1) candidate task A — node-suspicion scoring trained on `train`-split ground-truth labels only, with labels never reaching `cal`, `eval`, `eval_utopo` or any decision-time input; optional task B — frozen node embeddings used as an IPPO observation encoder (`learned_embedding` mode), run only if task A completes; (2) models — plain-PyTorch `SageDense` (mean aggregation) and `GatDense` (masked attention) over the existing `GraphBatch`, with PyTorch Geometric not used; (3) go criteria G1-G5 (data, compute, split disjointness, passing shape/mask/forward/backward tests, and the matched reference existing); (4) a matched conventional-versus-graph-feature-versus-learned-embedding ablation on identical `eval` and `eval_utopo` pairs, with origin-ranking metrics per scorer. A no-go is valid only with a specific, evidenced reason recorded in the stage `StageDecision`; "no measurable improvement over the graph-feature baseline" is a valid reported result, not a no-go. Dependencies and compute are not blockers (D-10). The conventional-versus-graph-feature ablation is required whatever the GNN decision. |
| Claims | No claim, in advance, that a learned graph model helps or does not help. Scores, rankings, attention weights and embeddings are predictive outputs, never causal explanations. |

### AD-12 | 2026-10-30 | Uncertainty-aware mechanism

| Field | Entry |
|-------|-------|
| Status | Decided (design step). |
| Decision | Signal: normalized entropy of the masked action distribution. Rule: evidence request for medium-impact actions on weakly evidenced nodes when `u >= tau_evidence`; deferral of high-impact actions when `u >= tau_defer` unless the evidence level is 3; a cap on consecutive substitutions; no random draws; no reward term. Matched control: the same wrapper with the rule disabled over the same checkpoint. Validation: calibration split distinct from train and eval, reliability bins, ECE, AUROC gate, status `ok`, `insufficient_data` or `not_informative`. Full specification and tests U-1 to U-13: RL_IMPLEMENTATION section 7. |
| Alternatives | Ensemble disagreement, MC-dropout (more networks or passes, larger budget); value-based evidence (no calibrated target). They remain the named alternatives if the entropy signal fails the gate. |

### AD-13 | 2026-10-30 | License

| Field | Entry |
|-------|-------|
| Status | Undecided. User decision pending. |
| Decision | No `LICENSE` file is created and no license text or SPDX identifier is added to `pyproject.toml` or any source file. The README states that the license is undecided. Third-party licences (CyMARL has none; CMARL-ACD, CC4, CybORG, NASim are MIT) do not matter because no third-party code is copied. |

### AD-14 | 2026-10-30 | Git identity and commits

| Field | Entry |
|-------|-------|
| Status | Open. User action required. |
| Decision | The local git identity was set in an earlier session with a guessed email. That email must not be reused. The workflow makes no commit, no push and no change to git configuration, remotes or branches. The user verifies `user.name` and `user.email` before any future commit. All work stays in the working tree on `main`. |

### AD-15 | 2026-10-30 | Graph pipeline is required infrastructure (current `DELTA.md` and `BRIEF.md`)

| Field | Entry |
|-------|-------|
| Status | Decided (current `DELTA.md` and `BRIEF.md`; user-confirmed). The graph pipeline is required infrastructure; this is settled. |
| Context | During the research delta an earlier, superseded `DELTA.md` sentence had called the graph pipeline and NetworkX optional. The current `DELTA.md` and `BRIEF.md` supersede it and require the lead-owned graph conversion and hand-engineered features before the RL environment. `AD-01` and the research delta follow the current file, and the user confirmed the requirement. |
| Decision | The design follows the current `DELTA.md` and `BRIEF.md`: the graph records (`contracts/graph.py`), the graph package (`graph/`), the `graph_features` observation mode and the non-neural graph-feature baseline (`baselines/graph_rule.py`) are REQUIRED infrastructure, built before the RL environment (stage S3, first part; stage S4). `graph/converter.py` provides the base-feature conversion shared by both the `conventional` and `graph_features` observation modes — an architecture fact, not a removal option. The learned GNN remains a staged go/no-go decided in its own stage (AD-11). |
| Reversal | Superseded and user-confirmed: the graph pipeline is required and there is no pending removal. A change would need an explicit user decision recorded in a new entry. |

---

## 7. Open items added by the design step

| # | Item | Needed from |
|---|------|-------------|
| 10 | Resolved: AD-15 closed — the graph pipeline is required (current `DELTA.md` and `BRIEF.md`), user-confirmed. | — (closed) |
| 11 | Resolved: the shared team reward and the MAPPO critic input are part of the design and need no separate confirmation (AD-07, AD-10; DELTA addendum 3). | — (closed) |
| 12 | Confirm the dashboard recommendation and Mayukh's skills (AD-09) | User |
| 13 | License decision (AD-13) | User |
| 14 | Verify the git identity before any commit (AD-14) | User |
| 15 | Free disk space before creating `.venv` (section 1; about 2 GB of headroom is a safe target) | User |
---

## 8. Review-reconciliation decisions (version 0.4)

Entries are dated 2026-10-30 and record the corrected decisions that resolve the blocking design review (`.agents/tasks/cyberswarm-foundation/gates/design-review.md`). They refer to documents that are SPECIFIED, NOT IMPLEMENTED; each names the review finding it closes and the later task that encodes it. AD-24, AD-25 and AD-26 capture the new design artifacts introduced by the HIGH findings (3, 4 and 1); their stale wording in `ARCHITECTURE.md` and the existing AD entries was corrected under task 2.1. `AD-` numbers are architecture decisions and are distinct from the `D-` technology rows in section 2.

### AD-16 | 2026-10-30 | `ground_truth()` returns the captured end-of-step-`t` snapshot

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 9 (MEDIUM). |
| Decision | `end_step()` captures the end-of-step-`t` ground-truth snapshot BEFORE it runs phase P0 of step `t+1`. `ground_truth()` returns that captured snapshot, with `GroundTruthSnapshot.step == StepResult.step_closed`, and keeps returning it until the next `end_step()`; `termination_reason()` refers to the same closed step `t`. A `restore_completed`, a new compromise or a new alert applied in P0 of `t+1` therefore does not appear in the step-`t` snapshot. Every step-`t` truth-derived quantity (reward `C_t`/`K_t`, `false_isolation`, `containment`, the outcome metrics) reads this captured snapshot, so a conformant simulator cannot pick the `t+1` reading. Conformance check SC-14 enforces it: in a scenario where `restore_completed` fires at P0 of `t+1`, the step-`t` snapshot still shows the node as `restoring` and still compromised. |
| Encoded by | `interfaces.simulator` docstring and conformance SC-14 (task 4.1); mock truth capture (task 5.2); engine ground-truth stream (task 7.1). |

### AD-17 | 2026-10-30 | Healthy-disruption is charged on `defense_availability`, not on attack-induced loss

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 10 (MEDIUM). |
| Decision | `GroundTruthSnapshot` carries two per-node availabilities: `service_availability` (status, dependency and compromise effects, used by `attack_impact`) and `defense_availability` (status and dependency effects only, with the compromise factor fixed at 1.0). `healthy_disruption`, `healthy_services_disrupted` and `healthy_disruption_cost` are computed from `defense_availability`, so attack-induced availability loss is charged to `attack_impact` only and never to the defender. Collateral from necessary containment stays in the metric by design, and the metric definition says so. Reward case R-10 and the demo fixture are updated accordingly: the demo `healthy_disruption_cost` becomes 1.75 (it was 2.125 under the old definition that mixed in 0.30 of attack-induced loss). |
| Encoded by | `GroundTruthSnapshot` contract (task 3.1); mock truth (task 5.2); reward model (task 7.2); `evaluation.outcome` (task 12.1). |

### AD-18 | 2026-10-30 | The mock `service_health` compromise proxy is weakened, with an AUROC guard

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 11 (MEDIUM). |
| Decision | On the mock, the `service_health` channel is weakened so it is informative but not an oracle for compromise: the compromised availability factor is 0.9 (was 0.7), the `service_health` noise is 0.08 (was 0.03), and the edge `traffic_level` compromise offset is 0.05 with noise 0.08 (was +0.15). `noise` is a standard normal truncated at ±3, drawn from the `health_noise` table. A mock guard test asserts that the best threshold on `service_health` predicts node compromise with AUROC below an agreed bound on a random-policy rollout, so a learned policy cannot read compromise off one input and the IPPO-versus-rule comparison does not partly measure access to an oracle-like channel. This qualifies the earlier claim that compared policies "differ only in the decision role": they also differ only through an intentionally weak, non-decisive health channel (the ARCHITECTURE §6.2 wording was corrected under task 2.1). |
| Encoded by | mock dynamics and observation (task 5.2); `service_health` AUROC guard test (task 5.3). |

### AD-19 | 2026-10-30 | Event-visibility table and SC-15

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 12 (MEDIUM). |
| Decision | A single event-visibility table in CONTRACTS §8.1 fixes `visible_to` for every event type: `episode_started` and `episode_ended` to all agents; `alert_raised`, `node_status_changed` and `restore_completed` to the node's zone agent plus any agent owning a node adjacent over a `network_link`; `action_applied` and `investigation_result` to the acting `agent_id` only; `recovery_verified` and `recovery_failed` to the node's zone agent. Conformance check SC-15 enforces that no agent's observation ever contains an `action_applied` or `investigation_result` caused by another agent, so an agent never sees another agent's actions or investigations. |
| Encoded by | mock observation `visible_to` (task 5.2); conformance SC-15 (tasks 4.1, 5.3). |

### AD-20 | 2026-10-30 | Recovery verification from agent-visible data, executed restores only

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 13 (MEDIUM). |
| Decision | Recovery verification uses agent-visible data only and tracks only executed `restore_node` actions. For a tracked execution on a node, arm at the `restore_completed` event at step `s`. Emit `recovery_verified` (producer `coordinator`, id prefix `cev-`, carrying `node_id` and `execution_id`, `visible_to` the node's zone agent) at the first step `t ≥ s` where, for the last `window` consecutive steps, the zone agent's observation shows `visible_status == up` and a non-null `service_health ≥ health_min`; a null `service_health` resets the streak. Emit `recovery_failed` if that condition is not met within `verify_timeout_steps` (default 6) after `s`, or if no `restore_completed` arrives within `restore_duration_steps + 2` after the execution. `quarantine` and `isolate` executions are not verified, because their recovery is the later restore. Defaults: `health_min = 0.8`, `window = 2`. |
| Encoded by | `coordinator/recovery.py` (task 9.1); verified/failed/null-breaks-streak/unfinished-at-episode-end tests (task 9.2). |

### AD-21 | 2026-10-30 | Coordinator reason-code dependency order and counter-assignment order

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 14 (MEDIUM). |
| Decision | The coordinator evaluates reason codes in a fixed dependency order, lists every independent failure, and skips only checks whose precondition already failed. `SCHEMA_INVALID` short-circuits all other checks. Once the input parses, `EPISODE_MISMATCH`, `EPISODE_NOT_ACTIVE`, `WRONG_STEP`, `AGENT_UNKNOWN`, `DUPLICATE_PROPOSAL` and `UNKNOWN_ACTION_KIND` always run. `ACTION_NOT_ALLOWLISTED`, `TARGET_REQUIRED`/`TARGET_NOT_ALLOWED` and `PARAMETER_INVALID` need a known action kind; `UNKNOWN_TARGET_NODE` needs a target; `TARGET_OUT_OF_AUTHORITY` needs a known target and a known agent; `ACTION_NOT_AVAILABLE` runs only if none of `UNKNOWN_ACTION_KIND`, the `TARGET_*` checks, `UNKNOWN_TARGET_NODE`, `TARGET_OUT_OF_AUTHORITY`, `PARAMETER_INVALID` or `AGENT_UNKNOWN` failed; `CONFLICTING_PROPOSAL` and `HIGH_IMPACT_BUDGET_EXCEEDED` run only when every earlier check passed. The `prop-`, `val-` and `exe-` counters are assigned in processing order, and proposals are processed in ascending `agent_id`. The storyboard and the CONTRACTS examples are regenerated to match: a core agent targeting `node-web-02` lists both `TARGET_OUT_OF_AUTHORITY` and `ACTION_NOT_AVAILABLE` where both apply, and the lower counter goes to the lower-sorting `agent_id`. |
| Encoded by | `coordinator/validator.py` (task 9.1); reason-code and ordering tests (task 9.2). |

### AD-22 | 2026-10-30 | Rule baselines re-investigate a contained compromised node (rule 2f)

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 18 (MEDIUM). |
| Decision | Both rule baselines (conventional and graph-feature) gain rule 2f so a contained compromised node is not left uncleared until truncation: a contained node whose evidence is null, 1 or 2 receives `investigate_node(n)` when `steps_since_investigated` is null or at least `investigate_cooldown`, otherwise `request_evidence(n)`, tagged with reason codes `evidence_insufficient` and `recovery_after_containment`. This lets a quarantined compromised node whose evidence expired be re-investigated within `investigate_cooldown + 1` steps, so the episode can reach `incident_resolved` and the baseline is not handicapped in every comparison. |
| Encoded by | `baselines/rule.py` and `baselines/graph_rule.py` (tasks 8.1, 8.2); re-investigation-reaches-`incident_resolved` test (task 8.3). |

### AD-23 | 2026-10-30 | `ShortText` drops `/ = + :` and rejects dotted-quads, `..` and a leading `-`

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 19 (MEDIUM). |
| Decision | `ShortText` is 1 to 160 characters restricted to `[A-Za-z0-9 .,()_-]`; the characters `/`, `=`, `+`, `:` (and quotes, backticks, pipes, ampersands, semicolons, dollar signs, angle brackets and backslashes) are not allowed. Additional validators reject a dotted quad (`\b\d{1,3}(\.\d{1,3}){3}\b`), the sequence `..`, and a value whose first character is `-`. So no text field (`rationale_text`, `label`, `description`, `note`, `message`) can hold an IP literal, a path such as `../etc/passwd`, a URL or a shell line like `rm -rf /`. Invalid fixtures `PolicyProposal.ip-in-rationale.json` and `PolicyProposal.path-in-rationale.json` cover it. (Already reflected in CONTRACTS §1 rule C8.) |
| Encoded by | `contracts/common.py` `ShortText` (task 3.1); invalid fixtures and tests (tasks 3.4, 3.5). |

### AD-24 | 2026-10-30 | Unseen-topology `eval_utopo` split and `topo-mock-small-v2`

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 3 (HIGH); new artifacts recorded here, the stale doc wording corrected under task 2.1. |
| Decision | An unseen-topology variant `configs/topologies/topo-mock-small-v2.json` is added with the same eight `node_id`s and the same two zones but a different `network_link` set, a different `service_dependency` DAG and different criticalities (for example the `ws-01`↔`file-01` link becomes `ws-01`↔`app-01`, the dependency `app-01`→`auth-01` becomes `db-01`→`auth-01`, and criticalities are changed: `web-02` 0.6, `file-01` 0.8, `ws-01` 0.4). Scenarios `scn-mock-lateral-utopo-01` and `-02` (`topology_id = topo-mock-small-v2`) and a split `eval_utopo` are added, and `eval_utopo` is added to the `Split` enum before schema 1.0 ships. Nothing on v2 is used for training, threshold selection or calibration; `assert_disjoint` additionally asserts that `train` and `cal` topology ids never appear in `eval_utopo`, enforced by a test. Checkpoint loading compares the node-id list and `ActionTable.digest()` (identical on v1 and v2), not the topology id, so a v1-trained checkpoint loads and runs on v2; the training and evaluation topology ids are recorded separately. The harness reports every metric separately for `eval` and `eval_utopo`. |
| Encoded by | `Split` enum (task 3.1); v2 topology, `utopo` scenarios and `splits.toml` (task 5.1); checkpoint node-id/action-table match (task 10.2); `splits.assert_disjoint` (task 12.1). |

### AD-25 | 2026-10-30 | `NodeScoreRecord` and the origin-ranking metric (top-1, MRR)

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 4 (HIGH); new artifacts recorded here, the stale doc wording corrected under task 2.1. |
| Decision | A registry record `NodeScoreRecord` is added (`schema_version`, `score_id`, `episode_id`, `step`, `agent_id`, `scorer_id` `ShortText` such as `detection.rule-v1` or `graph-rule-v1`, `scorer_kind` in {`detection_rule`, `graph_feature_rule`, `learned_graph_model`}, and `nodes` as a list of `{node_id, score: UnitFloat}` over the nodes the agent observes), streamed to `node_scores.jsonl`. It is produced by the detection role, by the graph-feature baseline (or `graph.features`), and by the learned graph model if that stage goes ahead. Metrics `origin_rank_top1` and `origin_mrr` are computed per scorer from evaluator-only truth: the origin node `o` is `incident.entry_node_id`; for each step `t` from `first_true_alert_step` to `containment_step` (or the last step if containment is never reached) and each agent whose observation contains `o`, take the rank of `o` among that agent's scored nodes by descending score (ties take the worst rank of the tie group); `origin_rank_top1` is the fraction of (step, agent) pairs with rank 1 and `origin_mrr` is the mean of `1/rank`, both null when there is no pair. `EpisodeResult.outcome` gains nullable `origin_rank_top1`/`origin_mrr` fields keyed by `scorer_id`. Everywhere this is labelled a ranking of predicted suspicion, never a root-cause or causal finding. Root-cause ranking is therefore an evaluated, secondary metric, which complements the task-2.1 correction of ARCHITECTURE §3 and §16 and of AD-01 decision 3. |
| Encoded by | `NodeScoreRecord` and `results` fields (task 3.1); detection role and baselines emit scores (tasks 6.1, 8.1, 8.2); `evaluation.outcome` origin metrics (task 12.1). |

### AD-26 | 2026-10-30 | Execution-derived manifest status fields

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 1 (HIGH), status-field part; the GNN/MAPPO staging itself is AD-11/AD-10 (updated under task 2.1). |
| Decision | The `RunManifest` status fields `ippo`, `learned_gnn` and `mappo` take values in {`not_run`, `implemented`, `no_go`, `deferred`} with an optional `ShortText` `reason` that is required when the value is `no_go` or `deferred`; the earlier `const` values and the `src/`-token ban are removed. The status is computed from what the run actually executed and is never hard-coded; `test_manifest_status` (which replaces the old `test_status_honesty`) asserts that the manifest status equals the executed outcome. A later go on the GNN or on MAPPO therefore needs no contract change. |
| Encoded by | `contracts/manifest.py` (task 3.1); runner `run_manifest` export (task 12.2); `tests/runner/test_manifest_status.py` (task 10.4). |

### AD-27 | 2026-10-30 | One evaluator-only import allowlist, with `evaluation.calibration_labels`

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 15 (MEDIUM). |
| Decision | There is a single evaluator-only import allowlist, `DESIGN` §3.2, which CONTRACTS §9 and ARCHITECTURE §9 point to rather than restating. The packages permitted to import evaluator-only ground truth are `rl.reward`, `evaluation`, `runner`, `engine.step_engine`, `interfaces.simulator`, `interfaces.ports`, `mock.*`, and the test suite; only `engine.step_engine` and tests call `ground_truth()`/`termination_reason()`. `uncertainty.calibration` must not import ground truth, so `evaluation.calibration_labels(proposals, truth) -> (u, correct)` produces the per-proposal uncertainty and correctness arrays and the runner passes them into calibration. The import-boundary test enforces this one list and the new function. |
| Encoded by | `tests/integration/test_import_boundaries.py` (task 4.2); `evaluation.calibration_labels` and `uncertainty.calibration` (task 11.2). |

### AD-28 | 2026-10-30 | The IPPO/MAPPO training CLI lives under `runner/`, not `marl/`

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 16 (MEDIUM). |
| Decision | The training entry points live in `runner/`, not `marl/`, because `marl` may import only `contracts`, `interfaces`, `common`, `rl`, `engine`, `numpy` and `torch`, while a training command must build the mock simulator, the real coordinator and the reward model (only `runner.factory` imports the mock). The IPPO CLI is `python -m cyberswarm.runner.train_ippo --config ... --out ...`, and the later MAPPO CLI lives under `runner/` likewise; `marl.ippo.IPPOTrainer(env, config, out_dir)` is unchanged. This keeps the import-boundary test passing without a special-case exception. |
| Encoded by | `runner/train_ippo.py` (task 10.3); import-boundary test (task 4.2). |

### AD-29 | 2026-10-30 | Keep failed runs; never tune on `eval` or `eval_utopo`

| Field | Entry |
|-------|-------|
| Status | Decided (review reconciliation). |
| Resolves | Design review finding 17 (MEDIUM). |
| Decision | Failed and underperforming runs stay in the evaluation outputs; they are reported, not hidden, and no tuning, threshold selection or model choice ever uses `eval` or `eval_utopo` (calibration uses `cal` only). Because episodes from differently seeded IPPO networks are not independent samples of one policy, `EvaluationMetrics` gains `policy_family_id` and `training_seed` (nullable), and an `AggregateMetrics` record reports, per `(policy_family_id, representation, split)`, the per-training-seed means plus the mean, sample standard deviation and percentile bootstrap interval across training seeds (unit = training seed; the interval is null below 5 seeds). The `ComparisonRecord` pairs are enumerated: `rule-conv` vs `rule-graph`; `ippo-conv` vs `ippo-graph`; `ippo-conv-ua` vs `ippo-conv-ctl`; `rule-conv` vs `ippo-conv`; `rule-graph` vs `ippo-graph`; each reported on `eval` and on `eval_utopo`. |
| Encoded by | `evaluation.compare`/`AggregateMetrics` and `stats` (task 12.1); `runner.evaluate` keeps failed runs and reports `eval`/`eval_utopo` separately (task 12.2). |
