# CyberSwarm Research Basis

**Status:** Version 0.2 (research delta). Version 0.1 was the Phase 1-2 research record. This version keeps that content and extends and corrects it, following the user's confirmed requirements in `.agents/tasks/cyberswarm-foundation/DELTA.md`.
**Fetched and checked:** 2026-10-30 (system clock of the machine that ran the checks). The delta re-fetched the primary source for every key claim on the same date.
**Scope:** What prior work exists, what its code actually does, what could and could not be reproduced, what technical foundations fit this project, what CyberSwarm proposes to add, and what is still unknown.

## What changed in version 0.2

| Change | Where |
|--------|-------|
| "CMARL-ACD" is kept separate from CyMARL. It is identified as the name of a GitHub repository and of a framework in a JAIST master's thesis. The journal paper it is said to accompany was **not** located. | Section 0 |
| Added the research comparison table (sources, algorithms, environment, observation and actions, rewards, metrics, code, verified yes/no, relevance). | Section 0.4 |
| Added the reproducibility assessment and the foundation comparison against CyberSwarm's requirements and the local Python. | Sections 2.5, 2.6 |
| Added a second CMARL-ACD source (the thesis) and the CyMARL code details that were missing. | Sections 1.2, 2.1 |
| Corrections to 0.1: TorchRL 0.14.1 does ship IPPO and MAPPO losses; BenchMARL lists IPPO; the CC4 pins resolve on Python 3.12; only three of the five CMARL-ACD defender action classes are active; the best_models claim that MAPPO is better is ambiguous. | Sections 1.2, 2.2, 2.3, 2.6 |
| The confirmed direction (multi-agent RL is primary; the GNN / Faultline concept is not central; graph representation and hand-engineered graph features are required lead-owned infrastructure; a learned GNN is a staged, secondary hypothesis) is reflected in the proposed extensions, the foundation comparison and the open questions. This follows the current `DELTA.md` (see the note on its two versions in `docs/DECISIONS.md` AD-01). | Sections 2.6, 3, 4; `docs/DECISIONS.md` entry AD-01 |

## How to read this document

Every factual statement carries one of four tags.

| Tag | Meaning |
|-----|---------|
| VERIFIED | I fetched the primary artifact in this session (paper text, repository file, registry record, documentation page) and the statement matches it. |
| REPORTED | A source states it (for example a README claim or a paper's own result). I did not check it independently and did not reproduce it. |
| UNVERIFIED | I could not fetch or confirm it. It is listed so nobody mistakes it for a finding. |
| INFERENCE | My reading. It follows from verified facts but no source states it. |

Nothing in this document is a result produced by CyberSwarm. No third-party code was executed or cloned. Files were read through raw-file and API URLs. The only things run were `pip install --dry-run` dependency resolutions inside throwaway virtual environments under `/tmp` (deleted afterwards). Nothing was installed into the system Python or into the repository. The method is in Appendix C.

**Framing rule for the whole project.** CyberSwarm's difference from prior work is a hypothesis to be tested by measured experiments. It is not "novel", not "first", and not "better". Section 3 states each proposed difference as a falsifiable hypothesis, and Section 1.6 lists prior work that already overlaps with several of them.

---

## 0. The "CMARL-ACD" question, kept separate from CyMARL

### 0.1 Finding

"CMARL-ACD" and "CyMARL" are different works. They share only the broad topic of multi-agent reinforcement learning for cyber defence.

| Artifact | What it is | Tag |
|----------|------------|-----|
| arXiv:2310.05939, "Learning Cyber Defence Tactics from Scratch with Multi-Agent Reinforcement Learning" (Wiebe, Al Mallah, Li) | The paper that introduces **CyMARL**. Workshop paper, 2023. | VERIFIED |
| GitHub repository `cyb3rlab/CMARL-ACD`, titled "CMARL-ACD: Competitive Multi-Agent Reinforcement Learning for Autonomous Cyber Defense" | Code from the Cybersecurity Research Lab at JAIST (Japan Advanced Institute of Science and Technology). Every exact-name search returned this one repository. | VERIFIED |
| JAIST master's thesis, hdl 10119/20030: Thanh Cong Nguyen, "Automated Cyber Defense Based on Reinforcement Learning Techniques", supervisor Razvan Beuran, dated August 2025 (repository issue date Sep-2025) | The thesis text calls its framework "CMARL-ACD (Competitive Multi-Agent Reinforcement Learning for Autonomous Cyber Defense)" (6 occurrences of "CMARL" in the extracted PDF text) and says the code and testbed configurations would be released as open source. | VERIFIED |
| "CMARL-ACD: Competitive Multi-Agent Reinforcement Learning for Autonomous Cyber Defense Under IDS Uncertainty Conditions", *Computers & Security* (under review), 2026 | A journal paper named in the repository README as the one the code accompanies. | REPORTED that it exists as a submission; the paper itself is UNVERIFIED (not found, Section 0.3) |

**Answer.** CMARL-ACD can be identified as a repository and as the name of a thesis framework. It cannot be identified as a published or peer-reviewed paper. It is not CyMARL, and nothing here assumes the two are the same.

**What differs.** Authors and institutions (RMC and DRDC Canada for CyMARL; JAIST for CMARL-ACD). Year (2023 workshop paper; 2025 thesis and 2026 repository). Setting (cooperative defenders against a heuristic attacker; versus a competitive attacker and defender trained against each other). Algorithms (value-based IQL and QMIX; versus clipped PPO as IPPO and MAPPO). Simulator (CybORG extended; versus a fork of NASim). Uncertainty (none modelled in CyMARL; simulated IDS false positives in CMARL-ACD). Details are in Sections 0.4, 1.1 and 1.2.

**What is still an inference.** The user's brief calls CMARL-ACD "the referenced research starting point". That the user means this repository or thesis is INFERENCE, because the name matches exactly and its IPPO and MAPPO pairing and IDS-uncertainty theme match the brief. Section 4, item 1 asks the user to confirm it.

### 0.2 Verdict on each claim in the earlier (orchestrator) finding

Re-checked on 2026-10-30. The CyMARL rows are unchanged from 0.1; the repository row is refined by the thesis finding.

| Earlier claim | Verdict |
|---------------|---------|
| arXiv 2310.05939 exists with this title and these authors (Wiebe, Al Mallah, Li) | Confirmed (arXiv abstract page and API). |
| Royal Military College of Canada and Defence Research and Development Canada | Confirmed from the PDF header. |
| Introduces CyMARL on CybORG (CAGE Challenge 2 base) with a PyMARL interface | Confirmed. |
| 9 hosts (5 user subnet, 4 operational subnet); importance 0.1 / 1 / 10 | Confirmed. |
| Two defender agents split by subnet; heuristic attacker; confidentiality, integrity, availability scenarios | Confirmed. |
| IQL and QMIX (not PPO), compared with a heuristic defender | Confirmed. The heuristic defender reads the true game state, which the paper states. |
| Actions monitor / remove / restore / analyze / data_repair / misinform; invalid actions penalized -0.1 | Confirmed. |
| Observation is six flags per host, one-hot encoded | Confirmed. The paper says "six flags ... one-hot encoded into a vector" and does not give the vector size. |
| "No public CyMARL code repository was found" | **Refuted.** `Poly-AIvsAI/CyMARL` exists (Section 2.1). |
| "CMARL-ACD has NOT been verified as a paper or repo name" | **Refuted for the repository name and for the thesis.** The journal paper was not located. |
| Venue (the earlier finding gave none) | The arXiv record says "Presented at 2nd International Workshop on Adaptive Cyber Defense, 2023"; its journal-reference field reads `ACD/2023/106`. |

### 0.3 Search log for the exact string "CMARL-ACD"

All searches were run on 2026-10-30. A negative result from an index is not proof that nothing exists.

| Index or method | Queries | Result |
|-----------------|---------|--------|
| GitHub repository search (unauthenticated API) | `CMARL-ACD`; `CMARL-ACD in:name`; `CMARL ACD`; `CMARL-ACD in:readme`; `competitive multi-agent reinforcement learning autonomous cyber defense`; `CMARL cyber`; `CMARL defense` | Each returned `total=1`: `cyb3rlab/CMARL-ACD` (created 2026-06-03). |
| same | `cmarl autonomous cyber defense IPPO MAPPO`; `IDS uncertainty multi-agent reinforcement learning cyber defense IPPO MAPPO` | 0 results. |
| same | `CMARL in:name,description` | 91 results. The ones inspected are unrelated (for example flexible job-shop scheduling, certified robustness, anomaly detection in cooperative MARL). Three newer repositories with no description were checked: one is about job-shop scheduling; the other two have no readable README (their subject is unknown, and nothing suggests cyber defence). |
| same | `CMARL defence`; `cmarl-acd`; `ACD cmarl in:readme cyber` | **Not completed**: the unauthenticated rate limit was reached. |
| arXiv API | `all:"CMARL-ACD"`; `all:CMARL AND all:cyber`; `all:"CMARL ACD"`; `ti:"competitive multi-agent reinforcement learning" AND abs:cyber`; `all:"IDS uncertainty" AND all:reinforcement`; `ti:"autonomous cyber defense" AND abs:"competitive multi-agent"`; `au:Beuran AND all:reinforcement`; `au:Nguyen AND au:Beuran` | 0 relevant results (the one hit, arXiv:2410.14121, is an IoT intrusion-detection paper). |
| Crossref | Bibliographic queries `CMARL-ACD`, the full journal title, and authors plus venue; a filtered query on *Computers & Security* (ISSN 0167-4048) from 2026 | No CMARL-ACD record. Nearest hits are unrelated (for example a 2026 *Expert Systems with Applications* paper by Zhang, Goel and Ahmad on adversarial MARL; different authors). |
| OpenAlex | `CMARL-ACD`; the full repository title | The first query returned 3 works. One is a 2025 work titled "Automated Cyber Defense Based on Reinforcement Learning Techniques", the same title as the JAIST thesis (likely the thesis record; not cross-checked); the others are unrelated. No journal-paper record. |
| DBLP | `CMARL-ACD` and two phrase queries | **Not searched**: the endpoint returned an anti-bot challenge page. |
| Semantic Scholar API | `CMARL-ACD ...` | **Not searched**: HTTP 429 (rate limited). |
| Zenodo; PyPI; Hugging Face | `CMARL-ACD`; `cmarl-acd`, `cmarl_acd` | 0 Zenodo records; both PyPI names return 404; the Hugging Face model search returned an empty list. |
| Web search (DuckDuckGo HTML endpoint) | exact `"CMARL-ACD"` | 8 results: 7 pages of the `cyb3rlab/CMARL-ACD` repository and 1 result, the JAIST thesis PDF. |
| same | the exact journal-paper title in quotes | 0 results. |
| Web search (Bing HTML) | same queries | Unusable: results were unrelated pages. Ignored. |
| JAIST Cybersecurity Research Lab website (publications page, fetched) | n/a | The page lists no CMARL-ACD entry. It may be out of date. |
| JAIST repository | thesis item page and PDF | Fetched from `dspace.jaist.ac.jp` (HTTP 200). The newer `jaist.repo.nii.ac.jp` record page returned HTTP 503 with a maintenance notice (from 28 Oct 2026), so the `hdl.handle.net` link could not be followed. |
| ScienceDirect | n/a | HTTP 403 to automated fetches; nothing could be read there. |

**Name collision to avoid.** "Collaborative multi-agent reinforcement learning (C-MARL) for automated red teaming in large-scale heterogeneous networks" (*Ain Shams Engineering Journal*, 2026, DOI 10.1016/j.asej.2026.104323, metadata read from Crossref) is a different work on offensive red teaming. It is not CMARL-ACD and is not used here. VERIFIED (metadata only).

### 0.4 Research comparison table

"Verified" means I read the primary artifact named in that row on 2026-10-30. "Not examined" means I did not read that part. All performance figures belong to their authors and are not CyberSwarm results.

| Source | What it is | Algorithms | Environment | Observation / actions | Rewards | Metrics | Code availability | Verified | Relevance to CyberSwarm |
|--------|------------|------------|-------------|-----------------------|---------|---------|-------------------|----------|-------------------------|
| **arXiv:2310.05939 (CyMARL)**. Wiebe, Al Mallah, Li, 2023 | Workshop paper (ACD 2023) plus a PyMARL-interfaced extension of CybORG | IQL and QMIX (PyMARL2 implementations; RNN agents) against a heuristic defender that reads the true state | CybORG (CAGE-2 base). 9 hosts, 2 subnets. Two defender agents, one per subnet. One heuristic attacker (confidentiality, integrity or availability goal) | 6 one-hot flags per host (scanned; exploit run; no malicious session; unknown; user session; root session). Actions monitor, remove, restore, analyze, data_repair, misinform. Invalid action -0.1 | Each compromised host costs the team its importance (0.1 user, 1 operational, 10 server). Restore penalty scales with importance. One shared team reward | Mean return and standard deviation. Score = average of the mean return of five evaluation runs, each using a separately trained static policy, over 1000 evaluation timesteps. Training curves | `Poly-AIvsAI/CyMARL` (no licence; link to the paper not stated by the paper; identity inferred). No code link in the paper | Yes: full paper text and repository files | Closest cooperative defender-MARL prior. Importance-weighted reward and a heuristic baseline are close to our design. Value-based, so not a PPO reference. Authors found IQL close to QMIX because the game needs little coordination |
| **CMARL-ACD**: repo `cyb3rlab/CMARL-ACD` and JAIST master's thesis (Nguyen, Beuran). Journal paper not located | Open-source code plus a thesis; the code is said to accompany a *Computers & Security* submission | IPPO and MAPPO (clipped PPO in TorchRL); `CRITIC_CENTRALIZED` flag selects MAPPO. Attacker and defender co-train; checkpoints form a pool for worst-case evaluation | Vendored NASim fork (Gymnasium). Competitive attacker and defender. Defender sees the network through a simulated IDS with SSH false positives. Thesis: 5 hosts, 4 subnets plus an internet-facing subnet, 150-step episodes | Defender: per-host rows (value, access, OS, services, processes, monitoring, decoys) plus 5 alert slots. Active actions: `decoy_services`, `ip_block` (IP1 to IP5), `noop`; invalid actions masked | Defender: action value minus cost, plus block reward or penalty, minus a time-weighted penalty per compromised host, plus a decoy bonus (thesis Eq. 3.3 to 3.6). The thesis calls the setting zero-sum with asymmetric objectives | Hosts Compromised, Decoy Interaction Ratio, True Block Rate; worst-case score and exploitability over checkpoint pools | Yes (MIT; GitHub reports licence "Other"). Includes a real-network testbed that must not be used | Repository: yes. Thesis: yes. Journal paper: **no** | The only IPPO and MAPPO plus IDS-noise prior found. TBR is a precision measure for blocking, close to our false-isolation concern. Competitive attacker, IP-block actions and real-testbed tooling are outside our scope |
| **CAGE Challenge 4 / CybORG**. Repos `cage-challenge/cage-challenge-4`, `cage-challenge/CybORG`; AAAI 2025 paper (Kiely et al.); AI Magazine 46(3) article | Public challenge environment, plus papers on it | Challenge is open to any method. `Requirements.txt` lists RLlib, sb3-contrib, PettingZoo and torch_geometric. The AAAI paper's own algorithms were not read | 5 blue agents (one per security zone; one covers three zones), red `FiniteStateRedAgent` strategies, green users who also raise false alerts, mission phases that change zone priority | `BlueEnterpriseWrapper` vector (mission phase, per-subnet info, four 8-bit messages). Actions Monitor, Analyse, DeployDecoy, Remove, Restore, BlockTraffic, AllowTraffic | Penalties when green work fails, when a compromised service is accessed and when red uses Impact; larger in mission-critical zones | Mean reward and standard deviation over 100 randomised episodes of 500 steps against `FiniteStateRedAgent` | Yes: public, MIT licence text in the CC4 repo; CybORG README says MIT | README, requirements and licence: yes. AAAI landing page and Crossref metadata: yes. AI Magazine full text: **no** (403) | Closest maintained analogue for "keep legitimate work running while containing". Design reference, not a dependency |
| **Yu et al. 2021, arXiv:2103.01955** and `marlbenchmark/on-policy` | NeurIPS 2022 Datasets and Benchmarks paper and its official code | MAPPO (PPO with a centralized value input) and IPPO (local inputs). GAE, value normalization | SMAC, Google Research Football, Hanabi, multi-agent particle world | Environment-specific (not cyber) | Shared team reward (Dec-POMDP with shared rewards) | Win rate or return per benchmark | `marlbenchmark/on-policy` (MIT). README installs Python 3.6.1 and torch 1.5.1 | Yes: paper text for definitions and suggestions; README | Algorithmic reference for IPPO and MAPPO and for the practical suggestions. Its shared-reward assumption needs a decision for our environment |
| **PyMARL2** (`hijkzzz/pymarl2`) with Hu et al., arXiv:2102.03479 | Fine-tuned value-based MARL code base and the paper on its tricks | QMIX family, VDN, QPLEX and others | SMAC (plus football and predator-prey configs) | Environment-specific | Environment-specific | Win rate | Yes (Apache-2.0). README installs Python 3.8 | README and metadata: yes. Hu et al.: abstract only | The framework CyMARL builds on. Value-based; no cyber content |
| **Yawning Titan** (`dstl/YAWNING-TITAN`) and arXiv:2207.12355 | Abstract graph-based cyber simulation, defender-only, against probabilistic red agents | Integrates Stable Baselines 3 and RLlib. The paper uses dynamic causal Bayesian optimisation as a blue agent | Abstract graph network | Not examined | Not examined | Not examined | Yes (MIT), **archived**; badge Python 3.8 to 3.10 | README, metadata and abstract: yes. Internals: no | Design reference for an abstract graph simulator. Not a dependency |
| **CyberBattleSim** (`microsoft/CyberBattleSim`) and Kunz et al., arXiv:2304.11052 | Attacker-focused enterprise-network simulation with a basic scripted defender; the paper adds trainable blue agents | Not examined for the base repo. Kunz et al. train blue agents alone or jointly with red | Abstract nodes and vulnerabilities, Gym interface. Defender example keeps network availability above an SLA (`maintain_sla=0.80`) | Not examined | Not examined | Not examined | Yes (MIT), active (last push 2026-10-30). Linux or WSL recommended; no macOS statement | README, metadata and abstract: yes. Internals: no | Low for defender learning. The SLA idea is a reminder to measure service availability |
| **arXiv:2606.13832** (Silva, 2026) and **arXiv:2401.10149** (Wilson et al., CAMLIS 2023) | Two papers that overlap with our proposed differences | IPPO, MAPPO-GAT, C-MAPPO-GAT, ACD3-GAT; versus shared-critic MAPPO and IPPO | CAGE Challenge 4; versus the IPMSRL maritime OT simulation | Not examined | Not examined | Downtime budgets, violation rate; episode outcome mean | Not examined | Abstracts only | Show that graph encoders, safety budgets, tail-risk estimation and MAPPO-over-IPPO results already exist. We cannot call these ideas new |

---

## 1. Verified existing research

### 1.1 CyMARL (Wiebe, Al Mallah, Li)

| Item | Detail | Tag |
|------|--------|-----|
| Title | Learning Cyber Defence Tactics from Scratch with Multi-Agent Reinforcement Learning | VERIFIED |
| Authors | Jacob Wiebe, Ranwa Al Mallah (Royal Military College of Canada), Li Li (Defence Research and Development Canada) | VERIFIED |
| Identifier | arXiv:2310.05939v1 [cs.CR]; v1 dated 25 Aug 2023; subjects cs.CR, cs.AI, cs.LG. The arXiv page lists one version. | VERIFIED |
| Venue | arXiv comment: "Presented at 2nd International Workshop on Adaptive Cyber Defense, 2023 (arXiv:2308.09520)"; journal-reference field `ACD/2023/106`. The linked record is "Proceedings of the 2nd International Workshop on Adaptive Cyber Defense" (arXiv:2308.09520; the arXiv record lists Carvalho, Marriott, Bilinski and Ridley). It is a workshop paper. No journal or conference publication was found. | VERIFIED |
| Authors' claim | "To the best of our knowledge, CyMARL is the first cooperative MARL training environment for enterprise network defence tasks." This is the authors' claim. CyberSwarm does not repeat or build on it. | REPORTED |
| Environment | CybORG extended with new game types, actions and topologies, plus a PyMARL(2) interface. CAGE Challenge 2 supplies the attacker models, topology and base actions. | VERIFIED |
| Topology | 9 hosts: 5 in a user subnet, 4 in an operational subnet; mixed Linux and Windows. | VERIFIED |
| Host importance | User hosts 0.1, operational hosts 1, operational server 10. Each compromised host costs the defender team a negative reward equal to its importance score. | VERIFIED |
| Attacker | One heuristic attacker with a foothold on a user host. It discovers a host, then a port, then moves laterally with an exploit; it can escalate privilege. Its session on the initial host cannot be removed. | VERIFIED |
| Scenarios | By attacker goal: confidentiality (spy), integrity (a `tamper` action modifies files), availability (a `deny` action starts a DoS process). In integrity and availability, a host counts as compromised only after the second action. | VERIFIED |
| Agents | Two defender agents, one per subnet. They do not communicate directly. | VERIFIED |
| Observation | Six flags per host in the agent's area: scanned last turn; exploit run on a service last turn; no malicious sessions; status unknown; malicious user-privilege session; malicious root/admin session. The authors state the observation does not alert on malicious files or processes. | VERIFIED |
| Actions | `monitor`, `remove`, `restore` (penalty scales with host importance), `analyze` and `data_repair` (integrity only), `remove` also for DoS processes (availability), `misinform` (decoy service; second set of trials). Invalid actions get reward -0.1. | VERIFIED |
| Heuristic defender | Assumed to have "a tailored expert system" giving a precise signal of malicious process creation; it relies on selected information from the true state, not on the MARL observation. Confidentiality: monitor all hosts in the subnet and restore the host named by the alert. Integrity: remove malware with 50% probability, else restore. | VERIFIED |
| Algorithms | IQL (PyMARL2 implementation: DQN-style agents with an RNN) and QMIX. | VERIFIED |
| Protocol | 2 million training timesteps. Evaluation over 1000 timesteps without learning. Score = mean of five evaluation runs using separately trained static policies. Hyperparameter grid (500k timesteps, 3 seeds, confidentiality scenario only): batch size {128, 256}, buffer {5000, 10000}, learning rate {0.005, 0.01}, TD-lambda {none, 0.6}. | VERIFIED |
| Reported outcome | Both IQL and QMIX scored better than the heuristic defender in all six scenario variants. QMIX learned slightly faster in the confidentiality and integrity scenarios; the final scores of IQL and QMIX were close. | REPORTED |
| Reported scores (Table 1, mean return ± standard deviation; returns are negative penalties, so closer to zero is better) | Confidentiality: IQL -26.19 ± 5.5, QMIX -23.01 ± 2.47, heuristic -65.96 ± 24.33. Integrity: -24.38 ± 5.44, -16.8 ± 2.73, -32.97 ± 20.82. Availability: -15.94 ± 3.78, -17.5 ± 4.68, -23.11 ± 18.88. With misinform: confidentiality -19.76 ± 4.18, -25.07 ± 7.1, -67.64 ± 24.03; integrity -18.17 ± 3.86, -16.75 ± 3.48, -31.9 ± 20.73; availability -14.47 ± 2.98, -11.63 ± 2.84, -21.39 ± 16.95. These are the authors' numbers, not reproduced. Note the heuristic's standard deviations are large relative to its gaps to IQL and QMIX in the integrity and availability rows. | REPORTED |
| Authors' own caveat | "The multi-agent host-monitoring game presented does not require explicit coordination for a defender team to be successful." This is the reason IQL came close to QMIX. | VERIFIED (quote) |
| Not stated in the paper text | Episode length; software versions; any link to CyMARL code. A keyword search of the full PDF text found no repository URL for CyMARL (it cites only the PyMARL repository). The code default for episode length is in Section 2.1. | VERIFIED (absence in text) |

Relevance to CyberSwarm: a small, importance-weighted topology, per-host partial observation and a heuristic baseline are close to our setting. The environment is attack-and-defence at the host-process level, which is more detailed than the abstract simulator specified in the brief. The finding that independent learners matched a centralized method in a weakly coupled game is a prior that IPPO may be competitive with MAPPO here as well. It is not a claim about our environment.

### 1.2 CMARL-ACD (Nguyen, Beuran)

Two sources exist: the repository (1.2.1) and the master's thesis (1.2.2). The journal paper was not located (Section 0.3), so its methods, results and claims are UNVERIFIED. The README's framing is REPORTED.

#### 1.2.1 Repository `cyb3rlab/CMARL-ACD`

| Item | Detail | Tag |
|------|--------|-----|
| Setting | Competitive: an attacker agent and a defender agent co-train. README: the defender "perceives the network only through a simulated IDS that emits noisy alerts (SSH false positives) and benign background traffic". | REPORTED |
| Owner and authors | Organization `cyb3rlab` ("Cybersecurity Research Lab @ JAIST" per the GitHub profile). LICENSE.md: "Copyright (c) 2026 Thanh Cong Nguyen, Razvan Beuran". README cites "T. C. Nguyen and R. Beuran". All four commits are by one account, `rzvnbr`. | VERIFIED |
| Accompanying paper | README: "CMARL-ACD: Competitive Multi-Agent Reinforcement Learning for Autonomous Cyber Defense Under IDS Uncertainty Conditions", *Computers & Security* (under review), 2026. | REPORTED |
| Algorithms | Clipped PPO via TorchRL. `CRITIC_CENTRALIZED = False` gives IPPO (default); `True` gives MAPPO. The README stresses "a single trainer, not two". | VERIFIED (script constants) |
| Agents | `POLICY_N_AGENTS = 2` (attacker, defender); `POLICY_SHARED_PARAMS = False`; `CRITIC_SHARED_PARAMS = False`. | VERIFIED |
| Network and PPO settings | MLP [256, 512, 512, 256] with Tanh; 2048 frames per batch; 10 PPO epochs; minibatch 64; learning rate 3e-4; grad-norm clip 1.0; clip epsilon 0.2; gamma 0.99; GAE lambda 0.95; entropy coefficient 0.1; default `TOTAL_FRAMES_TRAINING` 30,000,000 (a constant in the script). | VERIFIED |
| Environment | A vendored fork of NASim (Network Attack Simulator, MIT) at `simulation/nasim`, version 0.12.0 in `setup.py`. Gymnasium-based. Default benchmark scenario in the script: `tiny-small`. | VERIFIED |
| Scenarios | Scenario 1 "Clean Baseline" (SSH false-positive rate 0.0) and Scenario 2 "Operational Noise" (rate 1.0), selected by the `SCENARIO` environment variable. The environment takes a per-service false-positive probability (`service_fp_rates`) and draws `random.random() < fp_rate` per alert. What a rate of 1.0 means for every SSH event is not explained in the files I read. | VERIFIED (names, values, code path) |
| Defender observation | One row per host: host value, discovery value, access level, OS, services, processes, monitoring flag, decoy services, and five alert slots (IP1 to IP5) with one flag per service. | VERIFIED (printing helper in `train_marl.py`) |
| Defender actions | `action.py` defines `IPBlock` (`ip_block`), `DecoyServices` (`decoy_services`), `ToogleServices` (`disable_services`), `ToggleMonitoring` (`toggle_monitoring`) and `NoOpDefender` (`noop`). **Only `DecoyServices`, `IPBlock` (targets IP1 to IP5) and `NoOpDefender` are added to the defender action list; the lines adding `ToogleServices` and `ToggleMonitoring` are commented out.** This matches the thesis, which lists three defender actions. Invalid actions are masked with TorchRL `ActionMask` and `MaskedCategorical`. | VERIFIED (source) |
| Defender reward terms | Defender reward = reward from its own action, plus a per-step time penalty for each compromised host, `-min(host_value * alpha * exp(-beta * dt), r_max)` (environment defaults alpha 1.0, beta 0.1, r_max 100; the training script sets `RMAX` default 10.0), plus a penalty for blocking wrongly (default 40.0, `PENALTY`), plus a bonus when the attacker interacts with a decoy (default 20.0). A constant `reward_factor_defender_correct_block = 0.5` exists; I did not read how it is used. Rewards are normalized with TorchRL `RewardNormalizer`. As written (and in thesis Eq. 3.5) the exponent makes the penalty per host decay as `dt` grows. I did not analyze the intent. | VERIFIED (source) |
| Evaluation | `evaluate_vs_baselines.py`: four matchups (trained defender vs random attacker; self-play; random attacker vs NoOp defender; trained attacker vs NoOp defender), 1000 episodes by default. `crosseval_*_checkpoints.py`: an N-by-N tournament over a checkpoint set, producing a "worst-case score" and an "exploitability score". | VERIFIED (README, filenames) |
| Reported outcomes | `best_models/README.md` lists worst-case and exploitability scores per checkpoint set. Scenario 1: worst-case 9.45 for both IPPO and MAPPO. Scenario 2: IPPO -29.66, MAPPO -64.81. It then states "MAPPO performs better than IPPO" in Scenario 2 without saying whether a higher or lower score is better. The thesis (Section 1.2.2) reports the opposite ordering for hosts compromised against worst-case attackers under noise (IPPO about 2, MAPPO about 3). I could not reconcile the two self-reports from the files read. The README also notes that early checkpoints (100K to 600K frames) often beat heavily trained ones. Checkpoints were chosen by the same cross-evaluation that yields the scores, so the scores may be optimistic. | REPORTED |
| Real-network validation | README claims the trained policies were validated, without retraining, on a KVM testbed with Snort and Metasploit. | REPORTED |
| Seeding | `seed_everything` seeds `random`, `numpy` and `torch`. Each environment instance draws its scenario seed with `random.randint(0, 1_000_000)`. Bitwise reproducibility was not tested. | VERIFIED (source) |
| Dependencies | `simulation/requirements.txt` pins `torch==2.7.0`, `torchrl==0.8.0`, `tensordict==0.8.2`, `gymnasium==1.1.1`, `numpy==2.0.2`, `networkx==3.2.1`, `pandas==2.2.3`, `tensorboard==2.19.0` (58 lines in all). `setup.py` sets `python_requires=">=3.9,<3.13"`. | VERIFIED |
| Python 3.13 | The README says numpy 2.0.2 and torch 2.7.0 have no 3.13 wheels. PyPI file lists show numpy 2.0.2 has no cp313 wheel but torch 2.7.0 does. The numpy pin and the `setup.py` cap are what block 3.13. A dry-run of the full pinned file found a solution on Python 3.12.12 and none on 3.13.2 (Section 2.5). | VERIFIED |
| API drift | `train_marl.py` imports `CompositeSpec`, `UnboundedContinuousTensorSpec`, `DiscreteTensorSpec`, `BinaryDiscreteTensorSpec` from `torchrl.data`. The 0.14.1 wheel's `torchrl/data/__init__.py` does not mention these names; it exports `Composite`, `Unbounded`, `Categorical` and `Binary`. I did not run an import. | VERIFIED (wheel contents, v0.1) |
| Tests | No `tests/` directory appears in the 210-entry file tree. | VERIFIED |
| Licence | `LICENSE.md` is the MIT License with a notice that the project derives from NASim. GitHub's API reports licence "Other" (NOASSERTION). | VERIFIED |
| Repository state | Created 2026-06-03, last push 2026-06-23, four commits, no stars or forks at fetch time, about 62 MB (checkpoints are 4.7 to 5.2 MB each). | VERIFIED (API) |
| Out-of-scope content | `real-testbed/` drives Nmap, Metasploit and SSH brute-force generators and applies real iptables rules, and requires root on a provisioned Linux host. It also contains `passwords.txt` and `users.txt`, which I did not open. CyberSwarm must not run, adapt or vendor any of it. | VERIFIED (file tree, README) |

#### 1.2.2 Master's thesis (JAIST, hdl 10119/20030)

I downloaded the PDF (77 pages) and extracted its text. Quoted section numbers are the thesis's own. Figures are not reproduced here.

| Item | Detail | Tag |
|------|--------|-----|
| Identity | Thanh Cong Nguyen, "Automated Cyber Defense Based on Reinforcement Learning Techniques", Master's Thesis (Information Science), Graduate School of Advanced Science and Technology, JAIST. Supervisor and main examiner Razvan Beuran; examiners Yasuo Tan, Yuto Lim, Satoshi Uda. Cover date August 2025; repository issue date Sep-2025. | VERIFIED |
| Status | A degree thesis with named examiners. It is not a peer-reviewed journal paper. How it relates to the *Computers & Security* submission is not stated in the text I read. | VERIFIED / INFERENCE (relation) |
| Name | Chapter 2 (related work) introduces "CMARL-ACD (Competitive Multi-Agent Reinforcement Learning for Autonomous Cyber Defense), a novel framework". "Novel" is the author's claim; CyberSwarm does not repeat it. | VERIFIED (quote), REPORTED (claim) |
| Formulation | Competitive multi-agent POMDP with two agents. Each timestep the defender acts first, then the attacker. | VERIFIED |
| State and observation | Full state: topology, services, OS, attacker discovery state, privilege levels, decoys, IDS alerts. The defender sees configuration, host values, monitoring and decoy status and IDS alerts; it cannot see which hosts are compromised, what the attacker has discovered, or its privilege levels. | VERIFIED |
| Actions (Table 3.1) | Red: service scan, OS scan, subnet scan, exploit, privilege escalation, noop. Blue: deploy decoy service, block IP address, noop. Action masking keeps selections valid. | VERIFIED |
| Rewards (Eq. 3.1 to 3.6) | Attacker: value of discovered and compromised hosts (not decoys) minus action cost. Defender: action value, minus action cost, plus block reward or minus wrong-block penalty, minus a time-based penalty per compromised host, plus a decoy-trigger reward. | VERIFIED |
| Algorithms (Table 3.2) | Clipped PPO; GAE gamma 0.99, lambda 0.95; entropy bonus 0.1; Smooth L1 value loss; Adam, learning rate 3e-4. Separate actor and critic MLPs [256, 512, 512, 256] with tanh; no parameter sharing between attacker and defender. IPPO: each critic sees only its own agent's state. MAPPO: the centralized critic "accessing both agent information during training", which includes the opponent's information. | VERIFIED |
| Network used | Four internal subnets plus an internet-facing subnet 0; five hosts (Table 4.1: host rewards 10, 40, 70, 70, 100); real testbed runs all hosts as Linux. | VERIFIED |
| Training and evaluation | Episodes limited to 150 timesteps. Models saved every 100,000 timesteps to form agent pools. Each defender model is evaluated against an attacker pool for 50 episodes to get its worst-case score; exploitability is the distance to the best defender's worst case. Convergence took about 400,000 timesteps (baseline) and 3 million (operational noise). | VERIFIED |
| Metrics | Hosts Compromised (HC, unique hosts with at least one successful exploit; lower is better). Decoy Interaction Ratio (DIR, exploits that hit decoys over all exploits). True Blocking Rate (TBR, malicious IPs correctly blocked over all block actions; a precision of blocking). | VERIFIED |
| Reported results (Table 4.3, text) | Baseline scenario, random attacker: HC 3.802 with no defender; 0.64 with either IPPO or MAPPO; DIR 36%; TBR 100%. Against worst-case trained attackers both limited compromise to 1 host. Operational noise, worst-case attackers: HC 2 for IPPO and 3 for MAPPO. Real-testbed TBR against the random attacker in the noise scenario: IPPO 98.0% and MAPPO 100%. Simulation and real results are described as within 5% of each other. | REPORTED |
| Internal inconsistencies | The abstract and the body disagree: the abstract gives "approximately 79% reduction ... (from 3.8 to 0.8 hosts)" and says MAPPO's TBR degraded to 96.1% in realistic environments; the body and Table 4.3 give 0.64 hosts (83%) and put the 98.0% on IPPO with MAPPO at 100%. | VERIFIED (text) |
| Uncertainty mechanism | "Uncertainty" means IDS false positives in the observation. I found no agent-side uncertainty estimate: no calibration, ensemble, dropout or confidence measure (searches of the text for those terms found none; "entropy" appears once, as the PPO exploration bonus). | VERIFIED (absence in text) |
| Seeds and spread | No mention of random seeds, standard deviation or confidence intervals in the extracted text. | VERIFIED (absence in text) |
| Authors' limitations | Small topology; scalability untested; discrete action spaces; static environment; simple benign-traffic models. | VERIFIED |

Differences from CyberSwarm's brief, from the code and thesis: CMARL-ACD is competitive with a learning attacker. The brief calls for seeded scenario scripts (no learning attacker), several cooperating defenders, an explicit coordinator with an allowlist, and abstract containment actions with no IP strings (CMARL-ACD blocks generic IP slots IP1 to IP5, not real addresses, in the simulator). Its IPPO-versus-MAPPO contrast is over a two-agent attacker and defender pair, and its MAPPO critic sees the opponent. So a CyberSwarm IPPO-versus-MAPPO comparison is not a replication of it.

### 1.3 Related CybORG family: CybORG, CAGE Challenge 4

| Item | Detail | Tag |
|------|--------|-----|
| CybORG paper | Standen, Lucas, Bowman, Richer, Kim, Marriott. "CybORG: A Gym for the Development of Autonomous Cyber Agents", arXiv:2108.09118. arXiv record: IJCAI-21 1st International Workshop on Adaptive Cyber Defense. Abstract page and API record only; PDF not read. | VERIFIED (abs page) |
| CybORG repository | `cage-challenge/CybORG` is public, README header "Copyright DST Group. Licensed under the MIT license", last push 2024-06-03. GitHub's API reports licence "Other" (NOASSERTION). Wrappers include `PettingZooParallelWrapper`. | VERIFIED |
| CAGE Challenge 4 repository | `cage-challenge/cage-challenge-4`, LICENSE.txt "Copyright (c) Commonwealth of Australia 2019, MIT LICENSE", last push 2025-10-27. | VERIFIED |
| CC4 results paper | Kiely et al. (30 authors), "Exploring the Efficacy of Multi-Agent Reinforcement Learning for Autonomous Cyber Defence: A CAGE Challenge 4 Perspective", AAAI 2025 (vol. 39, no. 28), DOI 10.1609/aaai.v39i28.35158. Only the AAAI landing page and the Crossref record were read, not the paper text. | VERIFIED (landing page, metadata) |
| CC4 companion article | "CAGE challenge 4: A scalable multi-agent reinforcement learning gym for autonomous cyber defence", *AI Magazine* 46(3), September 2025, DOI 10.1002/aaai.70021. Metadata confirmed through Crossref. The Wiley page returns HTTP 403 and was not read. | VERIFIED (metadata), UNVERIFIED (content) |
| CC4 scenario | Five blue agents, each defending a security zone (one agent covers three zones); red starts on a contractor-network host and uses randomly selected strategies; green users generate legitimate work and occasional false alerts (a 1% `fp_detection_rate` is listed for `GreenAccessService`). Mission phases change which zones are high priority. Some blue agents can exchange 8-bit messages. | VERIFIED (README) |
| CC4 blue actions | Monitor, Analyse, DeployDecoy, Remove, Restore, BlockTraffic, AllowTraffic (listed in Appendix A of the README; other actions may exist). | VERIFIED (README) |
| CC4 reward | Penalties when green agents cannot work, when a compromised service is accessed, and when red uses `Impact`. Penalties are larger for zones that are mission-critical in the current phase. Blue is also penalized when its firewall changes block green work. | VERIFIED (README) |
| CC4 evaluation | 100 randomized episodes of 500 timesteps against `FiniteStateRedAgent`. | VERIFIED (README) |
| CC4 interface | `BlueEnterpriseWrapper` conforms to the PettingZoo environment API. Observation vector sized for up to 16 hosts per subnet (10 user, 6 servers) and 9 subnets. | VERIFIED (README) |
| CC4 pins | `Requirements.txt`: `gymnasium==0.28.1`, `pettingzoo==1.24.3`, `torch==2.2.0`, `numpy==1.26.4`, `sb3-contrib==2.2.1`, `ray[rllib]`, `torch_geometric==2.5.0`, plus `paramiko` and `sshtunnel`. Neither numpy 1.26.4 nor torch 2.2.0 has a cp313 wheel on PyPI. `setup.py` asserts Python 3.7 or newer. A dry-run resolves on Python 3.12.12 and fails on 3.13.2 (Section 2.5). | VERIFIED |

Relevance: CC4 is the closest maintained reference for "defend zones, keep legitimate work running, handle false alerts, penalize disruptive containment". CyberSwarm's own simulator is smaller and abstract by design (brief), so CybORG is a design reference, not a dependency.

### 1.4 Algorithm and framework sources

| Source | Detail | Tag |
|--------|--------|-----|
| Yu et al., "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games", arXiv:2103.01955 | Authors Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, Yi Wu. First posted 2021-03-02; arXiv comment: accepted at NeurIPS 2022 Datasets and Benchmarks. Full PDF read in 0.1; the definitions and the five suggestions were re-read in 0.2. | VERIFIED |
| Definitions in that paper | "PPO with centralized value function inputs as MAPPO ... and PPO with local inputs for both the policy and value function as IPPO". Studied as decentralized POMDPs with shared rewards. Parameter sharing is used when agents are homogeneous. GAE with advantage normalization and value clipping are used. | VERIFIED |
| Testbeds and finding | Particle-world, StarCraft multi-agent challenge, Google Research Football, Hanabi. PPO-based methods were competitive with off-policy methods. IPPO matched or came close to MAPPO in several settings; MAPPO showed a clearer margin in Hanabi as the number of players grew. | REPORTED |
| Practical suggestions | (1) value normalization; (2) include both local and global features in the value input; (3) at most 10 epochs on hard tasks, 15 on easy ones, and avoid mini-batch splitting; (4) clip ratio under 0.2; (5) large batch size. | VERIFIED (text of the paper) |
| Official MAPPO code | `marlbenchmark/on-policy`, MIT, last push 2024-07-18. README installs Python 3.6.1 with torch 1.5.1 and lists support only for StarCraft II (SMAC and SMAC v2), Hanabi, MPE and Google Research Football. The README itself warns that "a lot of papers do not reproduce the mappo results correctly, probably due to the rough hyper-parameters description". | VERIFIED |
| PyMARL2 and Hu et al. | `hijkzzz/pymarl2`, Apache-2.0, last push 2024-05-18, tuned for SMAC, value-based algorithms (QMIX family). README installs Python 3.8. Hu, Jiang, Harding, Wu, Liao, "Rethinking the Implementation Tricks and Monotonicity Constraint in Cooperative Multi-Agent Reinforcement Learning", arXiv:2102.03479 (arXiv comment: accepted by ICLR BlogTrack 2023). Not a PPO source. | VERIFIED |
| EPyMARL | `uoe-agents/epymarl`, Apache-2.0, last push 2024-09-24. Adds IPPO, MAPPO, IA2C and others; moved to Gymnasium; supports PettingZoo environments. It is run as `python src/main.py` from a cloned repository. Not published on PyPI (404). | VERIFIED (README, PyPI) |
| Papoudakis et al. 2020, arXiv:2006.07869 | "Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks" (Papoudakis, Christianos, Schäfer, Albrecht), NeurIPS 2021 Datasets and Benchmarks per the arXiv comment. The CyMARL paper cites it for CTDE methods generally outperforming independent learning. Only the arXiv record was read. | VERIFIED (record), UNVERIFIED (content) |

### 1.5 Simulation environments named in the brief

| Environment | Detail | Tag |
|-------------|--------|-----|
| Yawning Titan | `dstl/YAWNING-TITAN`: abstract, graph-based cyber simulation for training defensive agents against probabilistic red agents; MIT; README badge Python 3.8 to 3.10; built with OpenAI Gym, NetworkX, Stable Baselines 3 and RLlib. The repository is **archived** (last push 2024-05-21). Paper: Andrew, Spillard, Collyer, Dhir, "Developing Optimal Causal Cyber-Defence Agents via Cyber Security Simulation", arXiv:2207.12355 (ML4Cyber workshop at ICML). | VERIFIED |
| CyberBattleSim | `microsoft/CyberBattleSim`: abstract enterprise-network simulation focused on attacker lateral movement; the included defender is a basic stochastic scripted agent that detects and re-images nodes (example: keep network availability above an SLA of 0.80); Gym/Gymnasium interface; MIT; last push 2026-10-30. The README recommends Linux or WSL, calls direct Windows setup unsupported, and its setup script uses a uv-managed Python 3.10. It does not mention macOS. Not a defender-learning benchmark. | VERIFIED |
| Multi-agent CyberBattleSim | Kunz, Fisher, La Novara-Gsell, Nguyen, Li, arXiv:2304.11052: adds trainable blue agents to CyberBattleSim and reports training them alone or jointly with red. Abstract read; code not examined. | VERIFIED (abstract) |
| NASim | `Jjschwartz/NetworkAttackSimulator`, MIT, last push 2024-02-28; the base of the CMARL-ACD fork. README: implements the Gymnasium interface. Not examined beyond README and metadata. | VERIFIED (metadata) |

### 1.6 Prior work that overlaps with CyberSwarm's proposed differences

These came from a limited search (arXiv API, GitHub search, Crossref). They are not a systematic literature review. Only abstracts were read.

| Work | What it shows | Tag |
|------|---------------|-----|
| Silva, "Safety-Contract Graph Multi-Agent Reinforcement Learning for Autonomous Network Security Response", arXiv:2606.13832 (2026-06-11), single author, preprint | Evaluated on CAGE Challenge 4 with budgets for mean time to recover, false-positive response and firewall-change disruption. Compares IPPO, MAPPO-GAT, C-MAPPO-GAT and ACD3-GAT (CVaR tail-risk estimation, opponent belief, graph counterfactual risk screening). Reports that unconstrained methods violated a downtime budget in 100% of evaluated episodes, with three 200-episode seeds. This already combines MAPPO, a graph encoder, risk estimation and operational-disruption constraints. Peer-review status unknown. | REPORTED (abstract) |
| Wilson et al., "Multi-Agent Reinforcement Learning for Maritime Operational Technology Cyber Security", arXiv:2401.10149 (CAMLIS 2023) | In their IPMSRL environment a shared-critic MAPPO outperformed IPPO; MARL defenders stayed effective when alert success probability fell to 0.75 or 0.9. | REPORTED (abstract) |
| CMARL-ACD (repository and thesis) | Already pairs IPPO and MAPPO with IDS-alert uncertainty (noisy alerts as observation uncertainty), in a competitive setting. | REPORTED |
| CC4 and CMARL-ACD | Both already penalize wrongful or disruptive containment (green-work failures; wrong IP blocks). | VERIFIED |

Search coverage: arXiv queries combining "uncertainty-aware", "cyber defense" and "reinforcement" returned 0 results. That says my queries found nothing, not that nothing exists.

Consequences: "uncertainty-aware" and "proportionate response" are not new ideas, and graph encoders in cyber-defence MARL already exist (consistent with `docs/DECISIONS.md` AD-01, which keeps graph learning a secondary, staged hypothesis rather than the central claim). Whatever CyberSwarm adds must be stated as a specific mechanism with a matched control (Section 3).

---

## 2. Verified implementation details

This section answers: what do the official or closest-to-official implementations actually provide, what stops us reproducing them, and what technical foundations fit CyberSwarm?

### 2.1 CyMARL code: `Poly-AIvsAI/CyMARL`

| Item | Detail | Tag |
|------|--------|-----|
| Existence | Public repository, created 2023-05-11, last push 2023-06-07, no description, default branch `master`, one star. | VERIFIED |
| Link to the paper | Its README: "CyMARL is a PyMARL environment that extends CAGE Challenge 2 ... adapted from PyMARL2 and CybORG v3.1". All 43 commits are by one account, `jgawiebe`; a commit message references a merge from `RMC-AIvsAI/CyMARL`, and that address now returns HTTP 301 to `Poly-AIvsAI/CyMARL` (the repository was renamed or moved). The paper itself does not link to it. The match to the paper is strong (same environment name, same topic, an organization name that matches the authors' institution, an account name resembling the first author's) but is inferred, not confirmed by the authors. | VERIFIED (contents, redirect), INFERENCE (identity) |
| Contents | `CybORG/` (modified), `pymarl2/` with `envs/cyborg/cyborg_env.py` and `config/envs/cyborg.yaml`, scenario files such as `confidentiality_small.yaml`, `integrity_small.yaml`, `availability_medium.yaml`, custom agents (`BlueReactAgent`, `RedConfidentialityAgent`, and others), `runner/` scripts, `results/`. | VERIFIED |
| Scenario sizes | `confidentiality_small.yaml`: agents Blue0, Blue1, Red; 5 hosts (Defender, Op_Server0, User0 to User2). `confidentiality_medium.yaml`: 10 hosts (Defender, Op_Server0, Op_Host0 to Op_Host2, User0 to User4), i.e. 5 user and 4 operational hosts besides the Defender host. So the paper's 9-host topology most likely corresponds to the "medium" files. | VERIFIED (files), INFERENCE (mapping) |
| Environment wrapper | `cyborg_env.py`: `time_limit=100` in the constructor and `time_limit: 30` in `config/envs/cyborg.yaml`; `wrapper_type: raw` (flattened state) by default, with a `table` wrapper that gives the vector of flags; `action_masking: False` by default; team reward = mean of the agents' rewards; global state = concatenation of the agents' observations; `seed()` returns the attribute rather than seeding. | VERIFIED |
| Run command | `python pymarl2\main.py --config=qmix --env-config=cyborg with env_args.map_name=confidentiality_small`. Default `t_max` 2,050,000 in the config (README says 1M by default); results go to `results\sacred\<map_name>\`. | VERIFIED (README, config) |
| Shipped experiment scripts | `runner/batch/100k_init.bat` (IQL and QMIX, `t_max=100500`) and `runner/batch/table_baseline.bat` (`t_max=1005000`, `wrapper_type=table`, medium maps). Neither is the paper's 2M-step protocol with five evaluation runs. | VERIFIED |
| Environment and dependencies | README: "Tested using Windows 10"; conda Python 3.9; `torch==1.8.2` from an extra index of PyTorch LTS wheels. `setup.py` pins `gym==0.26.0`, `protobuf==3.19.5`, `pyyaml==5.3.1`, `sacred==0.8.2`, `scikit-learn==0.24.2`. | VERIFIED |
| Licence | No LICENSE file in the repository; GitHub reports none. Without a licence, code reuse is not permitted by default. | VERIFIED |
| Tests, seeds | No tests directory seen. Seed handling: see the `seed()` note above. | VERIFIED (listing) |
| Related | `Poly-AIvsAI/CyMARL-CommFormer` (Apache-2.0, last push 2026-02-06) adapts CyMARL and CommFormer; its README credits a thesis by Faizan Contractor and Dr. Ranwa Al Mallah. README read only. | VERIFIED (README) |

### 2.2 CMARL-ACD code

See Section 1.2.1. Reproducibility limits, collected:

1. The journal paper is not located. The thesis is a primary text, but it disagrees with itself in places (Section 1.2.2) and with the repository's best_models summary.
2. Python must be 3.9 to 3.12 and numpy 2.0.2 and torch 2.7.0 are pinned. This machine's default is Python 3.13.2 (Python 3.12.12 is also installed).
3. TorchRL is pinned to 0.8.0; the latest TorchRL (0.14.1) renamed the spec classes the script imports.
4. No tests are included.
5. Scenario seeds come from `random.randint` at environment creation; the training loop's determinism is untested. The thesis does not mention seeds or variance.
6. The default training budget in the script is 30 million frames; the thesis reports convergence at 0.4 to 3 million timesteps. Neither the script nor the thesis says how many independent training runs back each reported number.
7. The `real-testbed` half needs a root Linux host with KVM, Snort, Metasploit and iptables. It cannot run on macOS and must not be used by CyberSwarm.
8. Checkpoints are selected with the same cross-evaluation that reports the scores.
9. Nothing was executed by us. We have not reproduced any CMARL-ACD number.

### 2.3 CAGE Challenge 4 implementation

Covered in Section 1.3. Reproducibility limit: pinned `numpy==1.26.4` and `torch==2.2.0` have no cp313 wheels, so Python 3.13 cannot install `Requirements.txt`. The file does resolve on Python 3.12.12 in a dry-run (Section 2.5). The requirements also pull network-capable packages (`paramiko`, `sshtunnel`, `ray`), which CyberSwarm's runtime has no use for.

### 2.4 Implementation facts that decide CyberSwarm's design

| Fact | Source | Tag |
|------|--------|-----|
| Gymnasium 1.4.0: `Env.reset(*, seed=None, options=None) -> (obs, info)`; `Env.step(action) -> (obs, reward, terminated, truncated, info)`. | Read from the installed package in a temporary environment (0.1); the documentation page https://gymnasium.farama.org/api/env/ returned HTTP 200 in 0.2. | VERIFIED |
| PettingZoo 1.27.0 `ParallelEnv`: `reset(seed, options) -> (obs dict, infos dict)`; `step(actions) -> (obs, rewards, terminations, truncations, infos)`; attributes `agents`, `possible_agents`; per-agent `observation_space` and `action_space`; an optional `state()` documented as "a global view of the environment appropriate for centralized training decentralized execution methods like QMIX". | Re-read from `pettingzoo/utils/env.py` on GitHub (0.2). | VERIFIED |
| pydantic v2 can export JSON Schema with `additionalProperties: false`, discriminated unions, and `null`-able optional fields; it rejects extra fields, unknown action kinds, out-of-range values and non-conforming identifiers. Its schemas follow JSON Schema Draft 2020-12 per its documentation. `model_json_schema()` does not emit a `$schema` key, so the export step must add one. | Ran a small check on pydantic 2.12.5 (already on the machine; nothing installed) in 0.1. | VERIFIED |
| Ajv 8.20.0 ships `dist/2020.js` (Draft 2020-12 validator) for JavaScript consumers of the exported schemas. | Listed the npm tarball (0.1). | VERIFIED |
| Stable-Baselines3 PPO is single-agent. The PettingZoo SB3 tutorial states: "SB3 is designed for single-agent RL and does not directly support multi-agent algorithms or environments. These tutorials are only intended for demonstration purposes", and that they use "PPO with parameter sharing, allowing a single model to control all the agents". `sb3-contrib` has a Maskable PPO (single-agent). SuperSuit's README says it "will be deprecated and rewritten" inside PettingZoo. | PettingZoo tutorial page (HTTP 200), SB3 algorithm table, SuperSuit README, all read in 0.2. | VERIFIED |
| TorchRL 0.14.1 contains `torchrl/objectives/multiagent/mappo.py` with classes `MAPPO` and `IPPO`, exports `IPPOLoss` and `MAPPOLoss`, and has `MultiAgentGAE`, `MultiAgentMLP`, `MaskedCategorical` and an example `sota-implementations/multiagent/mappo_ippo.py`. Its multi-agent PPO tutorial says "In MAPPO the critic is centralised and takes as input the global state of the system. This can be a global observation or simply the concatenation of the agents' observation." Version 0.1 of this document said only that `ppo.py` and `multiagent.py` exist; that was incomplete. | Opened the 3.7 MB cp313 macOS wheel's file list and source (deleted afterwards); read the tutorial page. | VERIFIED |
| TorchRL 0.14.1 requires `tensordict>=0.14.3,<0.15.0` (latest tensordict is 0.15.3) and new dependencies `pyvers` and `hoptorch`. BenchMARL 1.5.2 (source distribution only, 2026-02-07) requires `torchrl>=0.10,<0.12`, `hydra-core`, `torchvision` and `av<14`. | PyPI metadata. | VERIFIED |
| BenchMARL's README table lists MAPPO and IPPO (on-policy, discrete and continuous actions), with a "full observability in critic" column that is Yes for MAPPO and No for IPPO. Version 0.1 had not checked IPPO. | README. | VERIFIED |
| A dry-run resolution on Python 3.13.2 for `torch`, `gymnasium`, `pettingzoo`, `pydantic`, `numpy`, `networkx`, `pytest`, `jsonschema` (wheels only) succeeded: torch 2.15.0 (cp313 macOS arm64 wheel, 138.6 MB), gymnasium 1.4.0, pettingzoo 1.27.0, pydantic 2.14.2, pydantic_core 2.50.2, networkx 3.7, numpy 2.5.4, pytest 9.2.0, jsonschema 4.26.0. The same set resolves on Python 3.12.12. Further dry-runs: torch_geometric 2.8.1 (3.13.2); torchrl alone (torchrl 0.14.1, tensordict 0.14.3); benchmarl (torchrl 0.11.1, tensordict 0.11.0, torch 2.15.0, torchvision 0.29.2, av 13.1.0). | `pip install --dry-run` in temporary environments. This proves the resolver finds a set. It does not prove runtime behaviour. | VERIFIED |

### 2.5 Reproducibility assessment

**Method and limit.** No third-party training or evaluation was run, and nothing was cloned. "Could be reproduced" below means: the artifact is obtainable, the licence allows use, and its pinned dependencies resolve for this machine (Apple silicon, macOS 15.7.4, Python 3.13.2 default, 3.12.12 available). It does **not** mean a number was reproduced. Resolution was tested with `pip install --dry-run` (wheels only unless noted) in throwaway environments.

| Source | What could be reproduced | What could not | Why |
|--------|--------------------------|----------------|-----|
| arXiv:2310.05939 and `Poly-AIvsAI/CyMARL` | The method and environment description can be reproduced on paper. The paper and the repository are obtainable, and the medium scenario files match the 9-host topology. | Running the code on this machine. Matching the paper's numbers. Reusing the code. | A dry-run of the `setup.py` requirements on Python 3.12.12 (source distributions allowed) failed: `scikit-learn==0.24.2` has no wheel for that Python and its source build failed in metadata preparation (`ModuleNotFoundError: pkg_resources`). `torch==1.8.2` is not on PyPI; it is on PyTorch's LTS index. The README's `.../lts/1.8/cu111` folder lists only Linux and Windows wheels, and the LTS index has no macOS arm64 wheel for Python 3.9 (for 1.8.2 it lists a macOS x86_64 wheel for cp39 and a macOS arm64 wheel for cp38 only; PyPI's torch 1.8.1 has macOS x86_64 wheels only). So this Apple silicon machine would need an x86_64 (Rosetta) Python 3.9 or a different torch (INFERENCE). The paper gives no episode length, versions or seeds for the main runs; the shipped batch scripts are not the 2M-step protocol; the heuristic baseline reads true state; no licence. Table 1 standard deviations for the heuristic are large. |
| CMARL-ACD repository | Obtainable (MIT text). Its full pinned `requirements.txt` resolves on Python 3.12.12 (wheels only, 58 lines). Checkpoints are shipped. | Resolving the pins on Python 3.13.2. Verifying the paper's claims. Running the real testbed. Matching reported scores. | `numpy==2.0.2` has no cp313 wheel, so the 3.13.2 resolve stops there. The journal paper is not located. No tests. No seeds or run counts in the script or thesis. The thesis abstract and body disagree on two headline numbers, and `best_models/README.md` and the thesis disagree on which algorithm is more robust under noise (or the sign convention is undefined). Checkpoint selection and scoring use the same cross-evaluation. `real-testbed/` needs root Linux, KVM, Snort, Metasploit and iptables and is out of scope and unsafe to run here. TorchRL 0.8.0 pin vs current API. |
| CAGE Challenge 4 / CybORG | Obtainable; MIT licence text in CC4; documented scenario, reward tables and evaluation protocol (100 episodes of 500 steps against `FiniteStateRedAgent`). `Requirements.txt` resolves on Python 3.12.12 when source distributions are allowed (torch 2.2.0, numpy 1.26.4, gymnasium 0.28.1, pettingzoo 1.24.3, ray 2.38.0, torch_geometric 2.5.0, stable_baselines3 2.3.2, sb3-contrib 2.2.1). | Python 3.13.2. Reading the AI Magazine article. Checking the AAAI paper's results. | The resolve with wheels only stopped at `gym==0.26.2` (3.12) and `PyYAML==6.0.1` (3.13); numpy 1.26.4 and torch 2.2.0 have no cp313 wheels. The Wiley page is behind HTTP 403; the AAAI paper text was not read. |
| Yu et al. and `marlbenchmark/on-policy` | The algorithm definitions and the five practical suggestions (read in the paper text). Obtainable MIT code. | Running the official code here. Any cyber-environment result. | README installs Python 3.6.1, torch 1.5.1, CUDA 10.1, and supports only SMAC, Hanabi, MPE and Google Research Football. The README warns that many papers fail to reproduce MAPPO because of under-specified hyperparameters. The paper's setting has shared rewards. |
| PyMARL2 and Hu et al. | Obtainable Apache-2.0 code and a documented install (conda, Python 3.8). | Anything run. | Tuned for SMAC and value-based methods; old Python; no cyber content. Dependencies were not resolved. |
| Yawning Titan | Obtainable MIT code. | Anything run. | Archived; README badge Python 3.8 to 3.10; the old Gym API. Dependencies were not resolved. |
| CyberBattleSim | Obtainable MIT code, actively maintained. | Anything run. | README recommends Linux or WSL with a uv-managed Python 3.10 and does not mention macOS. Dependencies were not resolved. |
| arXiv:2606.13832 and arXiv:2401.10149 | Their abstracts. | Anything else. | Only abstracts were read; no code was looked for. |
| CyberSwarm's own work | Nothing exists yet. | n/a | n/a |

**Consequence.** For this project the only reproduction that will exist is our own, run on equivalent scenarios and seeds with spread, on our own simulator. Third-party numbers are context only.

### 2.6 Foundation comparison against CyberSwarm's requirements and the local Python

**Requirements taken from the brief and DELTA.md.**

| ID | Requirement |
|----|-------------|
| Q1 | A multi-agent environment API with per-agent observations, rewards, termination and truncation, and action masks. |
| Q2 | Versioned contracts as typed models with JSON Schema export, consumed also by a non-Python dashboard. |
| Q3 | Independent PPO actors (IPPO), seeded training and checkpoints; MAPPO only if it can be done correctly (decentralized actors, centralized critic, correct advantages). |
| Q4 | Access to policy logits and entropy, and room for ensembles or dropout, for the uncertainty mechanism, with a matched control that shares code. |
| Q5 | Small CPU workloads on macOS arm64 under Python 3.12 or 3.13 in a repository-local virtual environment, with about 650 MiB free disk space (measured; see `docs/DECISIONS.md`, section 1). |
| Q6 | A defense coordinator and runtime with no shell, subprocess, socket or network behaviour, so dependencies with network or SSH functions need a reason. |
| Q7 | Topology as stable node and edge records, plus the required lead-owned graph pipeline (deterministic observation-to-graph conversion, hand-engineered features, masks) with a model-compatible tensor interface that does not need a learned-GNN library. |
| Q8 | A dashboard that runs on fixtures first, supports replay, is testable and shows missing values honestly. |

**A. Reinforcement-learning foundations** (versions from PyPI on 2026-10-30; research-level assessment only; decisions are in `docs/DECISIONS.md`)

| Candidate | Version and Python | Multi-agent support | Fit with Q1 to Q6 | Footprint and compatibility here | Assessment |
|-----------|--------------------|---------------------|-------------------|----------------------------------|------------|
| PyTorch | 2.15.0 (2026-10-28), `>=3.11`; cp312 and cp313 macOS arm64 wheels (`macosx_14_0`; macOS here is 15.7.4) | n/a | Needed for any learned policy | Wheel 138.6 MB, about 565 MB installed. Peak install space about 0.7 to 1 GB. **Free disk is about 650 MiB, so the install may fail until space is freed.** | Required. CPU only. |
| Gymnasium | 1.4.0, `>=3.10`, pure-Python wheel (0.5 MB) | Spaces and the terminated/truncated convention only | Q1 (spaces, API conventions) | Small | Required |
| PettingZoo | 1.27.0, `<3.15,>=3.10`, pure-Python wheel (0.7 MB) | `ParallelEnv` for simultaneous agents; `state()` hook for centralized critics | Q1, Q3 | Small; needs numpy, gymnasium, typing-extensions | Required for the environment API |
| Stable-Baselines3 and sb3-contrib | 2.9.2, `>=3.10`; requires `torch>=2.8` | **Single-agent** (documented by PettingZoo's tutorial); parameter-sharing only through wrappers; Maskable PPO is single-agent | Fails Q3 for independent actors and MAPPO; parameter sharing conflicts with per-agent policies | Resolves with torch 2.15.0 on 3.13. SuperSuit 3.11.0 depends on `tinyscaler`, which has wheels for cp310 to cp312 only; on 3.13.2 the wheels-only resolve of SuperSuit 3.11.0 fails, and the combined resolve silently fell back to SuperSuit 0.1.10, which is a trap | Not a dependency. Fine for a single-agent sanity check at most |
| TorchRL | 0.14.1, `>=3.10`, requires `tensordict>=0.14.3,<0.15`, `pyvers`, `hoptorch` | `IPPOLoss`, `MAPPOLoss`, `MultiAgentGAE`, `MultiAgentMLP`, `MaskedCategorical`; the wheel also contains `PettingZooWrapper` and `PettingZooEnv` (`torchrl/envs/libs/pettingzoo.py`; not exercised) | Strongest library fit for Q3 and masks. Needs TensorDict specs around our environment (more adaptation code), and the uncertainty hooks must be added around its modules | cp313 macOS arm64 wheels; resolves on 3.13.2 (with tensordict 0.14.3). CMARL-ACD's use of TorchRL 0.8.0 broke against 0.14.1 names, which shows API churn | Best maintained alternative. Candidate for a parity cross-check or a fallback for the loss only |
| BenchMARL | 1.5.2, source distribution only; requires `torchrl>=0.10,<0.12`, hydra-core, `torchvision`, `av<14` | MAPPO and IPPO in its README table | Needs our environment as a TorchRL task and Hydra configs; heavy extras for a benchmark runner | Resolves on 3.13.2 (torchrl 0.11.1, tensordict 0.11.0, torch 2.15.0); README badge lists Python 3.8 to 3.11; runtime not tested | Not for the core |
| EPyMARL | not on PyPI; clone and run `src/main.py`; `requirements.txt` pins `PyYAML==5.3.1` (wheels to cp39 only) | IPPO, MAPPO, IA2C and others; PettingZoo and Gymnasium supported | A research framework, not a library; per-agent masks for our contracts not checked | Not installed | Reference only |
| `marlbenchmark/on-policy` | not on PyPI; Python 3.6.1, torch 1.5.1 instructions | The official MAPPO and IPPO | Environment wrappers for four benchmarks only | Not installable here as documented | Reference for the algorithm and its tricks only |
| MARLlib | 1.0.3 (2023-04-25), source distribution only | Ray-based | Not evaluated beyond metadata; stale | n/a | Not considered |
| Compact self-written IPPO (and MAPPO from the same code path) | torch and numpy only | We write it: IPPO and, only if verified, MAPPO | Exact fit with Q1 to Q4: direct access to logits and masks, proposals as output, shared code for the matched control | No extra dependency | Feasible. Correctness risk is ours, so it needs the hand-computed GAE, clipped-loss and masking tests listed in `docs/DECISIONS.md` section 3 |

**B. Graph and contract tooling**

| Candidate | Version and Python | Facts | Assessment for the confirmed scope |
|-----------|--------------------|-------|-----------------------------------|
| NetworkX | 3.7 (2026-09-21), `>=3.12, !=3.14.1`; pure wheel 2.1 MB | Standard graph algorithms (reachability, paths, hop distance, degree, neighbours). | Default topology and graph-processing foundation for the required graph pipeline and the simulator and mock (current `DELTA.md`). The contracts package must not import it, so the dashboard and the schemas stay independent of it (Q2, Q7) |
| PyTorch Geometric | 2.8.1 (2026-10-12), `>=3.10`; pure wheel 1.3 MB; docs say Python 3.10 to 3.14; resolves on 3.13.2 | Core dependencies include aiohttp, requests, fsspec, jinja2, psutil, pyparsing, tqdm and xxhash (network libraries; Q6). It does not declare torch. CC4's requirements pin `torch_geometric==2.5.0`; arXiv:2606.13832 uses a graph attention encoder. | Not needed for the required pipeline, whose tensor and batch interface must not depend on it. Staged: considered only after reliable IPPO, and only if the GNN go decision is taken (task, supervision, tensor contracts, budget and matched evaluation documented first). A plain-PyTorch dense-adjacency layer would suffice for a first ablation |
| pydantic | 2.14.2 (2026-10-29), `>=3.10`; pydantic_core 2.50.2 | JSON Schema export, `extra="forbid"`, discriminated unions (Section 2.4). The machine already has 2.12.5. | Required (Q2) |
| jsonschema | 4.26.0, `>=3.10` | Validates fixtures against the exported schema files (development only). | Development dependency |
| numpy | 2.5.4, `>=3.12` | Needed by gymnasium, torch interop and the statistics. | Required |
| pytest | 9.2.0 | Test runner named in the brief. | Development dependency |

**C. Dashboard options** (npm registry and PyPI on 2026-10-30; the framework comparison and recommendation stay in `docs/DECISIONS.md`, section 4)

| Candidate | Version | Facts | Assessment |
|-----------|---------|-------|------------|
| React, Vite, TypeScript | 19.3.1; 8.3.5 (Node `^20.19.0 \|\| >=22.12.0`); 7.0.2 | Local Node v26.8.1 satisfies Vite and Vitest. | Recommended stack (depends on Mayukh's JS and TypeScript skills) |
| Cytoscape.js | 3.34.4 | Built for network graphs. The React wrapper `react-cytoscapejs` 2.0.0 was last published 2022-09-02, so use Cytoscape directly from a small React component. | Graph view |
| Vitest, Testing Library, Ajv | 5.0.5 (Node `^22.12.0 \|\| ^24.0.0 \|\| >=26.0.0`); `@testing-library/react` 16.3.3; ajv 8.20.0 | Headless tests and Draft 2020-12 validation of the exported schemas. | Tests and validation |
| Other graph libraries | `@xyflow/react` 12.12.0; sigma 4.0.4; vis-network 10.1.4 | Node-editor library; WebGL graph renderer; general network library. Metadata only. | Not preferred; no concrete advantage found for this view |
| Python UI frameworks | Streamlit 1.66.1; Dash 4.5.0 with `dash-cytoscape` 1.0.2 (source distribution only, last released 2024-07-15); NiceGUI 3.18.0 | Pure-Python options. Replay and scrubbing fit the Streamlit rerun model poorly; Dash needs server round trips. NiceGUI was checked by metadata only. | Fallback if the dashboard owner prefers Python |

---

## 3. Our proposed extensions

Each item is a hypothesis. None has been tested. Each could turn out false, and several overlap with prior work (Section 1.6). Under the confirmed direction (`docs/DECISIONS.md` AD-01) multi-agent RL is the primary line: the core items are E1 to E5 and E7. E8 is the controlled comparison that the required graph infrastructure makes possible. E6 (a learned graph model) is a staged, secondary hypothesis.

| ID | Proposed difference | How it would be tested | Overlap with prior work |
|----|---------------------|------------------------|-------------------------|
| E1 | A small, fully seeded, abstract simulator behind a versioned contract, so that RL code, coordinator, evaluator and dashboard share the same records. | Contract tests; simulator conformance suite; identical-seed determinism test. | Engineering, not a research claim. |
| E2 | Detection-to-containment evaluation that reports false-isolation count and healthy-service disruption next to containment time, over repeated seeds with spread, on held-out scenarios disjoint from training. | Evaluation harness; train/eval disjointness test. | CC4 and CMARL-ACD already penalize disruptive containment, and CMARL-ACD's TBR measures blocking precision. We do not claim the idea is new, only that we measure it explicitly. |
| E3 | An explicit defense coordinator with an action allowlist between policy and simulator. Rejected proposals never change state. | Allowlist and unchanged-state tests. | Safety-contract work exists (arXiv:2606.13832). |
| E4 | Uncertainty-aware decisions: when estimated uncertainty is high, prefer an evidence-gathering action; defer high-impact containment unless justified. **H-E4:** at equal training budget, the uncertainty-aware policy has fewer false isolations and no worse detection-to-containment time than a matched control (same architecture, data and budget, uncertainty unused) on held-out scenarios. | Matched-control ablation; calibration check (reliability/ECE) on held-out episodes. | CMARL-ACD models alert noise in the observation but, as far as the repository and thesis show, has no agent-side uncertainty estimate; arXiv:2606.13832 uses tail-risk estimation. Candidate uncertainty estimators (policy entropy, ensemble disagreement, MC-dropout, evidence-based) are unselected and unvalidated. |
| E5 | IPPO vs MAPPO under a contract-valid, coordinator-mediated pipeline. No expectation of either winning. | Same scenarios and seeds for both; MAPPO only if verified correct. | Mixed priors: IQL close to QMIX (CyMARL), MAPPO ahead of IPPO (arXiv:2401.10149), and CMARL-ACD's two self-reports that disagree (Section 1.2.1). |
| E6 | **Staged, secondary.** A learned graph model (GraphSAGE, GAT or another) as an observation encoder or a node-scoring head. Considered only after the conventional and graph-feature paths and IPPO are stable, and only with a documented concrete task, a supervision source that keeps evaluator-only labels out of decision-time features, tensor contracts, a resource budget and a matched evaluation. A documented no-go satisfies the stage. Scores, rankings, attention weights and embeddings would be predictive outputs, not causal explanations. | Ablation against the non-graph and the hand-engineered-feature observations on the same scenarios and seeds. | arXiv:2606.13832 uses a graph attention encoder with MAPPO; CC4's requirements include `torch_geometric`. |
| E7 | Reward-weight sensitivity analysis instead of tuning weights until a method wins. | Sweep over documented weights, reported with spread. | None claimed. |
| E8 | A controlled comparison of conventional observations against non-neural, hand-engineered graph features (for example degree and criticality context, neighbour-alert aggregation) with the same policy architecture, scenarios, seeds and training budget. **H-E8:** the graph-feature view changes containment time or false isolations relative to the conventional view. The direction of any change is not assumed. | Matched scenario and seed runs of both views, repeated seeds with spread; leakage tests showing the features use agent-visible data only. | Graph state encodings already appear in cyber-defence MARL (arXiv:2606.13832; CC4's graph-related requirements). Not claimed as new. |

Not claimed: that any of these improves containment, that CyberSwarm is the first system with any of these properties, or that learned policies beat the rule-based baseline.

---

## 4. Unresolved assumptions and open questions

| # | Item | What would resolve it |
|---|------|-----------------------|
| 1 | I assumed the brief's "CMARL-ACD" means `cyb3rlab/CMARL-ACD` and the JAIST thesis (Section 0). The name matches exactly, but the user has not confirmed it. `docs/ARCHITECTURE.md` instead names CyMARL as the reference and never mentions CMARL-ACD. | User confirms which was meant, or names another source. |
| 2 | The CMARL-ACD journal paper was not found. The thesis is the nearest primary text. Whether the paper's numbers equal the thesis's (which disagree with themselves in places) is unknown. | Locate the paper or ask the authors. |
| 3 | `Poly-AIvsAI/CyMARL` is very likely the paper's code (the old `RMC-AIvsAI/CyMARL` address redirects to it) but is not linked from the paper. | Authors' confirmation or a link in a later version. |
| 4 | CyMARL episode length in the paper's runs is not stated; the code defaults to 30 turns in the config and 100 in the constructor. Which one the paper used is unknown. | Ask the authors, or read their run configuration. |
| 5 | The CC4 AI Magazine article (Wiley) and the AAAI paper text were not read. | A fetch from a network that is not blocked, or the author-posted copy. |
| 6 | No prior result was reproduced. All numbers from other groups are theirs. | Out of scope for this project unless the user wants a replication. |
| 7 | MAPPO centralized-critic input. Yu et al. recommend including global features. TorchRL's tutorial says the global state "can be a global observation or simply the concatenation of the agents' observation". CMARL-ACD's critic also sees the opponent's information. The brief says ground truth is used only by the evaluator and simulator-side reward, never in observations. Whether the critic may see simulator-held global state (not ground-truth labels) is undecided. The safe default is a concatenation of agent-visible observations. Yu et al. also assume a shared team reward, which our environment must decide. | Design step plus user decision. |
| 8 | Number and roles of agents (per zone, or per function such as detect, investigate, contain). This sets whether parameter sharing is valid and how much MAPPO can differ from IPPO. | Design step. |
| 9 | What "uncertainty" means: uncertainty about the observation (noisy alerts), about the policy's decision, or both. Which estimator to use is open. | Design step (to be written in `docs/RL_IMPLEMENTATION.md`). |
| 10 | Whether a self-written IPPO/MAPPO is correct. It will need unit tests and a toy-task learning check. TorchRL's `IPPOLoss` and `MAPPOLoss` could serve as a parity cross-check, but no runtime test of TorchRL was done here. | Implementation step with executed tests. |
| 11 | Mayukh's JavaScript/TypeScript experience is unknown. The dashboard recommendation in `docs/DECISIONS.md` assumes he can learn basic React. | User confirms. |
| 12 | Package versions in `docs/DECISIONS.md` are as of 2026-10-30 and will move. A resolver solution is not a runtime test. | Pin ranges in `pyproject.toml`; re-check at install time. |
| 13 | Disk space. Free space on the data volume was 272 MiB at first measurement, about 911 MiB later and about 650 MiB at the end of this delta. The torch wheel needs 565 MB installed (peak 0.7 to 1 GB during install), so creating the in-repo virtual environment is at risk. | User frees space (see `docs/DECISIONS.md`, section 1). |
| 14 | Project licence is undecided. CyMARL has no licence, so its code cannot be copied. CMARL-ACD, CC4, CybORG and NASim are MIT, but we plan no code reuse. | User decision. |
| 15 | Search coverage for overlapping work is limited to arXiv, GitHub, Crossref, OpenAlex and one web search engine; DBLP and Semantic Scholar could not be queried. | A proper literature review before any novelty-style claim. |
| 16 | The thesis contradicts itself on two headline numbers, and the repository's best_models summary and the thesis give opposite impressions of which algorithm is more robust under noise. Which is right is unknown. | Authors' errata or the journal paper. |
| 17 | Whether hand-engineered graph features help the policies (E8) and whether a learned GNN is justified (E6) are untested. The GNN go/no-go depends on a task, a supervision source, a compute budget and evidence that the hand-engineered baseline leaves room for improvement; none exists yet. A risk to test, not a finding: on a small topology the hand-engineered features may already capture most of what message passing would add. | The matched experiments above, after IPPO is stable. |
| 18 | `DELTA.md` changed while this research delta ran. The version read at the start (12,728 bytes) called the graph pipeline optional; the current one (modified 2026-10-30 19:49, 18,005 bytes) supersedes that and makes it required. The step instructions for this delta used the earlier wording. This document and `docs/DECISIONS.md` follow the current file. | User or orchestrator confirms the current wording is intended. |

---

## Appendix A. Repository documents reviewed

| File | State | What it is |
|------|-------|------------|
| `README.md` | Tracked, 156 lines, last modified 2026-10-25 | "CyberSwarm-GX: Graph Neural Network-Powered Autonomous Cyber Defense". A GNN root-cause-localization project ("Faultline Mode"). It lists candidate root-cause ranking metrics, mentions reinforcement learning or contextual bandits only as one possible "adaptive defense policy", and states "No experimental results ... have been established yet." It uses a check mark and an hourglass symbol in its status list. No team names. **Its GNN-centred framing is superseded by the confirmed direction (`docs/DECISIONS.md` AD-01); graph representation stays as lead-owned infrastructure. Not modified in this research delta.** |
| `docs/ARCHITECTURE.md` | Untracked, 546 lines, 22,443 bytes, modified 2026-10-29 19:46 (four days after the initial commit) | "CyberSwarm Architecture Document", version 0.1, status "Concept -> Detailed Design Phase". No author or signature. From the repository alone I cannot tell whether the user or an assistant session wrote it; it is a draft, not a reviewed decision record. **Not modified in this research delta.** |

**What `docs/ARCHITECTURE.md` contains.** It already takes the multi-agent reinforcement-learning direction of the new brief: executive summary; research foundation naming CyMARL; an ASCII layered diagram; component table with owners (Raksha for the simulator, Mayukh for the dashboard, project lead for the rest); descriptions of the simulator, RL pipeline, safety-constrained coordinator, evaluator, dashboard and logging; episode lifecycle; sketch observation, action, event and episode-outcome structures; module ownership; design principles; an action allowlist; and technology choices. It explicitly says GNNs are optional, distinct from MARL, and to be added only with a measurable hypothesis. That is consistent with the confirmed pivot and with the brief.

**Points where it conflicts with the brief or with verified facts** (for the design step to reconcile; nothing has been changed):

1. It names CyMARL (IQL/QMIX) as the reference and never mentions CMARL-ACD (IPPO/MAPPO).
2. It says the CyMARL repository is "not publicly available". `Poly-AIvsAI/CyMARL` exists.
3. It says CybORG is "not open-sourced in code". `cage-challenge/CybORG` is public and MIT-licensed per its README.
4. It says CyMARL was "Published: October 2023". The arXiv v1 is dated 25 Aug 2023 and the work was presented at a workshop.
5. It has the simulator implement `gymnasium.Env` and return `obs, event, reward, done, info` from `step`. The brief requires a separate Simulator interface that the RL environment adapts, with ground truth behind an evaluator-only accessor.
6. Its event record has a `ground_truth` field inside the same record type, "only populated by evaluator". The brief wants hidden labels kept out of agent-visible records entirely.
7. Its observation sketch includes `process_list`, and its allowlist uses `investigate_host`, `isolate_host`, `restore_host` and `monitor_network`. The brief asks for one canonical abstract action vocabulary (monitor/investigate, request more evidence, rate-limit, quarantine, isolate, restore, do nothing) with no host-level artifacts.
8. It lists Ray/RLlib in the technology section; nothing in the brief needs it.
9. It lists "root-cause rank accuracy" among per-episode metrics and a "root-cause investigation" enhancement, carry-overs from the GNN framing; neither appears among the metrics the confirmed scope lists (a ranking or scoring task would belong to the staged GNN hypothesis, E6).
10. Its timestamp field is named `timestamp` and holds a step number; the brief requires explicit step semantics.

## Appendix B. Sources

Status is the HTTP status of a fetch made on 2026-10-30 (redirects followed). "Read" means I read the content; "listed" means only the status or metadata was checked.

**Papers and records**

| URL | What it is | Status |
|-----|------------|--------|
| https://arxiv.org/abs/2310.05939 | CyMARL paper, abstract page (title, authors, comment, journal reference) | 200, read |
| https://arxiv.org/pdf/2310.05939 | CyMARL paper, full text (10 pages) | 200, read |
| https://export.arxiv.org/api/query?id_list=2310.05939 | arXiv API record for the same paper | 200, read |
| https://arxiv.org/abs/2308.09520 | Proceedings of the 2nd International Workshop on Adaptive Cyber Defense | 200, listed (record read through the API) |
| https://arxiv.org/abs/2103.01955 and https://arxiv.org/pdf/2103.01955 | Yu et al., MAPPO and IPPO | 200, PDF text read |
| https://arxiv.org/abs/2102.03479 | Hu et al., PyMARL2 tricks | 200, abstract read through the API |
| https://arxiv.org/abs/2207.12355 | Andrew et al., Yawning Titan paper | 200, abstract read through the API |
| https://arxiv.org/abs/2108.09118 | CybORG paper | 200, abstract read through the API |
| https://arxiv.org/abs/2304.11052 | Kunz et al., multi-agent CyberBattleSim | 200, abstract read through the API |
| https://arxiv.org/abs/2006.07869 | Papoudakis et al., MARL benchmark | 200, record read through the API |
| https://arxiv.org/abs/2606.13832 | Silva, safety-contract graph MARL | 200, abstract read through the API |
| https://arxiv.org/abs/2401.10149 | Wilson et al., maritime OT MARL | 200, abstract read through the API |
| https://dspace.jaist.ac.jp/dspace/handle/10119/20030 | JAIST thesis item page (Nguyen, 2025) | 200, read |
| https://dspace.jaist.ac.jp/dspace/bitstream/10119/20030/5/paper.pdf | JAIST thesis full text (77 pages) | 200, read |
| https://hdl.handle.net/10119/20030 | Handle link for the thesis | **503** (the newer JAIST repository is under maintenance); not followed |
| https://www.jaist.ac.jp/is/labs/beuran-lab/publications.html | JAIST Cybersecurity Research Lab publications page | 200, read (no CMARL-ACD entry) |
| https://ojs.aaai.org/index.php/AAAI/article/view/35158 | Kiely et al., AAAI 2025 landing page | 200, metadata read |
| https://api.crossref.org/works/10.1609/aaai.v39i28.35158 | Crossref record for the AAAI paper | 200, read |
| https://api.crossref.org/works/10.1002/aaai.70021 | Crossref record for the AI Magazine CC4 article | 200, read |
| https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.70021 | AI Magazine CC4 article | **403**, not read, UNVERIFIED content |
| https://www.sciencedirect.com/science/article/pii/S2090447926003540 | "C-MARL" red-teaming article (name collision) | **403**, not read; metadata from Crossref |

**Code and repositories**

| URL | Files read | Status |
|-----|------------|--------|
| https://github.com/cyb3rlab/CMARL-ACD | `README.md`, `LICENSE.md`, `simulation/README.md`, `best_models/README.md`, `simulation/requirements.txt`, `simulation/setup.py`, `simulation/scripts/train_marl.py`, `simulation/nasim/envs/action.py`, `simulation/nasim/envs/environment.py`, file tree, commits (all re-fetched in 0.2; `evaluate_vs_baselines.py` was read in 0.1 and re-fetched in 0.2 for status only) | 200 |
| https://github.com/cyb3rlab | Organization profile | 200 |
| https://github.com/Poly-AIvsAI/CyMARL | `README.md`, `setup.py`, `pymarl2/envs/cyborg/cyborg_env.py`, `pymarl2/config/envs/cyborg.yaml`, batch scripts, two scenario files, file tree, commits | 200 |
| https://github.com/RMC-AIvsAI/CyMARL | Old address; redirects to `Poly-AIvsAI/CyMARL` | 200 after redirect (API: 301) |
| https://github.com/Poly-AIvsAI/CyMARL-CommFormer | `README.md` (0.1) | 200 |
| https://github.com/cage-challenge/cage-challenge-4 | `README.md`, `Requirements.txt`, `setup.py`, `LICENSE.txt` | 200 |
| https://github.com/cage-challenge/CybORG | `README.md` | 200 |
| https://github.com/marlbenchmark/on-policy | `README.md` | 200 |
| https://github.com/hijkzzz/pymarl2 | `README.md` | 200 |
| https://github.com/dstl/YAWNING-TITAN | `README.md` | 200 |
| https://github.com/microsoft/CyberBattleSim | `README.md` | 200 |
| https://github.com/Jjschwartz/NetworkAttackSimulator | `README.rst`, metadata | 200 |
| https://github.com/uoe-agents/epymarl | `README.md`, `requirements.txt` | 200 |
| https://github.com/facebookresearch/BenchMARL | `README.md` | 200 |
| https://github.com/pytorch/rl | `README.md`; the 0.14.1 wheel from PyPI (file list and source) | 200 |
| https://github.com/cytoscape/cytoscape.js and https://github.com/plotly/dash-cytoscape | Listed only | 200 |

**Documentation and registries**

| URL | What was read | Status |
|-----|---------------|--------|
| https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html | TorchRL multi-agent PPO tutorial (MAPPO critic wording) | 200, read |
| https://benchmarl.readthedocs.io/en/latest/ | BenchMARL documentation | 200, listed |
| https://pettingzoo.farama.org/api/parallel/ | PettingZoo Parallel API page | 200, listed |
| https://pettingzoo.farama.org/tutorials/sb3/index.html | PettingZoo SB3 tutorial (single-agent warning) | 200, read |
| https://raw.githubusercontent.com/Farama-Foundation/PettingZoo/master/pettingzoo/utils/env.py | `ParallelEnv` source | 200, read |
| https://raw.githubusercontent.com/Farama-Foundation/SuperSuit/master/README.md | SuperSuit deprecation note | 200, read |
| https://gymnasium.farama.org/api/env/ | Gymnasium Env API | 200, listed |
| https://stable-baselines3.readthedocs.io/en/master/guide/algos.html | SB3 algorithm table | 200, read |
| https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html | Maskable PPO page | 200, listed |
| https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html | PyG installation (Python 3.10 to 3.14) | 200, read |
| https://networkx.org/documentation/stable/ | NetworkX documentation | 200, listed |
| https://docs.pydantic.dev/latest/concepts/json_schema/ | pydantic JSON Schema (now served from pydantic.dev/docs) | 200 after redirect, listed (content read in 0.1) |
| https://pypi.org/pypi/torch/json, https://pypi.org/pypi/torchrl/json, https://pypi.org/pypi/torch_geometric/json | PyPI JSON API (the same endpoint pattern was used for every package version cited) | 200, read |
| https://registry.npmjs.org/react/latest, https://registry.npmjs.org/cytoscape/latest, https://registry.npmjs.org/vite/latest | npm registry (same pattern for every npm version cited) | 200, read |

**Cited by the sources above but not fetched (do not rely on them):** the Contractor and Al Mallah thesis referenced by CyMARL-CommFormer; DBLP and Semantic Scholar search pages (blocked or rate-limited, Section 0.3).

**Fetched with a browser-style User-Agent only:** the PyTorch LTS wheel index named in the CyMARL README (https://download.pytorch.org/whl/lts/1.8/torch_lts.html) returned HTTP 403 to my first request with the default `curl` User-Agent and HTTP 200 (22,842 bytes, file names read) with a browser-style User-Agent. Its file list is the basis for the torch 1.8.2 statement in Section 2.5.

## Appendix C. Verification method

Tools: `curl` (follow redirects, `--compressed`), Python 3.13 standard library, `jq`, and `pypdf` plus `pyyaml` installed only inside a throwaway virtual environment under `/tmp/cs-research` (removed after the work). No browser tool was available, so sites that block scripted clients (Wiley, ScienceDirect, DBLP, Semantic Scholar under load) could not be read.

Representative commands (all unauthenticated):

```
curl -sS -L https://export.arxiv.org/api/query?id_list=2310.05939
curl -sS -G --data-urlencode "q=CMARL-ACD in:name" https://api.github.com/search/repositories
curl -sS -G --data-urlencode "query.bibliographic=CMARL-ACD" https://api.crossref.org/works
curl -sS https://pypi.org/pypi/<package>/json
python3 -m venv /tmp/<scratch> && /tmp/<scratch>/bin/pip install --dry-run --no-cache-dir --only-binary=:all: --report r.json <packages>
```

Limits: GitHub's unauthenticated search allows 10 requests per minute, which cut off three planned queries. A `pip` dry-run proves the resolver finds a set of files; it does not install or import anything. Text extracted from PDFs with `pypdf` can mis-order table cells, so table figures were read against the surrounding prose where possible.
