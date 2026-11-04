# Research delta summary (verified and unverified facts)

Checked 2026-10-30. Full record: `docs/RESEARCH.md` (v0.2). Decisions: `docs/DECISIONS.md` (v0.2, entries AD-01 to AD-03).
Tags: VERIFIED = fetched and read this session. REPORTED = a source's own claim, not reproduced. UNVERIFIED = not fetched or not found. INFERENCE = my reading, not stated by a source.
Nothing third-party was run or cloned. Only `pip install --dry-run` resolutions ran, in throwaway venvs under `/tmp` (removed). The system Python is unchanged.

## 1. Was "CMARL-ACD" identified?

Yes as a name, no as a published paper. It is **not** CyMARL.

- VERIFIED: GitHub repo `cyb3rlab/CMARL-ACD` ("CMARL-ACD: Competitive Multi-Agent Reinforcement Learning for Autonomous Cyber Defense"), Cybersecurity Research Lab at JAIST, MIT text derived from NASim, created 2026-06-03, 4 commits by `rzvnbr`. Seven GitHub repository searches (four exact-name variants, three descriptive) each returned only this repository.
- VERIFIED: JAIST master's thesis (hdl 10119/20030), Thanh Cong Nguyen, supervisor Razvan Beuran, Aug 2025 (issue date Sep-2025), "Automated Cyber Defense Based on Reinforcement Learning Techniques". Its text names its framework "CMARL-ACD" (6 occurrences) and says the code will be released.
- REPORTED, paper UNVERIFIED: the README says the code accompanies "CMARL-ACD: ... Under IDS Uncertainty Conditions", *Computers & Security* (under review), 2026. Not found in arXiv, Crossref, OpenAlex, web search or the lab's publications page. DBLP (anti-bot page) and Semantic Scholar (HTTP 429) could not be queried.
- INFERENCE: this is what the brief means. The user has not confirmed it.

## 2. arXiv:2310.05939 (CyMARL), all VERIFIED unless marked

- Title: Learning Cyber Defence Tactics from Scratch with Multi-Agent Reinforcement Learning. Authors: Jacob Wiebe, Ranwa Al Mallah (Royal Military College of Canada), Li Li (Defence Research and Development Canada). v1 only, 25 Aug 2023, cs.CR/cs.AI/cs.LG.
- Venue: workshop paper, "2nd International Workshop on Adaptive Cyber Defense, 2023" (arXiv:2308.09520; journal-ref `ACD/2023/106`). No journal or conference version found.
- Algorithms: IQL and QMIX (PyMARL2 implementations, RNN agents), against a heuristic defender that reads true state. Not PPO.
- Environment: CybORG (CAGE-2 base) with a PyMARL interface; 9 hosts (5 user, 4 operational); importance 0.1 / 1 / 10; two defenders split by subnet; one heuristic attacker (confidentiality, integrity or availability goal).
- Observation: six one-hot flags per host. Actions: monitor, remove, restore, analyze, data_repair, misinform; invalid action -0.1.
- Reward: each compromised host costs the team its importance score; restore penalty scales with importance; shared team reward.
- Metrics: mean return and standard deviation; 2M training timesteps; 1000-step evaluation; five evaluation runs with separately trained static policies. REPORTED results: IQL and QMIX beat the heuristic in all six variants (Table 1 in RESEARCH.md 1.1); authors say the game needs no explicit coordination.
- Code: no link in the paper. `Poly-AIvsAI/CyMARL` exists (43 commits by `jgawiebe`; `RMC-AIvsAI/CyMARL` redirects to it; no licence; default branch `master`). Identity with the paper is INFERENCE.
- Dependencies and limits: torch 1.8.2 from PyTorch's LTS index (not on PyPI; the README's cu111 folder has Linux and Windows wheels only; the index has no macOS arm64 wheel for Python 3.9), gym 0.26.0, scikit-learn 0.24.2, PyYAML 5.3.1; "tested using Windows 10", Python 3.9. A dry-run of `setup.py` requirements on Python 3.12.12 failed (scikit-learn build). Paper gives no episode length or software versions; code defaults to 30 turns (config) and `t_max` 2,050,000; shipped batch scripts are 100.5k and 1.005M steps, not the 2M protocol.

## 3. CMARL-ACD facts (repo and thesis)

- VERIFIED: clipped PPO in TorchRL; `CRITIC_CENTRALIZED` False gives IPPO, True gives MAPPO; attacker and defender co-train (competitive); NASim fork; simulated IDS with SSH false positives (Scenario 1 clean, Scenario 2 noisy); defender actions active in the action list: `decoy_services`, `ip_block` (IP1 to IP5), `noop` (`disable_services` and `toggle_monitoring` exist but are commented out); invalid actions masked; reward = action value - cost + block reward or wrong-block penalty - time-weighted penalty per compromised host + decoy bonus; metrics Hosts Compromised, Decoy Interaction Ratio, True Blocking Rate, worst-case score and exploitability.
- VERIFIED (thesis): five hosts, 150-step episodes, checkpoints every 100k timesteps, 50 episodes per checkpoint pair; convergence about 0.4M (clean) and 3M (noisy) timesteps; no seeds, standard deviations or confidence intervals mentioned; no agent-side uncertainty estimate (the IDS noise is the "uncertainty").
- REPORTED: IPPO and MAPPO equal in the clean scenario; under noise IPPO limits worst-case compromise to about 2 hosts versus 3 for MAPPO; sim-to-real within 5%.
- VERIFIED inconsistencies: the thesis abstract (79%, 3.8 to 0.8 hosts, MAPPO TBR 96.1%) disagrees with its body and Table 4.3 (83%, 0.64 hosts, IPPO 98.0%, MAPPO 100%). `best_models/README.md` says MAPPO is better in Scenario 2 (worst case -64.81 vs -29.66) without defining the sign, which does not match the thesis ordering.
- Reproducibility: pinned `requirements.txt` resolves on Python 3.12.12, fails on 3.13.2 (`numpy==2.0.2` has no cp313 wheel); no tests; default 30M frames; `real-testbed/` needs root Linux, KVM, Snort, Metasploit, iptables and must never be run or adapted.

## 4. Other sources

- Yu et al. arXiv:2103.01955 (NeurIPS 2022 Datasets and Benchmarks): MAPPO = centralized value input, IPPO = local; shared-reward setting; five practical suggestions re-read. Official code `marlbenchmark/on-policy` (MIT; Python 3.6.1 / torch 1.5.1; README warns MAPPO results are often not reproduced).
- PyMARL2 (`hijkzzz/pymarl2`, Apache-2.0, Python 3.8, SMAC, value-based). Hu et al. arXiv:2102.03479.
- CAGE Challenge 4: repo, README, requirements, licence text read; AAAI 2025 paper landing page and Crossref metadata read; AI Magazine article metadata via Crossref only (Wiley 403). Requirements resolve on Python 3.12.12, not 3.13.2.
- Yawning Titan (archived, MIT), CyberBattleSim (MIT, active, Linux/WSL), Kunz et al. arXiv:2304.11052, NASim (MIT): READMEs and abstracts only.
- Overlap (abstracts only): arXiv:2606.13832 (CAGE-4, MAPPO with graph attention encoder, safety budgets, CVaR), arXiv:2401.10149 (MAPPO beat IPPO in a maritime OT sim). "Uncertainty-aware" and "graph MARL" are not new; CyberSwarm's difference is a hypothesis only.

## 5. Technology (PyPI/npm 2026-10-30; resolver dry-runs only)

| Item | Finding |
|------|---------|
| Local Python | 3.13.2 default; 3.12.12 also present; macOS 15.7.4 arm64. Core set (torch 2.15.0, gymnasium 1.4.0, pettingzoo 1.27.0, pydantic 2.14.2, numpy 2.5.4, networkx 3.7, pytest 9.2.0, jsonschema 4.26.0) resolves on both. |
| Stable-Baselines3 2.9.2 | Single-agent. PettingZoo's SB3 tutorial: "does not directly support multi-agent algorithms or environments". SuperSuit 3.11.0 needs `tinyscaler` (no cp313 wheel). Not a dependency. |
| TorchRL 0.14.1 | Ships `IPPOLoss`, `MAPPOLoss`, `MultiAgentGAE`, `MultiAgentMLP`, `MaskedCategorical`, `PettingZooWrapper` (corrects v0.1). Needs `tensordict<0.15`. Named contingency or parity cross-check. |
| BenchMARL 1.5.2 | README lists MAPPO and IPPO. Needs `torchrl<0.12`, hydra, torchvision, `av<14`. Resolves on 3.13.2; runtime untested. Not for the core. |
| EPyMARL, on-policy | Reference only (not on PyPI; old pins). |
| Self-written compact IPPO | Recommended (D-06, status changed to Recommended); MAPPO only if tests pass and the critic input is settled. |
| NetworkX 3.7 | Default topology and graph-processing foundation for the required graph pipeline and the simulator and mock (current `DELTA.md`); not imported by the contracts package (D-09). |
| Graph pipeline and hand-engineered graph features | Required, lead-owned, before the RL environment; no learned message passing, no PyG (D-18). New hypothesis E8: conventional versus graph-feature observations on matched scenarios and seeds. |
| PyG 2.8.1 | Staged, optional extra, never core; supports Python 3.10 to 3.14; resolves on 3.13.2; pulls aiohttp and requests (D-10). |
| GraphSAGE / GAT | Go/no-go taken after IPPO is stable; a documented no-go satisfies the stage. Dependency feasibility is not the blocker; task, supervision source and matched control are not yet defined (D-19). |
| pydantic 2.14.2 | Required for contracts (D-03). |
| Dashboard | React 19.3.1, Vite 8.3.5, TypeScript 7.0.2, Cytoscape.js 3.34.4, Vitest 5.0.5 (unchanged). `react-cytoscapejs` last published 2022: use Cytoscape directly. |
| Disk | About 650 MiB free at the end. Torch needs about 0.7 to 1 GB peak. The user should free space before the venv step. |

## 6. Could not fetch (UNVERIFIED)

Wiley AI Magazine article text (403); ScienceDirect (403); DBLP (anti-bot); Semantic Scholar (429); `hdl.handle.net` thesis link (503, repository maintenance; the `dspace.jaist.ac.jp` copy worked); three of ten planned GitHub searches (rate limit); the CMARL-ACD journal paper (not found); CC4 AAAI paper text (not read).

## 6b. DELTA.md changed during this work (needs a decision)

The `DELTA.md` read at the start (12,728 bytes) said the graph pipeline and graph-feature baseline were not required, NetworkX was optional support, and PyG, GraphSAGE/GAT and the graph-feature pipeline were optional or future. The current `DELTA.md` (modified 2026-10-30 19:49, 18,005 bytes) says it supersedes that sentence: MARL is primary, the GNN / Faultline concept is not the central claim, graph representation and hand-engineered graph features are REQUIRED lead-owned foundations built before the RL environment, NetworkX is the default foundation, and a learned GNN is a staged secondary hypothesis. The step instructions for this delta used the earlier wording. `docs/DECISIONS.md` (AD-01 supersession note, D-09, D-10, D-17 to D-19) and `docs/RESEARCH.md` (E6, E8, sections 2.6 and 4) follow the current file. If the earlier wording was intended, those are the places to change.

## 7. Corrections to the earlier (0.1) research

CMARL-ACD is not "unverified as a name"; TorchRL has IPPO/MAPPO losses; BenchMARL lists IPPO; CC4 pins resolve on 3.12; only three CMARL-ACD defender actions are active; best_models "MAPPO better" is ambiguous; CyMARL repo identity strengthened by the 301 redirect; free disk now about 650 MiB.

## 8. Files changed in this delta

- `/Users/guru/CyberSwarm-GX/docs/RESEARCH.md` (rewritten as v0.2, 0.1 content kept and corrected)
- `/Users/guru/CyberSwarm-GX/docs/DECISIONS.md` (v0.2: AD-01 pivot, AD-02, AD-03, table updates D-01 to D-19, section 1, 3, 4, 5)
- `/Users/guru/CyberSwarm-GX/.agents/tasks/cyberswarm-foundation/research-delta.md` (this file)
- Not modified: `README.md`, `docs/ARCHITECTURE.md`. No git commit, push or config change.
