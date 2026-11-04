# Phase 1-2 research summary (verified facts)

Checked 2026-10-30. Full record: `docs/RESEARCH.md`. Technology choices: `docs/DECISIONS.md`.
Tags: VERIFIED = source fetched and read this session; REPORTED = a source's own claim, not reproduced; UNVERIFIED = not fetched or not found.

## CMARL-ACD
- VERIFIED: a GitHub repo `cyb3rlab/CMARL-ACD` exists, titled "CMARL-ACD: Competitive Multi-Agent Reinforcement Learning for Autonomous Cyber Defense". Owner: org `cyb3rlab` (README: Cybersecurity Research Lab, JAIST). LICENSE.md: MIT, copyright Thanh Cong Nguyen and Razvan Beuran, derived from NASim. Created 2026-06-03, last push 2026-06-23, 4 commits, about 62 MB.
- VERIFIED (source): clipped PPO in TorchRL; one flag `CRITIC_CENTRALIZED` gives IPPO (False, default) or MAPPO (True); attacker and defender co-train (competitive, 2 agents); NASim fork; defender sees the network through a simulated noisy IDS (SSH false positives); invalid actions masked; pins torch 2.7.0, torchrl 0.8.0, numpy 2.0.2; `python_requires` below 3.13; no tests directory.
- REPORTED: accompanying paper "CMARL-ACD: ... Under IDS Uncertainty Conditions", Computers & Security (under review), 2026; real-testbed validation with Snort and Metasploit. Paper NOT located (arXiv API and Crossref searches returned nothing relevant). Its `real-testbed/` half uses attack tooling and real iptables: never run, adapt or vendor.
- Inference (needs user confirmation): this is the intended "CMARL-ACD".

## CyMARL (the earlier finding)
- VERIFIED: arXiv 2310.05939 is a real paper: Jacob Wiebe, Ranwa Al Mallah, Li Li; v1 dated 25 Aug 2023; workshop paper at the 2nd International Workshop on Adaptive Cyber Defense, 2023. It is NOT the source of the name "CMARL-ACD".
- VERIFIED details: CybORG-based, PyMARL2 interface; 9 hosts (5 user, 4 operational), importance 0.1/1/10; 2 defenders split by subnet; heuristic attacker; confidentiality/integrity/availability scenarios; IQL and QMIX; 6 flags per host; invalid action -0.1; 2M training timesteps; heuristic defender uses true state; authors say the game needs no explicit coordination.
- Earlier claim "no public CyMARL repo" is REFUTED: `Poly-AIvsAI/CyMARL` (43 commits by `jgawiebe`, last push 2023-06-07, no LICENSE file, Python 3.9 and torch 1.8.2, tested on Windows 10). Link to the paper is inferred, not stated by the paper.

## Other sources
- Yu et al. arXiv 2103.01955 (NeurIPS 2022 Datasets and Benchmarks): MAPPO = PPO with centralized value input; IPPO = local inputs; official code `marlbenchmark/on-policy` (MIT; Python 3.6.1 / torch 1.5.1 instructions).
- PyMARL2 (`hijkzzz/pymarl2`, Apache-2.0, value-based, SMAC); Hu et al. arXiv 2102.03479.
- CAGE Challenge 4: AAAI 2025 paper page fetched; 5 blue agents, PettingZoo wrapper, green false alerts, penalties for disrupting green work; pins numpy 1.26.4 / torch 2.2.0 (no cp313 wheels). CybORG is public and MIT (so `docs/ARCHITECTURE.md` "not open-sourced" is wrong). Wiley AI Magazine article NOT fetched (blocked).
- Yawning Titan: archived (last push 2024-05-21), MIT. CyberBattleSim: MIT, attacker-focused, scripted defender.
- Overlapping prior work (abstracts only): arXiv 2606.13832 (CAGE-4, MAPPO + graph encoder + risk/budget constraints), arXiv 2401.10149 (MAPPO beat IPPO in a maritime OT sim). So uncertainty-aware and proportionate-response ideas are not new; CyberSwarm's difference is a hypothesis.

## docs/ARCHITECTURE.md
- Untracked, 546 lines, modified 2026-10-29 (four days after the initial commit); unsigned draft v0.1. Already uses the multi-agent RL direction, names Raksha, Mayukh and the project lead, and says GNNs are optional. Conflicts to reconcile are listed in `docs/RESEARCH.md` Appendix A (names CyMARL not CMARL-ACD; wrong repo and CybORG claims; simulator as `gymnasium.Env` returning reward; `ground_truth` field inside event records; different action names; Ray/RLlib; root-cause metric). Not modified.

## Technology (details and versions in docs/DECISIONS.md)
- Python floor 3.12, dev on 3.13.2 (3.12.12 also installed). Resolver dry run on 3.13.2 succeeded for torch 2.15.0, gymnasium 1.4.0, pettingzoo 1.27.0, pydantic 2.14.2, networkx 3.7, numpy 2.5.4, pytest 9.2.0.
- Contracts: pydantic v2 + exported JSON Schema (checked: forbids extras, unknown action kinds, raw IPs). Env API: PettingZoo ParallelEnv. IPPO/MAPPO: self-written PyTorch, one shared code path. SB3, TorchRL, BenchMARL, EPyMARL, RLlib not used. PyG not adopted now. NetworkX for simulator/mock only. Logging as JSONL/CSV. Dashboard: React + Vite + TypeScript + Cytoscape.js (Recommended; depends on Mayukh's JS/TS skills).

## Blockers and open items
- Disk: 272 MiB free when measured (about 911 MiB after cleanup); torch needs 564.6 MB installed. A temp install already failed with ENOSPC. User must free space before the `.venv` step.
- Open: confirm CMARL-ACD identity; MAPPO critic input (no ground truth); agent roles; Mayukh's skills; Python floor; license.
