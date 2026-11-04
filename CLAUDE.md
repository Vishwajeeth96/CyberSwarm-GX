# 🚀 CyberSwarm - 12 Task Guide

**Who are you?**
- `raksha` → Simulator, engine, coordinator tasks
- `mayukh` → Evaluation, dashboard tasks
- `shared` → Anyone can do these

---

## Task Map

| # | Task | Owner | Status | Milestone |
|---|------|-------|--------|-----------|
| 1 | Coordinator Tests | Raksha | 🔴 Not Started | Run 20+ tests, all pass |
| 2 | IPPO Training Core | Shared | 🔴 Not Started | Trains for 1 epoch without error |
| 3 | IPPO Evaluation Slice | Mayukh | 🔴 Not Started | Produces AggregateMetrics |
| 4 | Uncertainty Wrapper | Shared | 🔴 Not Started | Wraps IPPO, same interface |
| 5 | Matched Control | Shared | 🔴 Not Started | Baseline for comparison |
| 6 | Evaluation Harness | Mayukh | 🔴 Not Started | Runs comparison, outputs JSON |
| 7 | Dashboard Bundle | Mayukh | 🔴 Not Started | JSON + visualizations |
| 8 | End-to-End Smoke | Shared | 🔴 Not Started | Full pipeline runs |
| 9 | Learned GNN (Go/No-Go) | Shared | 🔴 Not Started | Decision recorded |
| 10 | MAPPO Implementation | Shared | 🔴 Not Started | Trains, compares to IPPO |
| 11 | Final Documentation | Shared | 🔴 Not Started | All docs in sync |
| 12 | Final Validation | Mayukh | 🔴 Not Started | 652+ tests pass |

---

## 🔴 TASK 1: Coordinator Tests (Raksha)

**Owner:** Raksha  
**Goal:** Write and run coordinator safety tests

### What's Needed
```bash
# Check existing coordinator code
ls src/cyberswarm/coordinator/

# Create test file
touch tests/coordinator/test_validator.py
touch tests/coordinator/test_recovery.py
```

### Tests to Write
1. Validator rejects invalid proposals (schema, unknown action, out-of-authority)
2. Validator accepts valid proposals
3. RecoveryVerifier tracks restore_node → up status
4. Coordinator applies accepted actions via simulator
5. Coordinator rejects without state change

### ✅ Milestone
```
.venv/bin/python -m pytest tests/coordinator -q
# Expected: 20+ passed, 0 failed
```

---

## 🔴 TASK 2: IPPO Training Core (Shared)

**Owner:** Raksha (help integrate)  
**Goal:** Implement Independent PPO training

### Read First
```bash
cat docs/RL_IMPLEMENTATION.md | grep -A 40 "## 6. IPPO"
```

### Create
```
src/cyberswarm/marl/
├── __init__.py
├── ippo.py        # Main IPPO implementation
├── networks.py    # Actor/Critic networks
└── trainer.py     # Training loop
```

### What to Build
1. **Actor Network** — Maps obs → action distribution (policy head per agent)
2. **Critic Network** — Maps state/obs → value estimate
3. **PPO Loss** — Clipped surrogate objective
4. **Trainer** — Collect episodes, update networks, save checkpoints

### Interface
```python
from cyberswarm.rl.env import CyberSwarmParallelEnv

env = CyberSwarmParallelEnv(engine_factory, scenarios)
# Must work with existing env!
```

### ✅ Milestone
```bash
.venv/bin/python -c "from cyberswarm.marl.ippo import IPPOTrainer; print('imports OK')"
# Trains for 1 epoch: python -m cyberswarm.marl.train --epochs 1
```

---

## 🔴 TASK 3: IPPO Evaluation Slice (Mayukh)

**Owner:** Mayukh  
**Goal:** Extract metrics from trained IPPO

### Read First
```bash
grep -n "AggregateMetrics\|origin_rank" src/cyberswarm/contracts/results.py
```

### Create
```
src/cyberswarm/marl/
└── eval.py        # Evaluation wrapper
```

### What to Build
1. Load trained checkpoint
2. Run episodes (固定 seed, multiple runs)
3. Compute:
   - Episode reward
   - Steps to resolve
   - Containment count
   - Service disruption
   - Origin top-1 accuracy (if scoring enabled)

### Output
```python
# Must produce:
AggregateMetrics(
    episode_id="...",
    policy_family_id="ippo",
    reward_mean=...,
    steps_to_resolve_mean=...,
    origin_rank_top1=0.75,  # if scoring enabled
    ...
)
```

### ✅ Milestone
```bash
.venv/bin/python -m cyberswarm.marl.eval --checkpoint checkpoints/ippo-v1 --runs 3
# Produces valid AggregateMetrics JSON
```

---

## 🔴 TASK 4: Uncertainty Wrapper (Shared)

**Owner:** Raksha  
**Goal:** Wrap IPPO with uncertainty estimation

### Read First
```bash
cat docs/RL_IMPLEMENTATION.md | grep -A 30 "## 7. Uncertainty"
```

### Create
```
src/cyberswarm/uncertainty/
├── __init__.py
├── wrapper.py     # Uncertainty-aware policy
└── head.py        # Epistemic uncertainty head
```

### What to Build
1. **Uncertainty Head** — Extra network output for confidence
2. **Decision Rule** — If uncertainty > threshold → request evidence else act
3. **Wrapper** — Wraps IPPO actor, same interface

### ✅ Milestone
```python
from cyberswarm.uncertainty import UncertaintyAwarePPO
policy = UncertaintyAwarePPO(checkpoint)
action, confidence = policy.step(obs)
# confidence in [0, 1]
```

---

## 🔴 TASK 5: Matched Control (Shared)

**Owner:** Raksha  
**Goal:** Non-uncertainty baseline for comparison

### What to Build
- Same network architecture as TASK 4
- Same training procedure
- NO uncertainty head
- Identical hyperparameters

### Purpose
Compare uncertainty-aware vs non-aware fairly

### ✅ Milestone
```bash
# Produces same metrics format as TASK 3
.venv/bin/python -m cyberswarm.marl.eval --variant control --runs 3
```

---

## 🔴 TASK 6: Evaluation Harness (Mayukh)

**Owner:** Mayukh  
**Goal:** Run full comparison across policies

### Read First
```bash
cat docs/ARCHITECTURE.md | grep -A 15 "## 7. Data flow"
```

### Create
```
src/cyberswarm/evaluation/
├── __init__.py
├── harness.py     # Main comparison runner
├── config.py      # Eval config (seeds, runs, scenarios)
└── compare.py     # Statistical comparison
```

### What to Build
1. **Harness** — Runs multiple policies on same scenarios/seeds
2. **Comparison** — T-test or bootstrap for significance
3. **Output** — JSON Lines with AggregateMetrics per policy

### ✅ Milestone
```bash
.venv/bin/python -m cyberswarm.evaluation.harness \
    --policies rule,ippo,uncertainty,control \
    --seeds 1,2,3,4,5 \
    --runs 3
# Output: comparison.jsonl with 4 policies × 5 seeds × 3 runs
```

---

## 🔴 TASK 7: Dashboard Bundle (Mayukh)

**Owner:** Mayukh  
**Goal:** Create visualization-ready output

### Read First
```bash
ls fixtures/records/valid/ | head -10
```

### Create
```
outputs/
├── summary.json           # Key metrics per policy
├── comparison.csv         # Full table for spreadsheet
├── episodes/              # Per-episode JSON logs
└── plots/                 # (Optional) matplotlib outputs
```

### What to Build
1. **Summary JSON** — Aggregated metrics, rankings
2. **CSV Table** — All runs, all metrics, easy to import
3. **Episode Logs** — Per-episode detail for debugging

### ✅ Milestone
```bash
ls outputs/
summary.json  comparison.csv  episodes/
```

---

## 🔴 TASK 8: End-to-End Smoke (Shared)

**Owner:** Shared  
**Goal:** Full pipeline runs without errors

### What to Test
```bash
# Full pipeline smoke test:
1. Train IPPO for 1 epoch
2. Evaluate on 3 seeds
3. Compare with rule baseline
4. Generate dashboard output

.venv/bin/python scripts/smoke_test.py
```

### ✅ Milestone
```
smoke_test.py
✓ Environment loads
✓ Training runs 1 epoch
✓ Evaluation produces metrics
✓ Comparison runs
✓ Dashboard generates
✓ All output files created
```

---

## 🔴 TASK 9: Learned GNN Go/No-Go (Shared)

**Owner:** Raksha  
**Goal:** Decide whether learned graph helps

### Read First
```bash
cat docs/RL_IMPLEMENTATION.md | grep -A 40 "## 8. Learned Graph"
```

### What to Do
1. Implement GraphSAGE or GAT encoder
2. Replace conventional obs with graph-encoded
3. Train for limited epochs (10)
4. Compare to graph-feature baseline

### Decision Criteria
- **GO:** Measurable improvement over baseline
- **NO-GO:** No improvement → report result, move on

### ✅ Milestone
```bash
# Record decision in manifest
python -m cyberswarm.evaluation.record_decision \
    --stage learned_gnn \
    --decision go|no_go \
    --reason "X% improvement over baseline"
```

---

## 🔴 TASK 10: MAPPO Implementation (Shared)

**Owner:** Raksha  
**Goal:** Implement MAPPO and compare to IPPO

### Read First
```bash
cat docs/RL_IMPLEMENTATION.md | grep -A 30 "## 9. MAPPO"
```

### What to Build
```
src/cyberswarm/marl/
└── mappo.py        # Centralized critic variant
```

### Key Differences from IPPO
- Centralized critic sees all agents' observations
- Shared value function
- Same PPO update

### ✅ Milestone
```bash
.venv/bin/python -m cyberswarm.marl.train --algorithm mappo --epochs 10
# Compare: MAPPO vs IPPO on same seeds
```

---

## 🔴 TASK 11: Final Documentation (Shared)

**Owner:** Mayukh  
**Goal:** Sync all docs, create teammate guides

### What to Do
1. Update ARCHITECTURE.md with implemented features
2. Update RL_IMPLEMENTATION.md with actual interfaces
3. Create module READMEs in each package
4. Final teammate handoff docs

### ✅ Milestone
```
docs/
├── GETTING_STARTED.md    # New!
├── ARCHITECTURE.md       # Updated
├── RL_IMPLEMENTATION.md  # Updated
src/cyberswarm/*/README.md  # Each package has docs
```

---

## 🔴 TASK 12: Final Validation (Mayukh)

**Owner:** Mayukh  
**Goal:** Full test suite passes, repo ready

### What to Do
```bash
# Run full test suite
.venv/bin/python -m pytest -q

# Must pass 652+ tests
# No regressions
```

### Final Checklist
- [ ] All 652+ tests pass
- [ ] No import boundary violations
- [ ] All schemas export correctly
- [ ] README updated
- [ ] PyPI-ready (if applicable)

### ✅ Milestone
```
========== 652 passed in X.XXs ==========
```

---

## 🏃 Quick Commands

```bash
# Activate
source .venv/bin/activate

# Run all tests
.venv/bin/python -m pytest -q

# Run specific area
.venv/bin/python -m pytest tests/rl -q
.venv/bin/python -m pytest tests/engine -q
.venv/bin/python -m pytest tests/contracts -q

# Check imports
.venv/bin/python -c "from cyberswarm.marl.ippo import IPPOTrainer"

# Run a task's milestone directly
# (each task has a specific command in its section above)
```

---

**👉 Type your name (raksha/mayukh/shared) to focus on your tasks!**
