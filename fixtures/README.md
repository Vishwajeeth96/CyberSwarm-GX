# CyberSwarm fixtures

**Every file in this directory is synthetic and illustrative. Nothing here is a
real result, a research finding or a simulator run.** The values are hand-written
to exercise the contracts and to give the simulator and dashboard teams concrete
examples to code against. Records that carry a `provenance` field are labelled
`fixture_synthetic`; the rest are documented as synthetic here.

All fixtures are generated and checked for drift, so they never diverge from the
contracts:

- `scripts/fixture_storyboard.py` holds the literal S1 data and imports only
  `cyberswarm.contracts` (no simulator, graph, coordinator or evaluator code).
- `scripts/graph_fixtures.py` *derives* the S3 graph records
  (`GraphSnapshot.dmz-step5` / `-step6`, `GraphFeatureRecord.dmz-step5`, and the
  two invalid graph snapshots) by running the real `cyberswarm.graph` converter
  on the storyboard's observation inputs. This keeps the storyboard
  contracts-only while the graph snapshots remain genuine converter output.
- `scripts/build_fixtures.py --write` writes every fixture below (storyboard +
  derived graph records).
- `scripts/build_fixtures.py --check` regenerates them in memory — re-running
  the converter for the graph records — and fails if a committed file differs by
  a single byte.

Run both from the repository root with the repo-local interpreter:

```
.venv/bin/python scripts/build_fixtures.py --write
.venv/bin/python scripts/build_fixtures.py --check
```

## Layout

- `records/valid/<RecordName>.<case>.json` — one valid example per record. The
  S1 records come from the storyboard; the S3 graph records
  (`GraphSnapshot.dmz-step5`, `GraphSnapshot.dmz-step6`,
  `GraphFeatureRecord.dmz-step5`) are the real converter's output. The two
  graph snapshots use an identical node and edge ordering (topology order) while
  their dynamic state differs, which is why step 6 looks changed but still lines
  up with step 5. Each file is a single canonical JSON document that validates
  both through its pydantic model and against the exported schema in `schemas/`.
  Optional fields that no producer computed are written as `null` (never a
  default number), so these files show the on-disk shape, not the abbreviated
  form used in the documentation examples.
- `records/invalid/<RecordName>.<case>.json` — one example per record that
  breaks **exactly one** rule. Each must be rejected by its model; the file name
  states the broken rule. Several of these deliberately try to smuggle unsafe
  text into a `ShortText` field (a raw IP such as a dotted quad, a path-traversal
  `..` sequence, or a character outside the allowed set) to prove the contracts
  reject it. These files are expected to fail validation and are excluded from
  any "valid fixtures" sweep by their directory.
- `mini_episode/` — one small, internally consistent episode on a healthy
  three-node topology, used for the cross-record consistency checks. It follows
  the run-directory layout: `topology.json`, the agent-visible record streams
  (`events`, `observations`, `available_actions`, `detection`, `node_scores`,
  `proposals`, `validations`, `executions`), `episode_result.json`, and the
  evaluator-only ground-truth stream under `evaluator_only/ground_truth.jsonl`.
  The episode includes a healthy start, an alert, an investigation, one rejected
  cross-authority proposal and one accepted containment, so the
  proposal/validation/execution chain and the accepted/rejected paths are both
  exercised. The ground-truth stream lives only under `evaluator_only/` and is
  never an agent input.

Only the project lead edits these fixtures, the contracts and the schemas. The
simulator and dashboard teams consume them; change requests go to the project
lead (see `docs/CONTRACTS.md`, "Compatibility policy").
