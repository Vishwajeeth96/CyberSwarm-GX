"""Cross-record consistency on the S1 mini-episode (DELTA checks DC-05, DC-04).

Loads the committed mini-episode from disk and runs the cross-record checker.
Rules X-1 through X-9 (id resolution, agent ids, episode and step ranges, event
contiguity, observation events, the proposal/validation/execution chain,
outcome events and accepted-vs-available) hold with zero issues, and a negative
control proves the checker actually fires. X-10 (no evaluator-only field in any
agent-visible file) and X-14 (uncomputed optional values are null) are checked
directly against the committed bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

from cyberswarm.contracts import (
    ActionKind,
    AgentObservation,
    AvailableActions,
    DetectionSummary,
    EpisodeRecordSet,
    EpisodeResult,
    EventRecord,
    ExecutedAction,
    GroundTruthSnapshot,
    NodeScoreRecord,
    PolicyProposal,
    Topology,
    ValidationResult,
    check_episode,
    check_no_evaluator_fields,
    read_json,
    read_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MINI = REPO_ROOT / "fixtures" / "mini_episode"


def _load_components() -> dict:
    return {
        "topology": read_json(MINI / "topology.json", Topology),
        "events": list(read_jsonl(MINI / "events.jsonl", EventRecord)),
        "observations": list(read_jsonl(MINI / "observations.jsonl", AgentObservation)),
        "available_actions": list(
            read_jsonl(MINI / "available_actions.jsonl", AvailableActions)
        ),
        "detection": list(read_jsonl(MINI / "detection.jsonl", DetectionSummary)),
        "node_scores": list(read_jsonl(MINI / "node_scores.jsonl", NodeScoreRecord)),
        "proposals": list(read_jsonl(MINI / "proposals.jsonl", PolicyProposal)),
        "validations": list(read_jsonl(MINI / "validations.jsonl", ValidationResult)),
        "executions": list(read_jsonl(MINI / "executions.jsonl", ExecutedAction)),
        "ground_truth": list(
            read_jsonl(MINI / "evaluator_only" / "ground_truth.jsonl", GroundTruthSnapshot)
        ),
        "episode_result": read_json(MINI / "episode_result.json", EpisodeResult),
    }


def _record_set(components: dict) -> EpisodeRecordSet:
    return EpisodeRecordSet(
        events=components["events"],
        observations=components["observations"],
        available_actions=components["available_actions"],
        detection=components["detection"],
        node_scores=components["node_scores"],
        proposals=components["proposals"],
        validations=components["validations"],
        executions=components["executions"],
        ground_truth=components["ground_truth"],
        episode_result=components["episode_result"],
    )


def _agent_visible_files() -> list[Path]:
    return [
        path
        for path in sorted(MINI.rglob("*.json*"))
        if "evaluator_only" not in path.parts
    ]


# --------------------------------------------------------------------------- #
# X-1 .. X-9 (and X-11, X-12, X-15, X-16, which the same checker covers)
# --------------------------------------------------------------------------- #
def test_mini_episode_is_cross_record_consistent() -> None:
    components = _load_components()
    issues = check_episode(_record_set(components), components["topology"])
    assert issues == [], "\n".join(f"{i.rule}: {i.message}" for i in issues)


def test_checker_detects_an_accepted_action_absent_from_available() -> None:
    # Negative control for X-9: drop isolate_node from the dmz step-2 available
    # actions, so the accepted isolation no longer appears there.
    components = _load_components()
    patched = []
    for record in components["available_actions"]:
        if record.step == 2 and record.agent_id == "agent-dmz":
            kept = [
                action
                for action in record.actions
                if action.action_kind is not ActionKind.isolate_node
            ]
            patched.append(record.model_copy(update={"actions": kept}))
        else:
            patched.append(record)
    components["available_actions"] = patched

    issues = check_episode(_record_set(components), components["topology"])
    assert any(issue.rule == "X-9" for issue in issues)


# --------------------------------------------------------------------------- #
# X-10: no evaluator-only field in any agent-visible file
# --------------------------------------------------------------------------- #
def test_x10_no_evaluator_fields_in_agent_visible_files() -> None:
    assert check_no_evaluator_fields(_agent_visible_files()) == []


def test_x10_skips_the_evaluator_only_directory() -> None:
    ground_truth = MINI / "evaluator_only" / "ground_truth.jsonl"
    # The file legitimately carries ground-truth fields, but the scan skips
    # anything under an evaluator_only/ directory.
    assert check_no_evaluator_fields([ground_truth]) == []


def test_x10_detects_a_planted_evaluator_field(tmp_path) -> None:
    planted = tmp_path / "observations.jsonl"
    planted.write_text(
        json.dumps({"observation_id": "obs-0000-agent-dmz", "compromised": True}) + "\n",
        encoding="utf-8",
    )
    violations = check_no_evaluator_fields([planted])
    assert violations
    assert any("compromised" in v for v in violations)


# --------------------------------------------------------------------------- #
# X-14: optional values that were not computed are null
# --------------------------------------------------------------------------- #
def test_x14_uncomputed_optionals_are_null() -> None:
    components = _load_components()
    result = components["episode_result"]
    assert result.reward is None
    assert result.uncertainty_summary is None
    # The rule-based policy computes no uncertainty signal.
    assert all(proposal.uncertainty is None for proposal in components["proposals"])
