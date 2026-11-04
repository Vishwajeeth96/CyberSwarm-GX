"""Episode record sinks (stage S3, DESIGN section 4.6).

The :class:`EpisodeEngine` hands every contract record it produces to a
:class:`Recorder`. Three implementations cover the needs of the project:

* :class:`EpisodeRecorder` writes the per-episode files of CONTRACTS section 15
  under one episode directory: the agent-visible ``*.jsonl`` streams, the
  single ``episode_result.json`` document, and the evaluator overlay
  ``evaluator_only/ground_truth.jsonl``. Each stream is opened lazily on its
  first record (so an unused optional file such as ``graph_model_outputs.jsonl``
  stays absent), except ``errors.jsonl``, which is created eagerly because the
  contract lists it as a file that "may be empty".
* :class:`MemoryRecorder` keeps the records in memory, grouped by record type.
  It is what the tests and the RL environment use when no files are wanted.
* :class:`NullRecorder` drops everything. Training uses it so a rollout writes
  nothing to disk.

A record is routed by its class name alone (``type(record).__name__``). This
module therefore imports no record classes - in particular it never imports
``cyberswarm.contracts.evaluator_only`` or ``GroundTruthSnapshot`` - so it sits
cleanly inside the engine package's import boundary (DESIGN section 3.2); only
``engine.step_engine`` is allowed to touch evaluator-only truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..contracts import ContractModel, JsonlWriter, write_json

__all__ = [
    "Recorder",
    "EpisodeRecorder",
    "MemoryRecorder",
    "NullRecorder",
    "EPISODE_JSONL_FILES",
    "EPISODE_JSON_FILES",
]

#: Record class name -> JSON Lines file, relative to the episode directory
#: (CONTRACTS section 15). ``GroundTruthSnapshot`` is the evaluator overlay and
#: is the only entry written under ``evaluator_only/``.
EPISODE_JSONL_FILES: dict[str, str] = {
    "EventRecord": "events.jsonl",
    "AgentObservation": "observations.jsonl",
    "AvailableActions": "available_actions.jsonl",
    "DetectionSummary": "detection.jsonl",
    "NodeScoreRecord": "node_scores.jsonl",
    "GraphSnapshot": "graph_snapshots.jsonl",
    "GraphFeatureRecord": "graph_features.jsonl",
    "GraphModelOutput": "graph_model_outputs.jsonl",
    "PolicyProposal": "proposals.jsonl",
    "ValidationResult": "validations.jsonl",
    "ExecutedAction": "executions.jsonl",
    "ErrorRecord": "errors.jsonl",
    "GroundTruthSnapshot": "evaluator_only/ground_truth.jsonl",
}

#: Record class name -> single JSON document, relative to the episode directory.
EPISODE_JSON_FILES: dict[str, str] = {
    "EpisodeResult": "episode_result.json",
}

#: Created eagerly so it always exists even with no records (CONTRACTS 15).
_EAGER_JSONL: tuple[str, ...] = ("ErrorRecord",)


@runtime_checkable
class Recorder(Protocol):
    """A sink for contract records (DESIGN section 4.6).

    ``write`` accepts any already-validated :class:`~cyberswarm.contracts.ContractModel`
    and is responsible for routing it; ``close`` releases any resources (open
    file handles) and must be safe to call more than once.
    """

    def write(self, record: ContractModel) -> None: ...

    def close(self) -> None: ...


class EpisodeRecorder:
    """Write one episode's records to the CONTRACTS section 15 file layout.

    ``directory`` is the episode directory ``episodes/<episode_id>/``; it is
    created (with parents) on construction. Streams are opened lazily on their
    first record, so only the files an episode actually produces appear on disk.
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, JsonlWriter] = {}
        self._closed = False
        for name in _EAGER_JSONL:
            self._writer_for(name)

    def _writer_for(self, record_name: str) -> JsonlWriter:
        writer = self._writers.get(record_name)
        if writer is None:
            writer = JsonlWriter(self._dir / EPISODE_JSONL_FILES[record_name])
            self._writers[record_name] = writer
        return writer

    def write(self, record: ContractModel) -> None:
        if self._closed:
            raise RuntimeError("EpisodeRecorder is closed")
        name = type(record).__name__
        if name in EPISODE_JSON_FILES:
            write_json(self._dir / EPISODE_JSON_FILES[name], record)
            return
        if name in EPISODE_JSONL_FILES:
            self._writer_for(name).write(record)
            return
        raise ValueError(f"EpisodeRecorder cannot route a {name} record")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

    def __enter__(self) -> "EpisodeRecorder":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class MemoryRecorder:
    """Keep every record in memory, grouped by type and in write order.

    ``records`` maps a record class name (for example ``"AgentObservation"``) to
    the list of records of that type in write order; ``log`` is the single
    ordered list of every record as written, which lets a caller inspect the
    cross-type phase ordering. :meth:`by_type` reads one group back.
    """

    def __init__(self) -> None:
        self.records: dict[str, list[ContractModel]] = {}
        self.log: list[ContractModel] = []

    def write(self, record: ContractModel) -> None:
        self.records.setdefault(type(record).__name__, []).append(record)
        self.log.append(record)

    def by_type(self, record_name: str) -> list[ContractModel]:
        """Return the records of one type in write order (empty if none)."""

        return list(self.records.get(record_name, []))

    def close(self) -> None:  # noqa: D401 - nothing to release
        """No-op: memory needs no cleanup."""


class NullRecorder:
    """Drop every record. Used for training rollouts that write nothing."""

    def write(self, record: ContractModel) -> None:  # noqa: D401 - intentional no-op
        """Ignore the record."""

    def close(self) -> None:  # noqa: D401 - intentional no-op
        """Nothing to release."""
