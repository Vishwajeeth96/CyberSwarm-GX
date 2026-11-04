"""Unit tests for :mod:`cyberswarm.common.logging`.

Covers canonical JSON-line ``LogRecord`` emission to a file sink and a stream,
the deterministic (injected) clock, and the UTC timestamp format of the single
wall-clock reader.
"""

from __future__ import annotations

import io
import re

from cyberswarm.common.logging import get_logger, utc_now_iso
from cyberswarm.contracts.common import ErrorCode
from cyberswarm.contracts.io import dumps_canonical, loads_record
from cyberswarm.contracts.manifest import Component, LogLevel, LogRecord

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _fixed_clock() -> str:
    return "2026-10-30T20:00:00Z"


def test_get_logger_writes_canonical_logrecord_lines(tmp_path) -> None:
    sink = tmp_path / "logs" / "run.log.jsonl"
    logger = get_logger(
        "runner", run_id="run-common-test", sink=sink, clock=_fixed_clock
    )

    first = logger.info("Run started")
    second = logger.warning("Careful now", step=5)
    third = logger.error("It failed", code=ErrorCode.SIMULATOR_FAILURE)
    logger.close()

    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    parsed = [loads_record("LogRecord", line) for line in lines]
    assert parsed == [first, second, third]
    # Every written line is in canonical form.
    assert lines == [dumps_canonical(record) for record in (first, second, third)]


def test_logrecord_fields_are_populated(tmp_path) -> None:
    sink = tmp_path / "run.log.jsonl"
    logger = get_logger(
        Component.runner, run_id="run-common-test", sink=sink, clock=_fixed_clock
    )
    logger.info("Run started")
    logger.warning("Careful now", step=5)
    logger.error("It failed", code=ErrorCode.SIMULATOR_FAILURE)
    logger.close()

    parsed = [
        loads_record("LogRecord", line)
        for line in sink.read_text(encoding="utf-8").splitlines()
    ]
    assert isinstance(parsed[0], LogRecord)
    assert parsed[0].component == Component.runner
    assert parsed[0].run_id == "run-common-test"
    assert parsed[0].ts == "2026-10-30T20:00:00Z"
    assert parsed[0].level == LogLevel.info
    assert parsed[1].level == LogLevel.warning
    assert parsed[1].step == 5
    assert parsed[2].level == LogLevel.error
    assert parsed[2].code == ErrorCode.SIMULATOR_FAILURE


def test_get_logger_writes_to_a_stream() -> None:
    buffer = io.StringIO()
    logger = get_logger("mock", stream=buffer, clock=_fixed_clock)
    record = logger.info("hello there")
    logger.close()

    line = buffer.getvalue().strip()
    assert loads_record("LogRecord", line) == record


def test_logger_default_clock_produces_utc_timestamp(tmp_path) -> None:
    sink = tmp_path / "run.log.jsonl"
    logger = get_logger(Component.config, sink=sink)
    record = logger.info("configured")
    logger.close()
    assert _TS_RE.match(record.ts)


def test_utc_now_iso_matches_contract_format() -> None:
    assert _TS_RE.match(utc_now_iso())


def test_logger_below_threshold_level_is_filtered(tmp_path) -> None:
    sink = tmp_path / "run.log.jsonl"
    logger = get_logger("runner", sink=sink, level=LogLevel.warning, clock=_fixed_clock)
    logger.info("suppressed below threshold")
    logger.warning("kept")
    logger.close()

    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert loads_record("LogRecord", lines[0]).message == "kept"
