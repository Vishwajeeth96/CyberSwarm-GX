"""JSON-line ``LogRecord`` emission and the single wall-clock read (stage S1).

This is the only module in the system that reads the wall clock (CONTRACTS C9):
:func:`utc_now_iso` formats the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ`` and
is the default timestamp source for every emitted
:class:`~cyberswarm.contracts.manifest.LogRecord`. Every other timestamp in the
system is a simulation step, and logs are excluded from determinism comparisons.

A :class:`RunLogger` is built on the standard-library :mod:`logging` module with
a formatter that serialises each line as one canonical ``LogRecord`` JSON object
(CONTRACTS 11.5). ``import logging`` below is an absolute import and resolves to
the standard library, not to this submodule.
"""

from __future__ import annotations

import logging as _stdlib_logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, TextIO

from cyberswarm.contracts.common import ErrorCode
from cyberswarm.contracts.io import dumps_canonical
from cyberswarm.contracts.manifest import Component, LogLevel, LogRecord

#: ``strftime`` pattern for the only wall-clock value in the system.
_TS_FORMAT: str = "%Y-%m-%dT%H:%M:%SZ"

_LEVEL_TO_STDLIB: dict[LogLevel, int] = {
    LogLevel.debug: _stdlib_logging.DEBUG,
    LogLevel.info: _stdlib_logging.INFO,
    LogLevel.warning: _stdlib_logging.WARNING,
    LogLevel.error: _stdlib_logging.ERROR,
}
_STDLIB_TO_LEVEL: dict[int, LogLevel] = {
    value: level for level, value in _LEVEL_TO_STDLIB.items()
}

#: A callable returning the ``ts`` string for a log line.
Clock = Callable[[], str]


def utc_now_iso() -> str:
    """Return the current UTC wall-clock time as ``YYYY-MM-DDTHH:MM:SSZ``.

    This is the single wall-clock read in the whole system (CONTRACTS C9).
    """

    return datetime.now(timezone.utc).strftime(_TS_FORMAT)


class _LogRecordFormatter(_stdlib_logging.Formatter):
    """Serialise a stdlib log record as one canonical ``LogRecord`` JSON line.

    :class:`RunLogger` attaches the already-built contract record to each stdlib
    record (attribute ``contract_record``) so the timestamp is read exactly once
    per line.
    """

    def __init__(self, clock: Clock) -> None:
        super().__init__()
        self._clock = clock

    def format(self, record: _stdlib_logging.LogRecord) -> str:
        contract = getattr(record, "contract_record", None)
        if contract is None:  # pragma: no cover - defensive; RunLogger always sets it
            contract = LogRecord(
                ts=self._clock(),
                level=_STDLIB_TO_LEVEL.get(record.levelno, LogLevel.info),
                component=Component.runner,
                message=record.getMessage(),
            )
        return dumps_canonical(contract)


class RunLogger:
    """A component logger that writes canonical ``LogRecord`` JSON lines.

    Each call stamps the line with ``clock()`` (the module's UTC reader by
    default) and writes it to ``sink`` when given, otherwise to ``stream``
    (defaulting to :data:`sys.stderr`). The emitting methods also return the
    built :class:`~cyberswarm.contracts.manifest.LogRecord` so callers can assert
    on or forward it. A message must be a ``ShortText``; an invalid message
    raises :class:`~cyberswarm.contracts.common.ContractError`.
    """

    def __init__(
        self,
        component: Component | str,
        *,
        run_id: str | None = None,
        sink: Path | str | None = None,
        stream: TextIO | None = None,
        level: LogLevel = LogLevel.info,
        clock: Clock = utc_now_iso,
    ) -> None:
        self.component: Component = Component(component)
        self.run_id = run_id
        self._clock = clock
        # A standalone Logger (not the global registry) so repeated construction
        # never accumulates duplicate handlers or leaks into other loggers.
        self._logger = _stdlib_logging.Logger(
            name=f"cyberswarm.{self.component.value}",
            level=_LEVEL_TO_STDLIB[level],
        )
        self._logger.propagate = False

        handler: _stdlib_logging.Handler
        if sink is not None:
            sink = Path(sink)
            sink.parent.mkdir(parents=True, exist_ok=True)
            handler = _stdlib_logging.FileHandler(sink, encoding="utf-8")
        else:
            handler = _stdlib_logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(_LogRecordFormatter(self._clock))
        self._logger.addHandler(handler)

    def _emit(
        self,
        level: LogLevel,
        message: str,
        *,
        episode_id: str | None = None,
        step: int | None = None,
        code: ErrorCode | str | None = None,
    ) -> LogRecord:
        record = LogRecord(
            ts=self._clock(),
            level=level,
            component=self.component,
            run_id=self.run_id,
            episode_id=episode_id,
            step=step,
            code=code,
            message=message,
        )
        self._logger.log(
            _LEVEL_TO_STDLIB[level],
            message,
            extra={"contract_record": record},
        )
        return record

    def log(
        self,
        level: LogLevel | str,
        message: str,
        *,
        episode_id: str | None = None,
        step: int | None = None,
        code: ErrorCode | str | None = None,
    ) -> LogRecord:
        return self._emit(
            LogLevel(level), message, episode_id=episode_id, step=step, code=code
        )

    def debug(self, message: str, **fields: object) -> LogRecord:
        return self._emit(LogLevel.debug, message, **fields)  # type: ignore[arg-type]

    def info(self, message: str, **fields: object) -> LogRecord:
        return self._emit(LogLevel.info, message, **fields)  # type: ignore[arg-type]

    def warning(self, message: str, **fields: object) -> LogRecord:
        return self._emit(LogLevel.warning, message, **fields)  # type: ignore[arg-type]

    def error(self, message: str, **fields: object) -> LogRecord:
        return self._emit(LogLevel.error, message, **fields)  # type: ignore[arg-type]

    def close(self) -> None:
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)


def get_logger(
    component: Component | str,
    *,
    run_id: str | None = None,
    sink: Path | str | None = None,
    stream: TextIO | None = None,
    level: LogLevel = LogLevel.info,
    clock: Clock = utc_now_iso,
) -> RunLogger:
    """Return a :class:`RunLogger` for ``component`` writing ``LogRecord`` lines."""

    return RunLogger(
        component,
        run_id=run_id,
        sink=sink,
        stream=stream,
        level=level,
        clock=clock,
    )
