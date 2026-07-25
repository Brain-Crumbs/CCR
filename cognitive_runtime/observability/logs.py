"""Logging configuration for CCR.

The training modules already log (``ccr.training.nursery``,
``ccr.training.cortex``), but nothing in the project ever configured the
logging package -- so the root logger's default WARNING level swallowed
every one of those ``log.info`` calls and a 30-epoch cortex run printed
nothing between "starting" and "done".  This module is the missing half:
one idempotent :func:`configure_logging` call, made by the CLI for every
subcommand and by the notebook helper, that attaches a formatter showing
elapsed time since configuration:

    [+00:12.4] INFO  nursery      recorded 24 train + 12 holdout sessions
    [+02:41.9] INFO  cortex         epoch 7/30  total=0.0412  pixel=0.0333

``--log-format json`` swaps that for one JSON object per line (CI/log
shipping), and ``--log-file`` tees everything to a file at DEBUG regardless
of the console level, so a console kept readable at INFO still leaves a
full-detail log on disk.

Records at WARNING and above are also forwarded into the active run trace
(:mod:`cognitive_runtime.observability.trace`), so a warning that scrolled
past in the console is still in ``trace.jsonl`` afterwards.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, Iterable, Optional

from cognitive_runtime.observability.trace import current_trace

#: Logger namespaces CCR code writes to.  Configuring these rather than the
#: root logger keeps a notebook's other libraries at their own levels.
LOG_NAMESPACES = (
    "ccr", "cognitive_runtime", "brain", "sleep", "motor", "development",
)

LEVELS = ("debug", "info", "warning", "error", "critical")

_HANDLER_MARK = "_ccr_handler"

_LEVEL_COLORS = {
    "DEBUG": "\033[38;5;244m",
    "INFO": "\033[38;5;39m",
    "WARNING": "\033[38;5;214m",
    "ERROR": "\033[38;5;203m",
    "CRITICAL": "\033[1;38;5;203m",
}
_RESET = "\033[0m"
_DIM = "\033[38;5;244m"


def _short_name(name: str) -> str:
    """``ccr.training.nursery`` -> ``nursery``; keeps the column narrow."""
    return name.rsplit(".", 1)[-1]


class ElapsedFormatter(logging.Formatter):
    """``[+MM:SS.s] LEVEL  logger  message`` -- elapsed time is what you want
    when watching a training run, not wall-clock timestamps."""

    def __init__(self, *, color: bool = False, start: Optional[float] = None) -> None:
        super().__init__()
        self.color = color
        self.start = start if start is not None else time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed = max(0.0, record.created - self.start)
        stamp = f"+{int(elapsed // 60):02d}:{elapsed % 60:04.1f}"
        level = record.levelname
        name = _short_name(record.name)
        message = record.getMessage()
        if record.exc_info:
            message = message + "\n" + self.formatException(record.exc_info)
        if self.color:
            tint = _LEVEL_COLORS.get(level, "")
            return (f"{_DIM}[{stamp}]{_RESET} {tint}{level:<7}{_RESET} "
                    f"{_DIM}{name:<12}{_RESET} {message}")
        return f"[{stamp}] {level:<7} {name:<12} {message}"


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log shipping and CI parsing."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: Dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace = current_trace()
        if trace is not None:
            payload["run_id"] = trace.run_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=repr)


class TraceForwardHandler(logging.Handler):
    """Forward WARNING+ records into the active run trace, so the durable
    record of a run includes the problems it hit, not just its metrics."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        trace = current_trace()
        if trace is None:
            return
        try:
            trace.event(
                "log",
                level=record.levelname,
                logger=record.name,
                message=record.getMessage(),
            )
        except Exception:  # noqa: BLE001 - logging must never break a run
            pass


def _in_notebook() -> bool:
    return "ipykernel" in sys.modules


def _want_color(stream: Any, color: Optional[bool]) -> bool:
    if color is not None:
        return color
    if os.environ.get("NO_COLOR"):
        return False
    if _in_notebook():
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _resolve_level(level: Optional[str]) -> int:
    name = (level or os.environ.get("CCR_LOG_LEVEL") or "info").strip().lower()
    return getattr(logging, name.upper(), logging.INFO)


def _clear_ccr_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001
                pass


def configure_logging(
    level: Optional[str] = None,
    *,
    log_file: Optional[str] = None,
    log_format: str = "human",
    color: Optional[bool] = None,
    stream: Any = None,
    namespaces: Iterable[str] = LOG_NAMESPACES,
    forward_to_trace: bool = True,
) -> None:
    """Configure CCR's loggers.  Idempotent: calling it again replaces the
    handlers it installed rather than stacking duplicates.

    ``level`` defaults to ``$CCR_LOG_LEVEL`` and then ``info``.  ``log_file``
    always records at DEBUG so a quiet console still leaves full detail on
    disk.
    """
    resolved = _resolve_level(level)
    if stream is None:
        # Notebooks render stderr as a red error box; stdout reads as output.
        stream = sys.stdout if _in_notebook() else sys.stderr
    start = time.time()

    if log_format == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = ElapsedFormatter(color=_want_color(stream, color), start=start)

    console = logging.StreamHandler(stream)
    console.setFormatter(formatter)
    console.setLevel(resolved)
    setattr(console, _HANDLER_MARK, True)

    handlers = [console]
    if log_file:
        directory = os.path.dirname(os.path.abspath(log_file))
        if directory:
            os.makedirs(directory, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            JsonFormatter() if log_format == "json"
            else ElapsedFormatter(color=False, start=start)
        )
        file_handler.setLevel(logging.DEBUG)
        setattr(file_handler, _HANDLER_MARK, True)
        handlers.append(file_handler)
    if forward_to_trace:
        forwarder = TraceForwardHandler()
        setattr(forwarder, _HANDLER_MARK, True)
        handlers.append(forwarder)

    effective = min([resolved] + [h.level for h in handlers if h.level])
    for namespace in namespaces:
        logger = logging.getLogger(namespace)
        _clear_ccr_handlers(logger)
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(effective)
        # These are top-level namespaces; propagating to root would duplicate
        # output under any root handler a notebook or host app installed.
        logger.propagate = False


def configure_notebook_logging(level: str = "info", **kwargs: Any) -> None:
    """Convenience for ``notebooks/build_and_diagnose_organism.ipynb``: log
    to stdout so training progress renders as cell output."""
    kwargs.setdefault("stream", sys.stdout)
    configure_logging(level, **kwargs)
