"""Observability for CCR runs: structured logging + run tracing.

Two halves of one instrumentation pass:

- :mod:`~cognitive_runtime.observability.logs` -- human-readable console
  output (elapsed-time formatter, optional JSON, optional log file).  Until
  this existed nothing configured the ``logging`` package, so every
  ``log.info`` in the training modules was silently discarded.
- :mod:`~cognitive_runtime.observability.trace` -- a crash-safe JSONL record
  of one run: config, git commit, package versions, phase timings, metric
  series, warnings and the final status.

Typical library-side use (no-ops when no run is active, so instrumented
code stays importable and cheap everywhere):

    from cognitive_runtime.observability import span, trace_event, trace_metrics

    with span("cortex.train", epochs=cfg.epochs):
        trace_metrics(step=epoch, total_loss=0.31, pixel_loss=0.02)

Typical entrypoint-side use:

    from cognitive_runtime.observability import configure_logging, start_run

    configure_logging("info")
    with start_run("nursery.joint", config={"epochs": 30}):
        ...
"""

from cognitive_runtime.observability.logs import (
    LOG_NAMESPACES,
    configure_logging,
    configure_notebook_logging,
)
from cognitive_runtime.observability.read import (
    format_run_list,
    format_trace_summary,
    list_runs,
    load_trace,
    resolve_run_dir,
)
from cognitive_runtime.observability.trace import (
    DEFAULT_TRACE_DIR,
    TRACE_SCHEMA,
    RunTrace,
    current_trace,
    default_trace_dir,
    span,
    start_run,
    trace_counter,
    trace_event,
    trace_metric,
    trace_metrics,
)

__all__ = [
    "DEFAULT_TRACE_DIR",
    "LOG_NAMESPACES",
    "RunTrace",
    "TRACE_SCHEMA",
    "configure_logging",
    "configure_notebook_logging",
    "current_trace",
    "default_trace_dir",
    "format_run_list",
    "format_trace_summary",
    "list_runs",
    "load_trace",
    "resolve_run_dir",
    "span",
    "start_run",
    "trace_counter",
    "trace_event",
    "trace_metric",
    "trace_metrics",
]
