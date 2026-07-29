# How to add or update tracing

Use CCR's built-in observability helpers to record durable, crash-safe
information about a run. They write an append-only `trace.jsonl` file and a
`manifest.json` summary under `runs/traces/<run-id>/`; they do not require an
external telemetry service.

This guide is for contributors instrumenting Python code. For operating and
reading traces from a training run, see
[Tracing & Logging](../v2/09-tracing-and-logging.md).

## Before you add instrumentation

1. Identify the command or Python entry point that owns the run. CLI commands
   that perform work are already wrapped by `cognitive_runtime.cli`; code
   called from those commands should normally add only instrumentation events,
   not another run.
2. Decide what question the trace should answer: how long a phase took, which
   decision was made, how a numeric value changed, or how often something
   occurred.
3. Keep names stable and hierarchical. Prefer names such as
   `nursery.record.episode` and `cortex.train.done` over generic names such as
   `recording` or `done`. Trace names are part of the diagnostic interface.

The library-level helpers are safe to call without an active run. They become
no-ops, so instrumentation can live in reusable library code without an
`if current_trace()` guard.

## Choose the right primitive

| Need | Use | Example |
| --- | --- | --- |
| Time a phase, including failure | `span` | recording one episode or training epochs |
| Record a discrete fact or decision | `trace_event` | a cache hit or quality-gate failure |
| Record one or more numeric series values | `trace_metric` or `trace_metrics` | loss or accuracy each epoch |
| Accumulate a run-wide total | `trace_counter` | episodes recorded or batches skipped |

Import from the public package rather than an implementation module:

```python
from cognitive_runtime.observability import (
    span,
    trace_counter,
    trace_event,
    trace_metrics,
)
```

## Instrument work with a span

Wrap work that has a meaningful duration in `span`. Put input context in its
opening fields and add values known only after the work completes with
`phase.set()`.

```python
with span("cortex.train", epochs=config.epochs, backbone=config.backbone) as phase:
    stats = train(model, dataset, config)
    phase.set(final_total_loss=stats["total_loss"])
```

Spans nest automatically. A successful span gets an `ok: true` end record;
if the body raises, it gets `ok: false`, duration, and error details before the
exception is re-raised. Do not catch an exception solely to trace it.

Use a short, bounded span for repeated units only when that detail is useful:

```python
with span("nursery.record.episode", session=session_id, seed=seed) as episode:
    frames = record_episode(...)
    episode.set(frames=len(frames))
```

Avoid creating a span for every cheap inner-loop operation. It makes the trace
large and less useful; prefer a metric for repeated progress.

## Record events, metrics, and counters

Use events for a fact that happened once, with enough context to diagnose it:

```python
trace_event(
    "nursery.record.cache_hit",
    scenario=scenario.name,
    seed=seed,
    sessions=len(cached_sessions),
)
```

Metric values must be numeric. Batch related values at the same step so they
are written as one trace record and can be compared directly:

```python
for epoch in range(config.epochs):
    total_loss, pixel_loss = train_epoch(...)
    trace_metrics(
        step=epoch + 1,
        **{
            "train/total_loss": total_loss,
            "train/pixel_loss": pixel_loss,
        },
    )
```

Use slash-separated metric names to group related series. A non-numeric metric
value is ignored, so record labels and structured detail with `trace_event`
instead.

Counters accumulate in the manifest and are useful when a time series adds no
value:

```python
trace_counter("episodes_recorded")
trace_counter("batches_skipped", amount=skipped_batches)
```

Values attached to events, spans, or run configuration are converted to JSON
where possible. Dataclasses, mappings, sequences, scalar-like tensors, and
paths are handled; other values fall back to `repr`. Keep fields small and do
not attach raw frames, large tensors, credentials, or other sensitive data.

## Add a trace lifecycle for a new Python entry point

Library code should not start a trace. A standalone script or notebook that
owns an entire run should configure logging and open one run around its work:

```python
from cognitive_runtime.observability import configure_logging, start_run

configure_logging("info")
with start_run(
    "my_pipeline.run",
    config={"epochs": config.epochs, "seed": config.seed},
):
    run_pipeline(config)
```

`start_run` captures the run configuration and environment, including Git and
package information. The default destination is `runs/traces`; callers may
pass `trace_dir` and `run_id` when needed. Always use the context manager so a
normal completion, error, or `KeyboardInterrupt` is recorded correctly.

## Logging and warnings

Use the module logger for human-readable progress as usual:

```python
import logging

log = logging.getLogger("ccr.training.example")
log.info("training %s for %d epochs", model_name, config.epochs)
```

When CCR logging has been configured, warning-level and more severe messages
are copied into the active trace automatically. Do not duplicate a warning as
a `trace_event` unless it needs additional structured context.

## Verify the change

Add or update focused tests in `tests/test_observability.py` when changing the
tracing implementation. For a call-site instrumentation change, test the
feature at its natural layer and assert the emitted event, metric, span result,
or counter as appropriate.

Run the observability suite:

```bash
pytest tests/test_observability.py
```

Then run the smallest relevant command or test and inspect its trace:

```bash
ccr trace list
ccr trace show
```

Confirm that the trace tells the intended story: phases are nested correctly,
metrics have a monotonic and meaningful `step`, event fields are sufficient for
diagnosis, and no noisy inner-loop records or sensitive payloads were added.

## Trace quality checklist

- Does the entry point create exactly one trace for one unit of work?
- Are spans reserved for meaningful durations and named consistently with
  nearby spans?
- Are recurring numeric measurements emitted as metrics with a clear step?
- Are one-off decisions and failures emitted as structured events?
- Are run-level totals counters rather than repeated events?
- Can a failed or interrupted run still explain what it was doing?
- Does the instrumentation stay non-fatal and avoid large or sensitive data?
