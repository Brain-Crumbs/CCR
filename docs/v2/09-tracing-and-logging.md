# Tracing & Logging

*How to see what a V2 crafter training run is doing while it does it, and
what it did after it's over.*

## The problem this solves

A V2 crafter training run is a chain of expensive phases:

```
record N scenarios x M seeds  →  quality gate  →  build dataset
    →  train the cortex for E epochs  →  evaluate per scenario
    →  zero-shot evaluate  →  probes
```

Before this existed:

- **Nothing configured Python's `logging`.** `cognitive_runtime.training.nursery`
  and `.action_world_model` both create loggers and call `log.info` at every
  interesting step — and the root logger's default WARNING level discarded
  every one of them. A 30-epoch `ccr nursery joint` printed a banner, went
  silent for forty minutes, then printed a report.
- **The only durable record was `--report`**, written after the last phase.
  A run that died at epoch 25/30, or failed the quality gate, or was killed
  by the OOM reaper, left nothing behind: no config, no partial loss curve,
  no timings.
- **Nothing tied a number to what produced it** — which commit, which
  torch version, which seeds, which flags.

## The two halves

| | what it is | who reads it |
|---|---|---|
| **Logging** | Human-readable progress on the console, elapsed-time stamped | you, while the run is going |
| **Tracing** | Append-only JSONL + a manifest, flushed line by line | you (or CI, or a script) afterwards |

Both come from the same instrumentation calls, so there's one thing to
maintain, not two.

## Using it: the console

Every `ccr` subcommand accepts the flags, before or after the subcommand:

```bash
ccr nursery joint --record-dir runs/Pixel --epochs 30 --log-level info
```

```
[+00:00.0] INFO    trace        run nursery-joint-20260724T1830-9f2ac1  trace=runs/traces/nursery-joint-20260724T1830-9f2ac1/trace.jsonl
[+00:00.1] INFO    nursery      === nursery joint: train=['walk_forward', ...] holdout=['approach_entity'] world=crafter ===
[+00:00.1] INFO    nursery      recording nursery-walk_forward-train-0  seed=0  ticks=400
[+00:04.7] INFO    trace        <- nursery.record.episode  4.61s
...
[+02:58.2] INFO    nursery      quality gate passed
[+02:58.4] INFO    nursery      horizons (ticks): [1, 10, 100] -> frames: [1, 10, 100]  (1.00 ticks/frame)
[+03:01.0] INFO    cortex       training cortex (autoregressive)  episodes=24  epochs=30  horizons=[1, 10, 100]  backbone=gru
[+03:12.4] INFO    cortex         epoch 1/30  total=0.4412  pixel=0.3901  latent=0.0511  (11.4s/epoch, eta 5m30s)
[+03:23.6] INFO    cortex         epoch 2/30  total=0.3120  pixel=0.2740  latent=0.0380  (11.2s/epoch, eta 5m14s)
```

| flag | effect |
|---|---|
| `--log-level debug\|info\|warning\|error` | console verbosity (default `info`, or `$CCR_LOG_LEVEL`). `debug` adds every span boundary and metric |
| `--log-file path.log` | tee **everything** at DEBUG to a file, whatever the console level is |
| `--log-format json` | one JSON object per line instead of the human format (CI, log shipping) |

In a notebook:

```python
from cognitive_runtime.observability import configure_notebook_logging
configure_notebook_logging("info")   # logs to stdout, so they render as cell output
```

## Using it: the trace

Every command writes a trace unless told not to:

```
runs/traces/
  latest -> nursery-joint-20260724T1830-9f2ac1
  nursery-joint-20260724T1830-9f2ac1/
    manifest.json      identity, config, status, metric summaries
    trace.jsonl        one event per line, flushed as it happens
```

| flag | effect |
|---|---|
| `--trace-dir DIR` | where traces go (default `runs/traces`, or `$CCR_TRACE_DIR`) |
| `--run-id NAME` | name the run instead of generating an id |
| `--no-trace` | write no trace at all (`CCR_TRACE=0` does the same) |

Read one back:

```bash
ccr trace list                  # every run, oldest first
ccr trace show                  # the latest run
ccr trace show nursery-joint-20260724T1830-9f2ac1 --tail 20
```

```
run       nursery-joint-20260724T1830-9f2ac1  (nursery.joint)
status    ok   duration 41m12.4s   started 2026-07-24T18:30:02
commit    a06c8060d1f4 (claude/v2/main)
packages  cognitive-runtime=0.1.0  crafter=1.8.3  numpy=1.26.4  torch=2.3.0
device    cuda_available=False  threads=8

config
  backbone                 gru
  epochs                   30
  horizons                 [1, 10, 100]
  ...

phases
      178.4s  nursery.record   scenarios=6 episodes=36
       29.7s    nursery.record.scenario   scenario=walk_forward
        4.6s      nursery.record.episode   session=nursery-walk_forward-train-0 seed=0
        1.2s  nursery.quality_gate   issues=0
        3.1s  nursery.dataset   transitions=9600 ticks_per_frame=1.0
     2210.8s  nursery.train   epochs=30 backbone=gru final_total_loss=0.0731
       94.2s  nursery.evaluate.in_distribution   scenarios=5
       18.9s  nursery.probes

counters
  episodes_recorded                36

metric                               n        first         last          min          max
train/total_loss                    30       0.4412       0.0731       0.0731       0.4412
eval/zero_shot/approach_entity/t+1/beats_copy_last
                                     1            1            1            1            1
```

## Instrumenting new code

Three primitives, all no-ops when no run is active — so library code calls
them unconditionally, and importing an instrumented module costs nothing:

```python
from cognitive_runtime.observability import span, trace_event, trace_metrics

with span("cortex.train", epochs=cfg.epochs) as phase:   # timed, nestable
    trace_event("dataset", transitions=len(dataset))     # a point in time
    for epoch in range(cfg.epochs):
        ...
        trace_metrics(step=epoch, **{"train/total_loss": loss})  # a series
    phase.set(final_total_loss=loss)                     # known only at the end
```

Rules of thumb:

- **`span`** for anything that takes measurable time. A span records its
  duration and outcome even when the body raises, so a failed run's trace
  still shows how far it got.
- **`trace_event`** for facts you'd want in a post-mortem: the quality
  gate's verdict, a probe result, the plan the run is about to execute.
- **`trace_metrics`** for anything that varies over steps. This is what
  makes a crashed run's loss curve survive.
- Anything logged at **WARNING or above lands in the trace automatically**,
  so `log.warning("frozen rollout detected")` needs no extra call.

Driving a run from Python (a notebook, a script) rather than the CLI:

```python
from cognitive_runtime.observability import configure_logging, start_run

configure_logging("info")
with start_run("nursery.joint", config={"epochs": 30}):
    model, report = run_nursery_joint("runs/Pixel", ...)
```

## Trace format (`ccr-trace-v1`)

`trace.jsonl`, one JSON object per line:

```json
{"t": 12.44, "wall": 1753380614.2, "kind": "span.start", "name": "nursery.train", "span": "s7", "parent": "s1", "data": {"epochs": 30}}
{"t": 23.90, "wall": 1753380625.7, "kind": "metrics", "name": "metrics", "span": "s7", "step": 1, "data": {"train/total_loss": 0.4412}}
{"t": 2234.7, "wall": 1753382836.5, "kind": "span.end", "name": "nursery.train", "span": "s7", "dur_s": 2210.8, "ok": true, "data": {...}}
```

| field | meaning |
|---|---|
| `t` | seconds since run start |
| `kind` | `run.start` / `run.end` / `span.start` / `span.end` / `event` / `metrics` |
| `span`, `parent` | span ids, for reconstructing the phase tree |
| `dur_s`, `ok` | on `span.end` only |
| `step` | on `metrics` only (epoch number, usually) |

`manifest.json` carries the run's identity (`run_id`, git commit + dirty
flag, package versions, device, argv, cwd, hostname), its `config`, its
final `status` (`running` / `ok` / `error` / `interrupted`), `counters`, and
a `first/last/min/max/count` summary per metric. It's rewritten at the end
of the run, so device facts (which need torch imported) are present even
though torch usually isn't loaded when the run starts.

Both files are plain text; a truncated final line (a killed run) is skipped
by the reader rather than treated as corruption.

**A run killed outright** (OOM, SIGKILL, a lost session) never writes
`run.end` and never rewrites its manifest, so its status stays `running` —
`ccr trace show` reports that as `running / killed (no run.end)`, marks the
phase that was in flight as still open (`..`), and rebuilds the metric table
from the flushed events rather than the manifest's missing copy. That's the
case the whole design is for: you still get the config, the phase you died
in, and the loss curve up to the last epoch that completed.

## Design notes

- **No new dependency.** Stdlib `logging` + JSONL. The project deliberately
  installs from a four-package core; a tracing solution that needed
  TensorBoard or W&B to read a loss curve would be the heaviest thing in
  the tree. `trace.jsonl` is one `json.loads` per line away from a
  DataFrame, and nothing stops a future sink from forwarding the same
  events onward.
- **Flush per line.** Throughput doesn't matter at these event rates; being
  readable mid-run and complete after a `kill -9` does.
- **Tracing must never break a run.** Unserializable values fall back to
  `repr`, a closed or full output file disables the trace instead of
  raising, and the environment capture swallows its own errors.
- **Loggers don't propagate to root.** `configure_logging` attaches
  handlers to the `ccr`, `cognitive_runtime`, `brain`, `sleep`, `motor` and
  `development` namespaces and turns off propagation, so a host app or
  notebook with its own root handler doesn't get double output.
