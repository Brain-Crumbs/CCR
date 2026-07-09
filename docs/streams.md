# Sensory & Motor Streams

The runtime consumes **time-indexed sensory streams**, not static
observations: instead of asking "what is the current observation?", it asks
**"what streams have arrived since the last cognitive tick?"**. This
document covers the stream substrate in `cognitive_runtime/core/streams/` —
the primitives the loop, recorder, replay and training are all built on
(see the migration status at the end of this document).

## Architecture

```
Program
  ↓  publishes
Sensory Streams (SensoryStreamBus)
  ↓
Stream Encoders (StreamEncoderRegistry)
  ↓
Shared Latent State (TemporalBuffer + fusion)
  ↓
Memory / World Model / Policy
  ↓  emits
Motor Streams (MotorStreamBus)
  ↓  consumed by
Program
```

Design rules:

1. Programs publish streams. The runtime subscribes.
2. Encoders convert stream events into latent tokens. The policy emits motor
   streams. Programs consume motor streams.
3. The runtime never asks for "the current observation" — it collects the
   window of events since the last cognitive tick (`TickSynchronizer`).
4. Different senses update at different rates (see cadence guidance below).

## Stream taxonomy

Every event lives on a named stream. Stream ids are **lowercase dotted
paths** whose first segment is, by convention, the modality
(`body.health`, `vision.frame.grid`, `event.damage_taken`). Modalities are
generic: Minecraft health, a Linux battery and robot joint stress are all
`body.*` streams; Minecraft frames, desktop pixels and robot cameras are all
`vision.*` streams. The brain consumes modalities, never environment fields.

| Modality | What flows on it | Example streams |
|---|---|---|
| `body` | Internal/self state | `body.health`, `body.hunger`, `body.oxygen` |
| `vision` | Frames, pixels, grids | `vision.frame.grid`, `vision.frame.pixels`, `vision.entities` |
| `spatial` | Position, orientation, geometry | `spatial.position`, `spatial.rotation` |
| `audio` | Sound events/levels | `audio.ambient` |
| `event` | Discrete world happenings | `event.damage_taken`, `event.item_collected` |
| `reward` | Reward components | `reward.scalar` |
| `language` | Text in/out of the world | `language.chat` |
| `input` | Raw human input (demos) | `input.keypress` |
| `world` | Global world state | `world.time`, `world.nearby_blocks` |
| `motor` | Actions, the other direction | `motor.command` |

## The primitives

| Type | Module | Role |
|---|---|---|
| `StreamEvent` | `events.py` | One time-indexed sample: id, modality, simulated timestamp, per-stream sequence number, JSON payload, confidence, source. Content-hashable (`.hash()`) — the replay-verification unit. |
| `StreamSpec` | `events.py` | A Program's advertisement of one stream it publishes (description, nominal rate, informal payload schema, plus optional encoder metadata: `range`, `legend`, `categories`, `neutral`). |
| `SensoryStreamBus` / `MotorStreamBus` | `bus.py` | Deterministic in-process pub/sub. `publish()` assigns per-stream monotonic sequence numbers; `drain()` returns pending events in deterministic order; `subscribe(pattern)` gives a glob-filtered view (`"body.*"`, `"*"`). Same mechanism both directions. |
| `TemporalBuffer` | `temporal_buffer.py` | Bounded per-stream history with per-modality capacities (vision short, events long). `latest`, `window(n)`, `events_since(t)`. |
| `TickSynchronizer` | `synchronizer.py` | Defines cognitive tick boundaries; `collect(bus)` drains into a `TickWindow` (events grouped by stream). Supports a `program_ticks_per_cognitive_tick` ratio and tracks per-stream arrival counts and silences for runtime health. |
| `StreamEncoderRegistry` | `encoder_registry.py` | Maps stream patterns to `StreamEncoder`s producing `LatentToken`s. |
| Modality encoders | `encoders/` | Fixed-width, spec-driven, environment-agnostic: `ScalarEncoder` (body/reward), `SpatialEncoder`, `GridVisionEncoder`, `EntityEncoder`, `EventEncoder`, `CategoryEncoder`. |
| `TemporalFusion` | `fusion.py` | Assembles per-stream tokens + recent history into one fixed-width `LatentState` (flat vector + named per-stream slices) with a deterministic, versioned `layout_hash`. |
| `StreamDeclaration` / `StreamRegistry` | `registry.py` | The per-stream schema registry (issue #21): one declaration per stream binding its encoder (or an explicit "raw"/no-fusion-slot stub), `trainable` flag, `checkpoint_key`, and `train_eval_behavior`, alongside the shape/rate already on its `StreamSpec`. `DEFAULT_STREAM_REGISTRY` holds the generic modality declarations (plus reserved ids for audio, keyboard and mouse/look inputs not yet published by any Program); `TemporalFusion`'s `default_encoder_registry()` is generated from it. A Program extends it (`StreamRegistry.extend`) for stream ids that don't fit a generic pattern — see `programs/minecraft/stream_registry.py`. `StreamRegistry.assert_complete(catalog)` fails loudly if any catalog stream has no declaration; `describe(catalog)` is what `runtime/loop.py` records into session metadata. |

## Encoders & fusion (Phase 4)

The neural-architecture shape is `stream event → modality encoder → latent
token → temporal fusion → latent state → policy`. Each encoder turns one
stream's recent window into a **fixed-width** vector using only generic
`StreamSpec` metadata (normalization `range`, grid `legend`, categorical
`categories`, `neutral` fill) — never world constants — so the same encoder
serves any Program. `TemporalFusion` lays the tokens out in a stable,
`stream_id`-ordered vector, filling silent streams with their neutral value so
the width is fixed, and hashes the layout so a model trained on one layout
fails loudly against an incompatible one. The dataset builder replays the
**same** fusion offline over recorded streams, so train-time and inference-time
features come from identical code (an online/offline parity test enforces it).

### Neural pixel vision

Fusion produces a *frozen, fixed-width* vector, which is the right shape for the
heuristic encoders but not for a **trainable** network.  So the pixel pathway
sits outside fusion: the `vision.frame.pixels` stream (a deterministic RGB frame
— colorized from the same semantic grid every backend emits, so the simulated
and mineflayer backends share it, and a real screenshot can later fill
`Observation.pixels` directly) is fed **raw** to a small CNN
(`models/vision.py`).  The CNN is trained **end to end** with the policy head by
behavioral cloning (`training/neural.py`), so the agent learns its own visual
features from the stream instead of a hand-written grid encoder.  Its companion
scalar input is the fused latent with every `vision.*` slice dropped
(`datasets.build_neural_dataset`), so the CNN is the sole visual pathway; the
`NeuralPolicy` reconstructs that same non-vision vector from the runtime's
`LatentState` at inference.  Because replay re-injects the recorded motor stream
(it never re-runs the policy), a neural encoder cannot affect the determinism
contract — only the deterministic renderer that produces the pixel stream
matters, and that is pure Python.

## Cadence guidance

Streams are multi-rate by design; publish at the rate that matches the sense:

- **vision** — once per program tick (the freshest frame is what matters).
- **body** — on change, plus a low-rate heartbeat so silence is
  distinguishable from stasis.
- **event** — irregular; only when something happens. An empty window is
  meaningful.
- **reward** — per tick (or irregular for sparse reward events).
- **motor** — at the policy's own cadence; an empty motor window is the NULL
  action, and it is an explicit, recorded decision.

`TickSynchronizer` tracks streams that go silent for many windows
(`silent_streams()`) so missing heartbeats surface as runtime-health
signals rather than silent staleness.

## Realtime multi-rate streaming (Phase 5)

Fast-forward runs everything on one deterministic clock as fast as the CPU
allows. **Realtime mode** (`--realtime`) instead lets each sense update at its
own rate in wall-clock time — vision at 10–30 Hz, a body heartbeat at 1–10 Hz,
events whenever they happen — while keeping determinism exactly where it is
promised.

### Two clocks, one deterministic

Every `StreamEvent` carries two timestamps:

- **`timestamp` — simulated time.** The deterministic replay clock. Windowing,
  hashing, replay and reward all use it. The realtime scheduler holds simulated
  time locked to the wall clock (it sleeps to keep the tick rate), so "10 Hz of
  simulated time" *is* "10 Hz of wall time" during a live run.
- **`arrived_at` — wall-clock arrival (metadata only).** The monotonic instant
  an event reached the bus, stamped only in realtime mode. It is **excluded
  from `hash()`**, so replay and hashing never depend on it. It exists purely so
  health metrics can measure the *actually delivered* cadence. Fast-forward logs
  omit it entirely and stay byte-identical.

### Rate-driven publication (the pacer)

`RatePacer` (in the Program's publisher) throttles a stream to a target rate.
Crucially it paces off **simulated** time, so pacing is deterministic:

- In **realtime** the scheduler keeps simulated ≈ wall, so vision paces to its
  target Hz in real time; irregular streams (`event.*`) carry no target rate
  and are never throttled; the per-tick world/reward streams run at the
  cognitive rate.
- In **fast-forward** the pacer is disabled and publication maps straight onto
  tick cadence — the established Phase-1 behavior, so tests stay fast and
  deterministic.

Because pacing is a pure function of simulated time, a realtime recording
**replays bit-for-bit in fast-forward**: replay re-enables the pacer off the
recorded simulated clock and regenerates the very same paced subset of frames.

### Asynchronous ingestion

`SensoryStreamBus(thread_safe=True)` adds a lock + condition variable so a real
backend (a mineflayer bridge, a screen-capture thread, the human-demo terminal
reader) can `publish()` from its own thread while the cognitive loop drains on
the main thread. The single-threaded simulated path is the default and stays
**lock-free and deterministic** — it pays nothing for machinery it does not use.

### Backpressure (bounded queues)

Realtime queues are bounded per stream; when one overflows, the policy declared
in its `StreamSpec.overflow` (or a per-modality default) decides what happens —
and every drop is **counted, never silent**:

| policy | behavior | default for |
|---|---|---|
| `coalesce` | collapse to the freshest event (a stale frame is worthless) | `vision`, `body` |
| `block` | never drop — the publisher waits for the consumer | `event` |
| `drop_oldest` | bounded ring: keep the most-recent `capacity` | everything else |

In the deterministic single-threaded path the queues are drained every tick, so
their bounds are never reached and ordering is unchanged.

### Missed-window & staleness health

`TickSynchronizer` and the `EpisodeSummary` account for realtime health:
**empty windows** (nothing arrived), **late windows** (a tick that started past
its deadline — the scheduler's missed ticks), **stale streams** (a rate-bearing
stream quiet for more than 2× its nominal period — a stopped publisher),
**motor emission rate**, **queue overflow counts**, and measured **wall-clock
rates** per stream. `dashboard` renders a realtime-health block for realtime
sessions.

### What determinism means in realtime mode

The motor log and every sensory-stream hash are still recorded. Replay re-runs
in fast-forward and verifies the simulated-time behavior tick-for-tick: the
recorded motor stream reproduces the same world trajectory, the same paced
sensory hashes (pacing reproduced off simulated time), and the same reward.
Wall-clock arrival times, being metadata, are free to differ between the live
run and its replay — that is the one thing that legitimately cannot be
reproduced, and nothing depends on it.

## The determinism contract

Determinism is infrastructure ([architecture.md](architecture.md)); streams
carry it forward:

1. **Simulated time drives everything deterministic.** `StreamEvent.timestamp`
   is simulated time; replay, hashing and pacing depend on it. Wall-clock
   arrival lives in `arrived_at` as metadata only (realtime mode) and is
   excluded from `hash()`.
2. **Per-stream monotonic sequence numbers**, assigned by the publishing
   bus, starting at 0 per episode (`bus.reset()`).
3. **Deterministic delivery order.** `drain()` sorts by
   `(timestamp, stream_id, sequence_number)`. Same publishes in ⇒ identical
   order out, regardless of publisher interleaving.
4. **Content hashes.** `StreamEvent.hash()` covers
   `(stream_id, sequence_number, timestamp, payload)` via canonical JSON —
   the unit of replay verification (streams-v2, Phase 3): re-inject the
   recorded motor stream and every regenerated sensory stream hash must match
   the log in order.
5. **Environment-agnostic.** Nothing under `core/streams/` imports from
   `programs/` (enforced by a test).

## Migration status

Completed phases: stream primitives (Phase 0), Program interface v2
(programs publish/consume streams), runtime loop v2 (cognitive ticks over
stream windows), stream-native recording/replay + tools (streams-v2),
modality encoders + temporal fusion with behavioral cloning on the latent
state (Phase 4), and **real-time multi-rate streaming (Phase 5)** — the
two-clock design, rate pacing, asynchronous ingestion, bounded-queue
backpressure and realtime health metrics described above.

The loop's policy `State` is now **derived from stream state**
(`Memory.latest_values().to_observation()`), not pulled from the Program:
`program.observe()` is no longer called by the loop (a test enforces it),
and the formerly observation-only fields are streams
(`body.in_water`, `body.alive`, `spatial.distance_from_spawn`). Remaining:
real neural encoders/fusion and learned world models. See the tracking
issue for the full plan.

Every stream in the Minecraft catalog now has an explicit `StreamDeclaration`
(issue #21: shape/schema and rate on `StreamSpec`, plus encoder binding,
trainable/fixed-stub, latent width, checkpoint key and train/eval behavior in
`registry.py`) — `test_stream_registry.py` asserts none are missing. Every
declaration today is a deliberate fixed stub; trainable
`StreamEncoderModule`s (`cognitive_runtime.neural`) are Phase B+.
