# Sensory & Motor Streams

The runtime is migrating from static observations to **time-indexed sensory
streams**: instead of asking "what is the current observation?", the runtime
asks **"what streams have arrived since the last cognitive tick?"**. This
document covers the Phase-0 stream substrate in
`cognitive_runtime/core/streams/` — the primitives everything later builds
on. The primitives are purely additive today: the legacy loop is unchanged
until Phase 2 of the migration.

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
(`body.health`, `vision.frame.grid`, `event.damage`). Modalities are
generic: Minecraft health, a Linux battery and robot joint stress are all
`body.*` streams; Minecraft frames, desktop pixels and robot cameras are all
`vision.*` streams. The brain consumes modalities, never environment fields.

| Modality | What flows on it | Example streams |
|---|---|---|
| `body` | Internal/self state | `body.health`, `body.hunger`, `body.position` |
| `vision` | Frames, pixels, grids | `vision.frame.grid` |
| `spatial` | Maps, locality, geometry | `spatial.nearby_blocks` |
| `audio` | Sound events/levels | `audio.ambient` |
| `event` | Discrete world happenings | `event.damage`, `event.item_collected` |
| `reward` | Reward components | `reward.survival` |
| `language` | Text in/out of the world | `language.chat` |
| `input` | Raw human input (demos) | `input.keyboard` |
| `world` | Global world state | `world.time_of_day`, `world.weather` |
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

## The determinism contract

Determinism is infrastructure ([architecture.md](architecture.md)); streams
carry it forward:

1. **Simulated time only.** `StreamEvent.timestamp` is simulated time,
   never wall clock. Replay depends on it.
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

Phase 0 (this document) is additive only. Completed phases: Program
interface v2 (programs publish/consume streams), runtime loop v2 (cognitive
ticks over stream windows), stream-native recording/replay + tools
(streams-v2), and modality encoders + temporal fusion with behavioral cloning
on the latent state (Phase 4). Remaining: real neural encoders/fusion, learned
world models, and real-time multi-rate streaming. See the tracking issue for
the full
plan.
