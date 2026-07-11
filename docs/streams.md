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
| `internal` | Interoception: the runtime's own modulation signals, published back onto the bus (issue #58) | `internal.prediction_error`, `internal.reward_prediction_error`, `internal.learning_progress`, `internal.novelty`, `internal.risk`, `internal.risk_gate`, `internal.safe_novelty`, `internal.predicted_risk_aversion` (published every tick, `core/modulation.py`; the last three are issue #61's risk-gated intrinsic-drive terms); `internal.attention.weights` (published every tick by the deterministic attention controller, issue #59, `core/attention.py`) |

The `internal` modality is the biological-modulation analog (dopamine as
reward-prediction-error, etc.): the agent's own error, progress, novelty and
risk signals are recordable, replayable streams like any sense, consumable
by the attention controller (#59), intrinsic drive (#61) and replay
prioritization — see "Internal modulation streams (issue #58)" under
Migration status below, and [neural-stream-agent.md](neural-stream-agent.md).

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

**Scope note (issue #44).** Byte-identical replay (points 1-4 above) is the
contract for the *simulated* backend, and stays a fast, dependency-free CI
smoke test of the loop/stream/recorder plumbing (`tests/test_tools.py`,
`python -m cognitive_runtime replay`) -- it still catches a broken publish
order, a hashing regression, or a recorder/replay round-trip bug. It is
**not** a regression gate for learning runs: neural online training breaks
byte-identical replay (torch/GPU nondeterminism, weights mutating
mid-episode), and the remote backend never had it (`NonDeterministicSessionError`).
For those, `cognitive_runtime.training.statistical_evaluation` reports mean
+/- confidence interval over N episodes on matched conditions (survival,
reward by tier, exploration coverage, world-model prediction error, death
causes) and flags a regression only when a candidate checkpoint's interval no
longer overlaps the baseline's on the worse side -- see
[online-learning.md](online-learning.md)'s "Evaluation Gates" section.

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
wiring the neural encoders/learned fusion into the live policy path (#57),
the budgeted attention layer between memory and fusion (#59), and the
generative multi-horizon world model (#39). See
[neural-stream-agent.md](neural-stream-agent.md) for the full roadmap.

Every stream in the Minecraft catalog now has an explicit `StreamDeclaration`
(issue #21: shape/schema and rate on `StreamSpec`, plus encoder binding,
trainable/fixed-stub, latent width, checkpoint key and train/eval behavior in
`registry.py`) — `test_stream_registry.py` asserts none are missing. Every
declaration is either wired to trainable neural encoder metadata
(`neural_encoder`, `neural_latent_width`) or called out as a deliberate fixed
stub. The legacy scalar `TemporalFusion` path still uses the fixed
`encoder_factory` bindings for checkpoint compatibility, while the neural
path uses `StreamEncoderModule`s in `cognitive_runtime.neural`.

The reserved `audio.*` declaration is intentionally a fixed neural stub:
`AudioEncoder` emits a stable zero latent and checkpoints its stub state, but
no Program publishes audio and no capture/spectrogram backend exists yet.

### Internal modulation streams (issue #58) and the intrinsic drive (issue #61)

The runtime already thinks in streams, so neuromodulation is modeled as
*internal streams*, not hidden variables: `cognitive_runtime/core/modulation.py`
computes and `runtime/loop.py` publishes eight `internal.*` streams every
cognitive tick, each with the uniform `{"value": <float>}` payload the
reward engine's `intrinsic_stream` component kind already expects
(`programs/minecraft/reward_profile.py`, so any of them can be wired
straight into a reward profile's `intrinsic` slots without a translation
layer). Five are the raw signals #58 introduced; three are #61's risk-gated
"safe surprise" terms, derived from the raw five and never recomputed
reward-side:

- `internal.prediction_error` — the world model's next-latent prediction
  error (`core.world_model.Prediction.prediction_error`, issue #26),
  promoted to a stream. `None` for the heuristic `TrendWorldModel`, which
  never populates it.
- `internal.reward_prediction_error` — actual reward minus the world
  model's predicted reward (`Prediction.predicted_reward`); the dopamine
  analog. It is meant to modulate replay priority and memory tagging, not
  act as the whole attention system (out of scope here). `None` without a
  reward head.
- `internal.learning_progress` — is prediction error improving? Computed by
  `LearningProgressTracker` as a two-timescale EMA difference
  (`slow_ema - fast_ema`): positive means the model is getting better at
  predicting, near zero for a plateaued or noisy-but-static error. `None`
  when `internal.prediction_error` is unavailable this tick.
- `internal.novelty` — the combined novelty scalar (issue #27,
  `core.novelty.combine_novelty`), promoted to a stream alongside the
  richer `model.novelty` (which keeps its `{novelty, world_model_error,
  entity_surprise}` breakdown for debugging).
- `internal.risk` — the world model's risk/p_death head output
  (`Prediction.risk`), made visible. Always published: `Prediction.risk`
  defaults to `0.0`, never `None`.
- `internal.risk_gate` — `safe_gate(risk) = sigmoid(-(risk - risk_threshold)
  / temperature)` (issue #61): `1.0` well below `risk_threshold` (safe to
  be curious), `0.0` well above it, `0.5` exactly at the threshold. Always
  published, like `internal.risk`. `risk_threshold`/`temperature` are
  `ModulationTracker` construction params, sourced from
  `RuntimeConfig.intrinsic_risk_threshold`/`intrinsic_risk_temperature`
  (`--intrinsic-risk-threshold`/`--intrinsic-risk-temperature`, defaults
  `0.5`/`0.15`).
- `internal.safe_novelty` — `internal.novelty * internal.risk_gate` (issue
  #61): surprise sought only when it doesn't forecast suffering. `None`
  exactly when `internal.novelty` is `None`.
- `internal.predicted_risk_aversion` — `-internal.risk` (issue #61),
  already sign-flipped so a *positive* reward-profile weight on this slot
  amplifies an aversive shaping term proportional to predicted risk —
  avoidance happens before damage, not after. Always published.

`ModulationTracker` owns the `learning_progress` EMA state across ticks and
episodes within a run (a world model's predictive skill is a property of
the model, not of any one episode, so episode resets don't reset it) and
exposes `state_dict()`/`load_state_dict()`, ready for a checkpoint's
`training_stats` (issue #20) so a resumed run doesn't reset the baselines —
wiring that into a concrete learner's checkpoint bundle is left to whichever
learner owns one. `risk_threshold`/`temperature` are constant for the
tracker's lifetime (a run-level config knob, not evolving state), so they
are not part of `state_dict()`; `CognitiveRuntime.run()` instead records
them into session metadata's `intrinsic_modulation` field for provenance.
The `internal.*` wildcard was already declared `agent_input` with
`AttentionMetadata(modality="internal")` in `DEFAULT_STREAM_REGISTRY` ahead
of this issue (issue #32); these are runtime-computed signals registered
directly on the sensory bus, like `model.novelty`/`model.value_estimate`,
not part of any Program's stream catalog. `episode_viewer.py` renders all
eight per recent decision, and `neural.replay_buffer.load_session_into_buffer`
reads `internal.novelty` and `internal.reward_prediction_error` per tick
(issue #28) instead of always leaving novelty `None` for loaded sessions.

### Stream classification and attention metadata (issue #32)

Every `StreamDeclaration` also carries a `classification` -- **agent_input**
(raw/near-raw sensory, proprioceptive, motor, reward and interoceptive
(`internal.*`) streams the policy should actually consume), **aux_debug**
(hand-computed semantic summaries -- `world.front_block`, `world.sheltered`,
`vision.entities`, `event.*` narrations -- useful for dashboards/replay and
auxiliary-loss targets, but not raw policy input), or **privileged**
(exact ground-truth simulator state -- the unbounded-vocabulary `_exact`
mirrors of *world* facts like `world.nearby_blocks_exact` -- recorded for
replay fidelity only, excluded from both policy input and aux-loss targets).
`StreamRegistry.assert_complete`/`missing` already fail loudly on an
undeclared stream, so requiring `classification` on every `StreamDeclaration`
(`StreamDeclaration.__post_init__`) makes the audit complete by construction;
`tests/test_stream_classification.py` covers it end to end.

Every `agent_input` declaration additionally carries an `AttentionMetadata`
(modality, expected sample rate, relative encoding compute cost, and whether
the stream can carry a direction/region localization hint) -- the per-stream
metadata the attention controller (#59) scores against and the orienting
reflex (#60) will consume for localizable streams. `describe()` includes both
the classification and the attention fields in session metadata, so a
recorded session is self-describing about which streams the policy consumed.

### Deterministic attention controller (issue #59)

`cognitive_runtime/core/attention.py` turns the `AttentionMetadata` above
into an actual per-tick allocation: every `agent_input`-classified stream
(including `internal.*`) gets an `AttentionSignal` from nothing but its own
recent history in the `TemporalBuffer` --

- `novelty` — did this tick's payload hash change from the last one?
- `prediction_error` — magnitude of a numeric stream's short-window trend.
- `reward_relevance` — |Pearson correlation| between the stream's recent
  numeric values and the catalog's `reward.*` stream over the same window.
- `risk` — the current `internal.risk` value, shared across every stream's
  signal this tick (a global "how dangerous is it right now" term).
- `recency` — the same half-life decay `TemporalFusion` uses for silent
  streams.
- `boredom` — fraction of the recent window repeating the same value.
- `compute_cost` — the registry's low/medium/high cost, as a penalty.

`AttentionController.compute()` blends these into a score per
`AttentionCoefficients`, then `AttentionBudget` forces a choice: only the
top `max_streams` streams (by score) get nonzero weight, capped at
`max_total_weight` in total. A hysteresis/dwell rule protects the selected
"focus" stream from thrashing between near-equal spikes (`dwell_ticks`
minimum persistence), while a challenger that clears `displacement_margin`
above the focus's captured score still captures it immediately -- bottom-up
capture. `attention="off"` (the CLI/`RuntimeConfig` default) instead gives
every stream weight `1.0`, byte-identical to no attention controller at all.

The runtime loop computes one `AttentionState` per tick
(`CognitiveRuntime.attention`), stores it on `Memory.attention_state()`,
publishes it as `internal.attention.weights`, folds its per-stream reason
breakdown into `DecisionRecord.attention`, and feeds its weights into both
`TemporalFusion.fuse(attention_weights=...)` (gating stream slices) and
`LiveLearnedFusion.fuse`/`maybe_train_step` (issue #57's
`LatentFusionModel.forward(attention_weights=...)` hook). `--attention
{off,budgeted}` selects the mode; `tools.episode_viewer` renders the focus
timeline with its reason breakdown, and `tools.metrics_dashboard` reports
per-stream focus totals and average budget spent across budgeted episodes.

`StreamRegistry.to_encoder_registry(classifications={"agent_input"})` builds
a fusion registry restricted to agent-input streams -- the "raw input"
profile: `ccr run --input-profile raw` fuses only that subset into the
online/actor-critic policy's state, while aux/debug and privileged streams
keep publishing and recording exactly as before (nothing about *what gets
recorded* changes, only what reaches the policy). The neural pixel-BC
pipeline has the analogous ablation via
`training.datasets.build_neural_dataset(..., stream_profile="raw")`: "pixel
only" (pixels + minimal agent-input proprioception) vs the default "full"
("pixels + semantics").
