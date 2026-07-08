# Architecture

## Vision

The Continuous Cognitive Runtime is an always-running cognitive system that
inhabits interactive worlds. It does not wait for prompts. Every tick it
observes, perceives, remembers, predicts, decides, acts (or idles), learns,
and records.

```
                 Continuous Cognitive Runtime
                            |
        ------------------------------------------------
        |              |              |                |
    Perception       Memory       World Model        Policy
        |              |              |                |
        ------------------------------------------------
                            |
                      Motor Output
                            |
                        Program API
                            |
        ------------------------------------------------
        |              |              |                |
     Minecraft       ToyOS        Linux VM        AI OS
```

## Design principles

- continuous execution
- **null action as default** — inaction is a decision the agent must learn
- environment-agnostic runtime — Programs define experience
- deterministic infrastructure — same seed + same actions ⇒ same world
- replayable sessions — if it can't be replayed, it can't be debugged
- safe sandboxed execution
- modular sensors and actions
- curriculum-based learning
- future compatibility with AI-native OS concepts

## Components

All core components live in `cognitive_runtime/core/` and know nothing
about any specific world.

| Component | Module | Role |
|---|---|---|
| Program | `core/program.py` | Universal environment interface (see [program-interface.md](program-interface.md)) |
| Observation | `core/observation.py` | Timestamped structured data + optional frame; content-hashable for replay/novelty |
| Action | `core/action.py` | Opaque named action with params; `NULL_ACTION` is first-class |
| Perception | `core/perception.py` | Encodes observations into runtime state (generic numeric flattening in the MVP) |
| Memory | `core/memory.py` | Stream-native: a `TemporalBuffer` of recent events + the fused `LatentState` + motor emissions; novelty, repetition and per-stream trend signals |
| WorldModel | `core/world_model.py` | Predicts from per-stream trends in memory; MVP ships a trend extrapolator with a generic vital-risk heuristic |
| Policy | `core/policy.py` | `emit(state, memory, prediction) -> list[Action]` (`[]` == NULL); `SingleActionPolicy` adapts one-action-per-tick policies |
| Learned policies | `policies/learned.py`, `policies/neural_policy.py` | `learned` — linear softmax BC over the fused latent (dependency-free); `neural` — an end-to-end **pixel-vision CNN** (`models/vision.py`) that learns its own vision from the `vision.frame.pixels` stream (optional, torch) |
| Learner | `core/learner.py` | Online `update(window)` hook reading `reward.scalar`; MVP learns offline via behavioral cloning |
| RewardSignal | `core/reward.py` | Scalar value + named components + semantic events |
| Streams | `core/streams/` | Time-indexed sensory/motor primitives, per-modality encoders and `TemporalFusion` producing a fixed-width `LatentState`; see [streams.md](streams.md) |

The runtime machinery lives in `cognitive_runtime/runtime/`:

| Component | Module | Role |
|---|---|---|
| CognitiveRuntime | `runtime/loop.py` | The continuous loop (v2: cognitive ticks over stream windows); runs episodes back to back |
| Streams | `core/streams/` | Buses, `TemporalBuffer`, `TickSynchronizer`, encoders — the sensory/motor substrate ([streams.md](streams.md)) |
| FixedTickScheduler | `runtime/scheduler.py` | Holds a fixed tick rate (realtime) or fast-forwards; tracks missed ticks |
| Recorder | `runtime/recorder.py` | Streams-v2 log: `*.streams.jsonl` (every StreamEvent, both directions) + `*.decisions.jsonl` (one cognitive tick each) + episode summaries (per-stream counts/rates) |
| Replay | `runtime/replay.py` | Re-injects the recorded motor stream tick-aligned through `step()` and verifies every regenerated sensory event hash in order |

## The loop (v2: cognitive ticks over stream windows)

The runtime no longer asks "what is the current observation?" — it asks
**"what streams have arrived since the last cognitive tick?"** (see
[streams.md](streams.md)):

```python
while running:
    scheduler.wait_for_next_tick()
    for _ in range(program_ticks_per_cognitive_tick):
        program.step()                          # drains motor bus, publishes streams
    window  = synchronizer.collect(sensory_bus) # events since the last cognitive tick
    memory.update(window)                       # TemporalBuffer of recent events
    latent  = fusion.fuse(window, memory.buffer)   # fixed-width LatentState
    state   = memory.latest_values().to_observation()  # stream-derived, no observe()
    pred    = world_model.predict(state, memory)
    motor   = policy.emit(state, memory, pred)  # list of motor emissions; [] == NULL
    for action in motor: motor_bus.publish(...)
    learner.update(window)                      # reads reward.scalar from the window
    recorder.write_cognitive_tick(window.events, motor, decision)
```

### One-tick actuation latency

Motor emissions from cognitive tick *t* sit on the motor bus and are applied
by `program.step()` at the **start of tick *t+1***. This one-tick latency is
intentional — it is how real sensorimotor loops behave — and it is stable
and documented because **replay and reward attribution depend on it**.
Replay mirrors the loop exactly (`step()` + the motor bus in the same
order), so recorded sessions still reproduce byte-for-byte: reset with the
recorded seed, re-inject the recorded motor stream tick-aligned, and every
regenerated sensory event hash must match the log in order.

### Cognitive vs program ticks

`program_ticks_per_cognitive_tick` (default 1) decouples the decision rate
from the world rate: with a ratio of *N* the world advances *N* program
ticks per decision, and the every-tick streams (vision, `world.time`,
`reward.scalar`) arrive **batched** *N*-to-a-window. The world still moves
every program tick; the agent simply decides less often.

Notes:

- **NULL is a real action.** An empty motor emission is an explicit,
  recorded decision (`selected_action == "NULL"`), counted in
  `EpisodeSummary.null_action_ticks`. The world still advances on a NULL
  tick (hunger drains, mobs move, time passes) and still publishes
  `reward.scalar`.
- **Perception is retired from the loop.** The per-modality stream encoders
  plus `TemporalFusion` produce the fixed-width `LatentState` in place of the
  old `StructuredPerception`; the default learned policy consumes that fused
  latent state.
- **Stream-derived policy state.** The `State` handed to policies is
  reconstructed from the latest value of each stream
  (`Memory.latest_values().to_observation()`); the loop never calls
  `program.observe()` (a test enforces it). Observation-based policies
  (scripted, human demo, the handcrafted A/B featurizer) read stream-keyed
  data — the latest value each stream has published.
- Decision latency (window collection → motor emission) is measured per
  cognitive tick and recorded, alongside per-stream event rates and
  silent-stream gaps.
- Timestamps inside stream events and observations are *simulated* time so
  that hashing and replay are wall-clock independent.

## Determinism and replay

Determinism is infrastructure, not an afterthought:

- Programs must reset deterministically from a seed.
- All randomness inside a Program flows through a seeded RNG whose state is
  captured by `snapshot()`.
- `python -m cognitive_runtime replay --session <dir>` re-simulates every
  episode and fails loudly on the first diverging tick or reward mismatch.

## The key architectural rule

Do not let Minecraft-specific code leak into the runtime.

Bad:

```
runtime.checkHunger()
runtime.mineBlock()
runtime.craftPickaxe()
```

Good:

```
program.observe()
policy.decide()
program.act(action)
reward.evaluate()
```

Minecraft knowledge belongs in the Minecraft Program
(`programs/minecraft/`), its reward module, and optional Minecraft-specific
policy experiments (`policies/scripted.py`, `training/features.py`). The
core runtime must remain environment-independent — the MVP completion
criterion is that a second Program can be added without rewriting it.
