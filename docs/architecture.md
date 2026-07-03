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
| Memory | `core/memory.py` | Bounded window of states/actions/hashes; novelty, repetition and trend signals |
| WorldModel | `core/world_model.py` | Predicts future state; MVP ships a trend extrapolator with a generic risk heuristic |
| Policy | `core/policy.py` | `decide(state, memory, prediction) -> Action` |
| Learner | `core/learner.py` | Online `(obs, action, reward)` hook; MVP learns offline via behavioral cloning |
| RewardSignal | `core/reward.py` | Scalar value + named components + semantic events |
| Streams | `core/streams/` | Time-indexed sensory/motor stream primitives (Phase 0, not yet wired into the loop); see [streams.md](streams.md) |

The runtime machinery lives in `cognitive_runtime/runtime/`:

| Component | Module | Role |
|---|---|---|
| CognitiveRuntime | `runtime/loop.py` | The continuous loop; runs episodes back to back |
| FixedTickScheduler | `runtime/scheduler.py` | Holds a fixed tick rate (realtime) or fast-forwards; tracks missed ticks |
| Recorder | `runtime/recorder.py` | JSONL tick records + episode summaries per session |
| Replay | `runtime/replay.py` | Re-simulates recorded actions and verifies observation hashes |

## The loop

```python
while running:
    observation = program.observe()
    state       = perception.encode(observation)
    memory.update(state)
    prediction  = world_model.predict(state, memory)
    action      = policy.decide(state, memory, prediction)
    program.act(action)                  # NULL is a real action
    reward      = program.reward()
    learner.update(observation, action, reward)
    recorder.write_tick(...)
```

Notes:

- The Program advances exactly one tick per `act()` call, including NULL.
  This makes recorded sessions exactly replayable: reset with the recorded
  seed, feed back the recorded actions, and every observation hash must match.
- Decision latency (perception → decision) is measured per tick and recorded.
- Timestamps inside observations are *simulated* time so that hashing and
  replay are wall-clock independent.

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
