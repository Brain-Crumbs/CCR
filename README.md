# Continuous Cognitive Runtime

A research-oriented AI runtime designed to **continuously inhabit interactive
worlds**. Unlike a chatbot or task-specific agent, the system does not wait
for prompts: it observes, updates memory, predicts, decides whether to act,
and either acts or deliberately remains idle — every tick, forever.

```
Runtime + Program = Experience
```

- **Runtime** — the always-running cognitive system (perception, memory,
  world model, policy, learning, recording). Environment-agnostic.
- **Program** — an environment the runtime can inhabit (Minecraft, ToyOS,
  a Linux VM, a browser, a future AI-native OS). Programs create experiences.
- **Experience** — the interaction between the two; what the runtime learns from.

This repository contains the runtime core and the first Program:
**MinecraftSurvivalBox** — survive as long as possible in a constrained
survival world. See [docs/minecraft-mvp.md](docs/minecraft-mvp.md).

> The MVP ships with a deterministic *simulated* survival backend so the
> entire stack — continuous loop, rewards, recording, replay, baselines,
> behavioral cloning — runs end to end with zero dependencies and no
> Minecraft server. A **real-Minecraft backend** (`--backend remote`) plugs in
> behind the same adapter seam (`SurvivalBackend`) via a mineflayer bridge,
> without touching the runtime — see
> [bridge/mineflayer/README.md](bridge/mineflayer/README.md).

## Quick start

Requires Python ≥ 3.10. No dependencies for the core stack (pytest for tests).
Pixel-vision neural training/inference is opt-in: `pip install -e .[neural]`
pulls PyTorch; everything else runs without it.

```bash
# Milestone 0: the continuous loop with the null policy
python -m cognitive_runtime run --policy null --episodes 1

# Baselines with recording (sessions/ by default).  Add --record-frames when
# the session is destined for training: frames are logged hash-only otherwise,
# so the latent policy's vision features would train blind.
python -m cognitive_runtime run --policy scripted --episodes 3 --record-frames

# Real-time multi-rate streaming: hold 20 Hz in wall-clock time, vision paced
# to 10 Hz and a 2 Hz body heartbeat (see docs/streams.md). Still replayable.
python -m cognitive_runtime run --policy scripted --realtime --episode-ticks 200

# Compare baseline policies on identical episode seeds
python -m cognitive_runtime evaluate --policies null,random,scripted --episodes 3

# Play yourself; the session becomes imitation-learning data
python -m cognitive_runtime demo

# Train the first learned policy (behavioral cloning) from any session(s)
python -m cognitive_runtime train --sessions sessions/<session-id> --out models/bc.json

# Evaluate the learned policy against the baselines
python -m cognitive_runtime evaluate --policies random,scripted,learned --model models/bc.json

# Train a pixel-vision CNN end to end: it learns its own vision from the RGB
# vision.frame.pixels stream (needs frames in the log + the optional neural extra)
#   pip install -e .[neural]
python -m cognitive_runtime train --model-type neural \
    --sessions sessions/<session-id> --out models/vision_bc.pt
python -m cognitive_runtime evaluate --policies scripted,neural --model models/vision_bc.pt

# Replay a recorded session and verify tick-for-tick determinism
python -m cognitive_runtime replay --session sessions/<session-id>

# Inspect an episode / aggregate metrics across sessions
python -m cognitive_runtime view --session sessions/<session-id> --episode episode_00000
python -m cognitive_runtime dashboard

# Inhabit a real Minecraft server via the mineflayer bridge (see
# bridge/mineflayer/README.md): npm install once, point at your server, then:
CCR_MINECRAFT_HOST=localhost python -m cognitive_runtime run \
    --backend remote --realtime --policy scripted --episode-ticks 400 --record-frames
```

Run the tests:

```bash
pytest
```

## The loop

The runtime asks **"what streams have arrived since the last cognitive
tick?"**, not "what is the current observation?" (see
[docs/streams.md](docs/streams.md)):

```python
while running:
    for _ in range(program_ticks_per_cognitive_tick):
        program.step()                          # drains motor bus, publishes streams
    window  = synchronizer.collect(sensory_bus) # events since the last cognitive tick
    memory.update(window)                       # TemporalBuffer of recent events
    latent  = fusion.fuse(window, memory.buffer)   # fixed-width LatentState
    state   = memory.latest_values().to_observation()   # stream-derived, no observe()
    prediction = world_model.predict(state, memory)
    motor      = policy.emit(state, memory, prediction)  # [] == NULL
    for action in motor: motor_bus.publish(...)          # applied next tick
    learner.update(window)
    recorder.write_cognitive_tick(window.events, motor, decision)  # streams-v2 log
```

**NULL is a real action.** An empty motor emission is an explicit, recorded
decision — the agent must learn when *not* to act. Motor emissions are
applied one tick later (a deliberate, replay-stable actuation latency); see
[docs/architecture.md](docs/architecture.md).

## Project structure

```
cognitive_runtime/
  core/        Program interface, Action/Observation/Reward, Policy,
               Perception, Memory, WorldModel, Learner  (environment-agnostic)
  runtime/     continuous loop, fixed-tick scheduler, config,
               recorder (streams-v2: stream log + decisions + summaries), replay/verify
  programs/
    minecraft/ SurvivalBox adapter, simulated backend, remote (mineflayer)
               backend, observations, actions, survival reward, metrics
  policies/    null, random, scripted survival, human-demo, learned (linear BC),
               neural (end-to-end pixel-vision CNN)
  training/    features, dataset builders (linear + neural), imitation trainer,
               neural trainer (end-to-end BC), policy comparison
  models/      neural models (pixel-vision CNN); optional, torch-only
  tools/       episode viewer, metrics dashboard, replay runner
bridge/
  mineflayer/  Node bridge to a live Minecraft server (real backend)
  fake/        SimulatedWorld over the same protocol (tests / reference)
docs/          architecture, program interface, Minecraft MVP, future AI-OS
tests/         determinism, rewards, replay fidelity, training, remote backend
```

## Key architectural rule

Minecraft-specific code must never leak into the runtime. The core runtime
only ever talks to the generic `Program` interface — `program.step()` plus
the sensory/motor stream buses; even the `State` handed to policies is
derived from stream state, never pulled from the Program. Minecraft
knowledge lives in the Minecraft Program, its reward module, and optional
Minecraft-specific policy experiments. Adding a second Program requires no
runtime changes; that is the point.

## Documentation

- [docs/neural-stream-agent.md](docs/neural-stream-agent.md) — **the target architecture and roadmap**: a neural, stream-native agent with learned encoders, fusion, world model, and actor/critic online learning
- [docs/architecture.md](docs/architecture.md) — runtime design and components
- [docs/streams.md](docs/streams.md) — sensory/motor stream primitives and the determinism contract
- [docs/online-learning.md](docs/online-learning.md) — trainable stream modules and the online Q baseline
- [docs/program-interface.md](docs/program-interface.md) — the universal Program contract
- [docs/minecraft-mvp.md](docs/minecraft-mvp.md) — SurvivalBox: scope, rewards, milestones
- [docs/future-ai-os.md](docs/future-ai-os.md) — the long-term AI-native OS direction
