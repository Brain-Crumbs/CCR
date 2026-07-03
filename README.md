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
> Minecraft server. A real-Minecraft backend plugs in behind the same
> adapter seam (`SurvivalBackend`) without touching the runtime.

## Quick start

Requires Python ≥ 3.10. No dependencies (pytest for tests).

```bash
# Milestone 0: the continuous loop with the null policy
python -m cognitive_runtime run --policy null --episodes 1

# Baselines with recording (sessions/ by default)
python -m cognitive_runtime run --policy scripted --episodes 3

# Compare baseline policies on identical episode seeds
python -m cognitive_runtime evaluate --policies null,random,scripted --episodes 3

# Play yourself; the session becomes imitation-learning data
python -m cognitive_runtime demo

# Train the first learned policy (behavioral cloning) from any session(s)
python -m cognitive_runtime train --sessions sessions/<session-id> --out models/bc.json

# Evaluate the learned policy against the baselines
python -m cognitive_runtime evaluate --policies random,scripted,learned --model models/bc.json

# Replay a recorded session and verify tick-for-tick determinism
python -m cognitive_runtime replay --session sessions/<session-id>

# Inspect an episode / aggregate metrics across sessions
python -m cognitive_runtime view --session sessions/<session-id> --episode episode_00000
python -m cognitive_runtime dashboard
```

Run the tests:

```bash
pytest
```

## The loop

```python
while running:
    observation = program.observe()
    state       = perception.encode(observation)
    memory.update(state)
    prediction  = world_model.predict(state, memory)
    action      = policy.decide(state, memory, prediction)
    program.act(action)                 # NULL is a real action
    reward      = program.reward()
    learner.update(observation, action, reward)
    recorder.write_tick(...)
```

**NULL is a real action.** The agent must learn when *not* to act.

## Project structure

```
cognitive_runtime/
  core/        Program interface, Action/Observation/Reward, Policy,
               Perception, Memory, WorldModel, Learner  (environment-agnostic)
  runtime/     continuous loop, fixed-tick scheduler, config,
               recorder (JSONL ticks + episode summaries), replay/verify
  programs/
    minecraft/ SurvivalBox adapter, simulated backend, observations,
               actions, survival reward, evaluation metrics
  policies/    null, random, scripted survival, human-demo, learned (BC)
  training/    features, dataset builder, imitation trainer, policy comparison
  tools/       episode viewer, metrics dashboard, replay runner
docs/          architecture, program interface, Minecraft MVP, future AI-OS
tests/         determinism, rewards, replay fidelity, training milestones
```

## Key architectural rule

Minecraft-specific code must never leak into the runtime. The core runtime
only ever calls `program.observe()`, `program.act(action)`,
`program.reward()` — Minecraft knowledge lives in the Minecraft Program,
its reward module, and optional Minecraft-specific policy experiments.
Adding a second Program requires no runtime changes; that is the point.

## Documentation

- [docs/architecture.md](docs/architecture.md) — runtime design and components
- [docs/streams.md](docs/streams.md) — sensory/motor stream primitives and the determinism contract
- [docs/program-interface.md](docs/program-interface.md) — the universal Program contract
- [docs/minecraft-mvp.md](docs/minecraft-mvp.md) — SurvivalBox: scope, rewards, milestones
- [docs/future-ai-os.md](docs/future-ai-os.md) — the long-term AI-native OS direction
