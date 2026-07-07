# The Universal Program Interface

A **Program** is an environment the runtime can inhabit: Minecraft Survival,
a ToyOS desktop, a Linux VM, a browser, a coding environment, a robot
simulator, or an AI-native OS workspace. Programs create experiences; the
runtime learns from them.

Every Program implements the same interface (`cognitive_runtime/core/program.py`):

```python
initialize(config)                 # prepare / connect / load config
observe() -> Observation           # current view of the world
act(action: Action) -> ActionResult   # apply one action; advances one tick
reward() -> RewardSignal           # reward for the most recent tick
is_complete() -> bool              # episode over?
reset(seed)                        # new episode, deterministic from seed
snapshot() -> snapshot_id          # capture full world state
restore(snapshot_id)               # restore captured state
metadata() -> ProgramMetadata      # name, version, action space, observation keys
episode_stats() -> dict            # optional: program-specific summary stats
```

This allows the same runtime to move between Minecraft, ToyOS, a VM, or a
future AI-native operating system without modification.

## The streams-first contract (interface v2)

The interface is migrating from pull-style observations to **time-indexed
streams** (see [streams.md](streams.md)): instead of the runtime pulling
`observe() -> Observation`, Programs publish `StreamEvent`s onto a
`SensoryStreamBus` and drain actions from a `MotorStreamBus`. Three methods
carry the new contract:

```python
stream_catalog() -> list[StreamSpec]     # streams this Program publishes
attach_buses(sensory, motor)             # register catalog, publish initial snapshot
step()                                   # advance one program tick
```

Contract details:

- **`step()` replaces `act(action)` as the tick driver.** It drains pending
  motor events, applies them in deterministic order, advances the world one
  tick, and publishes this tick's sensory/reward/event streams. **Zero
  motor events is a NULL tick** — the world still advances: hunger drains,
  mobs move, time passes.
- **Motor events** are `motor.command` events with payload
  `{"action": <Action.key()>}` (`core/streams/motor.py`). The action space
  stays program-defined and opaque to the runtime.
- **Invalid motor events never raise.** Malformed or out-of-space commands
  are rejected by publishing `event.action_rejected` (payload: reason);
  the world steps anyway.
- **Every publication uses simulated time** and flows through the bus so
  per-stream sequence numbers stay monotonic.
- **`reset(seed)` also resets both buses** (`bus.reset()`) and republishes
  an initial full snapshot per stream, so subscribers never start blind.
- **Reward is a stream too**: publish `reward.scalar`
  (`{"value", "components"}`) every tick; semantic happenings are
  irregular `event.*` streams.

### Cadence guidance

Publish each stream at its native rate, not everything every tick:
vision per tick; body vitals on change plus a heartbeat (so silence is
distinguishable from a dead sensor); spatial/world state on change;
events irregularly; reward per tick. `DeltaPublisher`
(`core/streams/delta.py`) keeps the on-change detection in one place.

### Migrating a legacy Program

1. Wrap it: `ObservationStreamShim(program)` (`core/streams/shim.py`)
   publishes generic streams by diffing consecutive `observe()` results —
   one `observation.<key>` stream per top-level data key, the frame as
   `vision.frame.grid`. This keeps any pull-style Program runnable on the
   stream substrate with zero changes.
2. Then migrate for real: define a `StreamSpec` catalog with native
   cadences and modality-prefixed ids, implement
   `attach_buses()`/`step()`, and move reward emission onto
   `reward.scalar`. `programs/minecraft/streams.py` +
   `MinecraftSurvivalBox.step()` are the reference migration.
3. The reverse view, `LatestValueView(buffer).to_observation()`,
   reconstructs an Observation-shaped snapshot from latest stream values
   for observation-based policies and featurizers.

The loop drives Programs through `step()` and the buses, and the `State` it
hands policies is derived from stream state (`LatestValueView`), so the loop
never calls `observe()` (a test enforces it). The legacy
`observe()`/`act()`/`reward()` contract remains for shim-wrapped pull-style
Programs and parity tests; the sections below describe it.

## Contract details

### Ticks

`act()` advances the world exactly one tick — including `NULL`. The runtime
always calls `observe()` before `act()` and `reward()` after it, once per
tick. Idle worlds still move: mobs approach, hunger drains, time passes.

### Observations

`Observation` carries:

- `timestamp` — *simulated* time (wall-clock independence keeps replay honest)
- `tick` — monotonically increasing world tick
- `data` — structured, JSON-serializable metadata (positions, vitals, inventory…)
- `frame` — optional 2-D grid standing in for screen pixels

`Observation.hash()` is a deterministic content hash over `tick`, `data`
and `frame`. Replay verification and novelty detection depend on it, so
Programs must round floats before putting them in `data`.

### Actions

Actions are opaque `(name, params)` pairs. The Program publishes its valid
action space via `metadata().action_space`. Unknown or malformed actions
must be rejected with `ActionResult(ok=False)` *before* the world is
stepped, so an invalid action never consumes a tick. The runtime still
records the attempt.

### Rewards

`reward()` returns the signal for the most recent tick: a scalar `value`,
a breakdown by named `components`, and semantic `events`
(e.g. `broke_block:tree`, `died`). Reward logic is part of the Program
side, never the runtime.

### Determinism

`reset(seed)` must produce identical worlds for identical seeds, and all
in-world randomness must flow through seeded RNG state captured by
`snapshot()`/`restore()`. The replay runner enforces this.

Determinism and snapshots are **capabilities**, not universal guarantees: a
Program backed by a live external world (a real Minecraft server, a real
browser) declares `metadata().deterministic = False` and may raise
`NotImplementedError` from `snapshot()`/`restore()`. Its sessions are still
recorded and trainable, but `replay` skips re-simulation for them with a
clear message.

## Adding a new Program

1. Create `cognitive_runtime/programs/<name>/`.
2. Implement the `Program` ABC (adapter + observations + actions + rewards).
3. Give it a deterministic backend (simulated or real, behind a seam like
   SurvivalBox's `SurvivalBackend`).
4. Wire it into the CLI.

No changes to `core/` or `runtime/` are required — if they are, that is a
bug in the runtime's design.

## Planned Program tracks

- **Track A — game worlds:** Minecraft SurvivalBox → ResourceGathering →
  Crafting → ShelterBuilding; Sokoban; 2-D survival.
- **Track B — interface worlds:** ToyOS, fake browser/file system/terminal,
  Linux VM, real browser, VS Code.
- **Track C — AI-native OS:** see [future-ai-os.md](future-ai-os.md).
