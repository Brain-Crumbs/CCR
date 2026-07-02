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
