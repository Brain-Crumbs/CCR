# Minecraft MVP: SurvivalBox

**Program name:** `MinecraftSurvivalBox`
**Initial objective:** stay alive for 5 minutes (6000 ticks at 20 tps).

The MVP proves that the runtime can run continuously, observe an external
world, choose actions, receive rewards, record sessions, evaluate behavior,
and support future learning.

## Scope

**In scope:** environment adapter, continuous loop, fixed tick rate,
observation model, action model, reward function, episode recording,
baseline policies, survival evaluation metrics.

**Out of scope:** general intelligence, full language understanding,
large-scale neural training, multiplayer, modded complexity, real OS
integration, AI-native OS implementation, autonomous web/desktop control.

## The backend

The adapter (`programs/minecraft/adapter.py`) talks to a pluggable
`SurvivalBackend`:

- **`SimulatedBackend` (shipped):** a deterministic, seeded survival world
  (`programs/minecraft/world.py`) with terrain and biomes (plains, forest,
  desert, lake), a day/night cycle, hunger/health/oxygen, zombies that
  spawn at night and burn at dawn, block breaking/placing, and an
  inventory + hotbar. Same seed + same actions ⇒ byte-identical
  observations, which is what makes replay verification possible.
- **`RemoteMinecraftBackend` (stub):** the seam for real Minecraft
  (mineflayer / Malmo / RCON driving a fixed-seed, world-bordered server).
  Implementing it requires no changes above the adapter.

Starting conditions follow the plan: fixed seed, limited world boundary
(walled), survival mode, daytime start, controlled difficulty, short
episodes, optional spawn-near-resources. The goal is not to make Minecraft
hard yet — the goal is to make learning measurable.

## Observations (MVP)

`timestamp`, screen `frame` (11×11 coarse top-down grid standing in for
pixels), `health`, `hunger`, `oxygen`, `position`, `yaw`/`pitch`,
`inventory` summary, `selected_slot`/`hotbar`, `nearby_blocks` (5×5 patch),
plus `biome`, `time_of_day`, `mobs` (distance/bearing), `front_block`,
`sheltered`, `distance_from_spawn`.

Future: audio, crafting state, tool durability, semantic events,
long-range map memory.

## Actions (MVP — kept small)

`NULL`, `MOVE_FORWARD/BACKWARD/LEFT/RIGHT`, `JUMP`, `SNEAK`, `SPRINT`,
`LOOK_LEFT/RIGHT/UP/DOWN`, `ATTACK`, `USE`, `SELECT_HOTBAR_SLOT(slot)`.

Later: `CRAFT`, `DROP_ITEM`, `OPEN_INVENTORY`, `MOVE_INVENTORY_ITEM`,
`PLACE_BLOCK`, `EQUIP_ITEM`, `TYPE_COMMAND`.

## Reward design (`programs/minecraft/rewards.py`)

The reward prevents the agent from learning pure passivity:

| Family | Rule | Value |
|---|---|---|
| Base survival | per tick alive / on death | +0.01 / −10.0 |
| Body state | 100 damage-free ticks at high hp | +0.05 |
| | per damage event | −0.5 |
| | per hunger point lost | −0.25 |
| | entering critical health / hunger | −1.0 each |
| Exploration | new block type observed (cap 2.0) | +0.1 |
| | new biome (cap 1.0) | +0.2 |
| | per 10 blocks of new max distance (cap 2.0) | +0.1 |
| Item diversity | new inventory item type (cap 5.0) | +0.5 |
| | first tool / first food | +1.0 each |
| | first block placed | +1.0 |
| Safety/shelter | enters enclosed space / creates light source / survives first night | +1.0 each, once |
| Anti-stagnation | identical action streak > 20 | −0.01/tick |
| | idle too long *without context* (no threat, healthy) | −0.05 |
| | spinning in place | −0.1 |
| | no observation novelty for 200 ticks | −0.1 |

Novelty rewards are **capped** so the agent cannot optimize for endless
wandering or junk collection. "First tool" and "light source" are dormant
in the simulated backend (no crafting yet) and activate with a richer
backend.

## Baseline policies

- **null** — always NULL; verifies runtime stability, passive baseline.
- **random** — uniform over the action space; lower-bound behavior.
- **scripted** — hardcoded survival heuristics (fight/flee, eat, harvest,
  unstick, wander); validates reward and metrics.
- **human** — you play in the terminal (`demo` command); the recorded
  session is imitation-learning data.

## First learned policy

Behavioral cloning (`training/`): structured-observation features + recent
action history → softmax over the action space (pure-Python softmax
regression — deliberately tiny, easily replaced by a real network).
Training data: human demonstrations, scripted traces, and successful
episode replays (`--min-reward` filter). Success target: outperform the
random baseline on survival time and reward (asserted in
`tests/test_training.py`).

## Recording, replay, metrics

The recorded artifact of a session is the **stream log** (streams-v2), not
reconstructed observations. Per episode:

- `episode_XXXXX.streams.jsonl` — one line per `StreamEvent`, both directions
  (`dir: sensory|motor`), in bus-drain order, each with its content `hash`.
  Streams elided for size (`exclude_streams`) keep a hash-only line so replay
  stays complete.
- `episode_XXXXX.decisions.jsonl` — one line per cognitive tick (window span,
  per-stream event counts, motor emitted, policy, latency, window reward); this
  is where NULL decisions are visible even though they emit no motor events.
- `episode_XXXXX.summary.json` — seed, duration, death reason, total reward,
  distance, items, damage, food, blocks broken/placed, success, and per-stream
  event counts + rates.

`replay` rebuilds the program from `session.json`, resets with the recorded
seed, re-injects the recorded **motor** stream tick-aligned, and verifies that
every re-generated **sensory** event hash matches the log in order (reporting
the first divergence as stream + seq + tick). If a session cannot be replayed,
it cannot be debugged or improved seriously.

Metrics (`evaluate` / `dashboard`): survival time, death rate,
damage/minute, food consumed, items collected, blocks broken/placed,
null-action rate, distance, reward/minute, ticks/second, decision latency,
missed ticks.

## Milestones

| # | Milestone | Status | Proof |
|---|---|---|---|
| 0 | Runtime skeleton: fixed-tick loop with NULL policy | ✅ | `run --policy null`, `test_runtime.py` |
| 1 | Minecraft adapter: observe + act | ✅ | `test_program.py` |
| 2 | Reward function: traces + summaries | ✅ | `test_rewards.py` |
| 3 | Baseline policies compared on same config | ✅ | `evaluate`, `test_policies.py` |
| 4 | Recorder + replay with verification | ✅ | `replay`, `test_runtime.py` |
| 5 | Human demonstrations as training data | ✅ | `demo` → `train` |
| 6 | Behavioral cloning beats random baseline | ✅ | `test_training.py` |

MVP completion criteria: a continuous runtime inhabits Minecraft, observes,
acts, receives rewards, records and replays episodes, compares baselines,
trains a first learned policy from demonstrations — and the architecture
supports another Program without rewriting the runtime.
