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
- **`RemoteMinecraftBackend` (shipped):** drives a **live Minecraft server**
  through a small mineflayer bridge over a line-delimited JSON protocol
  (`programs/minecraft/remote.py` + `bridge/mineflayer/`). It declares itself
  non-deterministic and snapshot-less, so everything above the adapter —
  streams, rewards, recording, training, realtime pacing — works unchanged.

The backend is selected with `--backend {simulated,remote}` on
`run`/`demo`/`evaluate` (default: `simulated`).

Starting conditions follow the plan: fixed seed, limited world boundary
(walled), survival mode, daytime start, controlled difficulty, short
episodes, optional spawn-near-resources. The goal is not to make Minecraft
hard yet — the goal is to make learning measurable.

### Real Minecraft via the mineflayer bridge

Full setup is in [`bridge/mineflayer/README.md`](../bridge/mineflayer/README.md).
The short version:

```bash
cd bridge/mineflayer && npm install          # Node ≥ 18
export CCR_MINECRAFT_HOST=localhost CCR_MINECRAFT_PORT=25565
python -m cognitive_runtime run --backend remote --realtime \
    --policy scripted --episodes 1 --episode-ticks 400 --record-frames
```

The runtime launches `node bridge/mineflayer/index.js` (override with
`CCR_MINECRAFT_BRIDGE_CMD`) and talks to it over stdio. The bridge maps the
action space onto mineflayer controls, builds the observation shape (mapping
Minecraft block/biome/item names into the SurvivalBox vocabulary), and
synthesizes the semantic event vocabulary — `damage:<reason>`,
`new_item:<item>`, `broke_block:<block>`, `placed_block`, `ate_food`,
`entered_shelter`, `survived_night`, `died` — that the stream publisher and
reward function already consume. The dormant reward rules (`first_tool`,
`created_light_source`) activate as soon as the bridge emits those events.

How the seam behaves for a live, non-deterministic world:

- **Capability flags.** `SurvivalBackend.deterministic` /
  `supports_snapshots` are both `False` for the remote backend. The adapter
  refuses `snapshot()`/`restore()` with a clear error, the session records
  `"deterministic": false`, and `replay` skips re-simulation with an
  explanation instead of reporting spurious divergence — the recording stays
  fully usable for `view` and `train`.
- **Realtime is the natural fit.** With `--realtime` the bridge advances one
  server tick per step while the runtime paces vision/body streams to
  wall-clock rates over bounded, overflow-counted queues; `dashboard` then
  reports realtime health for the session.

Testing without a server: the Python **fake bridge**
(`bridge/fake/sim_bridge.py`) speaks the same protocol backed by the
simulated world, so `tests/test_remote_backend.py` drives the entire remote
path and asserts it reproduces the in-process backend byte-for-byte. The
bridge's JavaScript is syntax-checked in CI; its live behaviour (block-mapping
fidelity, action timing) is what you tune against your server.

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

By default the policy input is the **fused latent state** (`--features
latent`): per-modality stream encoders → `TemporalFusion` → a fixed-width
vector, produced by the same code online and offline. The hand-written
Minecraft featurizer stays available (`--features handcrafted`) for A/B
comparison; on the night scenario both survive full episodes and beat random,
with the latent path within a comparable share of the handcrafted reward.

Current learning behavior remains offline behavioral cloning: record sessions,
then train a model from those recordings. Phase 1 introduces the pure
`OnlineQModel` checkpoint/update layer, but online updates are not wired into
the runtime loop until the online policy/learner phase.

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
