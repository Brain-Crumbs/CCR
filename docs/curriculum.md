# Curriculum: staged world configs and reward goals

Part of the neural-stream-agent target ("Make Minecraft The Development
Nursery", `docs/neural-stream-agent.md`) and tracked by issue #30. A
**curriculum preset** (`cognitive_runtime/programs/minecraft/curriculum.py`)
is a named, documented bundle of:

- **World config** — `SurvivalBoxConfig` overrides (difficulty, `max_mobs`,
  `day_length`, `start_time`, `episode_ticks`, `world_size`).
- **Reward weights** — `SurvivalRewardConfig` overrides, applied over the
  defaults (`programs/minecraft/rewards.py`).
- **A fixed default seed** — so the preset gives the same episode content
  run to run, for regression comparisons across code/model changes.

Select one with `--curriculum <name>` on `run`/`demo`/`evaluate`; any world
flag you also pass explicitly (`--seed`, `--world-size`, `--difficulty`, ...)
still overrides that one value from the preset (see `_resolve_world_args` in
`cognitive_runtime/cli.py`). The chosen name is recorded into session
metadata (`session.json`'s `"curriculum"` field) and each episode's summary,
so `cognitive_runtime dashboard` groups runs by curriculum step as well as
policy.

Presets only tune existing world/reward knobs — no policy or heuristic
behavior is encoded here; rewards state goals, never actions to take.

## Stage zero: the nursery scenario suite (planned)

Below the survival curriculum sits a planned "infant" stage (issue #62):
scripted micro-scenarios — `walk_forward`, `turn_in_place`,
`object_permanence`, `day_night`, `approach_entity` — that each isolate one
worldly regularity, generate clean recorded sessions, and benchmark the
world model's multi-horizon prediction (t+1, t+5, t+20) on held-out seeds
against copy-last-frame/mean-frame baselines. No reward learning happens
here; the point is that the world model learns the world is *lawful*
(ego-motion produces optical flow, hidden objects persist, night follows
day) before the policy is asked to survive in it. Nursery checkpoints seed
step 1 below.

Also planned: an automated **curriculum runner** (issue #43) that promotes
the agent between the steps below only when statistical metrics pass
(issue #44), carrying the same checkpoint bundle across stage boundaries.

## The six steps

| Step | Name | World | Reward emphasis | Seed |
|---|---|---|---|---|
| 1 | `flat-safe` | no mobs, day never ends, small world | survival basics (`tick_alive`, `health_maintained`) | 100 |
| 2 | `resource-world` | no mobs, full-size world | block/item novelty (`new_block_type`, `new_item`) | 200 |
| 3 | `night-survival` | short day/night cycle, mobs on | shelter/light/night (`shelter`, `light_source`, `survived_night`) | 300 |
| 4 | `caves` | no mobs (no literal caves yet, see below) | mining + tool use (`new_block_type`, `tool_used_item`) | 400 |
| 5 | `combat` | night at spawn, more mobs, harder | damage avoidance + tool use (`damage_taken`, `tool_used_item`) | 500 |
| 6 | `crafting` | no mobs, longer episode | crafting chain (`craft_progress`, `first_tool`) | 600 |

Full field-level values live in `CURRICULA` (`curriculum.py`) — the table
above is the summary; read the module for exact numbers and descriptions.

**Caves is an approximation.** The simulated world (`world.py`) has no
literal underground/cave generation. Per issue #30's out-of-scope note
("caves/combat parity in the remote Mineflayer backend can be follow-ups if
the simulated world needs new mechanics"), the `caves` preset instead turns
off mobs and emphasizes the mining/tool-use reward signal on the existing
stone/coal_ore terrain features. A literal cave biome is future work.

## Running a preset

```bash
# Step 1, sanity-check with the random policy (per-preset acceptance run):
python -m cognitive_runtime run --curriculum flat-safe --policy random \
    --episodes 1 --record-dir sessions

# Step 3, with the scripted baseline, recorded and later replayable:
python -m cognitive_runtime run --curriculum night-survival --policy scripted \
    --episodes 3 --record-dir sessions

# Compare policies on an identical curriculum episode set:
python -m cognitive_runtime evaluate --curriculum caves \
    --policies null,random,scripted --episodes 3

# Group dashboard metrics by curriculum step:
python -m cognitive_runtime dashboard --record-dir sessions
```

Each preset's `--policy random` run above is the regression check from the
issue's acceptance criteria: same curriculum name + seed reproduces the same
episode content, so a session recorded today is directly comparable to one
recorded after a later change (`replay --session ... --verify` also confirms
byte-identical re-simulation for the deterministic simulated backend).

## Reward goals filled in for the curriculum (issue #30)

The curriculum needed reward-goal gaps filled first; see the reward table in
[`docs/minecraft-mvp.md`](minecraft-mvp.md#reward-design-programsminecraftrewardspy)
for the full list. In summary:

- **Exploration — new-chunk/new-cell visitation** (`new_chunk`): rewards
  covering new *area* (grouped into `chunk_size`-wide cells), independent of
  `new_block_type` (new terrain) or `distance` (max distance from spawn).
- **Tool use** (`tool_used_item`): rewards actually swinging an equipped
  tool/weapon, once per distinct type — distinct from `first_tool`, which
  rewards merely acquiring one.
- **Light placement** (`light_source`, pre-existing but dormant): completed
  by giving the simulated world a `torch` item, craftable at a furnace from
  coal alone (`smelt_torch`) and placeable like any other placeable item;
  placing one now emits `event.created_light_source`.
- **Crafting progress** (`craft_progress`): rewards each distinct recipe
  crafted (`event.crafted`'s `recipe` id), capped — so the gather → craft →
  smelt chain has its own signal independent of item pickup.

`first_tool` also stopped being dormant: the crafting table now has a second
recipe, `planks_to_pickaxe` (3 planks → 1 `wooden_pickaxe`), tried once
`log_to_planks`'s log input runs out — see `RECIPES` in `world.py`.

All four new/completed components are unit-tested against synthetic stream
events in `tests/test_rewards.py`, and the underlying world mechanics
(pickaxe/torch crafting, the `used_tool`/`created_light_source` events) are
covered end to end in `tests/test_program_streams.py`.
