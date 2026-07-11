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

An automated **curriculum runner** (issue #43,
`cognitive_runtime/training/curriculum_runner.py`) promotes the agent between
staged goals only when a metric passes, carrying the same checkpoint bundle
across stage boundaries -- see "Curriculum runner" below. It gates on the
plain mean of one summary metric over a fixed eval sample size -- deliberately
the simplest thing that is still an N-episode aggregate, not a single-episode
fluke. Issue #44's full statistical harness
(`cognitive_runtime.training.statistical_evaluation`: confidence intervals
across survival/reward-by-tier/coverage/prediction-error, with
regression/improvement flagging) has landed and is available for richer
inspection of a curriculum run's recorded sessions
(`statistical-evaluate --from-sessions <record_dir>`,
`dashboard --statistical`), but the runner's own promotion gate intentionally
stays on the plain-mean criterion for now.

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

## Curriculum runner (issue #43)

The presets above are single-stage tuning: pick one, run it, look at the
metrics. The **curriculum runner**
(`cognitive_runtime/training/curriculum_runner.py`) chains stages together
unattended: train an actor/critic on a stage, evaluate it, and promote to the
next stage only when a metric clears a threshold -- holding (and logging why)
otherwise, never spinning silently. The same policy/critic/optimizer -- the
"brain" -- carries across every stage; only the world/reward config changes.

A **curriculum definition** is a YAML/JSON file, distinct from a
`CurriculumPreset` above (which only carries world/reward overrides, no
promotion logic):

```yaml
name: toy-two-stage
stages:
  - name: flat-safe-toy
    world_config: {world_size: 32, episode_ticks: 300, difficulty: 0.0, max_mobs: 0}
    reward_config: {tick_alive: 0.02}       # or reward_profile_path: goals/survival.yaml
    train_episodes: 2
    promotion: {metric: average_ticks, threshold: 250, sample_size: 2}
    max_attempts: 2
  - name: night-survival-toy
    world_config: {world_size: 32, episode_ticks: 300, day_length: 400, difficulty: 1.0, max_mobs: 2}
    reward_config: {survived_night: 2.0}
    train_episodes: 2
    promotion: {metric: survival_rate, threshold: 0.0, sample_size: 2}
    max_attempts: 2
```

- `world_config` / `reward_config` mirror `SurvivalBoxConfig`/`SurvivalRewardConfig`
  overrides, same as a `CurriculumPreset`. `reward_profile_path` (issue #41)
  is the alternative to `reward_config` -- the two are mutually exclusive per
  stage.
- `promotion.metric` is one of `average_reward`, `average_ticks`,
  `total_reward`, `total_ticks`, `survival_rate` (fraction of eval episodes
  that didn't end in death); `threshold` is compared against the *mean* over
  `sample_size` eval episodes.
- `max_attempts` is the demotion/plateau rule: a stage that hasn't met its
  promotion criteria after this many train+evaluate attempts holds instead of
  retrying forever.
- All stages must share the same `world_size` (and any other knob that
  reshapes the stream catalog) -- `load_curriculum_definition` checks this at
  load time, since a stream-layout change would make the checkpoint
  incompatible mid-curriculum.

Curriculum progress (current stage, attempts, promotion history) is stored in
the checkpoint bundle's `training_stats["curriculum"]` (issue #20), so
interrupting the runner and restarting it resumes at the correct stage from
the same checkpoint:

```bash
python -m cognitive_runtime curriculum-run \
    --curriculum-file goals/curricula/toy_two_stage.yaml \
    --checkpoint checkpoints/curriculum.pt

# Resumes automatically from the checkpoint's saved stage; --stage/--force-promote
# are manual overrides for experimentation:
python -m cognitive_runtime curriculum-run \
    --curriculum-file goals/curricula/toy_two_stage.yaml \
    --checkpoint checkpoints/curriculum.pt --stage 1 --force-promote
```

Each stage's train/eval episodes are tagged with the stage name and index in
session metadata (`session.json`'s `curriculum`/`curriculum_stage_index`
fields, alongside every plain `--curriculum` run's), so `dashboard` groups a
curriculum run's progress by stage.
