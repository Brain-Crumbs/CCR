# How to add or update the nursery

The nursery is CCR's suite of scripted micro-scenarios. Each scenario
isolates a regularity the world model should learn, records separate training
and held-out episodes, applies data-quality checks, trains or evaluates a
model, and writes diagnostic artifacts.

This guide is for contributors adding a scenario or changing the nursery
pipeline. For the broader V2 system context, see the
[onboarding guide](../v2/03-onboarding-guide.md).

## Understand the two nursery modes

The public commands serve different purposes:

| Command | What it does |
| --- | --- |
| `ccr nursery list --world <world>` | Lists the scenario registry for a world. |
| `ccr nursery run <scenario>` | Records train and held-out seeds for one scenario, trains a per-scenario visual model, and evaluates held-out prediction. |
| `ccr nursery run all` | Runs every scenario in the selected world's registry independently. |
| `ccr nursery joint` | Records multiple scenarios and trains one action-conditioned recurrent world model, with held-out-seed and zero-shot-scenario evaluation. |

The default world is Crafter. Minecraft scenarios remain available through
`--world minecraft`; its `simulated` backend is the reliable choice for
scripted scene setup. A scenario belongs to exactly one registry:
`CRAFTER_SCENARIOS` or `NURSERY_SCENARIOS` in
`cognitive_runtime/training/nursery.py`.

## Before you change a scenario

Write down the regularity and the evidence that would prove it was present in
the recording. A useful nursery scenario has:

- A narrow, observable causal pattern, such as ego-motion, a view change,
  occlusion and reappearance, or an entity approaching.
- A deterministic scripted policy and/or scene setup, parameterized by seed.
- Distinct training and holdout conditions. Held-out episodes must test the
  same regularity without being duplicates of training episodes.
- A quality expectation that catches the scenario becoming inert, truncated,
  or otherwise invalid.

Do not add a scenario just because it produces frames. A static or accidental
world variation cannot support a learning or generalization claim.

## Add a scenario

Define a builder that accepts `seed` and `NurseryConfig` and returns a
`ScenarioRecording`:

```python
from cognitive_runtime.core.action import Action
from cognitive_runtime.policies.constant_action import ConstantActionPolicy
from cognitive_runtime.training.nursery import (
    NurseryConfig,
    NurseryScenario,
    ScenarioRecording,
)


def _short_forward(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    return ScenarioRecording(
        policy=ConstantActionPolicy(Action("MOVE_FORWARD")),
        program_config_extra={"max_mobs": 0},
        episode_ticks=100,
    )
```

The builder may provide:

- `policy`: the action source used while recording.
- `program_config_extra`: scenario-specific world configuration merged with
  the resolved program configuration.
- `scene_setup`: an optional one-time hook to arrange terrain, entities, or
  other scripted state before the episode begins.
- `episode_ticks`: an optional scenario-specific length when the regularity
  needs a fixed cycle.

Register the scenario in the registry for its world, with a unique stable
name and a description that says what it isolates:

```python
CRAFTER_SCENARIOS["short_forward"] = NurseryScenario(
    "short_forward",
    "short constant forward movement over varied seeds -- ego-motion.",
    _short_forward,
    min_moving_transition_fraction=0.10,
    min_semantic_change_fraction=0.05,
)
```

For a Minecraft scene hook, type the argument as `MinecraftSurvivalBox`. For a
Crafter scene hook, type it as `CrafterWorld`. The recording code calls hooks
for Crafter and for Minecraft's `simulated` backend only; a remote Minecraft
server cannot be modified in-process. Do not register one scenario in both
registries unless each builder and its assumptions work in its target world.

## Configure the quality gate

`NurseryScenario` carries scenario-specific expectations that
`validate_nursery_recordings` applies before training. Set only conditions
that are necessary for the regularity:

| Field | Use it to require or limit |
| --- | --- |
| `min_blocks_per_tick` / `max_blocks_per_tick` | Net agent displacement. |
| `min_unique_frame_fraction` | Meaningful visual variation. |
| `min_yaw_sweep_degrees` | Continuous Minecraft view rotation. |
| `min_unique_facings` | Discrete Crafter facing variation. |
| `min_moving_transition_fraction` | Movement throughout the episode, not only at the start. |
| `min_semantic_change_fraction` | Real scene change rather than pixel flicker. |
| `max_longest_stationary_tail` / `max_longest_stationary_run` | A stuck final tail or long stationary period. |
| `require_completed` | Whether an early-terminated episode is invalid; this defaults to `True`. |

Start from the smallest meaningful threshold and validate it against both a
healthy recording and a known-bad one. Do not solve a failed gate by setting
`--skip-data-quality-gate` or loosening a threshold until the scenario's
intended signal is no longer protected.

The optional split-overlap gate checks whether held-out episodes duplicate
training episodes. Keep train and holdout seeds disjoint; the runner rejects
overlapping seed sets before it records anything.

## Keep the experiment split honest

`nursery run` turns the requested counts into train seeds `0..N-1` and
holdout seeds `N..N+M-1`. Scenario builders should use the seed to vary the
background, placement, trajectory, or other context without removing the
regularity under study.

For `nursery joint`, be deliberate about both axes of evaluation:

- Held-out seeds test in-distribution generalization within a trained
  scenario.
- `--holdout-scenarios` excludes whole scenarios from training and tests
  zero-shot generalization. The default holdout is `approach_entity`.

If a scenario has parameter ranges, partition them by split rather than
letting training and holdout draw the same scripted cases. The existing
occlusion and approach scenarios are examples of this pattern.

## Update a pipeline or model contract carefully

`NurseryConfig` is the shared contract for recording and per-scenario
training. Add a field there when a setting must be available to scenario
builders and runners; preserve a safe default so existing callers continue to
work. Thread a user-facing setting through the relevant CLI parser and the
`NurseryConfig` construction in `cognitive_runtime/cli.py`.

For a change that affects joint training, also review:

- `run_nursery_joint`, which records scenario groups, checks quality and split
  overlap, builds the action sequence dataset, trains, and evaluates.
- `ActionWorldModelConfig`, which owns recurrent model and objective settings.
- The checkpoint and report payloads, so downstream viewers and comparison
  tools can identify the configuration that produced a result.

Do not change the action vocabulary based on the actions a scenario happened
to use. The joint dataset intentionally uses the selected world's complete,
stable action space so checkpoints can be reused safely across stages.

## Run and inspect a small experiment

First confirm that the scenario is visible in the expected registry:

```bash
ccr nursery list --world crafter
```

Use a small local smoke run while iterating:

```bash
ccr nursery run short_forward \
  --world crafter \
  --record-dir runs/nursery-smoke \
  --train-seeds 2 \
  --holdout-seeds 1 \
  --episode-ticks 50 \
  --horizons 1 5 \
  --epochs 1 \
  --consistency-epochs 0
```

Then inspect the result rather than relying on process success alone:

- The data-quality gate passes for the intended reason.
- Train and holdout session metadata have `curriculum` set to
  `nursery/<scenario>`.
- Each requested horizon has model, copy-last, and mean-frame metrics.
- Held-out dream strips and exported prediction files are present when those
  artifacts are enabled.
- The trace (`ccr trace show`) shows recording, quality, dataset, training,
  and evaluation phases without a frozen rollout warning.

Remote Minecraft runs have important limitations: sessions share a persistent
server world, seeds do not vary terrain, and scripted scene hooks are skipped.
Use them only when that is the system under test and expect the quality gate to
reject recordings that do not contain the required signal.

## Test the change

Put scenario and runner coverage in `tests/test_nursery.py`; add world-specific
coverage in the relevant Crafter or Minecraft tests when needed. A scenario
test should use a small `NurseryConfig` and verify at least:

- Registration, description, and appropriate quality constraints.
- The expected number of train and held-out sessions.
- Disjoint splits and correct `nursery/<scenario>` metadata.
- Finite per-horizon metrics and any scenario-specific diagnostic.
- The failure path for an invalid recording or configuration when it is new.

Run the focused test file:

```bash
pytest tests/test_nursery.py
```

## Nursery change checklist

- Does the scenario isolate one observable regularity?
- Is it registered only for compatible worlds and backends?
- Do its seeds and scripted parameters make a genuine held-out split?
- Will the quality gate reject a stuck, static, or truncated recording?
- Are training, evaluation, report, checkpoint, and viewer expectations still
  aligned after the change?
- Does a small smoke run produce meaningful traces and artifacts?
- Does focused test coverage cover both the expected result and the important
  failure mode?
