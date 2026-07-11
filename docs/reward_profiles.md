# Reward profiles: YAML tiers, milestones, normalized returns

Tracked by issue #41. A **reward profile** is a YAML/JSON document that
drives the Minecraft reward function entirely declaratively, instead of the
hard-coded Python in `programs/minecraft/rewards.py` (`SurvivalReward`,
still the default when no profile is given -- profiles are additive, not a
breaking change). Load one with:

```
ccr run --reward-profile goals/survival.yaml ...
ccr run --reward-profile goals/ender_dragon.yaml ...
```

A malformed profile fails immediately with a diagnosis naming the exact
field -- never mid-run.

`ccr replay --session ...` needs the exact same profile a session was
recorded with: pass it again via `--reward-profile` (`replay_session`'s
`reward_profile=` in code). Session metadata's `reward_profile` field
records `{name, content_hash, ...}`, so replay fails loudly -- rather than
silently re-scoring `reward.scalar` against the default `SurvivalReward` --
whenever a `--reward-profile` is missing or its content hash doesn't match
the session's recorded one.

## The compass: tiers

Components are grouped into named **tiers**. The recommended shape is a
three-tier "reward compass" plus one cross-cutting bucket:

- **`survival`** -- stay alive: tick-alive, death, damage, hunger, critical
  vitals.
- **`capability`** -- learn the world: exploration, item diversity, tool
  use, crafting, shelter/light/night.
- **`quest`** -- sparse milestones/events: village, Nether, dragon.
- **`shaping`** -- anti-stagnation penalties (repeated actions, idling,
  spinning, no novelty). Not reward *content*, so it isn't one of the three
  compass tiers, but it lives in the same profile.

Tier names beyond these four are accepted (the engine doesn't special-case
tier names, only component `kind`s), but dashboards and docs assume this
set.

## Component shape

```yaml
tiers:
  capability:
    new_item:
      kind: capped_novelty       # required: selects the evaluation rule
      value: 0.5                 # reward per trigger
      cap: 5.0                   # total cap across this component's scope
      decay: 1.0                 # geometric decay per repeat (decaying_repeat only)
      decay_floor: 0.0           # floor after decay
      cooldown_ticks: 0          # min ticks between triggers
      scope: life                # "life" (resets each episode) | "brain" (persists)
      disabled: false
      params:
        source: "event:new_item"
```

`value`, `cap`, `decay`, `cooldown_ticks` and `scope` are anti-farming
controls common to every kind. `params` carries kind-specific knobs.

### Known kinds

| kind | fires when | required `params` |
| --- | --- | --- |
| `tick` | every non-death tick | -- |
| `death` | the `died` event | -- |
| `event_count` | `params.event_prefix:*` events, `value * count` | `event_prefix` |
| `delta_decrease` | `params.field` (`health`/`hunger`) drops by whole points | `field` |
| `threshold_enter` | `params.field` crosses below `params.threshold` (edge-triggered) | `field`, `threshold` |
| `periodic_no_event` | every `params.window` ticks without `params.event_prefix`, gated on `params.min_field >= params.min_value` | `event_prefix`, `window`, `min_field`, `min_value` |
| `capped_novelty` | a new distinct key from `params.source` (`nearby_blocks`, `biome`, `position_chunk`, or `event:<prefix>`) | `source`, and a `cap` |
| `distance_ladder` | every `params.unit` of new max distance from spawn | `unit`, and a `cap` |
| `once_predicate` | the first key from `params.source` matching `params.predicate` (`is_tool`/`is_food`) | `source`, `predicate` |
| `once_event` | the first occurrence of the exact event `params.event` | `event` |
| `decaying_repeat` | every occurrence of a key from `params.source`, value decaying `value * decay^n` per repeat of that key, floored at `decay_floor` | `source` |
| `streak_penalty` | the same action repeated more than `params.threshold` ticks | `threshold` |
| `idle_penalty` | a NULL-action streak past `params.threshold`, unless threatened (`params.low_health_threshold`, default 10.0, or mobs visible) | `threshold` |
| `spinning_penalty` | the last `params.window` actions are all in `params.actions` | `window`, `actions` |
| `no_novelty_penalty` | `params.ticks` in a row with no new observation hash | `ticks` |

`capped_novelty` and `distance_ladder` require an explicit `cap` at load
time -- an uncapped novelty/distance component is very likely an authoring
mistake (unbounded reward farming).

## Milestone state: `scope: life` vs `scope: brain`

One-time components (`once_event`, `once_predicate`) and running totals
carry a `scope`:

- **`life`** (default) -- resets every episode. Fine for anything that
  should be re-earnable each life (e.g. "found shelter tonight").
- **`brain`** -- persists across episodes and across a checkpoint
  interrupt/resume. Use this for genuine one-time-ever milestones (quest
  progression: "entered the village", "defeated the dragon") so an
  interrupted training run does not re-grant them on resume.

`ProfileRewardEngine.state_dict()` / `.load_state_dict()` carry
`brain`-scoped state (plus the running-return normalizer); wire these into
a checkpoint bundle's `extra_metadata` alongside the neural checkpoint.
`load_state_dict()` refuses to load state saved under a different profile
(compared by content hash below), since milestone keys/semantics aren't
guaranteed compatible across profiles.

## Intrinsic component slots

A separate `intrinsic` section holds named, pluggable intrinsic-drive
components -- `learning_progress`, `safe_novelty`, `predicted_risk_aversion`
(the terms themselves are computed by issue #61's `core/modulation.py` from
the `internal.*` streams #58 publishes; this schema just supplies the slot
they plug into):

```yaml
intrinsic:
  learning_progress:
    stream: internal.learning_progress   # required: source stream id
    weight: 1.0                          # multiplier on the raw stream value
    cap: 1.0
    disabled: false
```

Each tick, the engine reads the *latest* published value of `stream`
(payload is either a bare number or `{"value": ...}`), multiplies by
`weight`, and applies the same `cap`/`disabled` controls as any other
component -- it never recomputes a signal itself. Deliberately **not**
supported here: raw world-model prediction error or raw prediction
accuracy as a reward. Raw error rewards irreducible noise and can walk an
agent off cliffs chasing unpredictable inputs; raw accuracy rewards
wall-staring at an already-predictable static scene.

### The three intrinsic terms (issue #61)

```
intrinsic =
    + w_lp   * learning_progress          # internal.learning_progress: error is IMPROVING here
    + w_nov  * safe_novelty                # internal.safe_novelty = novelty * risk_gate
    + w_risk * predicted_risk_aversion     # internal.predicted_risk_aversion = -risk
```

`cognitive_runtime/core/modulation.py` computes all three as `internal.*`
streams every tick, alongside the five raw signals #58 publishes:

- **`internal.learning_progress`** -- already published by #58
  (`LearningProgressTracker`): positive while prediction error is
  *improving*, near zero on both mastered scenes and irreducible noise (the
  noisy-TV problem self-extinguishes here, not via a cap).
- **`internal.risk_gate`** -- `safe_gate(risk) = sigmoid(-(risk -
  risk_threshold) / temperature)`: `1.0` well below `risk_threshold` (safe
  to be curious), `0.0` well above it ("surprising but painful is not
  curiosity, it is a warning"), `0.5` exactly at the threshold. Logged as
  its own stream mainly for dashboards/debugging the gate itself; a
  profile does not normally wire it into a reward component directly.
- **`internal.safe_novelty`** -- `internal.novelty * internal.risk_gate`:
  surprise sought only when it doesn't forecast suffering.
- **`internal.predicted_risk_aversion`** -- `-internal.risk`, already
  sign-flipped, so a *positive* `weight` on this slot (as in the shipped
  profiles) amplifies an aversive (negative) shaping term proportional to
  predicted risk -- avoidance happens before damage, not after.

`risk_threshold`/`temperature` are run-level config
(`RuntimeConfig.intrinsic_risk_threshold`/`intrinsic_risk_temperature`,
`--intrinsic-risk-threshold`/`--intrinsic-risk-temperature` on the CLI;
defaults `0.5`/`0.15`), recorded into session metadata's
`intrinsic_modulation` field alongside each intrinsic slot's
stream/weight/cap (`reward_profile.metadata()`'s `intrinsic` field) so #44's
harness can compare drives tuned differently across runs.

A profile can be intrinsic-only (nursery stages, issue #62) by leaving
`tiers` mostly or entirely empty (see `goals/intrinsic_only.yaml`), or
extrinsic-only by leaving `intrinsic` empty -- both sections are optional.

## Two-scale rewards: raw vs training value

Every `reward.scalar` stream event carries both:

- **`value`** / **`components`** -- the raw, unclipped magnitudes. Always
  logged; this is what episode totals and dashboards show, and what
  `core.learner.window_reward()` sums.
- **`training_value`** -- normalized and clipped, from the profile's
  `normalization` block. This is what an optimizer should actually consume
  (`core.learner.window_training_reward()`, used by the online-Q and
  actor-critic learners) -- a linear TD update already assumes small
  errors, so a raw `1_000_000.0` dragon-kill reward must never hit it
  directly.

```yaml
normalization:
  method: running     # "running" (Welford mean/std) | "none" (pass raw through, still clipped)
  clip: 5.0            # null disables clipping
  epsilon: 0.0001       # numerical floor added to the running std
  warmup_ticks: 100    # ticks before "running" normalization kicks in (raw+clip until then)
```

When no profile is loaded (the legacy `SurvivalReward` path),
`training_value` is simply set equal to `value` -- normalization is only
ever profile-driven.

## Profile identity: content hash

`RewardProfile.content_hash` is a stable SHA-1 over the profile's full
content (independent of file path). Session metadata (`session.json`'s
`"reward_profile"` field, when a profile is active) carries `{name,
content_hash, intrinsic}` (`intrinsic` spelling out each intrinsic slot's
stream/weight/cap/disabled), so `cognitive_runtime dashboard` groups runs by
*exact* profile content rather than by filename -- two files with the same
content hash the same; the same filename edited between runs hashes
differently.

## Shipped profiles

- `goals/survival.yaml` -- the default profile, reproducing the values
  `SurvivalRewardConfig` has always shipped (`survival` + `capability` +
  `shaping` tiers, no `quest` tier, no intrinsic slots). Equivalent to
  passing no `--reward-profile` at all.
- `goals/ender_dragon.yaml` -- the full compass example: survival +
  capability + a `quest` tier with brain-scoped milestones toward
  `defeated_dragon`, plus all three intrinsic slots wired to `internal.*`
  streams, plus a worked anti-farming example (`first_log` vs
  `repeated_common_item`). The quest tier names events the simulated
  backend does not emit yet (village/Nether/dragon progression); the
  components simply never fire until a backend does, which is harmless --
  the file documents the target schema shape for a richer backend.
- `goals/intrinsic_only.yaml` -- the nursery example (issue #61/#62): no
  `tiers` at all, just the three intrinsic slots. A newborn agent's only
  reward is curiosity -- want to predict the world, seek safe surprise,
  avoid predicted pain.

## Adding your own

Write a new `.yaml`/`.yml`/`.json` file (`reward_profile.load_reward_profile`
loads either), start from `goals/survival.yaml`, and run it with
`--reward-profile <path>`. If it's malformed, the CLI exits immediately with
the exact field that's wrong -- fix and retry; nothing partially runs.
