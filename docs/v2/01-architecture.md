# V2 Architecture — The Anatomy of a Predictive Organism

> Companion to the [overview](00-overview.md) (the *why*) and the
> [implementation plan](02-implementation-plan.md) (the *when*). This document
> is the *how*: the organs, how signals flow between them, and how each maps
> onto the code that exists today.

## Design commitments (the answers this is built on)

1. **Re-architect + rename, don't rewrite.** Keep the proven substrate (streams,
   record, world-model, bridges, async trainer); rename it to the biology;
   build the three missing organs (hippocampus, sleep/dreams, motor-from-
   prediction).
2. **Predict in latent space, always carry a decoder.** Learning happens on
   compressed representations (robust, avoids the pixel identity-attractor
   documented in `nursery-turn-in-place-analysis.md`), but every horizon can be
   *decoded to the same input shape* so predictions stay viewable — your "output
   is the same shape as the input" requirement, satisfied without the fragility.
3. **Prediction is action-conditioned.** `z_{t+1} = f(z_t, a_t)`. A predictor
   that never sees the action it took cannot tell "I kept turning" from "I
   stopped." This is the single most important fix over today's per-scenario
   predictors.
4. **Two worlds, one brain.** A fast deterministic nursery world (Crafter/
   Craftax) and Minecraft (mineflayer) behind the same World seam.
5. **One learned voluntary path; hardcoded reflexes override it.** Voluntary
   action is the world-model forecast decoded to motor (active inference). Above
   it sits a configured, genetic **reflex stack** (orienting, threat/withdrawal,
   the former scripted behaviours) and, in the nursery, a **caregiver override**,
   with strict precedence. Every tick records voluntary-vs-actuated action, so
   reflex integration is measurable. (A policy head stays as an optional
   alternative *voluntary* controller for A/B — it is not a reflex.)
6. **Don't constrain for determinism; diagnose instead.** Byte-exact replay stays
   available for the sim nursery as a cheap plumbing check, but it is no longer a
   design constraint. Rich diagnostics replace it as the trust mechanism.
7. **Named organism.** One config value → model id → prefix for every generated
   file.

## The map: old name → V2 name

The rename is the spine of the redesign. Nothing in this column is invented; it
is the biological name for a mechanism already present (or a clearly-scoped new
organ). Module paths are targets, not a demand to move everything on day one —
the [implementation plan](02-implementation-plan.md) sequences it.

| Today (`cognitive_runtime/…`) | V2 organ | Biological role |
|---|---|---|
| the whole runtime | **the Organism** (`organism/`) | the animal; carries a configurable `name`/id |
| `core/program.py` (Program) | **World** (`world/`) | the *Umwelt* the organism inhabits |
| `programs/minecraft/` | **worlds/minecraft**, **worlds/crafter** | concrete habitats |
| sensory streams | **afferent senses** (`senses/`) | vision, audio, interoception (body) |
| motor streams | **efferents** (`motor/`) | motor nerves; carry efference copy back as input |
| `core/attention.py` (AttentionController) | **Thalamus** (`brain/thalamus.py`) | budgeted sensory gating into cortex |
| per-stream neural encoders | **sensory cortices** (`brain/cortex/sensory/`) | visual/auditory/somatosensory encoders |
| `core/streams/fusion.py` (TemporalFusion / LatentState) | **Workspace** (`brain/workspace.py`) | the bound momentary percept (global workspace) |
| `neural/world_model.py` (MultiHorizon…) | **Predictive Cortex** (`brain/cortex/predictive.py`) | the generative world model; multi-horizon forecaster |
| `core/memory.py` TemporalBuffer | **Working memory** (`brain/working_memory.py`) | seconds-scale recent events |
| *(new)* | **Hippocampus** (`brain/hippocampus.py`) | fast episodic store of dream *seeds*; retrieval |
| `neural/replay_buffer.py` + `training/async_trainer.py` | **Sleep** (`sleep/`) | consolidation via dreamed + real replay |
| *(new, generative rollout)* | **Dreams** (`sleep/dream.py`) | seed → cortex rollout with senses off |
| `core/modulation.py` (`internal.*`) | **Neuromodulators** (`brain/neuromod/`) | dopamine, threat/adrenaline, acetylcholine |
| `internal.risk` + predicted-pain | **Amygdala** (`brain/amygdala.py`) | threat appraisal → adrenaline release |
| *(the state machine your rant describes)* | **Arbiter** (`brain/arbiter.py`) | picks reward-seeking / info-gathering / fight-or-flight |
| *(new, on-analogy path)* | **Voluntary motor** (`motor/voluntary.py`) | learned; decode the forecast into the action that fulfils it (active inference) |
| `policies/actor_critic.py` | **alt. voluntary controller** (`motor/policy.py`) | optional learned policy head, kept for A/B against voluntary motor |
| `policies/scripted.py`, `core/orienting_reflex.py` | **Reflexes** (`motor/reflexes.py`) | hardcoded genetic stimulus→action overrides (colliculus/orienting, threat/withdrawal) |
| NULL action / go–no-go | **basal-ganglia gate** (in `motor/`) | inaction as a real, gated *voluntary* choice |
| `training/curriculum_runner.py` + nursery | **Development** (`development/`) | staged ontogeny with gated promotion |
| `tools/`, `viewer/` | **Clinic** (`clinic/` server + web) | diagnostics & control front-end |
| recorder / record format | **the Record** (`record/`) | the organism's life history |

> Naming is a proposal, not a religion. If "Predictive Cortex" reads better as
> "Neocortex," or "Workspace" as "Percept," rename freely — the *structure*
> below is the commitment, the labels are ergonomics.

## The waking data flow

```
        ┌──────────────────────────── the World (Umwelt) ───────────────────────────┐
        │   Crafter / Minecraft: publishes afferent senses, consumes efferents       │
        └───────────────┬───────────────────────────────────────────▲───────────────┘
                        │ senses (pixels, audio, body, efference copy)│ motor command
                        ▼                                             │
                   ┌─────────┐   budget    ┌──────────────┐           │
    afferents ────►│ Thalamus│────────────►│ sensory      │           │
                   │ (gate)  │  which sense │ cortices     │           │
                   └────┬────┘  this tick   └──────┬───────┘           │
                        │                          │ per-sense latents │
                        │                          ▼                   │
                        │                    ┌───────────┐             │
                        │                    │ Workspace │  bound percept z_t
                        │                    └─────┬─────┘             │
                        │                          │                   │
                        │         ┌────────────────┼───────────────┐   │
                        │         ▼                ▼               ▼   │
                        │  ┌────────────┐   ┌───────────┐   ┌──────────┴─────┐
                        │  │ Predictive │   │ Amygdala  │   │ Motor:         │
                        │  │ Cortex     │   │ (threat)  │   │ voluntary      │
                        │  │ ẑ_{t+1,4,8}│   └─────┬─────┘   │ (active-inf)   │
                        │  │ + decode   │         │         │ ◄ reflex stack │
                        │  └─────┬──────┘         │         │   overrides    │
                        │        │ pred. error    │ threat  └──────┬─────────┘
                        │        ▼                ▼                │
                        │   ┌──────────────────────────┐          │
                        │   │ Neuromodulators           │          │
                        │   │ dopamine / ACh / adrenaline│         │
                        │   └────────────┬─────────────┘          │
                        │                ▼                         │
                        │           ┌─────────┐                    │
                        └──────────►│ Arbiter │  mode ─────────────┘
                                    │ (3 modes)│  gates motor & attention
                                    └────┬─────┘
                                         ▼
                                    the Record  ──►  Hippocampus (episodic seed)
```

Every arrow in that diagram is a **stream** — including the neuromodulators and
the arbiter's chosen mode. Interoception is input; the organism can attend to
its own dopamine the way it attends to vision. Nothing here is Minecraft-aware.

## The prediction core (the heart)

### Shape of the forecast

At each waking tick the **Predictive Cortex** consumes the workspace latent
`z_t` and the action `a_t`, and emits, for each configured horizon `h`
(default {1, 4, 8}, arbitrary and per-organism):

```
ẑ_{t+h}, decode(ẑ_{t+h}) → same-shape sense,  σ_{t+h} (uncertainty),
r̂_{t+h} (reward),  d̂_{t+h} (terminal/death),  risk_{t+h} (predicted pain)
```

- **Latent + decoder (commitment 2).** `ẑ` is the learning target; `decode(ẑ)` is
  the viewable frame/waveform. Losses live mostly in latent space (stable);
  reconstruction is an auxiliary loss and the diagnostic surface.
- **Action-conditioned + recurrent (commitment 3).** `f` carries a recurrent
  state so a single frame need not be a Markov state — the fix the turn-in-place
  analysis prescribed (`training/action_world_model.py` already prototypes this;
  V2 promotes it to *the* cortex).
- **Multi-horizon with uncertainty.** Per-horizon heads (not 100-step backprop-
  through-composition, which selects for the identity) plus short-rollout
  scheduled sampling. `neural/world_model.py`'s `MultiHorizonMLPWorldModel` is
  the seed; it gains the recurrent, action-conditioned, decoded body.
- **Horizons are counted in ticks/seconds**, stored with the checkpoint, so
  "T+8" means the same thing across worlds and sample rates.

### The training signal is free

The world records its actual future. So the target for the T+h forecast made at
tick `t` is simply the real workspace latent (and real sense) at tick `t+h`,
already on disk. Self-supervised, unlimited, unlabelled. This is the engine.

### Scoring (honesty gates)

Never raw MSE alone. Every run reports, per horizon:
`MSE(model) / MSE(copy-last)` and, for periodic scenarios,
`MSE(model) / MSE(period-oracle)`; plus a **frozen-rollout detector** (predicted
frames identical across horizons while actuals differ ⇒ flag red). These already
exist in `training/` and become first-class clinic diagnostics.

## Memory: three timescales

Straight [Complementary Learning Systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC9606815/):

| Organ | Timescale | Holds | Learns |
|---|---|---|---|
| **Working memory** | seconds | recent stream events (the `TemporalBuffer`) | nothing; a window |
| **Hippocampus** *(new)* | a session / a "day" | sparse **episodic seeds**: `(z_t, action-sequence, surprise/reward tags)` | fast, one-shot, pattern-separated |
| **Predictive Cortex** | the organism's life | the generative model weights | slow, general, over many dreams |

The **hippocampus is the missing organ**. It stores compact seeds prioritised by
surprise, reward, threat, and novelty (the priorities already computed as
`internal.*`), and it is what **dreams are launched from**. Retrieval (bringing
back a relevant past episode when the present resembles it) is a later capability
(today's `neural.replay_buffer` prioritisation is the starting point).

## Sleep & Dreams

### A dream

```
dream(seed, length):
    z ← seed.z                         # an initial condition
    for k in range(length):
        a ← seed.actions[k]  OR  motor.imagined(z)   # replay, or imagine
        z ← PredictiveCortex.f(z, a)   # roll forward, senses OFF
        yield decode(z)                # the regenerated experience
```

A dream is a generative rollout of the cortex from a hippocampal seed with **no
live senses**. That is simultaneously recall, the substrate for consolidation,
and the arena for imagination training.

### The sleep cycle (continuous training, off the tick thread)

This is the existing async actor/learner split, re-framed and given the dream
mechanism:

- **Wake** (the tick thread): act, encode episodes into the hippocampus, take
  *cheap* online updates that fit the tick budget. Never blocks.
- **Sleep** (a separate process): drain a mix of **real** replay and **dreamed**
  trajectories, take the heavy cortical gradient steps (world-model +,
  optionally, imagination-based motor learning), and **publish new weights** back
  to the waking body between ticks (`neural/weight_publisher.py`).

Sleep can be *periodic micro-sleeps* during a run or a *long consolidation* at a
session boundary. Generative replay (dreaming old seeds while learning new ones)
is the defence against catastrophic forgetting — the organism doesn't forget how
to crawl when it learns to forage. The clinic can trigger, watch, and inspect a
sleep phase (dream strips of what it dreamed, loss curves, forgetting metrics).

## Neuromodulation & the three modes

### The chemicals (grounded, behaviour-changing — commitment from Q7)

Start with the handful that each *do* something, all published as `internal.*`
(now human-named) streams:

- **Dopamine** = reward prediction error (`internal.reward_prediction_error`
  today). Tags memories for replay priority and gears learning rate.
- **Acetylcholine** = attention/precision: how much to trust each sense and how
  sharply to focus. Feeds the thalamus. (Derived from expected uncertainty and
  learning-progress signals already computed.)
- **Amygdala → adrenaline (noradrenaline)** = threat: the predicted-pain/risk
  head (`internal.risk`, `internal.predicted_risk_aversion`) appraised into a
  fast fight-or-flight release that can pre-empt deliberation and gate reflexes.

(Serotonin/patience, explicit norepinephrine-arousal etc. are deferred until a
concrete behaviour needs them — no chemistry cosplay.)

### The Arbiter — your rant as a state machine

The organism is always in exactly one mode, chosen from two scalars it already
computes each tick — **surprise** (prediction error) and **predicted pain**
(amygdala/risk):

```
                    predicted pain?
                 no                 yes
            ┌───────────────┬───────────────────┐
 surprise   │  INFO-GATHER  │   FIGHT / FLIGHT   │  high
  high      │  (curious)    │   (afraid)         │
            │  orient &     │   reflex overrides │
            │  sample to    │   deliberation;    │
            │  reduce error │   adrenaline       │
            ├───────────────┼───────────────────┤
 surprise   │ REWARD-SEEK   │   (wary reward-    │  low
  low       │ (bored)       │    seek; caution)  │
            │ exploit for   │                    │
            │ reward        │                    │
            └───────────────┴───────────────────┘
```

- **Reward-seeking** (not surprised): the policy/active-inference motor pursues
  reward; attention is broad and cheap.
- **Information-gathering** (surprised, safe): the colliculus orients toward the
  surprising, localizable stimulus and the motor system samples it; the intrinsic
  "safe surprise" drive (`internal.safe_novelty`) rewards driving the error down.
  This is your "point the head at the loud-but-harmless boom."
- **Fight-or-flight** (surprised, threatened): adrenaline releases a fast avoidance
  reflex that **overrides** the policy (the existing reflex-veto precedence:
  never suppress fleeing/eating). This is your "duck, cover, run."

The arbiter's chosen mode is a recorded stream and a headline diagnostic — you
can watch the organism flip from bored to curious to afraid as the world
surprises it, which is exactly the behaviour you already saw error-spike in the
nursery runs.

## The motor system: one voluntary path, a reflex stack over it (commitment 5)

Motor output is a **precedence stack**, not competing brains. From lowest
authority to highest:

1. **Voluntary action (cortical, learned).** The default. The Predictive
   Cortex's forecast of "what I expect to sense next" is decoded into the action
   that would bring it about — active inference, the T+1-output→encoder→motor
   path you described (`motor/voluntary.py`). This is the *only* motor path that
   learns. An actor/critic **policy head** (`motor/policy.py`, today's
   `actor_critic.py`) remains available as an *alternative* voluntary controller
   for A/B — but it is voluntary, not a reflex.

2. **Reflexes (subcortical, hardcoded — the "genome").** A configured set of
   stimulus→action rules that **override** the voluntary action when their
   trigger fires. Reflexes do not learn; they are the organism's genetic priors.
   The *stimuli available* to trigger them are declared at the **World interface**
   (a World can advertise a localizable threat, a looming object, a damage
   event); *which reflexes this organism has, and their thresholds/priorities* is
   **model configuration** — the genotype/phenotype split. This is exactly what
   the old scripted/hardcoded policies become: the `OrientingReflex` (orient
   toward salience) and the threat/withdrawal response are the first two, and
   scripted survival behaviours migrate here as configured reflexes rather than
   "agent intelligence."

3. **Caregiver override (external).** During nursery stages the experimenter
   injects motor commands directly (babbling, guided movement). This is the same
   override slot as a reflex, driven from outside — the top of the stack.

Precedence: `caregiver override > reflex (by priority) > voluntary`. NULL
(inaction) remains a real, gated *voluntary* choice (basal-ganglia gate).

**Every tick records the whole stack, not just the outcome:** the voluntary
(predicted) action, which reflex fired and why (if any), whether a caregiver
override applied, and the final actuated action. So *predicted action vs.
reflexive/overridden action* is always reconstructable — the efference signal of
"what I meant to do vs. what my body did."

The action space stays World-defined and opaque to the brain
(`core/action_registry.py`'s world-changing/information-gathering classification
carries over, so reflexes and the arbiter can reason about an action's *kind*
without knowing what it *is*).

### Why tracking predicted-vs-reflex matters: reflex integration

That record is also a **teaching signal**, and it yields a genuinely biological
developmental curve. Human infant reflexes (grasp, rooting, Moro) are hardcoded
and then *integrated* — voluntary cortical control gradually learns to reproduce
and finally supersede them, and the raw reflex fades. Because the organism logs
the reflex action next to its own predicted action, the cortex can learn to
predict/pre-empt what its reflexes do; over development the voluntary path
internalises the reflex and the hardcoded override fires less often. The clinic
charts this directly as **reflex-activation rate falling** with maturity — a
concrete, on-analogy milestone rather than a metaphor.

## The World seam (unchanged principle, renamed)

A **World** publishes afferent senses and consumes efferents; that's the entire
contract (`world/base.py`, née `core/program.py`). The brain never learns a
World's vocabulary. V2 ships two:

- **Crafter/Craftax** (`worlds/crafter/`) — the **nursery**: 2-D pixel
  Minecraft, pip-installable, fast, deterministic; no server, no headless-GL.
  Fixes the recording-quality problems the turn-in-place analysis traced to live
  survival Minecraft. Scenarios (babbling, crawling, object permanence) are set
  up here.
- **Minecraft** (`worlds/minecraft/`) — the **graduation** world: the existing
  simulated backend + mineflayer bridge, untouched behind the renamed seam.

Adding a robot sim, a browser, or the future AI-OS is "write a World"; the brain
doesn't change. That is the whole point, preserved.

## Development (ontogeny)

The staged ladder from the [overview](00-overview.md) is driven by the existing
gated curriculum runner, generalised so a stage promotes only when its
milestones pass. Each stage declares: which World + scenario, which senses are
active, whether motor is frozen/overridden/learned, which losses are on, and its
promotion milestones. The organism carries **one checkpoint** across every stage
— the same brain grows up; only its world and its freedoms change.

## The Clinic (front-end)

A **Node/React app** backed by a small Node/HTTP service over the Record and the
organism's control plane. Reuses the existing `viewer/` pixel-horizon component.

**Read-only first (V1):**

- Session browser scoped by **organism name**.
- **Dream strips** (predicted vs actual per horizon) — the existing component.
- **EEG panel**: neuromodulator + prediction-error + arbiter-mode timelines.
- **Attention/focus timeline** with per-stream reasons.
- **Developmental chart**: milestones per stage per organism.
- **Data-quality gates**: pixel provenance, motion floors, completed-episode,
  frozen-rollout flags — a session is green/amber/red before you ever train on
  it.

**Control later (V2):** launch/stop runs, drive scenarios, **override motor**
during nursery, trigger a sleep/consolidation phase and watch it, promote stages.

The clinic talks to the brain only through the Record and a thin control API, so
it stays as World-agnostic as the brain.

## Diagnostics as the trust mechanism (replacing determinism)

Because we no longer constrain for byte-exact replay (commitment 6), trust comes
from **observability**. Every run is judged by: statistical evaluation over N
episodes (CIs, regression flags — already in `training/statistical_evaluation.py`),
the scoring gates above, the frozen-rollout detector, data-quality gates, and the
milestone checks. Byte-exact replay stays available for the deterministic sim
nursery as a cheap plumbing smoke test — useful, no longer load-bearing.

## What is kept, renamed, and newly built

**Kept as-is (renamed only):** stream primitives & buses, record/replay,
temporal fusion, multi-horizon world model, attention controller, orienting
reflex, intrinsic drive, curriculum runner, reward profiles, mineflayer bridge,
simulated backend, async trainer, statistical evaluation, the pixel-horizon
viewer.

**Renamed + extended:** world model → recurrent, action-conditioned, decoded
Predictive Cortex; async trainer → Sleep with dreams; modulation → named
Neuromodulators + Amygdala; attention → Thalamus fed by acetylcholine; scripted
policies + orienting reflex → the hardcoded **reflex stack**; actor/critic → an
optional alternative voluntary controller.

**Newly built:** Hippocampus (episodic seed store), Dreams (generative rollout
subsystem), the Arbiter (three-mode state machine), the Crafter nursery World,
the voluntary (active-inference) motor path + the reflex-override stack with
predicted-vs-actuated tracking, and the Clinic front-end.

See the [implementation plan](02-implementation-plan.md) for the order.
