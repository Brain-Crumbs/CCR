# V2 Overview — A Predictive Organism We Raise

> **Status:** design proposal for the `world-model-biological-redesign` branch.
> This document is the *why* and the *what*. The [architecture](01-architecture.md)
> is the *how*, and the [implementation plan](02-implementation-plan.md) is the
> *in what order*.

## One sentence

We are growing a small artificial animal: a system that continuously watches,
listens to, and acts inside a world, *predicts what its own senses will feel
next*, is surprised when it is wrong, and — while it sleeps — **dreams** those
surprises back into a slowly-improving model of the world it lives in.

## The thesis

Animal brains are, at their core, **prediction machines**. The dominant
computational theory of the cortex — *predictive processing* / *active
inference* under the *free energy principle* — says the brain is a hierarchical
generative model whose job is to minimise the difference between what it
predicted and what actually arrived (its *prediction error*). Perception is the
brain updating its model to match the world; action is the brain moving the
body so the world matches its model. Everything else — attention, emotion,
curiosity, fear, memory, sleep — is machinery in service of that loop.

This project takes that thesis literally and builds an organism around it:

```
        senses (T0)  ──►  predict senses at T+1, T+4, T+8  ──►  act
             ▲                        │                          │
             └──────────  the world answers  ◄──────────────────┘
                                      │
                          prediction error = surprise
                                      │
                    ┌─────────────────┼──────────────────┐
              bored (no error)   curious (error,      afraid (error,
              → seek reward       no predicted pain)   predicted pain)
                                  → gather info        → fight / flight
```

The organism records the world's actual future inputs, so **yesterday's
prediction becomes today's training target, for free, forever.** No labels, no
human in the loop — the world grades every guess. That is the engine that lets
a latent world-model bootstrap itself from raw experience.

### Why this is a redesign, not a new project

The current codebase (`cognitive_runtime/`) already grew ~80% of this biology
under engineering names. It already has a stream-native perception loop, a
multi-horizon world model with uncertainty, a "dopamine analog"
(`internal.reward_prediction_error`), a predicted-pain/risk signal, a budgeted
attention controller, a brainstem orienting reflex, a curiosity ("safe
surprise") drive, and a nursery of developmental micro-scenarios. It even calls
its prediction visualisations *dream strips*.

V2's job is therefore **not to invent the biology — it is to name it honestly,
finish the organs that are missing, and re-centre the whole thing on the loop
above.** See the [architecture](01-architecture.md) for the full old→new map.
The three organs that are genuinely missing today:

1. **Episodic memory (a hippocampus).** Today there is a replay *buffer* but no
   episodic store of "initial conditions" to dream from.
2. **Sleep & dreams (consolidation).** Today there is an async trainer; V2 makes
   it a *sleep cycle* that replays hippocampal seeds as generative dreams and
   consolidates them into the cortex without forgetting.
3. **Motor-from-prediction.** Today a separate policy head chooses actions; V2
   adds the on-analogy path where the *forecast itself* drives the motor system.

## The core loop, precisely

Every waking tick the organism:

1. **Senses.** Input streams arrive — pixels, audio, body/interoception, the
   motor command it just issued (efference copy). Actions taken at T0 are part
   of the input.
2. **Attends.** A thalamic gate decides, under a fixed compute budget, which
   senses get high-resolution processing this tick. A spike of novelty, threat,
   or reward captures focus bottom-up.
3. **Binds.** Attended senses are encoded and fused into one workspace state —
   the organism's momentary "percept."
4. **Predicts.** The cortex forecasts *the same-shaped inputs it expects* at
   several horizons (default T+1, T+4, T+8 — fully configurable), each with an
   uncertainty. Predictions are made in a latent space but always carry a
   decoder, so every horizon yields a frame/waveform you can look at.
5. **Feels.** Comparing forecast to reality produces the neuromodulators:
   *dopamine* (reward surprise), *acetylcholine* (where to focus / how much to
   trust each sense), and an *amygdala* threat estimate that can release
   *adrenaline*. These are recorded as ordinary streams — interoception is input
   too.
6. **Decides which mode it is in** (the heart of your rant):
   - **Reward-seeking** when nothing is surprising (bored) — exploit what it
     knows to chase reward.
   - **Information-gathering** when surprised *but not* predicting pain — orient
     toward the surprise and sample it to drive the error down (curiosity).
   - **Fight-or-flight** when surprised *and* predicting pain — reflexive
     avoidance overrides deliberation.
7. **Acts.** Its default (voluntary) action is the one that would fulfil its own
   forecast — active inference. Above that sits a stack of hardcoded **reflexes**
   (orienting toward salience, threat/withdrawal) that *override* the voluntary
   action when their stimulus fires, and — in the nursery — a direct **caregiver
   override**. Every tick records what it *meant* to do (voluntary) versus what
   its body actually did (reflex/override). NULL (do nothing) is always a real,
   recorded voluntary choice.
8. **Remembers.** The tick is written to the record and a sparse *seed* is
   handed to the hippocampus.

Then, periodically, it **sleeps**: it stops acting, replays hippocampal seeds as
generative dreams through the cortex, and consolidates them into slow cortical
weights — the mechanism that turns a day of episodes into lasting skill without
overwriting yesterday.

## Dreams (your instinct, made concrete)

> *"memories are, after all, just initial conditions going through a network
> which simulates the previous event."*

That is exactly a generative world-model rollout, and V2 elevates it to a named
subsystem. A **dream** is: take a stored initial condition (a hippocampal seed —
a latent state plus the action sequence that followed), and run the cortex
forward *with no live senses*, regenerating the experience. One mechanism, three
uses:

- **Recall** — replaying a seed *is* remembering the episode.
- **Consolidation** — training the slow cortex on dreamed trajectories is how it
  learns the general structure of many episodes without forgetting (generative
  replay, the standard defence against catastrophic forgetting).
- **Imagination training** — the motor system can practise inside dreams, the way
  DreamerV3 trains its actor entirely in imagined rollouts, so the body improves
  faster than real, expensive ticks allow.

Dreams are also the best diagnostic surface we have: a "dream strip" of
predicted-vs-actual frames at each horizon shows, at a glance, whether the world
model has learned the world is *lawful*.

## Two kinds of learning, always on

The organism **trains as it lives** — you asked for this explicitly, and it maps
cleanly onto biology's two memory systems ([Complementary Learning
Systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC9606815/)):

- **Wake (fast, hippocampal):** cheap online updates every tick and quick
  episodic encoding, running in the tick budget without ever stalling the loop.
- **Sleep (slow, cortical):** heavier gradient work on dreamed + real replay,
  run off the tick thread in a separate process, publishing new weights back to
  the waking body between ticks.

This is the async actor/learner split that already exists in the repo, re-framed
as a wake/sleep cycle — so it is a rename over working code, not a rebuild.

## How we raise it: development, not a training run

We don't train the whole thing at once; we **grow it like an infant**, one
capability at a time, validating before advancing — a developmental ladder the
existing curriculum runner already knows how to walk with gated promotion:

| Stage | Nickname | What it learns | Motor |
|---|---|---|---|
| 0 | **Gestation** | just *see and hear*, and habituate — learn sensory regularities and a calm baseline (don't be frightened by everything) | frozen |
| 1 | **Babbling** | its own body: that its actions cause sensory change (forward/inverse models) | random, overridden |
| 2 | **Crawling** | ego-motion, optical flow, view rotation (`walk_forward`, `turn_in_place`) | scripted / overridden |
| 3 | **Objects** | object permanence, affordances, approach & scale (`object_permanence`, `approach_entity`) | scripted / learned |
| 4 | **Foraging** | goal-directed, reward-seeking behaviour (the survival curriculum) | learned |
| 5 | **Speaking** | communication / language streams (later) | learned |

Each stage has explicit **milestones** (e.g. "the world model beats copy-last on
held-out seeds", "the latent linearly decodes yaw", "predicted-pain aversion
fires before any damage"), and the organism is not promoted until it passes
them. During early stages we **override the motor system directly** — motor
babbling and guided movement — instead of letting it steer, exactly as a
caregiver moves an infant's limbs.

## The organism has a name

Each organism instance is **named**, and that name is a configuration value used
as the model id and the prefix for every file it generates — checkpoints,
recorded sessions, dream exports, diagnostics. You raise *Pixel*, or *Mote*, or
*Sprout*; its whole life is stored under its name. (The name is cosmetic to the
architecture but load-bearing for provenance: one organism, one identity, one
paper trail of everything it ever saw, predicted, dreamed, and learned.)

## Where it lives (worlds)

The runtime stays **world-agnostic**: a *World* is any environment that
implements the interface, and the organism can inhabit any of them unchanged.
V2 raises the infant in a **fast, deterministic, pixel-native nursery world**
([Crafter / Craftax](https://arxiv.org/abs/2109.06780) — a pip-installable 2-D
"Minecraft" built for open-ended agent research, no server, no headless-GL
pain), and keeps **Minecraft via mineflayer** as the richer *graduation* world.
Both are just Worlds behind the same seam; adding a third requires no changes to
the brain. (This directly fixes the recording-quality problems documented in
`docs/nursery-turn-in-place-analysis.md`, which were mostly headless-render and
non-determinism artifacts of using live survival Minecraft as a nursery.)

## What you interact with: the clinic

Day-to-day, you work through a **Node/React front-end — a developmental
clinic**, not a CLI (a CLI still exists underneath). The clinic starts
**read-only**, reusing the existing pixel-horizon viewer, and shows:

- **Dream strips** — predicted vs actual senses at each horizon, per session.
- **Neuromodulator timelines** — dopamine, threat/adrenaline, acetylcholine,
  prediction error, and which of the three modes the organism was in, tick by
  tick (the organism's "EEG").
- **Attention / focus timelines** — what it looked at and why.
- **Developmental chart** — milestones passed per stage, per organism.
- **Data-quality gates** — is a recorded session even usable? (pixel provenance,
  motion floors, completed episodes — the checks the turn-in-place analysis
  asked for.)

It later grows *control*: launch and stop runs, drive scenarios, override the
motor during nursery stages, trigger sleep/consolidation, and promote stages.

## Goals of V2, restated as success criteria

The redesign has succeeded when:

- The organism **predicts its own future senses** at configurable horizons, in a
  latent space with a decoder, and the predictions are *viewable* and beat
  trivial baselines on held-out nursery seeds.
- It **trains continuously as it inhabits a world**, via an always-on wake/sleep
  cycle, without ever stalling the tick.
- **Dreams are real**: hippocampal seeds replay generatively and demonstrably
  consolidate skill and reduce forgetting.
- The **three modes** (reward-seeking / information-gathering / fight-or-flight)
  arise from the predict→surprise→(threat?) loop and visibly change behaviour.
- It is **raised developmentally**, stage by stage, with validated milestones and
  direct motor override in the nursery.
- Every organism has a **configurable name** that ids its model and all its
  files.
- You **operate it from a front-end** with rich diagnostics, not a CLI.
- **None of the brain knows a single Minecraft (or Crafter) concept** — swap the
  World and the same organism inhabits a new one.

## Reading order

1. **This overview** — the vision and the loop.
2. [**Architecture**](01-architecture.md) — the anatomy, the old→new naming map,
   the prediction/memory/neuromodulation/dream subsystems, the World seam, the
   front-end, and what is kept vs. built new.
3. [**Implementation plan**](02-implementation-plan.md) — the phased,
   file-by-file path from today's `cognitive_runtime/` to the V2 organism, with
   milestones and validation gates.

## Sources

- Free energy principle / predictive coding — [Springer overview](https://link.springer.com/chapter/10.1007/978-981-95-1327-7_14),
  [active inference & cortical architecture](https://neupsykey.com/active-inference-predictive-coding-and-cortical-architecture/)
- Complementary Learning Systems (hippocampus/neocortex, replay, consolidation) —
  [PMC review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9606815/)
- Dopamine as reward prediction error — [Schultz, J. Neurophysiol.](https://journals.physiology.org/doi/full/10.1152/jn.1998.80.1.1)
- Superior colliculus & orienting / dopamine gating — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5529105/)
- World models in ML — [DreamerV3-lineage](https://arxiv.org/pdf/2405.15083),
  [V-JEPA 2](https://arxiv.org/pdf/2506.09985)
