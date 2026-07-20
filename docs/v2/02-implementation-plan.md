# V2 Implementation Plan — From `cognitive_runtime/` to the Organism

> Companion to the [overview](00-overview.md) and [architecture](01-architecture.md).
> This is the ordered path: what to build, in what sequence, and the milestone
> that proves each phase before the next begins. The guiding rule, inherited from
> the current roadmap: **build the smallest living loop that measurably learns
> before widening the senses.**

## Principles for the migration

- **Rename behind shims, don't break the world.** Every rename lands as a new
  module plus a re-export from the old path, so nothing breaks mid-migration and
  tests keep passing. Delete old paths only once nothing imports them.
- **One organism, one checkpoint, one name.** The `name` config lands first and
  threads through everything, because provenance is easiest to add before there
  are files to rename.
- **Each phase ends at a green milestone.** No phase is "done" because code
  exists; it is done when its milestone metric passes on held-out data and shows
  up in the clinic.
- **The nursery world comes early.** Most recording-quality pain in the repo
  traces to using live survival Minecraft as a nursery; a clean deterministic
  pixel world removes that variable before we tune the hard parts.

---

## Phase 0 — Identity & scaffolding (small, unblocks everything)

**Goal:** the organism has a name, and the V2 package skeleton exists alongside
the old one.

- Add `OrganismConfig.name` (string id). Thread it into: checkpoint bundle
  metadata, session-id prefix, dream-export filenames, clinic session grouping.
  Default to a generated name; `--name Pixel` overrides. *(This is the whole of
  Phase 0 — cheap provenance that's easiest to add before there are files to
  rename.)*
- **Defer the rename to after Milestone 2.** The target package tree (`organism/`,
  `world/`, `senses/`, `motor/`, `brain/`, `sleep/`, `record/`, `development/`,
  `clinic/`) and `ARCHITECTURE_MAP.md` land **once the cortex is proven**, not now.
  Empty re-export namespaces are bikeshedding vocabulary before the science holds;
  the critique's sharpest process point is *don't rename ~100 files before the
  three missing organs exist.* Build Phases 1–2 in the current names, prove
  Milestone 2, then rename behind shims — or never.

**Milestone 0:** a run recorded as `Pixel-<session>`, its checkpoint carrying
`name: Pixel`, and every generated file discoverable by organism name.

---

## Phase 1 — The nursery World (Crafter/Craftax)

**Goal:** a fast, deterministic, pixel-native World to raise the infant in.

- Add `worlds/crafter/` implementing the World seam (`world/base.py`, née
  `core/program.py`): publish `vision.frame.pixels`, `body.*`, reward, and
  `motor.command`; consume efferents. Keep it behind the exact same
  catalog/stream-registry contract Minecraft uses, so the brain is unchanged.
- Wire an `--world crafter|minecraft` selector (CLI now, clinic later).
- Port the nursery scenarios to run in Crafter, where determinism and pixel
  provenance are free: `walk_forward`, `object_permanence`, `approach_entity`, and
  a discrete-facing `turn` (Crafter's facing is a discrete flip, not the continuous
  rotation `turn_in_place` assumed — re-scope, don't port the optical-flow premise).
- Bring the **data-quality gates** forward from `nursery-turn-in-place-analysis.md`
  (pixel provenance, motion floor, completed-episode, yaw-sweep) as a reusable
  `record/quality.py`.

**Milestone 1:** `walk_forward` (and a discrete-facing `turn`) recorded in Crafter
pass the data-quality gates deterministically, with genuine frame-to-frame motion.
The win here is **determinism, pixel provenance, speed, and clean translational
motion** — *not* perspective: Crafter is 2-D top-down like the old Minecraft
render, so ego-motion / optical-flow is explicitly out of scope until the
first-person graduation world (see overview).

---

## Phase 2 — The Predictive Cortex (the heart)

**Goal:** one action-conditioned, recurrent, decoded, multi-horizon world model
that serves every scenario — replacing the per-scenario tiny predictors.

- Promote `training/action_world_model.py` (already a recurrent, action-
  conditioned prototype) to `brain/cortex/predictive.py`. Merge in
  `neural/world_model.py`'s multi-horizon + uncertainty heads.
- **Latent + decoder:** learn on `z`, keep a decoder so every horizon decodes to
  the same input shape (viewable). Reconstruction is auxiliary; latent
  prediction + short-rollout scheduled sampling is primary. No 100-step backprop-
  through-composition (it selects for the identity).
- Horizons in **ticks**, stored with the checkpoint; default {1, 4, 8}, fully
  configurable per organism.
- **Temporal backbone is a choice — benchmark it.** GRU is the default; also try a
  **dilated temporal-conv or small transformer over a frame window** (parallel over
  time, multi-timescale in one forward pass) with a **context-length curriculum**
  (1 frame → 2 → k). Same interface, so it's an A/B, not a fork.
- Scoring gates wired in: `model/copy-last`, `model/oracle`, **frozen-rollout
  detector** — as structured report fields, not just logs.

**Milestone 2 (the pivotal proof):** on held-out Crafter seeds, the cortex beats
copy-last at every horizon on `walk_forward`, and *withholding the action stream
measurably hurts* `turn_in_place` (proving it actually uses actions). This is the
"promise becomes proof" milestone the current roadmap already names.

---

## Phase 3 — Neuromodulators, Amygdala & the Arbiter

**Goal:** the three modes are selected each tick by the switch over (surprise,
predicted pain) and visibly change behaviour.

- Rename `core/modulation.py` → `brain/neuromod/` with human-named signals
  (dopamine, acetylcholine, adrenaline) over the existing `internal.*` math. No
  new math for dopamine — it is already RPE.
- `brain/amygdala.py`: appraise the cortex's risk head into a fast threat/
  adrenaline release (existing `internal.risk` + `predicted_risk_aversion`).
- `brain/arbiter.py`: the three-mode state machine over (surprise, predicted
  pain). It gates attention breadth and which motor path/reflex wins. Mode is a
  recorded stream.
- Feed acetylcholine into the renamed **Thalamus** (`core/attention.py`) as a
  precision term.
- **Calibrate surprise; add hysteresis.** The arbiter is a 2×2 lookup, so it is
  only as good as its inputs: produce cortex uncertainty (ensemble or predicted-
  error head), **calibrate and report it** (reliability diagram / temperature
  scaling on the rolling holdout), and require a mode change to persist k ticks
  before it takes.

**Milestone 3 (the three-region test, generalised):** in a scripted scene with a
harmless surprise and a harmful one, the organism demonstrably enters
info-gathering for the first (orients toward it) and fight-or-flight for the
second (reflex overrides policy, adrenaline spikes), and reward-seeking when
bored — each visible in the arbiter-mode timeline.

---

## Phase 4 — Hippocampus & Dreams

**Goal:** episodic seeds exist, and dreaming from them works.

- `brain/hippocampus.py`: a fast, capacity-bounded episodic store of seeds
  `(z_t, action-sequence, dopamine/threat/novelty tags)`, prioritised by the
  neuromodulator tags. Built over `neural/replay_buffer.py`'s prioritisation.
- `sleep/dream.py`: `dream(seed, length)` — roll the cortex forward from a seed
  with senses off, replay-action or imagined-action, yielding decoded frames.
- Export dreams for the clinic (reuse `training/prediction_export.py`'s format).

**Milestone 4:** a dream launched from a stored seed regenerates the original
episode's frames to within the cortex's own T+h accuracy, and the clinic renders
the dream strip. Recall works.

---

## Phase 5 — Sleep as continuous consolidation

**Goal:** the organism trains as it lives, via a wake/sleep cycle, without
stalling the tick — and dreaming prevents forgetting.

- Re-frame `training/async_trainer.py` + `neural/weight_publisher.py` as `sleep/`:
  Wake takes cheap in-tick updates + episodic encoding; Sleep drains **real +
  dreamed** replay, takes heavy cortical steps, publishes weights back between
  ticks. The learned spine is the **world model** (self-supervised regression =
  stable); the motor only plans over it, so this loop has no bootstrapped-policy
  instability.
- **Phasic before concurrent.** Ship the simple schedule first — act, pause,
  consolidate, resume — which has no weight staleness. Only then enable the
  concurrent separate-process trainer; when you do, publish **EMA-averaged**
  weights with a **monotonic version stamp** so the actor can bound staleness.
- Add **generative replay** with the bootstrap guardrail: keep a **reservoir of
  real transitions** (never train on dreams alone) and **gate the dream fraction on
  model quality** (0% until the cortex beats copy-last on held-out; ramp with the
  ratio; cap ≈0.5). Interleaving dreamed old seeds with new experience is the
  forgetting defence; report a forgetting metric (does `walk_forward` accuracy
  survive learning `object_permanence`?).
- Micro-sleep during runs + long consolidation at session boundaries; both
  clinic-triggerable later.

**Milestone 5:** a continuous run learns a new scenario during wake+sleep while
retaining a previously-mastered one (forgetting metric stays within tolerance),
with zero missed-tick regression versus a no-sleep baseline.

---

## Phase 6 — The motor system (voluntary + reflex stack)

**Goal:** one voluntary path (planning over the world model) with a hardcoded
reflex stack overriding it, full predicted-vs-actuated tracking, and nursery
caregiver override.

- `motor/voluntary.py`: the default voluntary controller is **one-step planning
  (MPC) over the Predictive Cortex** — roll the cortex forward for each of
  Crafter's ~17 actions and pick the best-scoring predicted next-state. Nothing
  here learns; the cortex does. Keep three alternatives behind the same seam for
  A/B (`motor.voluntary = mpc | active | imagination | policy`): **active-inference
  decoding** (the T+1→encoder→motor inverse path), a **DreamerV3-style imagination
  actor** trained in dreams, and the existing **actor/critic policy head**
  (`policies/actor_critic.py` → `motor/policy.py`). MPC is the spine; the others
  are experiments.
- `motor/reflexes.py`: a configured set of hardcoded stimulus→action reflexes
  that override voluntary output by priority. Migrate `OrientingReflex` and the
  threat/withdrawal response here; move scripted survival behaviours in as
  configured reflexes. Trigger *stimuli* come from World-declared streams
  (localization/threat hints); the reflex *set + thresholds* are organism config
  (the genome).
- **Caregiver override**: a development-stage hook injecting motor commands
  directly (babbling / guided movement) at the top of the precedence stack.
- **Record the whole stack** every tick: voluntary action, reflex fired
  (which/why), override applied, final actuated action.
- Precedence enforced: `caregiver override > reflex > voluntary`; NULL stays a
  real gated voluntary choice.

**Milestone 6:** in the babbling stage, caregiver-overridden motor produces clean
forward/inverse-model data; on a locomotion+threat scenario, a reflex demonstrably
overrides the voluntary action when its stimulus fires, the predicted-vs-actuated
divergence is logged, and the clinic charts **reflex-activation rate** — the curve
expected to fall as the cortex learns to pre-empt its reflexes (reflex
integration).

---

## Phase 7 — Development (ontogeny) end to end

**Goal:** raise one organism from Gestation to Foraging with gated promotion.

- Generalise `training/curriculum_runner.py` into `development/`: each stage
  declares World+scenario, active senses, motor freedom
  (frozen/overridden/learned), active losses, and **milestone gates**.
- Encode the ladder: Gestation → Babbling → Crawling → Objects → Foraging (→
  Speaking later). One checkpoint carried across all stages.
- Promotion uses the milestone metrics from Phases 2–6, not a single scalar.

**Milestone 7:** a single named organism walks the ladder unattended through at
least Crawling, passing each stage's milestone, resumable from its checkpoint,
its whole life inspectable by name in the clinic.

---

## Phase 8 — The Clinic (front-end)

Runs partly in parallel from Phase 1 (read-only panels can land as soon as there
is data), but hardens here.

**8a — Read-only (target: usable by end of Phase 2):**

- Node/HTTP service over the Record; React app.
- Session browser by organism name; **dream strips** (reuse `viewer/`'s
  `pixel-horizon-viewer`); **EEG panel** (neuromodulators + prediction error +
  arbiter mode); attention/focus timeline; developmental chart; data-quality
  gate results (green/amber/red per session).

**8b — Control (after Phase 7):**

- Launch/stop runs, pick World + scenario, **override motor** during nursery,
  trigger and watch a sleep/consolidation phase, promote stages — all over a thin
  control API, never reaching into brain internals.

**Milestone 8:** you can run a full nursery session, watch its EEG and dream
strips live-ish, see its data-quality verdict, and (8b) trigger its
consolidation — without touching the CLI.

---

## Dependency order (at a glance)

```
Phase 0 (name/scaffold)
      │
      ▼
Phase 1 (Crafter nursery) ──────────────┐
      │                                  │
      ▼                                  ▼
Phase 2 (Predictive Cortex) ◄── the pivotal proof (M2)
      │                                  │
      ├───────────────┐                  │
      ▼               ▼                  ▼
Phase 3 (modes)   Phase 4 (dreams)   Phase 8a (read-only clinic)
      │               │
      └──────┬────────┘
             ▼
      Phase 5 (sleep/consolidation)
             │
             ▼
      Phase 6 (motor: voluntary + reflex stack)
             │
             ▼
      Phase 7 (development ladder)
             │
             ▼
      Phase 8b (clinic control)
```

Phases 3, 4, and 8a can proceed in parallel once Phase 2's cortex exists. Phase 5
needs 4 (dreams to replay). Everything above 7 assumes the ladder can carry one
checkpoint across stages, which 5 and 6 must land first.

## Testing & validation strategy

- **Keep the sim-nursery byte-exact replay smoke test** as a cheap plumbing check
  (it still catches publish-order/recorder regressions), but do not extend it to
  learning runs.
- **Milestone gates are the real tests**: each phase's milestone metric is a CI-
  runnable check on held-out seeds where feasible (Phases 1, 2, 5, 6), and a
  recorded-scenario assertion otherwise (Phases 3, 4).
- **Statistical evaluation** (`training/statistical_evaluation.py`) is the
  regression referee for anything that learns — CIs over N episodes, regression
  flagged when a candidate's interval clears the baseline on the worse side.
- **Data-quality gates run before training**, in the clinic and in CI, so no
  phase trains on a red session.

## What this plan deliberately defers

- **Speaking / language** (developmental Stage 5) — after Foraging works.
- **Hippocampal *retrieval*** (context-cued recall of a relevant past episode)
  is implemented as a guarded follow-up to Phase 4: cosine kNN over fused
  latent keys, surprise/provenance gating, and prepend-to-cortex context.
- **Neural attention** and **learned neuromodulator scoring** — the deterministic
  versions ship first; learned successors stay gated until there's data to learn
  from (unchanged from the current roadmap's stance).
- **Extra neurochemistry** (serotonin, explicit norepinephrine-arousal) — added
  only when a concrete behaviour needs it.
- **Full package-path migration** — the renames land behind shims; physically
  moving every file to the new tree is bookkeeping that can trail the working
  organism, tracked by `ARCHITECTURE_MAP.md`.

## First concrete step

Phase 0 + the start of Phase 1: land `OrganismConfig.name` and a stub
`worlds/crafter/` World that publishes a pixel stream and consumes a motor command
— the smallest thing that proves the seam holds for a second, cleaner world. The
rename scaffolding waits until after Milestone 2; everything else builds on the
cortex that Phase 2 grows in that world.
