# V2 Organism Audit — Did we build the animal we designed?

> **Reviewer's frame.** Read as a research audit, not a QA pass. The question
> is not "does the code run" (it does; 43/43 of the cortex + forgetting tests
> pass here). The question is whether the *thing described in
> [`00-overview.md`](00-overview.md)* — a single organism that continuously
> predicts its own next senses, is surprised, dreams, and consolidates without
> forgetting — actually exists as one running system. This document answers the
> five questions in the review brief, with file:line evidence.
>
> **Environment note.** Analysis was done with `torch 2.13`, `numpy`,
> `namesgenerator` installed on demand; `crafter` could not be installed
> (wheel host blocked by the proxy), so no live Crafter roll was executed. All
> claims below are grounded in the source and in the unit tests that *did* run.

---

## TL;DR

| # | Question | Verdict |
|---|---|---|
| 1 | Implemented in the spirit of the objective? | **Organs: yes. Organism: no.** Every subsystem (cortex, arbiter, amygdala, neuromodulators, hippocampus, dreams, generative replay, motor stack, ladder, clinic) exists and is unit-tested. But they are **not assembled into one living loop**: the running runtime still predicts with a trivial `TrendWorldModel`, and the Predictive Cortex is an *offline, batch-trained artifact* that never drives a live tick. |
| 2 | Legacy to remove | The **live actor-critic RL spine** (`sleep/async_trainer.py`, `neural/policy.py`, `neural/value.py`, `neural/world_model.py`, `policies/actor_critic.py`) contradicts the V2 thesis and should stop being the online learner. The **Minecraft survival economy** (`programs/minecraft/`, ~4,700 LOC of crafting/hunger/reward-engine) and the **reward-profile system** are nursery-era baggage. But **sequencing matters** — the live loop currently *depends* on this stack, so it can't be deleted until the cortex is wired in as the live world model (see #1). |
| 3 | Major bugs / training-feasibility | **(a)** Cortex is never trained continuously — "trains as it lives" is unrealized for the cortex. **(b)** The cortex's `reward/terminal/risk/uncertainty` heads have **no training loss anywhere** — they are random projections at inference, yet the amygdala/arbiter/MPC are meant to read them. **(c)** Latent-prediction target is the *same encoder's* moving output with no stop-grad/EMA target → representation-collapse risk (JEPA pitfall), only partially masked by the auxiliary pixel loss. |
| 4 | Notebook plan | A 13-step `notebooks/build_and_diagnose_organism.ipynb` — record → quality-gate → train cortex → beat copy-last (M2) → action-ablation (M2) → yaw probe → dreams (M4) → generative-replay forgetting (M5) → export for clinic. Scaffold committed alongside this doc. |
| 5 | Clinic meets "saw / predicted / actual" per frame? | **Was partially missing, now fixed.** The viewer showed *predicted vs actual vs error* but not **what the model saw at t**. Added the "seen t" panel so every strip reads **seen → predicted → actual → |error|**. Remaining gap: the EEG/mode timeline is not cursor-synced to the scrubbed frame. |

---

## 1. Feature completeness vs. the objective

### What is genuinely built (and tested)

Phase-by-phase, the organs exist and are individually solid:

- **Predictive Cortex** (`brain/cortex/predictive.py`): recurrent, action-conditioned, multi-horizon, decoded, with GRU / dilated-conv / transformer backbones (`brain/cortex/backbones.py`) and a context-length curriculum. This is a faithful realization of architecture commitments 2–3.
- **Neuromodulators + Amygdala + Arbiter** (`brain/neuromod/`, `brain/amygdala.py`, `brain/arbiter.py`): the 2×2 (surprise, pain) switch with hysteresis and a rolling-holdout surprise **calibrator** (`brain/calibration.py`). These *are* wired into the live loop (`cognitive_runtime/runtime/loop.py:37–47`).
- **Hippocampus + Dreams** (`brain/hippocampus.py`, `sleep/dream.py`): seed store and generative rollout with senses off; dream export in the viewer's `pixel-predictions-v1` format.
- **Generative replay + forgetting metric** (`sleep/replay_mix.py`, `sleep/forgetting.py`): the bootstrap-guardrail mixer (frozen-snapshot dreams, reservoir of real transitions, quality-gated dream fraction) and the CI-refereed forgetting report. **43/43 cortex + forgetting tests pass.**
- **Motor stack** (`motor/voluntary.py`, `motor/reflexes.py`, `motor/policy.py`): MPC seam + reflex-override precedence + alt controllers.
- **Development ladder** (`development/`) and **read-only Clinic** (`viewer/`).

### The central gap: the organs are not one organism

The overview promises a single continuous loop: *sense → cortex predicts next senses → feel → arbiter picks mode → act by planning over the cortex → remember → sleep consolidates the cortex.* **That spine is not assembled.** Three seams are open:

1. **The cortex is not the live world model.** The runtime predicts with
   `self.world_model.predict(...)` (`cognitive_runtime/runtime/loop.py:528`),
   where `world_model` defaults to `TrendWorldModel` — a trivial trend
   extrapolator — or, at best, the *legacy memoryless* `MLPWorldModel` via the
   issue-#26 bridge (`cognitive_runtime/cli.py:190` `_make_world_model`). There
   is **no code path** that installs the recurrent `PredictiveCortex` as the
   organism's live predictor. The cortex is trained and evaluated only
   *offline*, in batch, from recorded sessions (`ccr nursery run` / `joint` →
   `train_action_world_model`).

2. **"Trains as it lives" does not train the cortex.** The wake/sleep engine
   the CLI actually runs (`--async`, `sleep/async_trainer.py`) builds and
   optimizes `MLPPolicyModel` + `MLPValueModel` + `MLPWorldModel` — the **legacy
   actor-critic RL stack** (`sleep/async_trainer.py:102–124`). It never touches
   the `PredictiveCortex`. The consolidation-of-the-cortex-from-dreams
   mechanism (`GenerativeReplayMixer`) is invoked **only** from
   `tests/test_forgetting_metric.py` / the `sleep/forgetting.py` harness — there
   is no runtime that dreams from the hippocampus and takes cortex gradient
   steps during a live run.

3. **The voluntary MPC path is not the live actor.** `motor/voluntary.py` can
   plan over a cortex, but the live loop's actor is still the old
   `cognitive_runtime.core.policy.Policy`; nothing wires `MPCController`
   (cortex-planning) in as the default controller.

**So the thing that runs today** is the Minecraft-MVP actor-critic runtime with
the new neuromodulator/arbiter/hippocampus *signals* bolted on, running **beside**
an offline cortex-training-and-visualization pipeline. The clinic's "model"
predictions are exported from that offline batch model, not from a living
organism. This is the biggest "spirit of the objective" shortfall: **the
milestones prove the parts; no milestone proves the assembled loop.**

### Secondary gaps

- **The cortex is pixels-only.** `PredictiveCortex` encodes a single pixel
  stream (`PixelStreamEncoder`) + action. The overview's senses include
  body/interoception and efference copy, and "the organism can attend to its
  own dopamine." None of that reaches the cortex — it is a vision-and-action
  world model, not a multimodal one. Multimodal/interoceptive prediction is
  unbuilt.
- **Milestone 2 is provable but unproven.** The harness to show "cortex beats
  copy-last on held-out Crafter" and "ablating actions hurts turn" exists
  (`evaluate_action_world_model`, `run_action_ablation_eval`) but no committed
  result / CI gate demonstrates it passing on real Crafter seeds.
- **Hippocampal retrieval** and **learned neuromodulator scoring** are deferred
  (acknowledged in the plan — fair).

---

## 2. Legacy to identify for removal

Guiding rule from the brief: keep only what a *continuously-predicting organism*
and its *biological frameworks* need. Three tiers, **sequenced** because the live
loop currently leans on tier A.

### Tier A — the legacy learning spine (remove as the online learner, keep one copy as an A/B alt)

The V2 thesis is explicit: the world model is the only thing that learns online
(self-supervised regression = stable), and **nothing in the motor path learns**
(`01-architecture.md` commitment 5). The online **actor-critic RL** stack is
therefore *counter-thesis* as the live learner:

- `sleep/async_trainer.py` (trains actor/critic/MLP-world-model online)
- `cognitive_runtime/neural/policy.py`, `neural/value.py`, `neural/optimizer.py`
- `cognitive_runtime/neural/world_model.py` (**memoryless** MLP world model — a
  second, redundant multi-horizon model that duplicates the cortex's job)
- `cognitive_runtime/policies/actor_critic.py`

**Keep** the actor/critic only where the architecture already files it: as one
*alternative voluntary controller* behind `motor/policy.py` for A/B. Remove it as
the online learning spine and as `async_trainer`'s target. **Precondition:** wire
the cortex in as the live world model first (gap #1), or you delete the only
working live brain.

### Tier B — the Minecraft survival economy (defer/slim; it's nursery-era baggage)

`programs/minecraft/` is ~4,700 LOC dominated by a survival economy the
predictive objective does not need — the objective is self-supervised on the
world's own future, not on crafted rewards:

- `reward_engine.py`, `rewards.py`, `reward_profile.py`, `docs/reward_profiles.md`,
  and the whole `--reward-profile` system (elaborate, legacy, hunger/crafting).
- crafting/inventory/auto-craft, `entity_persistence` (Minecraft-specific).

Minecraft stays in the design as the *graduation* world, but per the plan it
comes **after** Milestone 5. Recommendation: reduce `worlds/minecraft` to a thin
pixels+motion+basic-body seam (mirroring the Crafter adapter) and delete the
survival economy, or quarantine the whole `programs/minecraft/` tree until the
nursery loop is a real organism.

### Tier C — legacy-data / back-compat cruft (safe to trim)

- "legacy fusion slot," "legacy single-`promotion`" shapes, back-compat action
  ordering (`programs/minecraft/actions.py` `USE`/`MOVE_BACKWARD` comments).
- **Keep** `runtime/replay.py`'s `LegacyFormatError` — it *rejects* pre-v2
  recordings loudly, which is the right behavior, not back-compat to carry.
- Historical docs to archive (they describe the pre-V2 world): `minecraft-mvp.md`,
  `nursery-turn-in-place-analysis.md`, `childhood-runs.md`, `online-learning.md`,
  `future-ai-os.md`, `reward_profiles.md`. Move to `docs/history/` rather than
  delete — they justify current design decisions.

> I did **not** delete anything under this section — removal is entangled with
> the integration fix (#1) and deserves an explicit go-ahead. See "Recommended
> sequence."

---

## 3. Major bugs & training-feasibility risks

**(a) The cortex never trains continuously.** (Detailed in #1.) Feasibility of
*offline* training is fine — short-rollout scheduled sampling, latent+pixel loss,
and the frozen-rollout detector are sound, and the tests pass. But the headline
capability — an AI that "predicts future ticks continuously" and learns while it
lives — is not runnable for the cortex today.

**(b) The cortex's prediction heads are untrained.** `PredictiveCortex` exposes
`reward_head`, `terminal_head`, `risk_head`, `uncertainty_head`
(`brain/cortex/predictive.py:182–188`), but the **only** cortex training loop,
`train_action_world_model`, optimizes `pixel_loss + latent_loss` and nothing else
(`training/action_world_model.py:456–459`). A repo-wide search confirms no loss
is ever applied to those heads on the cortex — the reward/risk losses in
`training/world_model.py` belong to the *legacy* `MultiHorizonMLPWorldModel`, a
different class. Consequences:

- The **amygdala** threat signal and **arbiter** "pain" input, if sourced from
  the cortex `risk_head`, read a random projection. (Today the runtime dodges
  this by feeding the arbiter a *runtime prediction-error stand-in*, not the
  cortex sigma — the calibrator docstring admits "no dedicated sigma head is
  wired into the WorldModel interface yet," `brain/arbiter.py:216–221`. So the
  heads are both untrained *and* unused, i.e. dead weight that looks load-bearing.)
- Any **MPC** scorer that reads predicted reward/risk (`motor/voluntary.py`
  requires a caller-supplied `scorer`) would plan over noise.

Fix: add reward/terminal/risk supervised terms to the cortex training loop
(targets are already on disk — the recorded reward/terminal/`internal.risk`
streams), and calibrate `uncertainty_head` against realized latent error (the
batch machinery in `training/world_model.py:442–448` shows the pattern).

**(c) Representation-collapse risk in the latent objective.** The primary latent
loss regresses the predicted latent onto `latents[:, idx+1].detach()` — the
*same encoder's* output, with the encoder training jointly and **no stop-gradient
target-encoder / EMA** (`training/action_world_model.py:444–447`). This is the
classic JEPA/BYOL collapse setup; `F.normalize` guards scale collapse but not
dimensional collapse, and only the *auxiliary* pixel-reconstruction term stands
between the model and a degenerate constant-latent solution. The `linear_probe_yaw`
diagnostic is the right canary — but it should be a **gate**, not an optional
probe, and a target-encoder (EMA) is worth adding before scaling up.

**(d) Lower-severity:** predictions live in a 16×16 downsampled reconstruction
space (all PSNR/"viewable frames" are 16×16 — coarse); the clinic server
shells out to `python -m ...quality_cli` once **per session** on every listing
(`viewer/server.js:33`), O(sessions) subprocess spawns.

---

## 4. Plan: `notebooks/build_and_diagnose_organism.ipynb`

A single notebook that builds a cortex from scratch and runs every diagnostic /
milestone gate. Committed as a scaffold alongside this doc; steps:

1. **Setup** — `pip install -e .[neural,crafter]`; choose an organism `name`.
2. **Record** nursery data in Crafter (`run_nursery_scenario` / `ccr nursery run
   walk_forward turn approach_entity object_permanence --world crafter
   --record-frames`) → train + holdout sessions.
3. **Data-quality gates** — `cognitive_runtime.record.quality` per session;
   render green/amber/red; **halt on red** (no training on bad data).
4. **Dataset** — `build_action_sequence_dataset(train_dirs, action_keys=<full
   ACTION_SPACE>)`.
5. **Train the cortex** — `train_action_world_model`; plot total/pixel/latent
   loss curves.
6. **Milestone 2a** — `evaluate_action_world_model` on holdout: per-horizon
   model/copy-last/oracle MSE, PSNR, `model_over_copy_last_mse`, and the
   **frozen-rollout** flag. Assert beats-copy-last at every horizon.
7. **Milestone 2b (action ablation)** — `run_action_ablation_eval`
   (`withhold_actions` off vs on); assert withholding actions measurably hurts.
8. **Representation probe** — `linear_probe_yaw`: latent vs hidden R² / angular
   error (collapse check from bug (c)).
9. **Heads diagnostic** — surface bug (b): show reward/risk/uncertainty heads are
   currently untrained (correlation of `uncertainty` vs realized error ≈ 0).
10. **Dreams (M4)** — encode hippocampal seeds, `dream()`, `export_dream_file`,
    render dreamed-vs-actual strip.
11. **Generative replay + forgetting (M5)** — train scenario A, hold-out loss;
    train B via `GenerativeReplayMixer`; `compute_forgetting_metric` → retained?
    Compare against flat-training baseline (the falsifiable claim).
12. **Export for the clinic** — `export_prediction_file` / `save_full_visual_model`;
    print the `node viewer/server.js` command.
13. **Statistical referee** — `statistical_evaluation` CIs on every metric above.

The notebook doubles as the missing **Milestone-2 proof** (gap #1) once run
against real Crafter seeds.

---

## 5. Clinic: "what it saw / predicted / actually saw" — every frame

**Requirement:** for every frame, show *what the model saw*, *what it predicted
it would see next*, and *what actually was shown*, plus other graphs.

**Finding:** `pixel-horizon-viewer` showed, per horizon, **predicted(t+h)**,
**actual(t+h)**, and **|error|** — but **not the input frame the model saw at t**.
The "what it saw" column was missing.

**Fixed in this change:** each horizon strip now renders four cells —
**seen t → predicted t+h → actual t+h → |error|** — so the requested triple is
visible for every scrubbed frame at every horizon
(`viewer/public/pixel-horizon-viewer.js`; all 7 clinic tests still pass). The
"seen" frame is drawn in the same (pooled) space as the model's targets so the
three images are directly comparable.

**Other graphs already present** (meet "in addition to other graphs"): EEG panel
(dopamine / acetylcholine / adrenaline / prediction-error sparklines + arbiter-mode
timeline), attention/focus table with reason breakdown, developmental-ladder
panel, per-session data-quality verdict, MSE-over-time chart with a frame cursor,
and dream strips (`?kind=dream`).

**Remaining clinic gaps (recommended, not yet done):**
- The EEG / arbiter-mode timeline is a separate panel, **not cursor-synced** to
  the scrubbed frame `t`. Add a shared time cursor so "what mode was the organism
  in *on this frame*" is answerable at a glance.
- Once the cortex is the live predictor (#1), the clinic should read predictions
  from the **live** organism, not only from offline `*-predictions_*.json`
  exports.

---

## Recommended sequence (what I'd do next, in order)

1. **Wire the cortex in as the live world model** behind the existing
   `world_model` seam (adapter: `Prediction` from a one-step cortex rollout),
   and add the reward/terminal/risk/uncertainty **head losses** (bug b). This
   closes the largest objective gap and makes the heads real.
2. **Run the notebook against real Crafter seeds** to bank the Milestone-2 proof.
3. **Add an EMA target-encoder + promote `linear_probe_yaw` to a gate** (bug c).
4. **Wire dream-based cortex consolidation into a live micro-sleep** (replace the
   actor-critic `async_trainer` target with the cortex + `GenerativeReplayMixer`).
5. **Then** retire Tier-A legacy and quarantine the Minecraft survival economy
   (Tier B), which by now nothing live depends on.
6. Cursor-sync the clinic EEG to the frame scrubber.

Items 1, 3, 4 are the difference between "a well-engineered library of biological
parts" and "the continuously-predicting organism the overview promises."
