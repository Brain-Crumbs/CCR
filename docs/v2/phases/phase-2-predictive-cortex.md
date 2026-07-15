# Phase 2 — The Predictive Cortex (the heart)

> Master plan: [Phase 2](../02-implementation-plan.md#phase-2--the-predictive-cortex-the-heart).
> **Goal:** one action-conditioned, recurrent, decoded, multi-horizon world model
> that serves every scenario — replacing the per-scenario tiny predictors.
>
> **This is the pivotal, make-or-break phase.** Everything above it (modes, dreams,
> sleep, motor, the ladder, the clinic) is only worth building once this clears its
> bar. See [decision log #10](../direction-and-critique-response.md).

## Dependencies

- **Phase 1** — a deterministic Crafter world with recorded scenarios and quality
  gates to train/evaluate the cortex on held-out seeds.

## Builds on (existing code)

- `cognitive_runtime/training/action_world_model.py` — **already a recurrent,
  action-conditioned prototype**: `ActionWorldModelConfig`,
  `build_action_world_model` (encoder + GRUCell transition + decoder, closed-loop
  `rollout`), `train_action_world_model`, `evaluate_action_world_model`,
  `_best_recurrence_lag`, `_pairwise_dispersion` (frozen-rollout signal). **Promote
  this to the cortex.**
- `cognitive_runtime/neural/world_model.py` — `MultiHorizonMLPWorldModel`,
  `HorizonPrediction`, `MultiHorizonWorldModelOutput`, per-horizon +
  uncertainty heads, `checkpoint_metadata()`. **Merge its multi-horizon +
  uncertainty structure in.**
- `cognitive_runtime/neural/pixel_stream_encoder.py` — `PixelStreamEncoder`.
- `cognitive_runtime/training/visual_representation.py` —
  `PixelReconstructionDecoder`, `_reconstruction_shape`.
- `cognitive_runtime/neural/checkpoint.py` — checkpoint bundle (store horizons +
  name here).
- `cognitive_runtime/training/statistical_evaluation.py` — CIs / regression
  refereeing for held-out scoring.
- `cognitive_runtime/tests/test_action_world_model.py`,
  `test_multi_horizon_world_model.py` — existing coverage to extend.

## Tasks

1. **Create `brain/cortex/predictive.py` by promoting the action world model.**
   - Move the `ActionConditionedWorldModel` (encoder + action-conditioned GRUCell
     transition + latent head + decoder) into the cortex module, behind a
     re-export shim from `training/action_world_model.py` so nothing breaks.
   - *Acceptance:* existing `test_action_world_model.py` passes unchanged against the
     new location via the shim.

2. **Merge in multi-horizon + uncertainty heads.**
   - Fold `MultiHorizonMLPWorldModel`'s per-horizon heads and uncertainty output
     into the recurrent cortex: for each configured horizon `h`, emit
     `ẑ_{t+h}`, `decode(ẑ_{t+h})` (same input shape), `σ_{t+h}` (uncertainty),
     and the `r̂` / `d̂` / `risk` heads.
   - Produce uncertainty **cheaply**: an ensemble or a predicted-error head (this
     σ is the input Phase 3's arbiter depends on — build it calibratable).
   - *Acceptance:* one forward pass yields all horizons with a decoded frame and a σ
     per horizon; heads are present for reward/terminal/risk.

3. **Latent + decoder discipline.**
   - Learn on `z` (latent) as the primary target; keep the decoder so every horizon
     decodes to the same input shape (viewable). Reconstruction is an **auxiliary**
     loss; latent prediction + short-rollout scheduled sampling is primary.
   - **No 100-step backprop-through-composition** — it selects for the identity
     attractor documented in the turn-in-place analysis. Keep the short
     `rollout_frames` (default 8) + `scheduled_sampling_p` design already in
     `ActionWorldModelConfig`.
   - *Acceptance:* the loss config exposes latent vs pixel weights; the training
     window is short-rollout, not long-composition.

4. **Horizons in ticks, stored with the checkpoint.**
   - Represent horizons as tick counts (default `{1, 4, 8}`, per-organism
     configurable), persisted in checkpoint metadata so "T+8" means the same across
     worlds and sample rates. Reuse `horizons_ticks_to_frames`.
   - *Acceptance:* a checkpoint records its horizons; loading restores them; changing
     sample rate does not change the tick semantics.

5. **Temporal backbone as an A/B choice.**
   - Keep the GRU as default; add an alternative **dilated temporal-conv
     (WaveNet-style) or small transformer over a frame window** behind the same
     interface, with a **context-length curriculum** (1 frame → 2 → k). This is a
     benchmark, not a fork.
   - *Acceptance:* the backbone is selectable by config; both train and evaluate
     through the identical cortex interface; a benchmark harness reports both.

6. **Wire the scoring gates as structured report fields.**
   - Every evaluation reports, per horizon: `MSE(model)/MSE(copy-last)` and, for
     periodic scenarios, `MSE(model)/MSE(period-oracle)`; plus a **frozen-rollout
     detector** (predicted frames identical across horizons while actuals differ ⇒
     red), built on `_pairwise_dispersion`.
   - Emit these as structured fields (JSON), not just logs, so the clinic and CI
     consume them directly.
   - *Acceptance:* `evaluate_*` returns a structured report with the three gate
     values per horizon; a frozen model trips the detector.

7. **Replace per-scenario predictors with the one cortex.**
   - Point the nursery/training paths at the single cortex for every scenario
     instead of tiny per-scenario models. Keep old paths importable via shim during
     migration.
   - *Acceptance:* `walk_forward`, `turn`, `object_permanence` all train/evaluate
     through the same cortex checkpoint.

## Deliverables

- `brain/cortex/predictive.py`: recurrent, action-conditioned, decoded,
  multi-horizon cortex with calibratable uncertainty and reward/terminal/risk heads.
- Tick-denominated horizons in the checkpoint.
- Selectable GRU vs temporal-conv/transformer backbone + context-length curriculum.
- Structured scoring report (`model/copy-last`, `model/oracle`, frozen-rollout flag)
  per horizon.
- One cortex serving all scenarios; shims from old module paths.

## Tests

- Extend `test_action_world_model.py` / `test_multi_horizon_world_model.py`:
  multi-horizon forward shape, decoded-frame shape equals input shape, σ present.
- `tests/test_predictive_cortex.py`: action-ablation harness — training with the
  action stream withheld measurably worsens `turn` (the M2 claim, as an assertion);
  frozen-rollout detector trips on a degenerate identity model; copy-last baseline
  computed correctly.
- Held-out scoring wired through `statistical_evaluation` with CIs.

## Milestone 2 (the pivotal proof — exit gate)

On held-out Crafter seeds, the cortex **beats copy-last at every horizon on
`walk_forward`**, *and* **withholding the action stream measurably hurts `turn`**
(proving it actually uses actions). Both are CI-runnable checks on held-out seeds
via `statistical_evaluation`. **Promise becomes proof here — do not start Phases
3–8 until this passes.**

## Risks / notes

- **The identity attractor is the enemy.** MSE over long compositions of one
  transition rewards "predict no change." Guard with: short rollout, scheduled
  sampling, latent-space loss primary, and the frozen-rollout detector as a red gate.
- **Uncertainty is load-bearing downstream.** Phase 3's mode switch flaps if σ is
  uncalibrated. Build σ now so Phase 3 only has to calibrate/report it, not invent it.
- **The rename deferral ends here.** Per Phase 0, the package-tree rename
  (`organism/`, `brain/`, `world/`, …) + `ARCHITECTURE_MAP.md` lands *after* this
  milestone passes — behind shims — or never. Do it once the cortex is proven.
- Action-conditioning is the single most important fix over today's per-scenario
  predictors: `z_{t+1} = f(z_t, a_t)`. A predictor that never sees its action can't
  tell "kept turning" from "stopped."
