# Phase 3 — Neuromodulators, Amygdala & the Arbiter

> Master plan: [Phase 3](../02-implementation-plan.md#phase-3--neuromodulators-amygdala--the-arbiter).
> **Goal:** the three modes (reward-seeking / info-gathering / fight-or-flight) are
> selected each tick by a switch over (surprise, predicted pain) and visibly change
> behaviour.

## Dependencies

- **Phase 2** — the cortex, its prediction-error signal, and its (calibratable)
  uncertainty σ and risk head. Can run in parallel with Phases 4 and 8a.

## Builds on (existing code)

- `cognitive_runtime/core/modulation.py` — `ModulationSignals`, `ModulationTracker`,
  `compute_reward_prediction_error`, `LearningProgressTracker`, `safe_gate`, and the
  `internal.*` streams (`internal.reward_prediction_error`, `internal.risk`,
  `internal.predicted_risk_aversion`, `internal.safe_novelty`). **Dopamine is
  already RPE — no new math.**
- `cognitive_runtime/core/attention.py` — `AttentionController` (the Thalamus):
  `AttentionSignal`, `AttentionCoefficients`, `AttentionBudget`, `StimulusDirection`.
  Feed acetylcholine in as a precision term.
- `cognitive_runtime/core/orienting_reflex.py` — `OrientingReflex`,
  `OrientingDecision` (the info-gathering orient; wired fully in Phase 6, referenced
  by the arbiter here).
- `cognitive_runtime/tests/test_modulation.py`, `test_attention.py`,
  `test_intrinsic_drive.py` — existing coverage.

## Tasks

1. **Rename `core/modulation.py` → `brain/neuromod/` with human-named signals.**
   - Map the existing `internal.*` math to named streams: **dopamine** =
     `internal.reward_prediction_error`; **acetylcholine** = a precision/uncertainty
     term derived from cortex σ + learning-progress; **adrenaline** = the appraised
     threat release (Task 2). Land behind re-export shims.
   - *Acceptance:* `test_modulation.py` passes through the shim; each named signal is
     published as an `internal.*` (human-named) stream.

2. **`brain/amygdala.py` — threat appraisal → adrenaline.**
   - Appraise the cortex's risk head (`internal.risk` +
     `internal.predicted_risk_aversion`) into a fast threat/adrenaline release that
     can pre-empt deliberation and gate reflexes. Reuse `safe_gate`.
   - *Acceptance:* a rising predicted-pain signal produces an adrenaline spike stream;
     a calm scene keeps it near zero.

3. **`brain/arbiter.py` — the three-mode state machine.**
   - A **hand-authored 2×2 lookup** over two scalars: **surprise** (prediction error)
     and **predicted pain** (amygdala/risk):
     - low surprise → **reward-seeking** (bored);
     - high surprise, safe → **info-gathering** (curious): orient + sample to reduce
       error (drives `internal.safe_novelty`);
     - high surprise, threatened → **fight-or-flight** (afraid): reflex overrides
       deliberation, adrenaline.
   - The mode gates attention breadth and which motor path/reflex wins. Publish the
     chosen mode as a recorded stream.
   - *Acceptance:* given scripted (surprise, pain) inputs, the arbiter returns the
     expected mode for each quadrant; the mode is recorded per tick.

4. **Calibrate surprise and report it.**
   - The 2×2 is only as good as its inputs. Take the cortex σ (ensemble or
     predicted-error head from Phase 2), **calibrate** it (temperature scaling /
     reliability diagram on the rolling holdout), and **report** calibration as a
     first-class metric.
   - *Acceptance:* a reliability-diagram / calibration metric is emitted; an
     uncalibrated head is visibly flagged.

5. **Add hysteresis to the switch.**
   - A mode change requires the threshold to hold for `k` consecutive ticks before it
     takes, so the mode doesn't flap tick-to-tick.
   - *Acceptance:* a single-tick threshold blip does not change the mode; a sustained
     crossing does after `k` ticks.

6. **Feed acetylcholine into the Thalamus.**
   - Wire the acetylcholine precision term into `core/attention.py`'s scoring so
     trust-per-sense / focus sharpness responds to expected uncertainty.
   - *Acceptance:* raising acetylcholine measurably narrows/sharpens attention under
     the same budget.

## Deliverables

- `brain/neuromod/` (dopamine / acetylcholine / adrenaline) over existing math,
  behind shims.
- `brain/amygdala.py` threat→adrenaline appraisal.
- `brain/arbiter.py` three-mode switch with calibrated surprise + hysteresis, mode
  recorded as a stream.
- Acetylcholine feeding the Thalamus.

## Tests

- `tests/test_arbiter.py`: 2×2 truth table over (surprise, pain); hysteresis holds a
  mode through a single-tick blip and flips after `k` sustained ticks.
- `tests/test_amygdala.py`: predicted-pain → adrenaline spike; calm → quiet.
- Extend `test_attention.py`: acetylcholine precision term changes attention
  coefficients as expected.
- Calibration metric unit test on a synthetic mis-calibrated head.

## Milestone 3 (the three-region test — exit gate)

In a scripted scene with a **harmless** surprise and a **harmful** one, the organism
demonstrably enters **info-gathering** for the first (orients toward it) and
**fight-or-flight** for the second (reflex overrides policy, adrenaline spikes), and
**reward-seeking** when bored — each visible in the arbiter-mode timeline. Verified
by a recorded-scenario assertion (per the master plan's test strategy: Phase 3's
gate is a recorded-scenario check, not a held-out-seed metric).

## Risks / notes

- **The arbiter is authored, not emergent.** It is a hand-written 2×2 lookup — do
  not describe it as "arising." Its correctness is the lookup + calibrated inputs +
  hysteresis, nothing more ([decision log #1](../direction-and-critique-response.md)).
- **No chemistry cosplay.** Ship only the three chemicals that each *do* something.
  Serotonin/patience and explicit norepinephrine-arousal are deferred until a
  concrete behaviour needs them.
- The full orient/override *behaviour* lands in Phase 6 (motor stack); here the
  arbiter selects the mode and gates, and the orienting reflex is referenced but its
  motor precedence is finalised in Phase 6.
