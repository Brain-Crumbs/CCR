# Phase 4 — Hippocampus & Dreams

> Master plan: [Phase 4](../02-implementation-plan.md#phase-4--hippocampus--dreams).
> **Goal:** episodic seeds exist, and dreaming from them works — recall as a
> generative cortex rollout with senses off.

## Dependencies

- **Phase 2** — the cortex (its transition `f(z, a)` and decoder are what a dream
  rolls forward). Can run in parallel with Phases 3 and 8a.

## Builds on (existing code)

- `cognitive_runtime/neural/replay_buffer.py` — `ReplayBuffer`, `Transition`,
  `PriorityWeights`, `transition_priority`, `ReplayBufferConfig`. **The
  prioritisation is the starting point for the hippocampus's surprise/reward/threat/
  novelty weighting.**
- `cognitive_runtime/core/modulation.py` (→ `brain/neuromod/` after Phase 3) — the
  `internal.*` tags (dopamine/RPE, risk, safe_novelty) that prioritise seeds.
- `cognitive_runtime/training/prediction_export.py` — `export_prediction_file`,
  `export_session_predictions`, `_b64_frame` (the dream-strip export format).
- `cognitive_runtime/training/action_world_model.py` / `brain/cortex/predictive.py`
  — the closed-loop `rollout(start_latent, actions, hidden)` a dream reuses.
- `viewer/public/pixel-horizon-viewer.js` — the strip renderer the clinic reuses.

## Tasks

1. **`brain/hippocampus.py` — the episodic seed store.**
   - A fast, **capacity-bounded** store of seeds
     `(z_t, action-sequence, dopamine/threat/novelty tags)`, prioritised by the
     neuromodulator tags. Build over `replay_buffer.py`'s priority machinery
     (`transition_priority`, `PriorityWeights`) rather than reinventing it.
   - Pattern-separated, one-shot writes: encoding a tick is cheap and happens in the
     wake tick budget.
   - *Acceptance:* encoding N ticks yields ≤ capacity seeds, ordered by priority;
     high-surprise/high-reward ticks are retained over bland ones when full.

2. **Encode seeds during the wake tick.**
   - Hook seed encoding into the loop's per-tick record path so every waking tick
     hands a sparse seed to the hippocampus (the Record already writes the tick;
     the seed is the compact `(z, actions, tags)` slice).
   - *Acceptance:* a recorded run produces a populated hippocampus without stalling
     the tick (encoding cost is bounded and off the critical path).

3. **`sleep/dream.py` — the generative rollout.**
   - Implement `dream(seed, length)`: start from `seed.z`, and for each step feed
     `seed.actions[k]` (replay) — imagined actions are a Phase 6 concern — advance
     the cortex `z ← f(z, a)` with **senses off**, and `yield decode(z)` (the
     regenerated frame). Reuse the cortex `rollout`.
   - *Acceptance:* `dream(seed, len)` returns a decoded frame per step from the
     stored seed with no live senses consumed.

4. **Export dreams for the clinic.**
   - Reuse `training/prediction_export.py`'s format to write a dream strip
     (dreamed vs the original episode's actual frames per horizon), name-prefixed
     (Phase 0).
   - *Acceptance:* a dream export loads in `pixel-horizon-viewer.js` and shows
     dreamed-vs-actual side by side.

## Deliverables

- `brain/hippocampus.py`: capacity-bounded, priority-weighted episodic seed store,
  encoded during wake.
- `sleep/dream.py`: `dream(seed, length)` generative rollout with senses off.
- Dream-strip export in the existing prediction-export format.

## Tests

- `tests/test_hippocampus.py`: capacity bound enforced; priority ordering
  (surprise/reward/threat/novelty) matches `transition_priority`; eviction keeps the
  high-priority seeds.
- `tests/test_dream.py`: `dream` from a seed reproduces the episode to within the
  cortex's own T+h accuracy on a held-out seed; senses-off invariant (no sensory bus
  reads during a dream).
- Export round-trip: dream file parses and matches the viewer's expected schema.

## Milestone 4 (exit gate)

A dream launched from a stored seed **regenerates the original episode's frames to
within the cortex's own T+h accuracy**, and the clinic renders the dream strip.
**Recall works.** Verified by `tests/test_dream.py` (recorded-scenario assertion per
the master plan's test strategy) + a rendered strip in the clinic (or the viewer
until 8a lands).

## Risks / notes

- **A dream is only as good as the cortex.** Dreaming from a half-trained cortex
  reproduces its errors — that is expected here (Phase 4 only proves recall). The
  *consolidation* use of dreams, and the bootstrap guardrail (dream fraction gated on
  measured quality), belong to **Phase 5**; do not train on dreams in this phase.
- **Retrieval is deferred.** Phase 4 builds the store + generative dreaming;
  context-cued recall of a *relevant* past episode is an explicit follow-up
  ([master plan deferrals](../02-implementation-plan.md#what-this-plan-deliberately-defers)).
- Keep the seed **sparse** — it is `(z, actions, tags)`, not full frames. Frames come
  back by *decoding the rollout*, which is the whole point.
