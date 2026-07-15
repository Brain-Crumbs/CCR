# Phase 5 — Sleep as Continuous Consolidation

> Master plan: [Phase 5](../02-implementation-plan.md#phase-5--sleep-as-continuous-consolidation).
> **Goal:** the organism trains as it lives, via a wake/sleep cycle, without
> stalling the tick — and dreaming demonstrably prevents forgetting. **This phase
> carries the one falsifiable claim of the whole project.**

## Dependencies

- **Phase 4** — dreams to replay (generative rollouts from hippocampal seeds).
- **Phase 2** — the cortex is the thing consolidated.

## The measured claim (why this phase matters most)

> Developmental staging + generative replay (dreaming old seeds while learning new
> ones) produces **measurably less catastrophic forgetting than flat training on the
> same data.** ([overview](../00-overview.md#the-one-measured-claim),
> [decision log #10](../direction-and-critique-response.md).)

Milestone 5 is designed to prove exactly this.

## Builds on (existing code)

- `cognitive_runtime/training/async_trainer.py` — `AsyncTrainer`,
  `spawn_trainer_process`, `TrainerSupervisor`, `run_forever`, `train_step`,
  `publish`, `ingest_live`, `load_recorded_sessions`. **The actor/learner split,
  re-framed as wake/sleep.**
- `cognitive_runtime/neural/weight_publisher.py` — `WeightPublisher`,
  `WeightSubscriber` (`poll_version`, `maybe_reload`). **The weight hand-off; extend
  with EMA + monotonic version stamp for the concurrent schedule.**
- `cognitive_runtime/neural/replay_buffer.py` — the reservoir of real transitions.
- `cognitive_runtime/brain/hippocampus.py`, `sleep/dream.py` (Phase 4) — seeds +
  dream rollout to mix into replay.
- `cognitive_runtime/training/statistical_evaluation.py` — the regression referee for
  the forgetting metric (CIs over N episodes).
- `cognitive_runtime/tests/test_async_trainer.py`,
  `test_async_actor_learner_integration.py` — existing coverage.

## Tasks

1. **Re-frame `training/async_trainer.py` + `weight_publisher.py` as `sleep/`.**
   - **Wake** (tick thread): cheap in-tick cortex updates that fit the budget +
     episodic seed encoding (Phase 4). Never blocks.
   - **Sleep** (separate process): drain a mix of **real + dreamed** replay, take the
     heavy cortical gradient steps, publish weights back between ticks. Land behind
     shims from the old module paths.
   - *Acceptance:* a run does cheap wake updates and a heavier sleep pass through the
     renamed `sleep/` path; old imports still resolve.

2. **Phasic wake/sleep first (no staleness).**
   - Ship the simple schedule: act for a while → pause acting → consolidate →
     resume. Nothing acts while the cortex updates, so there is **no weight
     staleness** — simplest to get right, closest to the biology.
   - *Acceptance:* a phasic run alternates act/consolidate cleanly; the actor uses
     only post-consolidation weights (no mid-update reads).

3. **Concurrent schedule later — EMA + version stamp.**
   - Only after phasic works, enable the concurrent separate-process trainer. When
     concurrent, publish an **EMA/Polyak-averaged** weight copy (a slow-moving target
     kills tick-to-tick oscillation) and stamp a **monotonic version** so the actor
     can log and bound how stale its weights are (extend `WeightPublisher` /
     `WeightSubscriber`).
   - *Acceptance:* concurrent mode publishes EMA weights with an increasing version;
     the actor reports and bounds its staleness.

4. **Generative replay with the bootstrap guardrail.**
   - Mix dreamed old seeds with new experience as the forgetting defence, subject to:
     (a) keep a **reservoir of real transitions** and **never train on dreams
     alone**; (b) **gate the dream fraction on measured model quality** — 0% until the
     cortex beats copy-last on held-out by a margin, ramp with the ratio, cap ≈ 0.5.
     The dream fraction is a *function of a metric*, not a constant.
   - *Acceptance:* with a weak cortex the dream fraction is 0; as held-out quality
     rises it ramps toward the cap; training never draws a dream-only batch.

5. **Report a forgetting metric.**
   - Measure whether `walk_forward` accuracy survives learning `object_permanence`
     (and generally: old-scenario accuracy after training a new one). Route through
     `statistical_evaluation` so regressions are flagged with CIs.
   - *Acceptance:* the metric is emitted per consolidation; a flat-training control
     shows worse retention than the staged+replay condition (the measured claim).

6. **Micro-sleep + long consolidation schedules.**
   - Support periodic micro-sleeps during a run and a long consolidation at session
     boundaries; both clinic-triggerable later (Phase 8b).
   - *Acceptance:* both schedules run; each records what it dreamed + loss curves +
     the forgetting metric.

## Deliverables

- `sleep/` wake/sleep cycle (phasic first, concurrent with EMA + version stamp
  later), behind shims.
- Generative replay with reservoir + quality-gated dream fraction.
- A reported forgetting metric through `statistical_evaluation`.
- Micro-sleep and session-boundary consolidation schedules.

## Tests

- Extend `test_async_trainer.py`: phasic schedule uses only post-consolidation
  weights; concurrent schedule publishes EMA weights with monotonic versions and
  bounded staleness.
- `tests/test_generative_replay.py`: dream fraction is 0 below the quality bar and
  ramps/caps above it; no dream-only batch is ever drawn; reservoir retained.
- `tests/test_forgetting_metric.py`: staged+replay retains a mastered scenario
  within tolerance while a flat-training control does not; zero missed-tick
  regression vs a no-sleep baseline.

## Milestone 5 (exit gate — the falsifiable result)

A continuous run **learns a new scenario during wake+sleep while retaining a
previously-mastered one** (forgetting metric stays within tolerance), **with zero
missed-tick regression** versus a no-sleep baseline. This is a CI-runnable check on
held-out seeds and is the project's headline measured claim.

## Risks / notes

- **The spine is self-supervised, and that is what keeps it stable.** The heavy
  thing learned in sleep is the *world model* (regression onto the world's recorded
  future), not a bootstrapped policy chasing a moving value target. Keep the
  make-or-break online learning on the world model; keep the motor as *planning* over
  it (Phase 6) — do not let a learning policy into this loop, or the async-RL
  instabilities return.
- **The dream bootstrap paradox is the sharp risk.** Dreams from a half-trained
  cortex reinforce its own errors. The reservoir + quality-gated dream fraction are
  not optional niceties — they are the guardrail that makes generative replay help
  instead of harm.
- Phasic-before-concurrent is a hard ordering: do not ship the concurrent trainer
  until phasic + the forgetting metric are green, or you will debug staleness and
  forgetting at the same time.
