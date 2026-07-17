# Issue 109 Implementation Bug Bash — Consolidated Findings

**Date:** 2026-07-17
**Scope:** the V2 implementation through Phase 7, checked against the task-level
acceptance criteria in `docs/v2/phases/`. Phase 8 is treated as future work rather
than as a regression because its implementation has not started.

**Test suite baseline:** 484 passed, 43 skipped (all pass clean).

---

## Executive result

The implementation is **not ready to satisfy the end-to-end V2 exit gates**. The
lower-level components have broad unit coverage and well-structured code, but
Phase 5's measured claim is absent, the Phase 7 runner carries the new stage fields
as metadata rather than applying them, and several code-level correctness bugs lurk
in under-tested edge cases. In particular, the ladder trains an actor-critic while
its promotion gates train separate, disposable predictive models. It therefore does
not raise or measure one organism through the declared developmental stages.

Two categories of findings:

- **Architectural / integration bugs (BB-109-01 through BB-109-08):** components not
  wired together, missing implementations, design-level problems where the runner
  ignores stage declarations.
- **Code-level correctness bugs (BB-109-11 through BB-109-21):** wrong behavior at
  specific lines — crashes, silent data corruption, dropped overrides, broken
  invariants. These are not caught by the existing test suite.

---

## Architectural / Integration Findings

### BB-109-01 — Critical — Phase 5 generative replay was never implemented

**Acceptance criteria affected:** Phase 5 tasks 1, 4, 5, and 6; Milestone 5.

The `sleep/` package contains scheduling, async training, and weight publication,
but it has no real-transition reservoir, no quality-controlled dream fraction, no
real+dream batch mixer, and no forgetting evaluator. The Phase 5-required
`tests/test_generative_replay.py` and `tests/test_forgetting_metric.py` do not exist.
The implementation itself acknowledges this absence when the ladder substitutes a
different gate for `forgetting_score`.

**Impact:** the headline claim — staged generative replay reduces catastrophic
forgetting — cannot be run, measured, or accepted. Concurrent EMA publication is not
a substitute for consolidation from generative dreams.

**Reproduction:** compare the Phase 5 deliverables and required test file names with
the contents of `sleep/` and `tests/`, then search for `dream_fraction`,
`forgetting_score`, or a replay reservoir.

---

### BB-109-02 — Critical — Stage motor freedom is ignored by the real ladder runner

**Acceptance criteria affected:** Phase 7 task 5 and Milestone 7; Phase 6's caregiver
integration gate.

**File:** `development/runner.py` — `_run_stage_episodes()` (lines 147-176)

`run_curriculum()` always creates an `ActorCriticPolicy` and
`ActorCriticLearner` for every stage. `_run_stage_episodes()` never reads
`stage.motor_freedom` and never calls `build_stage_policy()`. Consequently:

- Gestation is not frozen.
- Babbling and Crawling do not use the scripted/caregiver path.
- Objects and Foraging do not use the Phase 6 voluntary-controller seam.

The motor-freedom tests exercise `build_stage_policy()` in isolation, not the
runner that is claimed to walk the ladder.

**Impact:** the organism performs learned actor-critic actions during every rung,
so its actual experience contradicts the ladder definition and cannot produce the
guided forward/inverse-model data required by Milestone 6.

**Reproduction:** instrument the policy passed to `CognitiveRuntime` in
`_run_stage_episodes()` while running the Gestation stage; it is an
`ActorCriticPolicy`, not a frozen `MotorFreedomPolicy`.

---

### BB-109-03 — Critical — Promotion metrics measure disposable models, not the checkpoint being promoted

**Acceptance criteria affected:** Phase 7 tasks 3 and 4 and Milestone 7.

**Files:** `development/runner.py:142`, `development/ladder.py:217-251`

The checkpoint persisted by `run_curriculum()` contains only an actor, critic, and
their optimizer (`has_world_model` is explicitly `False` at `runner.py:142`). The
predictive gates then call `run_nursery_scenario()` and `run_action_ablation_eval()`,
which record fresh data and train fresh predictive models outside that checkpoint.
Neither helper is given the organism's checkpoint or weights.

**Impact:** a random/untrained ladder checkpoint can promote whenever an unrelated
temporary nursery model passes. Conversely, the organism's own learning cannot
improve those gates. The history gives the misleading appearance that the same
brain passed a held-out cortex milestone.

**Reproduction:** hash the ladder checkpoint immediately before and after
calling `ladder_milestone_metrics()`; the gate computation does not load it.
Inspect checkpoint metadata and observe `has_world_model: false`.

---

### BB-109-04 — High — Declared scenarios, senses, and losses are not applied

**Acceptance criteria affected:** Phase 7 tasks 1, 2, and 3.

**File:** `development/runner.py` — `_program_for_stage()` (lines 97-117),
`_run_stage_episodes()` (lines 147-176)

`_program_for_stage()` honors only `world` and `world_config`.
`_run_stage_episodes()` does not reference `stage.scenario`, `stage.senses`, or
`stage.losses`. All Crafter stages therefore run the generic environment with the
same full fusion layout and actor-critic objective; `object_permanence`, `turn`,
`walk_forward`, and `approach_entity` are labels used only by the separate gate
helpers.

**Impact:** changing these supposedly load-bearing fields has no effect on the
organism's training run. The phase acceptance test currently proves that fields can
be serialized, not that the runner enforces them.

**Reproduction:** replace a stage's `scenario`, `senses`, and `losses`
with other valid strings and run with fixed seeds. The runner's constructed
program/policy/learner and episode trajectory are unchanged.

---

### BB-109-05 — High — Later promotion gates are constant synthetic checks unrelated to development

**Acceptance criteria affected:** Phase 7 task 4; Phase 6 task 7.

**File:** `development/ladder.py` — `_reflex_override_precedence()` (lines 254-279),
`_voluntary_reliance_score()` (lines 282-303)

The Objects gate runs a newly constructed `ReflexStack` for 20 synthetic ticks and
is structurally guaranteed to return `1.0`. The Foraging gate runs another new
stack against a fixed RNG threat stream and reports `1 - activation_rate`. Neither
uses the organism, its recorded stage session, its cortex, or its maturation
history.

**Impact:** Objects and Foraging can promote irrespective of learned behavior. The
claimed downward developmental reflex trend is not measured; the input threat rate
is simply chosen low enough to pass.

**Reproduction:** call the two helpers before any curriculum training;
they return the same passing values that they return after training.

---

### BB-109-06 — High — The reflex metric changes meaning without changing its name

**Acceptance criteria affected:** Phase 6 task 7 and Phase 7 task 4.

**File:** `development/ladder.py:303`

`reflex_activation_rate` normally means the fraction of ticks on which a reflex
activated (lower is better). The ladder stores `1.0 - reflexes.activation_rate`
under that same key so it can use the gate evaluator's hard-coded `>=` comparison.
Recorded history thus labels voluntary reliance as reflex activation rate.

**Impact:** dashboards and offline analysis will interpret a high value as frequent
reflexes while promotion interprets it as few reflexes. This silently reverses the
meaning of a Phase 6 metric at the Phase 7 boundary.

**Suggested fix:** add comparison direction to `PromotionCriteria`, or expose a
separately named `voluntary_reliance_rate` metric.

---

### BB-109-07 — High — Milestone gate sample sizes are ignored

**Acceptance criteria affected:** Phase 7 task 4's held-out-data gate.

**File:** `development/runner.py:345,355`

Each milestone `PromotionCriteria` owns a `sample_size`, but the runner chooses eval
episode count and seed spacing exclusively from `stage.promotion.sample_size`.
Changing a gate's sample size therefore has no effect. Multiple gates with different
sample sizes cannot be honored at all.

**Impact:** a gate declared as an N-episode aggregate may actually promote from the
legacy promotion field's smaller sample, making the validation weaker than its
configuration says.

**Reproduction:** declare a milestone gate with `sample_size=5` and a legacy
promotion with `sample_size=1`; the runner evaluates one episode.

---

### BB-109-08 — High — No supported command runs the built-in ladder and its real gates

**Acceptance criteria affected:** Phase 7's unattended-run deliverable and
Milestone 7.

**File:** `cognitive_runtime/cli.py` — `cmd_curriculum_run` (line 1938)

The CLI exposes generic `curriculum-run` for a user-supplied definition file, but it
imports from the old `training.curriculum_runner`, not the new `development.runner`.
It does not expose `GESTATION_TO_FORAGING` or automatically attach
`ladder_milestone_metrics`. A YAML representation also cannot serialize the Python
metric-provider callback. The only way to invoke the advertised ladder is a custom
Python call that manually supplies a partially applied provider and record path.

**Impact:** users cannot perform the milestone's unattended named-organism run from
the supported application surface. Omitting the callback fails once a milestone
metric is requested.

---

## Code-Level Correctness Findings

### BB-109-11 — High — Caregiver override silently dropped when reflexes=None in "overridden" mode

**File:** `motor/organism_policy.py:107-108`

When `motor_freedom="overridden"` and `reflexes is None`, the `decide()` method
returns the scripted action directly at line 108 without ever draining
`self.caregiver`. Injected caregiver overrides accumulate indefinitely and are never
acted upon — violating the documented precedence contract ("caregiver always wins").

```python
if self.reflexes is None:
    return voluntary_action  # BUG: caregiver.drain() never called
```

**Failure scenario:** `motor_freedom="overridden"`, `reflexes=None`, caregiver has
injected override -> override is ignored, scripted action returned instead.

---

### BB-109-12 — High — `_encode_goal` crashes on 2D non-latent pixel tensors

**File:** `motor/policy.py:55`

A 2D tensor (e.g. `[H, W]` grayscale frame) where `shape[-1] != cortex.latent_width`
falls through to `goal.permute(2, 0, 1)`, which requires exactly 3 dimensions ->
`RuntimeError`. The docstring says "H x W x C RGB pixel frame" but a 2D frame
(single-channel or collapsed) isn't handled.

**Failure scenario:** `goal` is a 2D tensor `[H, W]` where `shape[-1] !=
cortex.latent_width` -> `permute(2, 0, 1)` raises RuntimeError.

---

### BB-109-13 — High — Batched latent goals silently produce wrong encoding

**File:** `motor/policy.py:52-53`

A batched latent `[B, latent_width]` with `B > 1` passes the `dim() <= 2` and
`shape[-1] == cortex.latent_width` checks, then `reshape(1, -1)` flattens it to
`[1, B*latent_width]` — a single vector of the wrong width. Downstream MSE
comparisons silently compute garbage or broadcast-error.

**Failure scenario:** `goal` is `[2, latent_width]` tensor -> `reshape(1, -1)`
produces `[1, 2*latent_width]` -> downstream MSE against `[1, latent_width]`
target gives wrong results or broadcast error.

---

### BB-109-14 — Medium — `ImaginationActor.act()` broken on unbatched input

**File:** `motor/policy.py:126-127`

`logits[0]` assumes a batch dimension. If `latent` is 1D `[latent_width]`, the
actor produces 1D `[n_actions]` logits; `logits[0]` then extracts the scalar logit
of the first action, and `argmax` of a scalar always returns `0` — the actor
always picks action 0 regardless of learned weights.

```python
logits = self.actor(latent)
return int(torch.argmax(logits[0]).item())  # BUG: logits[0] is scalar for 1D input
```

**Failure scenario:** `latent` is 1D `[latent_width]` -> actor outputs 1D
`[n_actions]` -> `logits[0]` is scalar -> `argmax` always returns 0.

---

### BB-109-15 — Medium — `build_policy_controller` returns `NULL_ACTION` outside offered actions

**File:** `motor/policy.py:229`

When `policy.emit()` returns `[]`, `choose` returns `NULL_ACTION`. But
`CallableController.choose` (the wrapper at `motor/voluntary.py:69`) validates
`action not in actions` and raises `ValueError` if `NULL_ACTION` isn't in the
offered action space.

**Failure scenario:** `policy.emit()` returns `[]` -> choose returns `NULL_ACTION`
-> `CallableController` raises ValueError if `NULL_ACTION not in actions`.

---

### BB-109-16 — Medium — `reset()` doesn't reset reflex stack state

**File:** `motor/organism_policy.py:96-98`

`MotorFreedomPolicy.reset()` only resets `self.scripted`. The `ReflexStack`'s tick
counters and `activation_rate` accumulate across episodes, meaning metrics from
episode N bleed into episode N+1.

**Failure scenario:** multiple episodes run -> `ReflexStack.activation_rate`
accumulates episode 1 data into episode 2 metrics.

---

### BB-109-17 — Medium — Stimuli frozen at construction time

**File:** `motor/organism_policy.py:93`

`self.stimuli = tuple(stimuli)` is set once in `__init__` and never updated. Every
tick sees the same stimuli — reflexes can't respond to dynamic environmental
changes (e.g. threat appearing/disappearing mid-episode). The `decide()` method
passes `self.stimuli` every tick with no hook to update it.

**Failure scenario:** threat appears mid-episode -> reflex stack never sees it
because stimuli were set at init and never change.

---

### BB-109-18 — Medium — `set_context_length(0)` silently uses full window

**File:** `brain/cortex/backbones.py:115`

```python
self._current_context = max(1, min(int(n), limit)) if n else limit
```

`n=0` is falsy in Python, so `if n` takes the `else` branch and sets
`_current_context` to the full window maximum. The caller's intent of "zero
context" (which should clamp to 1) is silently promoted to "maximum context" — the
exact opposite.

**Failure scenario:** `set_context_length(0)` -> `n=0` is falsy ->
`_current_context` set to `limit` (max) instead of being clamped to 1.

---

### BB-109-19 — Low — `bearing=0` (dead ahead) classified as "salience-left"

**File:** `motor/reflexes.py:184`

```python
kind = "salience-right" if bearing > 0 else "salience-left"
```

A bearing of exactly 0 degrees (dead ahead) is classified as `"salience-left"`,
triggering an orienting reflex that turns the organism left when the stimulus is
already centered. Should be filtered out or treated as no-orient.

**Failure scenario:** stimulus at `bearing_deg=0` (dead ahead) -> classified as
`"salience-left"` -> orienting reflex turns organism left unnecessarily.

---

### BB-109-20 — Low — NaN scores cause nondeterministic action selection

**File:** `motor/voluntary.py:57`

`max(range(len(actions)), key=scores.__getitem__)` with NaN values in `scores`
produces nondeterministic results because NaN comparisons are always False in
Python. The "best" action depends on list order and Python's `max` implementation
details.

**Failure scenario:** scorer returns NaN for some actions -> `max()` with NaN key
values gives implementation-dependent result.

---

### BB-109-21 — Low — Scalar (0D) tensor goal raises cryptic IndexError

**File:** `motor/policy.py:52`

A 0D scalar tensor has `dim() == 0` (passes `<= 2`), but `goal.shape[-1]` raises
`IndexError: tuple index out of range` on an empty shape tuple. Should give a
descriptive error instead.

**Failure scenario:** `goal` is 0D scalar tensor -> `goal.shape` is `()` ->
`goal.shape[-1]` raises `IndexError: tuple index out of range`.

---

## Acceptance Coverage Summary

| Phase | Bug-bash status |
|---|---|
| 0 — Identity | Broad automated coverage present; no additional defect confirmed. |
| 1 — Nursery | Broad automated coverage present; Crafter-dependent checks skipped when optional dep absent. |
| 2 — Cortex | Component tests exist, but metrics disconnected from promoted ladder checkpoint (BB-109-03). Backbone context-length bug (BB-109-18). |
| 3 — Neuromodulators | Component tests exist; no additional defect confirmed. |
| 4 — Hippocampus/dreams | Recall/export components exist, but consolidation does not consume them as required (BB-109-01). |
| 5 — Sleep | **Exit gate blocked** by missing generative replay and forgetting measurement (BB-109-01). |
| 6 — Motor | Component precedence works in isolation, but developmental integration/trend measurement absent (BB-109-02, BB-109-05, BB-109-06). Multiple code-level bugs (BB-109-11 through BB-109-17, BB-109-19 through BB-109-21). |
| 7 — Development | **Exit gate blocked**: stage controls are metadata-only and gates do not measure the carried brain (BB-109-02 through BB-109-08). |

---

## Recommended Triage Order

1. Implement Phase 5's real/dream replay mixer and forgetting experiment before
   treating concurrent scheduling as Phase 5 completion.
2. Decide what the single developmental checkpoint contains; it must include the
   predictive cortex that milestone gates evaluate.
3. Make the runner enforce scenario, senses, losses, and motor freedom, with an
   end-to-end test that inspects actual recorded actions/streams at each rung.
4. Compute every gate from the named organism's held-out records and weights; remove
   guaranteed synthetic gates and preserve metric semantics.
5. Add a supported CLI entry point for the built-in ladder and make its real
   Gestation->Crawling path a CI acceptance test.
6. Fix the code-level correctness bugs (BB-109-11 through BB-109-21), prioritizing
   the three high-severity ones in `_encode_goal` and caregiver-override handling.
