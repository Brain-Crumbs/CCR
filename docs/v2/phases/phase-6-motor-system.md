# Phase 6 — The Motor System (voluntary + reflex stack)

> Master plan: [Phase 6](../02-implementation-plan.md#phase-6--the-motor-system-voluntary--reflex-stack).
> **Goal:** one voluntary path (planning over the world model) with a hardcoded
> reflex stack overriding it, full predicted-vs-actuated tracking, and nursery
> caregiver override.

## Dependencies

- **Phase 5** — a consolidating cortex worth planning over.
- **Phase 3** — the arbiter/mode signal that gates which motor path/reflex wins.

## Design commitment (the answer this phase is built on)

**MPC-first.** The default voluntary controller is **one-step planning over the
Predictive Cortex** — nothing in the motor path learns; the cortex it plans over is
the only thing that improves. Active-inference decoding, a DreamerV3 imagination
actor, and the policy head are **alternative controllers for A/B**, not the spine
([decision log #2](../direction-and-critique-response.md)). This is what keeps the
online loop stable (Phase 5).

## Builds on (existing code)

- `cognitive_runtime/brain/cortex/predictive.py` (Phase 2) — the model MPC rolls
  forward for each candidate action; `rollout(start_latent, actions, hidden)`.
- `cognitive_runtime/policies/actor_critic.py` — `ActorCriticPolicy`,
  `ActorCriticLearner` → the optional learned-policy **alt** controller
  (`motor/policy.py`).
- `cognitive_runtime/policies/scripted.py`, `policies/scripted_sequence.py`,
  `policies/null_policy.py` — scripted survival behaviours → migrate into configured
  reflexes; NULL → the basal-ganglia gate.
- `cognitive_runtime/core/orienting_reflex.py` — `OrientingReflex`,
  `OrientingDecision`, `_is_survival_critical` → first reflex in the stack.
- `cognitive_runtime/core/action_registry.py` — world-changing vs
  information-gathering classification (so reflexes/arbiter reason about an action's
  *kind* without knowing what it *is*).
- `cognitive_runtime/core/action.py`, `core/action_space.py` — the action space the
  world defines and the brain treats as opaque.
- `cognitive_runtime/programs/crafter/action_registry.py` (Phase 1) — Crafter's ~17
  discrete actions MPC enumerates.

## Tasks

1. **`motor/voluntary.py` — MPC over the cortex (the default).**
   - For each of Crafter's ~17 discrete actions, roll the cortex forward one step and
     score the predicted next state against the current goal (predicted reward /
     achievement progress / calibrated novelty, gated by the arbiter mode). Pick the
     best. MPC over ~17 actions is cheap and embarrassingly parallel.
   - **Nothing here learns.** No gradients in the motor path.
   - *Acceptance:* on a fixed cortex, MPC selects the highest-scoring action; the
     choice is deterministic given the cortex and goal.

2. **Keep three alt controllers behind the same seam.**
   - `motor.voluntary = mpc | active | imagination | policy`:
     **active-inference decoding** (T+1 output → encoder → motor inverse path —
     decode the forecast into the action that fulfils it); a **DreamerV3-style
     imagination actor** trained inside dreams (Phase 4/5 dreams); the existing
     **actor/critic policy head** (`policies/actor_critic.py` → `motor/policy.py`).
     These are experiments for A/B, not the spine.
   - *Acceptance:* all four controllers satisfy one `voluntary` interface and are
     swappable by config; MPC is the default.

3. **`motor/reflexes.py` — the hardcoded reflex stack (the "genome").**
   - A configured set of stimulus→action reflexes that **override** the voluntary
     action by priority. Migrate `OrientingReflex` (orient toward salience) and the
     threat/withdrawal response in first; move scripted survival behaviours in as
     *configured reflexes*, not "agent intelligence."
   - **Genotype/phenotype split:** the trigger *stimuli* are declared at the World
     interface (a World advertises a localizable threat, a looming object, a damage
     event); *which reflexes this organism has + their thresholds/priorities* is
     organism config.
   - *Acceptance:* a declared stimulus fires its reflex and overrides voluntary
     output; reflex set + thresholds come from organism config, stimuli from the
     World.

4. **Caregiver override — the top of the stack.**
   - A development-stage hook injecting motor commands directly (babbling / guided
     movement) at the top of the precedence stack (same slot as a reflex, driven from
     outside).
   - *Acceptance:* during a nursery stage the caregiver command supersedes both
     reflex and voluntary output.

5. **Enforce precedence + the basal-ganglia gate.**
   - `caregiver override > reflex (by priority) > voluntary`. NULL (inaction) remains
     a real, gated *voluntary* choice (basal-ganglia go/no-go), not the absence of a
     choice.
   - *Acceptance:* precedence holds in all combinations; NULL is recorded as a chosen
     voluntary action.

6. **Record the whole stack every tick.**
   - Log: the voluntary (predicted) action, which reflex fired and why (if any),
     whether a caregiver override applied, and the final actuated action — so
     *predicted vs actuated* is always reconstructable (the efference signal of "what
     I meant to do vs what my body did").
   - *Acceptance:* every tick's record contains all four fields; predicted-vs-actuated
     divergence is computable offline.

7. **Chart reflex-activation rate (reflex integration).**
   - Expose reflex-activation rate over development as a metric/series — expected to
     **fall** as the cortex learns to pre-empt its reflexes (human infant reflex
     integration, as a measured curve not a metaphor). Feeds the clinic.
   - *Acceptance:* the rate is emitted per session and trends downward as the cortex
     matures on a locomotion+threat scenario.

## Deliverables

- `motor/voluntary.py` (MPC default + 3 A/B controllers behind one seam).
- `motor/reflexes.py` (configured reflex stack; orienting + threat/withdrawal first;
  scripted behaviours migrated in).
- Caregiver override hook; enforced precedence; basal-ganglia NULL gate.
- Full predicted-vs-actuated per-tick record; reflex-activation-rate metric.

## Tests

- `tests/test_voluntary_motor.py`: MPC picks the best-scoring action on a fixed
  cortex; all four controllers satisfy the seam; nothing in the motor path takes a
  gradient step.
- `tests/test_reflexes.py`: a World-declared stimulus fires the right reflex and
  overrides voluntary; precedence `caregiver > reflex > voluntary` holds; NULL is a
  recorded voluntary choice.
- `tests/test_efference_record.py`: every tick records voluntary/reflex/override/
  actuated; divergence reconstructs correctly.

## Milestone 6 (exit gate)

In the **babbling** stage, caregiver-overridden motor produces clean forward/inverse
-model data; on a **locomotion+threat** scenario, a reflex demonstrably overrides the
voluntary action when its stimulus fires, the predicted-vs-actuated divergence is
logged, and the clinic charts **reflex-activation rate** (the curve expected to fall
as the cortex learns to pre-empt its reflexes). CI-runnable on held-out seeds where
feasible; a recorded-scenario assertion for the override behaviour.

## Risks / notes

- **The motor must not learn in the online loop.** MPC is the spine precisely
  because it adds no bootstrapped-policy instability on top of Phase 5's
  self-supervised cortex. The imagination actor / policy head are A/B experiments —
  keep them off the critical path.
- **Action space stays World-defined and opaque to the brain.** Reflexes and the
  arbiter reason about an action's *kind* (world-changing / information-gathering)
  via `action_registry.py`, never about what it *is*.
- Reflexes are the organism's **genetic priors** — they do not learn. Reflex
  *integration* (the falling activation rate) is the cortex learning to pre-empt
  them, not the reflexes changing.
