# V2 direction: a continuously-trained world model on Crafter

Response to the external critique of the V2 redesign. This doc records the
decisions taken in reply to that review, sharpens the project's single claim,
and answers the engineering critiques (async instability, generative-replay
bootstrap, speed, data handling) with concrete, buildable workarounds.
Optimization is explicitly out of scope here — these are correctness- and
stability-first choices, not performance tuning.

It is grounded in the code that exists today: the action-conditioned recurrent
world model (`training/action_world_model.py`), the async actor/learner split
(`training/async_trainer.py`), the `internal.*` modulation streams, and the
diagnosis in `docs/nursery-turn-in-place-analysis.md`. Where the critique's
concern is already handled in code, this doc says so; where it isn't, it
proposes the smallest thing that closes the gap.

---

## 0. First: this is a simulation, not biology

The critique spends two sections on "biology becoming load-bearing." That
concern dissolves once we state the actual position plainly:

**There is no biology in this system. None.** "Amygdala," "dopamine,"
"adrenaline," "reflex" are variable names for scalar signals and small heads —
a risk scalar, a reward-prediction-error scalar, a thresholded mode selector.
They are ergonomic labels, chosen because they are easier to hold in the head
than `risk_head_output`. They carry **zero** behavioral commitment.

The rule that keeps this honest — and it is the only rule needed:

> A label is never an argument. If a behavior is not in the code and not in a
> measured number, the biological word does not grant it.

Consequences, adopted:

- The "Arbiter" is a hand-authored 2×2 over two scalars (surprise,
  predicted-pain). It is a lookup table, so we call it a **mode selector** and
  describe it as thresholds, not as something that "arises" or "emerges."
  Scripted is fine; scripted-described-as-emergent is what invites the
  critique. (See §5c for making its inputs trustworthy.)
- The "reflex-integration curve" (reflex activation rate falling as the model
  learns to pre-empt it) is a genuinely interesting, **testable** signal — so
  we treat it as a metric to measure, never a property to assert by analogy.

Net effect: the entire "biology as self-deception" critique is answered not by
removing the labels but by demoting them to what they are. No section of the
system's *behavior* depends on a biological word being true, because none of
them are claims about biology.

---

## 1. The one claim

Refined and scoped to Crafter:

> **A single model, trained continuously and online, that learns the dynamics
> of the world it inhabits by predicting its own next observations — and acts
> to make those predictions come true.**

The engine is self-supervised and free: the world grades every guess, and
last tick's prediction is this tick's label. That is the load-bearing insight
and it is sound.

The falsifiable, Crafter-scoped version (this is the thing to prove, not just
plumbing):

1. **Prediction.** An action-conditioned, recurrent, multi-horizon predictor
   trained *purely online* (no fixed dataset epochs — each tick's observation
   is the label for the previous tick's prediction) beats copy-last on a
   rolling held-out set of seeds, and keeps improving rather than collapsing to
   a fixed point. The oracle-ratio methodology from the turn-in-place analysis
   is the measuring stick.
2. **Action from prediction.** The *same* weights, driven by a motor path that
   turns the next-frame forecast into the action that realizes a goal, forage
   and survive measurably better than random and scripted baselines on
   held-out seeds.
3. **Continuity.** Trained across changing regimes (§6), the model forgets
   measurably less than flat/offline training on the same data — the
   continual-learning result is the sharp contribution.

If (1) does not clear its bar, nothing above it matters — build and prove it
before anything else (§7).

### What this claim does *not* include (honesty about the nursery)

Crafter is 2-D and top-down; "turning" is a discrete facing flip, not
continuous rotation producing optical flow. So this stage learns **object
dynamics, action→effect causality, resource/inventory consequence, and
short-horizon prediction** — *not* ego-motion perception, parallax, depth
ordering, or optical flow. All of that language is dropped for the Crafter
stage. Ego-motion is a Minecraft-era goal, and (accepted) the learned weights
will **not** transfer across programs yet — moving to Minecraft means starting
the model over on a first-person world. The World seam keeps the *code*
world-agnostic; it does not make the *learning* portable, and we don't pretend
it does.

This is the direct answer to the critique's "glaring hole": we are not
claiming Crafter fixes ego-motion. We changed the goal to match the tool, not
the tool to match an unreachable goal.

---

## 2. Why Crafter fits (and where it doesn't)

**What Crafter buys us** — and it retires almost the entire "data-quality
findings" section of the turn-in-place analysis in one move:

- Determinism given a seed → the bit-exact replay smoke test that remote
  Minecraft could never support comes back.
- Pixel provenance is trivial: one renderer, one shape (64×64), no silent
  headless-GL fallback to a rotating minimap, no `pixel_source` ambiguity.
- Speed: fast, headless, no `xvfb`/`node-canvas-webgl`/`three` native-dep
  pain; env FPS stops being the bottleneck.
- Reward structure: 22 achievements give a real extrinsic signal for step (2)
  without hand-authoring a reward profile per scenario.
- Seeds actually vary the world (unlike the shared-world-state remote runs
  where "seeds do nothing").

**What Crafter does not buy us:** ego-motion perception (see §1). We take the
wins and scope the claim to what a top-down gridworld can actually teach.

---

## 3. Motor from prediction — the honest mechanism

The critique is right that "decode the T+1 forecast into the action that
fulfils it" as *active inference* (inverting/planning over a generative model)
is a research problem, and that Dreamer trains an actor in imagination rather
than inverting anything. We resolve the ordering the critique worried about by
choosing the low-risk mechanism as the **default**, not the optional
alternative:

- **Default — one-step planning / MPC over the world model.** Crafter has ~17
  discrete actions. Each tick, roll the world model forward one step for every
  candidate action, score each predicted next-state against the goal (predicted
  reward / achievement progress / calibrated novelty), and take the argmax.
  This *is* "motor from the predicted next frame," it is embarrassingly
  parallel over 17 candidates, it needs **no separate policy training**, it has
  **no policy-gradient instability**, and it improves automatically as the
  world model improves. With a decent world model this is exactly standard MPC
  and it works. Extend to a short (2–3 step) CEM/random-shooting search only if
  one-step is too myopic.
- **Later — Dreamer-style imagination actor** as the eventual spine if
  one-step planning plateaus: train an actor/critic in imagined rollouts. This
  is the proven path and becomes the real motor if MPC isn't enough.
- **Experiment only — active-inference inversion.** Kept as a labeled
  experiment, never the thing the project hinges on.

All three sit behind the same motor seam so they are A/B-swappable. The point:
the project's "organism acts to fulfil its predictions" story now rests on the
*most* proven component (MPC over a learned model), not the least.

---

## 4. Instabilities — async actor/learner

The critique flags async weight publication as known-finicky (staleness,
oscillation, nonstationarity). Two things are already true in the code and
worth stating before the fixes:

- **Failure isolation already exists** (`async_trainer.py`): the actor writes
  to a `SharedExperienceRing` (non-blocking) and polls a checkpoint file via
  `WeightSubscriber`; neither depends on the trainer being alive.
  `TrainerSupervisor` restarts a dead trainer and it resumes from checkpoint.
- The trainer already **grad-clips** (`grad_clip_norm=5.0`) and **publishes on
  an interval** (`publish_every_steps`), which bounds how much work a crash
  loses.

The single biggest stability lever, though, is a **design choice, not a knob**:

> Put the make-or-break online learning on the **world model**, whose objective
> is self-supervised regression (stable), and keep the policy as **one-step
> planning over it** (§3, no policy gradient at all). This sidesteps the great
> majority of async-RL instability by not doing async RL for the spine.

On top of that, small closable gaps:

- **Bound and smooth published weights.** Publish a Polyak/EMA copy of the
  learner weights rather than the raw latest; the actor sees a slow-moving
  target instead of tick-to-tick jitter. (One extra state-dict in the
  publisher.)
- **Bound staleness explicitly.** Stamp a monotonic version on each published
  checkpoint; have the actor log the actor-vs-learner version gap and warn if
  it grows without bound (learner falling behind ingestion). A simple counter,
  no new machinery.
- **Off-policy is fine here** because the world-model loss is supervised;
  resist the urge to add importance-sampling corrections now. Small LR +
  bounded replay + EMA weights is enough. (Revisit only if/when a Dreamer actor
  becomes the spine.)

---

## 5. Instabilities — generative replay, forgetting, calibration

### 5a. The dream bootstrap paradox

The critique is correct: dreams from a half-trained model reinforce that
model's own errors; generative replay only works once the generator is decent.
Workarounds:

- **Never train on dreams alone.** Keep a bounded **reservoir of real
  transitions** (reservoir sampling so the buffer represents the *whole* run,
  not just the recent window) and always mix real in.
- **Schedule the dream fraction on measured quality.** `dream_fraction = 0`
  until the model beats copy-last on the rolling held-out by a margin; ramp
  only as the held-out ratio improves; cap it (e.g. ≤ 0.5). The schedule is a
  function of a metric you already compute, not a constant.

### 5b. Catastrophic forgetting — and turning it into the contribution

- The **reservoir** above is also the primary anti-forgetting mechanism (real
  old data keeps getting replayed).
- **Rolling holdout as a forgetting detector.** Continuously hold out the most
  recent K seeds as validation; the existing statistical-evaluation harness
  (`training/statistical_evaluation.py`, confidence-interval regression test)
  flags a regression when the CI moves to the worse side — freeze/roll back the
  published weights on a confirmed regression.
- **Measure it, don't assert it.** Train sequentially across Crafter regimes
  (e.g. different seeds/biomes, or day→night difficulty), evaluate *backward
  transfer*, and report forgetting with-vs-without the reservoir and dream
  schedule. That measured delta is the falsifiable claim in §1(3).

### 5c. Uncertainty calibration (the mode selector depends on it)

The mode selector's correctness depends on calibrated surprise; an
uncalibrated σ/error head makes the mode flap tick-to-tick.

- **Produce uncertainty cheaply and honestly:** either a small ensemble (2–4
  world-model heads, disagreement = uncertainty) or a predicted-error head
  trained against realized error.
- **Calibrate it and report the calibration** (reliability diagram / temperature
  scaling on the rolling holdout) as a first-class metric — treat "is surprise
  trustworthy" as something measured, not assumed.
- **Add hysteresis** to the mode selector: require surprise to cross the
  threshold for *k* consecutive ticks before switching modes. Kills flapping
  even with imperfect calibration.

---

## 6. Processing speed and data handling

**Speed.** Crafter removes the headless-GL/xvfb pain and the ~17-tps, ~96%
-missed-tick jitter of the remote runs. The async design already decouples env
FPS from gradient FPS; on Crafter, env stepping is cheap (single-core,
thousands/s) and the learner is bounded by batch/GPU. Keep the visual encoder
small (Crafter is 64×64; downscale if useful). No further speed work now.

**Data handling** — carry forward every lesson the turn-in-place analysis paid
for:

- **One canonical observation shape, stamped and gated.** The single most
  expensive failure in the old dataset was a silent fallback that changed the
  observation distribution. Pin the Crafter shape, record it in session
  metadata, and keep the data-quality gate that refuses mixed provenance.
- **Horizons in ticks, not frames** (already fixed in `action_world_model.py`
  via `horizons_ticks_to_frames`) — keep it; Crafter's fixed step rate makes it
  clean.
- **Self-supervised label discipline.** "Yesterday's prediction is today's
  label" is a *delay buffer*: hold predictions for h ticks, then pair with the
  realized observation. Get the bookkeeping right per-horizon (the old bug was
  aliasing horizons against a sub-tick vision rate; Crafter removes the
  aliasing).
- **Bounded reservoir replay** on the existing `ReplayBuffer` /
  `SharedExperienceRing`, with reservoir sampling (§5a) so the buffer is a fair
  sample of the whole run.
- **Determinism restored.** Crafter is seed-deterministic, so the bit-exact
  replay smoke test (`runtime/replay.py`) applies again — use it as the
  plumbing regression test that remote Minecraft forced us to drop.

---

## 7. Re-sequencing: risk first, rename last (or never)

Adopting the critique's reorder, tuned to the decisions above. Each step gates
the next; do not start a step until the previous one is *measured*, not just
built.

1. **Port Crafter behind the World seam.** Fixed 64×64 encoder, discrete
   ~17-action space, seed-deterministic. Keep the brain world-agnostic (the
   seam already exists). Current names.
2. **Prove the world model online (make-or-break).** Action-conditioned
   recurrent multi-horizon predictor, trained *continuously* (not epoched),
   beats copy-last and keeps improving on a rolling held-out. This is §1(1).
   If it fails here, stop and fix here.
3. **Close the loop with MPC (§3).** One-step planning over the world model →
   achievements/survival > random and scripted on held-out seeds. This is
   §1(2): "acts to fulfil its predictions" via the least-risky mechanism.
4. **The continual-learning result (§5b, §6).** Sequential regimes, reservoir +
   dream schedule, measured backward transfer. This is §1(3) and the sharp,
   falsifiable contribution.
5. **Defer:** the rename, the React clinic, the Dreamer imagination actor, and
   active-inference inversion. All real, none on the risk path. The structure
   is the commitment; the labels can wait — or never come.

### Risks accepted going in

- One-step MPC may be too myopic for multi-step Crafter achievements → fallback
  is short-horizon CEM, then a Dreamer actor (§3), in that order.
- The intrinsic→extrinsic reward handoff (foraging after intrinsic-only stages)
  is a known curriculum failure point — watch it explicitly at step 3.
- Learning does not persist to Minecraft; the first-person world is a fresh
  start, by choice.
