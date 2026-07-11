# Neural Stream Agent Target

This document defines the intended end-state for CCR: a Minecraft-born agent
that learns continuously from input streams, builds latent representations,
learns object permanence/world dynamics, and improves its policy online from
reward.

The current online Q learner is only a baseline.  The target architecture is a
neural, stream-native agent — and, beyond that, an *attentive* one: a system
that manages its own limited perception, memory, compute and action bandwidth
the way biology does, updating its current most-important context based on
which sense is being triggered and reacting to it (looking at it, moving
toward or away from it).

## Target Shape

The desired agent is built around trainable input modules:

```text
video / pixels        -> vision CNN / video encoder        -> visual latent
keyboard / controls   -> motor encoder                     -> motor latent
audio                 -> audio CNN / spectrogram encoder   -> audio latent
inventory / entities  -> embedding or small MLP            -> symbolic latent
body state / rewards  -> scalar MLP                        -> body/reward latent
internal signals      -> scalar MLP                        -> interoceptive latent

[all stream latents + memory state]
  -> attention controller (budgeted salience; selects what matters this tick)
  -> learned latent fusion model (weighted by attention)
  -> learned world model (multi-horizon: t+1, t+5, t+20; with uncertainty)
  -> actor / critic policy heads (+ scripted orienting reflex below them)
  -> Minecraft action mapping
```

The agent should learn in the loop:

1. Observe streams.
2. Encode each stream into latent space.
3. Score attention: which streams deserve high-resolution processing this
   tick, under a budget.  A salience spike (novelty, damage, reward change,
   prediction error) captures focus bottom-up.
4. Fuse the attended stream latents into an agent state.
5. Predict what comes next — at multiple horizons, with uncertainty.
6. Choose an action.  Salient localizable stimuli trigger orienting
   (look/turn toward the source); predicted pain repels before it lands.
7. Receive reward and new observations.
8. Compute internal modulation signals (prediction error, reward prediction
   error, learning progress, novelty, risk) and publish them as streams.
9. Update encoders, fusion, world model, and policy from the transition.
10. Save checkpoints often enough that interruption is survivable.

The mental model is closer to a child learning than to a hand-written bot:
first visual regularities, then objects, then object permanence, then
affordances, then reward-directed behavior.  Attention is what makes that
tractable: the agent cannot process everything, so it must learn what is
worth its compute — and moving its eyes (camera, mouse, body) is part of
attending, not separate from it.

### Motivation: safe surprise

The drive design (issue #61) rewards neither raw prediction accuracy
(wall-staring becomes optimal) nor raw prediction error (irreducible noise
becomes fascinating and cliffs become interesting).  Instead:

- **learning progress** — prediction error that is *improving* is rewarded;
  mastered scenes and pure noise both self-extinguish;
- **safe novelty** — surprising, less-predictable situations are sought out,
  gated by predicted risk;
- **predicted-pain aversion** — the world model's risk head makes predicted
  injury/death aversive *before* it happens, so avoidance is anticipatory.

Surprise that does not forecast suffering is curiosity; surprise that does is
a warning.  Internal modulation signals (issue #58) are published as
`internal.*` streams — recordable, replayable interoception, consumed by the
attention controller, the reward profile, and replay prioritization alike.

## What Exists Today

Useful pieces already in CCR:

- Stream-native runtime loop.
- Minecraft simulated and Mineflayer remote backends.
- Stream logs, dashboard, episode viewer, replay for deterministic simulation.
- `vision.frame.pixels` stream.
- Optional pixel CNN behavioral-cloning model.
- `TrainableStreamModule` interface and fixed wrappers.
- Online checkpoint lifecycle.
- Online Q baseline over the fixed fused latent state.
- The Phase A–E neural stack, landed in first-cut form
  (`cognitive_runtime/neural/`): checkpoint bundle format (#20), stream
  schema registry (#21), reusable `PixelStreamEncoder` (#22), visual
  representation losses (#23), trainable encoders for motor/body/reward
  streams plus an audio stub (#24), `LatentFusionModel` (#25), an
  action-conditioned MLP world model (#26), entity persistence and a
  combined novelty signal (#27), a prioritized replay buffer (#28), and an
  MLP actor/critic with an `OnlineOptimizer` behind `--policy actor-critic`
  (#29).  Curriculum world/reward presets landed via #30.

Important limitations:

- `--policy online` uses fixed stream encoders plus a linear Q model; it
  stays as the baseline.  `--policy actor-critic` trains an MLP
  policy/critic (and optionally an MLP world model) over the fused latent
  state; the fused latent itself defaults to `TemporalFusion`'s fixed
  concatenation, but `--fusion learned` (issue #57) now runs trainable
  stream encoders (`cognitive_runtime.neural.live_fusion`, bound per-stream
  from each `StreamDeclaration.neural_encoder`) through `LatentFusionModel`
  in the live tick instead — the fused agent state the policy/critic consume
  either way, so no downstream consumer needs to know which mode produced
  it. `LatentFusionModel.forward` also takes an optional per-stream
  `attention_weights` argument now (uniform/all-ones by default, byte-
  equivalent to omitting it) so the attention controller (#59) can plug in
  later without another interface change. A checkpoint's fusion mode is part
  of its compatibility hash: resuming under the other mode fails loudly
  instead of silently feeding the policy a differently-shaped, differently-
  meaning vector. Training the trainable encoders + fusion model today is a
  small synchronous online update (a reward-prediction auxiliary head,
  every tick) behind the same checkpoint/learner interfaces the async
  trainer (#37) will eventually replace it through — still CNN pixel
  encoding is out of scope (`vision.frame.pixels` stays a deliberate raw
  stub in both fusion modes) pending a trainable `PixelStreamEncoder`
  binding.
- Per-tick prediction error and novelty are computed but only written into
  decision records; they are not yet first-class `internal.*` streams
  (issue #58).
- There is no attention system: every stream contributes equally to the
  fused state every tick, and nothing records what mattered (issue #59).

## Necessary Changes

### 1. Define The Neural Module Contracts

Add first-class contracts for:

- `StreamEncoderModule`: trainable per-input encoder.
- `LatentFusionModel`: maps per-stream latent slices into one agent state.
- `WorldModel`: predicts next latent state, reward, terminal/death, risk, and
  prediction error — at multiple horizons (t+1, t+5, t+20; issue #39), each
  with an uncertainty estimate.
- `PolicyModel`: maps fused latent and world-model outputs to action logits.
- `ValueModel` / critic: predicts expected return.
- `OnlineOptimizer`: owns losses, gradient steps, clipping, target networks,
  and checkpoint state.

These should be PyTorch-backed but isolated so the runtime can still import
without torch unless a neural policy is selected.  *Status: contracts landed
(#19); the multi-horizon/uncertainty extension of the world model landed
(#39, `MultiHorizonMLPWorldModel`).*

### 2. Make Input Streams Explicit

The target project should focus on streams that represent actual agent input
and action:

- video/pixels
- audio
- keyboard/control/motor history
- mouse/look controls if used
- body state
- reward
- internal modulation signals (`internal.*`, issue #58) — interoception is
  agent input too
- inventory/entity/world facts only when they come from Minecraft backend
  streams and are treated as inputs, not privileged control logic

Every stream needs:

- shape/schema
- sample rate
- encoder module
- latent width
- checkpoint keys
- train/eval behavior
- attention metadata: modality, relative compute cost, and whether the
  stream can carry a direction/region localization hint (issues #32, #59,
  #60)

*Status: registry landed (#21); the raw-vs-aux-vs-privileged classification
and attention metadata landed (#32) -- every `StreamDeclaration` carries a
`classification` (agent_input/aux_debug/privileged) and every agent-input
declaration an `AttentionMetadata` (modality, expected sample rate, relative
compute cost, localization hint); `ccr run --input-profile raw` and
`training.datasets.build_neural_dataset(..., stream_profile="raw")` fuse only
the agent-input subset for pixel-only-vs-pixels+semantics ablations
(docs/streams.md "Stream classification and attention metadata").*

### 3. Replace Fixed Fusion With Learned Fusion

Current `TemporalFusion` concatenates fixed hand-written encoder outputs.
Keep it as a debugging baseline, but add a learned fusion path:

```text
stream_id -> latent slice + mask + recency + attention weight
all slices -> fusion transformer/CNN/MLP -> fused agent state
```

The fusion model should know which streams are present, stale, or missing.
It should not assume every sensor fires every tick — and it should accept
per-stream attention weights (uniform by default) so the attention
controller can plug in without an interface change.

*Status: `LatentFusionModel` landed (#25) and is now wired into the live
policy path behind `--fusion learned` (#57), with the attention-weight hook
(uniform by default) on `LatentFusionModel.forward`.*

### 4. Add Representation Learning Losses

Before the agent can be expected to survive well, it needs pressure to model
the world.  Add losses for:

- next visual latent prediction
- contrastive visual consistency across adjacent frames
- object permanence: predict hidden/stale object latents after occlusion
- reward prediction
- damage/death/risk prediction
- action-conditioned next-state prediction, at multiple horizons
- novelty/prediction-error signals

This is where "baby learns objects" begins.  Object identity can start as a
latent slot or entity-token prediction before becoming a full object-centric
model.  *Status: first cuts landed (#23, #26, #27); the generative
multi-horizon world model with uncertainty landed (#39).*

### 5. Add Actor/Critic Online Learning

Linear Q is too weak for the target.  Add a neural policy stack:

```text
stream modules -> attention -> learned fusion -> world model features
  -> actor head: action distribution
  -> critic head: value estimate
```

The online update should use:

- one-tick delayed reward attribution
- advantage/value loss
- entropy bonus for exploration
- reward normalization
- gradient clipping
- checkpoint every N ticks
- eval mode with no mutation

Start with discrete Minecraft actions.  Later split action heads into movement,
look, inventory, attack/use, and crafting.  *Status: first cut landed (#29);
gates against baselines are #31; gradient steps move off the tick thread via
the async actor/learner split (#37).*

### 6. Add Replay Buffer And Mixed Training

The agent should learn from live experience and recent history:

- online transition buffer
- prioritized replay by reward, death, damage, novelty, and prediction error
  (priorities read from the recorded `internal.*` streams, #58)
- short on-policy updates every tick/window
- periodic replay minibatches
- checkpointed optimizer state

Recorded sessions should be reusable for pretraining and regression.
*Status: buffer landed (#28); the shared live+recorded trainer is #37.*

### 7. Make Minecraft The Development Nursery

Minecraft remains the test world, but should provide sensory input rather than
scripted intelligence:

- Mineflayer backend for live experience.
- Simulated backend for deterministic regression.
- Reward goals for survival, exploration, tool use, shelter, light, food.
- Curriculum configs: flat safe world, resource world, night survival, caves,
  combat, crafting (landed, #30; automated stage promotion is #43).
- **Nursery scenario suite (issue #62)** below the survival curriculum:
  scripted micro-scenarios (`walk_forward`, `turn_in_place`,
  `object_permanence`, `day_night`, …) that each isolate one worldly
  regularity, generate clean recorded sessions, and benchmark multi-horizon
  prediction (t+1, t+5, t+20) on held-out seeds.  This is stage zero of the
  childhood: the world model learns that the world is lawful before the
  policy learns to survive in it.

The backend exposes raw/near-raw streams where possible (landed, #32): every
published stream is classified agent_input/aux_debug/privileged in the
stream registry, `input.mouse_look` (mouse/look control history) is now a
published motor stream on both backends, and the Mineflayer bridge can
optionally capture real rendered pixels via prismarine-viewer
(`bridge/mineflayer/pixels.js`, best-effort, resized to the fixed pixel
shape) instead of the colorized semantic-grid fallback.  Semantic streams
(`world.front_block`, `world.sheltered`, `vision.entities`, `event.*`) are
useful for debugging and auxiliary losses and stay classified `aux_debug`,
but `ccr run --input-profile raw` excludes them from the online/actor-critic
policy's fused state so the core agent does not depend on hand-written
survival heuristics.

### 8. Add Internal Modulation (The Dopamine Analog)

Issue #58.  Model neuromodulation as *internal streams*, not hidden
variables — the runtime already thinks in streams, so interoception should be
recorded, replayed and consumed exactly like sensory input:

- `internal.prediction_error` — world-model next-latent error.
- `internal.reward_prediction_error` — actual minus predicted reward; the
  dopamine analog.  It modulates replay priority, memory tagging and
  salience — it is *not* the whole attention system.
- `internal.learning_progress` — is prediction error improving here?
- `internal.novelty` — the combined novelty scalar, promoted to a stream.
- `internal.risk` — predicted pain/injury/death, made visible.

### 9. Add Attention As A Runtime Subsystem

Issue #59 (deterministic first), #63 (neural successor, gated).  Attention is
resource allocation, not a Transformer block: which streams get
high-resolution processing this tick, what enters the fused context, and —
via the orienting reflex — where the sensors move next.

- Per-stream `AttentionSignal`: novelty, prediction error, uncertainty,
  reward relevance, risk, staleness, repetition, compute cost.
- `AttentionController` + `AttentionBudget`: deterministic scoring under a
  hard budget, with bottom-up capture (a spiking sense displaces the current
  focus) and dwell/hysteresis (focus does not thrash).
- `AttentionState` recorded every tick with per-stream reasons — the first
  milestone is *observability*: every tick can explain what the agent
  attended to and why.
- Weights feed learned fusion (#57's hook) and the policy's context.

### 10. Add The Orienting Reflex And Active Perception

Issue #60.  In an embodied agent, attention acts through motor output:
looking at something is an action.  Camera/look/mouse movement, waiting and
inspecting are information-gathering actions, distinct in the action
registry from world-changing actions.  A scripted, brainstem-style reflex
turns toward localizable salient stimuli (bounded, recorded, vetoed by high
predicted risk and survival-critical policy actions); the learned policy can
later inherit or override it.

### 11. Add The Intrinsic Drive

Issue #61, plugged into the reward-profile schema (#41) as weighted,
per-stage components: learning progress + risk-gated novelty − predicted
risk.  Nursery stages (#62) may run intrinsic-only; quest stages run mostly
extrinsic.  See "Motivation: safe surprise" above for the design rationale.

## What To Remove Or Deprecate

Remove only after replacement tests exist.  Some current pieces are useful
scaffolding even if they are not the final agent.

### Keep

- `core/streams/`
- runtime loop
- recorder/dashboard/view tools
- Minecraft simulated backend
- Mineflayer remote backend
- reward streams and reward components
- checkpoint/replay infrastructure
- `TrainableStreamModule`
- neural pixel BC as an offline bootstrap path

### Deprecate Once Neural Policy Is Stable

- `OnlineQModel`, `OnlineQPolicy`, `OnlineQLearner`
  - Keep as a baseline until actor/critic beats it.
- fixed hand-written stream encoders
  - Keep as debug/reference encoders and for tests.
- `LearnedPolicy` linear BC
  - Keep only as an offline baseline.
- handcrafted feature extraction in `training/features.py`
  - Keep temporarily for parity tests; remove from the main training path.
- scripted survival policy
  - Keep as a teacher/baseline, not as agent intelligence.
- random/null policies
  - Keep for smoke tests and lower-bound metrics.
- the deterministic attention controller, eventually
  - Only after the neural scorer (#63) beats it statistically (#44); it then
    remains as debug/fallback, never deleted.

### Candidates To Remove From The Main Product Path

These are not aligned with the final neural stream agent except as tools/tests:

- generic `StructuredPerception` numeric flattening
- hand-authored Minecraft survival feature logic
- linear softmax behavioral cloning as the primary learned policy
- dependency-free linear online Q as the primary online learner
- any reward or policy code that directly encodes a fixed survival strategy

### Do Not Remove

Do not remove the stream recorder, dashboard, replay, or Minecraft backends.
They are how the agent's childhood gets inspected.  Without recordings, there
is no way to understand what it saw, did, predicted, or learned.  (Note the
re-scope in issue #44: byte-identical replay becomes a fast sim smoke test of
the plumbing, while *statistical* evaluation over N episodes becomes the
regression gate for learning runs — live continuous training was never going
to be byte-reproducible.)

## Proposed Implementation Phases

### Phase A: Neural Stream Package — landed

- `cognitive_runtime/neural/` with torch-optional contracts (#19) and the
  checkpoint bundle format (#20).

### Phase B: Pixel Stream Encoder — landed

- Reusable `PixelStreamEncoder` (#22) with reconstruction/next-latent
  prediction losses (#23); trainable non-vision encoders + audio stub (#24).

### Phase C: Learned Fusion — landed, live behind `--fusion learned`

- `LatentFusionModel` with stream masks and recency (#25).
- Wired into the live actor/critic path via `--fusion learned`, with the
  attention-weight hook, landed (#57): trainable stream encoders
  (`cognitive_runtime.neural.live_fusion.build_trainable_encoder_registry`,
  bound per stream from `StreamDeclaration.neural_encoder`) feed
  `LatentFusionModel` every tick; `--fusion fixed` (`TemporalFusion`) stays
  the default until #31's gates pass a learned-fusion run. The checkpoint
  bundle carries the trainable encoders and fusion model, keyed by a
  fusion-mode-tagged compatibility hash distinct from the fixed path's, so a
  checkpoint trained under one mode fails loudly if resumed under the
  other. Online training of the encoders + fusion model is a small
  synchronous reward-prediction update for now; the async trainer (#37)
  owns replacing it with the full transition-based training loop.

### Phase D: World Model And Object Permanence — first cut landed

- Action-conditioned prediction (#26), entity persistence + novelty (#27).
- Multi-horizon, uncertainty-aware world model landed (#39):
  `MultiHorizonMLPWorldModel` (`cognitive_runtime/neural/world_model.py`)
  predicts `(next_latent, reward, terminal, risk, prediction_error)` at every
  configured horizon (default t+1/t+5/t+20) over the fused agent-state
  latent, each with a learned `uncertainty` field trained via a
  heteroscedastic NLL loss (`ccr train --model-type
  multi-horizon-world-model`); `next_latent` is checked per horizon against
  copy-last-latent and mean-latent baselines
  (`training.world_model.evaluate_multi_horizon_model`), and uncertainty is
  checked against realized error (`uncertainty_calibration`). `forward()`
  stays a plain single-step `WorldModelOutput` (horizon 1, no uncertainty)
  so every existing caller (`ActorCriticOptimizer`, the `NeuralWorldModel`
  loop bridge) is unaffected. The ego-motion canary
  (`cognitive_runtime/training/ego_motion_canary.py`, `ccr
  ego-motion-canary`) records `walk_forward` (constant `MOVE_FORWARD`,
  optional action noise) episodes at multiple seeds via the simulated
  backend, pretrains a pixel encoder/decoder/next-latent predictor on a
  train-seed subset, fine-tunes it against a horizon-consistency loss (the
  iterated-rollout alternative to dedicated per-horizon heads), and reports
  per-horizon PSNR/SSIM against copy-last-frame/mean-frame baselines on
  held-out seeds. In the current simulated backend the `vision.frame.pixels`
  render changes very little tick to tick (the constant-forward walker
  routinely stalls against terrain within the first few dozen ticks), which
  makes copy-last an unusually strong baseline for any lossy encoder/decoder
  to beat -- the canary harness and its held-out-seed evaluation are
  complete and CLI-wired, but actually clearing the "beats both baselines"
  bar needs either a more dynamic simulated render or the mineflayer remote
  backend. The `walk_forward` scenario formally lands in the nursery suite
  (#62), which owns tuning the environment for genuine motion; this module
  is the reusable benchmark it is expected to call into.

### Phase E: Actor/Critic Policy — first cut landed

- Add neural online policy. Landed: `MLPPolicyModel`/`MLPValueModel`/
  `ActorCriticOptimizer` (`cognitive_runtime/neural/`), wired to the runtime
  as `--policy actor-critic` (`cognitive_runtime/policies/actor_critic.py`).
- Train in simulation first. Landed: a smoke acceptance run
  (`cognitive_runtime/training/actor_critic_acceptance.py`) checks it beats
  random on identical seeds.
- Evaluate against random/scripted/linear Q. Landed (issue #31): the
  evaluation-gate harness `cognitive_runtime/training/evaluation_gates.py`, wired
  as `python -m cognitive_runtime evaluation-gates`. It trains the actor/critic
  *and* the linear online-Q baseline in simulation, evaluates both plus
  scripted and random with no mutation on identical seeds, and reports three
  gates: (1) actor/critic > random (hard requirement), (2) actor/critic >
  linear Q (unlocks deprecating `OnlineQ*` as primary), (3) reproducible
  improvement across reruns with the same seeds. Recorded eval sessions feed
  the existing dashboard; the gate report is written into the checkpoint
  bundle's training stats.
- Only then run live Mineflayer fine-tuning (issue #33).

### Phase F: Live Childhood Runs — landed

- Start each run from a checkpoint, or explicit `--fresh`: the CLI refuses a
  live (`--backend remote`) run with neither (issue #33).
- Save frequently: periodic tick-count checkpointing, plus checkpoint on
  clean shutdown, uncaught exception, `KeyboardInterrupt`, and a recoverable
  bridge crash (`BridgeError` / `RecoverableEpisodeError`,
  `cognitive_runtime/core/program.py`) -- a dropped live connection ends that
  episode, not the process; the next episode's `reset()` respawns the bridge.
- Record every session, frames included -- enforced automatically for live
  runs, not opt-in.
- Model-side streams (`model.novelty`, `model.value_estimate`) are recorded
  every tick so a session explains what the agent predicted, not just what
  it did.
- `review` (`cognitive_runtime/tools/review.py`) is the post-run command:
  summarize a run, compare it against baseline sessions on the same
  curriculum, and show per-episode detail in one call
  (`docs/childhood-runs.md`).
- Workflow documented end-to-end in `docs/childhood-runs.md`: pretrain in
  sim → eval gates → live fine-tune → review → next curriculum step.
- Curriculum promotion automation is #43.

### Async Actor/Learner Split — landed (issue #37)

Continuous training must never block the realtime tick. Neural gradient
steps don't fit in a tick budget, so the synchronous in-tick
`ActorCriticLearner` (Phase E) now has an actor/learner-split counterpart:

- `cognitive_runtime/neural/experience_queue.py`'s `SharedExperienceRing`:
  a bounded shared-memory ring of transitions the actor pushes to every
  tick (O(1), never blocks) and a separate trainer process drains in
  batches. Backpressure is drop-oldest.
- `cognitive_runtime/neural/weight_publisher.py`'s `WeightPublisher`/
  `WeightSubscriber`: the trainer publishes versioned checkpoint snapshots;
  the actor polls and hot-swaps them into its live policy/critic modules
  in place, between ticks.
- `cognitive_runtime/training/async_trainer.py`'s `AsyncTrainer`: the
  learner side -- optimizer, replay buffer, checkpoint ownership. The same
  `ReplayBuffer` is fed from a live ring and/or `load_session_into_buffer`
  recorded sessions (issue #28), so live training and offline pretraining
  are the same code path with different source mixes.
  `spawn_trainer_process`/`TrainerSupervisor` run it in a real OS process
  (not a thread -- the GIL), isolated so a trainer crash never kills the
  actor; a restarted trainer resumes from the last published checkpoint.
- `cognitive_runtime.policies.actor_critic.AsyncActorCriticLearner`: the
  actor side. Requires zero changes to `runtime/loop.py` -- it satisfies
  the same `Learner` contract `ActorCriticLearner` does, but only captures
  data and polls for new weights instead of taking gradient steps inline.
- CLI: `ccr run --policy actor-critic --async-trainer` wires the split for
  a live run; `ccr trainer --sessions ... --out ...` runs an `AsyncTrainer`
  to completion in the foreground, pointed only at recorded sessions with
  no live actor, for offline pretraining.

### Phase G: Attention, Modulation And Motivation

The follow-up roadmap (2026-07), in dependency order:

1. **#37** async actor/learner split and **#58** internal modulation streams
   — independent of each other; both unblock nearly everything.
2. **#57** learned fusion into the live path and **#39** multi-horizon
   generative world model — the learning stack.
3. **#59** deterministic attention controller (needs #58) and **#62**
   nursery scenario suite (needs #37/#39) — parallel tracks.
4. **#41** reward profiles → **#61** risk-gated surprise intrinsic drive →
   **#60** orienting reflex.
5. **#44/#31** referee everything; **#43/#42/#33/#34** proceed as planned;
   **#63** (neural attention, episodic memory retrieval) stays gated until
   #57 + #59 have produced data to learn from.

The guiding constraint: build the smallest living loop that measurably
learns before widening the senses.  The single most important upcoming
milestone is the first `walk_forward` nursery benchmark where the world
model beats copy-last-frame on held-out seeds — everything before that is
promise; that is proof.

## Success Criteria

The project is aligned with the target when:

- each major input stream has a trainable encoder or a deliberate fixed stub
- learned fusion is the primary policy input
- the agent predicts next state/reward/death — at t+1, t+5 and t+20, with
  calibrated-enough uncertainty to tell novelty from noise
- internal modulation (`internal.*`) is recorded every tick like any sense
- every tick records what the agent attended to and why, and salient stimuli
  visibly capture focus and trigger orienting
- the intrinsic drive demonstrably prefers novel low-risk situations over
  both boring and dangerous ones (the three-region test, #61)
- nursery scenarios show the world model beating trivial baselines on
  held-out seeds before survival training begins
- online updates mutate neural weights during play without missing ticks
- checkpoints include encoders, fusion, world model, policy, critic,
  optimizer, replay metadata, attention/modulation baselines, and training
  stats
- simulation shows statistically reproducible improvement (#44)
- live Minecraft runs do not crash and can resume after interruption
- recorded sessions can explain what the agent saw, predicted, attended to,
  did, and learned
- none of the attention/modulation machinery knows a single Minecraft
  concept — swap the Program and the same organism inhabits a new world
