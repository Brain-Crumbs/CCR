# Neural Stream Agent Target

This document defines the intended end-state for CCR: a Minecraft-born agent
that learns continuously from input streams, builds latent representations,
learns object permanence/world dynamics, and improves its policy online from
reward.

The current online Q learner is only a baseline.  The target architecture is a
neural, stream-native agent.

## Target Shape

The desired agent is built around trainable input modules:

```text
video / pixels        -> vision CNN / video encoder        -> visual latent
keyboard / controls   -> motor encoder                     -> motor latent
audio                 -> audio CNN / spectrogram encoder   -> audio latent
inventory / entities  -> embedding or small MLP            -> symbolic latent
body state / rewards  -> scalar MLP                        -> body/reward latent

[all stream latents + memory state]
  -> learned latent fusion model
  -> learned world model
  -> actor / critic policy heads
  -> Minecraft action mapping
```

The agent should learn in the loop:

1. Observe streams.
2. Encode each stream into latent space.
3. Fuse stream latents into an agent state.
4. Predict what comes next.
5. Choose an action.
6. Receive reward and new observations.
7. Update encoders, fusion, world model, and policy from the transition.
8. Save checkpoints often enough that interruption is survivable.

The mental model is closer to a child learning than to a hand-written bot:
first visual regularities, then objects, then object permanence, then
affordances, then reward-directed behavior.

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
- Neural module contracts (`cognitive_runtime/neural/`): stream encoders,
  learned fusion, an MLP world model, and now (Phase E) an MLP actor/critic
  policy/value pair with an `OnlineOptimizer` online update, behind
  `--policy actor-critic`.

Important limitation:

`--policy online` uses fixed stream encoders plus a linear Q model; it stays
as the baseline. `--policy actor-critic` trains an MLP policy/critic (and
optionally an MLP world model) over the fused latent state, but the fused
latent itself is still `TemporalFusion`'s fixed concatenation -- CNN stream
encoders and learned fusion are not yet wired into the online policy path.

## Necessary Changes

### 1. Define The Neural Module Contracts

Add first-class contracts for:

- `StreamEncoderModule`: trainable per-input encoder.
- `LatentFusionModel`: maps per-stream latent slices into one agent state.
- `WorldModel`: predicts next latent state, reward, terminal/death, risk, and
  prediction error.
- `PolicyModel`: maps fused latent and world-model outputs to action logits.
- `ValueModel` / critic: predicts expected return.
- `OnlineOptimizer`: owns losses, gradient steps, clipping, target networks,
  and checkpoint state.

These should be PyTorch-backed but isolated so the runtime can still import
without torch unless a neural policy is selected.

### 2. Make Input Streams Explicit

The target project should focus on streams that represent actual agent input
and action:

- video/pixels
- audio
- keyboard/control/motor history
- mouse/look controls if used
- body state
- reward
- inventory/entity/world facts only when they come from Minecraft backend
  streams and are treated as inputs, not privileged control logic

Every stream needs:

- shape/schema
- sample rate
- encoder module
- latent width
- checkpoint keys
- train/eval behavior

### 3. Replace Fixed Fusion With Learned Fusion

Current `TemporalFusion` concatenates fixed hand-written encoder outputs.
Keep it as a debugging baseline, but add a learned fusion path:

```text
stream_id -> latent slice + mask + recency
all slices -> fusion transformer/CNN/MLP -> fused agent state
```

The fusion model should know which streams are present, stale, or missing.  It
should not assume every sensor fires every tick.

### 4. Add Representation Learning Losses

Before the agent can be expected to survive well, it needs pressure to model
the world.  Add losses for:

- next visual latent prediction
- contrastive visual consistency across adjacent frames
- object permanence: predict hidden/stale object latents after occlusion
- reward prediction
- damage/death/risk prediction
- action-conditioned next-state prediction
- novelty/prediction-error signals

This is where "baby learns objects" begins.  Object identity can start as a
latent slot or entity-token prediction before becoming a full object-centric
model.

### 5. Add Actor/Critic Online Learning

Linear Q is too weak for the target.  Add a neural policy stack:

```text
stream modules -> learned fusion -> world model features
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
look, inventory, attack/use, and crafting.

### 6. Add Replay Buffer And Mixed Training

The agent should learn from live experience and recent history:

- online transition buffer
- prioritized replay by reward, death, damage, novelty, and prediction error
- short on-policy updates every tick/window
- periodic replay minibatches
- checkpointed optimizer state

Recorded sessions should be reusable for pretraining and regression.

### 7. Make Minecraft The Development Nursery

Minecraft remains the test world, but should provide sensory input rather than
scripted intelligence:

- Mineflayer backend for live experience.
- Simulated backend for deterministic regression.
- Reward goals for survival, exploration, tool use, shelter, light, food.
- Curriculum configs: flat safe world, resource world, night survival, caves,
  combat, crafting.

The backend should expose raw/near-raw streams where possible.  Semantic streams
are useful for debugging and auxiliary losses, but the core agent should not
depend on hand-written survival heuristics.

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
is no way to understand what it saw, did, predicted, or learned.

## Proposed Implementation Phases

### Phase A: Neural Stream Package

- Add `cognitive_runtime/neural/`.
- Move torch imports behind lazy neural modules.
- Add base module interfaces and checkpoint bundle format.
- Add tests that import the non-neural runtime without torch.

### Phase B: Pixel Stream Encoder

- Convert `VisionPolicyNet` into a reusable `PixelStreamEncoder`.
- It outputs a visual latent, not actions.
- Add reconstruction/next-latent prediction losses.

### Phase C: Learned Fusion

- Add `LatentFusionModel`.
- Support stream masks and recency.
- Train from recorded sessions to predict actions/rewards/next latents.

### Phase D: World Model And Object Permanence

- Add action-conditioned prediction.
- Add losses for next latent, reward, death/risk, and entity persistence.
- Track prediction error as curiosity/novelty input.

### Phase E: Actor/Critic Policy

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

### Phase F: Live Childhood Runs

- Start each run from a checkpoint.
- Save frequently.
- Record every session.
- Use dashboard and viewer after each curriculum step.
- Compare against baselines before increasing difficulty.

## Success Criteria

The project is aligned with the target when:

- each major input stream has a trainable encoder or a deliberate fixed stub
- learned fusion is the primary policy input
- the agent predicts next state/reward/death
- online updates mutate neural weights during play
- checkpoints include encoders, fusion, world model, policy, critic, optimizer,
  replay metadata, and training stats
- simulation shows reproducible improvement
- live Minecraft runs do not crash and can resume after interruption
- recorded sessions can explain what the agent saw, predicted, did, and learned
