# V2 Onboarding Guide — From Zero to the Whole System

This is the start-here guide for understanding CCR V2 as a system rather than
as a collection of files. It explains the research idea, runtime substrate,
biological architecture, current code layout, and main workflows. Use
[04-contracts-and-data-flow.md](04-contracts-and-data-flow.md) as the exact
interface reference and [05-presentation-runbook.md](05-presentation-runbook.md)
when teaching the project to someone else.

## What you should understand after reading this

You should be able to answer five questions:

1. What is CCR trying to prove?
2. What happens during one waking tick?
3. How do the World, streams, memory, cortex, motor system, sleep, and Record fit
   together?
4. Which code path should you follow for a live run, cortex training, dreaming,
   development, or the Clinic?
5. Which parts form one running system today, which are specialized pipelines,
   and which are deliberately deferred?

## The project in one paragraph

CCR is a research runtime for raising a small predictive organism inside an
interactive World. The World publishes sensory streams and consumes motor
streams. On every cognitive tick the runtime collects new events, stores them in
working memory, allocates attention, fuses the selected senses, predicts what
happens next, chooses or overrides an action, derives internal signals such as
surprise and threat, stores a sparse episodic seed, and records the entire tick.
Offline or during a sleep phase, a learned Predictive Cortex can be trained from
those records and can roll stored seeds forward as dreams. Development is staged:
the same named organism advances from passive observation to guided movement and
eventually voluntary control only when explicit milestone gates pass.

```text
World + continuous stream loop + predictive learning + memory + motor control
    = an organism with an inspectable life history
```

## The research claim, not just the product shape

The system contains many established ideas: predictive processing, recurrent
world models, reward prediction error, replay, model-predictive control, and
developmental curricula. The V2 docs make one sharp claim that can fail:

> Developmental staging plus generative replay should cause less catastrophic
> forgetting than flat training on the same experience.

That is why the architecture has three memory timescales and why Milestone 5 is
more important than merely showing that a dream can be rendered.

| Timescale | V2 name | Current implementation | Purpose |
|---|---|---|---|
| Seconds | Working memory | `Memory` + `TemporalBuffer` | Recent stream history and the current fused state |
| Session/day | Hippocampus | `brain/hippocampus.py` | High-priority latent/action seeds for recall and dreams |
| Lifetime | Predictive Cortex | `brain/cortex/` | Slow generative knowledge stored in model weights |

## The most important distinction: substrate, organs, and assembled organism

There are three layers of truth in this repository:

1. **The runtime substrate is live and integrated.** The stream buses, tick loop,
   memory, fusion, policies, recording, replay, attention, neuromodulator
   publication, arbiter, and hippocampal seed writes all run in
   `CognitiveRuntime`.
2. **The V2 organs exist.** The recurrent Predictive Cortex, dreams, generative
   replay, MPC controller, full reflex stack, developmental ladder, and read-only
   Clinic each have concrete modules and tests.
3. **Not every organ is yet the default spine of one live process.** The live
   `WorldModel` seam defaults to `TrendWorldModel` or can use the older neural
   world-model adapter. The recurrent `PredictiveCortex` is primarily trained by
   the nursery pipeline. The live async trainer still targets the actor/critic
   stack, and the full MPC/reflex motor stack is driven by development-specific
   policy seams rather than being the default `run` policy.

The July 2026 revision audit summarizes this as **“organs: yes; organism: not
yet.”** That is an assembly boundary, not a reason to ignore the implemented
parts. When reading the code, always ask whether a module is:

- in the default waking loop;
- used by a specialized training/development workflow; or
- a target design/deferred capability.

## The architectural vocabulary

V2 uses biological names as ergonomic labels. A name does not imply a behavior
that is absent from code or metrics.

| Concept | Meaning | Main code |
|---|---|---|
| Organism | One named agent and its life history | `RuntimeConfig.name`, checkpoints, sessions |
| World | Environment adapter; still named `Program` in the core interface | `cognitive_runtime/core/program.py` |
| Afferents | Sensory, body, reward, event, and internal streams | `cognitive_runtime/core/streams/` |
| Thalamus | Budgeted sensory gate | `cognitive_runtime/core/attention.py` |
| Workspace | One fused momentary percept | `TemporalFusion` / `LatentState` |
| Predictive Cortex | Action-conditioned recurrent decoded world model | `brain/cortex/` |
| Working memory | Recent per-stream windows | `Memory` / `TemporalBuffer` |
| Hippocampus | Capacity-bounded episodic seed store | `brain/hippocampus.py` |
| Neuromodulators | Dopamine, acetylcholine, adrenaline signals | `brain/neuromod/`, `brain/amygdala.py` |
| Arbiter | Authored three-mode switch | `brain/arbiter.py` |
| Voluntary motor | MPC by default; alternatives behind one seam | `motor/voluntary.py`, `motor/policy.py` |
| Reflexes | Configured stimulus-to-action overrides | `motor/reflexes.py` |
| Sleep and dreams | Consolidation and generative replay | `sleep/` |
| Development | Milestone-gated ontogeny | `development/` |
| Record | Immutable evidence of what happened | recorder, frame store, session files |
| Clinic | Read-only inspection UI and HTTP API today | `viewer/` |

## The system at a glance

```text
                           WAKING TICK

  World.step()
      │ publishes post-step senses/reward/events
      ▼
  SensoryStreamBus ──► TickSynchronizer ──► TickWindow
                                                │
                                                ▼
                                      Memory / TemporalBuffer
                                                │
                           ┌────────────────────┴───────────────────┐
                           ▼                                        ▼
                    Attention gate                         Latest-value view
                           │                                        │
                           ▼                                        ▼
                    Temporal fusion                          policy State
                           │                                        │
                           └────────────► fused latent ◄─────────────┘
                                                │
                         ┌──────────────────────┼─────────────────────┐
                         ▼                      ▼                     ▼
                   World model             Policy/motor       Entity persistence
                         │                      │                     │
                         └──────── surprise/risk/reward ──────────────┘
                                                │
                           Neuromodulators → Amygdala → Arbiter
                                                │
                           attention/motor mode for later decisions
                                                │
                         ┌──────────────────────┴─────────────────────┐
                         ▼                                            ▼
                 Hippocampal seed                              Recorder
                                                                      │
  action emitted to MotorStreamBus ── applied by next World.step()    ▼
                                                                  the Record

                           SLEEP / OFFLINE

  Record + real-transition reservoir + hippocampal seeds
                         │
                         ▼
             Predictive Cortex training / dream rollout
                         │
                         ▼
            versioned checkpoint / published weights / Clinic exports
```

The action arrow loops back with **one-tick actuation latency**. An action
chosen on cognitive tick `t` sits on `MotorStreamBus`; the World drains and
applies it at the beginning of tick `t+1`. Reward and internal signals follow
the same causal ordering. Do not mentally collapse this into a synchronous
`observe → act → immediate result` call.

## One tick, slowly

Suppose a Crafter stage is guiding the organism to move up.

1. `CrafterWorld.step()` drains the motor bus. On the first tick it finds no
   command, so it applies Crafter's `NULL`/noop action. The world still advances.
2. Crafter publishes the new pixel frame, semantic grid, body state, reward, and
   any achievement/death events with simulated timestamps.
3. `TickSynchronizer.collect()` drains all events since the last cognitive
   boundary into a `TickWindow` grouped by stream id.
4. `Memory.update()` appends those events to bounded per-stream histories.
5. `AttentionController.compute()` assigns weights. In `off` mode all eligible
   inputs receive `1.0`; in `budgeted` mode the controller spends a fixed budget
   using novelty, trend/error, reward relevance, risk, recency, boredom, and
   compute cost.
6. `TemporalFusion.fuse()` encodes the latest stream histories into stable,
   named slices of one `LatentState`. Its `layout_hash` protects checkpoint
   compatibility.
7. The core `WorldModel.predict()` returns a `Prediction`; a policy consumes the
   latest-value State, Memory, and Prediction. In the target architecture this
   prediction comes from a one-step Predictive Cortex rollout.
8. The selected action may pass through a reflex or caregiver precedence stack.
   The default live loop currently has the older orienting-reflex integration;
   the complete `caregiver > reflex > voluntary` record is implemented in
   `motor/reflexes.py` and driven by `MotorFreedomPolicy` in developmental code.
9. The final emissions are published as `motor.command` events. If the policy
   emits `[]`, that is an explicit NULL decision; no motor event is published,
   but the decision is still recorded.
10. Prediction error, reward prediction error, learning progress, risk, safe
    novelty, dopamine, acetylcholine, and adrenaline are computed and published
    as internal streams. The arbiter chooses reward-seeking, information-
    gathering, or fight-or-flight with hysteresis.
11. The current fused latent, emitted action keys, and salient tags are offered
    to the Hippocampus. When full, it keeps the highest-priority seeds rather
    than the newest seeds.
12. The Recorder writes the sensory events, motor events, and one decision
    record. On the next tick, the World consumes the action from step 9.

This ordering explains two common surprises:

- an action and its consequence live on adjacent ticks, not the same tick;
- runtime-produced `internal.*` streams are visible to attention/fusion on a
  later window, because they are derived after the current window was collected.

## Repository tour

### `cognitive_runtime/core/` — stable environment-agnostic contracts

Start here when you want to understand what any World, policy, learner, or
stream component must implement.

- `program.py`: the World seam, still named `Program` for compatibility.
- `action.py`, `policy.py`, `learner.py`, `world_model.py`: small contracts used
  by the live loop.
- `memory.py`: recent stream history, fused latent, attention state, and recent
  actions.
- `attention.py`: the deterministic Thalamus-like budget allocator.
- `streams/`: events, schemas, buses, synchronization, encoders, fusion,
  buffering, and motor command encoding.

The core rule is strict: this package must not import Minecraft or Crafter.

### `cognitive_runtime/runtime/` — the continuous machine

- `loop.py`: assembles the live tick.
- `config.py`: run identity, rates, recording, attention/reflex settings, and
  stage sense masks.
- `scheduler.py`: realtime pacing and missed-tick accounting.
- `recorder.py`: the `streams-v2` disk contract.
- `frame_store.py`: bounded binary storage for large ndarray frames.
- `replay.py`: re-simulation and stream-hash verification for deterministic
  Worlds.

When debugging “what actually happens in a run,” read `loop.py` before any
neural module.

### `cognitive_runtime/programs/` and `bridge/` — Worlds

`programs/crafter/` is the deterministic pixel-native nursery. It translates
Crafter observations and actions into the generic stream and action contracts.

`programs/minecraft/` is the graduation World. Its adapter can use an in-process
deterministic simulated backend or `RemoteMinecraftBackend`, which speaks
line-delimited JSON to the Node mineflayer bridge.

`bridge/fake/` implements the same bridge protocol over the simulated world for
tests. Protocol and process handling can therefore be tested without a
Minecraft server.

### `brain/` — predictive and regulatory organs

- `cortex/predictive.py`: `PredictiveCortex`, its outputs, and checkpoint
  metadata.
- `cortex/backbones.py`: GRU, dilated causal convolution, and transformer
  implementations behind a common temporal-state contract.
- `neuromod/`: internal modulation math plus human-named stream aliases.
- `amygdala.py`: smoothed risk-to-adrenaline appraisal.
- `arbiter.py`: calibrated, hysteretic three-mode lookup.
- `hippocampus.py`: sparse priority store of dream seeds.

### `motor/` — intention and override

- `voluntary.py`: one-step MPC and the shared voluntary-controller protocol.
- `policy.py`: active-inference, imagination-actor, and policy-controller A/B
  adapters.
- `reflexes.py`: World-advertised stimuli, organism-owned reflex configuration,
  caregiver override, and the complete efference record.
- `organism_policy.py`: adapts developmental motor freedom to the live `Policy`
  contract.

### `sleep/` — consolidation machinery

- `dream.py`: sensory-free generative rollout from a hippocampal seed.
- `replay_mix.py`: real/dream batch mixing with bootstrap guardrails.
- `schedule.py`: staleness-free phasic wake/sleep coordination.
- `async_trainer.py`, `weight_publisher.py`: process-level training and atomic,
  versioned weight handoff; concurrent publication uses EMA weights.
- `forgetting.py`: measured retention/forgetting comparison.

### `development/` — raising one organism

- `definitions.py`: stage schema, promotion gates, sense masks, serialization.
- `runner.py`: train/evaluate/promote/hold orchestration and checkpoint resume.
- `ladder.py`: concrete Gestation → Babbling → Crawling → Objects → Foraging
  definition and milestone computations.

Speaking, cross-world weight transfer, and hippocampal retrieval are not part of
the current ladder.

### `cognitive_runtime/training/` and `cognitive_runtime/neural/`

These directories contain both the earlier neural-stream-agent substrate and
the V2 cortex training path. The most important V2-oriented file is
`training/action_world_model.py`: it builds aligned frame/action datasets,
trains the Predictive Cortex with short closed-loop rollouts, evaluates
copy-last/oracle baselines, detects frozen rollouts, runs action ablations, and
exports checkpoints.

Do not assume every class under `neural/` is the Predictive Cortex. In
particular, `neural/world_model.py` is the earlier fused-latent MLP world model;
`brain/cortex/predictive.py` is the V2 recurrent decoded cortex.

### `viewer/` — the Clinic

The current Clinic is a zero-dependency Node server plus browser-native custom
elements and DOM modules. It is not presently a bundled React application,
despite the target architecture using “Node/React” language. It provides a
read-only session browser, prediction/dream strips, EEG-style internal-signal
plots, attention reasons, developmental status, and data-quality verdicts.

The revision audit adds an important presentation requirement: every horizon
strip should communicate **seen at t → predicted at t+h → actual at t+h →
absolute error**, not only predicted versus actual.

## Set up the repository from scratch

Requirements:

- Python 3.10 or newer;
- Node 18 or newer for the Clinic and mineflayer bridge;
- PyTorch only for neural/cortex workflows;
- Crafter only for the nursery World.

PowerShell setup:

```powershell
git clone <repository-url> CCR
Set-Location CCR
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,neural,crafter]"
```

Minimal core-only setup:

```powershell
python -m pip install -e ".[dev]"
```

Useful verification:

```powershell
python -m cognitive_runtime --help
python -m cognitive_runtime nursery list
python -m pytest -q --basetemp=.pytest-tmp
$env:PYTHON = (Resolve-Path .venv\Scripts\python.exe)
Set-Location viewer
npm.cmd test
```

The full Python suite contains neural/training tests and can take several
minutes. If the operating system's default temp directory is not writable, use
`--basetemp` inside the checkout as shown.

Current-checkout caveat: `tests/test_dream.py::test_dream_never_reads_live_senses`
still monkeypatches a removed `StreamBus.read_since` method, so that test fails
at setup rather than exercising dream isolation. Treat it as stale test debt,
not a passing dream-safety proof; the production dream path should still be
reviewed for the no-sensory-read invariant.

## First hands-on path

### 1. Run the deterministic substrate

```powershell
python -m cognitive_runtime run `
  --world minecraft --backend simulated `
  --policy null --episodes 1 --episode-ticks 40 `
  --name Pixel --record-dir sessions
```

Inspect the new `sessions/Pixel-.../` directory. A NULL policy is useful here:
it proves the World advances and records even without a motor event.

### 2. Run a behavioral baseline with frames

```powershell
python -m cognitive_runtime run `
  --world minecraft --backend simulated `
  --policy scripted --episodes 1 --episode-ticks 100 `
  --name Pixel --record-frames --record-dir sessions
```

Then inspect and replay it:

```powershell
python -m cognitive_runtime view --session sessions\<session-id> --episode episode_00000
python -m cognitive_runtime replay --session sessions\<session-id>
```

### 3. Record and evaluate nursery prediction

```powershell
python -m cognitive_runtime nursery run walk_forward `
  --world crafter --record-dir sessions --out-dir models\nursery

python -m cognitive_runtime nursery joint `
  --record-dir sessions --horizons 1 4 8 `
  --backbone gru --out-dir models\joint `
  --report models\joint\report.json
```

The first command trains/evaluates one scenario. `joint` trains one
action-conditioned cortex across scenarios and reports held-out behavior,
copy-last/oracle ratios, frozen-rollout diagnostics, and representation probes.

### 4. Open the Clinic

From the repository root on Windows, point the Node service at the venv Python
so its quality-check subprocess resolves correctly:

```powershell
$env:PYTHON = (Resolve-Path .venv\Scripts\python.exe)
node viewer\server.js --data-dir sessions --port 8787
```

Open `http://localhost:8787` and follow one session from quality verdict to
episode, prediction/dream strip, internal signals, and attention decisions.

## How the model learns from the Record

The core self-supervised signal is alignment, not human labels. A prediction
made from time `t` at horizon `h` is trained against what the Record actually
contains at `t+h`.

```text
recorded frames:    x0  x1  x2  x3  x4  ...
recorded actions:   a0  a1  a2  a3  ...
encoder:            x0 → z0
cortex:             f(z0, a0) → predicted z1
target:                          encoded actual x1
decoder:                         predicted z1 → viewable predicted frame
```

The current cortex training loop uses short rollout windows and scheduled
sampling to avoid learning the identity function over long compositions. It
scores the model against copy-last and periodic-oracle baselines; raw MSE alone
is not considered evidence that dynamics were learned.

The autoregressive-latent revision proposal goes further: use the full fused
workspace latent as the token and train every causal prefix and every prediction
horizon in parallel, LLM-style. That proposal is not the current training
implementation. It also proposes online hippocampal retrieval into the rolling
context window; retrieval remains deferred.

## The three behavioral modes

The Arbiter is deliberately authored, not emergent.

| Surprise | Predicted pain | Mode | Intended effect |
|---|---|---|---|
| Low | Low | Reward-seeking | Exploit known reward opportunities |
| High | Low | Information-gathering | Orient and sample to reduce safe surprise |
| High | High | Fight-or-flight | Let threat reflexes pre-empt deliberation |
| Low | High | Cautious reward-seeking region | Remain wary without treating every tick as a new surprise |

The input is calibrated and the switch has `k`-tick hysteresis. The live loop
currently uses prediction error as a stand-in where a dedicated live cortex
uncertainty head is not connected.

## Sleep, dreams, and the bootstrap guardrail

A dream is a cortex rollout with senses off:

```text
Seed(z_t, actions, tags)
    → cortex.rollout(z_t, recorded actions)
    → predicted future latents
    → decoder
    → viewable dreamed frames
```

Dreaming is not automatically beneficial. A weak generator rehearses its own
mistakes. `GenerativeReplayMixer` therefore enforces two rules:

1. every batch contains real experience; dreams can never be the whole batch;
2. dream fraction is zero until held-out model quality beats copy-last, then
   ramps to a configured cap.

Phasic sleep is the simplest safe schedule: stop acting, consolidate, publish a
completed version, reload it, resume acting. Concurrent sleep exists for the
legacy async stack and publishes a slower EMA snapshot with a monotonic version
so actor staleness can be measured.

## Development as orchestration

The developmental ladder changes the organism's freedoms, not its identity.

| Stage | Main experience | Motor freedom |
|---|---|---|
| Gestation | Passive sensory regularities | Frozen |
| Babbling | Action-to-sensory consequence | Scripted/caregiver overridden |
| Crawling | Predictable locomotion | Scripted/caregiver overridden |
| Objects | Permanence and approach | Voluntary learned path |
| Foraging | Goal-directed behavior | Voluntary learned path |

Each `CurriculumStageSpec` declares its World, scenario, active senses, motor
freedom, active losses, and one or more promotion gates. `run_curriculum()`
persists stage index, attempts, and history in checkpoint metadata so a run can
resume. A failed gate holds the organism rather than silently advancing.

## What is explicitly not included right now

These are design boundaries, not accidental omissions:

- language/Speaking;
- cross-world weight transfer from top-down Crafter pixels to first-person
  Minecraft pixels;
- context-cued hippocampal retrieval during wake;
- learned attention and learned neuromodulator scoring as the default;
- extra neurochemistry without a behavior that needs it;
- byte-exact determinism as a learning-quality gate;
- a control-plane Clinic for starting/stopping runs, injecting caregiver motor,
  triggering sleep, or promoting stages;
- a fully unified live path in which the recurrent cortex is simultaneously the
  predictor, MPC planner, and wake/sleep training target.

## Known documentation and implementation tensions

- “World” is the V2 concept; the foundational Python class is still `Program`.
- The target Clinic says Node/React; the current UI is Node plus browser-native
  JavaScript/custom elements.
- The V2 default motor is MPC; the general `run` command still exposes legacy
  policies and actor/critic paths.
- The current cortex is pixel/action based; the revision proposal's token is a
  multimodal fused workspace latent.
- Cortex reward/terminal/risk/uncertainty outputs exist, but the revision audit
  found that the primary cortex training loop does not yet train all those heads.
- One dream-isolation test targets the removed `StreamBus.read_since` API and
  currently fails before reaching the behavior it intends to assert.
- `README.md` is still primarily the pre-V2 runtime/Minecraft entry point. Use
  this guide and the V2 docs for the organism mental model.

## Recommended source-reading order

Follow one datum before reading every class:

1. `cognitive_runtime/core/program.py`
2. `cognitive_runtime/core/streams/events.py`
3. `cognitive_runtime/programs/crafter/adapter.py`
4. `cognitive_runtime/runtime/loop.py`
5. `cognitive_runtime/core/memory.py`
6. `cognitive_runtime/core/attention.py`
7. `cognitive_runtime/core/streams/fusion.py`
8. `cognitive_runtime/core/world_model.py`
9. `brain/cortex/predictive.py`
10. `brain/neuromod/`, `brain/amygdala.py`, `brain/arbiter.py`
11. `brain/hippocampus.py`, `sleep/dream.py`, `sleep/replay_mix.py`
12. `motor/voluntary.py`, `motor/reflexes.py`, `motor/organism_policy.py`
13. `development/runner.py`, `development/ladder.py`
14. `cognitive_runtime/runtime/recorder.py`, then one real session directory
15. `viewer/server.js` and `viewer/public/`

At each step, ask: what comes in, what goes out, who owns the vocabulary, what
is persisted, and whether the code runs in wake, sleep/offline, or development.
