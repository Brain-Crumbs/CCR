# CCR — Train a CNN World Model That Builds Like an LLM

A research project for growing a **Predictive Cortex**: a recurrent,
action-conditioned CNN world model that learns to predict its own future
senses, trained incrementally from recorded experience — record scenarios,
train, measure, repeat.

The measurable claim: **the cortex beats a copy-last-frame baseline on
held-out seeds at every prediction horizon, and withholding action
information measurably hurts its predictions** (Milestone 2).

```
Record scenarios  →  Quality-gate the data  →  Train the Predictive Cortex
     →  Evaluate on held-out seeds  →  Analyze results
```

## Getting started

### Notebook (recommended)

The primary entrypoint is the Jupyter notebook — it runs the full pipeline
end-to-end in one place: record, gate, train, evaluate, diagnose.

```bash
pip install -e ".[neural]"      # installs PyTorch + Crafter + all deps
jupyter notebook notebooks/build_and_diagnose_organism.ipynb
```

The notebook walks through every step with inline results and plots. Start
here if you want to understand the project or reproduce the results.

### CLI

Every notebook step has a CLI equivalent for scripting, CI, or headless runs.

```bash
pip install -e ".[neural]"
```

#### 1. Record nursery scenarios (train + holdout data)

The nursery records scripted micro-scenarios in Crafter (a deterministic,
pip-installable 2-D world) with pixel frames at multiple seeds, splitting
into train and holdout sets.

```bash
# List available scenarios
ccr nursery list

# Record one scenario (train seeds 0-5, holdout seeds 6-7, 400 ticks each)
ccr nursery run walk_forward --record-dir runs/Pixel --name Pixel

# Record all scenarios at once
ccr nursery run all --record-dir runs/Pixel --name Pixel
```

#### 2. Train the Predictive Cortex (joint world model)

Train one recurrent, action-conditioned world model across all recorded
scenarios. The cortex predicts future pixel frames and latent states at
configurable horizons (default t+1, t+10, t+100).

```bash
ccr nursery joint \
    --record-dir runs/Pixel \
    --epochs 30 \
    --backbone gru \
    --training-objective autoregressive \
    --out-dir models/Pixel \
    --report results/joint-report.json
```

#### 3. Evaluate: does the cortex beat baselines?

The nursery evaluation compares the trained model against copy-last-frame
and mean-frame baselines on held-out seeds, reports per-horizon metrics
(MSE, PSNR, SSIM), checks for frozen rollouts (a collapsed model), and
runs a yaw linear probe on the hidden state.

```bash
# Per-scenario evaluation (run during nursery run)
ccr nursery run walk_forward --record-dir runs/Pixel --out-dir models/Pixel --name Pixel

# Joint evaluation with zero-shot held-out scenarios
ccr nursery joint --record-dir runs/Pixel --out-dir models/Pixel --report results/report.json
```

#### 4. Inspect results in the clinic

The clinic is a browser-based viewer that shows dream strips (predicted vs
actual frames at each horizon), neuromodulator timelines, and data-quality
diagnostics.

```bash
node viewer/server.js --data-dir runs/Pixel
# open http://localhost:8787
```

### What each command does

| Command | Purpose |
|---|---|
| `ccr nursery list` | List available nursery scenarios |
| `ccr nursery run <scenario\|all>` | Record train/holdout episodes, train a pixel encoder, evaluate per-scenario |
| `ccr nursery joint` | Train ONE joint world model across all scenarios, evaluate in-distribution + zero-shot |
| `ccr nursery backbone-benchmark` | Compare GRU vs dilated-conv vs transformer backbones on identical data |
| `ccr run` | Run the live cognitive loop (Crafter by default) |
| `ccr train --model-type ...` | Offline training for specific model types (neural, world-model, etc.) |
| `ccr evaluate` | Compare baseline policies on identical episodes |
| `ccr statistical-evaluate` | Mean +/- CI evaluation over N episodes with regression flagging |
| `ccr dashboard` | Aggregate metrics across recorded sessions |
| `ccr replay --session <path>` | Replay a recorded session and verify determinism |
| `ccr view --session <path> --episode <id>` | Inspect a single recorded episode |
| `ccr review --session <path>` | Post-run session review with baseline comparison |

Run `ccr <command> --help` for full option details.

## What gets measured

The project has explicit milestones with falsifiable acceptance criteria:

**Milestone 2 — Prediction quality (the core result):**
- (a) The cortex beats copy-last-frame at every horizon on held-out seeds
- (b) Withholding actions from the model measurably hurts predictions
  (action-ablation test)
- No frozen rollouts (the model hasn't collapsed to a fixed point)

**Milestone 4 — Dreams:**
- Hippocampal seeds replay generatively through the cortex
- Dream strips (predicted vs actual) are viewable and exported

**Milestone 5 — Forgetting (the sharp research bet):**
- Developmental staging + generative replay produces measurably less
  catastrophic forgetting than flat training on the same data

**Representation quality:**
- The hidden state linearly decodes heading (yaw probe)
- No latent collapse (effective rank, variance checks)
- Reward/terminal/risk/uncertainty heads beat constant-predictor baselines

## Temporal backbones

The Predictive Cortex's recurrent core is a swappable **temporal backbone**
(`brain/cortex/backbones.py`) — the module that advances the world state
from one `(latent, action)` pair to the next. All three share the same
contract (`initial_state`, `step`, `readout`) and the same cortex
prediction/scoring heads, so swapping the backbone is an A/B test, not a
fork.

| Backbone | `--backbone` | How it works | Context | Tradeoff |
|---|---|---|---|---|
| **GRU** | `gru` | A single `GRUCell` processes one input per step; the recurrent hidden state accumulates the full history implicitly. | Unbounded (recurrent) | Simple, fast per-step, proven on sequence prediction. Can't attend to distant tokens in parallel — long-range dependencies decay through the hidden state. |
| **Dilated Causal Conv** | `dilated_conv` | WaveNet-style 1-D convolution stack with exponentially growing dilation (2, 4, 8...). Processes the last `--context-length` inputs in one parallel pass each step. | Fixed window (`--context-length`, default 8) | Reads multiple timescales simultaneously; efficient parallel training. Receptive field is bounded by depth and dilation — can't see beyond the window. |
| **Causal Transformer** | `transformer` | A small `TransformerEncoder` with ALiBi (Attention with Linear Biases) positional encoding over the last `--context-length` inputs. Full pairwise attention within the window. | Fixed window (`--context-length`, default 8) | Attends to every position in the window equally (no decay); ALiBi means positions learned at short curriculum widths generalize to longer windows at inference. Quadratic cost in window length. |

**Context-length curriculum:** The windowed backbones (dilated-conv,
transformer) start training with a window of 1 frame and ramp up to
`--context-length` over the course of training. This prevents the model
from being asked to exploit a long context window before it has learned
what one step of dynamics looks like.

**Benchmarking backbones:**
```bash
ccr nursery backbone-benchmark \
    --backbones gru dilated_conv transformer \
    --baseline-backbone gru \
    --train-scenarios walk_forward turn_in_place \
    --eval-scenario turn_in_place \
    --report results/backbone-comparison.json
```

## Project structure

```
notebooks/
  build_and_diagnose_organism.ipynb    ** START HERE ** — full pipeline

cognitive_runtime/
  cli.py               CLI entrypoint (ccr command)
  core/                Program interface, streams, memory, fusion, policy
  runtime/             Tick loop, scheduler, recorder, replay
  programs/
    crafter/           Crafter nursery world (default)
    minecraft/         Legacy survival-economy world (opt-in via --world minecraft)
  training/            Dataset builders, nursery runner, evaluation harnesses
  neural/              Trainable stream encoders, pixel encoder, world model
  policies/            null, random, scripted, learned, neural, cortex world model
  tools/               Episode viewer, dashboard, replay runner, review

brain/
  cortex/              PredictiveCortex — the recurrent action-conditioned world model
  hippocampus.py       Episodic seed store for dreaming and recall
  amygdala.py          Threat estimation from prediction error
  arbiter.py           Mode selection (reward-seeking / info-gathering / fight-or-flight)
  neuromod/            Neuromodulator signals (dopamine, acetylcholine, adrenaline)
  calibration.py       Rolling-holdout surprise calibrator

sleep/
  dream.py             Generative world-model rollouts from hippocampal seeds
  cortex_consolidation.py  Online cortex learning from replay
  replay_mix.py        Generative replay mixer (real + dream transitions)
  forgetting.py        CI-refereed forgetting metric

motor/
  voluntary.py         MPC over the world model
  reflexes.py          Hardcoded reflex stack (orienting, threat withdrawal)
  policy.py            Reflex-override precedence

development/           Developmental ladder with gated stage promotion
viewer/                Browser-based clinic (Node/JS, read-only diagnostics)
docs/v2/               V2 design docs (architecture, phases, onboarding)
tests/                 ~190 tests
```

## Installation

Requires Python >= 3.10.

```bash
# Core only (runtime loop, recording, Crafter world — no neural training)
pip install -e .

# With neural training (PyTorch + everything needed for the notebook)
pip install -e ".[neural]"

# Development (adds pytest)
pip install -e ".[dev,neural]"
```

Run the tests:

```bash
pytest
```

## The architecture in brief

The system is a **continuously-running cognitive loop** that observes an
interactive world, predicts what happens next, and records everything:

```python
while running:
    streams  = world.step()                    # sensory input
    memory.update(streams)                     # temporal buffer
    latent   = fusion.fuse(streams, memory)    # fixed-width state
    forecast = cortex.predict(latent, action)  # multi-horizon prediction
    action   = motor.select(forecast)          # act by planning over the model
    recorder.write(streams, action, forecast)  # full tick record
```

The **Predictive Cortex** (`brain/cortex/predictive.py`) is the core model:
- Recurrent (GRU backbone by default; dilated-conv and transformer alternatives)
- Action-conditioned (actions are part of the input, not just the output)
- Multi-horizon (predicts at t+1, t+4, t+8 — configurable)
- Decoded (every prediction has a pixel decoder so you can see what it imagined)
- Trains from recorded experience with an autoregressive objective

The world is **Crafter** by default — a deterministic, 2-D, pip-installable
environment that provides pixels, actions, rewards, and achievements without
any server or GPU rendering dependencies.

## Documentation

- [docs/v2/00-overview.md](docs/v2/00-overview.md) — the research vision and
  the predict-surprise-act loop
- [docs/v2/01-architecture.md](docs/v2/01-architecture.md) — the anatomy of
  the organism, the old-to-new naming map
- [docs/v2/03-onboarding-guide.md](docs/v2/03-onboarding-guide.md) — from-scratch
  mental model of the system
- [docs/v2/04-contracts-and-data-flow.md](docs/v2/04-contracts-and-data-flow.md) —
  exact Python/tensor/stream contracts
- [docs/v2/REVIEW-2026-07-organism-audit.md](docs/v2/REVIEW-2026-07-organism-audit.md) —
  honest assessment of what's assembled vs what's separate pipelines
