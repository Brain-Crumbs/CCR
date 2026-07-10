# Online Learning And Modular Streams

CCR's first online learner is still intentionally small: a dependency-free
linear Q model over the existing fused latent vector.  The neural modular path
is represented by module interfaces plus a checkpoint bundle format, not yet
as a behavior change.

The linear Q learner is a **baseline**, not the target.  The end-state — a
neural, stream-native agent with trainable encoders, learned fusion, a world
model, and an actor/critic policy — is specified in
[neural-stream-agent.md](neural-stream-agent.md).

## Trainable Stream Modules

`TrainableStreamModule` extends the existing fixed `StreamEncoder` contract with
future learning hooks:

- `encode(events, spec)` produces the current stream latent token.
- `predict_next(latent_slice)` is reserved for stream-local prediction.
- `update(loss_signal)` is reserved for module-local learning.
- `state_dict()` / `load_state_dict()` and `checkpoint_payload()` define
  checkpoint state for future neural modules.

`FixedStreamModule` wraps today's fixed encoders and returns no trainable state.
That means `TemporalFusion` and the v1 online Q feature layout remain unchanged.

`cognitive_runtime.neural.checkpoint.NeuralAgentCheckpoint` is the Phase A
bundle format for future neural learners: it saves encoders, learned fusion,
world model, actor, critic, optimizer state, training counters, RNG state and
replay-buffer metadata with torch, while writing a JSON sidecar so tooling can
inspect checkpoint provenance without deserializing tensors.

## Future Neural Path

The intended upgrade path is incremental:

1. Keep fixed stream encoders as the regression baseline.
2. Replace selected streams with trainable encoders that expose the same fixed
   slice width and checkpoint hooks.
3. Add learned fusion over per-stream slices while preserving layout/version
   checks for saved models.
4. Add a learned world model that predicts next latent state, expected reward,
   terminal/death probability, risk, and prediction error.
5. Use those predictions as inputs to an actor/critic policy, keeping the
   linear online Q learner as the baseline and smoke-test target.

## Simulated Pretraining

Pretrain in the deterministic simulator first:

```bash
python -m cognitive_runtime run --backend simulated --policy online \
  --episodes 20 --episode-ticks 1200 --world-size 32 \
  --day-length 800 --start-time 300 \
  --online-model models/online-q.json --online-save-every 1000 \
  --epsilon-start 0.8 --epsilon-min 0.05 --epsilon-decay-ticks 20000 \
  --online-lr 0.05 --online-gamma 0.99 \
  --record-dir sessions --session-id online-pretrain
```

Evaluate without further mutation:

```bash
python -m cognitive_runtime run --backend simulated --policy online \
  --no-online-train --episodes 3 --episode-ticks 1200 \
  --online-model models/online-q.json \
  --record-dir sessions --session-id online-eval
```

## Phase E Evaluation Gates

Before the actor/critic can replace the linear-Q baseline (or go live), it has
to clear the deprecation gates from
[`neural-stream-agent.md`](neural-stream-agent.md) Phase E. The
`phase-e-gates` subcommand (issue #31) is the one-liner: it trains both the
actor/critic and the linear online-Q in simulation, evaluates both plus
`scripted` and `random` with no mutation on identical seeds, and reports:

1. actor/critic > random — hard requirement.
2. actor/critic > linear Q — unlocks deprecating `OnlineQ*` as primary.
3. reproducible improvement — the same seeds reproduce gate 1 across reruns.

A policy "beats" another when it earns more total reward *or* survives more
total ticks on the shared seeds (the acceptance convention): random wanders
into more incidental novelty reward than the trained learners do, so the
meaningful signal against it is survival.

Small deterministic run (the automated `tests/test_phase_e_gates.py` budget),
recording eval sessions and writing the gate report into the checkpoint:

```bash
python -m cognitive_runtime phase-e-gates \
  --checkpoint models/actor-critic.pt --record-dir sessions
```

Reproduce gates 2–3 on a larger curriculum preset, then inspect the recorded
sessions in the dashboard:

```bash
python -m cognitive_runtime phase-e-gates --curriculum night-survival \
  --train-episodes 40 --reproducible \
  --checkpoint models/actor-critic-night.pt --record-dir sessions
python -m cognitive_runtime dashboard --record-dir sessions
```

The gate report lands in the checkpoint bundle's training stats under
`phase_e_gates` (issue #20), so `read_checkpoint_metadata(path)` recovers the
gate booleans, per-policy eval reward/ticks, and the seeds used.

## Live Mineflayer Rollout

Install and configure the bridge as described in
[`bridge/mineflayer/README.md`](../bridge/mineflayer/README.md), then start with
eval-only live smoke:

```bash
set CCR_MINECRAFT_HOST=localhost
set CCR_MINECRAFT_PORT=25565
python -m cognitive_runtime run --backend remote --realtime \
  --policy online --no-online-train --episodes 1 --episode-ticks 400 \
  --online-model models/online-q.json \
  --record-dir sessions --session-id live-online-eval
```

Fine-tune live with frequent checkpointing:

```bash
python -m cognitive_runtime run --backend remote --realtime \
  --policy online --episodes 1 --episode-ticks 1200 \
  --online-model models/online-q.json --online-save-every 100 \
  --epsilon-start 0.1 --epsilon-min 0.02 --epsilon-decay-ticks 10000 \
  --record-dir sessions --session-id live-online-train
```

Compare against baselines by recording comparable live sessions:

```bash
python -m cognitive_runtime run --backend remote --realtime \
  --policy random --episodes 1 --episode-ticks 1200 \
  --record-dir sessions --session-id live-random

python -m cognitive_runtime run --backend remote --realtime \
  --policy scripted --episodes 1 --episode-ticks 1200 \
  --record-dir sessions --session-id live-scripted

python -m cognitive_runtime dashboard --record-dir sessions
```

Remote Minecraft sessions remain non-deterministic and snapshot-less. Replay
verification intentionally skips them with a clear message, but the recordings
remain usable for `view`, `dashboard`, and offline training datasets.

