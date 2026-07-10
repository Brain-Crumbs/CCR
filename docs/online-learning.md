# Online Learning And Modular Streams

CCR's first online learner is intentionally small: a dependency-free linear
Q model over the existing fused latent vector.  The neural path now also has
a first behavioral cut — an MLP actor/critic behind `--policy actor-critic`
(issue #29) — though it still consumes the fixed fused latent rather than
learned fusion (issue #57).

The linear Q learner is a **baseline**, not the target.  The end-state — a
neural, stream-native agent with trainable encoders, budgeted attention,
learned fusion, a multi-horizon world model, internal modulation streams,
and an actor/critic policy driven by both extrinsic reward and a
"safe surprise" intrinsic drive — is specified in
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
   slice width and checkpoint hooks (landed, #24).
3. Add learned fusion over per-stream slices while preserving layout/version
   checks for saved models (module landed, #25; live wiring is #57).
4. Add a learned world model that predicts next latent state, expected reward,
   terminal/death probability, risk, and prediction error (first cut landed,
   #26; multi-horizon generative version with uncertainty is #39).
5. Use those predictions as inputs to an actor/critic policy, keeping the
   linear online Q learner as the baseline and smoke-test target (first cut
   landed, #29; evaluation gates are #31).
6. Publish internal modulation signals (prediction error, reward prediction
   error, learning progress, novelty, risk) as `internal.*` streams (#58).
7. Add the budgeted attention controller between memory and fusion, with a
   recorded per-tick `AttentionState` (#59) and the orienting reflex (#60).
8. Move gradient steps off the tick thread via the async actor/learner split
   (#37), and add the risk-gated surprise intrinsic drive through the reward
   profile schema (#41, #61).

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

