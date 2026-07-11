# Online Learning And Modular Streams

CCR's first online learner is intentionally small: a dependency-free linear
Q model over the existing fused latent vector.  The neural path now also has
a first behavioral cut — an MLP actor/critic behind `--policy actor-critic`
(issue #29) — which consumes the fixed fused latent by default, or the
learned-fusion path (`--fusion learned`, issue #57) instead.

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
   checks for saved models (module landed, #25; live wiring behind
   `--fusion learned` landed, #57).
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

## Evaluation Gates

Before the actor/critic can replace the linear-Q baseline (or go live), it has
to clear the deprecation gates from
[`neural-stream-agent.md`](neural-stream-agent.md) Phase E. The
`evaluation-gates` subcommand (issue #31) is the one-liner: it trains both the
actor/critic and the linear online-Q in simulation, evaluates both plus
`scripted` and `random` with no mutation on identical seeds, and reports:

1. actor/critic > random — hard requirement.
2. actor/critic > linear Q — unlocks deprecating `OnlineQ*` as primary.
3. reproducible improvement — the same seeds reproduce gate 1 across reruns.

A policy "beats" another when it earns more total reward *or* survives more
total ticks on the shared seeds (the acceptance convention): random wanders
into more incidental novelty reward than the trained learners do, so the
meaningful signal against it is survival.

Small deterministic run (the automated `tests/test_evaluation_gates.py` budget),
recording eval sessions and writing the gate report into the checkpoint:

```bash
python -m cognitive_runtime evaluation-gates \
  --checkpoint models/actor-critic.pt --record-dir sessions
```

Reproduce gates 2–3 on a larger curriculum preset, then inspect the recorded
sessions in the dashboard:

```bash
python -m cognitive_runtime evaluation-gates --curriculum night-survival \
  --train-episodes 40 --reproducible \
  --checkpoint models/actor-critic-night.pt --record-dir sessions
python -m cognitive_runtime dashboard --record-dir sessions
```

The gate report lands in the checkpoint bundle's training stats under
`evaluation_gates` (issue #20), so `read_checkpoint_metadata(path)` recovers the
gate booleans, per-policy eval reward/ticks, and the seeds used.

## Statistical Evaluation Harness

Deterministic replay (`docs/streams.md`'s determinism contract) proves
plumbing correctness by re-simulating one episode bit-for-bit; it cannot be
the regression story once weights mutate mid-episode (neural online
training) or the backend is non-deterministic (`--backend remote`). Issue
#44 re-scopes replay + the simulated backend down to a fast CI smoke test of
loop/stream/recorder plumbing, and adds
`cognitive_runtime.training.statistical_evaluation`: run N episodes per
policy/checkpoint on matched conditions (same curriculum stage; same seed
set in sim; same server/time budget live) and report **mean +/- confidence
interval** on survival ticks, total reward, reward by tier (issue #41),
exploration coverage, world-model prediction error/novelty (issue #39), and
death causes. A candidate is flagged as a regression on a metric only when
its confidence interval no longer overlaps the baseline's, on the worse
side -- an incidental one-episode dip does not fail a gate, but a
consistently, significantly worse checkpoint does.

`evaluation-gates` (issue #31) reports this alongside gates 1-2's single-run
"beats" ordering (`EvaluationGateResult.statistics`/`gate1_comparisons`/
`gate2_comparisons`); a standalone `statistical-evaluate` subcommand runs it
directly, either fresh in sim or against already-recorded sessions:

```bash
# Fresh sim run, N=20 episodes per policy, flag regressions against random:
python -m cognitive_runtime statistical-evaluate \
  --policies random,scripted,online --episodes 20 --baseline random

# From already-recorded sessions, grouped by (curriculum, policy):
python -m cognitive_runtime statistical-evaluate --from-sessions sessions

# The plain-mean dashboard can append the same report:
python -m cognitive_runtime dashboard --record-dir sessions --statistical
```

What determinism still promises: given the same seed and action sequence,
the *simulated* backend reproduces byte-identical observations/rewards
(`replay`'s smoke test), and a checkpoint's saved RNG state (Python, NumPy,
torch, plus torch's deterministic-algorithm/cuDNN flags) lets a single run be
reconstructed for debugging. What it does not promise: that two training
runs (sim or live) produce the same weights, or that a live run reproduces
at all -- regressions there are a statistical question, answered by this
harness, not a bit-comparison one.

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

## Live Childhood Run Protocol

The rules a live (`--backend remote`) run enforces -- start from a checkpoint
or explicit `--fresh`, always record with frames, checkpoint on every kind of
exit including a crashed bridge connection -- plus the `review` command that
closes the loop after a run, are documented in
[`childhood-runs.md`](childhood-runs.md) (issue #33).

