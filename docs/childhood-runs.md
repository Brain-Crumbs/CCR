# Live Childhood Runs

Phase F of [`neural-stream-agent.md`](neural-stream-agent.md) (issue #33): the
protocol for taking a trained policy off the deterministic simulator and onto
a live Mineflayer connection, without losing training progress or the ability
to explain what the agent did. Live Minecraft runs are non-deterministic and
snapshot-less (`bridge/mineflayer/`, `programs/minecraft/remote.py`) -- the
guarantees below come from checkpointing and recording, not from replay.

The full workflow, in order:

```text
pretrain in sim  ->  eval gates  ->  live fine-tune  ->  review  ->  next curriculum step
```

## 1. Pretrain In Simulation

Train and clear the deprecation gates before ever touching a live server --
see [`online-learning.md`](online-learning.md)'s "Simulated Pretraining" and
"Evaluation Gates" sections. Live ticks are expensive and non-reproducible;
simulation is where the bulk of learning should happen.

## 2. Live Fine-Tune: The Run Protocol

A live run (`--backend remote`) enforces three rules the CLI checks before it
ever spawns the bridge subprocess:

1. **Start from a checkpoint, or say so explicitly.** `--online-model`/
   `--actor-critic-model` must point at an existing checkpoint, or the run
   must pass `--fresh`. This is a hard CLI error, not a silent fallback --
   losing a childhood's worth of weights to a typo'd path is exactly the
   failure this guards against:

   ```bash
   python -m cognitive_runtime run --backend remote --realtime \
     --policy actor-critic --fresh \
     --actor-critic-model models/actor-critic.pt \
     --record-dir sessions --session-id live-childhood-001
   ```

2. **Always record, frames included.** `--no-record` is rejected outright
   for live runs, and frame recording turns on automatically -- no need to
   remember `--record-frames`. A live session that cannot be replayed by
   re-simulation is still fully reviewable through `view`/`dashboard`/
   `review` precisely because every frame and stream event is on disk.

3. **Save often, and on every kind of exit.** `--online-save-every`/
   `--actor-critic-save-every` checkpoint periodically by tick count; the
   runtime's `finally` block (`CognitiveRuntime.run`) also checkpoints on
   clean shutdown, on an uncaught exception, and on `KeyboardInterrupt`
   (`Ctrl-C`/`SIGINT`) -- interrupting a live run leaves a valid checkpoint
   that the next run resumes from, tick counters and weights included:

   ```bash
   python -m cognitive_runtime run --backend remote --realtime \
     --policy actor-critic --actor-critic-model models/actor-critic.pt \
     --actor-critic-save-every 200 \
     --record-dir sessions --session-id live-childhood-002
   ```

### Crash-Resume: A Dying Bridge Doesn't Kill The Run

The Mineflayer bridge is a subprocess talking to a real server over a
socket -- it can crash or the connection can drop. When that happens
mid-episode, `RemoteBridge` raises `BridgeError`
(`cognitive_runtime.programs.minecraft.remote`), a subclass of the generic
`RecoverableEpisodeError` (`cognitive_runtime.core.program`). The runtime
loop catches it, checkpoints the online learner with reason `"bridge_error"`,
and ends *that episode* (`termination_reason: "bridge_error"` in its
summary) instead of crashing the process. The next episode's `reset()`
respawns the bridge subprocess automatically (`RemoteBridge.start()` detects
a dead process and relaunches it), so a multi-episode live run survives a
connection drop and keeps going.

If the bridge command itself is broken (missing binary, bad
`$CCR_MINECRAFT_BRIDGE_CMD`), `reset()` fails the same way on every attempt --
that surfaces as a real crash, checkpointed on the way out, rather than an
infinite respawn loop.

### What Gets Recorded

Every cognitive tick, alongside sensory/motor streams and the reward, the
runtime publishes model-introspection streams so a recording explains not
just what the agent did but what it *expected*:

- `model.novelty` -- world-model prediction error + entity-persistence
  surprise (issues #26/#27).
- `model.value_estimate` -- the critic's value estimate for the current
  fused state (actor/critic policies only).

Both show up in `view`/`review` decision lines (`pred_error=... novelty=...
value_estimate=...`) next to the reward the agent actually received.

## 3. Review

After a run, one command summarizes it, compares it against baseline
sessions recorded on the same curriculum step, and shows recent-episode
detail:

```bash
python -m cognitive_runtime review \
  --session sessions/live-childhood-002 --record-dir sessions
```

This is the loop-closing step: "did this run beat the baselines on this
curriculum step enough to advance?" `review` is `dashboard` (aggregate
comparison) plus `view` (per-episode decisions) in one call, scoped to one
run's baselines rather than every session on disk. Record baselines the same
way you would for the dashboard (`docs/online-learning.md`'s "Live
Mineflayer Rollout" section) -- e.g. a `random`/`scripted` session on the
same `--curriculum`.

## 4. Next Curriculum Step

Once a run reviews well against its baselines, move to the next curriculum
step (`docs/curriculum.md`) and repeat from step 2, still pointing
`--actor-critic-model`/`--online-model` at the same checkpoint -- the
childhood continues from where it left off, not from scratch.
