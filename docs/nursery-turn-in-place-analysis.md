# Analysis: first real `nursery/turn_in_place` runs, and the road to a general world model

Review of `shared/nursery-turn_in_place-train-{0..5}` on the
`first-trained-session-data` branch (recorded 2026-07-12, remote backend).
Companion to `docs/nursery-walk-forward-validation.md`; every number below is
reproducible from the stream logs with the snippets in the appendix.

The two symptoms reported from the viewer:

1. the view is still top-down, not first-person;
2. the multi-horizon predictions (t+1 / t+10 / t+100) are all the same frame,
   instead of picking up the spinning pattern.

Both are real, both are diagnosable from the code + data, and neither is a
"turn up the epochs" problem. Findings first, then the research-project shape
that fixes them for good.

## Symptom 1: the view is top-down

**The first-person render path exists but never ran; the fallback is a
rotating minimap, and nothing records which one you got.**

- The recorded frames are 33×33×3 — the colorized *semantic-grid fallback*
  (`pixels_from_frame`), not the prismarine-viewer first-person capture the
  branch added (`bridge/mineflayer/pixels.js`, which renders at 160×120 and
  resizes to its own `PIXEL_SHAPE`). The viewer path depends on optional
  native deps (`node-canvas-webgl`, headless-GL, `three`); when they are
  missing or fail to init, `PixelViewer.start()` logs to stderr and the
  session silently degrades to the grid. That is what happened on the server:
  `config.pixel_source` defaults to `"viewer"`, but the frames prove the grid
  path produced every recorded pixel.
- The session metadata does not say which pixel source was active.
  `session.json` declares the stream as an RGB camera frame either way, so
  the only way to notice is to look at the pictures — which is how it was
  noticed.
- What *did* change on this branch: the fallback now rotates the top-down
  grid by yaw (`world.py:_orient_frame_grid`, and
  `remote.py` passes `yaw_degrees` into `pixels_from_frame`). That is why the
  run "does look like it's spinning": the frames recur almost exactly every
  12 vision frames (mean |diff| 0.6–1.9/255 at lag 12 vs ~15/255 at lag 1),
  i.e. one full 360° revolution (30°/frame — see the sampling-rate finding
  below). But it is still an orthographic minimap that rotates as a rigid
  image: no horizon, no perspective, no parallax, no depth ordering. The
  regularity available to learn is "image rotates by a constant angle each
  step", not "camera rotation in a 3D scene".
- Latent bug for when the viewer *does* activate: `bridge/mineflayer/pixels.js`
  declares `PIXEL_SHAPE = [128, 128, 3]` ("keep in sync with
  streams.py:PIXEL_SHAPE") while `streams.py` computes 33×33×3 from
  `PIXEL_RADIUS`/`PIXEL_SCALE`. The Python side (`_pixels_array`) accepts any
  H×W×3, so the first successful viewer session will record frames that
  contradict the catalog's declared shape and be incompatible with models
  trained on the 33×33 fallback.

## Symptom 2: all horizons predict the same frame

**The rollout has collapsed to a fixed point, and the current architecture
cannot represent the alternative.**

The signature in the viewer is exact: MSE at t+10 equals MSE at t+100 to
three significant figures (3.88e-3 in the reported run). The rollout reaches
a fixed frame within ~10 steps and stays there; the predicted image is the
same at every horizon.

Why, in order of importance:

1. **The predictor cannot see motion.** `NextLatentPredictor` is a two-layer
   MLP `z_t → z_{t+1}` on a single frame's 32-dim latent — no action input,
   no recurrent state, no frame stacking. A single top-down frame of a
   rotating scene does not determine the next frame: rotation speed and
   direction are simply not in the Markov state. It is learnable *here* only
   because every training episode spins the same direction at the same rate,
   so the model could bake "always rotate 30°" into the weights — but nothing
   pushes it to, and the fixed-point solution is cheaper (see 2 and 3).

2. **The identity is a strong attractor under this loss.**
   `train_horizon_consistency` decodes the h-step *iterated* rollout and
   MSE-matches the true frame at t+h, for h ∈ {1, 4, 8}, backpropagating
   through up to 100 compositions of the same MLP. The lowest-effort way to
   keep a 100-step composition stable is `f ≈ identity`; the decoder then
   pays a constant, modest MSE at every horizon. On mostly-uniform terrain
   (the sim's grass world, or the remote sand/grass patch) a static frame is
   already close to the mean-frame optimum, so the gradient toward actually
   modelling the rotation is tiny compared to the gradient toward not
   exploding.

3. **The only strongly-moving object is unpredictable.** In the sim viewer
   run, the terrain is near-uniform green, so rotation is *invisible* except
   through the mob — and the mob's wandering is stochastic. The MSE-optimal
   deterministic prediction of an unpredictable object is to freeze or blur
   it. `turn_in_place` on uniform terrain therefore contains almost no
   learnable signal at all: the predictable part (rotation) is invisible, and
   the visible part (the mob) is unpredictable.

4. **Aliasing makes the horizons nearly indistinguishable.** Vision was
   recorded at ~10 Hz against 20 Hz ticks (201 pixel events / 400 ticks —
   same as the walk-forward finding), so each recorded frame step is 30° and
   the rotation period is 12 frames. Horizons count *frame steps*
   (`load_episode_pixel_frames`), so t+100 ≡ t+4 (mod 12): the target
   distributions at t+1, t+10, t+100 are nearly identical, which further
   flattens any per-horizon gradient — and makes "predict a plausible static
   frame" score about equally at every horizon.

**How much signal is the model leaving on the table?** Computed on the six
real sessions (MSE on [0,1] pixels, averaged over sessions):

| horizon | copy-last | mean-frame | period-12 oracle |
|---------|-----------|------------|------------------|
| t+1     | 0.0143    | 0.0125     | **0.0023**       |
| t+10    | 0.0211    | 0.0125     | **0.0023**       |
| t+100   | 0.0385    | 0.0147     | **0.0024**       |

The "period-12 oracle" simply reuses the frame from one revolution earlier —
what any model that learned the rotation would approximate. It beats
copy-last by 6–16× and is *flat across horizons*: on this scenario a correct
world model loses nothing with horizon, so the benchmark can cleanly separate
"learned the regularity" from "collapsed to a static frame". The current
model is architecturally unable to reach the oracle row.

**The data to fix this is already recorded.** `input.mouse_look` is logged at
20 Hz in every session (`{d_yaw: -15.0}` on 399/400 ticks) — the
action-conditioning signal exists in every log; the training pipeline just
never reads it.

## Data-quality findings specific to these runs

Extends the walk-forward review; the same remote-backend caveats apply
(seeds do nothing, sessions inherit world state, ~96% missed realtime ticks
at ~17 tps, vision at half the declared rate, and no holdout sessions in
`shared/`, so the benchmark itself is not reproducible from this folder).

New for `turn_in_place`:

1. **The agent did not stay in place.** Position drifts 11.7–24.4 blocks in
   5 of 6 sessions while the policy only issues `LOOK_LEFT`. A live server
   applies knockback, water currents and gravity; a "pure view rotation"
   scenario does not survive contact with survival-mode physics.
2. **train-0 was killed mid-run.** `termination_reason: "death:hit"` at tick
   167/400, `total_reward -12.8`, `event.damage_taken` at 0.36 Hz — hostile
   mobs beat the spinning agent to death. Episode lengths across the six
   sessions are 85/201/164/149/199/166 frames: inconsistent for reasons the
   scenario metadata does not surface.
3. **The data-quality gate waves this through.** `turn_in_place` declares no
   expectations (`min_blocks_per_tick=0.0`, `min_unique_frame_fraction=0.0`)
   with a rationale that is now stale — "the top-down render doesn't rotate
   with yaw" — written before `_orient_frame_grid` landed on this same
   branch. A gate that would catch these sessions needs the *opposite*
   checks: yaw sweep ≥ 360°, displacement ≤ ε (flag drift), episode ran to
   completion (flag death), and ≥ 1/12 unique-frame fraction now that
   rotation produces changing pixels.
4. Housekeeping: `shared/nursery-turn_in_place-train-0 copy/` is a duplicate
   folder checked into the data branch.

## The research project: from per-scenario canaries to one general model

The nursery currently trains **one tiny model per scenario, with the policy
baked into the dynamics**. That is the right shape for a canary and the wrong
shape for a general world model — a predictor that never sees actions cannot
distinguish "I will keep turning left" from "I will stop", so it can only
ever model a single policy's closed-loop dynamics. The project below turns
the same harness into something that trains one model across scenarios.

### Phase 0 — make the percept trustworthy (infrastructure)

- Get the first-person viewer running on the recording host (headless-GL via
  xvfb or a GPU node), or explicitly decide the semantic-grid render *is* the
  percept for this stage and say so in the stream spec.
- Reconcile the pixel-shape contract (33×33 grid vs 128×128 viewer): pick one
  recorded shape (64×64 is a reasonable meet-in-the-middle), enforce it at
  the bridge boundary, and version it in the catalog.
- Record **pixel provenance** (`pixel_source: viewer|grid`) in
  `session.json`, and make the nursery data-quality gate refuse to mix
  sources within one training run. A silent stderr fallback that changes the
  observation distribution is the single most expensive failure mode in this
  dataset.
- Nursery recordings belong on the simulated backend (deterministic, seeds
  vary terrain, no realtime jitter) or on a *protected* remote agent
  (peaceful mode / invulnerable / teleport-reset per session). Remote survival
  mode is an evaluation environment, not a curriculum recorder.

### Phase 1 — make the problem well-posed (model interface)

- **Action-conditioned transition:** `z_{t+1} = f(z_t, a_t)` — embed the
  already-recorded `input.mouse_look` / motor stream and concatenate with the
  latent. This single change is what lets one model serve every scenario, and
  turn_in_place is its perfect unit test: with actions, direction is known;
  without, it is unknowable.
- **State with memory:** a single frame is not a Markov state (finding 1
  above). Either stack k frames at the encoder or make `f` recurrent (GRU
  over latents). Success criterion: the model predicts rotation *rate* from
  observation history alone when the action stream is withheld.
- **Horizon semantics:** count horizons in ticks (or seconds), not recorded
  frame steps, and store the vision rate with the checkpoint so t+100 means
  the same thing on every backend. (The catalog honesty fix on the data
  branch makes this possible; the training code still ignores it.)

### Phase 2 — make the training signal honest (objective)

- Replace 100-step backprop-through-composition with per-horizon heads (the
  codebase already has `MultiHorizonMLPWorldModel` on the fused latent) or
  short-rollout scheduled sampling (roll 5–10 steps, resample start points).
  Long compositions of one MLP under MSE select for the identity — that is
  symptom 2 in one sentence.
- Score every run against the cheap reference predictors, not just raw MSE:
  report `MSE(model)/MSE(copy-last)` per horizon, and for periodic scenarios
  `MSE(model)/MSE(period oracle)`. The oracle table above is the target
  curve; a model between copy-last and oracle is learning, a model at
  copy-last is not.
- **Frozen-rollout detector:** if the variance across horizons of the
  *predicted* frames is ~0 while the actual frames differ, flag the run red
  in the report. This exact check would have auto-diagnosed the screenshot
  that prompted this analysis (identical predictions and identical MSE at
  t+10/t+100).

### Phase 3 — train one model, evaluate generality

- Joint pretraining over all nursery scenarios (shared encoder + shared
  action-conditioned transition), with **held-out scenarios** as the
  generality metric — e.g. train on walk_forward + turn_in_place +
  strafe_and_stop, test zero-shot on approach_entity (composition of
  ego-motion and scale change).
- Scenario-conditional probes: does the latent linearly decode yaw? mob
  bearing? time of day? These are the cheap interpretability checks that say
  *what* the general model actually captured.
- Only then move up the ladder to the survival curriculum (issue #43), with
  the nursery suite kept as a regression battery: every future encoder or
  world-model change must not lose the rotation/ego-motion/permanence
  regularities it once had.

### Immediate, low-cost fixes (independent of the phases)

1. Add `pixel_source` to session metadata; gate on it.
2. Fix `bridge/mineflayer/pixels.js` `PIXEL_SHAPE` to match the catalog.
3. Give `turn_in_place` a real data-quality gate (yaw sweep, max
   displacement, completed episode, unique-frame floor) and delete the stale
   "doesn't rotate with yaw" comment.
4. Record nursery holdout sessions into `shared/` (or wherever run data
   lands) so the benchmark is reproducible from the artifact alone.
5. Remove `nursery-turn_in_place-train-0 copy/`.

## Implementation status

Everything below landed with this document's branch; what remains is
host-side (installing headless-GL/`node-canvas-webgl` on the recording
server so the viewer path activates) and re-recording.

- **Phase 0** — the bridge reports `pixel_source` (`viewer`/`grid`) per
  observation; both backends surface it through `stats()` into
  `summary.program_stats.pixel_sources`; the data-quality gate refuses mixed
  provenance and (via `NurseryConfig.expected_pixel_source`) provenance that
  contradicts the run's intent. `pixels.js` `PIXEL_SHAPE` now matches the
  catalog's 33×33×3. The duplicate `train-0 copy/` folder is gone.
- **Phase 1** — `training/action_world_model.py`: action-conditioned
  recurrent world model (`z`, `a` → GRU state → next `z`), built from the
  already-recorded `motor.command` stream; horizons are declared in ticks
  and converted to recorded-frame steps via the measured vision rate
  (`horizons_ticks_to_frames`), for both the joint model and the existing
  per-scenario harness (`NurseryScenarioReport.horizon_frames` /
  `ticks_per_frame`).
- **Phase 2** — training uses short-rollout scheduled sampling
  (`warmup_frames`/`rollout_frames`), not 100-step compositions; evaluation
  reports `model_over_copy_last_mse` and `model_over_oracle_mse` (recurrence
  oracle) per horizon; `evaluate_rollout_health` /
  `rollout_health` flag frozen rollouts on both the new and the legacy
  predictor (`ccr nursery run` prints a FROZEN ROLLOUT warning).
- **Phase 3** — `run_nursery_joint` / `ccr nursery joint`: one model trained
  across scenarios with a vocabulary pinned to the full action space,
  evaluated on held-out seeds per scenario and zero-shot on held-out
  scenarios, plus `linear_probe_yaw` (does the latent/hidden state decode
  heading?). The gate holds `turn_in_place` to its premise: ≥ 360° yaw
  sweep, ≤ 0.02 blocks/tick drift, completed episode, ≥ 5% unique frames.

## Appendix: reproduction

All snippets read the session folders directly; frames resolve via
`frame_ref` into `frames/segment_*.bin` using the offsets in
`segment_*.index.jsonl`.

```python
import json, os
import numpy as np

def load_frames(sess):
    idx = [json.loads(l) for l in open(os.path.join(sess, "frames/segment_00000.index.jsonl"))]
    blob = open(os.path.join(sess, "frames/segment_00000.bin"), "rb").read()
    store = {e["hash"]: np.frombuffer(blob[e["offset"]:e["offset"]+e["length"]],
                                      dtype=e["dtype"]).reshape(e["shape"]) for e in idx}
    events = sorted(
        (r["timestamp"], r["frame_ref"])
        for r in map(json.loads, open(os.path.join(sess, "episode_00000.streams.jsonl")))
        if r.get("stream_id") == "vision.frame.pixels")
    return np.stack([store[h] for _, h in events]).astype(np.float64) / 255.0

seq = load_frames("shared/nursery-turn_in_place-train-1")

# rotation period: |diff| collapses at lag 12 (one 360° revolution)
for lag in (1, 6, 12, 24):
    print(lag, np.mean(np.abs(np.diff(seq[::1], axis=0))) if lag == 1 else
          np.mean(np.abs(seq[lag:] - seq[:-lag])))

# horizon table: copy-last vs mean-frame vs period-12 oracle
for h in (1, 4, 8):
    cur, tgt = seq[:-h], seq[h:]
    copy = np.mean((cur - tgt) ** 2)
    mean = np.mean((seq.mean(0) - tgt) ** 2)
    pairs = [(t + h, t + h - 12) for t in range(len(seq) - h) if t + h - 12 >= 0]
    oracle = np.mean([np.mean((seq[i] - seq[j]) ** 2) for i, j in pairs])
    print(f"t+{h}: copy {copy:.4f}  mean {mean:.4f}  oracle {oracle:.4f}")
```

Key per-session facts (from `episode_00000.summary.json` /
`spatial.position` / `input.mouse_look`):

| session | frames | position drift (blocks) | outcome |
|---------|--------|-------------------------|---------|
| train-0 | 85     | 12.6                    | died (`death:hit`) at tick 167 |
| train-1 | 201    | 0.0                     | completed |
| train-2 | 164    | 11.7                    | completed |
| train-3 | 149    | 16.2                    | completed |
| train-4 | 199    | 15.5                    | completed |
| train-5 | 166    | 24.4                    | completed |

`input.mouse_look` records `d_yaw = -15.0` on every acting tick in all six
sessions — the action stream needed for an action-conditioned world model is
already present in the data.
