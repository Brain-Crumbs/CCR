# Validation notes: first real `nursery/walk_forward` run (`shared/`)

Review of `shared/nursery-walk_forward-train-{0..5}` (2026-07-12). Every
check below is reproducible from the stream logs; the viewer under
`viewer/` renders the frames these findings describe.

## What checks out

- **Frame store integrity**: every `segment_*.bin` frame matches its index
  hash under `content-bytes-v1` (`ndarray|dtype|shape|` prefix + raw bytes),
  and every `frame_ref` in the stream logs resolves in the store.
- **Log consistency**: `streams.jsonl`, `decisions.jsonl` and
  `summary.json` agree on reward totals (0.91 each), tick counts (400),
  and motor emissions (400× `MOVE_FORWARD`).
- **Shapes**: all pixel frames are 33×33×3 uint8 as the catalog declares.

## Issues found

1. **The run recorded the remote backend's persistent world, so the seeds
   did nothing.** All six sessions carry `program_tags: [... "remote"]` and
   `source: "remote"`. `walk_forward` is designed around
   `world.reset(seed)` producing varied terrain per seed; on the remote
   backend (auto-selected when `CCR_MINECRAFT_HOST` is set, which also
   forces `realtime`) the server's current scene persists **across
   sessions**: train-1's first frame hash equals train-0's last frame hash,
   and train-1..5 all start at exactly (-38.4, 24.3) — where train-0 ended.

2. **The agent is stuck for ~95 % of the recorded data.** In train-0 it
   walks 3.76 blocks in the first 2.9 s, then its position never changes
   again despite `MOVE_FORWARD` every tick (no `event.bumped` is emitted).
   In train-1..5 net displacement is 0.00 for the whole episode. Unique
   frames per session: 12, 4, 4, 5, 4, 5 — out of 201 pixel events each.
   The dataset therefore contains almost no ego-motion signal (the one
   regularity `walk_forward` exists to capture), nearly every adjacent
   training pair is (identical, identical), and the copy-last baseline has
   ~zero MSE (PSNR → ∞), making `beats_copy_last` unattainable. Expect the
   first metrics to look catastrophic for data reasons, not model reasons.

3. **Vision streams run at half the declared rate.** `vision.frame.pixels`
   and `vision.frame.grid` log 201 events per 400 ticks (~10 Hz in
   tick-time; 8.6 Hz wall-clock) while the catalog declares 20 Hz — with a
   doubled event on tick 0. Horizons are counted in *frame steps*
   (`load_episode_pixel_frames`), so on this data `t+1/t+10/t+100` means
   0.1 s/1 s/10 s — twice the simulated-backend timescale. Any comparison
   against simulated-run metrics is apples-to-oranges until the rate is
   fixed or horizons are rescaled.

4. **Realtime scheduling couldn't keep up.** `missed_ticks`/`late_windows`
   = 396/400 (99 %), 17.2 ticks/s against the 20 Hz target. For recording
   training data, prefer the simulated backend (`realtime=False`) or lower
   the tick rate; a recording where nearly every window is late has soft
   timing semantics throughout.

5. **No holdout sessions are present.** `run_nursery_scenario` records
   `nursery-walk_forward-holdout-*` right after the train seeds; `shared/`
   has only train-0..5. Evaluation and dream strips run on holdout sessions
   only, so this data alone can't reproduce the benchmark.

6. **Metadata mismatches on remote.** `session.json` declares
   `spatial.position` range `[0, 48]` (`world_size`), but remote
   coordinates are negative (x ≈ -39). Anything normalizing by the declared
   range mis-scales. Cosmetic but confusing: `summary.silent_streams`
   flags `vision.frame.pixels`/`grid` as silent (they publish every other
   window), and `success: true` merely means the episode ran to
   `episode_ticks`.

7. **Reward flatlines by design.** After ~20 repeated actions the
   `repeated_action: -0.01` penalty exactly cancels `tick_alive: +0.01`, so
   every session totals 0.91 regardless of whether the agent moved.
   Harmless for world-model pretraining, but total reward is useless as a
   data-quality signal for nursery runs.

8. **Checkpoints can't regenerate predictions.**
   `save_nursery_scenario_checkpoint` persists only the pixel *encoder*;
   the decoder and next-latent predictor are lost when the process exits.
   Use `viewer/export_predictions.py` (`save_full_visual_model` /
   `export_prediction_file`) right after training to keep predicted frames
   inspectable.

## Recommendations

- Record nursery data on the **simulated backend** (unset
  `CCR_NURSERY_BACKEND`/`CCR_MINECRAFT_HOST`, or pass `--backend simulated`)
  so seeds vary terrain and `realtime` stays off.
- If remote recording is required: teleport/reposition the agent per
  session, verify displacement per episode (a `max_distance_from_spawn`
  floor would have caught 5 of these 6 sessions), and reconcile the vision
  stream rate with the catalog.
- Add a post-record sanity gate before training: minimum unique-frame
  count and minimum displacement per episode.

## Fixes landed since this review

- **Data-quality gate** (finding 2): `run_nursery_scenario` now measures
  every recorded episode (unique-frame fraction, net displacement per tick)
  against the scenario's declared expectations and refuses to train on
  recordings without the scenario's signal
  (`NurseryConfig.data_quality_gate`, `--skip-data-quality-gate`). The six
  sessions reviewed here all fail it.
- **Catalog honesty in realtime** (findings 3 & 6): `stream_catalog()` now
  declares the paced rates (`realtime_vision_hz`, `realtime_body_heartbeat_hz`)
  in realtime mode instead of the 20 Hz tick cadence, and the remote backend
  drops the `[0, world_size]` position/distance ranges its live-server
  coordinates never respected.
- **Predictions survive the run** (finding 8): after training,
  `run_nursery_scenario` exports `predictions_<episode>.json` (the pixel
  viewer's "model" source) for every recorded session by default
  (`NurseryConfig.export_predictions`, `--no-export-predictions`), and
  `nursery run --out-dir` also saves `<scenario>-full.pt` — a full
  encoder+decoder+predictor bundle
  (`cognitive_runtime.training.prediction_export`) that can re-export
  predictions later without retraining.
- **Louder remote warning** (finding 1): `nursery run` on a non-simulated
  backend now spells out that seeds do not vary terrain, sessions inherit
  the previous session's agent position, and vision is paced below the tick
  rate.

Still open: the agent getting stuck without an `event.bumped` (the
mineflayer bridge only emits it when the blocking block has
`boundingBox === 'block'` — whatever blocked this run didn't) needs a live
server to reproduce; the half-rate recording itself is by design
(`realtime_vision_hz`), so horizon comparisons across backends must
rescale, which the corrected catalog metadata now makes visible.
