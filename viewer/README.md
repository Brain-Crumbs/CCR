# CCR Clinic

The clinic is the presentation layer for the Model Factory (epic #212): a
read-only Node/HTTP API over recorded streams-v2 sessions and factory runs
(`server.js`), and a React front end (`web/`, built into `public/`) that lets
you inspect one experiment run at a time -- its pixels, its predictions at
every horizon, the actions taken tick by tick, and the factory's own
lineage/contract/champion evidence.

## Run

```bash
node viewer/server.js                       # serves runs + episode_cache + corpora on :8787
node viewer/server.js --runs-dir runs --corpus-root corpora   # match wherever `ccr factory ...` writes
node viewer/server.js --data-dir /path/to/sessions --port 9000
```

Open http://localhost:8787 -- pick an organism, run, recording, and episode.

For front-end development, run the API server as above and, in a second
terminal, start the Vite dev server (proxies `/api` to `:8787`):

```bash
cd viewer/web && npm install && npm run dev   # http://localhost:5173
```

`npm run build` (from `viewer/` or `viewer/web/`) writes the production
bundle into `viewer/public/`, which `server.js` serves as static files
alongside its API -- the clinic is still one deployable.

## What it shows

Per selected organism/run, tabbed by concern:

- **Episode** -- one strip per tick: the pixel frame seen at *t* paired with
  the action taken then (voluntary vs. actuated, and any reflex/caregiver
  override that changed it), followed by one panel per prediction horizon
  with the model's predicted frame, the actual frame, an |error| heatmap,
  and MSE/PSNR against the copy-last and mean-frame baselines -- plus an
  MSE-over-time chart so a horizon's accuracy is visible across the whole
  episode, not just one tick. EEG (neuromodulators, prediction error, arbiter
  mode) and attention/focus panels share the same tick cursor, so clicking
  any of them scrubs the rest.
- **Development** -- the organism's developmental-ladder stages and
  milestones (Phase 7 gates).
- **Experiment** -- the run's Model Factory manifests: lineage (fresh /
  clone / resume / fine_tune, parent, weight donor), architecture/data/
  training contract hashes, execution provenance (commit, device,
  precision), the promotion verdict and its reasons, and -- when this run is
  a clone/fine_tune -- its paired comparison against its parent (mean delta,
  win rate, bootstrap/permutation confidence intervals).
- **Factory** -- every run launched for the organism and its
  `state.json` position (queued/running/checkpointing/completed/
  budget_exceeded/failed/cancelled), not just promoted ones -- click a row
  to select that run everywhere else in the clinic -- plus the champion
  registry: every (family, tier, objective) slot's leading champion and
  retained population, with a metrics table (often horizon-scoped, e.g.
  `rollout.t+4.model_mse`) so you can compare model outputs across the whole
  retained population at a glance, plus promote/hold history.
- **Compare** -- two runs' predictions on the same session, side by side,
  each with its own run summary and tick-synced horizon strips -- "inspect
  models against each other" rather than one at a time. Works fully when
  both runs were evaluated against the same frozen corpus (the normal case
  for a clone/fine_tune lineage); a run missing an export for the chosen
  session just falls back to the baselines in that column.

Each recording also shows its `record/quality.py` green/amber/red verdict
before you ever train on it.

## Directory layout

```text
viewer/
  server.js            # read-only Node/HTTP API (no framework, no deps)
  export_predictions.py
  public/              # built React bundle (npm run build writes here)
  web/                 # React source
    src/
      lib/             # pure data transforms (frame/prediction math, diagnostics, actions, format)
      hooks/           # usePixelHorizonData, useDarkMode
      components/      # PixelHorizonViewer, SeenFramePanel, ActionTimeline, EEGPanel, ...
      App.jsx
  test/                # server.js contract tests (node:test)
```

## Prediction sources

- **copy-last** and **mean-frame** work on any recorded session with pixel
  frames (`--record-frames`); they are exactly the baselines
  `evaluate_ego_motion_holdout` benchmarks the model against.
- **model** appears when a `<id>-predictions_<episode>.json` file exists for
  the selected experiment. Two pipelines write these automatically:
  - `run_nursery_scenario` (and `ccr nursery run`) exports one for every
    recorded session by default (`NurseryConfig.export_predictions` /
    `--no-export-predictions`), next to the episode's stream log, because
    the nursery checkpoint only persists the pixel *encoder* -- predicted
    frames are unrecoverable after the run unless exported. `nursery run
    --out-dir` also saves `<scenario>-full.pt`, a full
    encoder+decoder+predictor bundle for re-exporting later.
  - `ccr factory baseline`/`ccr factory clone` export from the selected
    checkpoint for up to `--export-predictions-max N` validation episodes
    (default 3; `--no-export-predictions` to skip) -- see
    [Model Factory exports](#model-factory-exports) below.

Live `CortexWorldModel` runs also place decoded horizon frames in each
`DecisionRecord`. When no offline export exists, the clinic assembles those
records into the same `pixel-predictions-v1` response and labels the source
**model (live record)**.

```bash
python -m cognitive_runtime.training.prediction_export \
    --model out/walk_forward-full.pt \
    --session shared/nursery-walk_forward-train-0 --horizons 1,10,100
```

or in code, while the trained model is in memory:

```python
from cognitive_runtime.training.prediction_export import (
    export_prediction_file, save_full_visual_model,
)

export_prediction_file(model, session_dir, "episode_00000", (1, 10, 100))
save_full_visual_model(model, "walk_forward-full.pt")
```

(`viewer/export_predictions.py` remains as a shim re-exporting the same
functions.)

## Model Factory exports

`run_trial` (`cognitive_runtime/training/model_factory/runner.py`) does two
things automatically, before and after training, purely for the clinic --
neither is required for the trial itself, and a failure in either never
fails the trial:

- Writes `clinic_sessions.json` into the run directory as soon as it's
  allocated (before training starts), listing every train/validation
  session from the frozen corpus. This is what makes a run's recordings
  browsable in the clinic immediately -- including while the trial is still
  `running` -- since EEG/attention/action data lives in each session's own
  streams/decisions files, independent of any prediction export.
- Exports pixel predictions from the selected (`best-validation.pt`)
  checkpoint for up to `--export-predictions-max` validation episodes,
  **into the run's own `predictions/` directory** -- deliberately not into
  the validation session's directory. A frozen corpus session's content
  hash (`corpus.py`'s `_session_hash`) covers every file in that directory;
  writing a prediction export there would silently break every later
  trial's reuse of the same corpus (`resolve_corpus` would see a "modified"
  session and refuse it). `viewer/server.js`'s predictions endpoint already
  resolves the currently-selected run's directory from `?run=`, so it
  checks there too, alongside the session's own directory (which is still
  where a nursery/`episode_cache` export legitimately lives).

Because each run's export lives in its own directory rather than the
shared corpus, two runs trained against the same corpus never collide --
each is independently addressable by `?experiment=<run_id>`, which is what
the **Compare** tab uses to fetch two runs' predictions for one session.

Model predictions live in the decoder's downsampled reconstruction space
(default 16x16), the same space the training losses and holdout PSNR/SSIM
use; the export bundles the identically pooled actual targets so the
viewer's model-mode numbers match the harness.

## API

Read-only JSON, all under `/api`. Clinic mode (default) joins a selected
organism/run to its recordings under `--runs-dir`, `--episode-cache-dir`
(nursery recordings), and `--corpus-root` (frozen Model Factory corpora);
pass `--data-dir` for the single-session-tree compatibility mode instead.

| Route | Returns |
| --- | --- |
| `GET /api/catalog` | organisms and runs available under `--runs-dir` |
| `GET /api/runs?organism=&run=` | the run's experiment_report.json header slice (model, training, evaluation, promotion) |
| `GET /api/experiments?organism=&run=` | the run's Model Factory manifests as-is: `trial_spec`, `contracts`, `lineage`, `data_manifest`, `execution`, `experiment_report`, and `metrics.{validation,test,comparison}` |
| `GET /api/registry?organism=` | the organism's champion registry (`registry.json`) |
| `GET /api/factory-runs?organism=` | every run's `state.json` position, lineage mode, and promotion verdict |
| `GET /api/sessions?organism=&run=&name=&...filters` | recordings for the selected run, with quality verdicts |
| `GET /api/sessions/:id` | one session's streams, decisions, exports, and quality verdict |
| `GET /api/sessions/:id/episodes/:eid/{streams,decisions,frames,predictions}` | per-episode records; `predictions` accepts `?kind=dream` and `?experiment=` |

## Reusing PixelHorizonViewer

`viewer/web/src/components/PixelHorizonViewer.jsx` is a plain React
component (no clinic-specific state beyond its props): `framesSrc` and
`predictionsSrc` point at the two endpoints above, `decisions` pairs the
leading seen-frame cell with the action taken that tick, and
`tick`/`onTickChange` let a parent share one time cursor across it and
sibling panels (see `App.jsx` for the wiring).

`frames-src` must return:

```json
{
  "shape": [33, 33, 3], "dtype": "uint8", "n_frames": 201,
  "frames": [{"i": 0, "t": 0.0, "seq": 0, "hash": "...", "data": "<base64 raw HWC uint8>"}]
}
```

`predictions-src` must return the `pixel-predictions-v1`/`-v2` format written
by `viewer/export_predictions.py` and
`cognitive_runtime.training.prediction_export` (documented in that module's
docstring). A 404 is fine -- the component falls back to the baselines.

## Tests

- `viewer/test/*.test.js` (`npm test`, `node:test`) -- server contract tests
  against recorded fixtures: sessions by organism, streams/decisions/exports,
  quality verdicts, Model Factory manifest and registry endpoints.
- `viewer/web/src/**/*.test.{js,jsx}` (`npm run test:web`, Vitest +
  React Testing Library) -- pure data-transform unit tests and component
  tests for every panel.
