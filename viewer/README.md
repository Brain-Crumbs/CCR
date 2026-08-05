# CCR Clinic

The clinic is the control and presentation layer for the Model Factory (epic
#212): a zero-dependency Node/HTTP API over recorded streams-v2 sessions,
factory runs, and detached jobs (`server.js`), and a React front end (`web/`,
built into `public/`) that lets you launch work and inspect one experiment run
at a time -- its pixels, its predictions at
every horizon, the actions taken tick by tick, and the factory's own
lineage/contract/champion evidence.

## Run

```bash
node viewer/server.js                       # serves runs + episode_cache + corpora on :8787
node viewer/server.js --data-dir /path/to/sessions --port 9000
node viewer/server.js --runs-dir runs --episode-cache-dir episode_cache --corpus-root corpora
node viewer/server.js --corpus-specs-dir specs/corpora
node viewer/server.js --host 0.0.0.0 --max-concurrent-jobs 4   # see Security below first
```

`--corpus-root` (default `<repo>/corpora`) is where a run's
`clinic_sessions.json` may point for frozen Model Factory corpus sessions
(`cognitive_runtime.training.model_factory.corpus`'s own default root),
mounted under a `corpus/` id prefix alongside the existing `run/`/`cache/`
recordings -- the same containment discipline as those two roots applies:
only a session inside one of the three configured roots is ever served.

Open http://localhost:8787 -- pick an organism, run, recording, and episode.

For front-end development, run the API server as above and, in a second
terminal, start the Vite dev server (proxies `/api` to `:8787`):

```bash
cd viewer/web && npm install && npm run dev   # http://localhost:5173
```

`npm run build` (from `viewer/` or `viewer/web/`) writes the production
bundle into `viewer/public/`, which `server.js` serves as static files
alongside its API -- the clinic is still one deployable.

## Security

The clinic's job-launch API (`POST /api/jobs/*`, `POST /api/preview/*`,
`POST /api/jobs/:id/cancel`) is **unauthenticated** -- anyone who can reach
the server can launch, preview, or cancel Model Factory training. There is
no login, no token, no per-route access control.

The mitigation is network exposure, not authentication: `--host` defaults to
`127.0.0.1`, so a plain `node viewer/server.js` is reachable only from the
same machine. Passing `--host 0.0.0.0` (or any other non-loopback address)
puts the job-launch API on that network with no further protection -- do
this only on a trusted network (e.g. behind your own firewall/VPN), never on
a network you don't control. Read-only clinic mode carries the same
consideration if you'd rather not share run contents broadly either.

`--max-concurrent-jobs` (default `2`) caps how many launched jobs may be
`running` at once -- a `POST /api/jobs/*` beyond the cap is refused with
`409`, since the training backend itself has no cross-run throttling for
concurrent CPU/`auto`-device trials (only per-file and per-device locks
exist). This bounds accidental pile-up, not malicious use.

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
- **Factory** -- the organism-wide state of the Factory build
  (`/api/factory-runs`: every run's mode, `state.json` state/reason, and
  last-updated time, whether or not it was ever promoted) above the
  champion registry: every (family, tier, objective) slot's leading
  champion and retained population, with a metrics table (often
  horizon-scoped, e.g. `rollout.t+4.model_mse`) so you can compare model
  outputs across the whole retained population at a glance, plus
  promote/hold history.
- **Compare** -- two runs' predictions for the same session/episode, side
  by side: pick Run A and Run B (scoped to the current organism), a shared
  session/episode picker (from Run A's recordings), and a shared tick
  cursor across both `PixelHorizonViewer` strips. Works fully when both
  runs trained against the same frozen corpus (the common clone/fine_tune
  case), since their prediction exports land in the same session
  directories; otherwise the side without a matching export falls back to
  baselines.
- **Build** -- launch a Model Factory trial without leaving the browser:
  pick an existing organism or create a new named organism, then choose a
  mode (fresh / clone / fine_tune / resume, sourced
  from `GET /api/factory-meta`). On a clean workspace, first launch one of
  the repository's corpus recipes and follow that corpus build in the same
  Jobs panel. The corpus builder remains available after the first corpus:
  a fresh baseline requires the generic action-effects recipe, while a
  goal-navigation corpus is fine-tune-only because it requires a parent
  checkpoint for retention measurement. Then fill in a form over `spec.data`/`model`/
  `training`/`evaluation` (fresh) or a completed run plus checkpoint and
  `--set` overrides (clone/fine_tune/resume), and launch it as a
  `POST /api/jobs/{baseline,clone,resume}` job. A Jobs panel underneath
  tracks every job launched for the organism -- status, each precomputed
  run's live state, a log tail, and a cancel action -- polling the exact
  routes described below rather than the hand-edited-dict notebook
  workflow this supersedes.
- **Evolve** -- run a bounded evolutionary campaign (`ccr factory search`)
  over one completed run's spec: that run's `trial_spec.json` becomes the
  base every candidate inherits (its `mode` and `parent` carry through, so a
  clone-mode base means a fixed-parent campaign), and only the declared
  genome schema's training genes vary. Set population size, generations,
  mutation rate, seed, and proposal method; the worst-case training count is
  stated up front, alongside the `{prefix}-p<generation>-c<candidate>` shape
  the campaign's runs will land under. **Dry-run preview** posts to
  `/api/preview/search`, which is the CLI's own torch-free `--dry-run`, and
  tabulates the resolved candidates against the dotted `training.*` paths
  that actually differ between them -- so the genome's active genes are
  *observed* from real proposals rather than re-declared in the front end.
  A **reference run** picker (defaulting to the organism's leading champion
  from `GET /api/registry`) sets `evaluation.reference_run`, making every
  candidate report `model_over_reference_mse` against that model's own
  predictions and gating the campaign on beating it -- the configurable
  comparison baseline, beyond copy-last-frame. Editing the selection metric
  sends one `--set evaluation.selection_metric=...` rather than a second
  copy of the evaluation block.

  After launch, a **population board** shows the campaign generation by
  generation: each candidate's live `state.json` pill and its resolved
  selection-metric value, the generation's leader marked, and each bred
  offspring's two parents. Only generation 0's run ids are precomputable --
  `run_evolutionary_search` fills a later generation's low slots with
  *carried* survivors, which keep the run ids they were first trained under
  -- so the board recovers each generation's survivor count from its lowest
  offspring slot and names the survivors from the offspring's own
  `lineage.json` parents. It re-reads only what the campaign wrote to disk;
  no selection or breeding decision is re-derived in the browser.

  An **Architecture (NAS)** toggle switches the tab to the nested campaign
  (`ccr factory search --genome architecture`): an outer evolutionary loop
  over model architectures, each scored by a complete inner training-gene
  campaign of its own. The outer loop gets its own schema registry
  (`architecture_genome.ARCHITECTURE_GENOME_SCHEMAS`, offered separately
  from the training schemas because the two are mutually exclusive by
  construction), population size, generations and mutation rate; the
  existing population/generation/mutation fields keep their meaning as that
  *inner* campaign's parameters, exactly as the CLI declares them. Because
  the two budgets multiply, the cost bound
  (`outer_size × outer_generations × inner_size × inner_generations`, times
  the base spec's own `training.max_training_seconds`) is stated as a figure
  rather than a hint, and **Launch campaign** stays disabled until an
  explicit checkbox acknowledges it -- any edit that changes the cost
  withdraws the acknowledgement. The dry run is free and stays available
  either way; once it returns, the figure quoted is
  `architecture_search.estimate_cost`'s own rather than the client-side one.
  After launch an **architecture board** replaces the population board:
  one row per architecture per outer generation, with its inner campaign's
  aggregate state and best selection-metric value, each expandable into that
  architecture's own population grid (the same component, fed from the one
  organism-wide poll rather than opening a poller per expanded row).

  Retention is the one campaign fact run ids cannot express: an architecture
  retained into the next outer generation is deliberately not re-trained, so
  it allocates no runs under that generation's prefix and is
  indistinguishable, from run directories alone, from one the campaign has
  not reached yet. So the campaign publishes its own decisions as it makes
  them, to `<runs>/<organism>/.architecture-campaigns/<campaign>.json`
  (`architecture_search.campaign_progress_path`; a dot-directory, inert to
  the run catalog scan), served by `GET /api/architecture-campaigns`. The
  board reads carried/retained/eliminated from it -- a carried slot is
  labelled as such, names the architecture whose evidence it carries, and
  opens onto the inner grid that architecture already ran. A campaign that
  never published one (an older run, a read-only root) still renders from
  run ids alone, just without naming its carried slots.
- **Breed** -- cross two completed runs of one organism into a single
  explicit-lineage child (`ccr factory breed`). Both parent pickers offer
  only `completed` runs, because breeding reads a parent's frozen genome and
  checkpoint and refuses anything that never finished evaluation -- the same
  gate promotion enforces. Editing any parent, checkpoint, schema, seed,
  mutation rate, objective, tier, generation or weight donor re-runs
  `POST /api/preview/breed` (the CLI's own torch-free `--dry-run`) after a
  short debounce, so the panel always shows either the exact child these
  settings would produce or why they cannot: an incompatible pair renders
  `check_compatible`'s own message verbatim, which names *every* mismatched
  dimension at once (architecture hash, corpus, stage, budget tier,
  evaluation contract) rather than just the first. The child preview is
  read straight out of the dry run's `breeding_lineage`: one row per gene
  with both parents' values, which parent uniform crossover took it from,
  and the child's value, flagged where mutation or repair then moved it --
  plus the resolved child spec's mode, weight-donor parent and checkpoint,
  corpus, objective and selection metric. **Breed child** stays disabled
  until a preview of *exactly these settings* has succeeded -- an edit made
  while the previous check is still on screen marks it stale rather than
  letting a click breed a child the panel isn't describing -- and launches
  the same subcommand minus `--dry-run` as a `POST /api/jobs/breed` job,
  tracked in the shared Jobs panel. The same seed, parents and schema always
  reproduce the same child.

Each recording also shows its `record/quality.py` green/amber/red verdict
before you ever train on it.

## Directory layout

```text
viewer/
  server.js            # Node/HTTP viewer + local control plane (no framework, no deps)
  export_predictions.py
  public/              # built React bundle (npm run build writes here)
  web/                 # React source
    src/
      lib/             # pure data transforms (frame/prediction math, diagnostics, actions, format)
      hooks/           # usePixelHorizonData, useDarkMode
      components/      # PixelHorizonViewer, SeenFramePanel, ActionTimeline, EEGPanel, BuildPanel, EvolvePanel, PopulationBoard, ArchitectureBoard, BreedPanel, JobsPanel, ...
      App.jsx
  test/                # server.js contract tests (node:test)
```

## Prediction sources

- **copy-last** and **mean-frame** work on any recorded session with pixel
  frames (`--record-frames`); they are exactly the baselines
  `evaluate_ego_motion_holdout` benchmarks the model against.
- **model** appears when a `predictions_<episode>.json` file sits next to the
  episode's stream log. `run_nursery_scenario` (and `ccr nursery run`) writes
  these for every recorded session by default
  (`NurseryConfig.export_predictions` / `--no-export-predictions`), because
  the nursery checkpoint only persists the pixel *encoder* -- predicted
  frames are unrecoverable after the run unless exported. `nursery run
  --out-dir` also saves `<scenario>-full.pt`, a full encoder+decoder+predictor
  bundle for re-exporting later. `ccr factory baseline`/`ccr factory clone`
  (`model_factory.runner.run_trial`) export the same way for up to 3
  validation episodes from the promoted checkpoint by default
  (`--export-predictions-max N` / `--no-export-predictions`) -- written into
  the frozen corpus session's own directory (`<experiment_id>-predictions_
  <episode>.json`), which is what lets the Compare tab show two runs
  trained against the same corpus side by side.

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

Model predictions live in the decoder's downsampled reconstruction space
(default 16x16), the same space the training losses and holdout PSNR/SSIM
use; the export bundles the identically pooled actual targets so the
viewer's model-mode numbers match the harness.

## API

JSON, all under `/api`. Clinic mode (default) joins a selected organism/run
to its recordings; pass `--data-dir` for the single-session-tree
compatibility mode instead. Every route below `GET` reads; the `POST` routes
launch, preview, or cancel Model Factory jobs and are unauthenticated --
see Security above before exposing them beyond localhost.

| Route | Returns |
| --- | --- |
| `GET /api/catalog` | organisms and runs available under `--runs-dir` |
| `GET /api/runs?organism=&run=` | the run's experiment_report.json header slice (model, training, evaluation, promotion) |
| `GET /api/experiments?organism=&run=` | the run's Model Factory manifests as-is: `trial_spec`, `contracts`, `lineage`, `data_manifest`, `execution`, `experiment_report`, and `metrics.{validation,test,comparison}` |
| `GET /api/registry?organism=` | the organism's champion registry (`registry.json`) |
| `GET /api/factory-runs?organism=` | every run under the organism with its lineage mode, `state.json` state/reason/updated_at, promotion outcome, its own `selection_metric` resolved to a `selection_metric_value` from `metrics/validation.json` (`null` until evaluated), and its `lineage.json` `configuration_parents` -- queued/running/completed/failed across the whole Factory build, not just promoted runs |
| `GET /api/architecture-campaigns?organism=` | every nested architecture (NAS) campaign's live progress document for the organism, newest first -- the outer loop's own retention/carry/breeding decisions as it makes them, which run ids alone cannot express (a retained architecture is not re-run, so it allocates no runs under the later generation's prefix). Empty for an organism that never ran one |
| `GET /api/sessions?organism=&run=&name=&...filters` | recordings for the selected run, with quality verdicts |
| `GET /api/sessions/:id` | one session's streams, decisions, exports, and quality verdict |
| `GET /api/sessions/:id/episodes/:eid/{streams,decisions,frames,predictions}` | per-episode records; `predictions` accepts `?kind=dream` and `?experiment=` |
| `POST /api/organisms` | creates a new organism namespace under `--corpus-root`; body: `{ "organism": "Name" }` |
| `POST /api/jobs/{corpus,baseline,clone,resume,search,breed}` | launches the matching `ccr factory ...` workflow as a detached subprocess; body shape per kind in `viewer/lib/jobs.js`'s `buildLaunch`. `409` past `--max-concurrent-jobs` |
| `GET /api/jobs?organism=` | every launched job (registry entry) plus each precomputed run's live `state.json`/heartbeat |
| `GET /api/jobs/:id/log?offset=` | the job's combined stdout+stderr log, tailed from a byte offset |
| `POST /api/jobs/:id/cancel` | asks every one of the job's runs to stop gracefully, then `SIGTERM`s the subprocess |
| `GET /api/corpora?organism=` | frozen Model Factory corpora under `--corpus-root` (id, data contract hash, session counts per split) |
| `GET /api/corpus-specs?organism=` | repository corpus recipes under `--corpus-specs-dir`; the Build tab can launch one when a clean workspace has no frozen corpus yet |
| `POST /api/preview/{search,breed}` | the same CLI invocation a launch would use, plus `--dry-run` -- candidate specs or the resolved child + lineage, no run allocated |
| `GET /api/factory-meta` | the factory's own declared modes/objectives/genome schema versions (training *and* architecture)/backbone presets (`ccr factory meta`), so a form can source its options from the backend instead of duplicating them |

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
