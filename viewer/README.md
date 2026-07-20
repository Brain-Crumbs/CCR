# CCR Clinic

The viewer now opens on a React session browser grouped by the organism name
recorded in Phase 0. Every card displays the authoritative Record quality
verdict and its failing checks before the session is used for training.

A lightweight, zero-dependency node server plus a reusable
`<pixel-horizon-viewer>` web component for inspecting recorded streams-v2
sessions: for each prediction horizon it shows the **actual** frame at `t+h`
next to a **predicted** frame at `t+h`, an |error| heatmap, MSE/PSNR
readouts, a scrubber/playback over the episode, and an MSE-over-time chart.

## Run

```bash
node viewer/server.js                       # serves ./shared on :8787
node viewer/server.js --data-dir /path/to/sessions --port 9000
```

Open http://localhost:8787 — pick a session and episode.

The read-only API supports `GET /api/sessions?name=Pixel` and
`GET /api/sessions/:id`; the detail response contains all recorded stream
events, JSON exports, and the `record.quality` verdict. Existing episode
`frames` and `predictions` endpoints remain available to viewer panels.

## Prediction sources

- **copy-last** and **mean-frame** work on any recorded session with pixel
  frames (`--record-frames`); they are exactly the baselines
  `evaluate_ego_motion_holdout` benchmarks the model against.
- **model** appears when a `predictions_<episode>.json` file sits next to the
  episode's stream log. `run_nursery_scenario` (and `ccr nursery run`) writes
  these for every recorded session by default
  (`NurseryConfig.export_predictions` / `--no-export-predictions`), because
  the nursery checkpoint only persists the pixel *encoder* — predicted frames
  are unrecoverable after the run unless exported. `nursery run --out-dir`
  also saves `<scenario>-full.pt`, a full encoder+decoder+predictor bundle
  for re-exporting later:

Live `CortexWorldModel` runs also place decoded horizon frames in each
`DecisionRecord`. When no offline export exists, the clinic assembles those
records into the same `pixel-predictions-v1` response and labels the source
**model (live record)**.

The episode frame scrubber is shared with the EEG and arbiter-mode timelines:
moving either pixel viewer highlights the matching cognitive tick, while
clicking a trace or mode tick moves both pixel viewers to its corresponding
recorded frame.

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
(default 16×16), the same space the training losses and holdout PSNR/SSIM
use; the export bundles the identically pooled actual targets so the
viewer's model-mode numbers match the harness.

## Reusing the component

The component is a plain custom element with no framework dependency —
`viewer/public/pixel-horizon-viewer.js` is the only file to copy. It only
needs two JSON endpoints (see below), not this server.

Plain HTML:

```html
<script type="module" src="/pixel-horizon-viewer.js"></script>
<pixel-horizon-viewer
    frames-src="/api/sessions/S/episodes/episode_00000/frames"
    predictions-src="/api/sessions/S/episodes/episode_00000/predictions"
    horizons="1,10,100" scale="6">
</pixel-horizon-viewer>
```

React (custom elements render as-is; attributes are strings):

```jsx
import "./pixel-horizon-viewer.js";

export function EpisodeViewer({ session, episode }) {
  return (
    <pixel-horizon-viewer
      frames-src={`/api/sessions/${session}/episodes/${episode}/frames`}
      predictions-src={`/api/sessions/${session}/episodes/${episode}/predictions`}
      horizons="1,10,100"
      scale="6"
    />
  );
}
```

Attributes: `frames-src` (required), `predictions-src` (optional),
`horizons` (default `1,10,100`), `scale` (CSS px per frame pixel, default 6).
Light/dark theme follows `prefers-color-scheme`.

## Endpoint contracts

`frames-src` must return:

```json
{
  "shape": [33, 33, 3], "dtype": "uint8", "n_frames": 201,
  "frames": [{"i": 0, "t": 0.0, "seq": 0, "hash": "…", "data": "<base64 raw HWC uint8>"}]
}
```

`predictions-src` must return the `pixel-predictions-v1` format written by
`viewer/export_predictions.py` (documented in that module's docstring). A
404 is fine — the component falls back to the baselines.
