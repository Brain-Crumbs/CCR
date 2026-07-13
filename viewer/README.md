# CCR pixel prediction viewer

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

```bash
python -m cognitive_runtime.training.prediction_export \
    --model models/nursery/pathfinder-world-model.pt \
    --session sessions/live-pathfinder/nursery-pathfinder-holdout-0 \
    --horizons 1,4,8
```

or in code, while the trained model is in memory:

```python
from cognitive_runtime.training.prediction_export import (
    export_action_prediction_file,
)

export_prediction_file(model, session_dir, "episode_00000", (1, 4, 8))
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
    horizons="1,4,8" scale="6">
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
      horizons="1,4,8"
      scale="6"
    />
  );
}
```

Attributes: `frames-src` (required), `predictions-src` (optional),
`horizons` (default `1,4,8`), `scale` (CSS px per frame pixel, default 6).
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
