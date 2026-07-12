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
  episode's stream log. The nursery checkpoint only persists the pixel
  *encoder*, so export predictions while the full model is in memory:

```python
from cognitive_runtime.training.nursery import run_nursery_scenario
from viewer.export_predictions import export_prediction_file, save_full_visual_model

model, report = run_nursery_scenario("shared", "walk_forward")
for session_dir in report.train_sessions + report.holdout_sessions:
    export_prediction_file(model, session_dir, "episode_00000", (1, 10, 100))
save_full_visual_model(model, "walk_forward-full.pt")   # re-export later without retraining
```

or later, from a saved full-model bundle:

```bash
python -m viewer.export_predictions --model walk_forward-full.pt \
    --session shared/nursery-walk_forward-train-0 --horizons 1,10,100
```

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
