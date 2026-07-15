"""Export model pixel predictions per horizon for the pixel viewer.

The viewer (``viewer/server.js`` + ``<pixel-horizon-viewer>``) renders the
copy-last and mean-frame baselines straight from a recorded session; to also
show a *trained model's* predicted frames it needs a
``predictions_<episode>.json`` file next to the episode's stream log. This
module writes that file.

``run_nursery_scenario`` calls :func:`export_prediction_file` for every
recorded session by default (``NurseryConfig.export_predictions``), because
the nursery checkpoint (``save_nursery_scenario_checkpoint``) persists only
the pixel *encoder* -- the decoder and next-latent predictor needed to roll
predictions forward would otherwise be lost when the process exits.  To
re-export later without retraining, persist the whole model with
:func:`save_full_visual_model` and use the CLI::

    python -m cognitive_runtime.training.prediction_export \
        --model walk_forward-full.pt \
        --session shared/nursery-walk_forward-train-0 --horizons 1,10,100

Output format ("pixel-predictions-v1")::

    {
      "format": "pixel-predictions-v1",
      "session_id": ..., "episode_id": ...,
      "horizons": [1, 10, 100],
      "prediction_shape": [16, 16, 3],       # decoder output (reconstruction space)
      "n_frames": 201,
      "predictions": {"1": {"frames": ["<b64 uint8 rgb>", ...]}},   # frames[t] = prediction for t+h
      "targets": ["<b64 uint8 rgb>", ...]     # pooled actual frames, index-aligned
    }

Predictions live in the decoder's downsampled reconstruction space (the same
space the training losses and PSNR/SSIM benchmarks use), so the viewer's
model-mode diffs match the harness numbers; the ``targets`` array carries the
identically pooled actual frames so no client-side resampling is needed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from typing import Any, Dict, Optional, Sequence

import torch

from cognitive_runtime.neural.pixel_stream_encoder import pixels_to_chw
from cognitive_runtime.runtime.replay import list_episodes
from cognitive_runtime.training.datasets import load_episode_pixel_frames
from cognitive_runtime.training.visual_representation import (
    VisualRepresentationModel,
    reconstruction_target,
)

FULL_MODEL_FORMAT = "visual-representation-full-v1"


def save_full_visual_model(model: VisualRepresentationModel, path: str) -> None:
    """Save encoder+decoder+next-predictor so predictions can be exported
    later (the nursery checkpoint keeps only the encoder)."""
    torch.save(
        {
            "format": FULL_MODEL_FORMAT,
            "pixel_shape": list(model.pixel_shape),
            "latent_width": model.latent_width,
            "reconstruction_shape": list(model.reconstruction_shape),
            "hidden_dim": model.decoder.net[0].out_features,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_full_visual_model(path: str) -> VisualRepresentationModel:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FULL_MODEL_FORMAT:
        raise ValueError(
            f"{path!r} is not a {FULL_MODEL_FORMAT} bundle (got {payload.get('format')!r}); "
            "save one with prediction_export.save_full_visual_model"
        )
    model = VisualRepresentationModel(
        tuple(payload["pixel_shape"]),
        latent_width=payload["latent_width"],
        reconstruction_shape=tuple(payload["reconstruction_shape"]),
        hidden_dim=payload["hidden_dim"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _session_name(session_dir: str) -> Optional[str]:
    """Best-effort read of the recorded organism name (issue #88) from
    ``session.json``, for prefixing export filenames; `None` for sessions
    recorded before the field existed or with no metadata at all."""
    try:
        with open(os.path.join(session_dir, "session.json"), encoding="utf-8") as fh:
            return json.load(fh).get("name")
    except (OSError, ValueError):
        return None


def _b64_frame(chw: torch.Tensor) -> str:
    """``Tensor[C, H, W]`` in [0, 1] -> base64 of HWC uint8 bytes."""
    hwc = (chw.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).permute(1, 2, 0)
    return base64.b64encode(hwc.contiguous().numpy().tobytes()).decode("ascii")


def export_prediction_file(
    model: VisualRepresentationModel,
    session_dir: str,
    episode_id: str,
    horizons: Sequence[int],
    out_path: str | None = None,
    name: Optional[str] = None,
) -> str:
    """Roll the model out from every start frame and write
    ``predictions_<episode>.json`` into ``session_dir``. Returns the path.

    The filename is prefixed with the organism ``name`` (issue #88) --
    explicit if given, else read from the session's own recorded metadata,
    else left unprefixed for a legacy nameless session."""

    horizons_sorted = sorted({int(h) for h in horizons if int(h) > 0})
    if not horizons_sorted:
        raise ValueError("horizons must contain at least one positive offset")
    frames = load_episode_pixel_frames(session_dir, episode_id)
    max_horizon = horizons_sorted[-1]
    if len(frames) <= max_horizon:
        raise ValueError(
            f"{session_dir}/{episode_id} has {len(frames)} frames, too short for horizon {max_horizon}"
        )

    pixel_tensors = torch.stack([pixels_to_chw(f) for f in frames])
    targets = reconstruction_target(pixel_tensors, model.reconstruction_shape)

    was_training = model.training
    model.eval()
    predictions: Dict[str, Any] = {str(h): {"frames": []} for h in horizons_sorted}
    with torch.no_grad():
        latents = model.encoder(pixel_tensors)
        for t in range(len(frames) - 1):
            rolled = latents[t : t + 1]
            for step in range(1, max_horizon + 1):
                if t + step >= len(frames):
                    break
                rolled = model.next_predictor(rolled)
                if step in horizons_sorted:
                    predictions[str(step)]["frames"].append(_b64_frame(model.decoder(rolled).squeeze(0)))
    if was_training:
        model.train()

    payload = {
        "format": "pixel-predictions-v1",
        "session_id": os.path.basename(os.path.normpath(session_dir)),
        "episode_id": episode_id,
        "horizons": horizons_sorted,
        "prediction_shape": list(model.reconstruction_shape),
        "n_frames": len(frames),
        "predictions": predictions,
        "targets": [_b64_frame(targets[i]) for i in range(len(frames))],
    }
    if out_path is None:
        resolved_name = name or _session_name(session_dir)
        filename = (
            f"{resolved_name}-predictions_{episode_id}.json"
            if resolved_name else f"predictions_{episode_id}.json"
        )
        out_path = os.path.join(session_dir, filename)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return out_path


def export_session_predictions(
    model: VisualRepresentationModel,
    session_dirs: Sequence[str],
    horizons: Sequence[int],
) -> Dict[str, str]:
    """Export every episode of every session; returns
    ``{f"{session_dir}/{episode_id}": prediction_file_path}``.  Episodes too
    short for the largest horizon are skipped rather than fatal -- a session
    can legitimately end early (death)."""
    written: Dict[str, str] = {}
    max_horizon = max(int(h) for h in horizons)
    for session_dir in session_dirs:
        for episode_id in list_episodes(session_dir):
            if len(load_episode_pixel_frames(session_dir, episode_id)) <= max_horizon:
                continue
            written[f"{session_dir}/{episode_id}"] = export_prediction_file(
                model, session_dir, episode_id, horizons
            )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export model pixel predictions per horizon for the pixel viewer."
    )
    parser.add_argument("--model", required=True, help="full-model bundle from save_full_visual_model")
    parser.add_argument("--session", required=True, action="append", help="session dir (repeatable)")
    parser.add_argument("--episode", default=None, help="episode id (default: every episode)")
    parser.add_argument("--horizons", default="1,10,100")
    args = parser.parse_args()

    model = load_full_visual_model(args.model)
    horizons = [int(h) for h in args.horizons.split(",")]
    for session_dir in args.session:
        episodes = [args.episode] if args.episode else list_episodes(session_dir)
        for episode_id in episodes:
            path = export_prediction_file(model, session_dir, episode_id, horizons)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
