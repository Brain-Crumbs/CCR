"""Export model pixel predictions per horizon for the pixel viewer.

The viewer (``viewer/server.js`` + ``<pixel-horizon-viewer>``) renders the
copy-last and mean-frame baselines straight from a recorded session; to also
show a *trained model's* predicted frames it needs a
``predictions_<episode>.json`` file next to the episode's stream log. This
module writes that file.

Legacy nursery runs called :func:`export_prediction_file` for every recorded
session because their checkpoints persisted only the pixel *encoder* -- the
decoder and next-latent predictor needed to roll predictions forward would
otherwise be lost when the process exited. To re-export later without
retraining, persist the whole visual-representation model with
:func:`save_full_visual_model` and use the CLI::

    python -m cognitive_runtime.training.prediction_export \
        --model walk_forward-full.pt \
        --session shared/nursery-walk_forward-train-0 --horizons 1,4,8

Output format ("pixel-predictions-v1")::

    {
      "format": "pixel-predictions-v1",
      "session_id": ..., "episode_id": ...,
      "horizons": [1, 4, 8],
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
from typing import Any, Dict, Sequence

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
) -> str:
    """Roll the model out from every start frame and write
    ``predictions_<episode>.json`` into ``session_dir``. Returns the path."""

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
    out_path = out_path or os.path.join(session_dir, f"predictions_{episode_id}.json")
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


def export_action_prediction_file(
    model: Any,
    session_dir: str,
    episode_id: str,
    horizons: Sequence[int],
    horizon_labels: Sequence[int] | None = None,
    out_path: str | None = None,
) -> str:
    """Export viewer predictions from an action-conditioned world model."""

    from cognitive_runtime.training.action_world_model import (
        _episode_tensors,
        build_action_sequence_dataset,
    )

    horizon_frames = [int(h) for h in horizons]
    if horizon_labels is None:
        labels = list(horizon_frames)
    else:
        labels = [int(h) for h in horizon_labels]
        if len(labels) != len(horizon_frames):
            raise ValueError("horizon_labels must have the same length as horizons")
    if not horizon_frames or any(h <= 0 for h in horizon_frames):
        raise ValueError("horizons must contain at least one positive offset")
    if any(label <= 0 for label in labels):
        raise ValueError("horizon labels must be positive")
    if len(set(labels)) != len(labels):
        raise ValueError(f"horizon labels must be unique, got {labels!r}")
    horizon_specs = list(zip(labels, horizon_frames))
    max_horizon = max(horizon_frames)

    dataset = build_action_sequence_dataset([session_dir], action_keys=model.action_keys)
    selected = [episode for episode in dataset.episodes if episode.episode_id == episode_id]
    if not selected:
        raise ValueError(f"{session_dir}/{episode_id}: no frame/action episode found")
    dataset.episodes = selected
    dataset.sources = [f"{session_dir}/{episode_id}"]

    action_index = {name: i for i, name in enumerate(model.action_keys)}
    episode, pixels, targets, _actions = _episode_tensors(dataset, model.reconstruction_shape)[0]
    if pixels.shape[0] <= max_horizon:
        raise ValueError(
            f"{session_dir}/{episode_id} has {pixels.shape[0]} frames, too short for horizon {max_horizon}"
        )
    playback_frame_count = episode.playback_frame_count or int(pixels.shape[0] - max_horizon)
    playback_frame_count = max(
        0, min(int(playback_frame_count), int(pixels.shape[0] - max_horizon))
    )
    if playback_frame_count <= 0:
        raise ValueError(
            f"{session_dir}/{episode_id} has no playback frames with horizon {max_horizon}"
        )
    remap = torch.tensor(
        [action_index[dataset.action_keys[a]] for a in episode.actions],
        dtype=torch.long,
    )

    was_training = model.training
    model.eval()
    predictions: Dict[str, Any] = {str(label): {"frames": []} for label, _h in horizon_specs}
    with torch.no_grad():
        latents = model.encoder(pixels)
        hiddens = [model.initial_state(1)]
        hidden = hiddens[0]
        for i in range(pixels.shape[0] - 1):
            _pred, hidden = model.step(latents[i : i + 1], remap[i : i + 1], hidden)
            hiddens.append(hidden)

        for t in range(playback_frame_count):
            available = min(max_horizon, pixels.shape[0] - 1 - t)
            if available <= 0:
                continue
            rolled, _hidden = model.rollout(
                latents[t : t + 1],
                remap[t : t + available].unsqueeze(0),
                hiddens[t],
            )
            for label, h in horizon_specs:
                if h <= available:
                    decoded = model.decoder(rolled[:, h - 1]).squeeze(0)
                    predictions[str(label)]["frames"].append(_b64_frame(decoded))
    if was_training:
        model.train()

    payload = {
        "format": "pixel-predictions-v1",
        "session_id": os.path.basename(os.path.normpath(session_dir)),
        "episode_id": episode_id,
        "horizons": labels,
        "horizon_frames": {str(label): int(h) for label, h in horizon_specs},
        "prediction_shape": list(model.reconstruction_shape),
        "n_frames": int(pixels.shape[0]),
        "playback_frame_count": int(playback_frame_count),
        "predictions": predictions,
        "targets": [_b64_frame(targets[i]) for i in range(pixels.shape[0])],
    }
    out_path = out_path or os.path.join(session_dir, f"predictions_{episode_id}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return out_path


def export_action_session_predictions(
    model: Any,
    session_dirs: Sequence[str],
    horizons: Sequence[int],
    horizon_labels: Sequence[int] | None = None,
) -> Dict[str, str]:
    """Export action-world-model predictions for every episode in sessions."""

    from cognitive_runtime.training.action_world_model import build_action_sequence_dataset

    written: Dict[str, str] = {}
    max_horizon = max(int(h) for h in horizons)
    for session_dir in session_dirs:
        dataset = build_action_sequence_dataset([session_dir], action_keys=model.action_keys)
        for episode in dataset.episodes:
            playback_frame_count = episode.playback_frame_count or len(episode.frames) - max_horizon
            if len(episode.frames) <= max_horizon or playback_frame_count <= 0:
                continue
            written[f"{session_dir}/{episode.episode_id}"] = export_action_prediction_file(
                model,
                session_dir,
                episode.episode_id,
                horizons,
                horizon_labels=horizon_labels,
            )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export model pixel predictions per horizon for the pixel viewer."
    )
    parser.add_argument("--model", required=True, help="full-model bundle from save_full_visual_model")
    parser.add_argument("--session", required=True, action="append", help="session dir (repeatable)")
    parser.add_argument("--episode", default=None, help="episode id (default: every episode)")
    parser.add_argument("--horizons", default="1,4,8")
    args = parser.parse_args()

    try:
        from cognitive_runtime.training.action_world_model import load_action_world_model

        model, _stats = load_action_world_model(args.model)
        export_file = export_action_prediction_file
    except ValueError:
        model = load_full_visual_model(args.model)
        export_file = export_prediction_file
    horizons = [int(h) for h in args.horizons.split(",")]
    for session_dir in args.session:
        episodes = [args.episode] if args.episode else list_episodes(session_dir)
        for episode_id in episodes:
            path = export_file(model, session_dir, episode_id, horizons)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
