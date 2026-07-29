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
import dataclasses
import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from typing import Any, Dict, Literal, Mapping, Optional, Sequence

import torch

from cognitive_runtime.neural.pixel_stream_encoder import pixels_to_chw
from cognitive_runtime.runtime.replay import list_episodes
from cognitive_runtime.training.datasets import load_episode_pixel_frames
from cognitive_runtime.training.visual_representation import (
    VisualRepresentationModel,
    reconstruction_target,
)

FULL_MODEL_FORMAT = "visual-representation-full-v1"


@dataclasses.dataclass(frozen=True)
class ExperimentIdentity:
    """Stable identity shared by recordings, a cortex checkpoint and exports."""

    experiment_id: str
    organism: str
    trace_id: Optional[str]
    created_at: str
    git_commit: Optional[str]
    #: Version tag for the persisted identity manifest.  Kept on the
    #: dataclass itself so every writer of ``experiment.json`` shares the
    #: same schema, including the Model Factory artifact allocator.
    format: str = "experiment-identity-v1"

    @classmethod
    def create(cls, experiment_id: str, organism: str, *, trace_id: Optional[str] = None) -> "ExperimentIdentity":
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            ).strip() or None
        except (OSError, subprocess.CalledProcessError):
            git_commit = None
        return cls(
            experiment_id=experiment_id,
            organism=organism,
            trace_id=trace_id,
            created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            git_commit=git_commit,
        )


def experiment_directory(root: str, experiment: ExperimentIdentity, *, resume: bool = False) -> str:
    """Create or explicitly resume a validated experiment directory.

    ``resume=False`` preserves the fail-fast default for an accidental reused
    identifier. ``resume=True`` is safe only when this directory's identity
    manifest matches the requested organism and experiment id; it allows a
    notebook to add new runs without mixing two experiments' artifacts.
    """
    path = os.path.join(root, experiment.organism, experiment.experiment_id)
    manifest_path = os.path.join(path, "experiment.json")
    if not os.path.exists(path):
        os.makedirs(path)
        # The identity is the guard for every later resume.  A crash must not
        # leave a syntactically partial JSON document that could be mistaken
        # for a valid experiment, so stage it beside the final path and swap
        # only after the complete document is flushed.
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path,
                prefix=".experiment.json.", suffix=".tmp", delete=False,
            ) as handle:
                temporary_path = handle.name
                json.dump(dataclasses.asdict(experiment), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, manifest_path)
        except BaseException:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
            raise
        return path
    if not resume:
        raise FileExistsError(
            f"experiment directory already exists: {path}; pass resume=True to append "
            "only after identity validation"
        )
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            saved = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(
            f"cannot safely resume legacy experiment directory without {manifest_path!r}"
        ) from exc
    if saved.get("experiment_id") != experiment.experiment_id or saved.get("organism") != experiment.organism:
        raise ValueError("resume identity does not match the existing experiment directory")
    return path


def checkpoint_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    # Exports are JSON/NumPy artifacts, while a CUDA-trained cortex returns
    # CUDA tensors.  Crossing to host memory belongs at this serialization
    # boundary, not in the inference path.
    return base64.b64encode(hwc.detach().cpu().contiguous().numpy().tobytes()).decode("ascii")


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

    device = next(model.parameters()).device
    pixel_tensors = torch.stack([pixels_to_chw(f) for f in frames]).to(device)
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


def export_cortex_session_predictions(
    model: Any,
    dataset: Any,
    session_dir: str,
    episode_id: str,
    *,
    horizon_frames: Sequence[int],
    prediction_mode: Literal["direct", "rollout"],
    checkpoint_path: str,
    experiment: ExperimentIdentity,
    training_stats: Mapping[str, Any],
    out_path: Optional[str] = None,
) -> str:
    """Export predictions from an evaluated :class:`PredictiveCortex` as v2.

    ``dataset`` must be the same action/workspace dataset used for evaluation.
    This deliberately refuses an incomplete workspace token or an action not
    present in the checkpoint rather than silently exporting a pixel-only
    approximation of a joint model.
    """
    from cognitive_runtime.neural.pixel_stream_encoder import pixels_to_chw
    from cognitive_runtime.training.action_world_model import _episode_workspace_tensors
    from cognitive_runtime.training.event_evaluation import extract_frame_event_labels, labels_to_json
    from cognitive_runtime.training.visual_representation import reconstruction_target

    if prediction_mode not in {"direct", "rollout"}:
        raise ValueError("prediction_mode must be 'direct' or 'rollout'")
    horizons = sorted({int(h) for h in horizon_frames if int(h) > 0})
    if not horizons:
        raise ValueError("horizon_frames must contain at least one positive offset")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"joint checkpoint does not exist: {checkpoint_path}")
    if prediction_mode == "direct":
        unsupported = set(horizons) - set(model.horizons_ticks)
        if unsupported:
            raise ValueError(
                "direct export requires configured direct horizon heads; unsupported frame horizons: "
                f"{sorted(unsupported)}"
            )
    episode = next(
        (item for item in dataset.episodes
         if item.session_dir == session_dir and item.episode_id == episode_id),
        None,
    )
    if episode is None:
        raise ValueError(f"{session_dir}/{episode_id} is not present in the evaluation dataset")
    vocabulary = {key: index for index, key in enumerate(model.action_keys)}
    unknown = [key for key in dataset.action_keys if key not in vocabulary]
    if unknown:
        raise ValueError(f"action vocabulary mismatch; checkpoint lacks {unknown!r}")

    frames = episode.frames
    if len(frames) <= horizons[-1]:
        raise ValueError(f"{session_dir}/{episode_id} is too short for horizon {horizons[-1]}")
    device = next(model.parameters()).device
    pixels = torch.stack([pixels_to_chw(frame) for frame in frames]).to(device)
    targets = reconstruction_target(pixels, model.reconstruction_shape)
    actions = torch.tensor(
        [vocabulary[dataset.action_keys[index]] for index in episode.actions],
        dtype=torch.long, device=device,
    )
    # This performs the layout and per-modality checks for workspace-aware models.
    workspace = _episode_workspace_tensors(episode, dataset, model, actions=actions)
    action_names = [dataset.action_keys[index] for index in episode.actions]
    event_labels = extract_frame_event_labels(
        episode.semantic_grids, positions=episode.positions, actions=action_names,
    ) if len(episode.semantic_grids) == len(episode.frames) else []

    predictions: Dict[str, Any] = {str(h): {"frames": []} for h in horizons}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        latents = model.encode_workspace(pixels, workspace)
        if prediction_mode == "direct":
            hidden = model.forward_sequence(latents[:-1].unsqueeze(0), actions.unsqueeze(0))
            for h in horizons:
                direct = model.sequence_prediction(
                    hidden, h,
                    reference_frame=targets[:-1].unsqueeze(0),
                    reference_spatial=model.encode_visual(pixels[:-1]).spatial.unsqueeze(0),
                ).decoded[0]
                for t in range(len(frames) - h):
                    predictions[str(h)]["frames"].append(_b64_frame(direct[t]))
        else:
            state = model.initial_state(1)
            states = [state]
            for t in range(len(actions)):
                _predicted, state = model.step(latents[t:t + 1], actions[t:t + 1], state)
                states.append(state)
            for t in range(len(frames) - horizons[-1]):
                rollout = model.forward_horizons(
                    latents[t:t + 1], actions[t:t + horizons[-1]].unsqueeze(0), states[t],
                    horizon_frames=horizons,
                    reference_frame=targets[t:t + 1],
                    reference_spatial=model.encode_visual(pixels[t:t + 1]).spatial,
                )
                for h in horizons:
                    predictions[str(h)]["frames"].append(_b64_frame(rollout[h].decoded.squeeze(0)))
    if was_training:
        model.train()

    payload = {
        "format": "pixel-predictions-v2",
        "session_id": os.path.basename(os.path.normpath(session_dir)),
        "episode_id": episode_id,
        "experiment": dataclasses.asdict(experiment),
        "model": {
            "model_type": "predictive_cortex",
            "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
            "backbone": model.config.backbone,
            "training_objective": training_stats.get("training_objective"),
            "uses_actions": True,
            "uses_workspace": bool(getattr(model, "workspace_modalities", {})),
            "horizons_ticks": list(model.horizons_ticks),
            "reconstruction_shape": list(model.reconstruction_shape),
        },
        "training_sources": list(training_stats.get("training_sources", [])),
        "evaluation_source": f"{session_dir}/{episode_id}",
        "prediction_mode": prediction_mode,
        "events": labels_to_json(event_labels),
        "horizons": horizons,
        "prediction_shape": list(model.reconstruction_shape),
        "n_frames": len(frames),
        "predictions": predictions,
        "targets": [_b64_frame(target) for target in targets],
    }
    if out_path is None:
        out_path = os.path.join(session_dir, f"{experiment.experiment_id}-predictions_{episode_id}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return out_path


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
