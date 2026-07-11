"""Ego-motion canary benchmark (issue #39).

The single most important upcoming milestone named in
``docs/neural-stream-agent.md``: generate ``walk_forward`` episodes at
multiple world seeds (constant ``MOVE_FORWARD`` every tick, optional action
noise), train a next-frame predictor on a subset of seeds, and check that it
beats a copy-last-frame baseline and a mean-frame baseline on **held-out**
seeds -- evidence the model learned ego-motion/optical-flow regularities
instead of memorizing a map.

Scripted generation goes through the simulated Minecraft backend today;
remote (mineflayer) generation is a later extension this module does not
preclude (the only backend-specific code is in ``_record_walk_forward_episode``).
This lands as a standalone benchmark; the ``walk_forward`` scenario in the
nursery suite (issue #62) is expected to reuse the same recording helper.

Design: predict in latent space, decode to pixels as an auxiliary (the
project's stated Dreamer-style approach) -- reuses
``training.visual_representation.VisualRepresentationModel`` (encoder +
reconstruction decoder + single-step next-latent predictor) rather than
inventing a new pixel model, and reaches multi-horizon predictions by
*iterated rollout* of the single-step latent predictor (one of the two
horizon strategies the interface allows per docs/neural-stream-agent.md;
:class:`~cognitive_runtime.neural.world_model.MultiHorizonMLPWorldModel`
uses the other -- dedicated per-horizon heads -- on the fused agent-state
latent).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from cognitive_runtime.core.action import Action
from cognitive_runtime.neural.pixel_stream_encoder import pixels_to_chw
from cognitive_runtime.policies.constant_action import ConstantActionPolicy
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import list_episodes
from cognitive_runtime.training.datasets import (
    build_pixel_sequence_dataset,
    load_episode_pixel_frames,
)
from cognitive_runtime.training.visual_representation import (
    VisualPretrainingConfig,
    VisualRepresentationModel,
    reconstruction_target,
    save_pixel_encoder_pretraining_checkpoint,
    train_pixel_encoder_pretraining,
)

MOVE_FORWARD = Action("MOVE_FORWARD")


@dataclass
class EgoMotionCanaryConfig:
    train_seeds: Sequence[int] = (0, 1, 2, 3)
    holdout_seeds: Sequence[int] = (1000, 1001)
    episode_ticks: int = 120
    world_size: int = 24
    #: Probability each tick's action is replaced by a random action instead
    #: of MOVE_FORWARD -- issue #39's "optional action noise".
    action_noise: float = 0.0
    horizons: Sequence[int] = (1, 5, 20)
    latent_width: int = 32
    hidden_dim: int = 64
    reconstruction_size: int = 16
    epochs: int = 15
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 0
    max_train_samples: Optional[int] = None
    #: SSIM window side length (box filter; see ``_ssim``).
    ssim_window: int = 3
    #: Fine-tuning epochs for the horizon-consistency loss (see
    #: ``_train_horizon_consistency``): 0 skips it and evaluates the raw
    #: single-step ``next_predictor`` rolled out iteratively, which drifts
    #: badly past a few steps without this.
    consistency_epochs: int = 15
    consistency_lr: float = 1e-3


@dataclass
class EgoMotionCanaryReport:
    config: EgoMotionCanaryConfig
    train_sessions: List[str] = field(default_factory=list)
    holdout_sessions: List[str] = field(default_factory=list)
    pretraining_stats: Dict[str, Any] = field(default_factory=dict)
    #: Loss curves from the horizon-consistency fine-tune stage (empty if
    #: ``consistency_epochs == 0``).
    consistency_stats: Dict[str, List[float]] = field(default_factory=dict)
    #: Per horizon: model/copy-last/mean-frame PSNR + SSIM, sample count,
    #: and whether the model beat each baseline on *both* metrics.
    horizon_metrics: Dict[int, Dict[str, Any]] = field(default_factory=dict)


def run_ego_motion_canary(
    record_dir: str,
    config: Optional[EgoMotionCanaryConfig] = None,
) -> Tuple[VisualRepresentationModel, EgoMotionCanaryReport]:
    """Run the full canary: record train/holdout walk-forward episodes,
    pretrain a pixel encoder+decoder+next-latent predictor on the train
    seeds only, then evaluate multi-horizon next-frame prediction on the
    held-out seeds against copy-last-frame and mean-frame baselines.
    """

    cfg = config or EgoMotionCanaryConfig()
    if not cfg.horizons:
        raise ValueError("horizons must be non-empty")
    if any(h <= 0 for h in cfg.horizons):
        raise ValueError(f"horizons must be positive tick offsets, got {cfg.horizons!r}")
    if set(cfg.train_seeds) & set(cfg.holdout_seeds):
        raise ValueError("train_seeds and holdout_seeds must not overlap")

    train_sessions = [
        _record_walk_forward_episode(record_dir, f"walk-forward-train-{seed}", seed, cfg)
        for seed in cfg.train_seeds
    ]
    holdout_sessions = [
        _record_walk_forward_episode(record_dir, f"walk-forward-holdout-{seed}", seed, cfg)
        for seed in cfg.holdout_seeds
    ]

    train_dataset = build_pixel_sequence_dataset(train_sessions, max_samples=cfg.max_train_samples)
    if len(train_dataset) == 0:
        raise ValueError(
            "ego-motion canary: no adjacent pixel pairs in the walk-forward training "
            "sessions (episode_ticks too small?)"
        )

    visual_config = VisualPretrainingConfig(
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        latent_width=cfg.latent_width,
        hidden_dim=cfg.hidden_dim,
        reconstruction_size=cfg.reconstruction_size,
    )
    model, pretraining_stats = train_pixel_encoder_pretraining(train_dataset, visual_config)

    consistency_stats: Dict[str, List[float]] = {}
    if cfg.consistency_epochs > 0:
        consistency_stats = train_horizon_consistency(
            model,
            train_sessions,
            cfg.horizons,
            epochs=cfg.consistency_epochs,
            lr=cfg.consistency_lr,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
        )

    max_horizon = max(cfg.horizons)
    for session_dir in holdout_sessions:
        for episode_id in list_episodes(session_dir):
            if len(load_episode_pixel_frames(session_dir, episode_id)) <= max_horizon:
                raise ValueError(
                    f"{session_dir}/{episode_id} is too short for the largest horizon "
                    f"({max_horizon}); increase episode_ticks"
                )

    horizon_metrics = evaluate_ego_motion_holdout(
        model, holdout_sessions, cfg.horizons, ssim_window=cfg.ssim_window
    )
    report = EgoMotionCanaryReport(
        config=cfg,
        train_sessions=train_sessions,
        holdout_sessions=holdout_sessions,
        pretraining_stats=pretraining_stats,
        consistency_stats=consistency_stats,
        horizon_metrics=horizon_metrics,
    )
    return model, report


def evaluate_ego_motion_holdout(
    model: VisualRepresentationModel,
    holdout_sessions: Sequence[str],
    horizons: Sequence[int],
    *,
    ssim_window: int = 3,
) -> Dict[int, Dict[str, Any]]:
    """Per-horizon PSNR/SSIM for the model's iterated-rollout prediction vs.
    the copy-last-frame and mean-frame baselines, on already-recorded
    held-out episodes. The mean-frame baseline is the per-episode mean of
    the (downsampled) reconstruction targets, not a single global constant.
    """

    horizons_sorted = sorted(set(int(h) for h in horizons))
    max_horizon = horizons_sorted[-1]
    was_training = model.training
    model.eval()

    samples: Dict[int, Dict[str, List[float]]] = {
        h: {"model_mse": [], "copy_last_mse": [], "mean_frame_mse": [],
            "model_ssim": [], "copy_last_ssim": [], "mean_frame_ssim": []}
        for h in horizons_sorted
    }

    with torch.no_grad():
        for session_dir in holdout_sessions:
            for episode_id in list_episodes(session_dir):
                frames = load_episode_pixel_frames(session_dir, episode_id)
                if len(frames) <= max_horizon:
                    continue
                pixel_tensors = torch.stack([pixels_to_chw(f) for f in frames])
                targets = reconstruction_target(pixel_tensors, model.reconstruction_shape)
                mean_frame = targets.mean(dim=0)

                latents = model.encoder(pixel_tensors)
                for t in range(len(frames) - max_horizon):
                    rolled = latents[t : t + 1]
                    per_horizon_latent: Dict[int, torch.Tensor] = {}
                    for step in range(1, max_horizon + 1):
                        rolled = model.next_predictor(rolled)
                        if step in samples:
                            per_horizon_latent[step] = rolled
                    for h in horizons_sorted:
                        recon = model.decoder(per_horizon_latent[h]).squeeze(0)
                        target = targets[t + h]
                        copy_last = targets[t]

                        samples[h]["model_mse"].append(float(F.mse_loss(recon, target)))
                        samples[h]["copy_last_mse"].append(float(F.mse_loss(copy_last, target)))
                        samples[h]["mean_frame_mse"].append(float(F.mse_loss(mean_frame, target)))
                        samples[h]["model_ssim"].append(_ssim(recon, target, window=ssim_window))
                        samples[h]["copy_last_ssim"].append(_ssim(copy_last, target, window=ssim_window))
                        samples[h]["mean_frame_ssim"].append(_ssim(mean_frame, target, window=ssim_window))

    if was_training:
        model.train()

    report: Dict[int, Dict[str, Any]] = {}
    for h in horizons_sorted:
        entry = samples[h]
        if not entry["model_mse"]:
            raise ValueError(f"no held-out samples at horizon {h}; check holdout_sessions")
        model_mse = sum(entry["model_mse"]) / len(entry["model_mse"])
        copy_last_mse = sum(entry["copy_last_mse"]) / len(entry["copy_last_mse"])
        mean_frame_mse = sum(entry["mean_frame_mse"]) / len(entry["mean_frame_mse"])
        model_ssim = sum(entry["model_ssim"]) / len(entry["model_ssim"])
        copy_last_ssim = sum(entry["copy_last_ssim"]) / len(entry["copy_last_ssim"])
        mean_frame_ssim = sum(entry["mean_frame_ssim"]) / len(entry["mean_frame_ssim"])
        psnr_model = _psnr_from_mse(model_mse)
        psnr_copy_last = _psnr_from_mse(copy_last_mse)
        psnr_mean_frame = _psnr_from_mse(mean_frame_mse)
        report[h] = {
            "n_samples": len(entry["model_mse"]),
            "psnr_model": psnr_model,
            "psnr_copy_last": psnr_copy_last,
            "psnr_mean_frame": psnr_mean_frame,
            "ssim_model": model_ssim,
            "ssim_copy_last": copy_last_ssim,
            "ssim_mean_frame": mean_frame_ssim,
            "beats_copy_last": bool(psnr_model > psnr_copy_last and model_ssim > copy_last_ssim),
            "beats_mean_frame": bool(psnr_model > psnr_mean_frame and model_ssim > mean_frame_ssim),
        }
    return report


def save_ego_motion_canary_checkpoint(
    path: str,
    model: VisualRepresentationModel,
    report: EgoMotionCanaryReport,
) -> Dict[str, Any]:
    """Save the trained encoder in the unified checkpoint format, with the
    canary's holdout metrics folded into training stats so a checkpoint
    carries proof of the acceptance criterion it was trained against."""

    stats = dict(report.pretraining_stats)
    stats["ego_motion_canary"] = {
        "horizons": list(report.config.horizons),
        "train_sessions": report.train_sessions,
        "holdout_sessions": report.holdout_sessions,
        "horizon_metrics": report.horizon_metrics,
    }
    dataset_stub = _StubPixelDataset(
        layout_hash=None,
        sources=report.train_sessions + report.holdout_sessions,
        pixel_shape=model.pixel_shape,
    )
    return save_pixel_encoder_pretraining_checkpoint(path, model, dataset_stub, stats)


@dataclass
class _StubPixelDataset:
    """Just enough of ``PixelSequenceDataset`` for
    :func:`save_pixel_encoder_pretraining_checkpoint`'s metadata."""

    layout_hash: Optional[str]
    sources: List[str]
    pixel_shape: Tuple[int, int, int]
    representation: str = "ego_motion_canary"

    def __len__(self) -> int:
        return len(self.sources)


def train_horizon_consistency(
    model: VisualRepresentationModel,
    train_sessions: Sequence[str],
    horizons: Sequence[int],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> Dict[str, List[float]]:
    """Fine-tune encoder + ``next_predictor`` + decoder jointly against a
    horizon-consistency loss: decode the ``h``-step iterated latent rollout
    and match it to the actual frame at ``t + h``, for every configured
    horizon.

    ``VisualRepresentationModel.next_predictor`` is trained single-step only
    (predict ``t+1`` from ``t``); iterated ``h`` times with no other
    pressure it drifts off the manifold the decoder was trained on within a
    few steps. This is the "iterated rollout with a horizon-consistency
    loss" alternative docs/neural-stream-agent.md calls out as a valid
    reading of the multi-horizon interface, alongside dedicated per-horizon
    heads (used by ``MultiHorizonMLPWorldModel`` on the fused latent).

    Scenario-agnostic (only reads recorded pixel frames), so the nursery
    scenario suite (issue #62) reuses this directly for every scenario
    rather than re-deriving it per scenario.
    """

    horizons_sorted = sorted(set(int(h) for h in horizons))
    max_horizon = horizons_sorted[-1]
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    current_by_horizon: Dict[int, List[torch.Tensor]] = {h: [] for h in horizons_sorted}
    target_by_horizon: Dict[int, List[torch.Tensor]] = {h: [] for h in horizons_sorted}
    for session_dir in train_sessions:
        for episode_id in list_episodes(session_dir):
            frames = load_episode_pixel_frames(session_dir, episode_id)
            if len(frames) <= max_horizon:
                continue
            pixel_tensors = torch.stack([pixels_to_chw(f) for f in frames])
            targets = reconstruction_target(pixel_tensors, model.reconstruction_shape)
            n = pixel_tensors.shape[0]
            # Same (t range) for every horizon so per-horizon batches align 1:1.
            usable = n - max_horizon
            for h in horizons_sorted:
                current_by_horizon[h].append(pixel_tensors[:usable])
                target_by_horizon[h].append(targets[h : h + usable])

    for h in horizons_sorted:
        if not current_by_horizon[h]:
            raise ValueError(
                f"no training episodes long enough for horizon-consistency at horizon {h}"
            )
    current_by_horizon = {h: torch.cat(v, dim=0) for h, v in current_by_horizon.items()}
    target_by_horizon = {h: torch.cat(v, dim=0) for h, v in target_by_horizon.items()}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    curves: Dict[str, List[float]] = {f"h{h}_loss": [] for h in horizons_sorted}
    curves["total_loss"] = []
    n_samples = current_by_horizon[horizons_sorted[0]].shape[0]

    for _epoch in range(epochs):
        perm = torch.randperm(n_samples, generator=generator)
        epoch_loss = {h: 0.0 for h in horizons_sorted}
        seen = 0
        for start in range(0, n_samples, batch_size):
            batch = perm[start : start + batch_size]
            optimizer.zero_grad()
            batch_losses: Dict[int, torch.Tensor] = {}
            for h in horizons_sorted:
                latent = model.encoder(current_by_horizon[h][batch])
                for _ in range(h):
                    latent = model.next_predictor(latent)
                recon = model.decoder(latent)
                batch_losses[h] = F.mse_loss(recon, target_by_horizon[h][batch])
            total = sum(batch_losses.values()) / len(horizons_sorted)
            total.backward()
            optimizer.step()

            bs = int(batch.numel())
            seen += bs
            for h in horizons_sorted:
                epoch_loss[h] += float(batch_losses[h].detach()) * bs
        for h in horizons_sorted:
            curves[f"h{h}_loss"].append(round(epoch_loss[h] / max(seen, 1), 6))
        curves["total_loss"].append(
            round(sum(epoch_loss.values()) / max(seen, 1) / len(horizons_sorted), 6)
        )
    return curves


def _record_walk_forward_episode(
    record_dir: str, session_id: str, seed: int, cfg: EgoMotionCanaryConfig
) -> str:
    program_config = {"episode_ticks": cfg.episode_ticks, "world_size": cfg.world_size}
    policy = ConstantActionPolicy(
        MOVE_FORWARD,
        noise=cfg.action_noise,
        action_space=ACTION_SPACE if cfg.action_noise > 0 else None,
        seed=seed,
    )
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=seed,
        max_ticks_per_episode=cfg.episode_ticks,
        record_dir=record_dir,
        session_id=session_id,
        program_config=program_config,
        record_frames=True,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=program_config),
        policy=policy,
        config=runtime_config,
    ).run()
    return os.path.join(record_dir, session_id)


def _psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


def _ssim(a: torch.Tensor, b: torch.Tensor, *, window: int = 3) -> float:
    """Single-scale SSIM with a uniform (box) window: a lightweight
    approximation of the standard Gaussian-window SSIM that needs no
    scipy/skimage dependency, adequate at the small reconstruction
    resolutions this canary decodes to. ``a``, ``b``: ``Tensor[C, H, W]`` in
    ``[0, 1]``.
    """
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    h, w = a.shape[-2:]
    window = max(1, min(window, h, w))
    pool = lambda x: F.avg_pool2d(x.unsqueeze(0), window, stride=1)
    mu_a, mu_b = pool(a), pool(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = pool(a * a) - mu_a2
    sigma_b2 = pool(b * b) - mu_b2
    sigma_ab = pool(a * b) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    )
    return float(ssim_map.mean())
