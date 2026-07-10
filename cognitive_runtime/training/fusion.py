"""Offline training for learned latent fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from cognitive_runtime.neural.checkpoint import FORMAT_VERSION, NeuralAgentCheckpoint
from cognitive_runtime.neural.fusion import LatentFusionModel
from cognitive_runtime.training.datasets import LatentFusionDataset


class LatentFusionTrainingModel(nn.Module):
    """Fusion model plus auxiliary offline-training heads."""

    def __init__(
        self,
        fusion: LatentFusionModel,
        *,
        action_count: int,
        latent_width: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        fused_width = fusion.fused_width()
        self.action_head = nn.Sequential(
            nn.Linear(fused_width, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_count),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(fused_width, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.next_latent_head = nn.Sequential(
            nn.Linear(fused_width, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_width),
        )

    def forward(
        self,
        latents: torch.Tensor,
        presence_mask: torch.Tensor,
        recency: torch.Tensor,
        staleness: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fused = self.fusion(latents, presence_mask, recency, staleness)
        return {
            "fused": fused,
            "action_logits": self.action_head(fused),
            "reward": self.reward_head(fused).squeeze(-1),
            "next_latent": self.next_latent_head(fused),
        }


@dataclass
class FusionTrainingConfig:
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 0
    fused_width: Optional[int] = None
    hidden_dim: int = 128
    depth: int = 2
    dropout: float = 0.0
    action_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    next_latent_loss_weight: float = 1.0


def train_latent_fusion_model(
    dataset: LatentFusionDataset,
    config: Optional[FusionTrainingConfig] = None,
) -> Tuple[LatentFusionTrainingModel, Dict[str, Any]]:
    """Train learned fusion to predict action, reward, and next latent."""

    if len(dataset) == 0:
        raise ValueError("latent fusion dataset is empty; record sessions first")
    if dataset.layout_hash is None:
        raise ValueError("latent fusion dataset is missing a layout_hash")
    cfg = config or FusionTrainingConfig()
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    tensors = _dataset_tensors(dataset)
    latent_width = tensors["latents"].shape[1]
    fusion = LatentFusionModel(
        dataset.stream_slices,
        layout_hash=dataset.layout_hash,
        fused_width=cfg.fused_width or latent_width,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
    )
    model = LatentFusionTrainingModel(
        fusion,
        action_count=len(dataset.action_keys),
        latent_width=latent_width,
        hidden_dim=cfg.hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    curves: Dict[str, list] = {
        "action_loss": [],
        "reward_loss": [],
        "next_latent_loss": [],
        "total_loss": [],
    }

    _append_eval_losses(model, tensors, curves, cfg)
    model.train()
    n = len(dataset)
    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, perm.numel(), cfg.batch_size):
            batch = perm[start : start + cfg.batch_size]
            optimizer.zero_grad()
            losses = _losses_for_batch(model, tensors, batch, cfg)
            losses["total_loss"].backward()
            optimizer.step()
        _append_eval_losses(model, tensors, curves, cfg)

    stats: Dict[str, Any] = {
        "samples": float(len(dataset)),
        "epochs": float(cfg.epochs),
        "batch_size": float(cfg.batch_size),
        "lr": float(cfg.lr),
        "fused_width": float(fusion.fused_width()),
        "layout_hash": dataset.layout_hash,
        "stream_count": float(len(dataset.stream_ids)),
        "loss_curves": curves,
    }
    for key in ("action_loss", "reward_loss", "next_latent_loss", "total_loss"):
        stats[f"initial_{key}"] = curves[key][0]
        stats[f"final_{key}"] = curves[key][-1]
        stats[f"{key}_decreased"] = bool(curves[key][-1] < curves[key][0])
    return model, stats


def save_latent_fusion_checkpoint(
    path: str,
    model: LatentFusionTrainingModel,
    dataset: LatentFusionDataset,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Save learned fusion and auxiliary heads in the unified checkpoint format."""

    manager = NeuralAgentCheckpoint(
        path,
        layout_hash=dataset.layout_hash or model.fusion.layout_hash,
        action_keys=list(dataset.action_keys),
        fusion=model.fusion,
        policy=model.action_head,
        critic=model.reward_head,
        world_model=model.next_latent_head,
        replay_metadata={
            "sources": list(dataset.sources),
            "representation": dataset.representation,
            "stream_ids": list(dataset.stream_ids),
            "samples": len(dataset),
        },
        training_stats=stats,
        training_ticks=len(dataset),
        extra_metadata={
            "model_type": "latent-fusion",
            "losses": ["behavior_cloning", "reward_prediction", "next_latent_prediction"],
        },
    )
    return manager.save(reason="latent_fusion_training")


def load_latent_fusion_checkpoint(
    path: str,
    *,
    expected_layout_hash: Optional[str] = None,
    map_location: str | torch.device = "cpu",
) -> Tuple[LatentFusionModel, Dict[str, Any]]:
    """Load the fusion module from a unified checkpoint, validating layout."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older torch without weights_only
        payload = torch.load(path, map_location=map_location)
    if payload.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported neural checkpoint format {payload.get('format')!r}")
    metadata = payload.get("metadata", {})
    fusion_meta = (
        metadata.get("modules", {})
        .get("fusion", {})
        .get("checkpoint_metadata", {})
    )
    stream_slices = {
        key: tuple(value) for key, value in fusion_meta.get("stream_slices", {}).items()
    }
    if not stream_slices:
        raise ValueError("checkpoint is missing fusion stream_slices metadata")
    model = LatentFusionModel(
        stream_slices,
        layout_hash=fusion_meta.get("layout_hash", metadata.get("layout_hash")),
        fused_width=int(fusion_meta["fused_width"]),
        hidden_dim=int(fusion_meta["hidden_dim"]),
        depth=int(fusion_meta["depth"]),
        dropout=float(fusion_meta["dropout"]),
    )
    manager = NeuralAgentCheckpoint(
        path,
        layout_hash=expected_layout_hash or model.layout_hash,
        action_keys=list(metadata.get("action_keys", [])),
        fusion=model,
    )
    loaded = manager.load(
        expected_layout_hash=expected_layout_hash or model.layout_hash,
        map_location=map_location,
    )
    model.eval()
    return model, loaded


def _dataset_tensors(dataset: LatentFusionDataset) -> Dict[str, torch.Tensor]:
    return {
        "latents": torch.tensor(dataset.latents, dtype=torch.float32),
        "presence": torch.tensor(dataset.presence_masks, dtype=torch.float32),
        "recency": torch.tensor(dataset.recency, dtype=torch.float32),
        "staleness": torch.tensor(dataset.staleness, dtype=torch.float32),
        "labels": torch.tensor(dataset.labels, dtype=torch.long),
        "rewards": torch.tensor(dataset.rewards, dtype=torch.float32),
        "next_latents": torch.tensor(dataset.next_latents, dtype=torch.float32),
    }


def _losses_for_batch(
    model: LatentFusionTrainingModel,
    tensors: Dict[str, torch.Tensor],
    batch: torch.Tensor,
    cfg: FusionTrainingConfig,
) -> Dict[str, torch.Tensor]:
    out = model(
        tensors["latents"][batch],
        tensors["presence"][batch],
        tensors["recency"][batch],
        tensors["staleness"][batch],
    )
    action_loss = F.cross_entropy(out["action_logits"], tensors["labels"][batch])
    reward_loss = F.mse_loss(out["reward"], tensors["rewards"][batch])
    next_loss = F.mse_loss(out["next_latent"], tensors["next_latents"][batch])
    total = (
        cfg.action_loss_weight * action_loss
        + cfg.reward_loss_weight * reward_loss
        + cfg.next_latent_loss_weight * next_loss
    )
    return {
        "action_loss": action_loss,
        "reward_loss": reward_loss,
        "next_latent_loss": next_loss,
        "total_loss": total,
    }


def _append_eval_losses(
    model: LatentFusionTrainingModel,
    tensors: Dict[str, torch.Tensor],
    curves: Dict[str, list],
    cfg: FusionTrainingConfig,
) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        batch = torch.arange(tensors["latents"].shape[0])
        losses = _losses_for_batch(model, tensors, batch, cfg)
    for key, value in losses.items():
        curves[key].append(round(float(value.detach()), 6))
    if was_training:
        model.train()
