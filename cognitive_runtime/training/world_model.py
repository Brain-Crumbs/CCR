"""Offline training for the action-conditioned neural world model (Phase D).

Implements the ``(fused_latent, action) -> (next_latent, reward, p_death,
risk, prediction_error)`` contract from ``cognitive_runtime/neural/world_model.py``
against recorded sessions:

- next-latent prediction: MSE against the realized next fused latent.
- reward prediction: MSE against the realized ``reward.scalar`` window total.
- death prediction: BCE-with-logits against ``event.died`` on the next tick.
- risk prediction: MSE against a binary "took damage or died" next-tick label.
- prediction error: MSE against the model's own realized next-latent error,
  a self-supervised curiosity/novelty signal (ICM-style), detached so it
  cannot shortcut the next-latent head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from cognitive_runtime.neural.checkpoint import FORMAT_VERSION, NeuralAgentCheckpoint
from cognitive_runtime.neural.world_model import MLPWorldModel
from cognitive_runtime.training.datasets import WorldModelDataset


@dataclass
class WorldModelTrainingConfig:
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 0
    hidden_dim: int = 128
    depth: int = 2
    dropout: float = 0.0
    next_latent_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    death_loss_weight: float = 1.0
    risk_loss_weight: float = 1.0
    prediction_error_loss_weight: float = 0.1


def train_world_model(
    dataset: WorldModelDataset,
    config: Optional[WorldModelTrainingConfig] = None,
) -> Tuple[MLPWorldModel, Dict[str, Any]]:
    """Train the action-conditioned world model on recorded transitions."""

    if len(dataset) == 0:
        raise ValueError("world-model dataset is empty; record sessions first")
    if dataset.layout_hash is None:
        raise ValueError("world-model dataset is missing a layout_hash")
    cfg = config or WorldModelTrainingConfig()
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    tensors = _dataset_tensors(dataset)
    fused_width = tensors["latents"].shape[1]
    model = MLPWorldModel(
        fused_width=fused_width,
        n_actions=len(dataset.action_keys),
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
        layout_hash=dataset.layout_hash,
        action_keys=dataset.action_keys,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    curves: Dict[str, list] = {
        "next_latent_loss": [],
        "reward_loss": [],
        "death_loss": [],
        "risk_loss": [],
        "prediction_error_loss": [],
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
        "deaths": float(dataset.death_count()),
        "epochs": float(cfg.epochs),
        "batch_size": float(cfg.batch_size),
        "lr": float(cfg.lr),
        "fused_width": float(fused_width),
        "layout_hash": dataset.layout_hash,
        "loss_curves": curves,
    }
    for key in curves:
        stats[f"initial_{key}"] = curves[key][0]
        stats[f"final_{key}"] = curves[key][-1]
        stats[f"{key}_decreased"] = bool(curves[key][-1] < curves[key][0])
    return model, stats


def death_prediction_auc(model: MLPWorldModel, dataset: WorldModelDataset) -> float:
    """AUC-style ranking score for ``p_death``: the probability a randomly
    chosen death-preceding tick scores higher than a randomly chosen
    non-death tick (1.0 = perfect ranking, 0.5 = random, ties count half).

    Raises if the dataset has no death-preceding ticks or no non-death
    ticks to rank against -- the score is undefined without both.
    """

    was_training = model.training
    model.eval()
    tensors = _dataset_tensors(dataset)
    with torch.no_grad():
        action_onehot = _action_onehot(tensors["actions"], model.n_actions)
        out = model(tensors["latents"], action_onehot)
        p_death = torch.sigmoid(out.terminal_logit)
    if was_training:
        model.train()

    dones = tensors["dones"]
    positive = p_death[dones >= 0.5]
    negative = p_death[dones < 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError(
            "death-ranking AUC needs both death-preceding and non-death ticks; "
            f"got {positive.numel()} positive, {negative.numel()} negative"
        )
    greater = (positive.unsqueeze(1) > negative.unsqueeze(0)).float().sum()
    ties = (positive.unsqueeze(1) == negative.unsqueeze(0)).float().sum()
    return float((greater + 0.5 * ties) / (positive.numel() * negative.numel()))


def save_world_model_checkpoint(
    path: str,
    model: MLPWorldModel,
    dataset: WorldModelDataset,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Save the trained world model in the unified checkpoint format."""

    manager = NeuralAgentCheckpoint(
        path,
        layout_hash=dataset.layout_hash or model.layout_hash,
        action_keys=list(dataset.action_keys),
        world_model=model,
        replay_metadata={
            "sources": list(dataset.sources),
            "representation": dataset.representation,
            "samples": len(dataset),
            "deaths": dataset.death_count(),
        },
        training_stats=stats,
        training_ticks=len(dataset),
        extra_metadata={
            "model_type": "world-model",
            "losses": [
                "next_latent_prediction",
                "reward_prediction",
                "death_prediction",
                "risk_prediction",
                "prediction_error",
            ],
        },
    )
    return manager.save(reason="world_model_training")


def load_world_model_checkpoint(
    path: str,
    *,
    expected_layout_hash: Optional[str] = None,
    expected_action_keys: Optional[list] = None,
    map_location: str | torch.device = "cpu",
) -> Tuple[MLPWorldModel, Dict[str, Any]]:
    """Load the world model from a unified checkpoint, validating layout."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older torch without weights_only
        payload = torch.load(path, map_location=map_location)
    if payload.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported neural checkpoint format {payload.get('format')!r}")
    metadata = payload.get("metadata", {})
    wm_meta = (
        metadata.get("modules", {})
        .get("world_model", {})
        .get("checkpoint_metadata", {})
    )
    if not wm_meta:
        raise ValueError("checkpoint is missing world_model checkpoint metadata")
    action_keys = wm_meta.get("action_keys") or list(metadata.get("action_keys", []))
    model = MLPWorldModel(
        fused_width=int(wm_meta["fused_width"]),
        n_actions=int(wm_meta["n_actions"]),
        hidden_dim=int(wm_meta["hidden_dim"]),
        depth=int(wm_meta["depth"]),
        dropout=float(wm_meta["dropout"]),
        layout_hash=wm_meta.get("layout_hash", metadata.get("layout_hash")),
        action_keys=action_keys,
    )
    manager = NeuralAgentCheckpoint(
        path,
        layout_hash=expected_layout_hash or model.layout_hash,
        action_keys=expected_action_keys or action_keys,
        world_model=model,
    )
    loaded = manager.load(
        expected_layout_hash=expected_layout_hash or model.layout_hash,
        expected_action_keys=expected_action_keys or action_keys,
        map_location=map_location,
    )
    model.eval()
    return model, loaded


def _dataset_tensors(dataset: WorldModelDataset) -> Dict[str, torch.Tensor]:
    return {
        "latents": torch.tensor(dataset.latents, dtype=torch.float32),
        "next_latents": torch.tensor(dataset.next_latents, dtype=torch.float32),
        "actions": torch.tensor(dataset.labels, dtype=torch.long),
        "rewards": torch.tensor(dataset.rewards, dtype=torch.float32),
        "dones": torch.tensor(dataset.dones, dtype=torch.float32),
        "risks": torch.tensor(dataset.risks, dtype=torch.float32),
    }


def _action_onehot(actions: torch.Tensor, n_actions: int) -> torch.Tensor:
    return F.one_hot(actions, num_classes=n_actions).float()


def _losses_for_batch(
    model: MLPWorldModel,
    tensors: Dict[str, torch.Tensor],
    batch: torch.Tensor,
    cfg: WorldModelTrainingConfig,
) -> Dict[str, torch.Tensor]:
    action_onehot = _action_onehot(tensors["actions"][batch], model.n_actions)
    out = model(tensors["latents"][batch], action_onehot)

    next_latent_target = tensors["next_latents"][batch]
    next_latent_loss = F.mse_loss(out.next_latent, next_latent_target)
    reward_loss = F.mse_loss(out.reward, tensors["rewards"][batch])
    death_loss = F.binary_cross_entropy_with_logits(out.terminal_logit, tensors["dones"][batch])
    risk_loss = F.mse_loss(torch.sigmoid(out.risk), tensors["risks"][batch])

    realized_error = (out.next_latent.detach() - next_latent_target).pow(2).mean(dim=1)
    prediction_error_loss = F.mse_loss(out.prediction_error, realized_error)

    total = (
        cfg.next_latent_loss_weight * next_latent_loss
        + cfg.reward_loss_weight * reward_loss
        + cfg.death_loss_weight * death_loss
        + cfg.risk_loss_weight * risk_loss
        + cfg.prediction_error_loss_weight * prediction_error_loss
    )
    return {
        "next_latent_loss": next_latent_loss,
        "reward_loss": reward_loss,
        "death_loss": death_loss,
        "risk_loss": risk_loss,
        "prediction_error_loss": prediction_error_loss,
        "total_loss": total,
    }


def _append_eval_losses(
    model: MLPWorldModel,
    tensors: Dict[str, torch.Tensor],
    curves: Dict[str, list],
    cfg: WorldModelTrainingConfig,
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
