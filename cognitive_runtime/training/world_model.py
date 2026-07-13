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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from cognitive_runtime.neural.checkpoint import FORMAT_VERSION, NeuralAgentCheckpoint
from cognitive_runtime.neural.world_model import MLPWorldModel, MultiHorizonMLPWorldModel
from cognitive_runtime.training.datasets import MultiHorizonWorldModelDataset, WorldModelDataset


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


# ---------------------------------------------------------------------------
# Multi-horizon, uncertainty-aware world model (issue #39)
# ---------------------------------------------------------------------------


@dataclass
class MultiHorizonWorldModelTrainingConfig:
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
    #: Floor added to the learned ``uncertainty`` before it divides the
    #: heteroscedastic next-latent loss, so an under-trained near-zero
    #: uncertainty head cannot blow the loss up early in training.
    uncertainty_eps: float = 1e-3


def train_multi_horizon_world_model(
    dataset: MultiHorizonWorldModelDataset,
    config: Optional[MultiHorizonWorldModelTrainingConfig] = None,
) -> Tuple[MultiHorizonMLPWorldModel, Dict[str, Any]]:
    """Train the multi-horizon, uncertainty-aware world model (issue #39) on
    recorded transitions.

    Each horizon's ``next_latent`` head is trained with a heteroscedastic
    Gaussian-style NLL against the learned ``uncertainty`` head (Nix &
    Weigend): ``0.5 * (mse / (uncertainty + eps) + log(uncertainty + eps))``.
    This gives the uncertainty head a real training signal -- it is rewarded
    for tracking realized error, not just for existing -- which is what makes
    the calibration sanity check in :func:`uncertainty_calibration`
    meaningful. ``reward``/``terminal``/``risk``/``prediction_error`` per
    horizon keep the same losses :func:`train_world_model` uses for ``t+1``.
    """

    if len(dataset) == 0:
        raise ValueError("multi-horizon world-model dataset is empty; record sessions first")
    if dataset.layout_hash is None:
        raise ValueError("multi-horizon world-model dataset is missing a layout_hash")
    cfg = config or MultiHorizonWorldModelTrainingConfig()
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    tensors = _multi_horizon_dataset_tensors(dataset)
    fused_width = tensors["latents"].shape[1]
    model = MultiHorizonMLPWorldModel(
        fused_width=fused_width,
        n_actions=len(dataset.action_keys),
        horizons=dataset.horizons,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
        layout_hash=dataset.layout_hash,
        action_keys=dataset.action_keys,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    curve_keys = ["total_loss"] + [
        f"h{h}_{name}"
        for h in dataset.horizons
        for name in (
            "next_latent_nll",
            "reward_loss",
            "death_loss",
            "risk_loss",
            "prediction_error_loss",
            "uncertainty_mean",
        )
    ]
    curves: Dict[str, list] = {key: [] for key in curve_keys}

    _append_multi_horizon_eval_losses(model, tensors, dataset.horizons, curves, cfg)
    model.train()
    n = len(dataset)
    for _epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, perm.numel(), cfg.batch_size):
            batch = perm[start : start + cfg.batch_size]
            optimizer.zero_grad()
            losses = _multi_horizon_losses_for_batch(model, tensors, dataset.horizons, batch, cfg)
            losses["total_loss"].backward()
            optimizer.step()
        _append_multi_horizon_eval_losses(model, tensors, dataset.horizons, curves, cfg)

    stats: Dict[str, Any] = {
        "samples": float(len(dataset)),
        "epochs": float(cfg.epochs),
        "batch_size": float(cfg.batch_size),
        "lr": float(cfg.lr),
        "fused_width": float(fused_width),
        "horizons": list(dataset.horizons),
        "layout_hash": dataset.layout_hash,
        "loss_curves": curves,
    }
    for key in curves:
        stats[f"initial_{key}"] = curves[key][0]
        stats[f"final_{key}"] = curves[key][-1]
    stats["baselines"] = multi_horizon_baseline_mse(dataset)
    stats["evaluation"] = evaluate_multi_horizon_model(model, dataset)
    return model, stats


def multi_horizon_baseline_mse(dataset: MultiHorizonWorldModelDataset) -> Dict[int, Dict[str, float]]:
    """Per-horizon copy-last-latent and mean-latent baseline MSEs (issue
    #39's held-out-baseline acceptance criterion). Data-only -- needs no
    model -- so a canary/dataset report can print baselines before training
    even starts.
    """

    tensors = _multi_horizon_dataset_tensors(dataset)
    report: Dict[int, Dict[str, float]] = {}
    latents = tensors["latents"]
    for h in dataset.horizons:
        target = tensors["future_latents"][h]
        copy_last_mse = F.mse_loss(latents, target)
        mean_latent = target.mean(dim=0, keepdim=True).expand_as(target)
        mean_latent_mse = F.mse_loss(mean_latent, target)
        report[h] = {
            "copy_last_mse": float(copy_last_mse),
            "mean_latent_mse": float(mean_latent_mse),
        }
    return report


def evaluate_multi_horizon_model(
    model: MultiHorizonMLPWorldModel, dataset: MultiHorizonWorldModelDataset
) -> Dict[int, Dict[str, float]]:
    """Per-horizon model MSE vs. the copy-last/mean baselines, plus an
    uncertainty-calibration sanity check (correlation between predicted
    uncertainty and realized squared error). Callers wanting a genuine
    held-out check should build ``dataset`` from sessions/seeds the model
    never trained on (see ``training/ego_motion_canary.py``).
    """

    was_training = model.training
    model.eval()
    tensors = _multi_horizon_dataset_tensors(dataset)
    baselines = multi_horizon_baseline_mse(dataset)
    report: Dict[int, Dict[str, float]] = {}
    with torch.no_grad():
        action_onehot = _action_onehot(tensors["actions"], model.n_actions)
        out = model.forward_horizons(tensors["latents"], action_onehot)
        for h in dataset.horizons:
            pred = out[h]
            target = tensors["future_latents"][h]
            sample_mse = (pred.next_latent - target).pow(2).mean(dim=1)
            model_mse = float(sample_mse.mean())
            correlation = _pearson_correlation(pred.uncertainty, sample_mse)
            report[h] = {
                "model_mse": model_mse,
                "copy_last_mse": baselines[h]["copy_last_mse"],
                "mean_latent_mse": baselines[h]["mean_latent_mse"],
                "beats_copy_last": bool(model_mse < baselines[h]["copy_last_mse"]),
                "beats_mean_latent": bool(model_mse < baselines[h]["mean_latent_mse"]),
                "uncertainty_error_correlation": correlation,
            }
    if was_training:
        model.train()
    return report


def uncertainty_calibration(
    model: MultiHorizonMLPWorldModel, dataset: MultiHorizonWorldModelDataset
) -> Dict[int, float]:
    """Per-horizon Pearson correlation between predicted ``uncertainty`` and
    realized next-latent squared error -- the calibration sanity check from
    issue #39's acceptance criteria ("uncertainty ... correlates with
    realized error on held-out data"). Positive and away from zero is the
    bar; this is not full calibration (e.g. no reliability diagram).
    """

    return {h: v["uncertainty_error_correlation"] for h, v in evaluate_multi_horizon_model(model, dataset).items()}


def save_multi_horizon_world_model_checkpoint(
    path: str,
    model: MultiHorizonMLPWorldModel,
    dataset: MultiHorizonWorldModelDataset,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Save the trained multi-horizon world model in the unified checkpoint
    format."""

    manager = NeuralAgentCheckpoint(
        path,
        layout_hash=dataset.layout_hash or model.layout_hash,
        action_keys=list(dataset.action_keys),
        world_model=model,
        replay_metadata={
            "sources": list(dataset.sources),
            "representation": dataset.representation,
            "samples": len(dataset),
            "horizons": list(dataset.horizons),
        },
        training_stats=stats,
        training_ticks=len(dataset),
        extra_metadata={
            "model_type": "multi-horizon-world-model",
            "losses": [
                "next_latent_heteroscedastic_nll",
                "reward_prediction",
                "death_prediction",
                "risk_prediction",
                "prediction_error",
            ],
        },
    )
    return manager.save(reason="multi_horizon_world_model_training")


def load_multi_horizon_world_model_checkpoint(
    path: str,
    *,
    expected_layout_hash: Optional[str] = None,
    expected_action_keys: Optional[list] = None,
    map_location: str | torch.device = "cpu",
) -> Tuple[MultiHorizonMLPWorldModel, Dict[str, Any]]:
    """Load the multi-horizon world model from a unified checkpoint,
    validating layout."""

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
    model = MultiHorizonMLPWorldModel(
        fused_width=int(wm_meta["fused_width"]),
        n_actions=int(wm_meta["n_actions"]),
        horizons=list(wm_meta.get("horizons", (1, 4, 8))),
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


def _multi_horizon_dataset_tensors(dataset: MultiHorizonWorldModelDataset) -> Dict[str, Any]:
    return {
        "latents": torch.tensor(dataset.latents, dtype=torch.float32),
        "actions": torch.tensor(dataset.labels, dtype=torch.long),
        "future_latents": {
            h: torch.tensor(dataset.future_latents[h], dtype=torch.float32) for h in dataset.horizons
        },
        "future_rewards": {
            h: torch.tensor(dataset.future_rewards[h], dtype=torch.float32) for h in dataset.horizons
        },
        "future_dones": {
            h: torch.tensor(dataset.future_dones[h], dtype=torch.float32) for h in dataset.horizons
        },
        "future_risks": {
            h: torch.tensor(dataset.future_risks[h], dtype=torch.float32) for h in dataset.horizons
        },
    }


def _multi_horizon_losses_for_batch(
    model: MultiHorizonMLPWorldModel,
    tensors: Dict[str, Any],
    horizons: List[int],
    batch: torch.Tensor,
    cfg: MultiHorizonWorldModelTrainingConfig,
) -> Dict[str, torch.Tensor]:
    action_onehot = _action_onehot(tensors["actions"][batch], model.n_actions)
    out = model.forward_horizons(tensors["latents"][batch], action_onehot)

    total = torch.zeros(())
    per_horizon: Dict[str, torch.Tensor] = {}
    for h in horizons:
        pred = out[h]
        target = tensors["future_latents"][h][batch]
        sample_mse = (pred.next_latent - target).pow(2).mean(dim=1)
        safe_uncertainty = pred.uncertainty + cfg.uncertainty_eps
        next_latent_nll = (
            0.5 * (sample_mse / safe_uncertainty + torch.log(safe_uncertainty))
        ).mean()

        reward_loss = F.mse_loss(pred.reward, tensors["future_rewards"][h][batch])
        death_loss = F.binary_cross_entropy_with_logits(
            pred.terminal_logit, tensors["future_dones"][h][batch]
        )
        risk_loss = F.mse_loss(torch.sigmoid(pred.risk), tensors["future_risks"][h][batch])

        realized_error = sample_mse.detach()
        prediction_error_loss = F.mse_loss(pred.prediction_error, realized_error)

        horizon_total = (
            cfg.next_latent_loss_weight * next_latent_nll
            + cfg.reward_loss_weight * reward_loss
            + cfg.death_loss_weight * death_loss
            + cfg.risk_loss_weight * risk_loss
            + cfg.prediction_error_loss_weight * prediction_error_loss
        )
        total = total + horizon_total
        per_horizon[f"h{h}_next_latent_nll"] = next_latent_nll
        per_horizon[f"h{h}_reward_loss"] = reward_loss
        per_horizon[f"h{h}_death_loss"] = death_loss
        per_horizon[f"h{h}_risk_loss"] = risk_loss
        per_horizon[f"h{h}_prediction_error_loss"] = prediction_error_loss
        per_horizon[f"h{h}_uncertainty_mean"] = pred.uncertainty.mean()

    per_horizon["total_loss"] = total / max(len(horizons), 1)
    return per_horizon


def _append_multi_horizon_eval_losses(
    model: MultiHorizonMLPWorldModel,
    tensors: Dict[str, Any],
    horizons: List[int],
    curves: Dict[str, list],
    cfg: MultiHorizonWorldModelTrainingConfig,
) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        batch = torch.arange(tensors["latents"].shape[0])
        losses = _multi_horizon_losses_for_batch(model, tensors, horizons, batch, cfg)
    for key, value in losses.items():
        curves[key].append(round(float(value.detach()), 6))
    if was_training:
        model.train()


def _pearson_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two 1-D tensors; ``0.0`` for
    degenerate inputs (fewer than 2 samples, or a constant series) rather
    than a NaN, since a scalar summary should not poison a stats dict."""

    if a.numel() < 2:
        return 0.0
    a = a.detach().float()
    b = b.detach().float()
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = torch.sqrt((a_centered.pow(2).sum()) * (b_centered.pow(2).sum()))
    if float(denom) == 0.0:
        return 0.0
    return float((a_centered * b_centered).sum() / denom)
