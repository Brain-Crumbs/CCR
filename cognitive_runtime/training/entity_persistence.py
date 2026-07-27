"""Offline training for the entity-persistence model (Phase D, issue #27).

Builds ``(last_feature, gap_ticks) -> feature_at_reappearance`` samples from
recorded sessions: every time a tracked entity (``core.entity_tracker
.EntityTracker``) goes occluded and then reappears, the reappearance tick's
realized feature is a free training label for what the model should have
predicted during the gap.

``vision.entities`` is delta-published (only on change), so the *current*
payload for a tick is whatever last arrived, not necessarily this tick's
sensory records -- the same latest-value semantics
``core.streams.temporal_buffer.TemporalBuffer`` gives the online loop.  A
naive per-tick scan would misread "unchanged, so not republished" as
"occluded", falsely inflating gaps whenever a visible mob holds still.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from cognitive_runtime.core.entity_features import (
    ENTITY_FEATURE_WIDTH,
    NEUTRAL_ENTITY_FEATURE,
    VISION_ENTITIES_STREAM,
)
from cognitive_runtime.core.entity_tracker import EntityTracker
from cognitive_runtime.core.streams.temporal_buffer import TemporalBuffer
from cognitive_runtime.neural.checkpoint import FORMAT_VERSION, NeuralAgentCheckpoint
from cognitive_runtime.neural.entity_persistence import (
    DEFAULT_GAP_CAP_TICKS,
    ENTITY_PERSISTENCE_CHECKPOINT_KEY,
    EntityPersistenceModel,
    normalize_gap,
)
from cognitive_runtime.runtime.recorder import stream_event_from_log
from cognitive_runtime.runtime.replay import (
    iter_cognitive_ticks,
    list_episodes,
    load_session_metadata,
    require_streams_v2,
)


@dataclass
class EntityPersistenceDataset:
    """One sample per occlusion-then-reappearance a tracked mob went through."""

    last_features: List[List[float]] = field(default_factory=list)
    gaps: List[float] = field(default_factory=list)
    target_features: List[List[float]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    feature_width: int = ENTITY_FEATURE_WIDTH

    def __len__(self) -> int:
        return len(self.gaps)

    def baseline_mse(self) -> float:
        """"Forget immediately" baseline: always predict the neutral (far,
        no-bearing) feature instead of tracking anything -- the comparison
        acceptance criterion #1 asks the trained model to beat."""
        if not self.target_features:
            raise ValueError("dataset is empty; record occlusion sessions first")
        total = 0.0
        for target in self.target_features:
            total += sum(
                (t - b) ** 2 for t, b in zip(target, NEUTRAL_ENTITY_FEATURE)
            )
        return total / (len(self.target_features) * len(NEUTRAL_ENTITY_FEATURE))


def build_entity_persistence_dataset(
    session_dirs: List[str],
    max_samples: Optional[int] = None,
    max_gap_ticks: int = 200,
) -> EntityPersistenceDataset:
    """Walk recorded sessions and emit persistence samples from every
    occlusion-then-reappearance the tracked mobs went through."""

    dataset = EntityPersistenceDataset()
    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            raise FileNotFoundError(f"session directory not found: {session_dir}")
        metadata = load_session_metadata(session_dir)
        require_streams_v2(metadata)
        for episode_id in list_episodes(session_dir):
            buffer = TemporalBuffer()
            tracker = EntityTracker(max_gap_ticks=max_gap_ticks)
            episode_samples = 0
            for _decision, sensory, _motor in iter_cognitive_ticks(session_dir, episode_id):
                for record in sensory:
                    if record.get("elided") or record.get("stream_id") != VISION_ENTITIES_STREAM:
                        continue
                    buffer.append(stream_event_from_log(record))
                latest = buffer.latest(VISION_ENTITIES_STREAM)
                entities: List[Dict[str, Any]] = (
                    latest.payload if latest is not None and isinstance(latest.payload, list) else []
                )
                for reappearance in tracker.update(entities):
                    dataset.last_features.append(reappearance.last_feature)
                    dataset.gaps.append(float(reappearance.gap_ticks))
                    dataset.target_features.append(reappearance.feature_now)
                    episode_samples += 1
                    if max_samples is not None and len(dataset) >= max_samples:
                        dataset.sources.append(f"{session_dir}/{episode_id} (truncated)")
                        return dataset
            if episode_samples:
                dataset.sources.append(f"{session_dir}/{episode_id}")
    return dataset


@dataclass
class EntityPersistenceTrainingConfig:
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 0
    hidden_dim: int = 32
    depth: int = 2
    dropout: float = 0.0
    gap_cap: float = DEFAULT_GAP_CAP_TICKS
    feature_loss_weight: float = 1.0
    surprise_loss_weight: float = 0.1


def train_entity_persistence_model(
    dataset: EntityPersistenceDataset,
    config: Optional[EntityPersistenceTrainingConfig] = None,
) -> Tuple[EntityPersistenceModel, Dict[str, Any]]:
    """Train the persistence model on recorded occlusion/reappearance pairs."""

    if len(dataset) == 0:
        raise ValueError(
            "entity-persistence dataset is empty; record sessions where a "
            "tracked mob is occluded and reappears (e.g. walks behind a block)"
        )
    cfg = config or EntityPersistenceTrainingConfig()
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    model = EntityPersistenceModel(
        feature_width=dataset.feature_width,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
        gap_cap=cfg.gap_cap,
    )
    tensors = _dataset_tensors(dataset, cfg.gap_cap)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    curves: Dict[str, list] = {
        "feature_loss": [],
        "surprise_loss": [],
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

    model.eval()
    with torch.no_grad():
        out = model(tensors["last_features"], tensors["gaps_norm"])
        model_mse = float(F.mse_loss(out.predicted_feature, tensors["targets"]))
    baseline_mse = dataset.baseline_mse()

    stats: Dict[str, Any] = {
        "samples": float(len(dataset)),
        "epochs": float(cfg.epochs),
        "batch_size": float(cfg.batch_size),
        "lr": float(cfg.lr),
        "feature_width": float(dataset.feature_width),
        "loss_curves": curves,
        "baseline_mse": baseline_mse,
        "model_mse": model_mse,
        "beats_forget_baseline": bool(model_mse < baseline_mse),
    }
    for key in curves:
        stats[f"initial_{key}"] = curves[key][0]
        stats[f"final_{key}"] = curves[key][-1]
        stats[f"{key}_decreased"] = bool(curves[key][-1] < curves[key][0])
    return model, stats


#: The persistence model depends on neither the fused stream layout nor the
#: action space (it only ever sees entity-tracker gap features), so -- like
#: ``training.visual_representation.save_pixel_encoder_pretraining_checkpoint``
#: -- it checkpoints under a fixed synthetic layout/action-space pair rather
#: than requiring the caller to supply the runtime's real ones.
ENTITY_PERSISTENCE_LAYOUT_HASH = "entity-persistence-layout-v1"
ENTITY_PERSISTENCE_ACTION_KEYS = ["noop"]


def save_entity_persistence_checkpoint(
    path: str,
    model: EntityPersistenceModel,
    dataset: EntityPersistenceDataset,
    stats: Dict[str, Any],
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Save the trained persistence model in the unified checkpoint format.

    There is no dedicated singleton slot for this module in
    ``NeuralAgentCheckpoint`` (fusion/world_model/policy/critic), so it rides
    the generic ``encoders`` mapping under ``ENTITY_PERSISTENCE_CHECKPOINT_KEY``
    -- the same slot shape ``stream_encoder.*`` modules use, just for a
    module that consumes tracked-entity gaps rather than a raw stream.
    """

    manager = NeuralAgentCheckpoint(
        path,
        layout_hash=ENTITY_PERSISTENCE_LAYOUT_HASH,
        action_keys=ENTITY_PERSISTENCE_ACTION_KEYS,
        encoders={ENTITY_PERSISTENCE_CHECKPOINT_KEY: model},
        replay_metadata={
            "sources": list(dataset.sources),
            "samples": len(dataset),
        },
        training_stats=stats,
        training_ticks=len(dataset),
        extra_metadata={
            "model_type": "entity-persistence",
            "losses": ["persistence_feature_prediction", "self_supervised_surprise"],
        },
        name=name,
    )
    return manager.save(reason="entity_persistence_training")


def load_entity_persistence_checkpoint(
    path: str,
    *,
    expected_layout_hash: Optional[str] = None,
    expected_action_keys: Optional[List[str]] = None,
    map_location: str | torch.device = "cpu",
) -> Tuple[EntityPersistenceModel, Dict[str, Any]]:
    """Load the persistence model from a unified checkpoint."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older torch without weights_only
        payload = torch.load(path, map_location=map_location)
    if payload.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported neural checkpoint format {payload.get('format')!r}")
    metadata = payload.get("metadata", {})
    module_meta = (
        metadata.get("modules", {})
        .get("encoders", {})
        .get(ENTITY_PERSISTENCE_CHECKPOINT_KEY, {})
        .get("checkpoint_metadata", {})
    )
    if not module_meta:
        raise ValueError("checkpoint is missing entity_persistence checkpoint metadata")
    model = EntityPersistenceModel(
        feature_width=int(module_meta["feature_width"]),
        hidden_dim=int(module_meta["hidden_dim"]),
        depth=int(module_meta["depth"]),
        dropout=float(module_meta["dropout"]),
        gap_cap=float(module_meta["gap_cap"]),
    )
    layout_hash = expected_layout_hash or metadata.get("layout_hash")
    action_keys = expected_action_keys or list(metadata.get("action_keys", []))
    manager = NeuralAgentCheckpoint(
        path,
        layout_hash=layout_hash,
        action_keys=action_keys,
        encoders={ENTITY_PERSISTENCE_CHECKPOINT_KEY: model},
    )
    loaded = manager.load(
        expected_layout_hash=layout_hash,
        expected_action_keys=action_keys,
        map_location=map_location,
    )
    model.eval()
    return model, loaded


def _dataset_tensors(
    dataset: EntityPersistenceDataset, gap_cap: float
) -> Dict[str, torch.Tensor]:
    return {
        "last_features": torch.tensor(dataset.last_features, dtype=torch.float32),
        "gaps_norm": torch.tensor(
            [normalize_gap(g, gap_cap) for g in dataset.gaps], dtype=torch.float32
        ),
        "targets": torch.tensor(dataset.target_features, dtype=torch.float32),
    }


def _losses_for_batch(
    model: EntityPersistenceModel,
    tensors: Dict[str, torch.Tensor],
    batch: torch.Tensor,
    cfg: EntityPersistenceTrainingConfig,
) -> Dict[str, torch.Tensor]:
    out = model(tensors["last_features"][batch], tensors["gaps_norm"][batch])
    target = tensors["targets"][batch]
    feature_loss = F.mse_loss(out.predicted_feature, target)

    realized_error = (out.predicted_feature.detach() - target).pow(2).mean(dim=1)
    surprise_loss = F.mse_loss(out.surprise, realized_error)

    total = cfg.feature_loss_weight * feature_loss + cfg.surprise_loss_weight * surprise_loss
    return {
        "feature_loss": feature_loss,
        "surprise_loss": surprise_loss,
        "total_loss": total,
    }


def _append_eval_losses(
    model: EntityPersistenceModel,
    tensors: Dict[str, torch.Tensor],
    curves: Dict[str, list],
    cfg: EntityPersistenceTrainingConfig,
) -> None:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        batch = torch.arange(tensors["last_features"].shape[0])
        losses = _losses_for_batch(model, tensors, batch, cfg)
    for key, value in losses.items():
        curves[key].append(round(float(value.detach()), 6))
    if was_training:
        model.train()
