"""Versioned, resumable checkpoints for Model Factory runs.

The older :mod:`action_world_model` checkpoint is intentionally retained for
existing callers.  This module is the factory-facing format: it records the
optimizer, scheduler, trainer progress, and random-number-generator state
needed to continue an interrupted run exactly where it stopped.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Mapping, Optional

from cognitive_runtime.training.model_factory.contracts import contract_hash


FORMAT = "action-world-model-factory-v1"
_LEGACY_V2 = "action-world-model-v2"
_LEGACY_V1 = "action-world-model-v1"


@dataclass
class FactoryCheckpoint:
    """A loaded checkpoint and, when requested, restored runtime objects.

    ``payload`` remains available for inspection and supports mapping-style
    access so callers do not lose provenance fields that newer formats add.
    """

    payload: Dict[str, Any]
    model: Any = None
    optimizer: Any = None
    scheduler: Any = None
    resumed: bool = False

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    @property
    def trainer_state(self) -> Dict[str, Any]:
        return self.payload["trainer_state"]

    @property
    def training_stats(self) -> Dict[str, Any]:
        return self.payload["training_stats"]


def _torch():
    import torch

    return torch


def _plain(value: Any) -> Any:
    """Convert frozen contract dataclasses and mappings to saveable values."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif is_dataclass(value):
        value = asdict(value)
    return _saveable(value)


def _saveable(value: Any) -> Any:
    """Detach MappingProxyType values used by frozen contracts recursively."""
    if isinstance(value, Mapping):
        return {key: _saveable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_saveable(item) for item in value)
    if isinstance(value, list):
        return [_saveable(item) for item in value]
    return value


def _architecture_payload(contract: Any) -> Dict[str, Any]:
    body = _plain(contract)
    embedded_hash = body.pop("hash", None)
    computed = contract_hash(body)
    if embedded_hash is not None and embedded_hash != computed:
        raise ValueError("architecture_contract hash does not match its fields")
    # Keep the fields at this level (easy to inspect) and include the exact
    # canonical hash produced by MF-A1's ArchitectureContract.
    return {"hash": computed, **body}


def _contract_payload(contract: Any) -> Dict[str, Any]:
    return _plain(contract)


def capture_rng_state() -> Dict[str, Any]:
    """Capture Python, NumPy, CPU torch, and all CUDA torch RNG streams."""
    torch = _torch()
    try:
        import numpy as np
    except ImportError:  # NumPy is optional in the core package.
        numpy_state = None
    else:
        numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng_state: Mapping[str, Any]) -> None:
    """Restore the RNG streams captured by :func:`capture_rng_state`."""
    torch = _torch()
    random.setstate(rng_state["python"])
    if rng_state.get("numpy") is not None:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("checkpoint includes NumPy RNG state but NumPy is unavailable") from exc
        np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    cuda_state = rng_state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint includes CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)


def _model_definition(model: Any) -> Dict[str, Any]:
    """Persist every constructor setting required to rebuild the cortex."""
    cfg = model.config
    visual_defaults = {
        "scene_height": None,
        "hud_loss_weight": 0.25,
        "semantic_loss_weight": 1.0,
        "semantic_classes": int(cfg.semantic_classes),
        "change_mask_sparsity_weight": float(cfg.change_mask_sparsity_weight),
        "change_mask_supervision_weight": 0.25,
        "motion_pixel_loss_weight": 4.0,
        "change_mask_threshold": 0.02,
        "direct_horizon_loss_weight": 1.0,
        "closed_loop_pixel_loss_weight": 0.25,
        "closed_loop_latent_loss_weight": 0.25,
    }
    visual_defaults.update(getattr(model, "training_visual_config", {}))
    return {
        "pixel_shape": list(model.pixel_shape),
        "action_keys": list(model.action_keys),
        "latent_width": int(model.latent_width),
        "hidden_dim": int(model.hidden_dim),
        "action_embed_dim": int(cfg.action_embed_dim),
        "reconstruction_size": int(cfg.reconstruction_size),
        "reconstruction_shape": list(model.reconstruction_shape),
        "visual_architecture": str(model.visual_architecture),
        "horizons_ticks": list(model.horizons_ticks),
        "backbone": str(cfg.backbone),
        "context_length": int(cfg.context_length),
        "backbone_kwargs": dict(cfg.backbone_kwargs),
        "workspace_modalities": dict(model.workspace_modalities),
        "workspace_layout_hash": model.workspace_layout_hash,
        "visual_settings": visual_defaults,
    }


def _build_model(definition: Mapping[str, Any]) -> Any:
    from cognitive_runtime.training.action_world_model import (
        ActionWorldModelConfig,
        build_action_world_model,
    )

    visual = definition.get("visual_settings", {})
    cfg = ActionWorldModelConfig(
        latent_width=int(definition["latent_width"]),
        hidden_dim=int(definition["hidden_dim"]),
        action_embed_dim=int(definition["action_embed_dim"]),
        reconstruction_size=int(definition["reconstruction_size"]),
        visual_architecture=str(definition["visual_architecture"]),
        semantic_classes=int(visual.get("semantic_classes", 0)),
        change_mask_sparsity_weight=float(visual.get("change_mask_sparsity_weight", 0.01)),
        horizons_ticks=tuple(definition["horizons_ticks"]),
        backbone=str(definition["backbone"]),
        context_length=int(definition["context_length"]),
        backbone_kwargs=dict(definition.get("backbone_kwargs", {})),
    )
    model = build_action_world_model(
        tuple(definition["pixel_shape"]), definition["action_keys"], cfg,
        workspace_modalities=dict(definition.get("workspace_modalities", {})),
        workspace_layout_hash=definition.get("workspace_layout_hash"),
    )
    model.training_visual_config = dict(visual)
    return model


def save_factory_checkpoint(
    path: str,
    model: Any,
    optimizer: Any,
    scheduler: Any = None,
    trainer_state: Optional[Mapping[str, Any]] = None,
    *,
    architecture_contract: Any,
    data_contract_hash: Any,
    training_contract: Any,
    parent_checkpoint_sha: Optional[str] = None,
    training_stats: Optional[Mapping[str, Any]] = None,
    rng_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Save a factory checkpoint that can be resumed without state loss."""
    torch = _torch()
    progress = dict(trainer_state or {})
    for field in ("epoch", "global_step", "best_validation_metric"):
        progress.setdefault(field, None)
    data_hash = getattr(data_contract_hash, "hash", data_contract_hash)
    if not isinstance(data_hash, str):
        raise TypeError("data_contract_hash must be a SHA-256 string or DataContract")
    payload = {
        "format": FORMAT,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "trainer_state": progress,
        "rng_state": dict(rng_state) if rng_state is not None else capture_rng_state(),
        "architecture_contract": _architecture_payload(architecture_contract),
        "data_contract_hash": data_hash,
        "training_contract": _contract_payload(training_contract),
        "parent_checkpoint_sha": parent_checkpoint_sha,
        "training_stats": dict(training_stats or {}),
        # Explicit reconstruction metadata; visual settings include every
        # resolved visual knob rather than relying on implicit defaults.
        "model_definition": _model_definition(model),
    }
    torch.save(payload, path)
    return payload


def load_factory_checkpoint(
    path: str,
    *,
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    resume: bool = False,
    map_location: str = "cpu",
) -> FactoryCheckpoint:
    """Load a factory checkpoint, or inspect/clone a legacy AWM checkpoint.

    Passing ``resume=True`` restores optimizer, scheduler and RNG state.  A
    legacy v2 checkpoint has no such state, so it can only be inspected or
    cloned and receives an explicit remediation error for resume attempts.
    """
    torch = _torch()
    payload = torch.load(path, map_location=map_location, weights_only=False)
    format_name = payload.get("format")
    if format_name in {_LEGACY_V1, _LEGACY_V2}:
        if resume:
            raise ValueError(
                f"{format_name} checkpoints cannot be resumed: they do not contain "
                "optimizer, scheduler, or RNG state; load it for inspection or clone."
            )
        from cognitive_runtime.training.action_world_model import load_action_world_model

        legacy_model, stats = load_action_world_model(path, inspection_only=format_name == _LEGACY_V1)
        return FactoryCheckpoint(
            payload={**payload, "training_stats": dict(stats), "legacy": True},
            model=legacy_model,
        )
    if format_name != FORMAT:
        raise ValueError(f"unsupported factory checkpoint format {format_name!r}")
    architecture = dict(payload["architecture_contract"])
    stored_hash = architecture.pop("hash", None)
    if stored_hash != contract_hash(architecture):
        raise ValueError("factory checkpoint architecture_contract hash is invalid")
    restored_model = model or _build_model(payload["model_definition"])
    restored_model.load_state_dict(payload["model_state_dict"])
    if resume:
        if optimizer is None:
            raise ValueError("resume requires an optimizer instance to restore")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if payload["scheduler_state_dict"] is not None:
            if scheduler is None:
                raise ValueError("resume requires a scheduler instance to restore")
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        restore_rng_state(payload["rng_state"])
    return FactoryCheckpoint(
        payload=payload, model=restored_model, optimizer=optimizer,
        scheduler=scheduler, resumed=resume,
    )
