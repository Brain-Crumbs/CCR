"""Versioned, resumable checkpoints for Model Factory runs.

The older :mod:`action_world_model` checkpoint is intentionally retained for
existing callers.  This module is the factory-facing format: it records the
optimizer, scheduler, trainer progress, and random-number-generator state
needed to continue an interrupted run exactly where it stopped.
"""

from __future__ import annotations

import random
import os
import tempfile
import json
import hashlib
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from cognitive_runtime.training.model_factory.contracts import contract_hash


FORMAT = "action-world-model-factory-v1"
HEADER_FORMAT = "model-factory-checkpoint-header-v1"
_LEGACY_V2 = "action-world-model-v2"
_LEGACY_V1 = "action-world-model-v1"

_CONTINUATION_MODES = frozenset(("resume", "clone", "fine_tune"))


@dataclass(frozen=True)
class ContinuationMismatch:
    """One incompatible contract field, suitable for human diagnostics."""

    field: str
    parent: Any
    requested: Any


@dataclass(frozen=True)
class ContinuationDecision:
    """The header-only result of a requested checkpoint continuation."""

    approved: bool
    mode: str
    lineage_mode: Optional[str]
    mismatches: Tuple[ContinuationMismatch, ...] = ()
    message: str = ""

    @property
    def rejection(self) -> Optional[Dict[str, Any]]:
        """Structured rejection data, or ``None`` when loading is approved."""

        if self.approved:
            return None
        return {
            "mode": self.mode,
            "message": self.message,
            "mismatches": [
                {"field": item.field, "parent": item.parent, "requested": item.requested}
                for item in self.mismatches
            ],
        }


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
    continuation: Optional[ContinuationDecision] = None

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


def _jsonable(value: Any) -> Any:
    """Make comparison/header values JSON-shaped without importing torch."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def factory_checkpoint_metadata_path(path: str) -> str:
    """Return the JSON compatibility header path for a factory checkpoint."""
    return f"{path}.json"


def read_factory_checkpoint_metadata(path: str) -> Dict[str, Any]:
    """Read the compatibility header without deserializing tensors."""
    with open(factory_checkpoint_metadata_path(path), encoding="utf-8") as handle:
        header = json.load(handle)
    if not isinstance(header, dict) or header.get("format") != HEADER_FORMAT:
        raise ValueError(f"invalid factory checkpoint compatibility header for {path!r}")
    return header


def _atomic_json_dump(path: str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _checkpoint_sha256(path: str) -> str:
    """Hash raw checkpoint bytes without deserializing tensors."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _data_contract_hash(contract_or_hash: Any) -> str:
    value = getattr(contract_or_hash, "hash", contract_or_hash)
    if not isinstance(value, str):
        raise TypeError("data_contract_hash must be a SHA-256 string or DataContract")
    return value


def _normalise_architecture(contract: Any) -> Dict[str, Any]:
    """Canonical header representation for an architecture contract."""
    return _jsonable(_architecture_payload(contract))


def _normalise_training(contract: Any) -> Dict[str, Any]:
    return _jsonable(_contract_payload(contract))


def _contract_value(contract: Mapping[str, Any], name: str) -> Any:
    """Accept the public ``*_contract`` names and concise internal aliases."""
    aliases = {
        "architecture_contract": ("architecture_contract", "architecture"),
        "data_contract_hash": ("data_contract_hash", "data"),
        "training_contract": ("training_contract", "training"),
    }
    for key in aliases[name]:
        if key in contract:
            return contract[key]
    return None


def _normalise_continuation_contract(contract: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise TypeError("continuation contracts must be mappings")
    architecture = _contract_value(contract, "architecture_contract")
    data = _contract_value(contract, "data_contract_hash")
    training = _contract_value(contract, "training_contract")
    return {
        "architecture_contract": _normalise_architecture(architecture) if architecture is not None else None,
        "data_contract_hash": _data_contract_hash(data) if data is not None else None,
        "training_contract": _normalise_training(training) if training is not None else None,
    }


def _field_differences(parent: Any, requested: Any, prefix: str) -> Tuple[ContinuationMismatch, ...]:
    """Return stable, leaf-level differences for nested JSON-shaped contracts."""
    if isinstance(parent, Mapping) and isinstance(requested, Mapping):
        differences = []
        for key in sorted(set(parent) | set(requested)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in parent:
                differences.append(ContinuationMismatch(child, "<missing>", requested[key]))
            elif key not in requested:
                differences.append(ContinuationMismatch(child, parent[key], "<missing>"))
            else:
                differences.extend(_field_differences(parent[key], requested[key], child))
        return tuple(differences)
    if parent != requested:
        return (ContinuationMismatch(prefix, parent, requested),)
    return ()


def _render_rejection(mode: str, category: str, mismatches: Sequence[ContinuationMismatch]) -> str:
    lines = [f"cannot {mode}: {category} contract differs"]
    lines.extend(
        f"  {item.field}: parent={item.parent!r} requested={item.requested!r}"
        for item in mismatches
    )
    if mode == "resume":
        lines.append("  hint: use mode=fine_tune to continue these weights under a new training contract")
    elif mode == "clone" and category == "data":
        lines.append("  hint: use mode=fine_tune to branch these weights onto different data")
    return "\n".join(lines)


def validate_continuation(
    parent_contract: Mapping[str, Any],
    requested_contract: Mapping[str, Any],
    mode: str,
) -> ContinuationDecision:
    """Approve or reject a continuation using contracts only, never tensors.

    ``resume`` must preserve every contract.  ``clone`` preserves model and
    data identity but intentionally starts a new training procedure.
    ``fine_tune`` preserves the architecture/input contract while allowing a
    new data and training contract; it is always represented as a distinct
    fine-tune lineage branch.
    """
    if mode not in _CONTINUATION_MODES:
        raise ValueError(f"unsupported continuation mode {mode!r}")
    parent = _normalise_continuation_contract(parent_contract)
    requested = _normalise_continuation_contract(requested_contract)
    required = {
        "resume": ("architecture_contract", "data_contract_hash", "training_contract"),
        "clone": ("architecture_contract", "data_contract_hash"),
        "fine_tune": ("architecture_contract",),
    }[mode]
    category_labels = {
        "architecture_contract": "architecture",
        "data_contract_hash": "data",
        "training_contract": "training",
    }
    for name in required:
        if requested[name] is None:
            mismatch = ContinuationMismatch(name, parent[name], "<missing>")
            return ContinuationDecision(
                False, mode, None, (mismatch,),
                f"cannot {mode}: requested {category_labels[name]} contract is required",
            )
        if parent[name] is None:
            mismatch = ContinuationMismatch(name, "<missing>", requested[name])
            return ContinuationDecision(
                False, mode, None, (mismatch,),
                f"cannot {mode}: checkpoint is missing its {category_labels[name]} contract",
            )
        parent_value = parent[name]
        requested_value = requested[name]
        # ``hash`` is a derived summary of the architecture fields, not the
        # operator-actionable source of incompatibility.  Compare it only via
        # those fields so diagnostics name (for example) backbone_kwargs.
        if name == "architecture_contract":
            parent_value = dict(parent_value)
            requested_value = dict(requested_value)
            parent_value.pop("hash", None)
            requested_value.pop("hash", None)
        differences = _field_differences(
            parent_value, requested_value, category_labels[name],
        )
        if differences:
            return ContinuationDecision(
                False, mode, None, differences,
                _render_rejection(mode, category_labels[name], differences),
            )
    return ContinuationDecision(True, mode, mode, (), f"{mode} continuation approved")


def _checkpoint_header(payload: Mapping[str, Any], *, checkpoint_sha256: Optional[str] = None) -> Dict[str, Any]:
    header = {
        "format": HEADER_FORMAT,
        "checkpoint_format": FORMAT,
        "checkpoint_id": payload["checkpoint_id"],
        "architecture_contract": _normalise_architecture(payload["architecture_contract"]),
        "data_contract_hash": payload["data_contract_hash"],
        "training_contract": _normalise_training(payload["training_contract"]),
        "parent_checkpoint_sha": payload.get("parent_checkpoint_sha"),
    }
    if checkpoint_sha256 is not None:
        header["checkpoint_sha256"] = checkpoint_sha256
    return header


def _verify_header_binding(header: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    """Ensure the already-validated header describes this loaded payload."""
    if header.get("checkpoint_format") != payload.get("format"):
        raise ValueError("factory checkpoint payload does not match its compatibility header format")
    if header.get("checkpoint_id") != payload.get("checkpoint_id"):
        raise ValueError(
            "factory checkpoint payload does not match its compatibility header identity; "
            "retry after the checkpoint writer has finished"
        )
    payload_header = _checkpoint_header(payload)
    for field in ("architecture_contract", "data_contract_hash", "training_contract"):
        if header.get(field) != payload_header[field]:
            raise ValueError(
                "factory checkpoint payload contracts do not match its compatibility header; "
                "retry after the checkpoint writer has finished"
            )


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
    data_hash = _data_contract_hash(data_contract_hash)
    payload = {
        "format": FORMAT,
        # The sidecar repeats this generation ID.  A loader verifies it after
        # deserializing the payload so an interrupted overwrite cannot pair
        # new weights with an older approved header.
        "checkpoint_id": uuid.uuid4().hex,
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
    destination = Path(path)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
        # This small, tensor-free header is the continuation guard.  It lets
        # callers reject a requested resume/clone/fine-tune before torch.load
        # can deserialize weights or a model constructor can reserve a device.
        _atomic_json_dump(
            factory_checkpoint_metadata_path(path),
            _checkpoint_header(payload, checkpoint_sha256=_checkpoint_sha256(path)),
        )
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise
    return payload


def load_factory_checkpoint(
    path: str,
    *,
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    resume: bool = False,
    mode: Optional[str] = None,
    map_location: str = "cpu",
    architecture_contract: Any = None,
    data_contract_hash: Any = None,
    training_contract: Any = None,
) -> FactoryCheckpoint:
    """Load a factory checkpoint, or inspect/clone a legacy AWM checkpoint.

    Explicit ``mode`` values validate the JSON compatibility header before
    ``torch.load`` or model construction: ``resume`` requires all contracts,
    ``clone`` requires architecture and data, and ``fine_tune`` requires the
    architecture/input contract while recording a new training branch.

    ``resume=True`` remains a compatibility spelling for ``mode='resume'``;
    calls with neither option retain the pre-MF-B2 inspect/clone behavior.
    A legacy v2 checkpoint has no resumable state, so it can only be inspected
    or cloned and receives an explicit remediation error for resume attempts.
    """
    if mode is not None and mode not in _CONTINUATION_MODES:
        raise ValueError(f"unsupported continuation mode {mode!r}")
    if mode is not None and resume and mode != "resume":
        raise ValueError("resume=True conflicts with continuation mode " + repr(mode))
    requested_mode = mode or ("resume" if resume else None)
    continuation: Optional[ContinuationDecision] = None
    header: Optional[Dict[str, Any]] = None
    if requested_mode is not None:
        try:
            header = read_factory_checkpoint_metadata(path)
        except FileNotFoundError:
            if mode is not None:
                raise ValueError(
                    "cannot safely validate continuation before loading weights: "
                    f"factory checkpoint compatibility header is missing for {path!r}"
                ) from None
            # Legacy ``resume=True`` callers retain the historical error
            # below after format inspection.  New explicit modes are always
            # header-first and therefore never touch tensors on rejection.
        else:
            expected_digest = header.get("checkpoint_sha256")
            if not isinstance(expected_digest, str) or not header.get("checkpoint_id"):
                if mode is not None:
                    raise ValueError(
                        "cannot safely validate continuation before loading weights: "
                        "factory checkpoint compatibility header lacks payload binding"
                    )
            elif _checkpoint_sha256(path) != expected_digest:
                raise ValueError(
                    "factory checkpoint bytes do not match their compatibility header; "
                    "retry after the checkpoint writer has finished"
                )
            continuation = validate_continuation(
                header,
                {
                    "architecture_contract": architecture_contract,
                    "data_contract_hash": data_contract_hash,
                    "training_contract": training_contract,
                },
                requested_mode,
            )
            if not continuation.approved:
                raise ValueError(continuation.message)

    torch = _torch()
    payload = torch.load(path, map_location=map_location, weights_only=False)
    format_name = payload.get("format")
    if format_name in {_LEGACY_V1, _LEGACY_V2}:
        if requested_mode == "resume":
            raise ValueError(
                f"{format_name} checkpoints cannot be resumed: they do not contain "
                "optimizer, scheduler, or RNG state; load it for inspection or clone."
            )
        from cognitive_runtime.training.action_world_model import load_action_world_model

        legacy_model, stats = load_action_world_model(path, inspection_only=format_name == _LEGACY_V1)
        return FactoryCheckpoint(
            payload={**payload, "training_stats": dict(stats), "legacy": True},
            model=legacy_model, continuation=continuation,
        )
    if format_name != FORMAT:
        raise ValueError(f"unsupported factory checkpoint format {format_name!r}")
    if continuation is not None:
        # Hashing the file before torch.load catches an already-stale sidecar.
        # This second check closes the overwrite race between that hash and
        # torch.load, before model/optimizer/RNG state can be restored.
        assert header is not None
        _verify_header_binding(header, payload)
        payload_decision = validate_continuation(
            payload,
            {
                "architecture_contract": architecture_contract,
                "data_contract_hash": data_contract_hash,
                "training_contract": training_contract,
            },
            requested_mode,
        )
        if not payload_decision.approved:
            raise ValueError(payload_decision.message)
        continuation = payload_decision
    architecture = dict(payload["architecture_contract"])
    stored_hash = architecture.pop("hash", None)
    if stored_hash != contract_hash(architecture):
        raise ValueError("factory checkpoint architecture_contract hash is invalid")
    if requested_mode == "resume" and continuation is None:
        missing = [
            name for name, value in (
                ("architecture_contract", architecture_contract),
                ("data_contract_hash", data_contract_hash),
                ("training_contract", training_contract),
            ) if value is None
        ]
        if missing:
            raise ValueError(
                "resume requires expected " + ", ".join(missing)
            )
        expected_architecture = _architecture_payload(architecture_contract)
        if expected_architecture != payload["architecture_contract"]:
            raise ValueError("cannot resume: architecture contract does not match checkpoint")
        if _data_contract_hash(data_contract_hash) != payload["data_contract_hash"]:
            raise ValueError("cannot resume: data contract does not match checkpoint")
        if _contract_payload(training_contract) != payload["training_contract"]:
            raise ValueError("cannot resume: training contract does not match checkpoint")
    restored_model = model or _build_model(payload["model_definition"])
    _load_model_state(restored_model, payload["model_state_dict"])
    restored_model.training_objective = payload["training_contract"].get(
        "objective", "windowed_rollout"
    )
    if requested_mode == "resume":
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
        scheduler=scheduler, resumed=requested_mode == "resume", continuation=continuation,
    )


def _load_model_state(model: Any, state_dict: Mapping[str, Any]) -> None:
    """Load strictly, except for the two intentional historical migrations."""
    incompatibility = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        key for key in model.state_dict()
        if key.startswith("multi_token_heads.")
    }
    allowed_unexpected = (
        {"transition_backbone.position_embedding.weight"}
        if getattr(model.config, "backbone", None) == "transformer"
        else set()
    )
    missing = set(incompatibility.missing_keys) - allowed_missing
    unexpected = set(incompatibility.unexpected_keys) - allowed_unexpected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing keys: {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected keys: {sorted(unexpected)}")
        raise ValueError("incompatible factory checkpoint (" + "; ".join(details) + ")")
