"""Unified neural-agent checkpoint bundles.

The bundle format is intentionally split in two:

- ``<path>`` is a ``torch.save`` payload carrying tensors, optimizer state,
  RNG state and the same metadata copied into the sidecar.
- ``<path>.json`` is a plain JSON sidecar that dashboards can inspect without
  importing torch or deserializing tensors.

The runtime loop already calls ``learner.checkpoint(reason=...)`` during
shutdown/interruption.  ``NeuralAgentCheckpoint`` exposes that same hook, plus
``save()``, ``load(path)`` and ``resume()`` for concrete neural learners.
"""

from __future__ import annotations

import json
import os
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from cognitive_runtime.core.hashing import canonical_json

FORMAT_VERSION = "neural-agent-checkpoint-v1"
ACTION_SPACE_HASH_VERSION = "action-space-v1"
COMPATIBILITY_HASH_VERSION = "neural-agent-compat-v1"


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint belongs to a different layout/action space."""


def action_space_hash(action_keys: Sequence[str]) -> str:
    """Stable hash for the ordered action space a policy head was trained on."""

    import hashlib

    blob = canonical_json([ACTION_SPACE_HASH_VERSION, list(action_keys)])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def compatibility_hash(layout_hash: str, action_hash: str) -> str:
    """Stable hash for the compatibility-critical runtime layout pair."""

    import hashlib

    blob = canonical_json([COMPATIBILITY_HASH_VERSION, layout_hash, action_hash])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def checkpoint_metadata_path(path: str) -> str:
    """Sidecar path for JSON-inspectable checkpoint metadata."""

    return f"{path}.json"


def read_checkpoint_metadata(path: str) -> Dict[str, Any]:
    """Read the JSON sidecar without touching the tensor checkpoint."""

    with open(checkpoint_metadata_path(path), encoding="utf-8") as fh:
        return json.load(fh)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def _atomic_json_dump(path: str, payload: Mapping[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(payload), fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _atomic_torch_save(path: str, payload: Mapping[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(payload), tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _torch_load(path: str, map_location: Optional[str | torch.device] = None) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older torch without weights_only
        return torch.load(path, map_location=map_location)


def _module_metadata(module: nn.Module) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "class": f"{type(module).__module__}.{type(module).__name__}",
        "state_keys": sorted(module.state_dict().keys()),
    }
    checkpoint_metadata = getattr(module, "checkpoint_metadata", None)
    if callable(checkpoint_metadata):
        metadata["checkpoint_metadata"] = _json_safe(checkpoint_metadata())
    return metadata


def _optimizer_metadata(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    state = optimizer.state_dict()
    return {
        "class": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
        "param_groups": len(state.get("param_groups", [])),
        "state_entries": len(state.get("state", {})),
    }


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():  # pragma: no cover - CI is CPU-only
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])  # type: ignore[arg-type]
    if "numpy" in state:
        np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    if "torch" in state:
        torch.random.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():  # pragma: no cover
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class NeuralAgentCheckpoint:
    """Save/load/resume helper for a concrete neural stream agent.

    ``modules`` are grouped by the neural target contracts.  Encoder states are
    keyed by stream checkpoint keys (for example ``stream_encoder.body_health``);
    the other slots are the singleton learned fusion, world model, policy and
    critic modules.  ``optimizers`` carries ordinary ``torch.optim`` state, and
    ``online_optimizer`` can carry a higher-level object implementing the
    ``OnlineOptimizer`` contract from :mod:`cognitive_runtime.neural.optimizer`.
    """

    def __init__(
        self,
        path: str,
        *,
        layout_hash: str,
        action_keys: Sequence[str],
        encoders: Optional[Mapping[str, nn.Module]] = None,
        fusion: Optional[nn.Module] = None,
        world_model: Optional[nn.Module] = None,
        policy: Optional[nn.Module] = None,
        critic: Optional[nn.Module] = None,
        optimizers: Optional[Mapping[str, torch.optim.Optimizer]] = None,
        online_optimizer: Optional[Any] = None,
        replay_metadata: Optional[Mapping[str, Any]] = None,
        training_stats: Optional[Mapping[str, Any]] = None,
        training_ticks: int = 0,
        extra_metadata: Optional[Mapping[str, Any]] = None,
        capture_rng: bool = True,
    ) -> None:
        self.path = path
        self.layout_hash = layout_hash
        self.action_keys = list(action_keys)
        self.encoders = dict(encoders or {})
        self.fusion = fusion
        self.world_model = world_model
        self.policy = policy
        self.critic = critic
        self.optimizers = dict(optimizers or {})
        self.online_optimizer = online_optimizer
        self.replay_metadata = dict(replay_metadata or {})
        self.training_stats = dict(training_stats or {})
        self.training_ticks = int(training_ticks)
        self.extra_metadata = dict(extra_metadata or {})
        self.capture_rng = capture_rng

    @property
    def action_space_hash(self) -> str:
        return action_space_hash(self.action_keys)

    @property
    def compatibility_hash(self) -> str:
        return compatibility_hash(self.layout_hash, self.action_space_hash)

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.path,
            "format": FORMAT_VERSION,
            "layout_hash": self.layout_hash,
            "action_space_hash": self.action_space_hash,
            "compatibility_hash": self.compatibility_hash,
            "training_ticks": self.training_ticks,
            "replay_metadata": _json_safe(self.replay_metadata),
            "training_stats": _json_safe(self.training_stats),
        }

    def checkpoint(self, reason: str = "manual") -> None:
        self.save(reason=reason)

    def resume(self, **kwargs: Any) -> Dict[str, Any]:
        return self.load(self.path, **kwargs)

    def save(self, path: Optional[str] = None, *, reason: str = "manual") -> Dict[str, Any]:
        out = path or self.path
        metadata = self._metadata(reason=reason)
        payload = {
            "format": FORMAT_VERSION,
            "metadata": metadata,
            "state": self._state_payload(),
        }
        _atomic_torch_save(out, payload)
        _atomic_json_dump(checkpoint_metadata_path(out), metadata)
        return metadata

    def load(
        self,
        path: Optional[str] = None,
        *,
        expected_layout_hash: Optional[str] = None,
        expected_action_keys: Optional[Sequence[str]] = None,
        expected_action_space_hash: Optional[str] = None,
        map_location: Optional[str | torch.device] = None,
        strict: bool = True,
        restore_rng: bool = True,
    ) -> Dict[str, Any]:
        source = path or self.path
        payload = _torch_load(source, map_location=map_location)
        metadata = self._validate_payload_metadata(
            payload,
            expected_layout_hash=expected_layout_hash or self.layout_hash,
            expected_action_keys=expected_action_keys or self.action_keys,
            expected_action_space_hash=expected_action_space_hash,
        )
        state = payload.get("state")
        if not isinstance(state, Mapping):
            raise ValueError(f"neural checkpoint {source!r} is missing a state payload")
        self._load_state_payload(state, strict=strict)
        self.training_ticks = int(metadata.get("training_ticks", self.training_ticks))
        self.training_stats = dict(metadata.get("training_stats", self.training_stats))
        self.replay_metadata = dict(metadata.get("replay_metadata", self.replay_metadata))
        if restore_rng and isinstance(state.get("rng"), Mapping):
            _restore_rng_state(state["rng"])
        return metadata

    def _metadata(self, *, reason: str) -> Dict[str, Any]:
        modules: Dict[str, Any] = {
            "encoders": {
                key: _module_metadata(module) for key, module in sorted(self.encoders.items())
            }
        }
        for key, module in self._singleton_modules().items():
            if module is not None:
                modules[key] = _module_metadata(module)

        optimizer_state = {
            key: _optimizer_metadata(optimizer)
            for key, optimizer in sorted(self.optimizers.items())
        }
        online_optimizer_metadata = None
        if self.online_optimizer is not None:
            raw = self.online_optimizer.state_dict()
            online_optimizer_metadata = {
                "class": (
                    f"{type(self.online_optimizer).__module__}."
                    f"{type(self.online_optimizer).__name__}"
                ),
                "state_keys": sorted(raw.keys()) if isinstance(raw, Mapping) else [],
                "step": raw.get("step") if isinstance(raw, Mapping) else None,
            }

        rng_state = {
            "python": self.capture_rng,
            "numpy": self.capture_rng,
            "torch": self.capture_rng,
            "torch_cuda": bool(self.capture_rng and torch.cuda.is_available()),
        }
        metadata: Dict[str, Any] = {
            "format": FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "layout_hash": self.layout_hash,
            "action_keys": list(self.action_keys),
            "action_space_hash": self.action_space_hash,
            "compatibility_hash": self.compatibility_hash,
            "modules": modules,
            "optimizers": optimizer_state,
            "online_optimizer": online_optimizer_metadata,
            "training_ticks": self.training_ticks,
            "training_stats": _json_safe(self.training_stats),
            "replay_metadata": _json_safe(self.replay_metadata),
            "rng_state": rng_state,
            "extra": _json_safe(self.extra_metadata),
        }
        return metadata

    def _state_payload(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "encoders": {
                key: module.state_dict() for key, module in sorted(self.encoders.items())
            },
            "optimizers": {
                key: optimizer.state_dict()
                for key, optimizer in sorted(self.optimizers.items())
            },
        }
        for key, module in self._singleton_modules().items():
            if module is not None:
                state[key] = module.state_dict()
        if self.online_optimizer is not None:
            state["online_optimizer"] = self.online_optimizer.state_dict()
        if self.capture_rng:
            state["rng"] = _capture_rng_state()
        return state

    def _singleton_modules(self) -> Dict[str, Optional[nn.Module]]:
        return {
            "fusion": self.fusion,
            "world_model": self.world_model,
            "policy": self.policy,
            "critic": self.critic,
        }

    def _load_state_payload(self, state: Mapping[str, Any], *, strict: bool) -> None:
        self._load_named_modules("encoders", self.encoders, state.get("encoders", {}), strict=strict)
        for key, module in self._singleton_modules().items():
            if module is not None:
                if key not in state:
                    raise ValueError(f"neural checkpoint is missing {key!r} module state")
                module.load_state_dict(state[key], strict=strict)

        optimizer_state = state.get("optimizers", {})
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("neural checkpoint optimizers state must be a mapping")
        self._load_optimizers(optimizer_state)
        if self.online_optimizer is not None:
            if "online_optimizer" not in state:
                raise ValueError("neural checkpoint is missing online_optimizer state")
            self.online_optimizer.load_state_dict(state["online_optimizer"])

    @staticmethod
    def _load_named_modules(
        label: str,
        modules: Mapping[str, nn.Module],
        state: Any,
        *,
        strict: bool,
    ) -> None:
        if not isinstance(state, Mapping):
            raise ValueError(f"neural checkpoint {label} state must be a mapping")
        missing = sorted(set(modules) - set(state))
        if missing:
            raise ValueError(f"neural checkpoint missing {label} state for {missing}")
        for key, module in modules.items():
            module.load_state_dict(state[key], strict=strict)

    def _load_optimizers(self, state: Mapping[str, Any]) -> None:
        missing = sorted(set(self.optimizers) - set(state))
        if missing:
            raise ValueError(f"neural checkpoint missing optimizer state for {missing}")
        for key, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state[key])

    @staticmethod
    def _validate_payload_metadata(
        payload: Mapping[str, Any],
        *,
        expected_layout_hash: Optional[str],
        expected_action_keys: Optional[Sequence[str]],
        expected_action_space_hash: Optional[str],
    ) -> Dict[str, Any]:
        if payload.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported neural checkpoint format {payload.get('format')!r}; "
                f"expected {FORMAT_VERSION}"
            )
        metadata = payload.get("metadata")
        if not isinstance(metadata, MutableMapping):
            raise ValueError("neural checkpoint is missing metadata")
        if metadata.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported neural checkpoint metadata format "
                f"{metadata.get('format')!r}; expected {FORMAT_VERSION}"
            )

        got_layout = metadata.get("layout_hash")
        if expected_layout_hash is not None and got_layout != expected_layout_hash:
            raise CheckpointCompatibilityError(
                "neural checkpoint layout mismatch: checkpoint was trained on "
                f"layout_hash {got_layout!r}, but runtime produced "
                f"{expected_layout_hash!r}; rebuild the same stream catalog/"
                "encoders or train a new checkpoint for this layout"
            )

        got_action_keys = list(metadata.get("action_keys", []))
        expected_hash = expected_action_space_hash
        if expected_hash is None and expected_action_keys is not None:
            expected_hash = action_space_hash(expected_action_keys)
        got_hash = metadata.get("action_space_hash")
        if expected_hash is not None and got_hash != expected_hash:
            expected_keys = list(expected_action_keys) if expected_action_keys is not None else None
            raise CheckpointCompatibilityError(
                "neural checkpoint action-space mismatch: checkpoint has "
                f"hash {got_hash!r} action_keys={got_action_keys}, but runtime "
                f"expects hash {expected_hash!r} action_keys={expected_keys}; "
                "use the same ordered Program action space or train a new "
                "checkpoint"
            )
        return dict(metadata)

