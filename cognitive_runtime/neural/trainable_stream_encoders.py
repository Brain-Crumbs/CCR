"""Trainable non-visual stream encoders.

These modules are the Phase B/C counterparts to the fixed scalar/entity
encoders used by ``TemporalFusion`` today.  They keep the same stream-facing
``encode`` contract while producing neural latents that can be checkpointed
with the unified neural-agent bundle.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch import nn

from cognitive_runtime.core.streams.encoders.common import normalize, scalar_leaf, spec_range
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec
from cognitive_runtime.models.online_q import motor_history_features_for_actions
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
from cognitive_runtime.neural.encoder import StreamEncoderModule

DEFAULT_ACTION_KEYS: List[str] = [action.key() for action in ACTION_SPACE]

MOTOR_HISTORY_STREAM_ID = "motor.history"
MOTOR_HISTORY_CHECKPOINT_KEY = "stream_encoder.motor_history"
BODY_STATE_CHECKPOINT_KEY = "stream_encoder.body_state"
REWARD_CHECKPOINT_KEY = "stream_encoder.reward"
ENTITY_CHECKPOINT_KEY = "stream_encoder.entities"
AUDIO_STREAM_PATTERN = "audio.*"
AUDIO_CHECKPOINT_KEY = "stream_encoder.audio"


def _mlp(input_width: int, hidden_width: int, latent_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_width, hidden_width),
        nn.ReLU(),
        nn.Linear(hidden_width, latent_width),
        nn.ReLU(),
    )


def _numeric_leaves(payload: Any) -> List[float]:
    if isinstance(payload, bool):
        return [1.0 if payload else 0.0]
    if isinstance(payload, (int, float)):
        return [float(payload)]
    if isinstance(payload, Mapping):
        out: List[float] = []
        for key in sorted(payload):
            out.extend(_numeric_leaves(payload[key]))
        return out
    if isinstance(payload, (list, tuple)):
        out = []
        for item in payload:
            out.extend(_numeric_leaves(item))
        return out
    return []


def _string_bucket(value: str, buckets: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % buckets


def _latest_event(events: Sequence[StreamEvent], expected_prefix: Optional[str] = None) -> StreamEvent:
    if not events:
        raise ValueError("expected at least one stream event")
    latest = events[-1]
    if expected_prefix is not None and not latest.stream_id.startswith(expected_prefix):
        raise ValueError(
            f"expected stream id starting with {expected_prefix!r}, got {latest.stream_id!r}"
        )
    return latest


def _action_key_from_payload(payload: Any) -> Optional[str]:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        for key in ("action_key", "key", "action", "command"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        nested = payload.get("payload")
        if nested is not None:
            return _action_key_from_payload(nested)
    return None


def _action_keys_from_payload(payload: Any) -> List[str]:
    if isinstance(payload, Mapping):
        for key in ("recent_action_keys", "actions", "keys", "history"):
            value = payload.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [item for item in (_action_key_from_payload(v) for v in value) if item]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return [item for item in (_action_key_from_payload(v) for v in payload) if item]
    one = _action_key_from_payload(payload)
    return [one] if one else []


class MotorHistoryEncoder(StreamEncoderModule):
    """Encode recent motor/action keys into a motor latent.

    ``parity_mode=True`` emits the exact one-hot vector used by
    ``training.features.motor_history_features`` and ``OnlineQModel``.  The
    trainable mode embeds the last ``window_size`` keys, flattens those
    embeddings, and maps them through a small MLP.
    """

    stream_id = MOTOR_HISTORY_STREAM_ID
    checkpoint_key = MOTOR_HISTORY_CHECKPOINT_KEY

    def __init__(
        self,
        action_keys: Optional[Sequence[str]] = None,
        *,
        window_size: int = 8,
        embedding_width: int = 8,
        latent_width: int = 16,
        parity_mode: bool = False,
    ) -> None:
        super().__init__()
        self.action_keys = list(action_keys or DEFAULT_ACTION_KEYS)
        if not self.action_keys:
            raise ValueError("MotorHistoryEncoder requires at least one action key")
        self.window_size = int(window_size)
        self.embedding_width = int(embedding_width)
        self.latent_width = len(self.action_keys) if parity_mode else int(latent_width)
        self.parity_mode = bool(parity_mode)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.embedding_width <= 0:
            raise ValueError("embedding_width must be positive")
        if self.latent_width <= 0:
            raise ValueError("latent_width must be positive")
        self._action_to_index = {key: i for i, key in enumerate(self.action_keys)}
        if not self.parity_mode:
            self.embedding = nn.Embedding(len(self.action_keys) + 1, self.embedding_width)
            self.proj = _mlp(self.window_size * self.embedding_width, self.latent_width, self.latent_width)

    def width(self, spec: Optional[StreamSpec] = None) -> int:
        return self.latent_width

    def neutral(self, spec: Optional[StreamSpec] = None) -> List[float]:
        return [0.0] * self.width(spec)

    def encode_keys(self, recent_action_keys: Sequence[str]) -> torch.Tensor:
        keys = list(recent_action_keys)
        if self.parity_mode:
            return torch.tensor(
                motor_history_features_for_actions(keys, self.action_keys),
                dtype=torch.float32,
                device=self._parameter_device(),
            )
        pad_index = len(self.action_keys)
        indices = [self._action_to_index.get(key, pad_index) for key in keys[-self.window_size :]]
        if len(indices) < self.window_size:
            indices = [pad_index] * (self.window_size - len(indices)) + indices
        index_tensor = torch.tensor(indices, dtype=torch.long, device=self._parameter_device())
        embedded = self.embedding(index_tensor).reshape(1, -1)
        return self.proj(embedded).squeeze(0)

    def encode_latent(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[torch.Tensor]:
        if not events:
            return None
        keys: List[str] = []
        if events[-1].stream_id == self.stream_id:
            keys = _action_keys_from_payload(events[-1].payload)
        if not keys:
            for event in events:
                key = _action_key_from_payload(event.payload)
                if key:
                    keys.append(key)
        return self.encode_keys(keys)

    def predict_next_latent(self, latent_slice: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "module": type(self).__name__,
            "trainable": not self.parity_mode,
            "stream_id": self.stream_id,
            "checkpoint_key": self.checkpoint_key,
            "action_keys": list(self.action_keys),
            "window_size": self.window_size,
            "embedding_width": self.embedding_width,
            "latent_width": self.latent_width,
            "parity_mode": self.parity_mode,
            "state_keys": sorted(self.state_dict().keys()),
        }

    def _parameter_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")


class _ScalarMLPEncoder(StreamEncoderModule):
    stream_prefix = ""
    checkpoint_key = ""

    def __init__(self, *, latent_width: int = 8, hidden_width: int = 16) -> None:
        super().__init__()
        self.latent_width = int(latent_width)
        self.hidden_width = int(hidden_width)
        if self.latent_width <= 0:
            raise ValueError("latent_width must be positive")
        if self.hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        self.proj = _mlp(4, self.hidden_width, self.latent_width)

    def width(self, spec: Optional[StreamSpec] = None) -> int:
        return self.latent_width

    def neutral(self, spec: Optional[StreamSpec] = None) -> List[float]:
        return [0.0] * self.width(spec)

    def scalar_features(
        self, events: Sequence[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[torch.Tensor]:
        rng = spec_range(spec)
        values = [
            normalize(v, rng)
            for v in (scalar_leaf(event.payload) for event in events)
            if v is not None
        ]
        if not values:
            return None
        latest = values[-1]
        prev = values[-2] if len(values) >= 2 else latest
        vector = [latest, latest - prev, sum(values) / len(values), max(values)]
        return torch.tensor(vector, dtype=torch.float32, device=self._parameter_device())

    def encode_latent(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[torch.Tensor]:
        if not events:
            return None
        _latest_event(events, self.stream_prefix)
        features = self.scalar_features(events, spec)
        if features is None:
            return None
        return self.proj(features.unsqueeze(0)).squeeze(0)

    def predict_next_latent(self, latent_slice: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "module": type(self).__name__,
            "trainable": True,
            "stream_pattern": f"{self.stream_prefix}*",
            "checkpoint_key": self.checkpoint_key,
            "latent_width": self.latent_width,
            "hidden_width": self.hidden_width,
            "state_keys": sorted(self.state_dict().keys()),
        }

    def _parameter_device(self) -> torch.device:
        return next(self.parameters()).device


class BodyStateEncoder(_ScalarMLPEncoder):
    """Small MLP over scalar ``body.*`` streams."""

    stream_prefix = "body."
    checkpoint_key = BODY_STATE_CHECKPOINT_KEY


class RewardEncoder(_ScalarMLPEncoder):
    """Small MLP over scalar ``reward.*`` streams."""

    stream_prefix = "reward."
    checkpoint_key = REWARD_CHECKPOINT_KEY


class EntityEncoder(StreamEncoderModule):
    """Minimal neural encoder for entity and inventory-style symbolic facts."""

    checkpoint_key = ENTITY_CHECKPOINT_KEY

    def __init__(
        self,
        *,
        latent_width: int = 16,
        hidden_width: int = 32,
        symbol_buckets: int = 64,
        top_k_symbols: int = 4,
    ) -> None:
        super().__init__()
        self.latent_width = int(latent_width)
        self.hidden_width = int(hidden_width)
        self.symbol_buckets = int(symbol_buckets)
        self.top_k_symbols = int(top_k_symbols)
        if self.latent_width <= 0 or self.hidden_width <= 0:
            raise ValueError("latent_width and hidden_width must be positive")
        if self.symbol_buckets <= 0 or self.top_k_symbols <= 0:
            raise ValueError("symbol_buckets and top_k_symbols must be positive")
        self.symbol_embedding = nn.Embedding(self.symbol_buckets, self.hidden_width)
        self.proj = _mlp(self.hidden_width + 6, self.hidden_width, self.latent_width)

    def width(self, spec: Optional[StreamSpec] = None) -> int:
        return self.latent_width

    def neutral(self, spec: Optional[StreamSpec] = None) -> List[float]:
        return [0.0] * self.width(spec)

    def encode_latent(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[torch.Tensor]:
        if not events:
            return None
        latest = events[-1]
        symbols = self._symbols(latest.payload)
        numeric = self._numeric_summary(latest.payload, spec)
        device = self._parameter_device()
        if symbols:
            indices = torch.tensor(
                [_string_bucket(s, self.symbol_buckets) for s in symbols[: self.top_k_symbols]],
                dtype=torch.long,
                device=device,
            )
            symbol_latent = self.symbol_embedding(indices).mean(dim=0)
        else:
            symbol_latent = torch.zeros(self.hidden_width, dtype=torch.float32, device=device)
        numeric_tensor = torch.tensor(numeric, dtype=torch.float32, device=device)
        return self.proj(torch.cat([symbol_latent, numeric_tensor]).unsqueeze(0)).squeeze(0)

    def predict_next_latent(self, latent_slice: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "module": type(self).__name__,
            "trainable": True,
            "stream_pattern": "vision.entities/body.inventory*",
            "checkpoint_key": self.checkpoint_key,
            "latent_width": self.latent_width,
            "hidden_width": self.hidden_width,
            "symbol_buckets": self.symbol_buckets,
            "top_k_symbols": self.top_k_symbols,
            "state_keys": sorted(self.state_dict().keys()),
        }

    def _symbols(self, payload: Any) -> List[str]:
        if isinstance(payload, str):
            return [payload]
        if isinstance(payload, Mapping):
            out: List[str] = []
            for key in sorted(payload):
                if isinstance(key, str):
                    out.append(key)
                out.extend(self._symbols(payload[key]))
            return out
        if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
            out = []
            for item in payload:
                out.extend(self._symbols(item))
            return out
        return []

    def _numeric_summary(self, payload: Any, spec: Optional[StreamSpec]) -> List[float]:
        values = _numeric_leaves(payload)
        if not values:
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        rng = spec_range(spec)
        normalized = [normalize(v, rng) for v in values]
        first_angle = 0.0
        if isinstance(payload, list) and payload and isinstance(payload[0], Mapping):
            first_angle = math.radians(float(payload[0].get("angle", 0.0)))
        return [
            min(len(values), 16.0) / 16.0,
            normalized[0],
            sum(normalized) / len(normalized),
            max(normalized),
            math.sin(first_angle),
            math.cos(first_angle),
        ]

    def _parameter_device(self) -> torch.device:
        return next(self.parameters()).device


class AudioEncoder(StreamEncoderModule):
    """Deliberate fixed stub for reserved ``audio.*`` streams.

    No audio source exists yet.  The module is checkpointable and emits a
    stable zero latent so the registry can reserve the stream id and latent
    contract without pretending capture or spectrogram learning exists.
    """

    stream_pattern = AUDIO_STREAM_PATTERN
    checkpoint_key = AUDIO_CHECKPOINT_KEY

    def __init__(self, latent_width: int = 8) -> None:
        super().__init__()
        self.latent_width = int(latent_width)
        if self.latent_width <= 0:
            raise ValueError("latent_width must be positive")
        self.register_buffer("_zero", torch.zeros(self.latent_width, dtype=torch.float32))

    def width(self, spec: Optional[StreamSpec] = None) -> int:
        return self.latent_width

    def neutral(self, spec: Optional[StreamSpec] = None) -> List[float]:
        return [0.0] * self.width(spec)

    def encode_latent(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[torch.Tensor]:
        if not events:
            return None
        _latest_event(events, "audio.")
        return self._zero.clone()

    def predict_next_latent(self, latent_slice: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "module": type(self).__name__,
            "trainable": False,
            "fixed_stub": True,
            "stream_pattern": self.stream_pattern,
            "checkpoint_key": self.checkpoint_key,
            "latent_width": self.latent_width,
            "note": "Deliberate fixed stub: no audio source exists yet.",
            "state_keys": sorted(self.state_dict().keys()),
        }
