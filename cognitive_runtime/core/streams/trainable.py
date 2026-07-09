"""Trainable stream-module interface for future modular neural learning.

The current runtime still uses fixed :class:`StreamEncoder` instances and
``TemporalFusion`` unchanged.  These modules define the next abstraction layer:
per-stream encoders that can later learn, predict their own next latent slice,
and save checkpoint state without forcing v1 online Q to become neural.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Mapping, Optional, Sequence

from cognitive_runtime.core.streams.encoder_registry import LatentToken, StreamEncoder
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec


class TrainableStreamModule(StreamEncoder, abc.ABC):
    """A stream encoder with optional online-training and checkpoint hooks."""

    def train_mode(self) -> None:
        """Enable training behavior for modules that have it."""

    def eval_mode(self) -> None:
        """Disable training behavior for deterministic evaluation."""

    def predict_next(self, latent_slice: Sequence[float]) -> Dict[str, Any]:
        """Predict future state for this stream slice.

        Fixed wrappers return an empty dict; future learned modules can return
        values such as ``{"next": [...], "risk": 0.2}``.
        """
        return {}

    def update(self, loss_signal: Mapping[str, Any]) -> Dict[str, float]:
        """Apply a module-local learning signal and return scalar metrics."""
        return {}

    def state_dict(self) -> Dict[str, Any]:
        """Serializable trainable state. Fixed wrappers have no weights."""
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore module state previously returned by :meth:`state_dict`."""

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "module": type(self).__name__,
            "trainable": True,
            "state_keys": sorted(self.state_dict().keys()),
        }

    def checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "format": "trainable-stream-module-v1",
            "metadata": self.checkpoint_metadata(),
            "state": self.state_dict(),
        }


class FixedStreamModule(TrainableStreamModule):
    """No-op trainable wrapper around an existing fixed ``StreamEncoder``."""

    def __init__(self, encoder: StreamEncoder):
        self.encoder = encoder
        self.training = False

    def train_mode(self) -> None:
        self.training = True

    def eval_mode(self) -> None:
        self.training = False

    def encode(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[LatentToken]:
        return self.encoder.encode(events, spec)

    def width(self, spec: Optional[StreamSpec] = None) -> int:
        return self.encoder.width(spec)

    def neutral(self, spec: Optional[StreamSpec] = None) -> List[float]:
        return self.encoder.neutral(spec)

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError(
                f"{type(self).__name__} wraps fixed encoder "
                f"{type(self.encoder).__name__} and cannot load trainable state"
            )

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "module": type(self).__name__,
            "encoder": type(self.encoder).__name__,
            "trainable": False,
            "training": self.training,
            "state_keys": [],
        }


def fixed_stream_module(encoder: StreamEncoder) -> FixedStreamModule:
    """Wrap a fixed encoder in the trainable module interface."""
    return FixedStreamModule(encoder)

