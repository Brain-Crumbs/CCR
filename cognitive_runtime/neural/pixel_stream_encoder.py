"""Reusable CNN encoder for the raw RGB pixel stream.

``PixelStreamEncoder`` is the standalone visual trunk that used to live inside
``VisionPolicyNet``.  It consumes ``vision.frame.pixels`` stream events and
emits a fixed-width visual latent, leaving action heads, fusion, actor/critic
models and world models free to reuse the same encoder weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec
from cognitive_runtime.neural.encoder import StreamEncoderModule

PIXEL_STREAM_ID = "vision.frame.pixels"
PIXEL_CHECKPOINT_KEY = "stream_encoder.vision_frame_pixels"


@dataclass(frozen=True)
class VisualEncoding:
    """A compact temporal token together with its spatial visual evidence."""

    token: torch.Tensor
    spatial: torch.Tensor


class SpatialVisualEncoder(nn.Module):
    """Stride-eight visual encoder used by the spatial residual cortex.

    This deliberately lives beside, rather than replaces, ``PixelStreamEncoder``:
    the latter is a stable stream-module contract used by policy checkpoints.
    """

    architecture = "spatial_residual_v1"

    def __init__(self, pixel_shape: Tuple[int, int, int], latent_width: int, spatial_width: int = 64):
        super().__init__()
        h, w, c = pixel_shape
        if c != 3 or h < 1 or w < 1:
            raise ValueError("spatial encoder requires non-empty RGB frames")
        self.pixel_shape = tuple(pixel_shape)
        self.latent_width = int(latent_width)
        self.spatial_width = int(spatial_width)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, self.spatial_width, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.token = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(self.spatial_width, latent_width), nn.ReLU())

    def encode(self, pixels: torch.Tensor) -> VisualEncoding:
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != (3, *self.pixel_shape[:2]):
            raise ValueError(f"pixels must be [B, 3, {self.pixel_shape[0]}, {self.pixel_shape[1]}]")
        spatial = self.features(pixels)
        return VisualEncoding(token=self.token(spatial), spatial=spatial)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Token-only compatibility surface; use :meth:`encode` for layout."""
        return self.encode(pixels).token


def pixels_to_chw(frame: Any) -> torch.Tensor:
    """Convert an H x W x C RGB frame into a normalized C x H x W tensor.

    ``frame`` is normally an ndarray (the live/recorded pixel stream); a
    nested list is also accepted for legacy sessions.  The ndarray path goes
    through ``torch.from_numpy`` (a view, not a per-element Python conversion).
    """
    array = frame if isinstance(frame, np.ndarray) else np.asarray(frame, dtype=np.uint8)
    if array.size == 0:
        raise ValueError("pixel frame must be a non-empty H x W x C array")
    if array.ndim != 3:
        raise ValueError(f"pixel frame must be 3-dimensional, got shape {tuple(array.shape)}")
    if array.shape[2] != 3:
        raise ValueError(f"pixel frame must have 3 RGB channels, got {array.shape[2]}")
    contiguous = np.ascontiguousarray(array)
    if not contiguous.flags.writeable:
        # A zero-copy mmap view (read-only) from the frame store; torch.from_numpy
        # requires a writable buffer, and we're about to cast to float anyway.
        contiguous = contiguous.copy()
    tensor = torch.from_numpy(contiguous).float()
    return tensor.permute(2, 0, 1).contiguous() / 255.0


class PixelStreamEncoder(StreamEncoderModule):
    """Small CNN over ``vision.frame.pixels`` producing a visual latent."""

    stream_id = PIXEL_STREAM_ID
    checkpoint_key = PIXEL_CHECKPOINT_KEY

    def __init__(self, pixel_shape: Tuple[int, int, int], latent_width: int = 64):
        super().__init__()
        if len(pixel_shape) != 3:
            raise ValueError(f"pixel_shape must be (H, W, C), got {pixel_shape!r}")
        h, w, c = pixel_shape
        if h <= 0 or w <= 0 or c <= 0:
            raise ValueError(f"pixel_shape dimensions must be positive, got {pixel_shape!r}")
        if c != 3:
            raise ValueError(f"pixel_shape must describe RGB frames with 3 channels, got {c}")
        self.pixel_shape = tuple(pixel_shape)
        self.latent_width = int(latent_width)
        if self.latent_width <= 0:
            raise ValueError(f"latent_width must be positive, got {latent_width!r}")

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, self.latent_width),
            nn.ReLU(),
        )

    def width(self, spec: Optional[StreamSpec] = None) -> int:
        if spec is not None and spec.shape is not None and tuple(spec.shape) != self.pixel_shape:
            raise ValueError(
                f"{self.stream_id} spec shape {tuple(spec.shape)} != encoder "
                f"pixel_shape {self.pixel_shape}"
            )
        return self.latent_width

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4:
            raise ValueError(f"pixels must be N x C x H x W, got {tuple(pixels.shape)}")
        expected = (self.pixel_shape[2], self.pixel_shape[0], self.pixel_shape[1])
        if tuple(pixels.shape[1:]) != expected:
            raise ValueError(
                f"pixels must have shape N x {expected[0]} x {expected[1]} x "
                f"{expected[2]}, got {tuple(pixels.shape)}"
            )
        return self.cnn(pixels)

    def encode_frame(self, frame: Any) -> torch.Tensor:
        """Encode one raw HWC frame into ``Tensor[latent_width]``."""
        pixel_batch = pixels_to_chw(frame).unsqueeze(0).to(self._parameter_device())
        return self.forward(pixel_batch).squeeze(0)

    def encode_latent(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[torch.Tensor]:
        if not events:
            return None
        self.width(spec)
        latest = events[-1]
        if latest.stream_id != self.stream_id:
            raise ValueError(
                f"{type(self).__name__} expected {self.stream_id!r} events, "
                f"got {latest.stream_id!r}"
            )
        return self.encode_frame(latest.payload)

    def predict_next_latent(self, latent_slice: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {}

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "module": type(self).__name__,
            "trainable": True,
            "stream_id": self.stream_id,
            "checkpoint_key": self.checkpoint_key,
            "pixel_shape": list(self.pixel_shape),
            "latent_width": self.latent_width,
            "state_keys": sorted(self.state_dict().keys()),
        }

    def _parameter_device(self) -> torch.device:
        return next(self.parameters()).device
