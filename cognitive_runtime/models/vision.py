"""PyTorch pixel-vision behavioral cloning model.

This module is intentionally isolated from the rest of the runtime so torch
remains optional.  Importing it requires PyTorch; default runtime, replay, and
linear BC paths never import it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

REPRESENTATION = "neural_pixels"


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


class VisionPolicyNet(nn.Module):
    """Small CNN over pixels plus an MLP over non-vision and motor features."""

    def __init__(
        self,
        pixel_shape: Tuple[int, int, int],
        n_non_vision: int,
        n_motor: int,
        n_actions: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        if len(pixel_shape) != 3:
            raise ValueError(f"pixel_shape must be (H, W, C), got {pixel_shape!r}")
        h, w, c = pixel_shape
        if h <= 0 or w <= 0 or c <= 0:
            raise ValueError(f"pixel_shape dimensions must be positive, got {pixel_shape!r}")
        self.pixel_shape = tuple(pixel_shape)
        self.n_non_vision = int(n_non_vision)
        self.n_motor = int(n_motor)
        self.n_actions = int(n_actions)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, embed_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim + self.n_non_vision + self.n_motor, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, pixels: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4:
            raise ValueError(f"pixels must be N x C x H x W, got {tuple(pixels.shape)}")
        expected_aux = self.n_non_vision + self.n_motor
        if aux.ndim != 2 or aux.shape[1] != expected_aux:
            raise ValueError(f"aux must be N x {expected_aux}, got {tuple(aux.shape)}")
        visual = self.cnn(pixels)
        return self.head(torch.cat([visual, aux], dim=1))

    def config(self) -> Dict[str, Any]:
        return {
            "pixel_shape": list(self.pixel_shape),
            "n_non_vision": self.n_non_vision,
            "n_motor": self.n_motor,
            "n_actions": self.n_actions,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
        }


@dataclass
class VisionBCModel:
    """Serializable pixel BC bundle used by neural training and inference."""

    net: VisionPolicyNet
    action_keys: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def pixel_shape(self) -> Tuple[int, int, int]:
        return self.net.pixel_shape

    def logits(
        self,
        pixels: Any,
        non_vision: Sequence[float],
        motor: Sequence[float],
    ) -> List[float]:
        expected_non_vision = self.net.n_non_vision
        expected_motor = self.net.n_motor
        if len(non_vision) != expected_non_vision:
            raise ValueError(
                f"non-vision feature width {len(non_vision)} != model width "
                f"{expected_non_vision}"
            )
        if len(motor) != expected_motor:
            raise ValueError(f"motor feature width {len(motor)} != model width {expected_motor}")
        self.net.eval()
        with torch.no_grad():
            pixel_batch = pixels_to_chw(pixels).unsqueeze(0)
            aux = torch.tensor([list(non_vision) + list(motor)], dtype=torch.float32)
            return self.net(pixel_batch, aux).squeeze(0).tolist()

    def predict_index(
        self,
        pixels: Any,
        non_vision: Sequence[float],
        motor: Sequence[float],
    ) -> int:
        scores = self.logits(pixels, non_vision, motor)
        return max(range(len(scores)), key=scores.__getitem__)

    def predict_key(
        self,
        pixels: Any,
        non_vision: Sequence[float],
        motor: Sequence[float],
    ) -> str:
        return self.action_keys[self.predict_index(pixels, non_vision, motor)]

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.net.config(),
                "state_dict": self.net.state_dict(),
                "action_keys": list(self.action_keys),
                "meta": dict(self.meta),
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "VisionBCModel":
        raw = torch.load(path, map_location="cpu")
        net = VisionPolicyNet(**raw["config"])
        net.load_state_dict(raw["state_dict"])
        net.eval()
        return VisionBCModel(
            net=net,
            action_keys=list(raw["action_keys"]),
            meta=dict(raw.get("meta", {})),
        )

