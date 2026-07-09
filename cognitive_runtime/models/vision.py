"""PyTorch pixel-vision behavioral cloning model.

This module is intentionally isolated from the rest of the runtime so torch
remains optional.  Importing it requires PyTorch; default runtime, replay, and
linear BC paths never import it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from torch import nn

from cognitive_runtime.neural.pixel_stream_encoder import (
    PixelStreamEncoder,
    pixels_to_chw,
)

REPRESENTATION = "neural_pixels"
VISION_BC_FORMAT = "vision-bc-v2"
LEGACY_VISION_BC_FORMAT = "vision-bc-v1"


class VisionPolicyNet(nn.Module):
    """Pixel stream encoder plus an MLP action head."""

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

        self.encoder = PixelStreamEncoder(self.pixel_shape, latent_width=self.embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim + self.n_non_vision + self.n_motor, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    @property
    def cnn(self) -> nn.Sequential:
        """Backward-compatible access to the visual trunk."""
        return self.encoder.cnn

    def visual_latent(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixels)

    def forward(self, pixels: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        expected_aux = self.n_non_vision + self.n_motor
        if aux.ndim != 2 or aux.shape[1] != expected_aux:
            raise ValueError(f"aux must be N x {expected_aux}, got {tuple(aux.shape)}")
        visual = self.visual_latent(pixels)
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
                "format": VISION_BC_FORMAT,
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
        bundle_format = raw.get("format", LEGACY_VISION_BC_FORMAT)
        if bundle_format not in {LEGACY_VISION_BC_FORMAT, VISION_BC_FORMAT}:
            raise ValueError(
                f"unsupported VisionBCModel bundle format {bundle_format!r}; "
                f"expected {VISION_BC_FORMAT!r}"
            )
        net = VisionPolicyNet(**raw["config"])
        net.load_state_dict(_migrate_state_dict(raw["state_dict"]))
        net.eval()
        return VisionBCModel(
            net=net,
            action_keys=list(raw["action_keys"]),
            meta=dict(raw.get("meta", {})),
        )


def _migrate_state_dict(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Map pre-Phase-B ``cnn.*`` keys onto the composed encoder module."""
    migrated: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("cnn."):
            migrated[f"encoder.{key}"] = value
        else:
            migrated[key] = value
    return migrated

