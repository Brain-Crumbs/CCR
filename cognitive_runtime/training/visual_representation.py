"""Self-supervised visual pretraining for the pixel stream encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from cognitive_runtime.neural.checkpoint import FORMAT_VERSION, NeuralAgentCheckpoint
from cognitive_runtime.neural.pixel_stream_encoder import (
    PIXEL_CHECKPOINT_KEY,
    PixelStreamEncoder,
    pixels_to_chw,
)
from cognitive_runtime.training.datasets import PixelSequenceDataset
from cognitive_runtime.training.features import ACTION_KEYS


class PixelReconstructionDecoder(nn.Module):
    """Decode a visual latent into a compact RGB reconstruction."""

    def __init__(
        self,
        latent_width: int,
        output_shape: Tuple[int, int, int],
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        h, w, c = output_shape
        if h <= 0 or w <= 0 or c != 3:
            raise ValueError(f"output_shape must be positive RGB HWC, got {output_shape!r}")
        self.output_shape = tuple(output_shape)
        self.net = nn.Sequential(
            nn.Linear(latent_width, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, h * w * c),
            nn.Sigmoid(),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        h, w, c = self.output_shape
        flat = self.net(latents)
        return flat.view(latents.shape[0], c, h, w)


class NextLatentPredictor(nn.Module):
    """Predict the next visual latent from the current visual latent."""

    def __init__(self, latent_width: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_width, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_width),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents)


class ContrastiveProjection(nn.Module):
    """Projection head used by adjacent-frame InfoNCE."""

    def __init__(self, latent_width: int, projection_dim: Optional[int] = None) -> None:
        super().__init__()
        width = int(projection_dim or latent_width)
        self.net = nn.Sequential(
            nn.Linear(latent_width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(latents)


class VisualRepresentationModel(nn.Module):
    """Pixel encoder plus self-supervised representation-learning heads."""

    def __init__(
        self,
        pixel_shape: Tuple[int, int, int],
        latent_width: int = 64,
        reconstruction_shape: Optional[Tuple[int, int, int]] = None,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.pixel_shape = tuple(pixel_shape)
        self.latent_width = int(latent_width)
        self.reconstruction_shape = reconstruction_shape or _default_reconstruction_shape(pixel_shape)
        self.encoder = PixelStreamEncoder(self.pixel_shape, latent_width=self.latent_width)
        self.decoder = PixelReconstructionDecoder(
            self.latent_width, self.reconstruction_shape, hidden_dim=hidden_dim
        )
        self.next_predictor = NextLatentPredictor(self.latent_width, hidden_dim=hidden_dim)
        self.projection = ContrastiveProjection(self.latent_width)

    def encode_pair(self, pixels: torch.Tensor, next_pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(pixels), self.encoder(next_pixels)


@dataclass
class VisualPretrainingConfig:
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 0
    latent_width: int = 64
    hidden_dim: int = 128
    reconstruction_size: int = 16
    reconstruction_weight: float = 1.0
    next_latent_weight: float = 1.0
    contrastive_weight: float = 1.0
    contrastive_temperature: float = 0.2


def reconstruction_loss(
    decoder: PixelReconstructionDecoder,
    latents: torch.Tensor,
    pixels: torch.Tensor,
) -> torch.Tensor:
    reconstruction = decoder(latents)
    target = reconstruction_target(pixels, decoder.output_shape)
    return F.mse_loss(reconstruction, target)


def reconstruction_target(
    pixels: torch.Tensor,
    output_shape: Tuple[int, int, int],
) -> torch.Tensor:
    h, w, _c = output_shape
    if tuple(pixels.shape[-2:]) == (h, w):
        return pixels
    return F.interpolate(pixels, size=(h, w), mode="area")


def next_latent_prediction_loss(
    predictor: NextLatentPredictor,
    latents: torch.Tensor,
    next_latents: torch.Tensor,
) -> torch.Tensor:
    predicted = F.normalize(predictor(latents), dim=1)
    target = F.normalize(next_latents.detach(), dim=1)
    return F.mse_loss(predicted, target)


def contrastive_consistency_loss(
    projection: ContrastiveProjection,
    latents: torch.Tensor,
    next_latents: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    if latents.shape[0] < 2:
        return latents.new_zeros(())
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")
    left = F.normalize(projection(latents), dim=1)
    right = F.normalize(projection(next_latents), dim=1)
    logits = left @ right.T / temperature
    labels = torch.arange(latents.shape[0], device=latents.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def train_pixel_encoder_pretraining(
    dataset: PixelSequenceDataset,
    config: Optional[VisualPretrainingConfig] = None,
) -> Tuple[VisualRepresentationModel, Dict[str, Any]]:
    """Pretrain ``PixelStreamEncoder`` with reconstruction/prediction/contrastive losses."""
    if len(dataset) == 0:
        raise ValueError("pixel sequence dataset is empty; record sessions with --record-frames")
    cfg = config or VisualPretrainingConfig()
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    pixel_shape = tuple(dataset.pixel_shape or _infer_shape(dataset.pixels[0]))
    reconstruction_shape = _reconstruction_shape(pixel_shape, cfg.reconstruction_size)

    pixels = torch.stack([pixels_to_chw(p) for p in dataset.pixels])
    next_pixels = torch.stack([pixels_to_chw(p) for p in dataset.next_pixels])
    model = VisualRepresentationModel(
        pixel_shape,
        latent_width=cfg.latent_width,
        reconstruction_shape=reconstruction_shape,
        hidden_dim=cfg.hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    curves: Dict[str, List[float]] = {
        "total_loss": [],
        "reconstruction_loss": [],
        "next_latent_loss": [],
        "contrastive_loss": [],
    }

    model.train()
    for _epoch in range(cfg.epochs):
        epoch = {key: 0.0 for key in curves}
        seen = 0
        perm = torch.randperm(len(dataset), generator=generator)
        for start in range(0, perm.numel(), cfg.batch_size):
            batch = perm[start : start + cfg.batch_size]
            current = pixels[batch]
            future = next_pixels[batch]
            optimizer.zero_grad()
            latents, next_latents = model.encode_pair(current, future)
            recon = reconstruction_loss(model.decoder, latents, current)
            nxt = next_latent_prediction_loss(model.next_predictor, latents, next_latents)
            contrast = contrastive_consistency_loss(
                model.projection,
                latents,
                next_latents,
                temperature=cfg.contrastive_temperature,
            )
            total = (
                cfg.reconstruction_weight * recon
                + cfg.next_latent_weight * nxt
                + cfg.contrastive_weight * contrast
            )
            total.backward()
            optimizer.step()

            batch_n = int(batch.numel())
            seen += batch_n
            epoch["total_loss"] += float(total.detach()) * batch_n
            epoch["reconstruction_loss"] += float(recon.detach()) * batch_n
            epoch["next_latent_loss"] += float(nxt.detach()) * batch_n
            epoch["contrastive_loss"] += float(contrast.detach()) * batch_n
        for key in curves:
            curves[key].append(round(epoch[key] / max(seen, 1), 6))

    stats: Dict[str, Any] = {
        "samples": float(len(dataset)),
        "epochs": float(cfg.epochs),
        "batch_size": float(cfg.batch_size),
        "lr": float(cfg.lr),
        "latent_width": float(cfg.latent_width),
        "reconstruction_shape": list(reconstruction_shape),
        "loss_curves": curves,
        "final_total_loss": curves["total_loss"][-1],
        "final_reconstruction_loss": curves["reconstruction_loss"][-1],
        "final_next_latent_loss": curves["next_latent_loss"][-1],
        "final_contrastive_loss": curves["contrastive_loss"][-1],
    }
    return model, stats


def save_pixel_encoder_pretraining_checkpoint(
    path: str,
    model: VisualRepresentationModel,
    dataset: PixelSequenceDataset,
    stats: Dict[str, Any],
    name: Optional[str] = None,
) -> Dict[str, Any]:
    manager = NeuralAgentCheckpoint(
        path,
        layout_hash=dataset.layout_hash or "pixel-sequence-layout",
        action_keys=list(ACTION_KEYS),
        encoders={PIXEL_CHECKPOINT_KEY: model.encoder},
        replay_metadata={
            "sources": list(dataset.sources),
            "representation": dataset.representation,
            "pixel_shape": list(model.pixel_shape),
        },
        training_stats=stats,
        training_ticks=len(dataset),
        extra_metadata={
            "model_type": "pixel-encoder",
            "losses": ["reconstruction", "next_latent_prediction", "contrastive_consistency"],
        },
        name=name,
    )
    return manager.save(reason="pixel_encoder_pretraining")


def load_pretrained_pixel_encoder(
    path: str,
    *,
    pixel_shape: Optional[Tuple[int, int, int]] = None,
    latent_width: Optional[int] = None,
    map_location: str | torch.device = "cpu",
) -> PixelStreamEncoder:
    """Load only the pixel encoder from a unified neural checkpoint bundle."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - older torch without weights_only
        payload = torch.load(path, map_location=map_location)
    if payload.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported neural checkpoint format {payload.get('format')!r}")
    metadata = payload.get("metadata", {})
    encoder_meta = (
        metadata.get("modules", {})
        .get("encoders", {})
        .get(PIXEL_CHECKPOINT_KEY, {})
        .get("checkpoint_metadata", {})
    )
    checkpoint_shape = tuple(encoder_meta.get("pixel_shape", pixel_shape or ()))
    checkpoint_width = int(encoder_meta.get("latent_width", latent_width or 0))
    if pixel_shape is not None and tuple(pixel_shape) != checkpoint_shape:
        raise ValueError(
            f"pretrained pixel encoder shape {checkpoint_shape} != requested {tuple(pixel_shape)}"
        )
    if latent_width is not None and int(latent_width) != checkpoint_width:
        raise ValueError(
            f"pretrained pixel encoder width {checkpoint_width} != requested {latent_width}"
        )
    if not checkpoint_shape or checkpoint_width <= 0:
        raise ValueError("pretrained pixel encoder checkpoint is missing shape/width metadata")
    state = payload.get("state", {}).get("encoders", {}).get(PIXEL_CHECKPOINT_KEY)
    if state is None:
        raise ValueError(f"checkpoint does not contain {PIXEL_CHECKPOINT_KEY!r}")
    encoder = PixelStreamEncoder(checkpoint_shape, latent_width=checkpoint_width)
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder


def _default_reconstruction_shape(pixel_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return _reconstruction_shape(pixel_shape, 16)


def _reconstruction_shape(pixel_shape: Tuple[int, int, int], max_side: int) -> Tuple[int, int, int]:
    h, w, c = pixel_shape
    if max_side <= 0:
        raise ValueError(f"reconstruction_size must be positive, got {max_side!r}")
    scale = min(1.0, float(max_side) / float(max(h, w)))
    return (max(1, round(h * scale)), max(1, round(w * scale)), c)


def _infer_shape(frame: Any) -> Tuple[int, int, int]:
    if hasattr(frame, "shape"):
        return tuple(frame.shape)  # type: ignore[return-value]
    h = len(frame)
    w = len(frame[0]) if h else 0
    c = len(frame[0][0]) if w else 0
    return (h, w, c)
