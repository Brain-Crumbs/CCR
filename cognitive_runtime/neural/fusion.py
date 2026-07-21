"""Learned fusion over fixed per-stream latent slices.

Where ``cognitive_runtime.core.streams.fusion.TemporalFusion`` deterministically
concatenates each stream's fixed-width slice (with a spec's neutral value
filling silent streams) into one versioned vector, :class:`LatentFusionModel`
is the learned upgrade path ``docs/history/online-learning.md`` describes: a model
that consumes those same per-stream slices -- plus which streams were present
this tick and how recently each last fired -- and produces one fused agent
state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch import nn

from cognitive_runtime.core.streams.fusion import TemporalFusion
from cognitive_runtime.core.streams.synchronizer import TickWindow
from cognitive_runtime.core.streams.temporal_buffer import TemporalBuffer


@dataclass(frozen=True)
class LatentFusionInputs:
    """Tensor-ready learned-fusion inputs for one or more ticks."""

    latents: torch.Tensor
    presence_mask: torch.Tensor
    recency: torch.Tensor
    staleness: torch.Tensor
    attention: torch.Tensor
    layout_hash: str
    stream_ids: List[str]


class LatentFusionModel(nn.Module):
    """Fuses per-stream latent slices into one agent-state vector.

    Input/output shapes
    --------------------
    - ``latents``: ``Tensor[batch, total_width]`` -- the streams' latent
      slices concatenated in the same deterministic, versioned stream-id
      order ``TemporalFusion`` uses (one fixed layout per stream catalog;
      see ``TemporalFusion.layout_hash``).
    - ``presence_mask``: ``Tensor[batch, n_streams]`` of 0./1. -- whether
      each stream produced a token this tick, since ``TemporalFusion`` already
      zero-fills silent streams with their neutral value and this model may
      still need to distinguish "silent" from "reported zero".
    - ``recency``: ``Tensor[batch, n_streams]`` in ``[0, 1]`` -- a
      recency-weighted activation per stream, the learned analogue of
      ``TemporalFusion._event_recency``.
    - ``attention_weights`` (optional): ``Tensor[batch, n_streams]``, the
      attention-controller hook (issue #57/#59) -- how much each stream
      should contribute this tick. Omitting it (or passing all-ones) is
      byte-equivalent: the channel defaults to ones, so a fresh model with
      no attention controller attached reproduces plain fusion exactly.
    - Returns ``Tensor[batch, fused_width]``, the fused agent state consumed
      by :class:`~cognitive_runtime.neural.world_model.WorldModel` and
      :class:`~cognitive_runtime.neural.policy.PolicyModel`.

    Construction
    ------------
    Pass ``stream_slices`` (``stream_id -> (lo, hi)``) and ``layout_hash`` to
    create the concrete Phase-C MLP model.  The no-argument constructor remains
    available to subclasses that implement their own architecture, preserving
    the Phase-A contract used by tests and toy modules.

    Checkpoint keys
    ---------------
    ``state_dict()``/``load_state_dict()`` are :class:`torch.nn.Module`'s own
    (parameters and buffers). A checkpoint loader also needs
    :meth:`fused_width` and the stream-id layout (order and per-stream width)
    the model was trained against, so it can refuse to load a bundle trained
    on an incompatible catalog the way ``OnlineQModel.check_compatible`` does
    today; ``cognitive_runtime.neural.checkpoint.NeuralAgentCheckpoint``
    carries those compatibility fields in the unified bundle metadata.
    """

    def __init__(
        self,
        stream_slices: Optional[Mapping[str, Tuple[int, int]]] = None,
        *,
        layout_hash: Optional[str] = None,
        fused_width: int = 128,
        hidden_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        nn.Module.__init__(self)
        if stream_slices is None:
            if type(self) is LatentFusionModel:
                raise TypeError(
                    "LatentFusionModel needs stream_slices/layout_hash; use "
                    "LatentFusionModel.from_temporal_fusion(...) for the "
                    "concrete learned fusion model"
                )
            return
        if layout_hash is None:
            raise ValueError("layout_hash is required for learned fusion compatibility")
        if fused_width <= 0:
            raise ValueError(f"fused_width must be positive, got {fused_width!r}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim!r}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth!r}")

        ordered = sorted((sid, tuple(bounds)) for sid, bounds in stream_slices.items())
        if not ordered:
            raise ValueError("LatentFusionModel needs at least one stream slice")
        input_width = 0
        normalized: Dict[str, Tuple[int, int]] = {}
        for stream_id, (lo, hi) in ordered:
            if lo < 0 or hi <= lo:
                raise ValueError(f"invalid slice for {stream_id!r}: {(lo, hi)!r}")
            input_width = max(input_width, hi)
            normalized[stream_id] = (int(lo), int(hi))

        self.stream_slices = normalized
        self.stream_ids = [stream_id for stream_id, _bounds in ordered]
        self.layout_hash = layout_hash
        self.input_width = int(input_width)
        self._fused_width = int(fused_width)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)

        model_input_width = self.input_width + 4 * len(self.stream_ids)
        layers: List[nn.Module] = []
        width = model_input_width
        for _ in range(depth):
            layers.append(nn.Linear(width, hidden_dim))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = hidden_dim
        layers.append(nn.Linear(width, fused_width))
        self.net = nn.Sequential(*layers)

    @classmethod
    def from_temporal_fusion(
        cls,
        fusion: TemporalFusion,
        *,
        fused_width: Optional[int] = None,
        hidden_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> "LatentFusionModel":
        """Create a learned fusion model from a ``TemporalFusion`` layout."""

        offset = 0
        slices: Dict[str, Tuple[int, int]] = {}
        for entry in fusion.layout:
            slices[entry.stream_id] = (offset, offset + entry.width)
            offset += entry.width
        return cls(
            slices,
            layout_hash=fusion.layout_hash,
            fused_width=fused_width or fusion.width,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        )

    def fused_width(self) -> int:
        """Width of the fused agent-state vector this model produces."""
        return self._fused_width

    def forward(
        self,
        latents: torch.Tensor,
        presence_mask: torch.Tensor,
        recency: torch.Tensor,
        staleness: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse per-stream latents (+ presence/recency/attention) into one
        agent state.

        Returns ``Tensor[batch, fused_width()]``.
        """
        if latents.ndim != 2 or latents.shape[1] != self.input_width:
            raise ValueError(
                f"latents shape must be [batch, {self.input_width}], got "
                f"{tuple(latents.shape)}"
            )
        expected_mask = (latents.shape[0], len(self.stream_ids))
        if tuple(presence_mask.shape) != expected_mask:
            raise ValueError(
                f"presence_mask shape must be {expected_mask}, got "
                f"{tuple(presence_mask.shape)}"
            )
        if tuple(recency.shape) != expected_mask:
            raise ValueError(
                f"recency shape must be {expected_mask}, got {tuple(recency.shape)}"
            )
        if staleness is None:
            staleness = 1.0 - recency.clamp(0.0, 1.0)
        if tuple(staleness.shape) != expected_mask:
            raise ValueError(
                f"staleness shape must be {expected_mask}, got "
                f"{tuple(staleness.shape)}"
            )
        if attention_weights is None:
            attention_weights = torch.ones(expected_mask, dtype=torch.float32)
        if tuple(attention_weights.shape) != expected_mask:
            raise ValueError(
                f"attention_weights shape must be {expected_mask}, got "
                f"{tuple(attention_weights.shape)}"
            )

        latents = latents.float()
        presence_mask = presence_mask.float()
        recency = recency.float()
        staleness = staleness.float()
        attention_weights = attention_weights.float()
        features = torch.cat(
            [latents, presence_mask, recency, staleness, attention_weights], dim=1
        )
        return self.net(features)

    def checkpoint_metadata(self) -> Dict[str, object]:
        if "stream_slices" not in self.__dict__:
            metadata: Dict[str, object] = {}
            try:
                metadata["fused_width"] = self.fused_width()
            except Exception:
                pass
            return metadata
        return {
            "layout_hash": self.layout_hash,
            "stream_ids": list(self.stream_ids),
            "stream_slices": {key: list(value) for key, value in self.stream_slices.items()},
            "input_width": self.input_width,
            "fused_width": self._fused_width,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "dropout": self.dropout,
        }


def latent_fusion_inputs_from_buffer(
    fusion: TemporalFusion,
    temporal_buffer: TemporalBuffer,
    *,
    tick_window: Optional[TickWindow] = None,
    present_stream_ids: Optional[Iterable[str]] = None,
    stale_streams: Optional[Iterable[str]] = None,
    attention_weights: Optional[Mapping[str, float]] = None,
) -> LatentFusionInputs:
    """Build the mask/recency tensors for learned fusion from stream time.

    ``TemporalFusion`` still owns the fixed vector and layout hash.  This
    helper adds the learned path's extra channels: which streams arrived in
    the current tick, how recently each stream last fired, a normalized
    staleness scalar, and -- the attention-controller hook (issue #59) --
    each stream's attention weight, defaulting to ``1.0`` (uniform, byte-
    equivalent to no attention controller) for any stream `attention_weights`
    doesn't mention. Stale streams reported by ``TickSynchronizer`` override
    the staleness scalar to 1.0.
    """

    latent = fusion.fuse(None, temporal_buffer)
    if tick_window is not None:
        present = set(tick_window.by_stream)
    else:
        present = set(present_stream_ids or [])
    stale = set(stale_streams or [])
    reference_time = fusion._reference_time(temporal_buffer)
    attention_weights = attention_weights or {}

    mask: List[float] = []
    recency: List[float] = []
    staleness: List[float] = []
    attention: List[float] = []
    for entry in fusion.layout:
        latest = temporal_buffer.latest(entry.stream_id)
        recent = (
            fusion._event_recency(latest.timestamp, reference_time)
            if latest is not None
            else 0.0
        )
        mask.append(1.0 if entry.stream_id in present else 0.0)
        recency.append(float(recent))
        staleness.append(1.0 if entry.stream_id in stale else 1.0 - float(recent))
        attention.append(float(attention_weights.get(entry.stream_id, 1.0)))

    return LatentFusionInputs(
        latents=torch.tensor([latent.vector], dtype=torch.float32),
        presence_mask=torch.tensor([mask], dtype=torch.float32),
        recency=torch.tensor([recency], dtype=torch.float32),
        staleness=torch.tensor([staleness], dtype=torch.float32),
        attention=torch.tensor([attention], dtype=torch.float32),
        layout_hash=latent.layout_hash,
        stream_ids=[entry.stream_id for entry in fusion.layout],
    )
