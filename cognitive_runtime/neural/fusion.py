"""Learned fusion contract (Phase A: interface only).

Where ``cognitive_runtime.core.streams.fusion.TemporalFusion`` deterministically
concatenates each stream's fixed-width slice (with a spec's neutral value
filling silent streams) into one versioned vector, :class:`LatentFusionModel`
is the learned upgrade path ``docs/online-learning.md`` describes: a model
that consumes those same per-stream slices -- plus which streams were present
this tick and how recently each last fired -- and produces one fused agent
state.

No concrete fusion architecture is implemented here.
"""

from __future__ import annotations

import abc

import torch
from torch import nn


class LatentFusionModel(nn.Module, abc.ABC):
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
    - Returns ``Tensor[batch, fused_width]``, the fused agent state consumed
      by :class:`~cognitive_runtime.neural.world_model.WorldModel` and
      :class:`~cognitive_runtime.neural.policy.PolicyModel`.

    Checkpoint keys
    ---------------
    ``state_dict()``/``load_state_dict()`` are :class:`torch.nn.Module`'s own
    (parameters and buffers). A checkpoint loader also needs
    :meth:`fused_width` and the stream-id layout (order and per-stream width)
    the model was trained against, so it can refuse to load a bundle trained
    on an incompatible catalog the way ``OnlineQModel.check_compatible`` does
    today; the checkpoint *bundle* format that carries those fields is a
    separate issue.
    """

    def __init__(self) -> None:
        nn.Module.__init__(self)

    @abc.abstractmethod
    def fused_width(self) -> int:
        """Width of the fused agent-state vector this model produces."""

    @abc.abstractmethod
    def forward(
        self,
        latents: torch.Tensor,
        presence_mask: torch.Tensor,
        recency: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse per-stream latents (+ presence/recency) into one agent state.

        Returns ``Tensor[batch, fused_width()]``.
        """
