"""Temporal backbones for the predictive cortex (issue #93,
docs/v2/phases/phase-2-predictive-cortex.md task 5): the GRU transition is
one choice of "how does the world state advance from ``(latent, action)``
pairs", not the only one. This module gives that choice a seam.

Every backbone implements the same three-method contract
:class:`PredictiveCortex` drives them through:

- ``initial_state(batch)`` -- an opaque per-batch state (a hidden tensor for
  the GRU, a ring buffer of recent inputs for the windowed backbones).
- ``step(x, state)`` -- advance by one ``(latent, action)`` input vector
  ``x``; returns ``(hidden, new_state)`` where ``hidden`` is always
  ``Tensor[batch, hidden_dim]`` -- the representation the cortex's
  ``latent_head``/reward/terminal/risk/uncertainty heads read.
- ``readout(state)`` -- recover that same ``hidden`` from a ``state`` object
  handed back later (heads are applied to whatever ``state`` the cortex's
  ``rollout``/``forward_horizons`` carries between steps, which for the
  windowed backbones is not the hidden tensor itself).

Because ``step``/``readout`` are the *entire* surface :class:`PredictiveCortex`
touches, swapping the backbone changes nothing about the cortex's external
forward/rollout contract or its structured scoring output -- an A/B, not a
fork, per decision log #9.

``GRUBackbone`` reads one ``(latent, action)`` pair at a time with unbounded
effective context (the recurrent state accumulates the whole history).
``DilatedConvBackbone`` (WaveNet-style) and ``TransformerBackbone`` instead
process a *window* of the last ``context_length`` inputs in one parallel
pass each step -- the "reads several timescales at once" alternative the
phase doc asks for. Their window is curriculum-controlled via
``set_context_length`` (task 5's "1 frame -> 2 -> k"): training starts each
windowed backbone seeing only its most recent input and ramps the window up
to ``context_length_max``, so the model isn't asked to exploit a long window
before it has learned what one step of dynamics looks like.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Backbone name -> constructor. Populated at the bottom of this module once
#: every backbone class is defined.
_BACKBONES: dict = {}


class TemporalBackbone(nn.Module):
    """Base class for the cortex's swappable transition backbone.

    ``context_length_max`` is ``None`` for backbones with no fixed window
    (the GRU); the training curriculum uses its presence to decide whether a
    backbone has a context length to ramp at all.
    """

    hidden_dim: int
    context_length_max: Optional[int] = None

    def initial_state(self, batch: int) -> Any:
        raise NotImplementedError

    def step(self, x: torch.Tensor, state: Any) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError

    def readout(self, state: Any) -> torch.Tensor:
        raise NotImplementedError

    def set_context_length(self, n: Optional[int]) -> None:
        """No-op unless overridden by a windowed backbone."""


class GRUBackbone(TemporalBackbone):
    """The original transition: a single ``GRUCell`` with unbounded
    (recurrent-state) context."""

    def __init__(self, input_dim: int, hidden_dim: int, **_unused: Any) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def initial_state(self, batch: int) -> torch.Tensor:
        return self.cell.weight_ih.new_zeros(batch, self.hidden_dim)

    def step(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        new_state = self.cell(x, state)
        return new_state, new_state

    def readout(self, state: torch.Tensor) -> torch.Tensor:
        return state


class _WindowedBackbone(TemporalBackbone):
    """Shared ring-buffer bookkeeping for the windowed backbones: ``state``
    is ``(buffer, last_hidden)`` where ``buffer`` is
    ``Tensor[batch, context_length_max, input_dim]`` holding the most recent
    raw ``(latent, action)`` inputs (oldest first, zero-padded at the
    start), and ``last_hidden`` is this step's ``Tensor[batch, hidden_dim]``
    readout.
    """

    def __init__(self, input_dim: int, hidden_dim: int, context_length: int) -> None:
        super().__init__()
        if context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {context_length}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_length_max = context_length
        self._current_context = context_length

    def set_context_length(self, n: Optional[int]) -> None:
        limit = self.context_length_max or 1
        self._current_context = max(1, min(int(n), limit)) if n else limit

    def initial_state(self, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        device_param = next(self.parameters())
        buffer = device_param.new_zeros(batch, self.context_length_max, self.input_dim)
        hidden = device_param.new_zeros(batch, self.hidden_dim)
        return buffer, hidden

    def readout(self, state: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return state[1]

    def _slide_window(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        buffer, _last_hidden = state
        return torch.cat([buffer[:, 1:], x.unsqueeze(1)], dim=1)

    def _windowed(self, buffer: torch.Tensor) -> torch.Tensor:
        return buffer[:, -self._current_context :]


class DilatedConvBackbone(_WindowedBackbone):
    """WaveNet-style causal dilated 1-D convolution stack over the window.

    Doubling dilation per layer gives an exponentially growing receptive
    field for a linear parameter/depth cost; ``n_layers`` defaults to just
    enough layers to cover ``context_length``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        context_length: int = 8,
        kernel_size: int = 2,
        n_layers: Optional[int] = None,
        **_unused: Any,
    ) -> None:
        super().__init__(input_dim, hidden_dim, context_length)
        if kernel_size < 2:
            raise ValueError(f"kernel_size must be >= 2, got {kernel_size}")
        depth = n_layers or max(1, math.ceil(math.log(max(context_length, 2), kernel_size)))
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv_layers = nn.ModuleList()
        self.dilations = []
        dilation = 1
        for _ in range(depth):
            self.conv_layers.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=dilation)
            )
            self.dilations.append(dilation)
            dilation *= kernel_size
        self.activation = nn.ReLU()

    def step(
        self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        buffer = self._slide_window(x, state)
        window = self._windowed(buffer)  # [B, C, input_dim]
        h = self.input_proj(window).transpose(1, 2)  # [B, hidden_dim, C]
        for conv, dilation in zip(self.conv_layers, self.dilations):
            pad = dilation * (conv.kernel_size[0] - 1)
            h = self.activation(conv(F.pad(h, (pad, 0))))
        hidden = h[:, :, -1]
        return hidden, (buffer, hidden)


class TransformerBackbone(_WindowedBackbone):
    """A small causal transformer encoder over the window: full pairwise
    attention across the last ``context_length`` inputs each step, in one
    parallel pass, rather than the GRU's one-token-at-a-time recurrence."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        context_length: int = 8,
        n_heads: int = 2,
        n_layers: int = 1,
        **_unused: Any,
    ) -> None:
        super().__init__(input_dim, hidden_dim, context_length)
        if hidden_dim % n_heads != 0:
            # Attention needs hidden_dim divisible by n_heads; fall back to
            # the largest divisor <= n_heads rather than erroring on odd
            # configs a caller didn't hand-tune for this backbone.
            n_heads = next(h for h in range(min(n_heads, hidden_dim), 0, -1) if hidden_dim % h == 0)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Embedding(context_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def step(
        self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        buffer = self._slide_window(x, state)
        window = self._windowed(buffer)  # [B, C, input_dim]
        length = window.shape[1]
        positions = torch.arange(length, device=window.device)
        projected = self.input_proj(window) + self.position_embedding(positions).unsqueeze(0)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(length).to(window.device)
        encoded = self.encoder(projected, mask=causal_mask)
        hidden = encoded[:, -1]
        return hidden, (buffer, hidden)


_BACKBONES.update(
    {
        "gru": GRUBackbone,
        "dilated_conv": DilatedConvBackbone,
        "transformer": TransformerBackbone,
    }
)


def build_backbone(name: str, input_dim: int, hidden_dim: int, **kwargs: Any) -> TemporalBackbone:
    """Construct the named backbone. ``kwargs`` are backbone-specific
    (``context_length``, ``kernel_size``, ``n_layers``, ``n_heads``, ...);
    each backbone ignores the ones it doesn't use."""
    try:
        cls = _BACKBONES[name]
    except KeyError:
        raise ValueError(f"unknown cortex backbone {name!r}; choices: {sorted(_BACKBONES)}") from None
    return cls(input_dim, hidden_dim, **kwargs)
