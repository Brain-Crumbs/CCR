"""World-model contract (Phase A: interface only).

:class:`WorldModel` is the learned predictor ``docs/online-learning.md``
describes as step 4 of the neural upgrade path: given the fused agent state
and the action taken, predict what happens next -- the next latent state,
expected reward, terminal/death probability, a risk estimate, and the
model's own prediction error -- so a policy/value pair (step 5) can condition
on those predictions instead of only the raw fused state.

No concrete world-model architecture is implemented here.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class WorldModelOutput:
    """One step of world-model predictions. All fields are batched tensors.

    - ``next_latent``: ``Tensor[batch, fused_width]`` -- predicted next fused
      agent state (same width as the ``LatentFusionModel`` output it
      predicts).
    - ``reward``: ``Tensor[batch]`` -- predicted scalar reward for the step.
    - ``terminal_logit``: ``Tensor[batch]`` -- pre-sigmoid logit for
      episode-end/death probability (apply ``sigmoid`` for a probability).
    - ``risk``: ``Tensor[batch]`` -- a scalar danger estimate in the model's
      own learned units (not necessarily a probability).
    - ``prediction_error``: ``Tensor[batch]`` -- the model's own estimate of
      its error on ``next_latent`` (e.g. predicted MSE against the realized
      next state), intended to drive curiosity/exploration bonuses
      downstream.
    """

    next_latent: torch.Tensor
    reward: torch.Tensor
    terminal_logit: torch.Tensor
    risk: torch.Tensor
    prediction_error: torch.Tensor


class WorldModel(nn.Module, abc.ABC):
    """Predicts next latent state, reward, terminal, risk, and prediction
    error from the current fused state and action.

    Input/output shapes
    --------------------
    - ``fused_latent``: ``Tensor[batch, fused_width]`` -- the current tick's
      fused agent state (``LatentFusionModel`` output).
    - ``action_onehot``: ``Tensor[batch, n_actions]`` -- the action taken
      this tick, one-hot over the program's action space.
    - Returns a :class:`WorldModelOutput`; see its docstring for per-field
      shapes.

    Checkpoint keys
    ---------------
    ``state_dict()``/``load_state_dict()`` are :class:`torch.nn.Module`'s own
    (parameters and buffers). A loader additionally needs ``fused_width`` and
    ``n_actions`` to validate compatibility before restoring weights, the
    same way ``OnlineQModel`` checks ``latent_width``/``action_keys`` today;
    ``NeuralAgentCheckpoint`` carries those compatibility fields.
    """

    def __init__(self) -> None:
        nn.Module.__init__(self)

    @abc.abstractmethod
    def forward(
        self, fused_latent: torch.Tensor, action_onehot: torch.Tensor
    ) -> WorldModelOutput:
        """Predict next-state/reward/terminal/risk/error given state+action."""


class MLPWorldModel(WorldModel):
    """Phase-D concrete world model: an MLP trunk over ``[fused_latent,
    action_onehot]`` feeding five linear heads, one per :class:`WorldModelOutput`
    field.

    ``prediction_error`` is trained self-supervised against the model's own
    realized ``next_latent`` MSE (see ``cognitive_runtime.training.world_model``),
    so it is kept non-negative with a softplus rather than a raw linear head.

    Construction
    ------------
    ``fused_width`` must match ``memory.fused_latent()``'s width (the
    ``TemporalFusion`` vector the runtime already computes each tick -- the
    same input :class:`~cognitive_runtime.neural.fusion.LatentFusionModel`
    eventually replaces).  ``n_actions`` is the ordered action-space size;
    ``action_keys`` is optional bookkeeping recorded in checkpoint metadata
    only, not used by the forward pass.
    """

    def __init__(
        self,
        fused_width: int,
        n_actions: int,
        *,
        hidden_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.0,
        layout_hash: Optional[str] = None,
        action_keys: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if fused_width <= 0:
            raise ValueError(f"fused_width must be positive, got {fused_width!r}")
        if n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {n_actions!r}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim!r}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth!r}")

        self._fused_width = int(fused_width)
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.layout_hash = layout_hash
        self.action_keys = list(action_keys) if action_keys is not None else None

        layers: List[nn.Module] = []
        width = self._fused_width + self.n_actions
        for _ in range(depth):
            layers.append(nn.Linear(width, hidden_dim))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.next_latent_head = nn.Linear(hidden_dim, self._fused_width)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.terminal_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.prediction_error_head = nn.Linear(hidden_dim, 1)

    def fused_width(self) -> int:
        """Width of the fused agent-state vector this model consumes/predicts."""
        return self._fused_width

    def forward(
        self, fused_latent: torch.Tensor, action_onehot: torch.Tensor
    ) -> WorldModelOutput:
        if fused_latent.ndim != 2 or fused_latent.shape[1] != self._fused_width:
            raise ValueError(
                f"fused_latent shape must be [batch, {self._fused_width}], got "
                f"{tuple(fused_latent.shape)}"
            )
        if action_onehot.ndim != 2 or action_onehot.shape[1] != self.n_actions:
            raise ValueError(
                f"action_onehot shape must be [batch, {self.n_actions}], got "
                f"{tuple(action_onehot.shape)}"
            )
        if fused_latent.shape[0] != action_onehot.shape[0]:
            raise ValueError(
                f"fused_latent batch {fused_latent.shape[0]} != action_onehot batch "
                f"{action_onehot.shape[0]}"
            )

        x = torch.cat([fused_latent.float(), action_onehot.float()], dim=1)
        hidden = self.trunk(x)
        return WorldModelOutput(
            next_latent=self.next_latent_head(hidden),
            reward=self.reward_head(hidden).squeeze(-1),
            terminal_logit=self.terminal_head(hidden).squeeze(-1),
            risk=self.risk_head(hidden).squeeze(-1),
            prediction_error=F.softplus(self.prediction_error_head(hidden)).squeeze(-1),
        )

    def checkpoint_metadata(self) -> Dict[str, object]:
        return {
            "fused_width": self._fused_width,
            "n_actions": self.n_actions,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "layout_hash": self.layout_hash,
            "action_keys": self.action_keys,
        }
