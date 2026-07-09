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

import torch
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
