"""Policy-model contract (Phase A: interface only).

:class:`PolicyModel` is the actor half of step 5 in ``docs/online-learning.md``'s
neural upgrade path: fused latent state plus whatever features the
:class:`~cognitive_runtime.neural.world_model.WorldModel` exposes (its
predicted next state, risk, prediction error, ...) map to action logits,
replacing the linear online Q learner while keeping it as the baseline and
smoke-test target.

No concrete policy architecture is implemented here.
"""

from __future__ import annotations

import abc

import torch
from torch import nn


class PolicyModel(nn.Module, abc.ABC):
    """Maps fused latent state + world-model features to action logits.

    Input/output shapes
    --------------------
    - ``fused_latent``: ``Tensor[batch, fused_width]`` -- the current tick's
      fused agent state (``LatentFusionModel`` output).
    - ``world_features``: ``Tensor[batch, world_feature_width]`` -- features
      derived from the ``WorldModel``'s predictions for the candidate/likely
      actions (e.g. predicted reward, risk, terminal probability,
      prediction error, concatenated or otherwise combined); concrete
      subclasses document the exact composition and width they expect.
    - Returns ``Tensor[batch, action_space_size()]``, unnormalized action
      logits over the program's action space (softmax/argmax happens in the
      calling policy, mirroring ``OnlineQModel.q_values`` returning raw
      scores rather than probabilities).

    Checkpoint keys
    ---------------
    ``state_dict()``/``load_state_dict()`` are :class:`torch.nn.Module`'s own
    (parameters and buffers). A loader additionally needs the ordered
    ``action_keys`` this model was trained against, so it can refuse to load
    a bundle trained on an incompatible action space the way
    ``OnlineQModel.check_compatible`` does today; the checkpoint bundle
    format itself is a separate issue.
    """

    def __init__(self) -> None:
        nn.Module.__init__(self)

    @abc.abstractmethod
    def action_space_size(self) -> int:
        """Number of actions this model produces logits for."""

    @abc.abstractmethod
    def forward(
        self, fused_latent: torch.Tensor, world_features: torch.Tensor
    ) -> torch.Tensor:
        """Return ``Tensor[batch, action_space_size()]`` action logits."""
