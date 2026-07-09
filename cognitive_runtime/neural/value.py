"""Value-model contract (Phase A: interface only).

:class:`ValueModel` is the critic half of step 5 in ``docs/online-learning.md``'s
neural upgrade path, providing the expected-return baseline an actor/critic
policy needs alongside :class:`~cognitive_runtime.neural.policy.PolicyModel`.

No concrete value architecture is implemented here.
"""

from __future__ import annotations

import abc

import torch
from torch import nn


class ValueModel(nn.Module, abc.ABC):
    """Estimates expected return from fused latent state + world-model
    features (the actor/critic baseline).

    Input/output shapes
    --------------------
    - ``fused_latent``: ``Tensor[batch, fused_width]`` -- the current tick's
      fused agent state (``LatentFusionModel`` output).
    - ``world_features``: ``Tensor[batch, world_feature_width]`` -- the same
      world-model-derived features :class:`~cognitive_runtime.neural.policy.PolicyModel`
      consumes.
    - Returns ``Tensor[batch]``, the scalar expected-return estimate for each
      item in the batch.

    Checkpoint keys
    ---------------
    ``state_dict()``/``load_state_dict()`` are :class:`torch.nn.Module`'s own
    (parameters and buffers); the checkpoint bundle format that pairs this
    with its matching ``PolicyModel``/``WorldModel`` is a separate issue.
    """

    def __init__(self) -> None:
        nn.Module.__init__(self)

    @abc.abstractmethod
    def forward(
        self, fused_latent: torch.Tensor, world_features: torch.Tensor
    ) -> torch.Tensor:
        """Return ``Tensor[batch]`` expected-return estimate."""
