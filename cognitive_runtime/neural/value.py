"""Value-model contract (Phase A: interface only).

:class:`ValueModel` is the critic half of step 5 in ``docs/online-learning.md``'s
neural upgrade path, providing the expected-return baseline an actor/critic
policy needs alongside :class:`~cognitive_runtime.neural.policy.PolicyModel`.

No concrete value architecture is implemented here.
"""

from __future__ import annotations

import abc
from typing import Dict, List, Optional, Sequence

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
    (parameters and buffers); ``NeuralAgentCheckpoint`` bundles this with its
    matching ``PolicyModel``/``WorldModel`` state.
    """

    def __init__(self) -> None:
        nn.Module.__init__(self)

    @abc.abstractmethod
    def forward(
        self, fused_latent: torch.Tensor, world_features: torch.Tensor
    ) -> torch.Tensor:
        """Return ``Tensor[batch]`` expected-return estimate."""


class MLPValueModel(ValueModel):
    """Phase-E concrete critic: an MLP trunk over ``[fused_latent,
    world_features]`` feeding a scalar value head.

    Mirrors :class:`~cognitive_runtime.neural.policy.MLPPolicyModel`'s
    construction so a policy/critic pair trained together share
    ``fused_width``/``world_feature_width`` by convention, though nothing
    enforces they use the same ``hidden_dim``/``depth``.
    """

    def __init__(
        self,
        fused_width: int,
        world_feature_width: int,
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
        if world_feature_width < 0:
            raise ValueError(
                f"world_feature_width must be non-negative, got {world_feature_width!r}"
            )
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim!r}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth!r}")

        self._fused_width = int(fused_width)
        self._world_feature_width = int(world_feature_width)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.layout_hash = layout_hash
        self.action_keys = list(action_keys) if action_keys is not None else None

        layers: List[nn.Module] = []
        width = self._fused_width + self._world_feature_width
        for _ in range(depth):
            layers.append(nn.Linear(width, hidden_dim))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, fused_latent: torch.Tensor, world_features: torch.Tensor
    ) -> torch.Tensor:
        if fused_latent.ndim != 2 or fused_latent.shape[1] != self._fused_width:
            raise ValueError(
                f"fused_latent shape must be [batch, {self._fused_width}], got "
                f"{tuple(fused_latent.shape)}"
            )
        if world_features.ndim != 2 or world_features.shape[1] != self._world_feature_width:
            raise ValueError(
                f"world_features shape must be [batch, {self._world_feature_width}], got "
                f"{tuple(world_features.shape)}"
            )
        if fused_latent.shape[0] != world_features.shape[0]:
            raise ValueError(
                f"fused_latent batch {fused_latent.shape[0]} != world_features batch "
                f"{world_features.shape[0]}"
            )
        x = torch.cat([fused_latent.float(), world_features.float()], dim=1)
        return self.value_head(self.trunk(x)).squeeze(-1)

    def checkpoint_metadata(self) -> Dict[str, object]:
        return {
            "fused_width": self._fused_width,
            "world_feature_width": self._world_feature_width,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "layout_hash": self.layout_hash,
            "action_keys": self.action_keys,
        }
