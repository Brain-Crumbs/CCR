"""Value-model contract (Phase A: interface only).

:class:`ValueModel` is the critic half of step 5 in ``docs/history/online-learning.md``'s
neural upgrade path, providing the expected-return baseline an actor/critic
policy needs alongside :class:`~cognitive_runtime.neural.policy.PolicyModel`.

No concrete value architecture is implemented here.
"""

from __future__ import annotations

import abc
from typing import Dict, List, Mapping, Optional, Sequence

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

    def load_state_dict_with_action_growth(
        self,
        old_state: Mapping[str, torch.Tensor],
        old_action_keys: Sequence[str],
        new_action_keys: Sequence[str],
    ) -> None:
        """Critic counterpart of ``MLPPolicyModel.load_state_dict_with_action_growth``
        (issue #42): the critic has no per-action output head, but its
        ``trunk.0`` input still grows with ``world_feature_width`` (the
        motor-history one-hot appended at the tail), so it needs the same
        column-preserving surgery.
        """
        old_keys = list(old_action_keys)
        new_keys = list(new_action_keys)
        if not old_keys or new_keys[: len(old_keys)] != old_keys:
            raise ValueError(
                "action-space growth requires old_action_keys to be a "
                f"non-empty ordered prefix of new_action_keys; old={old_keys} "
                f"new={new_keys}"
            )
        if len(new_keys) <= len(old_keys):
            raise ValueError(
                "load_state_dict_with_action_growth expects a strict "
                f"superset; old has {len(old_keys)} actions, new has "
                f"{len(new_keys)}"
            )
        merged = dict(self.state_dict())
        for key, old_tensor in old_state.items():
            if key not in merged:
                continue
            new_tensor = merged[key]
            if tuple(old_tensor.shape) == tuple(new_tensor.shape):
                merged[key] = old_tensor
            elif key == "trunk.0.weight":
                grown = new_tensor.clone()
                grown[:, : old_tensor.shape[1]] = old_tensor
                merged[key] = grown
            else:
                raise ValueError(
                    f"cannot grow action space: unexpected shape change in "
                    f"{key!r} ({tuple(old_tensor.shape)} -> "
                    f"{tuple(new_tensor.shape)})"
                )
        self.load_state_dict(merged, strict=True)
        self.action_keys = new_keys
