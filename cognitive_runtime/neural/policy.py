"""Policy-model contract (Phase A: interface only).

:class:`PolicyModel` is the actor half of step 5 in ``docs/history/online-learning.md``'s
neural upgrade path: fused latent state plus whatever features the
:class:`~cognitive_runtime.neural.world_model.WorldModel` exposes (its
predicted next state, risk, prediction error, ...) map to action logits,
replacing the linear online Q learner while keeping it as the baseline and
smoke-test target.

No concrete policy architecture is implemented here.
"""

from __future__ import annotations

import abc
from typing import Dict, List, Mapping, Optional, Sequence

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
    ``OnlineQModel.check_compatible`` does today; the unified checkpoint
    bundle stores this in ``NeuralAgentCheckpoint`` metadata.
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


class MLPPolicyModel(PolicyModel):
    """Phase-E concrete actor head: an MLP trunk over ``[fused_latent,
    world_features]`` feeding a linear action-logit head.

    Construction
    ------------
    ``fused_width`` must match the runtime's fused-latent width
    (``memory.fused_latent()``/``TemporalFusion.width``); ``world_feature_width``
    must match whatever composition of world-model-derived features the
    caller builds (see ``cognitive_runtime.policies.actor_critic.world_features_vector``).
    ``n_actions`` is the ordered action-space size; ``action_keys`` is
    optional bookkeeping recorded in checkpoint metadata only.
    """

    def __init__(
        self,
        fused_width: int,
        world_feature_width: int,
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
        if world_feature_width < 0:
            raise ValueError(
                f"world_feature_width must be non-negative, got {world_feature_width!r}"
            )
        if n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {n_actions!r}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim!r}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth!r}")

        self._fused_width = int(fused_width)
        self._world_feature_width = int(world_feature_width)
        self.n_actions = int(n_actions)
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
        self.logits_head = nn.Linear(hidden_dim, self.n_actions)

    def action_space_size(self) -> int:
        return self.n_actions

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
        return self.logits_head(self.trunk(x))

    def checkpoint_metadata(self) -> Dict[str, object]:
        return {
            "fused_width": self._fused_width,
            "world_feature_width": self._world_feature_width,
            "n_actions": self.n_actions,
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
        """Load a checkpoint trained on a smaller, ordered-prefix action space
        into this (already larger) model in place (issue #42's action-space
        growth story): every weight/bias row or column that already existed
        under ``old_action_keys`` is copied verbatim; the rows/columns this
        model added for the new tail actions keep their own fresh
        initialization instead of being overwritten by garbage or zeros.

        Only ``logits_head`` (grows by output row per new action) and
        ``trunk.0`` (grows by input column: ``world_feature_width`` appends
        the motor-history one-hot at its tail, see
        ``cognitive_runtime.policies.actor_critic.world_feature_width``) can
        legitimately change shape; any other shape mismatch is a real
        incompatibility (a different ``hidden_dim``/``depth``/``fused_width``),
        not action-space growth, and is raised rather than silently patched.
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
        if self.n_actions != len(new_keys):
            raise ValueError(
                f"model is sized for {self.n_actions} actions, but "
                f"new_action_keys has {len(new_keys)}"
            )
        merged = dict(self.state_dict())
        for key, old_tensor in old_state.items():
            if key not in merged:
                continue
            new_tensor = merged[key]
            if tuple(old_tensor.shape) == tuple(new_tensor.shape):
                merged[key] = old_tensor
            elif key == "logits_head.weight":
                grown = new_tensor.clone()
                grown[: old_tensor.shape[0], :] = old_tensor
                merged[key] = grown
            elif key == "logits_head.bias":
                grown = new_tensor.clone()
                grown[: old_tensor.shape[0]] = old_tensor
                merged[key] = grown
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
