"""Entity-persistence model: predicts a tracked entity's feature during an
occlusion gap (issue #27, Phase D "object permanence").

Given the entity's last-seen feature vector (``core.entity_features
.entity_feature_vector``) and how long it has been occluded, predicts what
its feature is *right now* -- position/identity, not full appearance -- plus
a self-supervised "surprise" (expected prediction error), the second
ingredient of the combined novelty score alongside the world model's
``prediction_error`` (``neural.world_model.WorldModelOutput``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from cognitive_runtime.core.entity_features import ENTITY_FEATURE_WIDTH

ENTITY_PERSISTENCE_CHECKPOINT_KEY = "entity_persistence"

#: Ticks-since-occluded is normalized by this cap before feeding the model,
#: so a very long gap doesn't blow up the input scale.
#: ``training.entity_persistence`` uses the same cap when building gap
#: features, and ``core.entity_tracker.EntityTracker.max_gap_ticks`` bounds
#: tracked gaps in the first place.
DEFAULT_GAP_CAP_TICKS = 200.0


def normalize_gap(gap_ticks: float, gap_cap: float = DEFAULT_GAP_CAP_TICKS) -> float:
    if gap_cap <= 0:
        return 0.0
    return min(1.0, max(0.0, float(gap_ticks) / gap_cap))


@dataclass(frozen=True)
class EntityPersistenceOutput:
    """One batch of persistence predictions.

    - ``predicted_feature``: ``Tensor[batch, feature_width]`` -- the model's
      guess at the entity's feature right now (position/identity, the same
      shape ``entity_feature_vector`` produces).
    - ``surprise``: ``Tensor[batch]`` -- the model's own non-negative
      estimate of its error on ``predicted_feature``, trained self-supervised
      against its realized error the same way
      ``MLPWorldModel.prediction_error`` is (see
      ``training.entity_persistence``); feeds the combined novelty score.
    """

    predicted_feature: torch.Tensor
    surprise: torch.Tensor


class EntityPersistenceModel(nn.Module):
    """MLP trunk over ``[last_feature, gap_norm]`` predicting an occluded
    entity's current feature plus a self-supervised surprise estimate.

    Checkpoint keys
    ---------------
    ``state_dict()``/``load_state_dict()`` are :class:`torch.nn.Module`'s
    own; a loader additionally needs ``checkpoint_metadata()`` to validate
    compatibility before restoring weights, the same way ``MLPWorldModel``
    does.
    """

    def __init__(
        self,
        *,
        feature_width: int = ENTITY_FEATURE_WIDTH,
        hidden_dim: int = 32,
        depth: int = 2,
        dropout: float = 0.0,
        gap_cap: float = DEFAULT_GAP_CAP_TICKS,
    ) -> None:
        super().__init__()
        if feature_width <= 0:
            raise ValueError(f"feature_width must be positive, got {feature_width!r}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim!r}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth!r}")

        self.feature_width = int(feature_width)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.gap_cap = float(gap_cap)

        layers: List[nn.Module] = []
        width = self.feature_width + 1
        for _ in range(depth):
            layers.append(nn.Linear(width, hidden_dim))
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
            width = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.feature_head = nn.Linear(hidden_dim, self.feature_width)
        self.surprise_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, last_feature: torch.Tensor, gap_norm: torch.Tensor
    ) -> EntityPersistenceOutput:
        if last_feature.ndim != 2 or last_feature.shape[1] != self.feature_width:
            raise ValueError(
                f"last_feature shape must be [batch, {self.feature_width}], got "
                f"{tuple(last_feature.shape)}"
            )
        if gap_norm.ndim != 1 or gap_norm.shape[0] != last_feature.shape[0]:
            raise ValueError(
                f"gap_norm shape must be [batch] matching last_feature's batch "
                f"{last_feature.shape[0]}, got {tuple(gap_norm.shape)}"
            )
        x = torch.cat([last_feature.float(), gap_norm.float().unsqueeze(-1)], dim=1)
        hidden = self.trunk(x)
        return EntityPersistenceOutput(
            predicted_feature=self.feature_head(hidden),
            surprise=F.softplus(self.surprise_head(hidden)).squeeze(-1),
        )

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "feature_width": self.feature_width,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "dropout": self.dropout,
            "gap_cap": self.gap_cap,
        }
