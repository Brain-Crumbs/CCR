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
from typing import Dict, List, Optional, Sequence, Tuple

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

        hidden = self._hidden(fused_latent, action_onehot)
        return WorldModelOutput(
            next_latent=self.next_latent_head(hidden),
            reward=self.reward_head(hidden).squeeze(-1),
            terminal_logit=self.terminal_head(hidden).squeeze(-1),
            risk=self.risk_head(hidden).squeeze(-1),
            prediction_error=F.softplus(self.prediction_error_head(hidden)).squeeze(-1),
        )

    def _hidden(self, fused_latent: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Shared trunk projection, split out of :meth:`forward` so subclasses
        (:class:`MultiHorizonMLPWorldModel`) can add heads without
        duplicating input validation."""
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
        return self.trunk(x)

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


@dataclass(frozen=True)
class HorizonPrediction:
    """World-model prediction for one horizon step, ``h`` ticks ahead of the
    input tick (issue #39). All fields are ``Tensor[batch, ...]``, same batch
    as the model's input.

    - ``next_latent``: ``Tensor[batch, fused_width]`` -- predicted mean fused
      latent at ``t + h``.
    - ``reward``: ``Tensor[batch]`` -- predicted reward accumulated over the
      ``h`` steps from ``t`` to ``t + h``.
    - ``terminal_logit``: ``Tensor[batch]`` -- pre-sigmoid logit for death/
      episode-end occurring at or before ``t + h``.
    - ``risk``: ``Tensor[batch]`` -- danger estimate at ``t + h``, model's own
      learned units.
    - ``prediction_error``: ``Tensor[batch]`` -- self-estimated realized MSE
      of ``next_latent`` at this horizon (curiosity/novelty signal).
    - ``uncertainty``: ``Tensor[batch]`` -- non-negative learned variance
      estimate for ``next_latent`` at this horizon. This is the one contract
      field every head funnels uncertainty through (docs/neural-stream-
      agent.md): an ensemble-variance or MC-dropout implementation would
      populate the same field without changing callers. Used both to
      separate "surprising because novel" from "surprising because noisy"
      (issue #61) and, downstream, as an attention signal (issue #59).
    """

    next_latent: torch.Tensor
    reward: torch.Tensor
    terminal_logit: torch.Tensor
    risk: torch.Tensor
    prediction_error: torch.Tensor
    uncertainty: torch.Tensor


@dataclass(frozen=True)
class MultiHorizonWorldModelOutput:
    """Predictions at every configured horizon, keyed by tick offset (e.g.
    ``{1: ..., 5: ..., 20: ...}``)."""

    horizons: Dict[int, HorizonPrediction]

    def __getitem__(self, horizon: int) -> HorizonPrediction:
        return self.horizons[horizon]


class MultiHorizonMLPWorldModel(MLPWorldModel):
    """Multi-horizon, uncertainty-aware extension of :class:`MLPWorldModel`
    (issue #39): predicts ``(next_latent, reward, terminal, risk,
    prediction_error, uncertainty)`` at every configured horizon (default
    ``t+1, t+10, t+100``) from the *same* ``(fused_latent, action_onehot)``
    input, via independent linear heads over one shared trunk -- a direct
    multi-head reading of "the interface takes a horizon list so heads can
    be added without contract changes", as opposed to iterated latent
    rollout (also a valid reading per docs/neural-stream-agent.md; the
    ego-motion canary in ``training/ego_motion_canary.py`` uses that
    approach instead, at the pixel level).

    ``horizons`` must include ``1``: horizon 1 reuses the base class's own
    heads (no duplicate parameters), so ``forward()`` is inherited unchanged
    and keeps returning a plain :class:`WorldModelOutput` for ``t+1`` --
    every existing single-step caller (``ActorCriticOptimizer``,
    ``NeuralWorldModel`` bridge, ``ccr train --model-type world-model``)
    keeps working with no changes. :meth:`forward_horizons` is the new
    multi-horizon entry point.
    """

    def __init__(
        self,
        fused_width: int,
        n_actions: int,
        *,
        horizons: Sequence[int] = (1, 10, 100),
        hidden_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.0,
        layout_hash: Optional[str] = None,
        action_keys: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(
            fused_width,
            n_actions,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            layout_hash=layout_hash,
            action_keys=action_keys,
        )
        horizons_sorted = tuple(sorted({int(h) for h in horizons}))
        if not horizons_sorted:
            raise ValueError("horizons must be non-empty")
        if horizons_sorted[0] <= 0:
            raise ValueError(f"horizons must be positive tick offsets, got {horizons!r}")
        if 1 not in horizons_sorted:
            raise ValueError(
                "horizons must include 1 (t+1) so WorldModel.forward() stays a valid "
                f"single-step contract for existing callers, got {horizons!r}"
            )
        self.horizons_list: Tuple[int, ...] = horizons_sorted

        self.uncertainty_heads = nn.ModuleDict(
            {str(h): nn.Linear(hidden_dim, 1) for h in horizons_sorted}
        )
        self.horizon_heads = nn.ModuleDict(
            {
                str(h): nn.ModuleDict(
                    {
                        "next_latent": nn.Linear(hidden_dim, self._fused_width),
                        "reward": nn.Linear(hidden_dim, 1),
                        "terminal": nn.Linear(hidden_dim, 1),
                        "risk": nn.Linear(hidden_dim, 1),
                        "prediction_error": nn.Linear(hidden_dim, 1),
                    }
                )
                for h in horizons_sorted
                if h != 1
            }
        )

    def forward_horizons(
        self, fused_latent: torch.Tensor, action_onehot: torch.Tensor
    ) -> MultiHorizonWorldModelOutput:
        """Predict at every configured horizon in one forward pass."""
        hidden = self._hidden(fused_latent, action_onehot)
        predictions: Dict[int, HorizonPrediction] = {}
        for h in self.horizons_list:
            if h == 1:
                next_latent = self.next_latent_head(hidden)
                reward = self.reward_head(hidden).squeeze(-1)
                terminal_logit = self.terminal_head(hidden).squeeze(-1)
                risk = self.risk_head(hidden).squeeze(-1)
                prediction_error = F.softplus(self.prediction_error_head(hidden)).squeeze(-1)
            else:
                heads = self.horizon_heads[str(h)]
                next_latent = heads["next_latent"](hidden)
                reward = heads["reward"](hidden).squeeze(-1)
                terminal_logit = heads["terminal"](hidden).squeeze(-1)
                risk = heads["risk"](hidden).squeeze(-1)
                prediction_error = F.softplus(heads["prediction_error"](hidden)).squeeze(-1)
            uncertainty = F.softplus(self.uncertainty_heads[str(h)](hidden)).squeeze(-1)
            predictions[h] = HorizonPrediction(
                next_latent=next_latent,
                reward=reward,
                terminal_logit=terminal_logit,
                risk=risk,
                prediction_error=prediction_error,
                uncertainty=uncertainty,
            )
        return MultiHorizonWorldModelOutput(horizons=predictions)

    def checkpoint_metadata(self) -> Dict[str, object]:
        metadata = super().checkpoint_metadata()
        metadata["horizons"] = list(self.horizons_list)
        return metadata
