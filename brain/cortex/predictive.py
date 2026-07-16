"""The Predictive Cortex: one recurrent, action-conditioned, decoded,
multi-horizon world model (docs/v2/phases/phase-2-predictive-cortex.md,
issue #91).

Promoted from ``cognitive_runtime.training.action_world_model``'s
``ActionConditionedWorldModel`` (an encoder + action-conditioned GRUCell
transition + latent head + decoder), which is kept as a re-export shim so
existing imports keep resolving. Two structural additions over the
promoted prototype:

- **Multi-horizon reward/terminal/risk/uncertainty heads.** A single
  closed-loop rollout (:meth:`PredictiveCortex.forward_horizons`) now
  yields, at *every* configured horizon, a decoded frame plus reward /
  terminal / risk / uncertainty predictions -- the per-horizon structure
  ``cognitive_runtime.neural.world_model.MultiHorizonMLPWorldModel`` uses
  for the (memoryless) fused-latent world model, folded into this
  recurrent one. Uncertainty is produced cheaply (a softplus'd linear head
  on the GRU hidden state, trained as a predicted-error estimate) rather
  than an ensemble, per the phase doc's "build it calibratable" note --
  Phase 3's arbiter mode-switch depends on this signal.
- **Horizons in ticks.** ``PredictiveCortexConfig.horizons_ticks`` is the
  configured horizon set in *ticks*, persisted with the checkpoint, so
  "T+8" means the same thing across worlds and recording sample rates.
  ``forward_horizons`` itself still operates in *frame* steps (the unit
  its rollout is indexed in); callers convert ticks to frames per-dataset
  via ``horizons_ticks_to_frames`` before calling it, exactly as
  ``evaluate_action_world_model`` already does.

Latent + decoder discipline (short rollout, scheduled sampling, latent
loss primary / pixel loss auxiliary) is a training-loop concern and stays
in ``training/action_world_model.py``, unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_HORIZONS_TICKS: Tuple[int, ...] = (1, 4, 8)


@dataclass
class PredictiveCortexConfig:
    """Architecture knobs for :class:`PredictiveCortex`.

    Training hyperparameters (epochs, lr, rollout schedule, ...) live in
    ``training.action_world_model.ActionWorldModelConfig``, which carries
    this subset plus its own training-only fields and translates into a
    ``PredictiveCortexConfig`` when building the model.
    """

    latent_width: int = 32
    hidden_dim: int = 64
    action_embed_dim: int = 8
    reconstruction_size: int = 16
    #: Horizons in ticks (per-organism configurable), stored with the
    #: checkpoint. Frame-space horizons for a given recording are derived
    #: via ``horizons_ticks_to_frames(horizons_ticks, ticks_per_frame)``.
    horizons_ticks: Tuple[int, ...] = field(default=DEFAULT_HORIZONS_TICKS)


@dataclass(frozen=True)
class CortexHorizonPrediction:
    """One rollout horizon's worth of predictions, all ``Tensor[batch, ...]``.

    - ``latent``: predicted latent at this horizon.
    - ``decoded``: ``self.decoder(latent)``, same shape as the model's own
      reconstruction target (``reconstruction_shape``) -- viewable at every
      horizon, not just the last.
    - ``reward`` / ``terminal_logit`` / ``risk``: the world-model heads
      ``cognitive_runtime.neural.world_model.WorldModelOutput`` defines,
      read off the GRU hidden state at this horizon.
    - ``uncertainty``: non-negative predicted-error estimate for ``latent``
      at this horizon -- the calibratable sigma Phase 3's arbiter reads.
    """

    latent: torch.Tensor
    decoded: torch.Tensor
    reward: torch.Tensor
    terminal_logit: torch.Tensor
    risk: torch.Tensor
    uncertainty: torch.Tensor


@dataclass(frozen=True)
class CortexRolloutOutput:
    """Predictions at every requested horizon, keyed by frame-step offset."""

    horizons: Dict[int, CortexHorizonPrediction]

    def __getitem__(self, horizon: int) -> CortexHorizonPrediction:
        return self.horizons[horizon]


class PredictiveCortex(nn.Module):
    """Encoder + action-conditioned GRU transition + decoder, with
    multi-horizon reward/terminal/risk/uncertainty heads.

    The GRU hidden state is the model's world state: it accumulates
    observation history (so rotation *rate* is representable) and is
    advanced by ``(latent, action)`` pairs. Closed-loop rollout feeds
    predicted latents back in, so evaluation exercises exactly the
    multi-horizon interface the nursery benchmarks.
    """

    def __init__(
        self,
        pixel_shape: Tuple[int, int, int],
        action_keys: Sequence[str],
        config: Optional[PredictiveCortexConfig] = None,
    ) -> None:
        super().__init__()
        from cognitive_runtime.neural.pixel_stream_encoder import PixelStreamEncoder
        from cognitive_runtime.training.visual_representation import (
            PixelReconstructionDecoder,
            _reconstruction_shape,
        )

        cfg = config or PredictiveCortexConfig()
        horizons_ticks = tuple(sorted({int(h) for h in cfg.horizons_ticks}))
        if not horizons_ticks or horizons_ticks[0] < 1:
            raise ValueError(f"horizons_ticks must be positive, got {cfg.horizons_ticks!r}")

        self.config = cfg
        self.pixel_shape = tuple(int(d) for d in pixel_shape)
        self.action_keys = list(action_keys)
        self.latent_width = cfg.latent_width
        self.hidden_dim = cfg.hidden_dim
        self.horizons_ticks: Tuple[int, ...] = horizons_ticks
        self.reconstruction_shape = _reconstruction_shape(
            self.pixel_shape, cfg.reconstruction_size
        )

        self.encoder = PixelStreamEncoder(self.pixel_shape, latent_width=cfg.latent_width)
        self.action_embedding = nn.Embedding(len(self.action_keys), cfg.action_embed_dim)
        self.transition = nn.GRUCell(cfg.latent_width + cfg.action_embed_dim, cfg.hidden_dim)
        self.latent_head = nn.Linear(cfg.hidden_dim, cfg.latent_width)
        self.decoder = PixelReconstructionDecoder(
            cfg.latent_width, self.reconstruction_shape, hidden_dim=cfg.hidden_dim
        )

        # Multi-horizon heads (task 2): applied to the GRU hidden state at
        # every rollout step, so one closed-loop rollout yields
        # reward/terminal/risk/uncertainty at *every* configured horizon
        # instead of requiring a separate head set per horizon the way the
        # memoryless MultiHorizonMLPWorldModel does.
        self.reward_head = nn.Linear(cfg.hidden_dim, 1)
        self.terminal_head = nn.Linear(cfg.hidden_dim, 1)
        self.risk_head = nn.Linear(cfg.hidden_dim, 1)
        #: Predicted-error head (softplus'd, non-negative) -- the cheap,
        #: calibratable sigma the phase doc asks for in place of an
        #: ensemble.
        self.uncertainty_head = nn.Linear(cfg.hidden_dim, 1)

    def initial_state(self, batch: int) -> torch.Tensor:
        weight = self.latent_head.weight
        return weight.new_zeros(batch, self.hidden_dim)

    def step(
        self, latent: torch.Tensor, action_idx: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance the world state by one ``(latent, action)`` pair; returns
        ``(predicted next latent, next hidden)``."""
        embedded = self.action_embedding(action_idx)
        hidden = self.transition(torch.cat([latent, embedded], dim=-1), hidden)
        return self.latent_head(hidden), hidden

    def heads(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reward / terminal-logit / risk / uncertainty read off one rollout
        step's hidden state; returns four ``Tensor[batch]``."""
        return (
            self.reward_head(hidden).squeeze(-1),
            self.terminal_head(hidden).squeeze(-1),
            F.softplus(self.risk_head(hidden)).squeeze(-1),
            F.softplus(self.uncertainty_head(hidden)).squeeze(-1),
        )

    def rollout(
        self, start_latent: torch.Tensor, actions: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Closed-loop rollout: from ``start_latent`` and world state
        ``hidden``, apply ``actions`` (``[B, R]`` indices) feeding each
        predicted latent back in; returns predicted latents ``[B, R, L]``."""
        predictions = []
        latent = start_latent
        for i in range(actions.shape[1]):
            latent, hidden = self.step(latent, actions[:, i], hidden)
            predictions.append(latent)
        return torch.stack(predictions, dim=1), hidden

    def forward_horizons(
        self,
        start_latent: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor,
        horizon_frames: Optional[Sequence[int]] = None,
    ) -> CortexRolloutOutput:
        """One closed-loop rollout, sliced at every requested horizon.

        ``horizon_frames`` are frame-step offsets (see module docstring);
        defaults to :attr:`horizons_ticks` taken as frame steps for a
        caller that has no ticks-per-frame conversion to apply (e.g. the
        common ~1-tick-per-frame case). ``actions`` must cover at least
        ``max(horizon_frames)`` steps.

        This is task 2's "one forward pass yields all horizons" surface:
        a single rollout produces a decoded frame, sigma, and
        reward/terminal/risk at every configured horizon, not a separate
        forward pass per horizon.
        """
        horizons = sorted({int(h) for h in (horizon_frames if horizon_frames is not None else self.horizons_ticks)})
        if not horizons or horizons[0] < 1:
            raise ValueError(f"horizons must be positive frame steps, got {horizons!r}")
        max_horizon = horizons[-1]
        if actions.shape[1] < max_horizon:
            raise ValueError(
                f"actions must cover at least {max_horizon} steps for horizon "
                f"{max_horizon}, got {actions.shape[1]}"
            )

        predictions: Dict[int, CortexHorizonPrediction] = {}
        latent = start_latent
        wanted = set(horizons)
        for step in range(max_horizon):
            latent, hidden = self.step(latent, actions[:, step], hidden)
            h = step + 1
            if h in wanted:
                reward, terminal_logit, risk, uncertainty = self.heads(hidden)
                predictions[h] = CortexHorizonPrediction(
                    latent=latent,
                    decoded=self.decoder(latent),
                    reward=reward,
                    terminal_logit=terminal_logit,
                    risk=risk,
                    uncertainty=uncertainty,
                )
        return CortexRolloutOutput(horizons=predictions)

    def checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "pixel_shape": list(self.pixel_shape),
            "action_keys": list(self.action_keys),
            "latent_width": self.latent_width,
            "hidden_dim": self.hidden_dim,
            "reconstruction_shape": list(self.reconstruction_shape),
            "horizons_ticks": list(self.horizons_ticks),
        }


def build_predictive_cortex(
    pixel_shape: Tuple[int, int, int],
    action_keys: Sequence[str],
    config: Optional[PredictiveCortexConfig] = None,
) -> PredictiveCortex:
    """Construct a :class:`PredictiveCortex` (requires torch)."""
    return PredictiveCortex(pixel_shape, action_keys, config)


#: Re-export shim (task 1): code written against the pre-promotion name
#: keeps working -- ``ActionConditionedWorldModel`` was the local class
#: name inside ``training.action_world_model.build_action_world_model``.
ActionConditionedWorldModel = PredictiveCortex
