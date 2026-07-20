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

**Temporal backbone as an A/B choice (issue #93, task 5).** The GRU
transition is one ``brain.cortex.backbones.TemporalBackbone`` among several
selectable via ``PredictiveCortexConfig.backbone``: a dilated temporal-conv
(WaveNet-style) or a small transformer, both processing a window of recent
``(latent, action)`` pairs in one parallel pass instead of the GRU's
one-step-at-a-time recurrence. All three backbones implement the same
``initial_state``/``step``/``readout`` contract (``brain.cortex.backbones``),
so :meth:`PredictiveCortex.step`, :meth:`rollout`, and
:meth:`forward_horizons` are backbone-agnostic -- swapping the backbone is a
config change, not a fork.
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
    #: Transition backbone (issue #93): ``"gru"`` (default, unbounded
    #: recurrent context), ``"dilated_conv"`` (WaveNet-style causal dilated
    #: convolutions over a window), or ``"transformer"`` (causal
    #: self-attention over a window). See ``brain.cortex.backbones``.
    backbone: str = "gru"
    #: Window size (in frame steps) the windowed backbones
    #: (``dilated_conv``/``transformer``) attend over; ignored by ``"gru"``.
    #: Persisted with the checkpoint so a reloaded model's window matches
    #: what it was trained with.
    context_length: int = 8
    #: Extra backbone-specific constructor kwargs (e.g. ``kernel_size``,
    #: ``n_layers``, ``n_heads``); not persisted with the checkpoint.
    backbone_kwargs: Dict[str, Any] = field(default_factory=dict)
    #: Non-visual slices of the bound workspace token.  Vision is encoded by
    #: the CNN below; these are the fixed-layout vectors produced by
    #: ``TemporalFusion`` plus the efference-copy one-hot.  Empty preserves
    #: loading and using pre-C2 pixel-only cortexes.
    workspace_modalities: Dict[str, int] = field(default_factory=dict)
    #: Stream-layout identity for the ``workspace`` slice, persisted with the
    #: checkpoint so a live cortex cannot silently consume another program's
    #: fused state.
    workspace_layout_hash: Optional[str] = None


@dataclass(frozen=True)
class CortexHorizonPrediction:
    """One rollout horizon's visual and workspace predictions."""

    latent: torch.Tensor
    decoded: torch.Tensor
    reward: torch.Tensor
    terminal_logit: torch.Tensor
    risk: torch.Tensor
    uncertainty: torch.Tensor
    modalities: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class CortexRolloutOutput:
    """Predictions at every requested horizon, keyed by frame-step offset."""

    horizons: Dict[int, CortexHorizonPrediction]

    def __getitem__(self, horizon: int) -> CortexHorizonPrediction:
        return self.horizons[horizon]


@dataclass(frozen=True)
class CortexSequencePrediction:
    """Direct prediction at every causal sequence position for one horizon."""

    latent: torch.Tensor
    decoded: torch.Tensor
    reward: torch.Tensor
    terminal_logit: torch.Tensor
    risk: torch.Tensor
    uncertainty: torch.Tensor
    modalities: Dict[str, torch.Tensor] = field(default_factory=dict)


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
        from brain.cortex.backbones import build_backbone
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
        if any(int(width) <= 0 for width in cfg.workspace_modalities.values()):
            raise ValueError(f"workspace modality widths must be positive, got {cfg.workspace_modalities!r}")
        self.workspace_modalities = {
            str(name): int(width) for name, width in cfg.workspace_modalities.items()
        }
        self.workspace_layout_hash = cfg.workspace_layout_hash
        self.reconstruction_shape = _reconstruction_shape(
            self.pixel_shape, cfg.reconstruction_size
        )

        self.encoder = PixelStreamEncoder(self.pixel_shape, latent_width=cfg.latent_width)
        self.workspace_width = sum(self.workspace_modalities.values())
        # The token is a learned binding of vision with the stream-native
        # workspace slices, rather than the pixels-only encoder output.
        self.workspace_fuser = (
            nn.Linear(cfg.latent_width + self.workspace_width, cfg.latent_width)
            if self.workspace_width
            else None
        )
        self.action_embedding = nn.Embedding(len(self.action_keys), cfg.action_embed_dim)
        self.transition_backbone = build_backbone(
            cfg.backbone,
            cfg.latent_width + cfg.action_embed_dim,
            cfg.hidden_dim,
            context_length=cfg.context_length,
            **cfg.backbone_kwargs,
        )
        self.latent_head = nn.Linear(cfg.hidden_dim, cfg.latent_width)
        self.decoder = PixelReconstructionDecoder(
            cfg.latent_width, self.reconstruction_shape, hidden_dim=cfg.hidden_dim
        )
        self.workspace_decoders = nn.ModuleDict({
            name: nn.Linear(cfg.latent_width, width)
            for name, width in self.workspace_modalities.items()
        })

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
        # C1's direct multi-token heads predict farther future positions
        # without forcing a deep composition through the one-step head.
        self.multi_token_heads = nn.ModuleDict()
        for horizon in self.horizons_ticks:
            if horizon > 1:
                self.multi_token_heads[str(horizon)] = nn.ModuleDict({
                    "latent": nn.Linear(cfg.hidden_dim, cfg.latent_width),
                    "reward": nn.Linear(cfg.hidden_dim, 1),
                    "terminal": nn.Linear(cfg.hidden_dim, 1),
                    "risk": nn.Linear(cfg.hidden_dim, 1),
                    "uncertainty": nn.Linear(cfg.hidden_dim, 1),
                })

    def initial_state(self, batch: int) -> Any:
        return self.transition_backbone.initial_state(batch)

    def encode_workspace(
        self,
        pixels: torch.Tensor,
        modalities: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Bind visual and fused-workspace slices into one prediction token.

        ``modalities`` is keyed by the persisted workspace layout (normally
        ``workspace`` and ``efference``).  A C2 checkpoint rejects a missing
        or differently-shaped slice instead of falling back to pixels-only.
        """
        visual = self.encoder(pixels)
        if not self.workspace_modalities:
            return visual
        if modalities is None:
            raise ValueError("this cortex requires fused workspace modalities")
        pieces = [visual]
        for name, width in self.workspace_modalities.items():
            value = modalities.get(name)
            if value is None or value.ndim != 2 or value.shape != (visual.shape[0], width):
                got = None if value is None else tuple(value.shape)
                raise ValueError(
                    f"workspace modality {name!r} must be [{visual.shape[0]}, {width}], got {got}"
                )
            pieces.append(value.to(device=visual.device, dtype=visual.dtype))
        assert self.workspace_fuser is not None
        return self.workspace_fuser(torch.cat(pieces, dim=-1))

    def decode_workspace(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode the visual frame and every non-visual workspace modality."""
        if latent.ndim not in {2, 3}:
            raise ValueError(f"latent must be [B, L] or [B, T, L], got {tuple(latent.shape)}")
        leading = latent.shape[:-1]
        flat = latent.reshape(-1, latent.shape[-1])
        vision = self.decoder(flat).reshape(
            *leading, self.reconstruction_shape[2], self.reconstruction_shape[0],
            self.reconstruction_shape[1],
        )
        decoded: Dict[str, torch.Tensor] = {"vision": vision}
        for name, head in self.workspace_decoders.items():
            decoded[name] = head(flat).reshape(*leading, self.workspace_modalities[name])
        return decoded

    def set_context_length(self, n: Optional[int]) -> None:
        """Context-length curriculum hook (issue #93, task 5): a no-op for
        the ``"gru"`` backbone, else restricts the windowed backbone's
        attended-over window to the last ``n`` steps of its buffer."""
        self.transition_backbone.set_context_length(n)

    @property
    def context_length_max(self) -> Optional[int]:
        """``None`` for backbones with no fixed window (``"gru"``); else the
        window size the curriculum ramps up to."""
        return self.transition_backbone.context_length_max

    def step(
        self, latent: torch.Tensor, action_idx: torch.Tensor, hidden: Any
    ) -> Tuple[torch.Tensor, Any]:
        """Advance the world state by one ``(latent, action)`` pair; returns
        ``(predicted next latent, next hidden)``.

        ``hidden`` is an opaque state object owned by the configured
        backbone (a ``Tensor[batch, hidden_dim]`` for ``"gru"``; a
        ``(window buffer, last hidden)`` pair for the windowed backbones) --
        callers thread it back into the next ``step``/``heads`` call without
        inspecting it, exactly as before backbones were selectable.
        """
        embedded = self.action_embedding(action_idx)
        x = torch.cat([latent, embedded], dim=-1)
        hidden_repr, next_state = self.transition_backbone.step(x, hidden)
        return self.latent_head(hidden_repr), next_state

    def forward_sequence(self, latents: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """Causally encode every ``(z_t, a_t)`` prefix in parallel.

        ``latents`` is ``[B, T, latent_width]`` and ``action_idx`` is
        ``[B, T]``; each output position predicts a future token from that
        prefix.  The live ``step`` API remains separate for closed-loop use.
        """
        if latents.ndim != 3:
            raise ValueError(f"latents must be [B, T, L], got {tuple(latents.shape)}")
        if action_idx.shape != latents.shape[:2]:
            raise ValueError(
                "action_idx must be [B, T] matching latents, got "
                f"{tuple(action_idx.shape)} for {tuple(latents.shape)}"
            )
        embedded = self.action_embedding(action_idx)
        return self.transition_backbone.forward_sequence(torch.cat([latents, embedded], dim=-1))

    def sequence_prediction(self, hidden: torch.Tensor, horizon: int) -> CortexSequencePrediction:
        """Apply direct horizon heads to every sequence position."""
        if horizon < 1:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if horizon == 1:
            latent = self.latent_head(hidden)
            reward = self.reward_head(hidden).squeeze(-1)
            terminal_logit = self.terminal_head(hidden).squeeze(-1)
            risk = F.softplus(self.risk_head(hidden)).squeeze(-1)
            uncertainty = F.softplus(self.uncertainty_head(hidden)).squeeze(-1)
        else:
            try:
                heads = self.multi_token_heads[str(horizon)]
            except KeyError:
                raise ValueError(
                    f"horizon {horizon} has no direct head; configured horizons are "
                    f"{list(self.horizons_ticks)}"
                ) from None
            latent = heads["latent"](hidden)
            reward = heads["reward"](hidden).squeeze(-1)
            terminal_logit = heads["terminal"](hidden).squeeze(-1)
            risk = F.softplus(heads["risk"](hidden)).squeeze(-1)
            uncertainty = F.softplus(heads["uncertainty"](hidden)).squeeze(-1)
        if latent.ndim != 3:
            raise ValueError(f"sequence hidden must be [B, T, H], got {tuple(hidden.shape)}")
        decoded_workspace = self.decode_workspace(latent)
        return CortexSequencePrediction(
            latent=latent,
            decoded=decoded_workspace["vision"],
            reward=reward,
            terminal_logit=terminal_logit,
            risk=risk,
            uncertainty=uncertainty,
            modalities={name: value for name, value in decoded_workspace.items() if name != "vision"},
        )

    def heads(self, hidden: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reward / terminal-logit / risk / uncertainty read off one rollout
        step's world state; returns four ``Tensor[batch]``."""
        hidden_repr = self.transition_backbone.readout(hidden)
        return (
            self.reward_head(hidden_repr).squeeze(-1),
            self.terminal_head(hidden_repr).squeeze(-1),
            F.softplus(self.risk_head(hidden_repr)).squeeze(-1),
            F.softplus(self.uncertainty_head(hidden_repr)).squeeze(-1),
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
                decoded_workspace = self.decode_workspace(latent)
                predictions[h] = CortexHorizonPrediction(
                    latent=latent,
                    decoded=decoded_workspace["vision"],
                    reward=reward,
                    terminal_logit=terminal_logit,
                    risk=risk,
                    uncertainty=uncertainty,
                    modalities={
                        name: value for name, value in decoded_workspace.items()
                        if name != "vision"
                    },
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
            "backbone": self.config.backbone,
            "context_length": self.config.context_length,
            "workspace_modalities": dict(self.workspace_modalities),
            "workspace_layout_hash": self.workspace_layout_hash,
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
