"""Predictive-cortex world-model bridge: drive a trained recurrent,
action-conditioned :class:`~brain.cortex.predictive.PredictiveCortex` as the
loop's *live* ``WorldModel`` (issue #166).

Until now the cortex was only trained/evaluated offline; the live runtime
predicted with the trivial ``TrendWorldModel`` or the memoryless
``MLPWorldModel``.  This adapter closes that gap behind the same
``core.world_model.WorldModel`` seam every policy already reads through, so no
loop or policy change is needed to switch the live world model over to the
recurrent one.

Unlike the memoryless bridges, the cortex carries a *world state* -- the
backbone hidden state -- that must persist across cognitive ticks and reset on
an episode boundary.  The loop already calls ``world_model.reset()`` at the
start of every episode, so the rolling state lives entirely inside this adapter
(no loop plumbing).

Each tick:

- encode the live ``vision.frame.pixels`` frame with the cortex's own encoder
  into a latent (the cortex's own visual pathway, not the fused latent the
  memoryless bridge reads);
- score *this* tick's prediction error as how wrong last tick's forecast of
  this latent was -- the genuine surprise/novelty signal, computed from the
  cortex's own closed-loop forecast rather than a self-reported estimate;
- advance the world state by one real ``(latent, last-action)`` step and read
  the reward / terminal / risk heads off it, caching the one-step latent
  forecast for next tick's prediction-error comparison.

Like the other bridges it cannot condition on the action about to be taken
(``predict`` runs before the policy chooses), so the rollout repeats the last
action the runtime emitted -- the steady-state "if we keep doing what we just
did" assumption a curiosity/novelty consumer has available this tick.

Imports torch (via the cortex), so the CLI imports it lazily; the rest of the
runtime stays torch-free.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.world_model import Prediction
from cognitive_runtime.core.world_model import WorldModel as CoreWorldModel
from cognitive_runtime.neural.pixel_stream_encoder import PIXEL_STREAM_ID


class CortexWorldModel(CoreWorldModel):
    """Bridges a trained ``PredictiveCortex`` into the loop's ``WorldModel``
    seam, holding the recurrent hidden state across ticks within an episode."""

    def __init__(
        self,
        model: Union["PredictiveCortex", str],  # noqa: F821 -- lazy torch type
        action_keys: Optional[Sequence[str]] = None,
        horizons: Optional[Sequence[int]] = None,
    ):
        if isinstance(model, str):
            from cognitive_runtime.training.action_world_model import (
                load_action_world_model,
            )

            model, _stats = load_action_world_model(model)
        self.model = model
        self.model.eval()

        keys = list(action_keys) if action_keys is not None else list(model.action_keys)
        if not keys:
            raise ValueError(
                "CortexWorldModel needs action_keys (the checkpoint carried none); "
                "pass the program's ordered action space explicitly"
            )
        self.action_keys = keys
        self._action_index = {key: i for i, key in enumerate(self.action_keys)}

        # Forecast horizons in frame steps. The live loop advances one frame per
        # cognitive tick, so the checkpoint's tick-space horizons double as frame
        # steps here (the common ~1-tick-per-frame case), matching
        # ``PredictiveCortex.forward_horizons``' own default.
        picked = horizons if horizons is not None else model.horizons_ticks
        self.horizons = sorted({int(h) for h in picked})
        if not self.horizons or self.horizons[0] < 1:
            raise ValueError(f"horizons must be positive frame steps, got {picked!r}")

        # Rolling world state (reset on episode boundary):
        self._hidden = None  # backbone hidden state, opaque to this adapter
        # The cortex's one-step latent forecast for the *current* tick, made
        # last tick -- compared against the actually-observed latent to score
        # this tick's prediction error. ``None`` at episode start.
        self._predicted_latent: Optional[torch.Tensor] = None
        # The most recently encoded observation latent and the hidden state
        # *before* predict()'s one-step advance — exposed for the cortex MPC
        # predictor (issue #168), which evaluates each candidate action from
        # the same pre-advance starting point.
        self._latent: Optional[torch.Tensor] = None
        self._pre_advance_hidden = None

    def _workspace_modalities(self, memory: Memory) -> dict[str, torch.Tensor]:
        """Bind the runtime's actual fused workspace plus efference copy."""
        if not self.model.workspace_modalities:
            return {}
        latent = memory.fused_latent()
        if latent is None:
            raise ValueError("fused workspace is required by this cortex checkpoint")
        if self.model.workspace_layout_hash != latent.layout_hash:
            raise ValueError(
                "fused workspace layout mismatch: cortex expects "
                f"{self.model.workspace_layout_hash!r}, runtime has {latent.layout_hash!r}"
            )
        values: dict[str, torch.Tensor] = {}
        if "workspace" in self.model.workspace_modalities:
            values["workspace"] = torch.tensor([latent.vector], dtype=torch.float32)
        if "efference" in self.model.workspace_modalities:
            action = self._last_action_column(memory)
            one_hot = torch.zeros(1, len(self.model.action_keys), dtype=torch.float32)
            one_hot.scatter_(1, action.unsqueeze(1), 1.0)
            values["efference"] = one_hot
        return values

    def reset(self) -> None:
        self._hidden = None
        self._predicted_latent = None
        self._latent = None
        self._pre_advance_hidden = None

    def _last_action_column(self, memory: Memory) -> torch.Tensor:
        """The last emitted action as a ``Tensor[1]`` index column (0 -- the
        first action -- when nothing has been emitted yet, matching the
        all-zero action one-hot the memoryless bridges fall back to)."""
        index = 0
        last_actions = memory.last_actions(1)
        if last_actions:
            found = self._action_index.get(last_actions[-1].key())
            if found is not None:
                index = found
        return torch.tensor([index], dtype=torch.long)

    def predict(self, state: State, memory: Memory) -> Prediction:
        latest_pixels = memory.buffer.latest(PIXEL_STREAM_ID)
        if latest_pixels is None:
            # No pixel frame yet (first tick of an episode before the program
            # publishes): no learned signal, like the memoryless bridge.
            return Prediction()

        frame = latest_pixels.payload
        shape = tuple(frame.shape) if isinstance(frame, np.ndarray) else None
        if shape is not None and shape != tuple(self.model.pixel_shape):
            raise ValueError(
                f"pixel-frame shape {shape} != cortex's {tuple(self.model.pixel_shape)}; "
                "re-train or align the render geometry"
            )

        with torch.no_grad():
            # ``encode_frame`` is intentionally not used as the token: C2's
            # token is the bound vision + TemporalFusion workspace state.
            from cognitive_runtime.neural.pixel_stream_encoder import pixels_to_chw
            pixel_batch = pixels_to_chw(frame).unsqueeze(0).to(next(self.model.parameters()).device)
            latent = self.model.encode_workspace(pixel_batch, self._workspace_modalities(memory))

            if self._hidden is None:
                self._hidden = self.model.initial_state(1)

            # This tick's prediction error: how wrong last tick's forecast of
            # *this* latent turned out to be. ``None`` on the first tick, when
            # there is no prior forecast to score against -- the same "no
            # learned signal yet" fall-back the heuristic model uses.
            prediction_error: Optional[float] = None
            if self._predicted_latent is not None:
                prediction_error = float(F.mse_loss(latent, self._predicted_latent))

            # Snapshot for cortex MPC (issue #168): the encoded observation
            # and the hidden state *before* the one-step advance below.
            self._latent = latent
            self._pre_advance_hidden = self._hidden

            # Closed-loop rollout from the current world state, repeating the
            # last action across every horizon (steady-state assumption). The
            # first step is the *real* advance whose hidden state and latent
            # forecast we persist; further steps are what-if look-ahead that
            # must not corrupt the rolling state.
            action_col = self._last_action_column(memory)
            hidden = self._hidden
            latent_i = latent
            first_hidden = None
            first_latent = None
            for step_i in range(self.horizons[-1]):
                latent_i, hidden = self.model.step(latent_i, action_col, hidden)
                if step_i == 0:
                    first_hidden, first_latent = hidden, latent_i

            # Persist exactly one real step of world state.
            self._hidden = first_hidden
            self._predicted_latent = first_latent

            reward, terminal_logit, risk, uncertainty = self.model.heads(first_hidden)

        return Prediction(
            # ``risk_head`` is softplus'd (non-negative, unbounded); clamp into
            # the heuristic model's 0..1 range the reflex/veto thresholds expect.
            risk=float(torch.clamp(risk, 0.0, 1.0)),
            p_death=float(torch.sigmoid(terminal_logit)),
            predicted_reward=float(reward),
            next_latent=first_latent.squeeze(0).tolist(),
            prediction_error=prediction_error,
            # The cortex's own forward-uncertainty head (issue #169), read
            # off the same real one-step advance as the other heads above --
            # the dedicated sigma the arbiter's surprise calibration reads
            # in preference to the prediction_error stand-in.
            predicted_uncertainty=float(uncertainty),
        )
