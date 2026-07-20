"""Cortex-backed MPC: the live voluntary controller (issue #168).

Wires a live ``CortexWorldModel``'s recurrent state into
``MPCController``'s predictor/scorer seams, realizing architecture
commitment 5: "voluntary action = one-step planning over the world model."

The predictor evaluates each candidate action from the same pre-advance
starting point (the encoded observation and backbone hidden state that
``CortexWorldModel.predict()`` snapshots before its own one-step
advance).  The scorer reads the cortex's reward/risk/uncertainty heads
off the resulting hidden and combines them into a single planning score:
``reward + novelty_weight * uncertainty``.

The cortex heads are untrained until B1 adds head losses to the
consolidation loop; until then the scorer plans over noise, but the
NaN-score guard (#149) keeps selection deterministic and the first real
action in action-space order wins ties.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from cognitive_runtime.core.action import Action
from cognitive_runtime.policies.cortex_world_model import CortexWorldModel
from development.definitions import CurriculumStageSpec
from motor.voluntary import MPCController, VoluntaryController


class _CortexPrediction:
    """Lightweight carrier from predictor to scorer: the backbone hidden
    state after one candidate-action step, so heads() can read it."""

    __slots__ = ("hidden",)

    def __init__(self, hidden: Any) -> None:
        self.hidden = hidden


def build_cortex_mpc(
    cortex_wm: CortexWorldModel,
    *,
    novelty_weight: float = 0.1,
) -> MPCController:
    """Build an ``MPCController`` whose predictor rolls the cortex one step
    per candidate action and whose scorer reads the reward + novelty heads.

    ``cortex_wm`` must be the same ``CortexWorldModel`` the loop's
    ``world_model`` slot holds, so its ``_latent`` / ``_pre_advance_hidden``
    snapshots (set each tick by ``predict()``) are fresh when the policy's
    ``emit()`` calls ``choose()``.

    ``novelty_weight`` scales the uncertainty head's contribution to the
    planning score (exploration bonus); the scorer seam is replaceable.
    """
    action_index = cortex_wm._action_index

    def predictor(state: Any, action: Action) -> _CortexPrediction:
        latent = cortex_wm._latent
        hidden = cortex_wm._pre_advance_hidden
        if latent is None or hidden is None:
            return _CortexPrediction(hidden=None)
        idx = action_index.get(action.key(), 0)
        action_col = torch.tensor([idx], dtype=torch.long, device=latent.device)
        _, next_hidden = cortex_wm.model.step(latent, action_col, hidden)
        return _CortexPrediction(next_hidden)

    def scorer(prediction: _CortexPrediction, goal: Any) -> float:
        if prediction.hidden is None:
            return float("nan")
        reward, _terminal, _risk, uncertainty = cortex_wm.model.heads(
            prediction.hidden
        )
        return float(reward) + novelty_weight * float(uncertainty)

    return MPCController(predictor, scorer, name="cortex-mpc")


def cortex_mpc_factory(
    cortex_wm: CortexWorldModel,
    *,
    novelty_weight: float = 0.1,
) -> "VoluntaryControllerFactory":
    """Return a ``VoluntaryControllerFactory`` (the hook ``run_curriculum``
    accepts) that builds a cortex-backed MPC for every ``learned`` stage.
    """

    def factory(
        stage: CurriculumStageSpec, action_space: Sequence[Action]
    ) -> VoluntaryController:
        return build_cortex_mpc(cortex_wm, novelty_weight=novelty_weight)

    return factory
