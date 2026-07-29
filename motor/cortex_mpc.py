"""Cortex-backed receding-horizon model-predictive control.

The cortex predicts *conditional* futures: an action planner supplies a
candidate action sequence, and the model rolls its latent world state forward
through that sequence.  This controller uses deterministic beam search to
compare those imagined futures, then emits only the first action of the best
plan.  The next real observation re-primes the cortex and planning begins
again, which avoids committing to an imagined open-loop trajectory.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Sequence

from cognitive_runtime.core.action import Action
from development.definitions import CurriculumStageSpec
from motor.voluntary import VoluntaryController

if TYPE_CHECKING:
    import torch

    from cognitive_runtime.policies.cortex_world_model import CortexWorldModel


def _torch() -> Any:
    """Import Torch only when cortex MPC is actually asked to plan."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ModuleNotFoundError(
                "cortex MPC requires the optional 'neural' dependency; "
                "install cognitive-runtime[neural]"
            ) from exc
        raise
    return torch


def _repeat_state(state: Any, repeats: int) -> Any:
    """Repeat a temporal-backbone state along its batch dimension."""
    if isinstance(state, _torch().Tensor):
        return state.repeat_interleave(repeats, dim=0)
    if isinstance(state, tuple):
        return tuple(_repeat_state(part, repeats) for part in state)
    if isinstance(state, list):
        return [_repeat_state(part, repeats) for part in state]
    raise TypeError(f"unsupported cortex backbone state {type(state)!r}")


def _select_state(state: Any, indices: torch.Tensor) -> Any:
    """Select temporal-backbone batch rows without mutating the live state."""
    if isinstance(state, _torch().Tensor):
        return state.index_select(0, indices)
    if isinstance(state, tuple):
        return tuple(_select_state(part, indices) for part in state)
    if isinstance(state, list):
        return [_select_state(part, indices) for part in state]
    raise TypeError(f"unsupported cortex backbone state {type(state)!r}")


class CortexMPCController:
    """Deterministic beam-search MPC over a live ``CortexWorldModel``.

    The returned action is the first item of the highest-scoring imagined
    sequence.  Rewards are accumulated with discounting; risk and terminal
    probability are penalties, while uncertainty retains the existing small
    exploration bonus.  The controller never writes the cortex's persistent
    state -- only ``CortexWorldModel.predict`` advances the real world state.
    """

    name = "cortex-mpc"

    def __init__(
        self,
        cortex_wm: CortexWorldModel,
        *,
        planning_horizon: Optional[int] = None,
        beam_width: int = 16,
        discount: float = 0.95,
        novelty_weight: float = 0.1,
        risk_penalty: float = 1.0,
        terminal_penalty: float = 1.0,
    ):
        if planning_horizon is not None and planning_horizon < 1:
            raise ValueError("planning_horizon must be positive")
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        self.cortex_wm = cortex_wm
        self.planning_horizon = planning_horizon
        self.beam_width = beam_width
        self.discount = discount
        self.novelty_weight = novelty_weight
        self.risk_penalty = risk_penalty
        self.terminal_penalty = terminal_penalty
        self.last_plan: tuple[str, ...] = ()

    def _horizon(self) -> int:
        if self.planning_horizon is not None:
            return self.planning_horizon
        horizons = getattr(self.cortex_wm, "horizons", None)
        if horizons is None:
            horizons = self.cortex_wm.model.horizons_ticks
        return max(int(horizon) for horizon in horizons)

    @staticmethod
    def _best_indices(scores: torch.Tensor, count: int) -> torch.Tensor:
        """Stable score ordering, preserving action-space order on ties."""
        torch = _torch()
        values = scores.detach().cpu().tolist()
        ordered = sorted(
            range(len(values)),
            key=lambda index: (
                1 if math.isnan(values[index]) else 0,
                -values[index] if not math.isnan(values[index]) else 0.0,
                index,
            ),
        )
        return torch.tensor(ordered[:count], dtype=torch.long, device=scores.device)

    def choose(
        self,
        state: Any,
        actions: Sequence[Action],
        goal: Any = None,
    ) -> Action:
        del state, goal  # Goal-conditioned scoring is a future extension.
        if not actions:
            raise ValueError("voluntary action space must not be empty")
        torch = _torch()

        latent = self.cortex_wm._latent
        hidden = self.cortex_wm._pre_advance_hidden
        if latent is None or hidden is None:
            self.last_plan = ()
            return actions[0]

        action_indices = [self.cortex_wm._action_index.get(action.key(), 0) for action in actions]
        action_count = len(actions)
        # Keep at least one beam rooted at every possible immediate action:
        # otherwise a narrow beam can turn the first step back into a greedy
        # choice before delayed rewards have a chance to appear.
        width = max(self.beam_width, action_count)
        current_latent = latent
        current_hidden = hidden
        scores = torch.zeros(1, dtype=latent.dtype, device=latent.device)
        first_actions = torch.zeros(1, dtype=torch.long, device=latent.device)
        plans: list[tuple[str, ...]] = [()]

        with torch.no_grad():
            for step in range(self._horizon()):
                parent_count = current_latent.shape[0]
                expanded_latent = current_latent.repeat_interleave(action_count, dim=0)
                expanded_hidden = _repeat_state(current_hidden, action_count)
                expanded_actions = torch.tensor(
                    action_indices * parent_count,
                    dtype=torch.long,
                    device=latent.device,
                )
                next_latent, next_hidden = self.cortex_wm.model.step(
                    expanded_latent, expanded_actions, expanded_hidden
                )
                reward, terminal_logit, risk, uncertainty = self.cortex_wm.model.heads(
                    next_hidden
                )
                step_value = (
                    reward
                    + self.novelty_weight * uncertainty
                    - self.risk_penalty * risk
                    - self.terminal_penalty * torch.sigmoid(terminal_logit)
                )
                expanded_scores = scores.repeat_interleave(action_count) + (
                    self.discount ** step
                ) * step_value
                expanded_plans = [
                    plan + (action.key(),)
                    for plan in plans
                    for action in actions
                ]
                expanded_first = (
                    torch.arange(action_count, device=latent.device)
                    if step == 0
                    else first_actions.repeat_interleave(action_count)
                )
                keep = self._best_indices(
                    expanded_scores, min(width, expanded_scores.numel())
                )
                current_latent = next_latent.index_select(0, keep)
                current_hidden = _select_state(next_hidden, keep)
                scores = expanded_scores.index_select(0, keep)
                first_actions = expanded_first.index_select(0, keep)
                plans = [expanded_plans[index] for index in keep.detach().cpu().tolist()]

        self.last_plan = plans[0]
        return actions[int(first_actions[0])]


def build_cortex_mpc(
    cortex_wm: CortexWorldModel,
    *,
    planning_horizon: Optional[int] = None,
    beam_width: int = 16,
    discount: float = 0.95,
    novelty_weight: float = 0.1,
    risk_penalty: float = 1.0,
    terminal_penalty: float = 1.0,
) -> CortexMPCController:
    """Build a receding-horizon controller backed by the live cortex.

    ``planning_horizon`` defaults to the furthest configured cortex horizon.
    Each call performs deterministic beam search over action sequences, scores
    discounted reward minus terminal/risk cost (plus optional novelty), and
    returns only the first action from the best sequence.
    """
    return CortexMPCController(
        cortex_wm,
        planning_horizon=planning_horizon,
        beam_width=beam_width,
        discount=discount,
        novelty_weight=novelty_weight,
        risk_penalty=risk_penalty,
        terminal_penalty=terminal_penalty,
    )


def cortex_mpc_factory(
    cortex_wm: CortexWorldModel,
    **planning_kwargs: Any,
) -> "VoluntaryControllerFactory":
    """Return a curriculum factory that builds cortex MPC controllers.

    The factory carries the same world model so the runtime can prime it each
    tick before the voluntary controller plans from its latest observation.
    """

    def factory(
        stage: CurriculumStageSpec, action_space: Sequence[Action]
    ) -> VoluntaryController:
        del stage, action_space
        return build_cortex_mpc(cortex_wm, **planning_kwargs)

    factory.world_model = cortex_wm
    return factory
