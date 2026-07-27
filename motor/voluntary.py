"""Non-learning voluntary motor controllers.

The default controller performs one-step model-predictive control (MPC).  It
deliberately depends on tiny callable seams rather than on a particular cortex
implementation, which keeps actions World-defined and makes experimental
controllers interchangeable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from cognitive_runtime.core.action import NULL_ACTION, Action


class VoluntaryController(Protocol):
    """Common seam implemented by every voluntary controller."""

    name: str

    def choose(self, state: Any, actions: Sequence[Action], goal: Any = None) -> Action:
        """Choose one action, including the explicit ``NULL`` action."""


Predictor = Callable[[Any, Action], Any]
Scorer = Callable[[Any, Any], float]


@dataclass
class MPCController:
    """Deterministic one-step planning over a fixed predictive cortex.

    Ties retain action-space order.  Prediction runs under ``torch.no_grad``
    when torch is installed, ensuring the motor path never builds a gradient
    graph or updates the cortex.
    """

    predictor: Predictor
    scorer: Scorer
    name: str = "mpc"

    def choose(self, state: Any, actions: Sequence[Action], goal: Any = None) -> Action:
        if not actions:
            raise ValueError("voluntary action space must not be empty")

        def evaluate(action: Action) -> float:
            return float(self.scorer(self.predictor(state, action), goal))

        try:
            import torch
        except ImportError:
            scores = [evaluate(action) for action in actions]
        else:
            with torch.no_grad():
                scores = [evaluate(action) for action in actions]
        # Treat an invalid model score as the worst possible prediction.
        # This preserves deterministic action-space tie-breaking even when
        # every scorer result is NaN.
        safe_scores = [float("-inf") if math.isnan(score) else score for score in scores]
        return actions[max(range(len(actions)), key=safe_scores.__getitem__)]


@dataclass
class CallableController:
    """Adapter used by the active/imagination/policy A/B controllers."""

    name: str
    chooser: Callable[[Any, Sequence[Action], Any], Action]

    def choose(self, state: Any, actions: Sequence[Action], goal: Any = None) -> Action:
        action = self.chooser(state, actions, goal)
        # NULL is the universal explicit "do nothing" motor result. Policy
        # heads represent it as an empty emission, so it remains valid even
        # when a World offers only its actuated actions to this chooser.
        if action != NULL_ACTION and action not in actions:
            raise ValueError(f"{self.name} chose action outside the World action space: {action}")
        return action


def build_voluntary_controller(
    kind: str = "mpc",
    *,
    predictor: Predictor | None = None,
    scorer: Scorer | None = None,
    alternatives: Mapping[str, Callable[[Any, Sequence[Action], Any], Action]] | None = None,
) -> VoluntaryController:
    """Build one of ``mpc|active|imagination|policy``; MPC is the default."""
    if kind == "mpc":
        if predictor is None or scorer is None:
            raise ValueError("mpc requires predictor and scorer")
        return MPCController(predictor, scorer)
    if kind not in {"active", "imagination", "policy"}:
        raise ValueError(f"unknown voluntary controller {kind!r}")
    chooser = (alternatives or {}).get(kind)
    if chooser is None:
        raise ValueError(f"{kind} controller requires an alternative chooser")
    return CallableController(kind, chooser)
