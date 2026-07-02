"""World model: predict future state from current state and memory.

The MVP ships a trivial trend extrapolator.  The interface exists so a
learned dynamics model can replace it without touching the loop.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict

from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State


@dataclass
class Prediction:
    expected_features: Dict[str, float] = field(default_factory=dict)
    risk: float = 0.0  # heuristic 0..1: how quickly things are getting worse


class WorldModel(abc.ABC):
    @abc.abstractmethod
    def predict(self, state: State, memory: Memory) -> Prediction:
        ...

    def reset(self) -> None:
        pass


class TrendWorldModel(WorldModel):
    """Linear extrapolation of each numeric feature over a short horizon."""

    def __init__(self, horizon: int = 10, window: int = 16):
        self.horizon = horizon
        self.window = window

    def predict(self, state: State, memory: Memory) -> Prediction:
        expected: Dict[str, float] = {}
        risk = 0.0
        for name, value in state.features.items():
            slope = memory.feature_trend(name, self.window)
            expected[name] = value + slope * self.horizon
        # Generic risk heuristic: any feature that looks like a vital
        # ("health", "hunger", "oxygen") and is trending down raises risk.
        for name, value in state.features.items():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("health", "hunger", "oxygen"):
                slope = memory.feature_trend(name, self.window)
                if slope < 0:
                    risk = min(1.0, risk + min(1.0, -slope * self.horizon / max(value, 1.0)))
        return Prediction(expected_features=expected, risk=risk)
