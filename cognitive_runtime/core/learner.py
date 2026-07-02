"""Learner interface: consume (observation, action, reward) each tick.

The MVP learns offline (behavioral cloning from recordings), so the default
online learner only accumulates statistics.  The hook exists so online
learning can be added without changing the loop.
"""

from __future__ import annotations

import abc

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.reward import RewardSignal


class Learner(abc.ABC):
    @abc.abstractmethod
    def update(self, observation: Observation, action: Action, reward: RewardSignal) -> None:
        ...

    def reset(self) -> None:
        pass


class NullLearner(Learner):
    """Accumulates reward statistics; performs no learning."""

    def __init__(self) -> None:
        self.total_reward = 0.0
        self.ticks = 0

    def reset(self) -> None:
        self.total_reward = 0.0
        self.ticks = 0

    def update(self, observation: Observation, action: Action, reward: RewardSignal) -> None:
        self.total_reward += reward.value
        self.ticks += 1
