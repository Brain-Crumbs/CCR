"""Action-burst policy: sample an action uniformly from a fixed subset and
hold it for a uniformly sampled number of ticks before resampling.

Motor babbling (epic #212 §12.2, issue #235): resampling a new random action
every tick produces a near-stationary agent whose position barely
decorrelates from one tick to the next. Holding each sampled action for a
short burst instead is what creates the multi-tick displacement ego-motion
learning needs. Shared by ``training.nursery``'s ``motor_babbling_open`` and
(per the epic doc) the future ``motor_babbling_walls``.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import SingleActionPolicy
from cognitive_runtime.core.world_model import Prediction


class ActionBurstPolicy(SingleActionPolicy):
    """Uniformly sample an action from ``action_space``, hold it for a
    uniformly sampled ``[min_burst_ticks, max_burst_ticks]`` ticks, then
    resample -- deterministic given ``seed``."""

    name = "action-burst"

    def __init__(
        self,
        action_space: Sequence[Action],
        *,
        min_burst_ticks: int = 1,
        max_burst_ticks: int = 4,
        seed: int = 0,
    ):
        if not action_space:
            raise ValueError("action_space must not be empty")
        if min_burst_ticks < 1:
            raise ValueError(f"min_burst_ticks must be >= 1, got {min_burst_ticks!r}")
        if max_burst_ticks < min_burst_ticks:
            raise ValueError(
                f"max_burst_ticks ({max_burst_ticks!r}) must be >= "
                f"min_burst_ticks ({min_burst_ticks!r})"
            )
        self.action_space: List[Action] = list(action_space)
        self.min_burst_ticks = int(min_burst_ticks)
        self.max_burst_ticks = int(max_burst_ticks)
        self.seed = seed
        self.rng = random.Random(seed)
        self._current: Optional[Action] = None
        self._remaining = 0

    def reset(self) -> None:
        self.rng = random.Random(self.seed)
        self._current = None
        self._remaining = 0

    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        if self._remaining <= 0:
            self._current = self.rng.choice(self.action_space)
            self._remaining = self.rng.randint(self.min_burst_ticks, self.max_burst_ticks)
        self._remaining -= 1
        assert self._current is not None
        return self._current
