"""Random policy: sample uniformly from the Program's action space.

Tests action execution and establishes the lower-bound behavior baseline.
"""

from __future__ import annotations

import random
from typing import List, Optional

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import Policy
from cognitive_runtime.core.world_model import Prediction


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, action_space: List[Action], seed: int = 0):
        if not action_space:
            raise ValueError("action_space must not be empty")
        self.action_space = list(action_space)
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.rng = random.Random(self.seed)

    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        return self.rng.choice(self.action_space)
