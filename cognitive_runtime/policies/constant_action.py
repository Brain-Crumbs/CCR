"""Constant-action policy: repeat one action every tick, with optional noise.

The scripted, brainstem-simplest baseline the nursery scenarios (issue #62)
and the ego-motion canary (issue #39) need: "walk forward" is
``ConstantActionPolicy(Action("MOVE_FORWARD"))``, no survival logic at all.
"""

from __future__ import annotations

import random
from typing import List, Optional

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import SingleActionPolicy
from cognitive_runtime.core.world_model import Prediction


class ConstantActionPolicy(SingleActionPolicy):
    """Emit ``action`` every tick; with probability ``noise`` emit a random
    action from ``action_space`` instead (``action_space`` is required
    whenever ``noise > 0``).

    Used to generate the ego-motion canary's constant-``MOVE_FORWARD``
    episodes: "predict the next frame while walking forward" is the
    simplest ego-motion regularity a world model can be checked against,
    and action noise (issue #39's "optional action noise") lets the canary
    also probe robustness to occasional non-forward ticks.
    """

    name = "constant"

    def __init__(
        self,
        action: Action,
        *,
        noise: float = 0.0,
        action_space: Optional[List[Action]] = None,
        seed: int = 0,
    ):
        if not 0.0 <= noise <= 1.0:
            raise ValueError(f"noise must be in [0, 1], got {noise!r}")
        if noise > 0.0 and not action_space:
            raise ValueError("action_space is required when noise > 0")
        self.action = action
        self.noise = float(noise)
        self.action_space = list(action_space) if action_space else None
        self.seed = seed
        self.rng = random.Random(seed) 

    def reset(self) -> None:
        self.rng = random.Random(self.seed)

    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        if self.noise > 0.0 and self.rng.random() < self.noise:
            assert self.action_space is not None
            return self.rng.choice(self.action_space)
        return self.action
