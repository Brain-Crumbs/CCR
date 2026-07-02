"""Null policy: always do nothing.

Verifies runtime stability and establishes the passive survival baseline.
"""

from __future__ import annotations

from typing import Optional

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import Policy
from cognitive_runtime.core.world_model import Prediction


class NullPolicy(Policy):
    name = "null"

    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        return NULL_ACTION
