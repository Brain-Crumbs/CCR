"""Policy interface: decide what to do (including nothing) each tick."""

from __future__ import annotations

import abc
from typing import Optional

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.world_model import Prediction


class Policy(abc.ABC):
    name: str = "policy"
    # Interactive policies (e.g. human demonstrations) set this to end the
    # session gracefully; the runtime checks it every tick.
    stop_requested: bool = False

    @abc.abstractmethod
    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        """Choose the next action.  Returning NULL_ACTION is a real choice."""

    def reset(self) -> None:
        """Called at the start of each episode."""
