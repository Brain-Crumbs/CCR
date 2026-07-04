"""Policy interface: decide what to emit (including nothing) each cognitive tick.

Loop v2 contract: `emit(state, memory, prediction) -> list[Action]` — the
motor emissions for this cognitive tick, published as `motor.command`
events.  **An empty list is NULL**: an explicit, recorded decision to do
nothing.

`SingleActionPolicy` adapts the classic one-`Action`-per-tick policies to the
new contract: subclasses keep implementing `decide(...) -> Action` and
`NULL_ACTION` maps to `[]`.
"""

from __future__ import annotations

import abc
from typing import List, Optional

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.world_model import Prediction


class Policy(abc.ABC):
    name: str = "policy"
    # Interactive policies (e.g. human demonstrations) set this to end the
    # session gracefully; the runtime checks it every cognitive tick.
    stop_requested: bool = False

    @abc.abstractmethod
    def emit(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> List[Action]:
        """Motor emissions for this cognitive tick.  `[]` is a real (NULL) choice."""

    def reset(self) -> None:
        """Called at the start of each episode."""


class SingleActionPolicy(Policy):
    """Adapter for policies that choose exactly one `Action` per tick."""

    @abc.abstractmethod
    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        """Choose one action.  Returning `NULL_ACTION` maps to an empty emission."""

    def emit(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> List[Action]:
        action = self.decide(state, memory, prediction)
        if action is None or action.is_null:
            return []
        return [action]
