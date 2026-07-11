"""Scripted action-sequence policy: cycle through fixed ``(action, duration)``
segments, looping back to the start once exhausted.

``ConstantActionPolicy`` only covers "repeat one action forever" scenarios
(``walk_forward``, ``turn_in_place``). The nursery suite's
``strafe_and_stop`` scenario (issue #62) needs alternating movement and
stillness -- this is the minimal scripted-sequence policy that covers it
and any other multi-phase scripted scenario.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import SingleActionPolicy
from cognitive_runtime.core.world_model import Prediction


class ScriptedSequencePolicy(SingleActionPolicy):
    """Emit ``segments[i].action`` for ``segments[i].duration`` ticks each,
    in order, looping back to ``segments[0]`` once the cycle completes."""

    name = "scripted-sequence"

    def __init__(self, segments: Sequence[Tuple[Action, int]]):
        segments = list(segments)
        if not segments:
            raise ValueError("segments must be non-empty")
        for action, duration in segments:
            if duration <= 0:
                raise ValueError(f"segment duration must be positive, got {duration!r} for {action!r}")
        self.segments: List[Tuple[Action, int]] = segments
        self._cycle_length = sum(duration for _, duration in segments)
        self._tick = 0

    def reset(self) -> None:
        self._tick = 0

    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        offset = self._tick % self._cycle_length
        self._tick += 1
        for action, duration in self.segments:
            if offset < duration:
                return action
            offset -= duration
        raise AssertionError("unreachable: offset exceeded cycle length")
