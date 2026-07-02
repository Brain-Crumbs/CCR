"""Short-term runtime memory.

Keeps a bounded window of recent states, actions and observation hashes and
exposes generic signals (novelty, repetition, movement) that policies and
world models may use.  Contains no environment-specific logic.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Set

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.perception import State


class Memory:
    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.states: Deque[State] = deque(maxlen=capacity)
        self.actions: Deque[Action] = deque(maxlen=capacity)
        self.observation_hashes: Deque[str] = deque(maxlen=capacity)
        self._seen_hashes: Set[str] = set()
        self._novel_last_update = True

    def reset(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.observation_hashes.clear()
        self._seen_hashes.clear()
        self._novel_last_update = True

    def update(self, state: State) -> None:
        self.states.append(state)
        obs_hash = state.observation.hash()
        self._novel_last_update = obs_hash not in self._seen_hashes
        self._seen_hashes.add(obs_hash)
        self.observation_hashes.append(obs_hash)

    def record_action(self, action: Action) -> None:
        self.actions.append(action)

    @property
    def last_state(self) -> Optional[State]:
        return self.states[-1] if self.states else None

    def last_actions(self, n: int) -> List[Action]:
        return list(self.actions)[-n:]

    def repeated_action_streak(self) -> int:
        """Length of the trailing run of identical actions."""
        streak = 0
        last = None
        for action in reversed(self.actions):
            if last is None:
                last = action
            if action != last:
                break
            streak += 1
        return streak

    def novelty_rate(self, window: int = 64) -> float:
        """Fraction of unique observation hashes in the recent window."""
        recent = list(self.observation_hashes)[-window:]
        if not recent:
            return 1.0
        return len(set(recent)) / len(recent)

    @property
    def last_observation_was_novel(self) -> bool:
        return self._novel_last_update

    def feature_trend(self, name: str, window: int = 16) -> float:
        """Simple slope estimate (per tick) of a numeric feature."""
        values = [s.features.get(name) for s in list(self.states)[-window:]]
        values = [v for v in values if v is not None]
        if len(values) < 2:
            return 0.0
        return (values[-1] - values[0]) / (len(values) - 1)
