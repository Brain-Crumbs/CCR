"""Learned policy: behavioral-cloning model over structured observations.

Uses the same featurizer as the trainer.  Keeps its own recent-action
history so the features it sees at inference match the ones it was trained
on.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import SingleActionPolicy
from cognitive_runtime.core.world_model import Prediction
from cognitive_runtime.training.features import featurize
from cognitive_runtime.training.imitation import BCModel


class LearnedPolicy(SingleActionPolicy):
    name = "learned"

    def __init__(self, model: BCModel | str, history: int = 8):
        self.model = BCModel.load(model) if isinstance(model, str) else model
        self.history = history
        self._recent: Deque[str] = deque(maxlen=history)

    def reset(self) -> None:
        self._recent.clear()

    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        features = featurize(state.observation.data, list(self._recent))
        key = self.model.predict_key(features)
        self._recent.append(key)
        return Action.from_key(key)
