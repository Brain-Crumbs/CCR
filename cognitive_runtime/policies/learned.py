"""Learned policy: behavioral-cloning model over the latent state.

The model declares which representation it was trained on:

- ``latent`` (default) reads the fused :class:`LatentState` the runtime already
  computed into memory this tick, and refuses to run against an incompatible
  fusion layout (``layout_hash`` mismatch) rather than silently mis-predicting.
- ``handcrafted`` uses the Minecraft featurizer over the observation, for A/B
  comparison against the latent path.

Either way it keeps its own recent-action history so inference-time motor
features match training.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.world_model import Prediction
from cognitive_runtime.core.policy import SingleActionPolicy
from cognitive_runtime.training.features import featurize, latent_features
from cognitive_runtime.training.imitation import BCModel


class LearnedPolicy(SingleActionPolicy):
    name = "learned"

    def __init__(self, model: BCModel | str, history: int = 8):
        self.model = BCModel.load(model) if isinstance(model, str) else model
        self.history = history
        self._recent: Deque[str] = deque(maxlen=history)
        self.representation = self.model.meta.get("representation", "handcrafted")
        self._expected_layout = self.model.meta.get("layout_hash")

    def reset(self) -> None:
        self._recent.clear()

    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        if self.representation == "latent":
            features = self._latent_features(memory)
        else:
            features = featurize(state.observation.data, list(self._recent))
        key = self.model.predict_key(features)
        self._recent.append(key)
        return Action.from_key(key)

    def _latent_features(self, memory: Memory):
        latent = memory.fused_latent()
        if latent is None:
            raise RuntimeError(
                "learned policy trained on latent state, but the runtime produced "
                "no fused LatentState this tick"
            )
        if self._expected_layout is not None and latent.layout_hash != self._expected_layout:
            raise ValueError(
                "latent layout mismatch: model was trained on layout "
                f"{self._expected_layout} but the runtime produced "
                f"{latent.layout_hash}; re-train or align the stream catalog"
            )
        return latent_features(latent.vector, list(self._recent))
