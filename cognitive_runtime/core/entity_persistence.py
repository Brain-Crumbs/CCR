"""Entity-persistence probe seam: the loop's "does anything look surprising
about tracked entities right now" hook, the same shape as
``core.world_model.WorldModel`` (issue #27).

The MVP ships a null probe that never signals; a learned bridge
(``policies.neural_entity_persistence.NeuralEntityPersistence``) plugs in
behind this interface without touching the loop, the same way
``policies.neural_world_model.NeuralWorldModel`` plugs into
``core.world_model.WorldModel``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State


@dataclass
class EntityPersistencePrediction:
    #: The persistence model's own estimate of its error predicting the
    #: state of currently-occluded tracked entities this tick -- `None` when
    #: nothing is occluded (or no learned probe is wired in), the same
    #: optionality ``core.world_model.Prediction.prediction_error`` has.
    surprise: Optional[float] = None


class EntityPersistence(abc.ABC):
    @abc.abstractmethod
    def predict(self, state: State, memory: Memory) -> EntityPersistencePrediction:
        ...

    def reset(self) -> None:
        pass


class NullEntityPersistence(EntityPersistence):
    """Default: no entity tracking, no surprise signal."""

    def predict(self, state: State, memory: Memory) -> EntityPersistencePrediction:
        return EntityPersistencePrediction()
