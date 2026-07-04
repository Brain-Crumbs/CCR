"""World model: predict where things are heading from memory.

The MVP ships a trivial trend extrapolator.  Loop v2: it reads per-stream
numeric slopes from the :class:`Memory` `TemporalBuffer` instead of a
flattened feature dict.  Same semantics: vitals trending down ⇒ risk.  The
interface exists so a learned dynamics model can replace it without touching
the loop.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List

from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State

#: Streams treated as vitals for the generic risk heuristic.  Generic ids,
#: not environment fields: any Program publishing these gets risk sensing.
VITAL_STREAMS: List[str] = ["body.health", "body.hunger", "body.oxygen"]


@dataclass
class Prediction:
    expected_features: Dict[str, float] = field(default_factory=dict)
    risk: float = 0.0  # heuristic 0..1: how quickly things are getting worse


class WorldModel(abc.ABC):
    @abc.abstractmethod
    def predict(self, state: State, memory: Memory) -> Prediction:
        ...

    def reset(self) -> None:
        pass


class TrendWorldModel(WorldModel):
    """Linear extrapolation of each numeric vital stream over a short horizon."""

    def __init__(self, horizon: int = 10, window: int = 16,
                 vitals: List[str] | None = None):
        self.horizon = horizon
        self.window = window
        self.vitals = list(vitals) if vitals is not None else list(VITAL_STREAMS)

    def predict(self, state: State, memory: Memory) -> Prediction:
        expected: Dict[str, float] = {}
        risk = 0.0
        for stream_id in self.vitals:
            latest = memory.buffer.latest(stream_id)
            if latest is None or not isinstance(latest.payload, (int, float)):
                continue
            value = float(latest.payload)
            slope = memory.stream_trend(stream_id, self.window)
            expected[stream_id] = value + slope * self.horizon
            if slope < 0:
                risk = min(1.0, risk + min(1.0, -slope * self.horizon / max(value, 1.0)))
        return Prediction(expected_features=expected, risk=risk)
