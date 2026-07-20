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
from typing import Dict, List, Optional

from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State

#: Streams treated as vitals for the generic risk heuristic.  Generic ids,
#: not environment fields: any Program publishing these gets risk sensing.
VITAL_STREAMS: List[str] = ["body.health", "body.hunger", "body.oxygen"]


@dataclass
class Prediction:
    expected_features: Dict[str, float] = field(default_factory=dict)
    risk: float = 0.0  # heuristic 0..1: how quickly things are getting worse
    # Additive Phase-D fields (issue #26): populated by a learned WorldModel
    # bridge, `None` for the heuristic `TrendWorldModel` and any older caller
    # that never set them.  Kept optional so `TrendWorldModel`/`Prediction()`
    # callers throughout the codebase (and every recorded session so far)
    # stay valid without touching the loop or existing policies.
    p_death: Optional[float] = None
    predicted_reward: Optional[float] = None
    next_latent: Optional[List[float]] = None
    prediction_error: Optional[float] = None
    #: A dedicated forward-uncertainty estimate (issue #169), e.g. the
    #: Predictive Cortex's ``uncertainty_head`` -- distinct from
    #: ``prediction_error`` (this tick's *realized* error) in that it is
    #: predicted *before* the outcome is known. ``None`` for any bridge
    #: without a trained uncertainty head (the heuristic ``TrendWorldModel``,
    #: the legacy ``MLPWorldModel`` bridge), in which case consumers fall
    #: back to ``prediction_error`` as the closest available stand-in.
    predicted_uncertainty: Optional[float] = None


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
