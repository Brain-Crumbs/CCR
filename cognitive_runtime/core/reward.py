"""Reward signal produced by a Program each tick."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class RewardSignal:
    value: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    events: Tuple[str, ...] = ()
    #: Two-scale rewards (issue #41): `value`/`components` are the raw,
    #: unclipped magnitudes for logging/dashboards.  `training_value` is the
    #: normalized/clipped scalar an optimizer should actually see -- `None`
    #: when the reward source has no normalization stage, in which case
    #: consumers fall back to `value`.
    training_value: Optional[float] = None

    @staticmethod
    def from_components(
        components: Dict[str, float],
        events: Tuple[str, ...] = (),
        training_value: Optional[float] = None,
    ) -> "RewardSignal":
        return RewardSignal(
            value=round(sum(components.values()), 6),
            components={k: round(v, 6) for k, v in components.items() if v != 0.0},
            events=events,
            training_value=round(training_value, 6) if training_value is not None else None,
        )
