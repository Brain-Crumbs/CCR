"""Reward signal produced by a Program each tick."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class RewardSignal:
    value: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    events: Tuple[str, ...] = ()

    @staticmethod
    def from_components(components: Dict[str, float], events: Tuple[str, ...] = ()) -> "RewardSignal":
        return RewardSignal(
            value=round(sum(components.values()), 6),
            components={k: round(v, 6) for k, v in components.items() if v != 0.0},
            events=events,
        )
