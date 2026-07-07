"""SurvivalBox configuration.

A constrained world: fixed seed, limited boundary, survival mode, daytime
start, controlled difficulty, short episodes.  The goal is not to make
Minecraft hard yet -- the goal is to make learning measurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SurvivalBoxConfig:
    world_size: int = 64            # bounded square world, walled at the edge
    episode_ticks: int = 6000       # 5 minutes at 20 ticks/sec
    day_length: int = 6000          # full day/night cycle in ticks
    start_time: int = 0             # 0 = dawn; night begins at day_length/2
    difficulty: float = 1.0         # scales mob spawn rate and damage
    max_mobs: int = 3
    start_near_resources: bool = True
    rare_events_disabled: bool = True  # reserved: no weather / raids in the sim
    # Realtime (--realtime) per-stream wall-clock pacing targets.  Ignored in
    # fast-forward mode, where publication maps onto tick cadence instead.
    realtime_vision_hz: float = 10.0          # vision frames paced to this rate
    realtime_body_heartbeat_hz: float = 2.0   # body vitals heartbeat rate

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "SurvivalBoxConfig":
        cfg = SurvivalBoxConfig()
        for key, value in raw.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg
