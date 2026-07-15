"""CrafterWorld configuration.

Much smaller than ``programs.minecraft.config.SurvivalBoxConfig``: Crafter
has no day/night-cycle knob, no difficulty/mob-count dial and no backend
choice (the ``crafter`` package *is* the backend).  ``episode_ticks`` reuses
the same field name the CLI's ``--episode-ticks`` flag already fills in
(``cli.py:_program_config``), so passing a Minecraft-shaped config dict here
is a safe no-op for everything Crafter doesn't understand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class CrafterConfig:
    episode_ticks: int = 10000        # crafter's own default episode length
    area: Tuple[int, int] = (64, 64)  # world grid size
    view: Tuple[int, int] = (9, 9)    # crafter's local render window, in cells
    size: Tuple[int, int] = (64, 64)  # rendered RGB pixel-frame resolution
    grid_radius: int = 4              # vision.frame.grid egocentric crop half-width
    # Realtime (--realtime) per-stream wall-clock pacing targets, matching
    # SurvivalBoxConfig's convention; ignored in fast-forward mode.
    realtime_vision_hz: float = 10.0
    realtime_body_heartbeat_hz: float = 2.0

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "CrafterConfig":
        cfg = CrafterConfig()
        for key, value in raw.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg
