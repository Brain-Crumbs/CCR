"""Runtime configuration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RuntimeConfig:
    tick_rate: float = 20.0            # target ticks per second
    realtime: bool = False             # False: fast-forward (no sleeping)
    max_ticks_per_episode: int = 6000  # 5 minutes at 20 tps
    episodes: int = 1
    seed: int = 0                      # episode i uses seed + i
    record: bool = True
    record_dir: str = "sessions"
    record_observations: bool = True   # store full structured observations
    record_frames: bool = False        # frames are bulky; opt in
    session_id: Optional[str] = None
    memory_capacity: int = 512
    # Cognitive ticks can run slower than program ticks: the loop steps the
    # program this many times per cognitive tick (Phase 2, default 1).
    program_ticks_per_cognitive_tick: int = 1
    program_config: Dict[str, Any] = field(default_factory=dict)

    def resolved_session_id(self, policy_name: str) -> str:
        if self.session_id:
            return self.session_id
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{policy_name}"
