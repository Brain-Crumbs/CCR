"""Runtime configuration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RuntimeConfig:
    tick_rate: float = 20.0            # target ticks per second
    realtime: bool = False             # False: fast-forward (no sleeping)
    max_ticks_per_episode: int = 6000  # 5 minutes at 20 tps
    episodes: int = 1
    seed: int = 0                      # episode i uses seed + i
    record: bool = True
    record_dir: str = "sessions"
    record_frames: bool = False        # frames are bulky; opt in (elided otherwise)
    session_id: Optional[str] = None
    memory_capacity: int = 512
    # Cognitive ticks can run slower than program ticks: the loop steps the
    # program this many times per cognitive tick (Phase 2, default 1).
    program_ticks_per_cognitive_tick: int = 1
    program_config: Dict[str, Any] = field(default_factory=dict)
    # Streams-v2 recording size control: which sensory streams keep their full
    # payload in the log.  Streams matched by exclude_streams (or not matched by
    # record_streams) are written as hash-only lines so replay verification
    # stays complete even when payloads are elided.  Globs, e.g. ["vision.*"].
    record_streams: List[str] = field(default_factory=lambda: ["*"])
    exclude_streams: List[str] = field(default_factory=list)

    def effective_exclude_streams(self) -> List[str]:
        """exclude_streams plus the frame stream when frames are opted out."""
        excluded = list(self.exclude_streams)
        if not self.record_frames and "vision.frame.grid" not in excluded:
            excluded.append("vision.frame.grid")
        return excluded

    def resolved_session_id(self, policy_name: str) -> str:
        if self.session_id:
            return self.session_id
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{policy_name}"
