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

    #: Named curriculum preset (issue #30), if any -- recorded into session
    #: metadata so dashboard comparisons can group runs by curriculum step.
    #: `None` for a plain (non-curriculum) run.
    curriculum: Optional[str] = None

    #: Ordered position of `curriculum` within its curriculum definition
    #: (issue #43's curriculum runner), so session metadata/episode summaries
    #: can be ordered chronologically by stage instead of just grouped by
    #: name. `None` for a plain run or a bare `--curriculum` preset run that
    #: isn't driven by the staged runner.
    curriculum_stage_index: Optional[int] = None

    # Rolling-window binary frame store (only used when a frame stream is
    # actually being recorded, i.e. record_frames=True or it's named in
    # record_streams).  A segment rotates on whichever threshold hits first;
    # rotation then reclaims disk by dropping the oldest *unpinned* segment
    # until the store is back under budget.
    frame_segment_max_mb: float = 32.0
    frame_segment_max_seconds: float = 60.0
    frame_disk_budget_mb: float = 512.0
    #: Streams that pin the frame store's current segment when they fire this
    #: tick, so the surrounding frames survive rotation (high-value moments:
    #: deaths, damage).  Glob patterns, e.g. ["event.died", "event.damage_taken"].
    pin_on_streams: List[str] = field(
        default_factory=lambda: ["event.died", "event.damage_taken"]
    )

    #: Bulky frame streams elided from the log unless ``record_frames`` is set.
    FRAME_STREAMS = ("vision.frame.grid", "vision.frame.pixels")

    #: Attention ablation (issue #59): ``"off"`` gives every agent-input
    #: stream uniform weight ``1.0`` (the pre-#59 behavior, byte-identical
    #: fused output); ``"budgeted"`` runs the deterministic
    #: ``AttentionController`` scoring under a hard budget. Defaults to
    #: ``"off"`` so existing sessions/checkpoints are unaffected unless a
    #: caller opts in.
    attention_mode: str = "off"

    def effective_exclude_streams(self) -> List[str]:
        """exclude_streams plus the frame streams when frames are opted out."""
        excluded = list(self.exclude_streams)
        if not self.record_frames:
            for stream_id in self.FRAME_STREAMS:
                if stream_id not in excluded:
                    excluded.append(stream_id)
        return excluded

    def resolved_session_id(self, policy_name: str) -> str:
        if self.session_id:
            return self.session_id
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{policy_name}"
