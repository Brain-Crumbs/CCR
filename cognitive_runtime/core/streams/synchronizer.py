"""Cognitive tick boundaries over stream time.

The runtime never asks "what is the current observation?" — it asks
**"what streams have arrived since the last cognitive tick?"**  The
:class:`TickSynchronizer` answers that question by draining a bus into
:class:`TickWindow`s, and tracks per-stream arrival counts and silences for
runtime-health metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cognitive_runtime.core.streams.bus import StreamBus
from cognitive_runtime.core.streams.events import StreamEvent


@dataclass
class TickWindow:
    """Everything that arrived during one cognitive tick."""

    tick_index: int
    started_at: float
    ended_at: float
    events: List[StreamEvent] = field(default_factory=list)
    by_stream: Dict[str, List[StreamEvent]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.events


class TickSynchronizer:
    def __init__(self, program_ticks_per_cognitive_tick: int = 1):
        if program_ticks_per_cognitive_tick < 1:
            raise ValueError("program_ticks_per_cognitive_tick must be >= 1")
        #: Phase 2 uses this ratio to decouple cognitive rate from program
        #: rate (several program ticks may elapse per cognitive tick).
        self.program_ticks_per_cognitive_tick = program_ticks_per_cognitive_tick
        self._tick_index = 0
        self._last_ended_at = 0.0
        self._arrival_counts: Dict[str, int] = {}
        self._silent_windows: Dict[str, int] = {}

    def is_cognitive_tick_boundary(self, program_tick: int) -> bool:
        """True when `program_tick` completes a cognitive tick."""
        return (program_tick + 1) % self.program_ticks_per_cognitive_tick == 0

    def collect(self, bus: StreamBus, now: Optional[float] = None) -> TickWindow:
        """Drain the bus into the next cognitive tick window.

        `now` (simulated time) marks the window's end; when omitted it falls
        back to the latest event timestamp, or the window start if the
        window is empty.
        """
        events = bus.drain()  # deterministic (timestamp, stream_id, seq) order
        started_at = self._last_ended_at
        if now is not None:
            ended_at = now
        elif events:
            ended_at = max(e.timestamp for e in events)
        else:
            ended_at = started_at

        by_stream: Dict[str, List[StreamEvent]] = {}
        for event in events:
            by_stream.setdefault(event.stream_id, []).append(event)

        for stream_id, stream_events in by_stream.items():
            self._arrival_counts[stream_id] = (
                self._arrival_counts.get(stream_id, 0) + len(stream_events)
            )
            self._silent_windows[stream_id] = 0
        for stream_id in self._silent_windows:
            if stream_id not in by_stream:
                self._silent_windows[stream_id] += 1

        window = TickWindow(
            tick_index=self._tick_index,
            started_at=started_at,
            ended_at=ended_at,
            events=events,
            by_stream=by_stream,
        )
        self._tick_index += 1
        self._last_ended_at = ended_at
        return window

    # -- runtime-health metrics ----------------------------------------------

    def arrival_counts(self) -> Dict[str, int]:
        """Total events seen per stream since the last reset."""
        return dict(self._arrival_counts)

    def silent_streams(self, min_windows: int = 1) -> List[str]:
        """Streams that have been silent for at least `min_windows` windows."""
        return sorted(
            stream_id
            for stream_id, silent in self._silent_windows.items()
            if silent >= min_windows
        )

    def reset(self) -> None:
        self._tick_index = 0
        self._last_ended_at = 0.0
        self._arrival_counts.clear()
        self._silent_windows.clear()
