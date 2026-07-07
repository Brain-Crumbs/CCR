"""Cognitive tick boundaries over stream time.

The runtime never asks "what is the current observation?" — it asks
**"what streams have arrived since the last cognitive tick?"**  The
:class:`TickSynchronizer` answers that question by draining a bus into
:class:`TickWindow`s, and tracks per-stream arrival counts and silences for
runtime-health metrics.

Realtime health (Phase 5).  When windows arrive in wall-clock time the
synchronizer also accounts for missed-window conditions: **empty** windows
(nothing arrived), **stale** streams (a rate-bearing stream that has gone
quiet for longer than 2× its nominal period — a stopped publisher), and
per-stream **wall-clock rates** measured from the metadata ``arrived_at``
clock.  None of this touches the deterministic simulated-time path: the
simulated clock still drives windowing and hashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


from cognitive_runtime.core.streams.bus import StreamBus
from cognitive_runtime.core.streams.events import StreamEvent

#: A rate-bearing stream is "stale" once it has been quiet for this many
#: nominal periods — i.e. it has missed at least one expected heartbeat.
STALE_PERIOD_FACTOR = 2.0


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
    def __init__(
        self,
        program_ticks_per_cognitive_tick: int = 1,
        nominal_rates: Optional[Dict[str, float]] = None,
    ):
        if program_ticks_per_cognitive_tick < 1:
            raise ValueError("program_ticks_per_cognitive_tick must be >= 1")
        #: Phase 2 uses this ratio to decouple cognitive rate from program
        #: rate (several program ticks may elapse per cognitive tick).
        self.program_ticks_per_cognitive_tick = program_ticks_per_cognitive_tick
        #: stream_id -> nominal Hz, for stale-stream detection (rate-bearing
        #: streams only; irregular streams have no expectation of arrival).
        self._nominal_rates: Dict[str, float] = {
            sid: rate for sid, rate in (nominal_rates or {}).items() if rate
        }
        self._tick_index = 0
        self._last_ended_at = 0.0
        self._arrival_counts: Dict[str, int] = {}
        self._silent_windows: Dict[str, int] = {}
        self._empty_windows = 0
        #: sim-time of the most recent arrival per stream (for staleness).
        self._last_arrival_sim: Dict[str, float] = {}
        #: wall-clock (arrived_at) span + count per stream, for realtime rates.
        self._wall_first: Dict[str, float] = {}
        self._wall_last: Dict[str, float] = {}

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
            self._last_arrival_sim[stream_id] = ended_at
            self._record_wall_arrivals(stream_id, stream_events)
        for stream_id in self._silent_windows:
            if stream_id not in by_stream:
                self._silent_windows[stream_id] += 1

        if not events:
            self._empty_windows += 1

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

    def _record_wall_arrivals(
        self, stream_id: str, stream_events: List[StreamEvent]
    ) -> None:
        for event in stream_events:
            if event.arrived_at is None:
                continue
            if stream_id not in self._wall_first:
                self._wall_first[stream_id] = event.arrived_at
            self._wall_last[stream_id] = event.arrived_at

    # -- runtime-health metrics ----------------------------------------------

    def arrival_counts(self) -> Dict[str, int]:
        """Total events seen per stream since the last reset."""
        return dict(self._arrival_counts)

    def empty_windows(self) -> int:
        """Cognitive ticks whose window held no events at all."""
        return self._empty_windows

    def silent_streams(self, min_windows: int = 1) -> List[str]:
        """Streams that have been silent for at least `min_windows` windows."""
        return sorted(
            stream_id
            for stream_id, silent in self._silent_windows.items()
            if silent >= min_windows
        )

    def stale_streams(
        self, now: float, factor: float = STALE_PERIOD_FACTOR
    ) -> List[str]:
        """Rate-bearing streams quiet for more than ``factor`` nominal periods.

        A stream with a nominal rate that has fallen silent longer than
        ``factor / rate`` seconds of simulated time (which tracks wall clock
        in realtime) has missed a heartbeat — its publisher likely stopped.
        Irregular streams (no nominal rate) are never stale.
        """
        stale: List[str] = []
        for stream_id, rate in self._nominal_rates.items():
            threshold = factor / rate
            last = self._last_arrival_sim.get(stream_id, 0.0)
            if now - last > threshold:
                stale.append(stream_id)
        return sorted(stale)

    def wall_clock_rates(self) -> Dict[str, float]:
        """Measured events/sec per stream over the wall-clock (``arrived_at``)
        span.  Empty in fast-forward mode, where events carry no wall clock.

        Uses the mean inter-arrival rate ``(count - 1) / span`` so a regular
        stream measures its true cadence without a fence-post bias from the
        initial snapshot event.
        """
        rates: Dict[str, float] = {}
        for stream_id, first in self._wall_first.items():
            last = self._wall_last.get(stream_id, first)
            span = last - first
            count = self._arrival_counts.get(stream_id, 0)
            if span > 0 and count >= 2:
                rates[stream_id] = round((count - 1) / span, 3)
        return rates

    def reset(self) -> None:
        self._tick_index = 0
        self._last_ended_at = 0.0
        self._arrival_counts.clear()
        self._silent_windows.clear()
        self._empty_windows = 0
        self._last_arrival_sim.clear()
        self._wall_first.clear()
        self._wall_last.clear()
