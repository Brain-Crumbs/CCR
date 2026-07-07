"""Wall-clock rate pacing for realtime publication.

A Program publishes at whatever cadence its logic dictates (vision every
tick, body vitals on change plus a heartbeat, events when they happen).  In
**fast-forward** that maps straight onto tick cadences and the pacer is inert
— tests stay fast and deterministic.  In **realtime** the pacer throttles a
stream to its target wall-clock rate so different senses genuinely update at
different rates: vision at 10–30 Hz, a body heartbeat at 1–10 Hz, while
irregular streams (events) opt out by carrying no target rate.

The pacer never touches the simulated timestamp on an event; it only decides
*whether* to publish at a given instant.  The runtime feeds it **simulated
time** — which the realtime scheduler holds locked to the wall clock — so the
same pacing that throttles a live run also reproduces deterministically when a
realtime recording is replayed in fast-forward.  Determinism is doubly safe:
the pacer is disabled entirely in fast-forward runs, and driven by the
deterministic simulated clock when enabled.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

#: The consumer samples the pacer at discrete cognitive ticks, so a stream
#: whose period is a near-multiple of the tick period can alias *down* (a
#: 10 Hz stream sampled by a 20 Hz loop keeps missing the boundary by a
#: fraction and fires every third tick ≈ 6.7 Hz).  Allowing a publication a
#: small fraction of a period early absorbs that jitter; because the schedule
#: still advances by exactly one period, the measured rate stays close to
#: nominal (never more than ``1/(1-slack)`` × nominal — within the 20% budget).
PACER_SLACK = 0.1


class RatePacer:
    """Per-stream wall-clock rate limiter.

    ``enabled=False`` (fast-forward) makes every :meth:`should_publish`
    return ``True`` — the Program's own cadence rules through.  When enabled,
    a stream with a target rate publishes at most once per ``1 / rate``
    seconds of wall clock, scheduled against an ideal timeline (like the tick
    scheduler) so it neither drifts nor bursts.
    """

    def __init__(
        self,
        enabled: bool = False,
        clock: Callable[[], float] = time.monotonic,
        rates: Optional[Dict[str, float]] = None,
    ):
        self.enabled = enabled
        self._clock = clock
        #: stream_id -> target Hz.  A stream absent here (or with a falsy
        #: rate) is irregular and never throttled.
        self._rates: Dict[str, float] = dict(rates or {})
        #: stream_id -> next scheduled wall-clock instant it may publish.
        self._next_due: Dict[str, float] = {}

    def set_rate(self, stream_id: str, rate_hz: Optional[float]) -> None:
        """Set (or clear, with ``None``/0) a stream's target publication rate."""
        if rate_hz:
            self._rates[stream_id] = float(rate_hz)
        else:
            self._rates.pop(stream_id, None)

    def target_rate(self, stream_id: str) -> Optional[float]:
        return self._rates.get(stream_id)

    def should_publish(self, stream_id: str, now: Optional[float] = None) -> bool:
        """True if ``stream_id`` may publish at this wall-clock instant.

        In fast-forward mode, or for a stream with no target rate, always
        ``True``.  Otherwise gate on an ideal schedule advancing by one period
        per publication, with a small slack that absorbs tick-sampling jitter.
        """
        rate = self._rates.get(stream_id)
        if not self.enabled or not rate:
            return True
        period = 1.0 / rate
        current = self._clock() if now is None else now
        due = self._next_due.get(stream_id)
        if due is None:
            self._next_due[stream_id] = current + period
            return True
        if current + period * PACER_SLACK >= due:
            next_due = due + period
            if next_due <= current:  # fell behind (a stall): resync, don't burst
                next_due = current + period
            self._next_due[stream_id] = next_due
            return True
        return False

    def reset(self) -> None:
        """Forget pacing history (called on episode reset)."""
        self._next_due.clear()
