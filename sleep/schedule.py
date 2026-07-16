"""Staleness-free phasic wake/sleep coordination.

The coordinator intentionally has no policy-learning dependency.  Wake work is
limited to caller-provided cheap cortex and episodic-encoding hooks; heavy
self-supervised consolidation happens only after acting has stopped.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional


class Phase(str, Enum):
    WAKE = "wake"
    SLEEP = "sleep"


@dataclass(frozen=True)
class ConsolidationResult:
    """The single weight hand-off produced at the end of a sleep phase."""

    published_version: int
    loaded_version: Optional[int]


class PhasicSleepSchedule:
    """Alternate a fixed number of acting ticks with one consolidation pass.

    ``act`` never starts sleep itself and therefore never blocks on heavy
    training.  Once ``sleep_due`` is true the caller pauses its actor and calls
    ``consolidate``.  Reloading is performed only after consolidation has
    completed and its final snapshot has been published.
    """

    def __init__(self, wake_ticks: int):
        if wake_ticks <= 0:
            raise ValueError(f"wake_ticks must be positive, got {wake_ticks!r}")
        self.wake_ticks = wake_ticks
        self.phase = Phase.WAKE
        self.ticks_in_phase = 0
        self.sleep_due = False
        self.consolidations = 0

    def act(
        self,
        action: Callable[[], Any],
        *,
        wake_update: Optional[Callable[[], None]] = None,
        encode_seed: Optional[Callable[[], None]] = None,
    ) -> Any:
        """Run one acting tick plus optional cheap wake-only hooks."""
        if self.phase is not Phase.WAKE or self.sleep_due:
            raise RuntimeError("acting is paused until consolidation completes")
        result = action()
        if wake_update is not None:
            wake_update()
        if encode_seed is not None:
            encode_seed()
        self.ticks_in_phase += 1
        self.sleep_due = self.ticks_in_phase >= self.wake_ticks
        return result

    def consolidate(
        self,
        sleep_pass: Callable[[], int],
        *,
        reload_weights: Optional[Callable[[], Optional[int]]] = None,
    ) -> ConsolidationResult:
        """Consolidate while acting is paused, then atomically hand off weights."""
        if not self.sleep_due or self.phase is not Phase.WAKE:
            raise RuntimeError("consolidation is only allowed after the wake phase")
        self.phase = Phase.SLEEP
        try:
            published_version = int(sleep_pass())
            loaded_version = reload_weights() if reload_weights is not None else None
            if loaded_version is not None and loaded_version != published_version:
                raise RuntimeError(
                    "actor did not load the completed consolidation snapshot: "
                    f"published {published_version}, loaded {loaded_version}"
                )
        except BaseException:
            # Remain asleep: resuming acting after a failed/partial update could
            # expose stale weights.  The caller may retry or abort explicitly.
            raise
        self.consolidations += 1
        self.ticks_in_phase = 0
        self.sleep_due = False
        self.phase = Phase.WAKE
        return ConsolidationResult(published_version, loaded_version)
