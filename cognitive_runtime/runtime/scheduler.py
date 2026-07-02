"""Fixed tick-rate scheduler.

In realtime mode the scheduler sleeps to hold the target tick rate and
counts missed ticks (ticks that started late by more than one period).
In fast-forward mode it never sleeps -- useful for evaluation and training
-- but still tracks timing statistics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class SchedulerStats:
    ticks: int = 0
    missed_ticks: int = 0
    elapsed_seconds: float = 0.0

    @property
    def ticks_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.ticks / self.elapsed_seconds


class FixedTickScheduler:
    def __init__(self, tick_rate: float = 20.0, realtime: bool = False):
        if tick_rate <= 0:
            raise ValueError("tick_rate must be positive")
        self.period = 1.0 / tick_rate
        self.realtime = realtime
        self.stats = SchedulerStats()
        self._start = time.monotonic()
        self._next_tick = self._start

    def reset(self) -> None:
        self.stats = SchedulerStats()
        self._start = time.monotonic()
        self._next_tick = self._start

    def wait_for_next_tick(self) -> None:
        now = time.monotonic()
        if self.realtime:
            if now < self._next_tick:
                time.sleep(self._next_tick - now)
            elif now - self._next_tick > self.period:
                self.stats.missed_ticks += 1
            self._next_tick += self.period
        self.stats.ticks += 1
        self.stats.elapsed_seconds = time.monotonic() - self._start
