"""Learner interface: consume each cognitive-tick window.

The MVP learns offline (behavioral cloning from recordings), so the default
online learner only accumulates reward statistics — now read from the
`reward.scalar` stream events in the window.  The hook exists so online
learning can be added without changing the loop.
"""

from __future__ import annotations

import abc

from cognitive_runtime.core.streams.synchronizer import TickWindow


def window_reward(window: TickWindow) -> float:
    """Sum the raw `reward.scalar` `value`s published during this window.

    This is the *logging*-facing scale (episode totals, dashboards, replay
    verification) -- unclipped, un-normalized.  Learners that update from a
    scalar reward should use :func:`window_training_reward` instead (issue
    #41's two-scale rewards: huge raw magnitudes must never hit an optimizer
    directly).
    """
    total = 0.0
    for event in window.by_stream.get("reward.scalar", []):
        payload = event.payload
        if isinstance(payload, dict) and isinstance(payload.get("value"), (int, float)):
            total += float(payload["value"])
    return total


def window_training_reward(window: TickWindow) -> float:
    """Sum the `training_value` (normalized/clipped when a reward profile is
    active) published during this window, falling back to the raw `value`
    for events that don't carry one -- e.g. non-Minecraft Programs, or the
    legacy hard-coded SurvivalReward path."""
    total = 0.0
    for event in window.by_stream.get("reward.scalar", []):
        payload = event.payload
        if not isinstance(payload, dict):
            continue
        value = payload.get("training_value", payload.get("value"))
        if isinstance(value, (int, float)):
            total += float(value)
    return total


class Learner(abc.ABC):
    @abc.abstractmethod
    def update(self, window: TickWindow) -> None:
        ...

    def reset(self) -> None:
        pass


class NullLearner(Learner):
    """Accumulates reward statistics from the window; performs no learning."""

    def __init__(self) -> None:
        self.total_reward = 0.0
        self.ticks = 0

    def reset(self) -> None:
        self.total_reward = 0.0
        self.ticks = 0

    def update(self, window: TickWindow) -> None:
        self.total_reward += window_reward(window)
        self.ticks += 1
