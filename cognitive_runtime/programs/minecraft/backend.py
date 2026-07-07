"""The ``SurvivalBackend`` seam and the shipped simulated backend.

The adapter (``adapter.py``) talks only to this interface, so a real-Minecraft
backend (``remote.py``) plugs in with no change above it, and the runtime
never changes at all.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig
from cognitive_runtime.programs.minecraft.observations import build_observation
from cognitive_runtime.programs.minecraft.world import SimulatedWorld


class SurvivalBackend(abc.ABC):
    """The seam between the SurvivalBox Program and an actual world.

    Capability flags (class attributes, overridden per backend):

    - ``deterministic`` — reset(seed) + the same action sequence reproduces
      the world byte-for-byte.  True for the simulated backend; False for a
      live server, whose recordings cannot be replay-verified by
      re-simulation.
    - ``supports_snapshots`` — ``snapshot()``/``restore()`` actually capture
      and restore full world state.  A live server cannot honor this; the
      adapter raises a clear error instead of pretending.
    """

    deterministic: bool = True
    supports_snapshots: bool = True

    @abc.abstractmethod
    def reset(self, seed: int) -> None: ...

    @abc.abstractmethod
    def step(self, action: Action) -> "list[str]":
        """Advance one tick; returns semantic events."""

    @abc.abstractmethod
    def observe(self, timestamp: float) -> Observation: ...

    @abc.abstractmethod
    def tick(self) -> int: ...

    @abc.abstractmethod
    def is_dead(self) -> bool: ...

    @abc.abstractmethod
    def death_reason(self) -> Optional[str]: ...

    @abc.abstractmethod
    def stats(self) -> Dict[str, Any]: ...

    @abc.abstractmethod
    def snapshot(self) -> str: ...

    @abc.abstractmethod
    def restore(self, snapshot_id: str) -> None: ...

    def close(self) -> None:
        """Release backend resources (a subprocess, a socket).  No-op by
        default; the simulated backend holds nothing to release."""


class SimulatedBackend(SurvivalBackend):
    def __init__(self, config: SurvivalBoxConfig):
        self.world = SimulatedWorld(config, seed=0)

    def reset(self, seed: int) -> None:
        self.world.reset(seed)

    def step(self, action: Action) -> "list[str]":
        return self.world.step(action)

    def observe(self, timestamp: float) -> Observation:
        return build_observation(self.world, timestamp)

    def tick(self) -> int:
        return self.world.tick

    def is_dead(self) -> bool:
        return self.world.dead

    def death_reason(self) -> Optional[str]:
        return self.world.death_reason

    def stats(self) -> Dict[str, Any]:
        return dict(self.world.stats)

    def snapshot(self) -> str:
        return self.world.snapshot()

    def restore(self, snapshot_id: str) -> None:
        self.world.restore(snapshot_id)
