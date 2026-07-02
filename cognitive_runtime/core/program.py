"""The universal Program interface.

A Program is an environment the runtime can inhabit: Minecraft, ToyOS, a
Linux VM, a browser, or a future AI-native OS workspace.  Programs create
experiences; the runtime learns from them.

Every Program implements the same interface so the same runtime can move
between worlds without modification:

    initialize(config)
    observe() -> Observation
    act(Action) -> ActionResult
    reward() -> RewardSignal
    is_complete() -> bool
    reset(seed)
    snapshot() -> snapshot_id
    restore(snapshot_id)
    metadata() -> ProgramMetadata
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.reward import RewardSignal


@dataclass
class ActionResult:
    ok: bool = True
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgramMetadata:
    name: str
    version: str
    description: str = ""
    action_space: List[Action] = field(default_factory=list)
    observation_keys: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class Program(abc.ABC):
    """Abstract environment adapter.  Subclasses hold all world-specific logic."""

    @abc.abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Prepare the Program (connect to the world, load config)."""

    @abc.abstractmethod
    def observe(self) -> Observation:
        """Return the current observation of the world."""

    @abc.abstractmethod
    def act(self, action: Action) -> ActionResult:
        """Apply an action.  The Program advances one tick per act() call;
        the runtime calls act() with NULL when the policy chooses inaction."""

    @abc.abstractmethod
    def reward(self) -> RewardSignal:
        """Return the reward signal for the most recent tick."""

    @abc.abstractmethod
    def is_complete(self) -> bool:
        """True when the current episode has ended."""

    @abc.abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Start a new episode, deterministically from `seed`."""

    @abc.abstractmethod
    def snapshot(self) -> str:
        """Capture the full world state; returns a snapshot id."""

    @abc.abstractmethod
    def restore(self, snapshot_id: str) -> None:
        """Restore a previously captured world state."""

    @abc.abstractmethod
    def metadata(self) -> ProgramMetadata:
        """Static description of the Program, including its action space."""

    def episode_stats(self) -> Dict[str, Any]:
        """Program-specific statistics for the episode summary (optional)."""
        return {}
