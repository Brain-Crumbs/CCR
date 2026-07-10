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

Streams-first contract (interface v2 — the loop's primary path):

    stream_catalog() -> list[StreamSpec]
    attach_buses(sensory, motor)
    step()                       # the tick driver; replaces act()

Programs publish StreamEvents onto the sensory bus and drain actions from
the motor bus.  ``observe()``/``act()``/``reward()`` remain as the legacy
pull-style contract: ``observe()`` is the loop's compatibility bridge for
observation-based policies, and the shim keeps pull-style Programs runnable
on the stream substrate.  See docs/program-interface.md.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.reward import RewardSignal

if TYPE_CHECKING:  # avoid a runtime cycle: core.streams.shim imports Program
    from cognitive_runtime.core.streams import (
        MotorStreamBus,
        SensoryStreamBus,
        StreamSpec,
    )


class RecoverableEpisodeError(RuntimeError):
    """A Program cannot continue this episode, but the process should not
    crash: the runtime ends the current episode, checkpoints any online
    learner, and moves on to the next episode (issue #33).  A live backend
    whose connection dies mid-episode (e.g. the Mineflayer bridge process)
    raises this instead of letting the failure propagate as process death --
    the world is unrecoverable mid-episode, but the run is not."""


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
    #: Whether reset(seed) + the same action sequence reproduces the world
    #: byte-for-byte.  Simulated backends are; a live server (e.g. real
    #: Minecraft) is not — its recordings cannot be replay-verified by
    #: re-simulation.
    deterministic: bool = True


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

    # -------------------------------------------------- streams-first contract

    def stream_catalog(self) -> "List[StreamSpec]":
        """The streams this Program publishes (sensory, event, reward).

        Default: empty — a legacy pull-style Program.  Wrap those in
        ``ObservationStreamShim`` to get generic streams.
        """
        return []

    def attach_buses(
        self, sensory: "SensoryStreamBus", motor: "MotorStreamBus"
    ) -> None:
        """Connect the Program to its buses.

        Stream-native Programs register their catalog on the sensory bus and
        publish an initial full snapshot per stream so subscribers never
        start blind.  The default just stores the buses for subclasses.
        """
        self._sensory_bus = sensory
        self._motor_bus = motor

    def set_realtime(self, enabled: bool, clock: Optional[Any] = None) -> None:
        """Tell the Program whether the runtime is running in realtime mode.

        Realtime-aware Programs pace per-stream publication to wall-clock
        rates (vision 10–30 Hz, a body heartbeat 1–10 Hz) while irregular
        streams stay event-driven; in fast-forward the same code publishes at
        tick cadence so tests stay fast and deterministic.  The default
        just records the flag — a Program that has no rate machinery ignores
        it.  ``clock`` is a ``() -> float`` wall-clock source (monotonic).
        """
        self._realtime = enabled
        self._wall_clock = clock

    def step(self) -> None:
        """Advance one program tick: drain pending motor events, apply them,
        advance the world, publish sensory/reward/event streams for this tick.

        Contract:
        - Replaces ``act(action)`` as the tick driver.  Motor events drained
          this tick are applied in deterministic order; **zero motor events
          is a NULL tick** — the world still advances (hunger drains, mobs
          move, time passes).
        - Every publication uses simulated time and flows through the bus so
          sequence numbers stay per-stream monotonic.
        - Invalid motor events are rejected by publishing
          ``event.action_rejected`` (payload: reason) rather than raising —
          the world still steps.
        - ``reset(seed)`` must also reset both buses and republish
          initial-state events.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is a legacy pull-style Program; drive it "
            "through observe()/act() or wrap it in ObservationStreamShim"
        )
