"""MinecraftSurvivalBox Program adapter.

Implements the universal Program interface on top of a pluggable survival
backend.  The MVP ships a deterministic simulated backend; a real-Minecraft
backend (e.g. driving a client via mineflayer, RCON, or Project Malmo) can
be added by implementing `SurvivalBackend` -- nothing above this file needs
to change, and the runtime itself never changes.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.program import ActionResult, Program, ProgramMetadata
from cognitive_runtime.core.reward import RewardSignal
from cognitive_runtime.core.streams.bus import MotorStreamBus, SensoryStreamBus
from cognitive_runtime.core.streams.events import StreamSpec
from cognitive_runtime.core.streams.motor import (
    MOTOR_COMMAND_SPEC,
    MOTOR_COMMAND_STREAM,
    action_from_motor_event,
)
from cognitive_runtime.core.streams.pacer import RatePacer
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE, HOTBAR_SLOTS
from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig
from cognitive_runtime.programs.minecraft.observations import (
    OBSERVATION_KEYS,
    build_observation,
)
from cognitive_runtime.programs.minecraft.rewards import SurvivalReward, SurvivalRewardConfig
from cognitive_runtime.programs.minecraft.streams import (
    BODY_HEARTBEAT_KEY,
    VISION_STREAM,
    SurvivalStreamPublisher,
    build_survival_stream_specs,
)
from cognitive_runtime.programs.minecraft.world import SimulatedWorld

_VALID_ACTION_NAMES = {a.name for a in ACTION_SPACE}
_SIM_SECONDS_PER_TICK = 0.05  # simulated time; deterministic for replay


class SurvivalBackend(abc.ABC):
    """The seam between the SurvivalBox Program and an actual world."""

    @abc.abstractmethod
    def reset(self, seed: int) -> None: ...

    @abc.abstractmethod
    def step(self, action: Action) -> List[str]:
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


class SimulatedBackend(SurvivalBackend):
    def __init__(self, config: SurvivalBoxConfig):
        self.world = SimulatedWorld(config, seed=0)

    def reset(self, seed: int) -> None:
        self.world.reset(seed)

    def step(self, action: Action) -> List[str]:
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


class RemoteMinecraftBackend(SurvivalBackend):
    """Placeholder for a real-Minecraft backend.

    Implementation sketch: run a headless client (mineflayer) or Malmo mod
    alongside a Java server with a fixed seed and world border; translate
    the MVP action space to client inputs; build Observations from the
    client's entity/health/inventory state and (optionally) frames.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "Real-Minecraft backend not implemented in the MVP. "
            "Implement SurvivalBackend against mineflayer/Malmo/RCON."
        )

    reset = step = observe = tick = is_dead = death_reason = stats = snapshot = restore = None  # type: ignore[assignment]


BACKENDS = {"simulated": SimulatedBackend, "remote": RemoteMinecraftBackend}


class MinecraftSurvivalBox(Program):
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        reward_config: Optional[SurvivalRewardConfig] = None,
        backend: str = "simulated",
    ):
        self._config = SurvivalBoxConfig()
        self._backend_name = backend
        self._backend: Optional[SurvivalBackend] = None
        self._reward_fn = SurvivalReward(reward_config)
        self._pending_events: List[str] = []
        self._last_action: Action = NULL_ACTION
        self._last_reward: RewardSignal = RewardSignal()
        self._seed = 0
        self._sensory_bus: Optional[SensoryStreamBus] = None
        self._motor_bus: Optional[MotorStreamBus] = None
        self._publisher: Optional[SurvivalStreamPublisher] = None
        #: Off until set_realtime() enables it; shared with the publisher so
        #: the runtime can flip realtime mode after attach_buses().
        self._pacer = RatePacer(enabled=False)
        self.initialize(config)

    # ------------------------------------------------------------ interface

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config:
            self._config = SurvivalBoxConfig.from_dict(config)
        backend_cls = BACKENDS[self._backend_name]
        self._backend = backend_cls(self._config)
        self.reset(seed=self._seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = seed
        assert self._backend is not None
        self._backend.reset(self._seed)
        self._reward_fn.reset()
        self._pending_events = []
        self._last_action = NULL_ACTION
        self._last_reward = RewardSignal()
        if self._sensory_bus is not None:
            assert self._motor_bus is not None and self._publisher is not None
            self._sensory_bus.reset()
            self._motor_bus.reset()
            self._publisher.reset()
            self._publish_initial_state()

    def observe(self) -> Observation:
        assert self._backend is not None
        timestamp = self._backend.tick() * _SIM_SECONDS_PER_TICK
        return self._backend.observe(timestamp)

    @staticmethod
    def _validation_error(action: Action) -> Optional[str]:
        if action.name not in _VALID_ACTION_NAMES:
            return f"unknown action {action.name}"
        if action.name == "SELECT_HOTBAR_SLOT":
            slot = action.param("slot")
            if not isinstance(slot, int) or not 0 <= slot < HOTBAR_SLOTS:
                return f"invalid slot {slot!r}"
        return None

    def act(self, action: Action) -> ActionResult:
        assert self._backend is not None
        error = self._validation_error(action)
        if error is not None:
            return ActionResult(ok=False, info={"error": error})
        events = self._backend.step(action)
        self._pending_events = events
        self._last_action = action
        return ActionResult(ok=True, info={"events": events})

    def reward(self) -> RewardSignal:
        """Reward for the most recent tick (post-action world state)."""
        observation = self.observe()
        self._last_reward = self._reward_fn.evaluate(
            observation.data,
            self._pending_events,
            self._last_action,
            observation.hash(),
        )
        self._pending_events = []
        return self._last_reward

    def is_complete(self) -> bool:
        assert self._backend is not None
        return self._backend.is_dead() or self._backend.tick() >= self._config.episode_ticks

    # ------------------------------------------------- streams-first interface

    def stream_catalog(self) -> List[StreamSpec]:
        return build_survival_stream_specs(self._config.world_size)

    def attach_buses(self, sensory: SensoryStreamBus, motor: MotorStreamBus) -> None:
        self._sensory_bus = sensory
        self._motor_bus = motor
        for spec in self.stream_catalog():
            sensory.register(spec)
        motor.register(MOTOR_COMMAND_SPEC)
        self._publisher = SurvivalStreamPublisher(
            sensory, source=self._backend_name, pacer=self._pacer
        )
        self._publish_initial_state()

    def set_realtime(
        self, enabled: bool, clock: Optional[Callable[[], float]] = None
    ) -> None:
        """Enable wall-clock pacing of vision + body heartbeat in realtime mode.

        Rates come from the world config (``realtime_vision_hz``,
        ``realtime_body_heartbeat_hz``); irregular streams (events) and the
        per-tick world/reward streams are left unthrottled.  In fast-forward
        (``enabled=False``) the pacer is inert and cadence follows ticks.
        """
        self._realtime = enabled
        self._wall_clock = clock
        self._pacer.enabled = enabled
        if clock is not None:
            self._pacer._clock = clock
        self._pacer.set_rate(VISION_STREAM, self._config.realtime_vision_hz)
        self._pacer.set_rate(
            BODY_HEARTBEAT_KEY, self._config.realtime_body_heartbeat_hz
        )
        self._pacer.reset()

    def step(self) -> None:
        """Advance one program tick from the motor bus.

        Zero motor events is a NULL tick — the world still advances.  At
        most one command applies per tick; malformed or surplus commands
        are rejected via ``event.action_rejected`` and the world steps
        anyway.
        """
        assert self._backend is not None
        assert self._motor_bus is not None and self._sensory_bus is not None, (
            "attach_buses() before step()"
        )
        assert self._publisher is not None
        action = NULL_ACTION
        chosen = False
        rejections: List[str] = []
        for event in self._motor_bus.drain():
            if event.stream_id != MOTOR_COMMAND_STREAM:
                rejections.append(f"unsupported motor stream {event.stream_id!r}")
                continue
            try:
                candidate = action_from_motor_event(event)
            except ValueError as exc:
                rejections.append(str(exc))
                continue
            error = self._validation_error(candidate)
            if error is not None:
                rejections.append(error)
                continue
            if chosen:
                rejections.append(f"superseded: one command per tick ({candidate.key()})")
                continue
            action, chosen = candidate, True

        world_events = self._backend.step(action)
        timestamp = self._backend.tick() * _SIM_SECONDS_PER_TICK
        observation = self._backend.observe(timestamp)
        tick_events = self._publisher.publish_tick(
            observation, world_events, death_reason=self._backend.death_reason()
        )
        for reason in rejections:
            self._sensory_bus.publish(
                "event.action_rejected", {"reason": reason}, observation.timestamp
            )
        signal = self._reward_fn.evaluate_stream_window(tick_events, action)
        self._sensory_bus.publish(
            "reward.scalar",
            {"value": signal.value, "components": dict(signal.components)},
            observation.timestamp,
        )
        self._last_action = action
        self._last_reward = signal

    def _publish_initial_state(self) -> None:
        """Full snapshot per stream so subscribers never start blind.

        ``paced=False`` bypasses the realtime pacer so every stream appears in
        the snapshot even when a stream's wall-clock token would not yet be due.
        """
        assert self._backend is not None and self._publisher is not None
        timestamp = self._backend.tick() * _SIM_SECONDS_PER_TICK
        snapshot = self._publisher.publish_tick(
            self._backend.observe(timestamp), [], paced=False
        )
        self._reward_fn.prime_stream_state(snapshot)

    def snapshot(self) -> str:
        assert self._backend is not None
        return self._backend.snapshot()

    def restore(self, snapshot_id: str) -> None:
        assert self._backend is not None
        self._backend.restore(snapshot_id)

    def metadata(self) -> ProgramMetadata:
        return ProgramMetadata(
            name="MinecraftSurvivalBox",
            version="0.1.0",
            description="Survive as long as possible in a constrained survival world.",
            action_space=list(ACTION_SPACE),
            observation_keys=list(OBSERVATION_KEYS),
            tags=["minecraft", "survival", "mvp", self._backend_name],
        )

    def episode_stats(self) -> Dict[str, Any]:
        assert self._backend is not None
        stats = self._backend.stats()
        dead = self._backend.is_dead()
        stats.update(
            {
                "success": not dead and self._backend.tick() >= self._config.episode_ticks,
                "termination_reason": (
                    f"death:{self._backend.death_reason()}" if dead
                    else "episode_ticks" if self._backend.tick() >= self._config.episode_ticks
                    else "running"
                ),
                "death_reason": self._backend.death_reason() if dead else None,
                "final_tick": self._backend.tick(),
            }
        )
        return stats
