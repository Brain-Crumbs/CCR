"""MinecraftSurvivalBox Program adapter.

Implements the universal Program interface on top of a pluggable survival
backend.  The MVP ships a deterministic simulated backend; a real-Minecraft
backend (e.g. driving a client via mineflayer, RCON, or Project Malmo) can
be added by implementing `SurvivalBackend` -- nothing above this file needs
to change, and the runtime itself never changes.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.program import ActionResult, Program, ProgramMetadata
from cognitive_runtime.core.reward import RewardSignal
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE, HOTBAR_SLOTS
from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig
from cognitive_runtime.programs.minecraft.observations import (
    OBSERVATION_KEYS,
    build_observation,
)
from cognitive_runtime.programs.minecraft.rewards import SurvivalReward, SurvivalRewardConfig
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

    def observe(self) -> Observation:
        assert self._backend is not None
        timestamp = self._backend.tick() * _SIM_SECONDS_PER_TICK
        return self._backend.observe(timestamp)

    def act(self, action: Action) -> ActionResult:
        assert self._backend is not None
        if action.name not in _VALID_ACTION_NAMES:
            return ActionResult(ok=False, info={"error": f"unknown action {action.name}"})
        if action.name == "SELECT_HOTBAR_SLOT":
            slot = action.param("slot")
            if not isinstance(slot, int) or not 0 <= slot < HOTBAR_SLOTS:
                return ActionResult(ok=False, info={"error": f"invalid slot {slot!r}"})
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
