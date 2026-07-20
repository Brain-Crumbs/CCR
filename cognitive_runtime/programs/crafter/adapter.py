"""CrafterWorld Program adapter (issue #89).

Wraps the third-party ``crafter`` package behind the exact streams-v2 seam
Minecraft's ``MinecraftSurvivalBox`` implements (``core/program.py``), so the
runtime, recorder and replay machinery run unmodified against a fast,
deterministic, pixel-native nursery world.  See
``docs/v2/phases/phase-1-nursery-world.md``.

``crafter`` is an optional dependency, imported lazily here (like
``cognitive_runtime.neural``'s torch imports) so other worlds/commands (e.g.
``--world minecraft``, ``replay``, ``dashboard``) never require it installed.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from cognitive_runtime.programs.crafter.actions import ACTION_NAME_TO_INDEX, ACTION_SPACE
from cognitive_runtime.programs.crafter.config import CrafterConfig
from cognitive_runtime.programs.crafter.observations import (
    OBSERVATION_KEYS,
    build_observation,
    build_state,
)
from cognitive_runtime.programs.crafter.streams import (
    BODY_HEARTBEAT_HZ,
    CrafterStreamPublisher,
    VISION_STREAM,
    build_crafter_stream_specs,
)

_VALID_ACTION_NAMES = {a.name for a in ACTION_SPACE}
#: Matches MinecraftSurvivalBox's tick->seconds convention (20 ticks/sec).
_SIM_SECONDS_PER_TICK = 0.05


def _import_crafter():
    try:
        import crafter
    except ImportError as exc:  # pragma: no cover - exercised via ImportError message
        raise ImportError(
            "the crafter world needs the 'crafter' package; install '.[crafter]'"
        ) from exc
    return crafter


class CrafterWorld(Program):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._crafter = _import_crafter()
        self._config = CrafterConfig()
        self._env: Any = None
        self._pending_reward: RewardSignal = RewardSignal()
        self._pending_achievement_events: List[Tuple[str, int]] = []
        self._pending_died = False
        self._last_action: Action = NULL_ACTION
        self._last_obs: Any = None  # pixels ndarray from the last step/reset
        self._last_state: Dict[str, Any] = {}
        self._tick = 0
        self._dead = False
        self._done = False
        self._seed = 0
        self._sensory_bus: Optional[SensoryStreamBus] = None
        self._motor_bus: Optional[MotorStreamBus] = None
        self._publisher: Optional[CrafterStreamPublisher] = None
        #: Off until set_realtime() enables it; shared with the publisher so
        #: the runtime can flip realtime mode after attach_buses().
        self._pacer = RatePacer(enabled=False)
        self._realtime = False
        self._snapshots: Dict[str, Any] = {}
        self._snapshot_serial = 0
        #: name -> highest count seen this episode (achievements are
        #: cumulative counters, not one-shot flags -- see event.achievement).
        self._achievements_earned: Dict[str, int] = {}
        #: Set by ``freeze_reset()`` (issue #90's scripted nursery scenarios,
        #: e.g. ``object_permanence``'s scripted mob path): once set,
        #: ``reset()`` re-derives state from the already-edited ``self._env``
        #: instead of constructing a fresh one, so a caller's scripted world
        #: edits survive ``CognitiveRuntime.run()``'s own ``reset(seed)`` call
        #: at episode start (mirrors Minecraft scene-setup's
        #: ``world.reset = lambda seed: None`` convention).
        self._env_frozen = False
        self.initialize(config)

    # ------------------------------------------------------------ interface

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config:
            self._config = CrafterConfig.from_dict(config)
        self.reset(seed=self._seed)

    def freeze_reset(self) -> None:
        """Defeat future ``reset()`` calls rebuilding ``self._env`` -- for a
        scripted nursery scenario (issue #90) that has just finished editing
        the live env's world (walls, a scripted mob path) and needs that
        state to survive ``CognitiveRuntime.run()``'s own ``reset(seed)`` at
        episode start."""
        self._env_frozen = True

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._seed = seed
        if self._env_frozen:
            # A frozen env keeps its scripted world exactly as edited; only
            # re-derive state/observation from it (no ``env.reset()``, which
            # would regenerate the world and evict every scripted edit).
            pixels = self._env.render()
        else:
            # A fresh Env per reset, not env.reset(): crafter.Env.reset()
            # seeds each episode off hash((self._seed, self._episode)), so
            # calling reset() twice on one instance produces two *different*
            # worlds. A new instance keyed on the seed alone is what makes
            # reset(seed) reproducible -- this Program's determinism
            # contract.
            self._env = self._crafter.Env(
                area=self._config.area, view=self._config.view, size=self._config.size,
                length=self._config.episode_ticks, seed=self._seed,
            )
            pixels = self._env.reset()
        self._tick = 0
        self._dead = False
        self._done = False
        self._last_action = NULL_ACTION
        self._last_obs = pixels
        self._last_state = build_state(self._env, self._config.grid_radius)
        self._achievements_earned = {
            name: count for name, count in self._last_state["achievements"].items() if count
        }
        self._pending_reward = RewardSignal()
        self._pending_achievement_events = []
        self._pending_died = False
        if self._sensory_bus is not None:
            assert self._motor_bus is not None and self._publisher is not None
            self._sensory_bus.reset()
            self._motor_bus.reset()
            self._publisher.reset()
            self._publish_initial_state()

    def observe(self) -> Observation:
        timestamp = self._tick * _SIM_SECONDS_PER_TICK
        return build_observation(self._last_state, self._last_obs, timestamp, self._tick)

    @staticmethod
    def _validation_error(action: Action) -> Optional[str]:
        if action.name not in _VALID_ACTION_NAMES:
            return f"unknown action {action.name}"
        return None

    def _advance(self, action: Action) -> None:
        """Apply one crafter step; refreshes cached state/reward/events.
        Shared by ``act()`` and ``step()`` so both paths advance identically
        (mirrors ``MinecraftSurvivalBox``)."""
        index = ACTION_NAME_TO_INDEX[action.name]
        pixels, reward, done, info = self._env.step(index)
        self._tick += 1
        self._last_action = action
        self._last_obs = pixels
        self._last_state = build_state(self._env, self._config.grid_radius)
        self._done = done
        was_dead = self._dead
        self._dead = self._last_state["health"] <= 0
        self._pending_died = self._dead and not was_dead
        self._pending_achievement_events = []
        for name, count in info["achievements"].items():
            if count > self._achievements_earned.get(name, 0):
                self._achievements_earned[name] = count
                self._pending_achievement_events.append((name, count))
        self._pending_reward = RewardSignal.from_components({"crafter": round(float(reward), 6)})

    def act(self, action: Action) -> ActionResult:
        error = self._validation_error(action)
        if error is not None:
            return ActionResult(ok=False, info={"error": error})
        self._advance(action)
        return ActionResult(ok=True, info={"achievements": list(self._pending_achievement_events)})

    def reward(self) -> RewardSignal:
        """Reward for the most recent tick (crafter's own step() reward)."""
        return self._pending_reward

    def is_complete(self) -> bool:
        return self._done

    # ------------------------------------------------- streams-first interface

    def stream_catalog(self) -> List[StreamSpec]:
        return build_crafter_stream_specs(
            grid_radius=self._config.grid_radius,
            pixel_shape=(self._config.size[0], self._config.size[1], 3),
            world_size=float(self._config.area[0]),
            vision_hz=self._config.realtime_vision_hz if self._realtime else 20.0,
            heartbeat_hz=(
                self._config.realtime_body_heartbeat_hz if self._realtime else BODY_HEARTBEAT_HZ
            ),
        )

    def attach_buses(self, sensory: SensoryStreamBus, motor: MotorStreamBus) -> None:
        self._sensory_bus = sensory
        self._motor_bus = motor
        for spec in self.stream_catalog():
            sensory.register(spec)
        motor.register(MOTOR_COMMAND_SPEC)
        self._publisher = CrafterStreamPublisher(sensory, source="crafter", pacer=self._pacer)
        self._publish_initial_state()

    def set_realtime(
        self, enabled: bool, clock: Optional[Callable[[], float]] = None
    ) -> None:
        self._realtime = enabled
        self._wall_clock = clock
        self._pacer.enabled = enabled
        if clock is not None:
            self._pacer._clock = clock
        self._pacer.set_rate(VISION_STREAM, self._config.realtime_vision_hz)
        self._pacer.set_rate("body.heartbeat", self._config.realtime_body_heartbeat_hz)
        self._pacer.reset()

    def step(self) -> None:
        """Advance one program tick from the motor bus.

        Zero motor events is a NULL tick -- the world still advances (crafter's
        own noop). At most one command applies per tick; malformed or surplus
        commands are rejected via ``event.action_rejected`` and the world
        steps anyway.
        """
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

        self._advance(action)
        timestamp = self._tick * _SIM_SECONDS_PER_TICK
        self._publisher.publish_tick(
            self._tick, self._last_state, self._last_obs, timestamp,
            self._pending_achievement_events, reward_signal=self._pending_reward,
            died=self._pending_died,
        )
        for reason in rejections:
            self._sensory_bus.publish("event.action_rejected", {"reason": reason}, timestamp)

    def _publish_initial_state(self) -> None:
        """Full snapshot per stream so subscribers never start blind."""
        assert self._publisher is not None
        timestamp = self._tick * _SIM_SECONDS_PER_TICK
        self._publisher.publish_tick(
            self._tick, self._last_state, self._last_obs, timestamp, [], paced=False,
        )

    def snapshot(self) -> str:
        self._snapshot_serial += 1
        snapshot_id = f"snap-{self._tick}-{self._snapshot_serial}"
        self._snapshots[snapshot_id] = copy.deepcopy(
            (
                self._env, self._tick, self._dead, self._done, self._last_action,
                self._last_obs, self._last_state, self._achievements_earned,
            )
        )
        return snapshot_id

    def restore(self, snapshot_id: str) -> None:
        (
            self._env, self._tick, self._dead, self._done, self._last_action,
            self._last_obs, self._last_state, self._achievements_earned,
        ) = copy.deepcopy(self._snapshots[snapshot_id])

    def metadata(self) -> ProgramMetadata:
        return ProgramMetadata(
            name="CrafterWorld",
            version="0.1.0",
            description="Nursery world: a fast, deterministic, pixel-native 2-D survival "
                         "world behind the same seam Minecraft uses (issue #89).",
            action_space=list(ACTION_SPACE),
            observation_keys=list(OBSERVATION_KEYS),
            tags=["crafter", "nursery", "mvp"],
            deterministic=True,
        )

    def episode_stats(self) -> Dict[str, Any]:
        return {
            "final_tick": self._tick,
            "success": not self._dead and self._tick >= self._config.episode_ticks,
            "termination_reason": (
                "death:health" if self._dead
                else "episode_ticks" if self._done
                else "running"
            ),
            "death_reason": "health" if self._dead else None,
            "achievements_unlocked": sum(1 for c in self._achievements_earned.values() if c),
            "achievement_counts": dict(self._achievements_earned),
            # Unlike Minecraft's grid/viewer duality, Crafter's pixel frame
            # always comes from the same native renderer -- issue #90's
            # data-quality gate (record.quality) still reads this key to
            # confirm provenance across worlds via one shared field.
            "pixel_sources": ["crafter"],
        }
