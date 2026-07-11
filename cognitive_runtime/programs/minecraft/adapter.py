"""MinecraftSurvivalBox Program adapter.

Implements the universal Program interface on top of a pluggable survival
backend.  The MVP ships a deterministic simulated backend; a real-Minecraft
backend (e.g. driving a client via mineflayer, RCON, or Project Malmo) can
be added by implementing `SurvivalBackend` -- nothing above this file needs
to change, and the runtime itself never changes.
"""

from __future__ import annotations

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
from cognitive_runtime.programs.minecraft.world import RECIPE_NAMES
from cognitive_runtime.programs.minecraft.backend import SimulatedBackend, SurvivalBackend
from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig
from cognitive_runtime.programs.minecraft.observations import OBSERVATION_KEYS
from cognitive_runtime.programs.minecraft.remote import RemoteMinecraftBackend
from cognitive_runtime.programs.minecraft.reward_engine import ProfileRewardEngine
from cognitive_runtime.programs.minecraft.reward_profile import RewardProfile
from cognitive_runtime.programs.minecraft.rewards import SurvivalReward, SurvivalRewardConfig
from cognitive_runtime.programs.minecraft.streams import (
    BODY_HEARTBEAT_KEY,
    MOUSE_LOOK_STREAM,
    VISION_STREAM,
    SurvivalStreamPublisher,
    build_survival_stream_specs,
    mouse_look_delta,
)

_VALID_ACTION_NAMES = {a.name for a in ACTION_SPACE}
_SIM_SECONDS_PER_TICK = 0.05  # simulated time; deterministic for replay


#: Backend registry; ``--backend`` selects one.  ``SurvivalBackend`` and
#: ``SimulatedBackend`` live in ``backend.py`` and ``RemoteMinecraftBackend``
#: in ``remote.py``; they are re-exported here so existing imports keep working.
BACKENDS = {"simulated": SimulatedBackend, "remote": RemoteMinecraftBackend}


class MinecraftSurvivalBox(Program):
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        reward_config: Optional[SurvivalRewardConfig] = None,
        backend: str = "simulated",
        reward_profile: Optional[RewardProfile] = None,
    ):
        self._config = SurvivalBoxConfig()
        self._backend_name = backend
        self._backend: Optional[SurvivalBackend] = None
        self._reward_profile = reward_profile
        #: A loaded profile (issue #41) drives rewards generically through
        #: ProfileRewardEngine; otherwise the historical hard-coded
        #: SurvivalReward (optionally tuned by `reward_config`) is unchanged.
        self._reward_fn = (
            ProfileRewardEngine(reward_profile)
            if reward_profile is not None
            else SurvivalReward(reward_config)
        )
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

    def close(self) -> None:
        """Release backend resources (e.g. the remote bridge subprocess).

        The simulated backend holds nothing; a remote backend closes its
        bridge.  The runtime does not require this — the remote bridge also
        cleans up at interpreter exit — but it is the tidy way to stop a live
        client when you are done with a Program instance.
        """
        if self._backend is not None:
            self._backend.close()

    @staticmethod
    def _validation_error(action: Action) -> Optional[str]:
        if action.name not in _VALID_ACTION_NAMES:
            return f"unknown action {action.name}"
        if action.name in ("SELECT_HOTBAR_SLOT", "EQUIP_ITEM", "PLACE_BLOCK", "USE_ITEM"):
            slot = action.param("slot")
            if not isinstance(slot, int) or not 0 <= slot < HOTBAR_SLOTS:
                return f"invalid slot {slot!r}"
        elif action.name == "MOVE_INVENTORY_ITEM":
            from_slot = action.param("from_slot")
            to_slot = action.param("to_slot")
            if (
                not isinstance(from_slot, int) or not 0 <= from_slot < HOTBAR_SLOTS
                or not isinstance(to_slot, int) or not 0 <= to_slot < HOTBAR_SLOTS
            ):
                return f"invalid slots {from_slot!r},{to_slot!r}"
            if from_slot == to_slot:
                return "from_slot and to_slot must differ"
        elif action.name == "CRAFT":
            recipe = action.param("recipe")
            if recipe not in RECIPE_NAMES:
                return f"unknown recipe {recipe!r}"
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
            {
                "value": signal.value,
                "components": dict(signal.components),
                # Two-scale rewards (issue #41): raw `value`/`components` for
                # dashboards/logging; `training_value` -- normalized/clipped
                # when a reward profile is active, otherwise identical to
                # `value` -- is what an optimizer should actually consume.
                "training_value": (
                    signal.training_value if signal.training_value is not None else signal.value
                ),
            },
            observation.timestamp,
        )
        self._sensory_bus.publish(
            MOUSE_LOOK_STREAM, mouse_look_delta(action.name), observation.timestamp
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
        if not self._backend.supports_snapshots:
            raise NotImplementedError(
                f"the {self._backend_name!r} backend does not support snapshots"
            )
        return self._backend.snapshot()

    def restore(self, snapshot_id: str) -> None:
        assert self._backend is not None
        if not self._backend.supports_snapshots:
            raise NotImplementedError(
                f"the {self._backend_name!r} backend does not support snapshots"
            )
        self._backend.restore(snapshot_id)

    def reward_profile_metadata(self) -> Optional[Dict[str, Any]]:
        """Profile name + content hash (issue #41), for session metadata so
        dashboards can group runs by profile.  `None` when no profile is
        active (the legacy hard-coded SurvivalReward path)."""
        if self._reward_profile is None:
            return None
        return self._reward_profile.metadata()

    def observe_external_streams(self, payloads: Dict[str, Any]) -> None:
        """Runtime-computed streams (issue #58's `internal.*`, issue #61's
        derived risk-gated terms) are published directly onto the shared
        sensory bus by `CognitiveRuntime`, *after* this Program's own
        `step()` already ran this tick -- they never flow through
        `SurvivalStreamPublisher.publish_tick`'s `tick_events`, so a
        profile's `intrinsic` slots would otherwise never see them. Primed
        here (called by `CognitiveRuntime` right after it publishes them)
        so the *next* tick's reward evaluation picks them up -- the same
        one-tick lag `model.novelty`/`internal.*` already have relative to
        the window that first observes them. A no-op unless a `RewardProfile`
        is active (the legacy `SurvivalReward` path never reads `internal.*`)."""
        if isinstance(self._reward_fn, ProfileRewardEngine):
            self._reward_fn.observe_external_streams(payloads)

    def reward_engine_state_dict(self) -> Optional[Dict[str, Any]]:
        """Brain-scoped milestone state + return normalizer (issue #41), for
        the checkpoint bundle.  `None` when no profile is active."""
        if not isinstance(self._reward_fn, ProfileRewardEngine):
            return None
        return self._reward_fn.state_dict()

    def load_reward_engine_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore brain-scoped milestone state on resume, so a checkpoint
        interrupt/resume does not re-grant one-time rewards already earned."""
        if not isinstance(self._reward_fn, ProfileRewardEngine):
            raise ValueError("no reward profile is active; nothing to restore state into")
        self._reward_fn.load_state_dict(state)

    def metadata(self) -> ProgramMetadata:
        return ProgramMetadata(
            name="MinecraftSurvivalBox",
            version="0.1.0",
            description="Survive as long as possible in a constrained survival world.",
            action_space=list(ACTION_SPACE),
            observation_keys=list(OBSERVATION_KEYS),
            tags=["minecraft", "survival", "mvp", self._backend_name],
            deterministic=(
                self._backend.deterministic if self._backend is not None
                else BACKENDS[self._backend_name].deterministic
            ),
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
        if isinstance(self._reward_fn, ProfileRewardEngine):
            stats["reward_by_tier"] = self._reward_fn.tier_totals()
        return stats
