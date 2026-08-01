"""CrafterWorld Program adapter (issue #89).

Wraps the third-party ``crafter`` package behind the exact streams-v2 seam
Minecraft's ``MinecraftSurvivalBox`` implements (``core/program.py``), so the
runtime, recorder and replay machinery run unmodified against a fast,
deterministic, pixel-native nursery world.  See
``docs/v2/phases/phase-1-nursery-world.md``.

``crafter`` is a core dependency (issue #176: it's the CLI's default
``--world``), but the import here stays lazy, inside ``__init__``, so other
worlds/commands (e.g. ``--world minecraft``, ``replay``, ``dashboard``)
never pay for it and a partial/dev install missing it still fails with an
actionable message instead of an import-time crash.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.goal import INACTIVE_GOAL_STATE, GoalState
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
from cognitive_runtime.programs.crafter.goals import CaregiverGoalSetter
from cognitive_runtime.programs.crafter.observations import (
    OBSERVATION_KEYS,
    build_observation,
    build_state,
)
from cognitive_runtime.programs.crafter.oracle import OracleLabel, OracleLabelWriter
from cognitive_runtime.programs.crafter.streams import (
    BODY_HEARTBEAT_HZ,
    CrafterStreamPublisher,
    VISION_STREAM,
    build_crafter_stream_specs,
)
from cognitive_runtime.training.navigation_reward import NavigationRewardTracker

_VALID_ACTION_NAMES = {a.name for a in ACTION_SPACE}
#: Crafter's four directional actions -- the only ones that can move the
#: agent, so the only ones a blocked-move "collision" penalty applies to
#: (epic #212 §12.5).
_MOVEMENT_ACTION_NAMES = frozenset({"MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN"})
#: Matches MinecraftSurvivalBox's tick->seconds convention (20 ticks/sec).
_SIM_SECONDS_PER_TICK = 0.05


def _import_crafter():
    try:
        import crafter
    except ImportError as exc:  # pragma: no cover - exercised via ImportError message
        raise ImportError(
            "the crafter world needs the 'crafter' package (a core dependency; "
            "install '.', or 'pip install crafter>=1.8,<2' into this environment)"
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
        #: One-shot-per-reset scripted-world hook (issue #202), e.g.
        #: ``training.nursery``'s wildlife removal/terrain clearing for
        #: scenarios that regenerate a fresh world every seed (not frozen):
        #: called on the freshly (re)built ``self._env`` *before*
        #: ``_last_state``/the initial reset snapshot are captured below, so
        #: the edit takes effect before the reset ever publishes a frame --
        #: not after. Without this, a scene edit applied to ``self._env``
        #: only after ``reset()`` returns is too late: the un-edited state
        #: (incidental world-generation wildlife included) was already
        #: captured into ``_last_state`` and published as the episode's
        #: first recorded frame.
        self._post_reset_hook: Optional[Callable[[Any], None]] = None
        #: Caregiver goal setting (epic #212 §12.4, issue #238); `None` when
        #: `CrafterConfig.goal_enabled` is off (the default -- see
        #: `initialize()`, which (re)builds this from the resolved config
        #: before the first `reset()`).
        self._goal_setter: Optional[CaregiverGoalSetter] = None
        #: A* oracle labels + potential-based geodesic reward shaping (epic
        #: #212 §12.4/12.5, issue #239); both `None` whenever `_goal_setter`
        #: is (there is nothing to plan/shape toward without an active goal).
        self._oracle_writer: Optional[OracleLabelWriter] = None
        self._reward_tracker: Optional[NavigationRewardTracker] = None
        #: This tick's goal/oracle state, computed once per world-advance
        #: (`reset()`/`_advance()`) and read by both the stream publisher
        #: (`_current_goal_state()`) and `oracle_labels()` -- never
        #: recomputed on demand, so a call to `oracle_labels()` never runs a
        #: second A* search for state `step()` already derived this tick.
        self._pending_goal_state: Optional[GoalState] = None
        self._pending_oracle_label: Optional[OracleLabel] = None
        self.initialize(config)

    # ------------------------------------------------------------ interface

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config:
            self._config = CrafterConfig.from_dict(config)
        self._goal_setter = (
            CaregiverGoalSetter(self._config.goal_distribution_config())
            if self._config.goal_enabled
            else None
        )
        self._oracle_writer = (
            OracleLabelWriter(arrival_radius=self._config.goal_arrival_radius)
            if self._config.goal_enabled else None
        )
        self._reward_tracker = (
            NavigationRewardTracker(self._config.goal_reward_config())
            if self._config.goal_enabled
            else None
        )
        self.reset(seed=self._seed)

    def freeze_reset(self) -> None:
        """Defeat future ``reset()`` calls rebuilding ``self._env`` -- for a
        scripted nursery scenario (issue #90) that has just finished editing
        the live env's world (walls, a scripted mob path) and needs that
        state to survive ``CognitiveRuntime.run()``'s own ``reset(seed)`` at
        episode start."""
        self._env_frozen = True

    def set_post_reset_hook(self, hook: Optional[Callable[[Any], None]]) -> None:
        """Register (or clear, with ``None``) the per-reset scripted-world
        hook -- see ``self._post_reset_hook``'s docstring above."""
        self._post_reset_hook = hook

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
        if self._post_reset_hook is not None:
            self._post_reset_hook(self._env)
            pixels = self._env.render()  # re-render: the hook may have edited the world
        self._tick = 0
        self._dead = False
        self._done = False
        self._last_action = NULL_ACTION
        self._last_obs = pixels
        self._last_state = build_state(self._env, self._config.grid_radius)
        self._achievements_earned = {
            name: count for name, count in self._last_state["achievements"].items() if count
        }
        if self._goal_setter is not None:
            # One caregiver goal per episode, proposed once here -- deterministic
            # in the episode seed alone, matching this Program's own determinism
            # contract (see the fresh-``Env``-per-reset comment above).
            self._goal_setter.reset(
                spawn_position=self._position_tuple(),
                world_size=self._config.area,
                seed=self._seed,
                fixed_offset=self._config.goal_fixed_offset,
            )
            self._oracle_writer.reset()
            self._pending_goal_state = self._goal_setter.tracker.state(self._position_tuple())
            self._pending_oracle_label = self._compute_oracle_label()
            self._reward_tracker.reset()
            initial_distance = (
                self._pending_oracle_label.geodesic_distance
                if self._pending_oracle_label is not None
                else None
            )
            self._reward_tracker.prime(geodesic_distance=initial_distance)
        else:
            self._pending_goal_state = None
            self._pending_oracle_label = None
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

    def _position_tuple(self) -> Tuple[float, float]:
        position = self._last_state["position"]
        return (float(position["x"]), float(position["y"]))

    def _current_goal_state(self) -> GoalState:
        """This tick's ``internal.goal`` payload -- the harness's own
        arrival decision (epic #212 §12.4's anti-gaming requirement), from
        ``self._last_state``'s ground-truth position, never from anything a
        policy/motor command wrote. Reads the cached value ``reset()``/
        ``_advance()`` already computed rather than re-evaluating arrival a
        second time this tick."""
        if self._goal_setter is None:
            return INACTIVE_GOAL_STATE
        assert self._pending_goal_state is not None
        return self._pending_goal_state

    def _compute_oracle_label(self) -> Optional[OracleLabel]:
        """This tick's A* oracle label (issue #239): geodesic distance, next
        optimal action, legal-action mask, path-progress delta and
        replan-required, from the simulator's *full* semantic map
        (``env._sem_view()``) -- never the egocentric crop the vision stream
        publishes (see ``oracle.build_walkable_grid``'s docstring for why).
        ``None`` when oracle labels are disabled (``goal_enabled=False``) or
        no goal has ever been proposed."""
        if self._oracle_writer is None or self._goal_setter is None:
            return None
        goal_position = self._goal_setter.tracker.position
        if goal_position is None:
            return None
        current = self._position_tuple()
        # `goal_position` stays a continuous point here (not rounded to a
        # cell): `OracleLabelWriter.label` plans to the walkable cells
        # within `arrival_radius` of it, matching `GoalTracker.
        # evaluate_arrival`'s own Euclidean check exactly (issue #239's
        # fix -- a rounded single target cell could report "unreachable"
        # for a goal sampled onto a wall even though an adjacent cell
        # satisfies arrival).
        return self._oracle_writer.label(
            semantic=self._env._sem_view(),
            position=(int(round(current[0])), int(round(current[1]))),
            goal_position=goal_position,
            world_size=self._config.area,
        )

    def _advance(self, action: Action) -> None:
        """Apply one crafter step; refreshes cached state/reward/events.
        Shared by ``act()`` and ``step()`` so both paths advance identically
        (mirrors ``MinecraftSurvivalBox``)."""
        previous_position = self._position_tuple()
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
        task_reward = round(float(reward), 6)
        if self._goal_setter is None:
            self._pending_reward = RewardSignal.from_components({"crafter": task_reward})
            return
        # Goal-conditioned navigation (epic #212 §12.4/12.5, issue #239):
        # goal progress, completion, collision and task reward stay
        # separate components, never summed into one opaque "crafter" key.
        current_position = self._position_tuple()
        was_active = self._goal_setter.tracker.active
        self._pending_goal_state = self._goal_setter.tick(current_position=current_position)
        just_arrived = was_active and not self._pending_goal_state.active
        self._pending_oracle_label = self._compute_oracle_label()
        distance = (
            self._pending_oracle_label.geodesic_distance
            if self._pending_oracle_label is not None
            else None
        )
        components = dict(
            self._reward_tracker.compute(
                geodesic_distance=distance,
                # The state *before* this tick's arrival evaluation, not
                # after: the shaping delta scores the transition into
                # wherever this move landed, including the final approach
                # step that lands exactly on the goal -- gating on the
                # post-transition `active` (already False once arrived)
                # would silently drop that last, most important delta.
                goal_active=was_active,
                just_arrived=just_arrived,
            )
        )
        components["task"] = task_reward
        blocked = (
            action.name in _MOVEMENT_ACTION_NAMES and current_position == previous_position
        )
        if blocked and self._config.goal_collision_penalty:
            components["collision"] = -abs(self._config.goal_collision_penalty)
        self._pending_reward = RewardSignal.from_components(components)

    def act(self, action: Action) -> ActionResult:
        error = self._validation_error(action)
        if error is not None:
            return ActionResult(ok=False, info={"error": error})
        self._advance(action)
        return ActionResult(ok=True, info={"achievements": list(self._pending_achievement_events)})

    def reward(self) -> RewardSignal:
        """Reward for the most recent tick (crafter's own step() reward)."""
        return self._pending_reward

    def oracle_labels(self) -> Optional[Dict[str, Any]]:
        """Training-only A* oracle labels for the most recent tick (epic
        #212 §12.4/12.5, issue #239): geodesic distance, next optimal
        action, legal-action mask, path-progress delta, replan-required, the
        oracle-planner version and its map-cost assumptions.

        Structurally never reaches the deployed sensory workspace: this
        method is not part of ``stream_catalog()``/``attach_buses()`` and
        never calls ``self._sensory_bus.publish(...)`` -- the runtime's own
        recorder wires it into the per-tick *decision* record instead (see
        ``runtime.loop``), never onto the sensory bus a policy reads from.
        ``None`` when goal-conditioned navigation is disabled or no goal has
        ever been proposed.
        """
        return self._pending_oracle_label.as_payload() if self._pending_oracle_label else None

    def refresh_oracle_labels(self) -> Optional[Dict[str, Any]]:
        """Replan after an in-tick scripted map edit.

        Normal Crafter steps refresh the oracle automatically.  The
        ``replan_after_block`` nursery generator (issue #240) deliberately
        inserts an obstacle *between* observations and action selection; it
        calls this narrow hook so the decision recorded for that same state
        carries the real route-invalidation label instead of a stale route.
        The result remains training-only and never enters the stream catalog.
        """
        self._pending_oracle_label = self._compute_oracle_label()
        return self.oracle_labels()

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
            died=self._pending_died, goal_state=self._current_goal_state(),
        )
        for reason in rejections:
            self._sensory_bus.publish("event.action_rejected", {"reason": reason}, timestamp)

    def _publish_initial_state(self) -> None:
        """Full snapshot per stream so subscribers never start blind."""
        assert self._publisher is not None
        timestamp = self._tick * _SIM_SECONDS_PER_TICK
        self._publisher.publish_tick(
            self._tick, self._last_state, self._last_obs, timestamp, [], paced=False,
            goal_state=self._current_goal_state(),
        )

    def snapshot(self) -> str:
        self._snapshot_serial += 1
        snapshot_id = f"snap-{self._tick}-{self._snapshot_serial}"
        self._snapshots[snapshot_id] = copy.deepcopy(
            (
                self._env, self._tick, self._dead, self._done, self._last_action,
                self._last_obs, self._last_state, self._achievements_earned,
                # The caregiver goal tracker (issue #238) is per-episode
                # mutable state too -- without it, restoring a snapshot taken
                # while a goal was still active (or before/after it was
                # reached) would rewind the world but leave the *current*
                # goal state in place, desyncing internal.goal from the
                # restored position.
                self._goal_setter,
                # The A* oracle writer's previous-distance memory and the
                # navigation reward tracker's previous-potential/bonus-paid
                # state (issue #239) are per-episode mutable state for the
                # same reason: without them, a restored rollout would
                # recompute `path_progress_delta`/`replan_required`/
                # `goal_progress` against the *next* episode's history
                # instead of a fresh one from the restored tick, and could
                # re-pay the arrival bonus a snapshot taken after arrival had
                # already spent.
                self._oracle_writer, self._reward_tracker,
                self._pending_goal_state, self._pending_oracle_label,
            )
        )
        return snapshot_id

    def restore(self, snapshot_id: str) -> None:
        (
            self._env, self._tick, self._dead, self._done, self._last_action,
            self._last_obs, self._last_state, self._achievements_earned,
            self._goal_setter,
            self._oracle_writer, self._reward_tracker,
            self._pending_goal_state, self._pending_oracle_label,
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
