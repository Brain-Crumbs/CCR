"""Compatibility bridges between pull-style Observations and streams.

``ObservationStreamShim`` wraps any legacy pull-style Program and publishes
generic streams by diffing consecutive ``observe()`` results — one
``observation.<key>`` stream per top-level data key, the frame as
``vision.frame.grid``.  It keeps third-party/legacy Programs runnable on
the stream substrate and doubles as the migration recipe.

``LatestValueView`` goes the other way: it reconstructs an
Observation-shaped snapshot from the latest value of each stream in a
``TemporalBuffer`` — the compatibility path for observation-based policies
and featurizers once the loop is stream-native.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.program import ActionResult, Program, ProgramMetadata
from cognitive_runtime.core.reward import RewardSignal
from cognitive_runtime.core.streams.bus import MotorStreamBus, SensoryStreamBus
from cognitive_runtime.core.streams.delta import DeltaPublisher
from cognitive_runtime.core.streams.events import StreamSpec
from cognitive_runtime.core.streams.motor import (
    MOTOR_COMMAND_SPEC,
    action_from_motor_event,
)
from cognitive_runtime.core.streams.temporal_buffer import TemporalBuffer

#: Legacy observation data keys become "observation.<key>" streams.  The
#: "observation" prefix is not a modality name, so the specs declare the
#: generic "world" modality — the shim cannot know what a key really is.
OBSERVATION_STREAM_PREFIX = "observation."


class ObservationStreamShim(Program):
    """Publish a legacy Program's observations as generic streams."""

    def __init__(self, program: Program):
        self._program = program
        self._sensory: Optional[SensoryStreamBus] = None
        self._motor: Optional[MotorStreamBus] = None
        self._delta: Optional[DeltaPublisher] = None

    # ------------------------------------------------- legacy passthrough

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._program.initialize(config)

    def observe(self) -> Observation:
        return self._program.observe()

    def act(self, action: Action) -> ActionResult:
        return self._program.act(action)

    def reward(self) -> RewardSignal:
        return self._program.reward()

    def is_complete(self) -> bool:
        return self._program.is_complete()

    def snapshot(self) -> str:
        return self._program.snapshot()

    def restore(self, snapshot_id: str) -> None:
        self._program.restore(snapshot_id)

    def metadata(self) -> ProgramMetadata:
        return self._program.metadata()

    def episode_stats(self) -> Dict[str, Any]:
        return self._program.episode_stats()

    # ---------------------------------------------------- streams-first

    def stream_catalog(self) -> List[StreamSpec]:
        specs = [
            StreamSpec(
                stream_id=f"{OBSERVATION_STREAM_PREFIX}{key}",
                modality="world",
                description=f"legacy observation key {key!r} (published on change)",
            )
            for key in self._program.metadata().observation_keys
        ]
        specs.append(
            StreamSpec(
                "vision.frame.grid",
                "vision",
                description="legacy observation frame (published on change)",
            )
        )
        specs.append(
            StreamSpec("reward.scalar", "reward", nominal_rate_hz=None,
                       payload_schema='{"value": float, "components": dict}')
        )
        specs.append(
            StreamSpec("event.action_rejected", "event",
                       payload_schema='{"reason": str}')
        )
        return specs

    def attach_buses(self, sensory: SensoryStreamBus, motor: MotorStreamBus) -> None:
        self._sensory = sensory
        self._motor = motor
        for spec in self.stream_catalog():
            sensory.register(spec)
        motor.register(MOTOR_COMMAND_SPEC)
        self._delta = DeltaPublisher(sensory)
        self._publish_observation(self._program.observe())

    def reset(self, seed: Optional[int] = None) -> None:
        self._program.reset(seed)
        if self._sensory is not None:
            self._sensory.reset()
            assert self._motor is not None and self._delta is not None
            self._motor.reset()
            self._delta.reset()
            self._publish_observation(self._program.observe())

    def step(self) -> None:
        assert self._sensory is not None and self._motor is not None, (
            "attach_buses() before step()"
        )
        action = NULL_ACTION
        chosen = False
        rejections: List[str] = []
        for event in self._motor.drain():
            try:
                candidate = action_from_motor_event(event)
            except ValueError as exc:
                rejections.append(str(exc))
                continue
            if chosen:
                rejections.append(f"superseded: one command per tick ({candidate.key()})")
                continue
            action, chosen = candidate, True

        result = self._program.act(action)
        if not result.ok:
            rejections.append(str(result.info.get("error", f"rejected: {action.key()}")))
            self._program.act(NULL_ACTION)  # the world still advances this tick

        observation = self._program.observe()
        self._publish_observation(observation)
        for reason in rejections:
            self._sensory.publish(
                "event.action_rejected", {"reason": reason}, observation.timestamp
            )
        signal = self._program.reward()
        self._sensory.publish(
            "reward.scalar",
            {"value": signal.value, "components": dict(signal.components)},
            observation.timestamp,
        )

    def _publish_observation(self, observation: Observation) -> None:
        assert self._sensory is not None and self._delta is not None
        for key, value in observation.data.items():
            stream_id = f"{OBSERVATION_STREAM_PREFIX}{key}"
            if self._sensory.spec(stream_id) is None:  # key beyond the metadata
                self._sensory.register(StreamSpec(stream_id, "world"))
            self._delta.publish(stream_id, value, observation.timestamp)
        if observation.frame is not None:
            self._delta.publish(
                "vision.frame.grid", observation.frame, observation.timestamp
            )


class LatestValueView:
    """Observation-shaped snapshot of the latest value of each stream."""

    #: Transient streams that have no place in a state snapshot.
    _EXCLUDED_PREFIXES = ("event.", "motor.", "reward.")

    def __init__(self, buffer: TemporalBuffer):
        self._buffer = buffer

    def to_observation(self, tick: int = 0) -> Observation:
        data: Dict[str, Any] = {}
        frame = None
        timestamp = 0.0
        for stream_id in self._buffer.streams():
            event = self._buffer.latest(stream_id)
            assert event is not None
            if stream_id.startswith(self._EXCLUDED_PREFIXES):
                continue
            timestamp = max(timestamp, event.timestamp)
            if stream_id == "vision.frame.grid":
                frame = event.payload
            elif stream_id.startswith(OBSERVATION_STREAM_PREFIX):
                data[stream_id[len(OBSERVATION_STREAM_PREFIX):]] = event.payload
            else:
                data[stream_id] = event.payload
        return Observation(timestamp=timestamp, tick=tick, data=data, frame=frame)
