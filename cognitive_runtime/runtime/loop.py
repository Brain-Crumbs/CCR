"""The continuous cognitive loop (v2: cognitive ticks over stream windows).

The runtime no longer asks "what is the current observation?" — it asks
**"what streams have arrived since the last cognitive tick?"**:

    while running:
        scheduler.wait_for_next_tick()
        for _ in range(program_ticks_per_cognitive_tick):
            program.step()                       # drains motor bus, publishes streams
        window  = synchronizer.collect(sensory_bus)
        tokens  = encoders.encode_window(window)
        memory.update(window, tokens)
        pred    = world_model.predict(state, memory)
        motor   = policy.emit(state, memory, pred)   # [] == NULL
        for action in motor: motor_bus.publish(...)
        learner.update(window)
        recorder.write_tick(...)

**One-tick actuation latency.** Motor events emitted at cognitive tick *t*
sit on the motor bus and are applied by `program.step()` at the start of
tick *t+1*.  This is how real sensorimotor loops behave; it is stable and
documented because replay and reward attribution depend on it.

The loop stays environment-agnostic: it only talks to the stream buses and
the Program interface.  `program.observe()` is used solely as the sanctioned
Phase-2 compatibility bridge that lets observation-based policies (and the
recorder/featurizer) keep working until Phase 4 moves them onto latent
tokens.
"""

from __future__ import annotations

import time
from typing import List, Optional

from cognitive_runtime.core.learner import Learner, NullLearner, window_reward
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import Policy
from cognitive_runtime.core.program import Program
from cognitive_runtime.core.streams import (
    MotorStreamBus,
    PassthroughEncoder,
    SensoryStreamBus,
    StreamEncoderRegistry,
    TickSynchronizer,
    publish_motor_command,
)
from cognitive_runtime.core.world_model import TrendWorldModel, WorldModel
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.recorder import (
    EpisodeSummary,
    NullRecorder,
    Recorder,
    TickRecord,
)
from cognitive_runtime.runtime.scheduler import FixedTickScheduler


def default_encoder_registry() -> StreamEncoderRegistry:
    """Phase-2 registry: passthrough encoders for the numeric streams.

    Real modality encoders arrive in Phase 4; until then this stands in for
    the retired StructuredPerception so the loop has latent tokens to fuse.
    """
    registry = StreamEncoderRegistry()
    registry.register("body.*", PassthroughEncoder())
    registry.register("spatial.*", PassthroughEncoder())
    registry.register("reward.*", PassthroughEncoder())
    return registry


class CognitiveRuntime:
    def __init__(
        self,
        program: Program,
        policy: Policy,
        config: Optional[RuntimeConfig] = None,
        world_model: Optional[WorldModel] = None,
        learner: Optional[Learner] = None,
        recorder: Optional[Recorder] = None,
        encoders: Optional[StreamEncoderRegistry] = None,
    ):
        self.program = program
        self.policy = policy
        self.config = config or RuntimeConfig()
        self.world_model = world_model or TrendWorldModel()
        self.learner = learner or NullLearner()
        self.encoders = encoders or default_encoder_registry()
        self.memory = Memory(capacity=self.config.memory_capacity)
        self.sensory_bus = SensoryStreamBus()
        self.motor_bus = MotorStreamBus()
        self.synchronizer = TickSynchronizer(
            program_ticks_per_cognitive_tick=self.config.program_ticks_per_cognitive_tick
        )
        self.scheduler = FixedTickScheduler(
            tick_rate=self.config.tick_rate, realtime=self.config.realtime
        )
        if recorder is not None:
            self.recorder = recorder
        elif self.config.record:
            self.recorder = Recorder(
                record_dir=self.config.record_dir,
                session_id=self.config.resolved_session_id(policy.name),
                record_observations=self.config.record_observations,
                record_frames=self.config.record_frames,
            )
        else:
            self.recorder = NullRecorder()
        self.program.attach_buses(self.sensory_bus, self.motor_bus)
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> List[EpisodeSummary]:
        """Run the configured number of episodes back to back."""
        meta = self.program.metadata()
        self.recorder.write_session_metadata(
            {
                "session_id": self.recorder.session_id,
                "program": meta.name,
                "program_version": meta.version,
                "policy": self.policy.name,
                "tick_rate": self.config.tick_rate,
                "realtime": self.config.realtime,
                "max_ticks_per_episode": self.config.max_ticks_per_episode,
                "program_ticks_per_cognitive_tick": self.config.program_ticks_per_cognitive_tick,
                "episodes": self.config.episodes,
                "base_seed": self.config.seed,
                "program_config": self.config.program_config,
                "action_space": [a.key() for a in meta.action_space],
                "stream_catalog": [s.to_dict() for s in self.program.stream_catalog()],
            }
        )
        summaries: List[EpisodeSummary] = []
        try:
            for episode_index in range(self.config.episodes):
                if self._stop_requested:
                    break
                seed = self.config.seed + episode_index
                summaries.append(self._run_episode(episode_index, seed))
        finally:
            self.recorder.close()
        return summaries

    def _run_episode(self, episode_index: int, seed: int) -> EpisodeSummary:
        self.program.reset(seed=seed)  # resets both buses + republishes snapshot
        self.policy.reset()
        self.memory.reset()
        self.world_model.reset()
        self.learner.reset()
        self.synchronizer.reset()
        self.scheduler.reset()
        episode_id = self.recorder.start_episode(episode_index)

        ratio = self.config.program_ticks_per_cognitive_tick
        total_reward = 0.0
        null_ticks = 0
        latency_total_ms = 0.0
        ticks = 0
        last_timestamp = 0.0

        while not self._stop_requested:
            if ticks >= self.config.max_ticks_per_episode or self.program.is_complete():
                break
            self.scheduler.wait_for_next_tick()

            for _ in range(ratio):
                self.program.step()

            # "What streams arrived since the last cognitive tick?"  Decision
            # latency is measured from here (window collection) to emission.
            decide_start = time.perf_counter()
            observation = self.program.observe()  # Phase-2 compatibility bridge
            window = self.synchronizer.collect(self.sensory_bus, now=observation.timestamp)
            tokens = self.encoders.encode_window(window)
            self.memory.update(window, tokens)
            state = State(observation=observation)
            prediction = self.world_model.predict(state, self.memory)
            emissions = self.policy.emit(state, self.memory, prediction)
            latency_ms = (time.perf_counter() - decide_start) * 1000.0
            if self.policy.stop_requested:
                self._stop_requested = True
                break

            for action in emissions:
                publish_motor_command(self.motor_bus, action, observation.timestamp)
            self.memory.record_actions(emissions)
            self.learner.update(window)

            reward_value = window_reward(window)
            components = _aggregate_components(window)
            # window.by_stream only holds streams that fired this window.
            events = sorted(sid for sid in window.by_stream if sid.startswith("event."))
            selected = emissions[0].key() if emissions else "NULL"
            total_reward += reward_value
            latency_total_ms += latency_ms
            last_timestamp = observation.timestamp
            if not emissions:
                null_ticks += 1
            ticks += 1

            self.recorder.write_tick(
                TickRecord(
                    session_id=self.recorder.session_id,
                    episode_id=episode_id,
                    tick_id=observation.tick,
                    timestamp=observation.timestamp,
                    observation_hash=observation.hash(),
                    selected_action=selected,
                    action_ok=True,
                    reward=round(reward_value, 6),
                    reward_components=components,
                    events=events,
                    policy_name=self.policy.name,
                    latency_ms=round(latency_ms, 3),
                    observation=observation.to_dict(include_frame=True),
                )
            )

        stats = self.program.episode_stats()
        summary = EpisodeSummary(
            session_id=self.recorder.session_id,
            episode_id=episode_id,
            seed=seed,
            policy_name=self.policy.name,
            duration_ticks=ticks,
            total_reward=round(total_reward, 4),
            success=bool(stats.get("success", not self.program.is_complete())),
            termination_reason=str(
                stats.get("termination_reason", "max_ticks" if ticks else "empty")
            ),
            null_action_ticks=null_ticks,
            avg_latency_ms=round(latency_total_ms / ticks, 3) if ticks else 0.0,
            ticks_per_second=round(self.scheduler.stats.ticks_per_second, 2),
            missed_ticks=self.scheduler.stats.missed_ticks,
            program_ticks_per_cognitive_tick=ratio,
            stream_event_rates=self._stream_rates(last_timestamp),
            silent_streams=self.synchronizer.silent_streams(min_windows=1),
            program_stats=stats,
        )
        self.recorder.write_summary(summary)
        self.recorder.end_episode_file()
        return summary

    def _stream_rates(self, sim_elapsed: float) -> dict:
        """Events/sec per stream_id over the episode's simulated duration."""
        counts = self.synchronizer.arrival_counts()
        if sim_elapsed <= 0:
            return {sid: 0.0 for sid in sorted(counts)}
        return {sid: round(counts[sid] / sim_elapsed, 3) for sid in sorted(counts)}


def _aggregate_components(window) -> dict:
    """Sum reward.scalar component dicts across a window."""
    components: dict = {}
    for event in window.by_stream.get("reward.scalar", []):
        payload = event.payload
        if isinstance(payload, dict):
            for name, value in (payload.get("components") or {}).items():
                if isinstance(value, (int, float)):
                    components[name] = round(components.get(name, 0.0) + float(value), 6)
    return components
