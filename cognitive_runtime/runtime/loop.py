"""The continuous cognitive loop (v2: cognitive ticks over stream windows).

The runtime no longer asks "what is the current observation?" — it asks
**"what streams have arrived since the last cognitive tick?"**:

    while running:
        scheduler.wait_for_next_tick()
        for _ in range(program_ticks_per_cognitive_tick):
            program.step()                       # drains motor bus, publishes streams
        window  = synchronizer.collect(sensory_bus)
        memory.update(window)
        latent  = fusion.fuse(window, memory.buffer)   # fixed-width LatentState
        state   = memory.latest_values().to_observation()  # stream-derived
        pred    = world_model.predict(state, memory)
        motor   = policy.emit(state, memory, pred)   # [] == NULL
        for action in motor: motor_bus.publish(...)
        learner.update(window)
        recorder.write_cognitive_tick(window.events, motor, decision)

**One-tick actuation latency.** Motor events emitted at cognitive tick *t*
sit on the motor bus and are applied by `program.step()` at the start of
tick *t+1*.  This is how real sensorimotor loops behave; it is stable and
documented because replay and reward attribution depend on it.

The loop stays environment-agnostic: it only talks to the stream buses and
the Program interface.  The `State` handed to policies is **derived from
stream state** (`Memory.latest_values().to_observation()`), never pulled
from the Program: observation-based policies read the latest value each
stream has published.  `program.observe()` is no longer called by the loop.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from cognitive_runtime.core.entity_persistence import EntityPersistence, NullEntityPersistence
from cognitive_runtime.core.learner import Learner, NullLearner, window_reward
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.novelty import combine_novelty
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import Policy
from cognitive_runtime.core.program import Program, RecoverableEpisodeError
from cognitive_runtime.core.streams import (
    DEFAULT_STREAM_REGISTRY,
    MotorStreamBus,
    SensoryStreamBus,
    StreamEncoderRegistry,
    StreamRegistry,
    StreamSpec,
    TemporalFusion,
    TickSynchronizer,
    default_encoder_registry,
    publish_motor_command,
)
from cognitive_runtime.core.world_model import TrendWorldModel, WorldModel
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.frame_store import FRAME_HASH_ALGORITHM
from cognitive_runtime.runtime.recorder import (
    DecisionRecord,
    EpisodeSummary,
    NullRecorder,
    Recorder,
)
from cognitive_runtime.runtime.scheduler import FixedTickScheduler

#: Combined novelty score (issue #27): world-model prediction error +
#: entity-persistence surprise, published as its own stream so it is
#: recorded, replayable, and visible in `view`/`dashboard` like any other
#: sensory signal instead of living only in the decision record.  "event" is
#: the closest existing modality for a model-introspection scalar; nothing
#: about the id requires a real "model" stream head (see
#: `core.streams.events.validate_stream_identity`).
NOVELTY_STREAM = "model.novelty"
NOVELTY_STREAM_SPEC = StreamSpec(
    NOVELTY_STREAM, "event",
    "Combined novelty: world-model prediction error + entity-persistence "
    "surprise, averaged over whichever are available this tick.",
    payload_schema="{novelty, world_model_error, entity_surprise}",
)

#: The policy's own value estimate for the current fused state (issue #33
#: Phase F: "record model-side streams ... so the viewer can show what the
#: agent predicted vs what happened"), published whenever the policy exposes
#: one (currently `ActorCriticPolicy.latest_decision.value_estimate`; other
#: policies simply never publish it).
VALUE_ESTIMATE_STREAM = "model.value_estimate"
VALUE_ESTIMATE_STREAM_SPEC = StreamSpec(
    VALUE_ESTIMATE_STREAM, "event",
    "The policy's critic value estimate for the current fused state, when "
    "the policy exposes one.",
    payload_schema="{value_estimate}",
)


class CognitiveRuntime:
    def __init__(
        self,
        program: Program,
        policy: Policy,
        config: Optional[RuntimeConfig] = None,
        world_model: Optional[WorldModel] = None,
        entity_persistence: Optional[EntityPersistence] = None,
        learner: Optional[Learner] = None,
        recorder: Optional[Recorder] = None,
        encoders: Optional[StreamEncoderRegistry] = None,
        stream_registry: Optional[StreamRegistry] = None,
    ):
        self.program = program
        self.policy = policy
        self.config = config or RuntimeConfig()
        self.world_model = world_model or TrendWorldModel()
        self.entity_persistence = entity_persistence or NullEntityPersistence()
        self.learner = learner or NullLearner()
        self.encoders = encoders or default_encoder_registry()
        #: Per-stream schema declarations (issue #21): shape/rate come from
        #: the Program's `StreamSpec` catalog, the rest -- encoder binding,
        #: trainable/fixed-stub, latent width, checkpoint key, train/eval
        #: behavior -- from here. Defaults to the generic registry; a Program
        #: with concrete streams that don't fit a generic pattern passes its
        #: own extended `StreamRegistry` (e.g.
        #: `programs.minecraft.stream_registry.MINECRAFT_STREAM_REGISTRY`).
        self.stream_registry = stream_registry or DEFAULT_STREAM_REGISTRY
        self.fusion = TemporalFusion(self.program.stream_catalog(), self.encoders)
        self.memory = Memory(capacity=self.config.memory_capacity)
        # Two clocks: the simulated timestamp drives windowing/hashing/replay;
        # in realtime a monotonic wall clock stamps StreamEvent.arrived_at
        # metadata and paces asynchronous publication.  Fast-forward keeps the
        # bus lock-free and wall-clock-free so it stays byte-identical.
        realtime = self.config.realtime
        wall_clock = time.monotonic if realtime else None
        self.sensory_bus = SensoryStreamBus(thread_safe=realtime, wall_clock=wall_clock)
        self.motor_bus = MotorStreamBus()
        nominal_rates = {
            spec.stream_id: spec.nominal_rate_hz
            for spec in self.program.stream_catalog()
            if spec.nominal_rate_hz
        }
        self.synchronizer = TickSynchronizer(
            program_ticks_per_cognitive_tick=self.config.program_ticks_per_cognitive_tick,
            nominal_rates=nominal_rates,
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
                record_streams=self.config.record_streams,
                exclude_streams=self.config.effective_exclude_streams(),
                pin_on_streams=self.config.pin_on_streams,
                frame_segment_max_bytes=int(self.config.frame_segment_max_mb * 1024 * 1024),
                frame_segment_max_seconds=self.config.frame_segment_max_seconds,
                frame_disk_budget_bytes=int(self.config.frame_disk_budget_mb * 1024 * 1024),
            )
        else:
            self.recorder = NullRecorder()
        self.program.attach_buses(self.sensory_bus, self.motor_bus)
        # Let realtime-aware Programs pace publication to wall-clock rates.
        self.program.set_realtime(realtime, clock=wall_clock)
        # Not part of any Program's catalog: a runtime/model-computed signal
        # rather than an environment sensor, registered directly so it can
        # still be published/recorded/replayed like any other stream.
        self.sensory_bus.register(NOVELTY_STREAM_SPEC)
        self.sensory_bus.register(VALUE_ESTIMATE_STREAM_SPEC)
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> List[EpisodeSummary]:
        """Run the configured number of episodes back to back."""
        meta = self.program.metadata()
        session_metadata = {
            "session_id": self.recorder.session_id,
            "program": meta.name,
            "program_version": meta.version,
            "program_tags": list(meta.tags),
            "deterministic": meta.deterministic,
            "policy": self.policy.name,
            "tick_rate": self.config.tick_rate,
            "realtime": self.config.realtime,
            "max_ticks_per_episode": self.config.max_ticks_per_episode,
            "program_ticks_per_cognitive_tick": self.config.program_ticks_per_cognitive_tick,
            "episodes": self.config.episodes,
            "base_seed": self.config.seed,
            "curriculum": self.config.curriculum,
            "program_config": self.config.program_config,
            "action_space": [a.key() for a in meta.action_space],
            "stream_catalog": [s.to_dict() for s in self.program.stream_catalog()],
            # Marks sessions whose ndarray-payload (frame) stream events hash
            # raw bytes rather than JSON -- see StreamEvent.hash() and
            # runtime.replay's legacy_frame_hash handling.  Absent on sessions
            # recorded before this format, so replay knows not to fault a
            # frame-hash mismatch there as tampering.
            "frame_hash_algorithm": FRAME_HASH_ALGORITHM,
            "stream_registry": self.stream_registry.describe(self.program.stream_catalog()),
        }
        online_metadata = self._online_metadata()
        if online_metadata:
            session_metadata["online_model"] = online_metadata
        # Profile name + content hash (issue #41), when the Program has an
        # active reward profile, so dashboards can group sessions "like with
        # like" instead of mixing runs tuned by different profiles.
        profile_meta_fn = getattr(self.program, "reward_profile_metadata", None)
        if callable(profile_meta_fn):
            profile_metadata = profile_meta_fn()
            if profile_metadata:
                session_metadata["reward_profile"] = profile_metadata
        self.recorder.write_session_metadata(session_metadata)
        summaries: List[EpisodeSummary] = []
        checkpoint_reason = "shutdown"
        try:
            for episode_index in range(self.config.episodes):
                if self._stop_requested:
                    break
                seed = self.config.seed + episode_index
                summaries.append(self._run_episode(episode_index, seed))
        except KeyboardInterrupt:
            checkpoint_reason = "keyboard_interrupt"
            raise
        except Exception:
            checkpoint_reason = "exception"
            raise
        finally:
            self._checkpoint_online(checkpoint_reason)
            self.recorder.close()
        return summaries

    def _run_episode(self, episode_index: int, seed: int) -> EpisodeSummary:
        self.program.reset(seed=seed)  # resets both buses + republishes snapshot
        self.policy.reset()
        self.memory.reset()
        self.world_model.reset()
        self.entity_persistence.reset()
        self.learner.reset()
        self.synchronizer.reset()
        self.scheduler.reset()
        episode_id = self.recorder.start_episode(episode_index)

        ratio = self.config.program_ticks_per_cognitive_tick
        total_reward = 0.0
        null_ticks = 0
        latency_total_ms = 0.0
        motor_emissions = 0
        ticks = 0
        last_timestamp = 0.0
        risk_total = 0.0
        prediction_error_total = 0.0
        prediction_error_ticks = 0
        novelty_total = 0.0
        novelty_ticks = 0
        recoverable_error: Optional[str] = None

        while not self._stop_requested:
            if ticks >= self.config.max_ticks_per_episode or self.program.is_complete():
                break
            self.scheduler.wait_for_next_tick()

            try:
                for _ in range(ratio):
                    self.program.step()
            except RecoverableEpisodeError as exc:
                # A live backend's connection died mid-episode (issue #33):
                # end this episode instead of crashing the run.  The next
                # episode's program.reset() reconnects (RemoteBridge.start()
                # respawns a dead subprocess).
                recoverable_error = str(exc)
                self._checkpoint_online("bridge_error")
                break

            # "What streams arrived since the last cognitive tick?"  Decision
            # latency is measured from here (window collection) to emission.
            decide_start = time.perf_counter()
            window = self.synchronizer.collect(self.sensory_bus)
            self.memory.update(window)
            self.memory.set_fused_latent(self.fusion.fuse(None, self.memory.buffer))
            # The policy's State is derived from stream state, not pulled from
            # the Program: the latest value each stream has published, shaped
            # like an Observation for observation-based policies.
            observation = self.memory.latest_values().to_observation(
                tick=(window.tick_index + 1) * ratio
            )
            state = State(observation=observation)
            prediction = self.world_model.predict(state, self.memory)
            entity_prediction = self.entity_persistence.predict(state, self.memory)
            novelty = combine_novelty(prediction.prediction_error, entity_prediction.surprise)
            if novelty is not None:
                novelty_total += novelty
                novelty_ticks += 1
                self.sensory_bus.publish(
                    NOVELTY_STREAM,
                    {
                        "novelty": round(novelty, 6),
                        "world_model_error": prediction.prediction_error,
                        "entity_surprise": entity_prediction.surprise,
                    },
                    window.ended_at,
                    source="model",
                )
            emissions = self.policy.emit(state, self.memory, prediction)
            policy_decision = getattr(self.policy, "latest_decision", None)
            value_estimate = getattr(policy_decision, "value_estimate", None)
            if value_estimate is not None:
                self.sensory_bus.publish(
                    VALUE_ESTIMATE_STREAM,
                    {"value_estimate": round(float(value_estimate), 6)},
                    window.ended_at,
                    source="model",
                )
            latency_ms = (time.perf_counter() - decide_start) * 1000.0
            if self.policy.stop_requested:
                self._stop_requested = True
                break

            motor_events = [
                publish_motor_command(self.motor_bus, action, window.ended_at)
                for action in emissions
            ]
            motor_emissions += len(motor_events)
            self.memory.record_actions(emissions)
            self.learner.update(window)

            reward_value = window_reward(window)
            total_reward += reward_value
            latency_total_ms += latency_ms
            last_timestamp = window.ended_at
            if not emissions:
                null_ticks += 1
            ticks += 1
            risk_total += prediction.risk
            if prediction.prediction_error is not None:
                prediction_error_total += prediction.prediction_error
                prediction_error_ticks += 1

            self.recorder.write_cognitive_tick(
                sensory_events=window.events,
                motor_events=motor_events,
                decision=DecisionRecord(
                    tick_index=window.tick_index,
                    window_span=[round(window.started_at, 3), round(window.ended_at, 3)],
                    n_events_by_stream={
                        sid: len(evs) for sid, evs in sorted(window.by_stream.items())
                    },
                    motor_emitted=[event.hash() for event in motor_events],
                    policy_name=self.policy.name,
                    latency_ms=round(latency_ms, 3),
                    reward_window_total=round(reward_value, 6),
                    risk=round(prediction.risk, 6),
                    p_death=(
                        round(prediction.p_death, 6) if prediction.p_death is not None else None
                    ),
                    prediction_error=(
                        round(prediction.prediction_error, 6)
                        if prediction.prediction_error is not None
                        else None
                    ),
                ),
            )

        stats = self.program.episode_stats()
        summary = EpisodeSummary(
            session_id=self.recorder.session_id,
            episode_id=episode_id,
            seed=seed,
            policy_name=self.policy.name,
            curriculum=self.config.curriculum,
            duration_ticks=ticks,
            total_reward=round(total_reward, 4),
            success=False if recoverable_error else bool(
                stats.get("success", not self.program.is_complete())
            ),
            termination_reason=(
                "bridge_error" if recoverable_error
                else str(stats.get("termination_reason", "max_ticks" if ticks else "empty"))
            ),
            null_action_ticks=null_ticks,
            avg_latency_ms=round(latency_total_ms / ticks, 3) if ticks else 0.0,
            ticks_per_second=round(self.scheduler.stats.ticks_per_second, 2),
            missed_ticks=self.scheduler.stats.missed_ticks,
            program_ticks_per_cognitive_tick=ratio,
            realtime=self.config.realtime,
            stream_event_rates=self._stream_rates(last_timestamp),
            stream_event_counts=dict(sorted(self.synchronizer.arrival_counts().items())),
            silent_streams=self.synchronizer.silent_streams(min_windows=1),
            empty_windows=self.synchronizer.empty_windows(),
            # A late window is a cognitive tick that started past its deadline;
            # the scheduler already counts these as missed ticks.
            late_windows=self.scheduler.stats.missed_ticks,
            stale_streams=self.synchronizer.stale_streams(now=last_timestamp),
            motor_emissions=motor_emissions,
            motor_emission_rate=(
                round(motor_emissions / last_timestamp, 3) if last_timestamp > 0 else 0.0
            ),
            stream_overflow_counts=self.sensory_bus.overflow_counts(),
            stream_wallclock_rates=self.synchronizer.wall_clock_rates(),
            program_stats=(
                {**stats, "recoverable_error": recoverable_error} if recoverable_error else stats
            ),
            avg_risk=round(risk_total / ticks, 6) if ticks else 0.0,
            avg_prediction_error=(
                round(prediction_error_total / prediction_error_ticks, 6)
                if prediction_error_ticks
                else None
            ),
            avg_novelty=(
                round(novelty_total / novelty_ticks, 6) if novelty_ticks else None
            ),
        )
        self.recorder.write_summary(summary)
        self.recorder.end_episode_file()
        # A recoverable-error episode already checkpointed with a reason that
        # names the cause (`bridge_error`); don't mask it with the generic
        # "episode_end" reason a normal episode close would write.
        if not recoverable_error:
            self._end_online_episode()
        return summary

    def _stream_rates(self, sim_elapsed: float) -> dict:
        """Events/sec per stream_id over the episode's simulated duration."""
        counts = self.synchronizer.arrival_counts()
        if sim_elapsed <= 0:
            return {sid: 0.0 for sid in sorted(counts)}
        return {sid: round(counts[sid] / sim_elapsed, 3) for sid in sorted(counts)}

    def _online_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        policy_meta = getattr(self.policy, "model_metadata", None)
        if callable(policy_meta):
            metadata["policy"] = policy_meta()
        learner_meta = getattr(self.learner, "checkpoint_metadata", None)
        if callable(learner_meta):
            metadata["learner"] = learner_meta()
        return metadata

    def _checkpoint_online(self, reason: str) -> None:
        checkpoint = getattr(self.learner, "checkpoint", None)
        if callable(checkpoint):
            checkpoint(reason=reason)

    def _end_online_episode(self) -> None:
        end_episode = getattr(self.learner, "end_episode", None)
        if callable(end_episode):
            end_episode()
