"""The continuous cognitive loop.

    while running:
        observation = program.observe()
        state       = perception.encode(observation)
        memory.update(state)
        prediction  = world_model.predict(state, memory)
        action      = policy.decide(state, memory, prediction)
        program.act(action)          # NULL is a real action
        reward      = program.reward()
        learner.update(observation, action, reward)
        recorder.write_tick(...)

The loop is environment-agnostic: it only talks to the Program interface.
"""

from __future__ import annotations

import time
from typing import List, Optional

from cognitive_runtime.core.learner import Learner, NullLearner
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import Perception, StructuredPerception
from cognitive_runtime.core.policy import Policy
from cognitive_runtime.core.program import Program
from cognitive_runtime.core.world_model import TrendWorldModel, WorldModel
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.recorder import (
    EpisodeSummary,
    NullRecorder,
    Recorder,
    TickRecord,
)
from cognitive_runtime.runtime.scheduler import FixedTickScheduler


class CognitiveRuntime:
    def __init__(
        self,
        program: Program,
        policy: Policy,
        config: Optional[RuntimeConfig] = None,
        perception: Optional[Perception] = None,
        world_model: Optional[WorldModel] = None,
        learner: Optional[Learner] = None,
        recorder: Optional[Recorder] = None,
    ):
        self.program = program
        self.policy = policy
        self.config = config or RuntimeConfig()
        self.perception = perception or StructuredPerception()
        self.world_model = world_model or TrendWorldModel()
        self.learner = learner or NullLearner()
        self.memory = Memory(capacity=self.config.memory_capacity)
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
                "episodes": self.config.episodes,
                "base_seed": self.config.seed,
                "program_config": self.config.program_config,
                "action_space": [a.key() for a in meta.action_space],
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
        self.program.reset(seed=seed)
        self.policy.reset()
        self.memory.reset()
        self.world_model.reset()
        self.learner.reset()
        self.scheduler.reset()
        episode_id = self.recorder.start_episode(episode_index)

        total_reward = 0.0
        null_ticks = 0
        latency_total_ms = 0.0
        ticks = 0

        while not self._stop_requested:
            if ticks >= self.config.max_ticks_per_episode or self.program.is_complete():
                break
            self.scheduler.wait_for_next_tick()

            observation = self.program.observe()
            decide_start = time.perf_counter()
            state = self.perception.encode(observation)
            self.memory.update(state)
            prediction = self.world_model.predict(state, self.memory)
            action = self.policy.decide(state, self.memory, prediction)
            latency_ms = (time.perf_counter() - decide_start) * 1000.0
            if self.policy.stop_requested:
                self._stop_requested = True
                break

            result = self.program.act(action)
            self.memory.record_action(action)
            reward = self.program.reward()
            self.learner.update(observation, action, reward)

            total_reward += reward.value
            latency_total_ms += latency_ms
            if action.is_null:
                null_ticks += 1
            ticks += 1

            self.recorder.write_tick(
                TickRecord(
                    session_id=self.recorder.session_id,
                    episode_id=episode_id,
                    tick_id=observation.tick,
                    timestamp=observation.timestamp,
                    observation_hash=observation.hash(),
                    selected_action=action.key(),
                    action_ok=result.ok,
                    reward=reward.value,
                    reward_components=reward.components,
                    events=list(reward.events),
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
            program_stats=stats,
        )
        self.recorder.write_summary(summary)
        self.recorder.end_episode_file()
        return summary
