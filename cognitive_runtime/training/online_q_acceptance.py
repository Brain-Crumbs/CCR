"""Deterministic simulated acceptance run for the online Q learner.

This is deliberately small and boring: train in the simulated Minecraft
backend for a fixed budget, evaluate with epsilon forced to zero, and compare
against a fixed-seed random policy on the same evaluation seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry
from cognitive_runtime.models.online_q import OnlineQModel
from cognitive_runtime.policies import RandomPolicy
from cognitive_runtime.policies.online_q import OnlineQLearner, OnlineQPolicy
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.recorder import EpisodeSummary, NullRecorder

DEFAULT_ACCEPTANCE_CONFIG: Dict[str, int] = {
    "episode_ticks": 1200,
    "world_size": 32,
    "day_length": 800,
    "start_time": 300,
}


@dataclass(frozen=True)
class EvaluationSummary:
    policy: str
    total_reward: float
    total_ticks: int
    average_reward: float
    average_ticks: float
    termination_reasons: List[str]

    @staticmethod
    def from_episodes(policy: str, episodes: List[EpisodeSummary]) -> "EvaluationSummary":
        total_reward = round(sum(s.total_reward for s in episodes), 6)
        total_ticks = sum(s.duration_ticks for s in episodes)
        n = max(len(episodes), 1)
        return EvaluationSummary(
            policy=policy,
            total_reward=total_reward,
            total_ticks=total_ticks,
            average_reward=round(total_reward / n, 6),
            average_ticks=round(total_ticks / n, 6),
            termination_reasons=[s.termination_reason for s in episodes],
        )


@dataclass(frozen=True)
class OnlineQAcceptanceResult:
    train_episodes: EvaluationSummary
    online_eval: EvaluationSummary
    random_eval: EvaluationSummary
    training_ticks: int
    accepted: bool
    acceptance_metric: str
    config: Dict[str, int]
    seeds: Dict[str, int]


def _new_model(config: Dict[str, int], seed: int) -> OnlineQModel:
    program = MinecraftSurvivalBox(config=config)
    fusion = TemporalFusion(program.stream_catalog(), default_encoder_registry())
    return OnlineQModel.initialize(
        [action.key() for action in program.metadata().action_space],
        latent_width=fusion.width,
        layout_hash=fusion.layout_hash,
        latent_feature_names=fusion.feature_names(),
        seed=seed,
        epsilon_start=0.8,
        epsilon_min=0.05,
        epsilon_decay_ticks=20000,
        lr=0.05,
        gamma=0.99,
        meta={"source": "simulated-online-acceptance"},
    )


def _run_online(
    model: OnlineQModel,
    config: Dict[str, int],
    episodes: int,
    seed: int,
    train: bool,
    record_dir: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[EpisodeSummary]:
    program = MinecraftSurvivalBox(config=config)
    policy = OnlineQPolicy(model, action_space=program.metadata().action_space, training=train)
    learner = OnlineQLearner(model, policy, training=train)
    runtime_config = RuntimeConfig(
        episodes=episodes,
        seed=seed,
        max_ticks_per_episode=config["episode_ticks"],
        record=record_dir is not None,
        record_dir=record_dir or "sessions",
        session_id=session_id,
        program_config=config,
    )
    recorder = None if record_dir is not None else NullRecorder()
    return CognitiveRuntime(
        program=program,
        policy=policy,
        learner=learner,
        config=runtime_config,
        recorder=recorder,
    ).run()


def _run_random(config: Dict[str, int], episodes: int, seed: int) -> List[EpisodeSummary]:
    program = MinecraftSurvivalBox(config=config)
    runtime_config = RuntimeConfig(
        episodes=episodes,
        seed=seed,
        max_ticks_per_episode=config["episode_ticks"],
        record=False,
        program_config=config,
    )
    return CognitiveRuntime(
        program=program,
        policy=RandomPolicy(ACTION_SPACE, seed=0),
        config=runtime_config,
        recorder=NullRecorder(),
    ).run()


def run_simulated_online_acceptance(
    *,
    config: Optional[Dict[str, int]] = None,
    model_seed: int = 1,
    train_seed: int = 100,
    eval_seed: int = 500,
    train_episodes: int = 20,
    eval_episodes: int = 2,
) -> OnlineQAcceptanceResult:
    """Train online Q in simulation and compare epsilon-0 eval to random.

    Fixed defaults intentionally reproduce the acceptance result used in the
    unit suite: online Q survives more total ticks than random on the same
    eval seeds, even when random receives more incidental novelty reward.
    """
    cfg = dict(config or DEFAULT_ACCEPTANCE_CONFIG)
    model = _new_model(cfg, model_seed)
    train = _run_online(model, cfg, train_episodes, train_seed, train=True)
    online = _run_online(model, cfg, eval_episodes, eval_seed, train=False)
    random = _run_random(cfg, eval_episodes, eval_seed)
    online_summary = EvaluationSummary.from_episodes("online", online)
    random_summary = EvaluationSummary.from_episodes("random", random)
    accepted = (
        online_summary.total_reward > random_summary.total_reward
        or online_summary.total_ticks > random_summary.total_ticks
    )
    metric = "reward" if online_summary.total_reward > random_summary.total_reward else "ticks"
    return OnlineQAcceptanceResult(
        train_episodes=EvaluationSummary.from_episodes("online-train", train),
        online_eval=online_summary,
        random_eval=random_summary,
        training_ticks=model.training_ticks,
        accepted=accepted,
        acceptance_metric=metric,
        config=cfg,
        seeds={"model": model_seed, "train": train_seed, "eval": eval_seed},
    )

