"""Deterministic simulated acceptance run for the actor/critic online learner
(issue #29, ``docs/neural-stream-agent.md`` Phase E).

Mirrors :mod:`cognitive_runtime.training.online_q_acceptance`'s shape: train
in the simulated Minecraft backend for a fixed budget, evaluate with the
policy forced greedy (argmax, no sampling), and compare against a
fixed-seed random policy on the same evaluation seeds. This is deliberately
a smoke check -- "beats random" -- not the full actor/critic-vs-online-Q gate
(issue #31).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry
from cognitive_runtime.neural import ActorCriticOptimizer, MLPPolicyModel, MLPValueModel
from cognitive_runtime.policies import RandomPolicy
from cognitive_runtime.policies.actor_critic import (
    ActorCriticLearner,
    ActorCriticPolicy,
    world_feature_width,
)
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.recorder import EpisodeSummary, NullRecorder
from cognitive_runtime.training.online_q_acceptance import EvaluationSummary

DEFAULT_ACCEPTANCE_CONFIG: Dict[str, int] = {
    "episode_ticks": 1200,
    "world_size": 32,
    "day_length": 800,
    "start_time": 300,
}


@dataclass(frozen=True)
class ActorCriticAcceptanceResult:
    train_episodes: EvaluationSummary
    actor_critic_eval: EvaluationSummary
    random_eval: EvaluationSummary
    training_steps: int
    accepted: bool
    acceptance_metric: str
    config: Dict[str, int]
    seeds: Dict[str, int]


def _new_stack(config: Dict[str, int], seed: int, *, lr: float, entropy_coef: float):
    program = MinecraftSurvivalBox(config=config)
    fusion = TemporalFusion(program.stream_catalog(), default_encoder_registry())
    action_keys = [action.key() for action in program.metadata().action_space]
    wf_width = world_feature_width(action_keys)
    # Deterministic weight init: ActorCriticOptimizer's own `seed` only covers
    # its later stochastic ops, not construction that already happened.
    torch.manual_seed(seed)
    policy_model = MLPPolicyModel(
        fusion.width, wf_width, len(action_keys),
        hidden_dim=32, layout_hash=fusion.layout_hash, action_keys=action_keys,
    )
    critic_model = MLPValueModel(
        fusion.width, wf_width,
        hidden_dim=32, layout_hash=fusion.layout_hash, action_keys=action_keys,
    )
    optimizer = ActorCriticOptimizer(
        policy_model, critic_model, lr=lr, entropy_coef=entropy_coef, seed=seed,
    )
    return action_keys, policy_model, critic_model, optimizer


def _run_actor_critic(
    policy_model: MLPPolicyModel,
    critic_model: MLPValueModel,
    optimizer: ActorCriticOptimizer,
    action_keys: List[str],
    config: Dict[str, int],
    episodes: int,
    seed: int,
    train: bool,
) -> List[EpisodeSummary]:
    program = MinecraftSurvivalBox(config=config)
    policy = ActorCriticPolicy(
        policy_model, critic_model, action_keys,
        action_space=program.metadata().action_space, training=train, seed=seed,
    )
    learner = ActorCriticLearner(optimizer, policy, training=train)
    runtime_config = RuntimeConfig(
        episodes=episodes,
        seed=seed,
        max_ticks_per_episode=config["episode_ticks"],
        record=False,
        program_config=config,
    )
    return CognitiveRuntime(
        program=program,
        policy=policy,
        learner=learner,
        config=runtime_config,
        recorder=NullRecorder(),
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


def run_simulated_actor_critic_acceptance(
    *,
    config: Optional[Dict[str, int]] = None,
    model_seed: int = 1,
    train_seed: int = 100,
    eval_seed: int = 500,
    train_episodes: int = 20,
    eval_episodes: int = 2,
    lr: float = 1e-2,
    entropy_coef: float = 0.05,
) -> ActorCriticAcceptanceResult:
    """Train actor-critic in simulation and compare a greedy eval to random.

    Fixed defaults intentionally reproduce the acceptance result used in the
    unit suite.
    """
    cfg = dict(config or DEFAULT_ACCEPTANCE_CONFIG)
    action_keys, policy_model, critic_model, optimizer = _new_stack(
        cfg, model_seed, lr=lr, entropy_coef=entropy_coef
    )
    train = _run_actor_critic(
        policy_model, critic_model, optimizer, action_keys, cfg, train_episodes, train_seed, train=True
    )
    evald = _run_actor_critic(
        policy_model, critic_model, optimizer, action_keys, cfg, eval_episodes, eval_seed, train=False
    )
    random = _run_random(cfg, eval_episodes, eval_seed)
    ac_summary = EvaluationSummary.from_episodes("actor-critic", evald)
    random_summary = EvaluationSummary.from_episodes("random", random)
    accepted = (
        ac_summary.total_reward > random_summary.total_reward
        or ac_summary.total_ticks > random_summary.total_ticks
    )
    metric = "reward" if ac_summary.total_reward > random_summary.total_reward else "ticks"
    return ActorCriticAcceptanceResult(
        train_episodes=EvaluationSummary.from_episodes("actor-critic-train", train),
        actor_critic_eval=ac_summary,
        random_eval=random_summary,
        training_steps=optimizer.step_count,
        accepted=accepted,
        acceptance_metric=metric,
        config=cfg,
        seeds={"model": model_seed, "train": train_seed, "eval": eval_seed},
    )
