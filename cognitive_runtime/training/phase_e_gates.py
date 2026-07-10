"""Phase E evaluation gates: actor/critic vs random/scripted/linear-Q on
identical seeds (issue #31, ``docs/neural-stream-agent.md`` Phase E "Evaluate
against random/scripted/linear Q").

Where :mod:`cognitive_runtime.training.actor_critic_acceptance` is a smoke
check ("actor/critic beats random"), this is the full deprecation gate. It
trains the actor/critic *and* the linear online-Q baseline in the simulated
backend on a fixed config (a small deterministic default for CI, or a named
curriculum preset for manual runs), evaluates both -- plus scripted and random
-- with no mutation on identical episode seeds, and reports the three gates
from the target doc:

  1. actor/critic > random     -- hard requirement.
  2. actor/critic > linear Q   -- unlocks deprecating ``OnlineQ*`` as primary.
  3. reproducible improvement  -- the same seeds reproduce the eval summaries
     and gate 1 across reruns ("simulation shows reproducible improvement").

Gates 1-2 use the same "beats" convention as the acceptance harnesses -- a
policy beats another when it earns more total reward *or* survives more total
ticks on identical seeds. In this simulated world random wanders into more
incidental novelty reward than the trained learners do, so the meaningful
signal against random is survival: the trained policies reach the end of
lethal night episodes that kill random. The eval runs can be recorded
(``record_dir``) for dashboard inspection, and the gate results can be written
into the actor/critic checkpoint bundle's training stats (``checkpoint_path``,
issue #20).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry
from cognitive_runtime.models.online_q import OnlineQModel
from cognitive_runtime.neural import (
    ActorCriticOptimizer,
    MLPPolicyModel,
    MLPValueModel,
    NeuralAgentCheckpoint,
)
from cognitive_runtime.policies import RandomPolicy, ScriptedSurvivalPolicy
from cognitive_runtime.policies.actor_critic import (
    ActorCriticLearner,
    ActorCriticPolicy,
    world_feature_width,
)
from cognitive_runtime.policies.online_q import OnlineQLearner, OnlineQPolicy
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.curriculum import get_curriculum
from cognitive_runtime.programs.minecraft.rewards import SurvivalRewardConfig
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.recorder import EpisodeSummary, NullRecorder
from cognitive_runtime.training.online_q_acceptance import EvaluationSummary

#: The small deterministic config for the automated gate-1 test (issue #31's
#: "small deterministic config for the automated test").  It mirrors the
#: acceptance harnesses' proven budget: 1200-tick episodes with a day/night
#: cycle whose night (ticks ~100-500) is lethal enough that random dies before
#: the end while the trained policies survive -- the survival-tick gap gate 1
#: keys on.  Larger, longer curriculum presets are the "documented config for
#: manual runs" (``--curriculum`` on the CLI).
DEFAULT_GATE_CONFIG: Dict[str, int] = {
    "episode_ticks": 1200,
    "world_size": 32,
    "day_length": 800,
    "start_time": 300,
}

_SCRIPTED_SEED = 1  # fixed so scripted eval is deterministic across reruns

#: How gates 1-2 rank policies: more total reward, or more total survival ticks
#: (the acceptance harnesses' convention -- see the module docstring).
GATE_METRIC = "reward-or-survival-ticks"


@dataclass(frozen=True)
class PhaseEGateResult:
    summaries: Dict[str, EvaluationSummary]
    gate1_beats_random: bool
    gate2_beats_linear_q: bool
    gate3_reproducible: Optional[bool]
    metric: str
    actor_critic_training_steps: int
    online_q_training_ticks: int
    config: Dict[str, int]
    seeds: Dict[str, int]
    curriculum: Optional[str]

    @property
    def accepted(self) -> bool:
        """Both hard/soft ordering gates hold (gate 3 checked separately)."""
        return self.gate1_beats_random and self.gate2_beats_linear_q

    def gate_summary(self) -> Dict[str, Any]:
        """JSON-safe gate report for checkpoint training stats / dashboards."""
        return {
            "issue": 31,
            "metric": self.metric,
            "curriculum": self.curriculum,
            "config": dict(self.config),
            "seeds": dict(self.seeds),
            "gates": {
                "actor_critic_gt_random": self.gate1_beats_random,
                "actor_critic_gt_linear_q": self.gate2_beats_linear_q,
                "reproducible_improvement": self.gate3_reproducible,
            },
            "training": {
                "actor_critic_steps": self.actor_critic_training_steps,
                "online_q_ticks": self.online_q_training_ticks,
            },
            "eval": {
                name: {
                    "total_reward": s.total_reward,
                    "total_ticks": s.total_ticks,
                    "average_reward": s.average_reward,
                }
                for name, s in self.summaries.items()
            },
        }


def _beats(a: EvaluationSummary, b: EvaluationSummary) -> bool:
    """The gate ordering: more total reward *or* more total survival ticks on
    identical seeds (the acceptance harnesses' convention)."""
    return a.total_reward > b.total_reward or a.total_ticks > b.total_ticks


def _resolve_config(
    curriculum: Optional[str],
    config: Optional[Dict[str, int]],
    reward_config: Optional[SurvivalRewardConfig],
):
    if curriculum:
        preset = get_curriculum(curriculum)
        world = dict(config or preset.world_config)
        rewards = reward_config or dataclasses.replace(
            SurvivalRewardConfig(), **preset.reward_config
        )
        return world, rewards
    return dict(config or DEFAULT_GATE_CONFIG), reward_config


def _runtime_config(
    world_config: Dict[str, int],
    episodes: int,
    seed: int,
    record_dir: Optional[str],
    session_id: Optional[str],
) -> RuntimeConfig:
    return RuntimeConfig(
        episodes=episodes,
        seed=seed,
        max_ticks_per_episode=world_config["episode_ticks"],
        record=record_dir is not None,
        record_dir=record_dir or "sessions",
        session_id=session_id,
        program_config=world_config,
    )


def _run(
    program: MinecraftSurvivalBox,
    policy,
    learner,
    world_config: Dict[str, int],
    episodes: int,
    seed: int,
    *,
    record_dir: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[EpisodeSummary]:
    runtime_config = _runtime_config(world_config, episodes, seed, record_dir, session_id)
    recorder = None if record_dir is not None else NullRecorder()
    return CognitiveRuntime(
        program=program,
        policy=policy,
        learner=learner,
        config=runtime_config,
        recorder=recorder,
    ).run()


def _new_actor_critic_stack(
    world_config: Dict[str, int], seed: int, *, lr: float, entropy_coef: float
):
    """Build the actor/critic stack over the fixed fused latent (mirrors
    :func:`actor_critic_acceptance._new_stack`, but also returns the fusion and
    arch this module needs for the checkpoint bundle)."""
    program = MinecraftSurvivalBox(config=world_config)
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
    arch = {
        "fused_width": fusion.width,
        "world_feature_width": wf_width,
        "n_actions": len(action_keys),
        "hidden_dim": 32,
        "has_world_model": False,
    }
    return fusion, action_keys, policy_model, critic_model, optimizer, arch


def _new_online_q_model(world_config: Dict[str, int], seed: int) -> OnlineQModel:
    program = MinecraftSurvivalBox(config=world_config)
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
        meta={"source": "phase-e-gates"},
    )


def _run_actor_critic(
    policy_model, critic_model, optimizer, action_keys, world_config, reward_config,
    episodes, seed, *, train, record_dir=None, session_id=None,
) -> List[EpisodeSummary]:
    program = MinecraftSurvivalBox(config=world_config, reward_config=reward_config)
    policy = ActorCriticPolicy(
        policy_model, critic_model, action_keys,
        action_space=program.metadata().action_space, training=train, seed=seed,
    )
    learner = ActorCriticLearner(optimizer, policy, training=train)
    return _run(program, policy, learner, world_config, episodes, seed,
                record_dir=record_dir, session_id=session_id)


def _run_online_q(
    model, world_config, reward_config, episodes, seed, *, train,
    record_dir=None, session_id=None,
) -> List[EpisodeSummary]:
    program = MinecraftSurvivalBox(config=world_config, reward_config=reward_config)
    policy = OnlineQPolicy(model, action_space=program.metadata().action_space, training=train)
    learner = OnlineQLearner(model, policy, training=train)
    return _run(program, policy, learner, world_config, episodes, seed,
                record_dir=record_dir, session_id=session_id)


def _run_scripted(
    world_config, reward_config, episodes, seed, *, record_dir=None, session_id=None,
) -> List[EpisodeSummary]:
    program = MinecraftSurvivalBox(config=world_config, reward_config=reward_config)
    policy = ScriptedSurvivalPolicy(seed=_SCRIPTED_SEED)
    return _run(program, policy, None, world_config, episodes, seed,
                record_dir=record_dir, session_id=session_id)


def _run_random(
    world_config, reward_config, episodes, seed, *, record_dir=None, session_id=None,
) -> List[EpisodeSummary]:
    program = MinecraftSurvivalBox(config=world_config, reward_config=reward_config)
    policy = RandomPolicy(ACTION_SPACE, seed=0)
    return _run(program, policy, None, world_config, episodes, seed,
                record_dir=record_dir, session_id=session_id)


def _evaluate_all(
    world_config, reward_config, *, model_seed, train_seed, eval_seed,
    train_episodes, eval_episodes, ac_lr, ac_entropy_coef, record_dir,
):
    """Train both learners then eval all four policies on identical seeds.

    When ``record_dir`` is set each policy's eval episodes are recorded under a
    distinct session id so the existing dashboard/viewer can inspect them.
    """
    fusion, action_keys, policy_model, critic_model, optimizer, arch = (
        _new_actor_critic_stack(world_config, model_seed, lr=ac_lr, entropy_coef=ac_entropy_coef)
    )
    online_model = _new_online_q_model(world_config, model_seed)

    _run_actor_critic(policy_model, critic_model, optimizer, action_keys,
                      world_config, reward_config, train_episodes, train_seed, train=True)
    _run_online_q(online_model, world_config, reward_config, train_episodes, train_seed, train=True)

    ac_eval = _run_actor_critic(
        policy_model, critic_model, optimizer, action_keys, world_config, reward_config,
        eval_episodes, eval_seed, train=False, record_dir=record_dir,
        session_id="phase-e-actor-critic",
    )
    oq_eval = _run_online_q(
        online_model, world_config, reward_config, eval_episodes, eval_seed, train=False,
        record_dir=record_dir, session_id="phase-e-online-q",
    )
    scripted_eval = _run_scripted(
        world_config, reward_config, eval_episodes, eval_seed,
        record_dir=record_dir, session_id="phase-e-scripted",
    )
    random_eval = _run_random(
        world_config, reward_config, eval_episodes, eval_seed,
        record_dir=record_dir, session_id="phase-e-random",
    )
    summaries = {
        "actor-critic": EvaluationSummary.from_episodes("actor-critic", ac_eval),
        "online": EvaluationSummary.from_episodes("online", oq_eval),
        "scripted": EvaluationSummary.from_episodes("scripted", scripted_eval),
        "random": EvaluationSummary.from_episodes("random", random_eval),
    }
    return summaries, optimizer, online_model, fusion, action_keys, policy_model, critic_model, arch


def run_phase_e_gates(
    *,
    curriculum: Optional[str] = None,
    config: Optional[Dict[str, int]] = None,
    reward_config: Optional[SurvivalRewardConfig] = None,
    model_seed: int = 1,
    train_seed: int = 100,
    eval_seed: int = 500,
    train_episodes: int = 20,
    eval_episodes: int = 2,
    ac_lr: float = 1e-2,
    ac_entropy_coef: float = 0.05,
    record_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    check_reproducible: bool = False,
) -> PhaseEGateResult:
    """Train actor/critic and linear online-Q, eval both plus scripted/random
    on identical seeds, and report the three Phase E gates.

    ``check_reproducible`` reruns the whole train+eval a second time with the
    same seeds (no recording/checkpoint the second time) and sets gate 3 to
    whether every eval summary reproduces exactly and gate 1 (the improvement
    over random) still holds.
    """
    world_config, rewards = _resolve_config(curriculum, config, reward_config)
    common = dict(
        model_seed=model_seed, train_seed=train_seed, eval_seed=eval_seed,
        train_episodes=train_episodes, eval_episodes=eval_episodes,
        ac_lr=ac_lr, ac_entropy_coef=ac_entropy_coef,
    )
    (summaries, optimizer, online_model, fusion, action_keys,
     policy_model, critic_model, arch) = _evaluate_all(
        world_config, rewards, record_dir=record_dir, **common
    )

    gate1 = _beats(summaries["actor-critic"], summaries["random"])
    gate2 = _beats(summaries["actor-critic"], summaries["online"])

    # Gate 3: the improvement over the random baseline (gate 1) reproduces
    # exactly across a same-seed rerun -- the "reproducible improvement"
    # success criterion.  Deterministic seeds make the eval summaries bit-equal.
    gate3: Optional[bool] = None
    if check_reproducible:
        rerun, *_ = _evaluate_all(world_config, rewards, record_dir=None, **common)
        gate3 = summaries == rerun and _beats(rerun["actor-critic"], rerun["random"])

    result = PhaseEGateResult(
        summaries=summaries,
        gate1_beats_random=gate1,
        gate2_beats_linear_q=gate2,
        gate3_reproducible=gate3,
        metric=GATE_METRIC,
        actor_critic_training_steps=optimizer.step_count,
        online_q_training_ticks=online_model.training_ticks,
        config=world_config,
        seeds={"model": model_seed, "train": train_seed, "eval": eval_seed},
        curriculum=curriculum,
    )

    if checkpoint_path is not None:
        _write_gate_checkpoint(
            checkpoint_path, fusion, action_keys, policy_model, critic_model,
            optimizer, arch, result,
        )
    return result


def _write_gate_checkpoint(
    checkpoint_path, fusion, action_keys, policy_model, critic_model, optimizer, arch, result,
) -> None:
    """Persist the trained actor/critic with the gate report in training stats
    (issue #20: gate results written into the checkpoint bundle)."""
    checkpoint = NeuralAgentCheckpoint(
        checkpoint_path,
        layout_hash=fusion.layout_hash,
        action_keys=action_keys,
        policy=policy_model,
        critic=critic_model,
        online_optimizer=optimizer,
        training_ticks=optimizer.step_count,
        training_stats={"phase_e_gates": result.gate_summary()},
        extra_metadata={"actor_critic": arch},
    )
    checkpoint.save(reason="phase-e-gates")
