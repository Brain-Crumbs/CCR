"""Command-line interface for the Continuous Cognitive Runtime.

    python -m cognitive_runtime run --policy scripted --episodes 3
    python -m cognitive_runtime demo
    python -m cognitive_runtime evaluate --episodes 3
    python -m cognitive_runtime statistical-evaluate --episodes 20 --baseline random
    python -m cognitive_runtime train --sessions sessions/<id> --out models/bc.json
    python -m cognitive_runtime replay --session sessions/<id> --verify
    python -m cognitive_runtime view --session sessions/<id> --episode episode_00000
    python -m cognitive_runtime dashboard
    python -m cognitive_runtime nursery list
    python -m cognitive_runtime nursery run walk_forward
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import Any, Callable, Dict, Optional, Sequence

from cognitive_runtime.core.attention import ATTENTION_MODES
from cognitive_runtime.core.orienting_reflex import REFLEX_MODES
from cognitive_runtime.core.policy import Policy
from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry
from cognitive_runtime.models.online_q import OnlineQModel
from cognitive_runtime.policies import (
    HumanDemoPolicy,
    LearnedPolicy,
    NullPolicy,
    OnlineQLearner,
    OnlineQPolicy,
    RandomPolicy,
    ScriptedSurvivalPolicy,
)
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
from cognitive_runtime.programs.minecraft.adapter import BACKENDS, MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.curriculum import CURRICULUM_ORDER, get_curriculum
from cognitive_runtime.programs.minecraft.evaluation import comparison_table, summarize_episodes
from cognitive_runtime.programs.minecraft.reward_profile import (
    RewardProfile,
    RewardProfileError,
    load_reward_profile,
)
from cognitive_runtime.programs.minecraft.rewards import SurvivalRewardConfig
from cognitive_runtime.programs.minecraft.action_registry import MINECRAFT_ACTION_REGISTRY
from cognitive_runtime.programs.minecraft.stream_registry import MINECRAFT_STREAM_REGISTRY
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import NonDeterministicSessionError
from cognitive_runtime.tools.episode_viewer import view_episode
from cognitive_runtime.tools.metrics_dashboard import dashboard
from cognitive_runtime.tools.replay_runner import format_results, replay_session
from cognitive_runtime.tools.review import review_run
from cognitive_runtime.training.datasets import build_dataset
from cognitive_runtime.training.evaluation import compare_policies
from cognitive_runtime.training.imitation import train_bc

DEFAULT_MODEL_OUT = "models/bc.json"
DEFAULT_ONLINE_MODEL_OUT = "models/online-q.json"
DEFAULT_ACTOR_CRITIC_MODEL_OUT = "models/actor-critic.pt"

#: Issue #32 "raw input" ablation: which stream classifications the online
#: policy's fused state is built from. "full" preserves the pre-#32 behavior
#: (encoders=None -> default_encoder_registry()); "raw" restricts fusion to
#: MINECRAFT_STREAM_REGISTRY streams classified agent_input, so hand-computed
#: semantic streams keep publishing/recording but stop reaching the policy.
INPUT_PROFILES = {"full", "raw"}

#: Issue #57 "learned fusion primary" bridge: which fusion path
#: `--policy actor-critic` reads its fused agent state from. "fixed" is the
#: `TemporalFusion` concatenation of hand-written encoders (the default,
#: unchanged); "learned" runs trainable stream encoders + `LatentFusionModel`
#: in the live tick (`cognitive_runtime.neural.live_fusion.LiveLearnedFusion`).
FUSION_MODES = {"fixed", "learned"}


def _default_nursery_backend() -> str:
    """Use the live backend by default when live connection env is present."""
    env_default = os.environ.get("CCR_NURSERY_BACKEND")
    if env_default in BACKENDS:
        return env_default
    if os.environ.get("CCR_MINECRAFT_HOST"):
        return "remote"
    return "simulated"


def _encoders_for_input_profile(profile: str, stream_registry=MINECRAFT_STREAM_REGISTRY):
    if profile == "full":
        return None
    return stream_registry.to_encoder_registry(classifications={"agent_input"})


#: Historical CLI defaults for the world knobs a curriculum preset can also
#: set (issue #30).  `_add_world_args` leaves these unset (`None`) so a
#: chosen curriculum's `world_config` can fill them in; an explicit flag
#: always wins over the curriculum, and this dict wins when neither is given.
_WORLD_DEFAULTS: Dict[str, Any] = {
    "episode_ticks": 6000,
    "difficulty": 1.0,
    "world_size": 64,
    "day_length": 6000,
    "start_time": 0,
    "max_mobs": 3,
    "pixel_source": "viewer",
}


def _resolve_world_args(args: argparse.Namespace) -> None:
    """Fill unset world/seed args from `--curriculum`'s preset, falling back
    to the historical CLI defaults; mutates `args` in place so every caller
    downstream sees plain resolved values, curriculum or not."""
    preset = get_curriculum(args.curriculum) if args.curriculum else None
    for key, default in _WORLD_DEFAULTS.items():
        if getattr(args, key, None) is None:
            value = preset.world_config.get(key, default) if preset else default
            setattr(args, key, value)
    if args.seed is None:
        args.seed = preset.seed if preset else 0


def _reward_config_for(args: argparse.Namespace) -> Optional[SurvivalRewardConfig]:
    """The curriculum's reward-weight bundle applied over the defaults, or
    `None` (default reward config) when no curriculum was chosen."""
    if not args.curriculum:
        return None
    preset = get_curriculum(args.curriculum)
    return dataclasses.replace(SurvivalRewardConfig(), **preset.reward_config)


def _reward_profile_for(args: argparse.Namespace) -> Optional[RewardProfile]:
    """The loaded `--reward-profile`, or `None` for the legacy hard-coded
    reward path.  Fails the whole invocation immediately (issue #41: "a
    malformed profile fails at startup with a clear message, not mid-run")
    rather than letting a bad profile surface as a mid-episode crash."""
    path = getattr(args, "reward_profile", None)
    if not path:
        return None
    try:
        return load_reward_profile(path)
    except RewardProfileError as exc:
        sys.exit(str(exc))


def _program_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "episode_ticks": args.episode_ticks,
        "difficulty": args.difficulty,
        "world_size": args.world_size,
        "day_length": args.day_length,
        "start_time": args.start_time,
        "max_mobs": args.max_mobs,
        "pixel_source": args.pixel_source,
    }


def _make_policy(
    name: str, args: argparse.Namespace, action_space: Optional[list] = None
) -> Policy:
    if name == "null":
        return NullPolicy()
    if name == "random":
        return RandomPolicy(action_space or ACTION_SPACE, seed=args.seed)
    if name == "scripted":
        if getattr(args, "world", "minecraft") != "minecraft":
            sys.exit(
                f"--policy scripted is Minecraft-specific (a hand-authored heuristic over "
                f"Minecraft's own actions); it does not support --world {args.world!r}. "
                "Pick --policy null/random/human/online/actor-critic/learned/neural instead."
            )
        return ScriptedSurvivalPolicy(seed=args.seed)
    if name == "human":
        return HumanDemoPolicy(realtime=getattr(args, "realtime", False))
    if name == "learned":
        if not args.model:
            sys.exit("--model is required for the learned policy")
        return LearnedPolicy(args.model)
    if name == "neural":
        if not args.model:
            sys.exit("--model is required for the neural policy (a .pt bundle)")
        try:
            from cognitive_runtime.policies.neural_policy import NeuralPolicy
        except ImportError as exc:  # torch not installed
            sys.exit(f"the neural policy needs PyTorch ({exc}); install '.[neural]'.")
        return NeuralPolicy(args.model)
    sys.exit(f"unknown policy: {name}")


def _make_world_model(args: argparse.Namespace, program: MinecraftSurvivalBox):
    """The heuristic default (`None`, `TrendWorldModel`), or a trained neural
    world-model checkpoint bridged behind the same `world_model` seam
    (issue #26); `--world-model` is unset unless the caller opts in."""
    path = getattr(args, "world_model", None)
    if not path:
        return None
    try:
        from cognitive_runtime.policies.neural_world_model import NeuralWorldModel
    except ImportError as exc:  # torch not installed
        sys.exit(f"the neural world model needs PyTorch ({exc}); install '.[neural]'.")
    action_keys = [action.key() for action in program.metadata().action_space]
    return NeuralWorldModel(path, action_keys=action_keys)


def _add_world_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--world-model", default=None,
                        help="path to a trained neural world-model checkpoint (.pt bundle, "
                             "--model-type world-model); default: the heuristic TrendWorldModel")


def _make_entity_persistence(args: argparse.Namespace):
    """`None` (no entity-persistence surprise contribution to novelty) unless
    `--entity-persistence` opts into a trained checkpoint (issue #27)."""
    path = getattr(args, "entity_persistence", None)
    if not path:
        return None
    try:
        from cognitive_runtime.policies.neural_entity_persistence import NeuralEntityPersistence
    except ImportError as exc:  # torch not installed
        sys.exit(f"the neural entity-persistence model needs PyTorch ({exc}); install '.[neural]'.")
    return NeuralEntityPersistence(path)


def _add_entity_persistence_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--entity-persistence", default=None,
                         help="path to a trained entity-persistence checkpoint (.pt bundle, "
                              "--model-type entity-persistence); default: no entity-persistence "
                              "contribution to the model.novelty stream")


def _make_online_policy_and_learner(
    args: argparse.Namespace, program: MinecraftSurvivalBox, encoders=None
) -> tuple[OnlineQPolicy, OnlineQLearner]:
    action_space = list(program.metadata().action_space)
    action_keys = [action.key() for action in action_space]
    fusion = TemporalFusion(program.stream_catalog(), encoders or default_encoder_registry())
    model_path = args.online_model
    try:
        if os.path.exists(model_path):
            model = OnlineQModel.load(
                model_path,
                expected_action_keys=action_keys,
                expected_layout_hash=fusion.layout_hash,
                expected_latent_width=fusion.width,
            )
        else:
            model = OnlineQModel.initialize(
                action_keys,
                latent_width=fusion.width,
                layout_hash=fusion.layout_hash,
                latent_feature_names=fusion.feature_names(),
                lr=args.online_lr,
                gamma=args.online_gamma,
                epsilon_start=args.epsilon_start,
                epsilon_min=args.epsilon_min,
                epsilon_decay_ticks=args.epsilon_decay_ticks,
                seed=args.seed,
                meta={
                    "source": "cli",
                    "policy": "online",
                    "program": program.metadata().name,
                    "program_version": program.metadata().version,
                },
            )
    except ValueError as exc:
        sys.exit(str(exc))
    policy = OnlineQPolicy(model, action_space=action_space, training=args.online_train)
    learner = OnlineQLearner(
        model,
        policy,
        training=args.online_train,
        checkpoint_path=model_path,
        save_every_updates=args.online_save_every,
    )
    return policy, learner


def _make_actor_critic_policy_and_learner(
    args: argparse.Namespace, program: MinecraftSurvivalBox, encoders=None,
    stream_registry=MINECRAFT_STREAM_REGISTRY,
):
    """``--policy actor-critic``: the neural actor/critic online policy
    (issue #29, docs/neural-stream-agent.md Phase E), wired the same way
    ``_make_online_policy_and_learner`` wires the linear online-Q baseline.
    Imported lazily -- torch stays optional for every other policy.
    """
    try:
        import torch

        from cognitive_runtime.neural import (
            ActorCriticOptimizer,
            MLPPolicyModel,
            MLPValueModel,
            MLPWorldModel,
            NeuralAgentCheckpoint,
            read_checkpoint_metadata,
        )
        from cognitive_runtime.neural.replay_buffer import MixedTrainingSchedule, ReplayBuffer
        from cognitive_runtime.policies.actor_critic import (
            ActorCriticLearner,
            ActorCriticPolicy,
            world_feature_width,
        )
    except ImportError as exc:  # torch not installed
        sys.exit(f"the actor-critic policy needs PyTorch ({exc}); install '.[neural]'.")

    action_space = list(program.metadata().action_space)
    action_keys = [action.key() for action in action_space]
    fusion = TemporalFusion(program.stream_catalog(), encoders or default_encoder_registry())
    model_path = args.actor_critic_model
    fusion_mode = getattr(args, "fusion", "fixed")
    if fusion_mode not in FUSION_MODES:
        sys.exit(f"unknown --fusion {fusion_mode!r}; expected one of {sorted(FUSION_MODES)}")

    arch: Dict[str, Any] = {
        "fused_width": fusion.width,
        "world_feature_width": world_feature_width(action_keys),
        "n_actions": len(action_keys),
        "hidden_dim": args.actor_critic_hidden_dim,
        "has_world_model": args.actor_critic_world_model_loss,
        "fusion_mode": fusion_mode,
    }
    if os.path.exists(model_path):
        saved_arch = read_checkpoint_metadata(model_path).get("extra", {}).get("actor_critic")
        if saved_arch:
            arch = saved_arch
            saved_fusion_mode = arch.get("fusion_mode", "fixed")
            if saved_fusion_mode != fusion_mode:
                sys.exit(
                    f"checkpoint {model_path!r} was trained with --fusion "
                    f"{saved_fusion_mode!r}, but this run requested --fusion "
                    f"{fusion_mode!r}; use the matching flag, or --fresh to start a new "
                    "checkpoint (issue #57: fusion mode is not silently interchangeable)"
                )

    if fusion_mode == "learned" and getattr(args, "actor_critic_async", False):
        sys.exit(
            "--fusion learned does not support --async-trainer yet (issue #37's async "
            "trainer only knows the fixed fused-latent transition shape); drop one of "
            "the two flags"
        )

    live_fusion = None
    if fusion_mode == "learned":
        from cognitive_runtime.neural.live_fusion import LiveLearnedFusion

        live_fusion = LiveLearnedFusion(
            program.stream_catalog(),
            stream_registry,
            base_layout_hash=fusion.layout_hash,
            fused_width=arch.get("fused_width"),
            hidden_dim=arch["hidden_dim"],
            lr=args.actor_critic_lr,
        )
        arch["fused_width"] = live_fusion.fused_width()
        layout_hash = live_fusion.layout_hash
    else:
        layout_hash = fusion.layout_hash

    if getattr(args, "actor_critic_async", False):
        if getattr(args, "async_schedule", "phasic") == "concurrent":
            return _make_concurrent_actor_critic_policy_and_learner(
                args, arch=arch, model_path=model_path,
                layout_hash=layout_hash, action_keys=action_keys, action_space=action_space,
            ) + (None,)
        return _make_async_actor_critic_policy_and_learner(
            args, arch=arch, model_path=model_path,
            layout_hash=layout_hash, action_keys=action_keys, action_space=action_space,
        ) + (None,)

    # Deterministic weight init: ActorCriticOptimizer's own `seed` only covers
    # its later stochastic ops, not construction, which happens before it exists.
    torch.manual_seed(args.seed)
    policy_model = MLPPolicyModel(
        arch["fused_width"], arch["world_feature_width"], arch["n_actions"],
        hidden_dim=arch["hidden_dim"], layout_hash=layout_hash, action_keys=action_keys,
    )
    critic_model = MLPValueModel(
        arch["fused_width"], arch["world_feature_width"],
        hidden_dim=arch["hidden_dim"], layout_hash=layout_hash, action_keys=action_keys,
    )
    world_model = None
    if arch["has_world_model"]:
        world_model = MLPWorldModel(
            arch["fused_width"], arch["n_actions"],
            hidden_dim=arch["hidden_dim"], layout_hash=layout_hash, action_keys=action_keys,
        )

    optimizer = ActorCriticOptimizer(
        policy_model,
        critic_model,
        world_model=world_model,
        lr=args.actor_critic_lr,
        gamma=args.actor_critic_gamma,
        entropy_coef=args.actor_critic_entropy_coef,
        grad_clip_norm=args.actor_critic_grad_clip_norm,
        seed=args.seed,
    )

    checkpoint_kwargs: Dict[str, Any] = dict(
        layout_hash=layout_hash,
        action_keys=action_keys,
        online_optimizer=optimizer,
        extra_metadata={"actor_critic": arch},
        name=getattr(args, "name", None),
    )
    if live_fusion is not None:
        checkpoint_kwargs["encoders"] = live_fusion.encoders
        checkpoint_kwargs["fusion"] = live_fusion.module
        checkpoint_kwargs["optimizers"] = {"live_fusion": live_fusion.optimizer}
    checkpoint = NeuralAgentCheckpoint(model_path, **checkpoint_kwargs)
    if os.path.exists(model_path):
        try:
            checkpoint.load()
        except ValueError as exc:
            sys.exit(str(exc))

    policy = ActorCriticPolicy(
        policy_model, critic_model, action_keys, action_space=action_space,
        history=args.actor_critic_history, training=args.actor_critic_train, seed=args.seed,
    )
    if live_fusion is not None:
        (live_fusion.train_mode if args.actor_critic_train else live_fusion.eval_mode)()
    replay_buffer = ReplayBuffer()
    learner = ActorCriticLearner(
        optimizer,
        policy,
        training=args.actor_critic_train,
        checkpoint=checkpoint,
        save_every_ticks=args.actor_critic_save_every,
        replay_buffer=replay_buffer,
        mixed_schedule=MixedTrainingSchedule(replay_every_n_ticks=args.actor_critic_replay_every),
        replay_batch_size=args.actor_critic_replay_batch_size,
        live_fusion=live_fusion,
    )
    return policy, learner, live_fusion


def _make_async_actor_critic_policy_and_learner(
    args: argparse.Namespace,
    *,
    arch: Dict[str, Any],
    model_path: str,
    layout_hash: str,
    action_keys: list,
    action_space: list,
):
    """Build the phasic actor/trainer split used by ``--async-trainer``.

    Acting remains inference-only during each wake phase.  Between wake
    phases, the runtime blocks before the next acting tick while the trainer
    drains experience and performs a bounded consolidation pass.  Only the
    completed pass is published and reloaded, so an acting tick can never see
    intermediate or stale consolidation weights.
    """
    from cognitive_runtime.neural.checkpoint import NeuralAgentCheckpoint
    from cognitive_runtime.neural.experience_queue import SharedExperienceRing
    from sleep.weight_publisher import WeightSubscriber
    from cognitive_runtime.policies.actor_critic import (
        ActorCriticPolicy,
        AsyncActorCriticLearner,
    )
    from sleep import PhasicSleepSchedule
    from sleep.async_trainer import ActorCriticArch, AsyncTrainer, build_actor_critic_modules

    trainer_arch = ActorCriticArch(
        fused_width=arch["fused_width"],
        world_feature_width=arch["world_feature_width"],
        n_actions=arch["n_actions"],
        action_keys=tuple(action_keys),
        layout_hash=layout_hash,
        hidden_dim=arch["hidden_dim"],
        has_world_model=arch["has_world_model"],
    )

    ring = SharedExperienceRing(args.async_ring_capacity, arch["fused_width"])
    trainer = AsyncTrainer(
        trainer_arch,
        model_path,
        live_ring_handle=ring.handle(),
        lr=args.actor_critic_lr,
        gamma=args.actor_critic_gamma,
        entropy_coef=args.actor_critic_entropy_coef,
        grad_clip_norm=args.actor_critic_grad_clip_norm,
        seed=args.seed,
        batch_size=args.async_batch_size,
        min_buffer_size=args.async_min_buffer_size,
        publish_every_steps=args.async_publish_every,
    )
    trainer.resume_if_checkpoint_exists()

    policy_model, critic_model, _world_model, _optimizer = build_actor_critic_modules(
        trainer_arch, seed=args.seed,
    )
    actor_bundle = NeuralAgentCheckpoint(
        model_path, layout_hash=layout_hash, action_keys=action_keys,
        policy=policy_model, critic=critic_model,
    )
    subscriber = WeightSubscriber(path=model_path, bundle=actor_bundle)
    # A resumed checkpoint is already a completed snapshot; load it before
    # the first wake tick rather than waiting for the first sleep boundary.
    subscriber.maybe_reload()

    policy = ActorCriticPolicy(
        policy_model, critic_model, action_keys, action_space=action_space,
        history=args.actor_critic_history, training=args.actor_critic_train, seed=args.seed,
    )
    actor_learner = AsyncActorCriticLearner(policy, ring, weight_subscriber=None)
    schedule = PhasicSleepSchedule(wake_ticks=args.async_wake_ticks)

    class _PhasicLearner:
        """Learner adapter: the loop's update boundary is between acting ticks."""

        def __getattr__(self, name: str) -> Any:
            return getattr(actor_learner, name)

        def update(self, window: Any) -> None:
            schedule.act(lambda: actor_learner.update(window))
            if schedule.sleep_due:
                schedule.consolidate(
                    lambda: trainer.consolidate(args.async_consolidation_steps),
                    reload_weights=subscriber.maybe_reload,
                )

        def reset(self) -> None:
            actor_learner.reset()

        def model_metadata(self) -> Dict[str, Any]:
            return actor_learner.model_metadata()

        def finish(self) -> None:
            """Consolidate a final, partial wake phase before teardown."""
            if schedule.request_sleep():
                schedule.consolidate(
                    lambda: trainer.consolidate(args.async_consolidation_steps),
                    reload_weights=subscriber.maybe_reload,
                )

    learner = _PhasicLearner()
    # Stashed for `cmd_run` cleanup; not part of the `Learner` contract.
    learner.async_resources = (ring,)
    learner.phasic_schedule = schedule
    learner.sleep_trainer = trainer
    return policy, learner


def _make_concurrent_actor_critic_policy_and_learner(
    args: argparse.Namespace,
    *,
    arch: Dict[str, Any],
    model_path: str,
    layout_hash: str,
    action_keys: list,
    action_space: list,
):
    """Build the concurrent actor/trainer split used by ``--async-trainer
    --async-schedule concurrent`` (issue #100).

    Unlike the phasic split, acting never pauses: the trainer runs
    continuously in its own OS process (:class:`~sleep.async_trainer.
    TrainerSupervisor`, restarting it if it crashes), and the actor polls
    for a newer weight snapshot every ``--async-reload-every-ticks`` ticks
    instead of blocking. Because a reload can land mid-play, the trainer
    publishes an EMA/Polyak-averaged snapshot (see
    ``sleep.weight_publisher.EMAWeightPublisher``) -- a slow-moving target
    that absorbs tick-to-tick gradient noise -- stamped with the optimizer's
    own monotonic step count, so the actor can measure and bound how many
    versions behind its live weights are (``WeightSubscriber.staleness()``).
    """
    from cognitive_runtime.neural.checkpoint import NeuralAgentCheckpoint
    from cognitive_runtime.neural.experience_queue import SharedExperienceRing
    from sleep.weight_publisher import WeightSubscriber, ema_publish_path
    from cognitive_runtime.policies.actor_critic import (
        ActorCriticPolicy,
        AsyncActorCriticLearner,
    )
    from sleep.async_trainer import ActorCriticArch, TrainerSupervisor, build_actor_critic_modules

    # `EMAWeightPublisher` only validates this inside the spawned trainer
    # process, which the parent would then supervise/restart forever
    # without it ever reporting the real error back to the CLI user.
    if not 0.0 < args.async_ema_decay < 1.0:
        sys.exit(f"--async-ema-decay must be in (0, 1), got {args.async_ema_decay!r}")

    trainer_arch = ActorCriticArch(
        fused_width=arch["fused_width"],
        world_feature_width=arch["world_feature_width"],
        n_actions=arch["n_actions"],
        action_keys=tuple(action_keys),
        layout_hash=layout_hash,
        hidden_dim=arch["hidden_dim"],
        has_world_model=arch["has_world_model"],
    )

    ring = SharedExperienceRing(args.async_ring_capacity, arch["fused_width"])
    supervisor = TrainerSupervisor(
        trainer_arch, model_path,
        live_ring_handle=ring.handle(),
        trainer_kwargs={
            "lr": args.actor_critic_lr,
            "gamma": args.actor_critic_gamma,
            "entropy_coef": args.actor_critic_entropy_coef,
            "grad_clip_norm": args.actor_critic_grad_clip_norm,
            "seed": args.seed,
            "batch_size": args.async_batch_size,
            "min_buffer_size": args.async_min_buffer_size,
            "publish_every_steps": args.async_publish_every,
            "ema_decay": args.async_ema_decay,
        },
    )
    supervisor.start()

    policy_model, critic_model, _world_model, _optimizer = build_actor_critic_modules(
        trainer_arch, seed=args.seed,
    )
    actor_bundle = NeuralAgentCheckpoint(
        model_path, layout_hash=layout_hash, action_keys=action_keys,
        policy=policy_model, critic=critic_model,
    )
    # The trainer's `EMAWeightPublisher` never writes EMA weights to
    # `model_path` itself -- that path doubles as the trainer's own resume
    # checkpoint, which must always stay raw (see EMAWeightPublisher's
    # docstring). The actor polls the separate EMA snapshot file instead.
    subscriber = WeightSubscriber(path=ema_publish_path(model_path), bundle=actor_bundle)
    # A resumed checkpoint is already a completed snapshot; load it before
    # the first tick rather than waiting for the first background publish.
    subscriber.maybe_reload()

    policy = ActorCriticPolicy(
        policy_model, critic_model, action_keys, action_space=action_space,
        history=args.actor_critic_history, training=args.actor_critic_train, seed=args.seed,
    )
    actor_learner = AsyncActorCriticLearner(
        policy, ring, weight_subscriber=subscriber,
        reload_every_ticks=args.async_reload_every_ticks,
    )

    class _ConcurrentLearner:
        """Learner adapter: supervises the trainer process alongside acting."""

        def __getattr__(self, name: str) -> Any:
            return getattr(actor_learner, name)

        def update(self, window: Any) -> None:
            actor_learner.update(window)
            # Cheap liveness check; restarts the trainer process if it died
            # (issue #37: "trainer crash must not kill the actor").
            supervisor.poll()

    learner = _ConcurrentLearner()
    # Stashed for `cmd_run` cleanup; not part of the `Learner` contract.
    learner.async_resources = (ring, supervisor)
    learner.sleep_trainer_supervisor = supervisor
    learner.weight_subscriber = subscriber
    return policy, learner


def _shutdown_async_trainer(learner: Optional[Any]) -> None:
    """Best-effort, always-runs cleanup for ``--async-trainer``: ask the
    trainer process to stop when using the legacy process resources, then
    release the shared-memory ring."""
    resources = getattr(learner, "async_resources", None)
    if resources is None:
        return
    ring = resources[0]
    try:
        finish = getattr(learner, "finish", None)
        if finish is not None:
            finish()
        if len(resources) == 2 and hasattr(resources[1], "stop"):
            resources[1].stop()
        elif len(resources) == 3:
            _, trainer_process, trainer_stop_event = resources
            trainer_stop_event.set()
            trainer_process.join(timeout=10)
            if trainer_process.is_alive():
                trainer_process.terminate()
                trainer_process.join(timeout=5)
    finally:
        ring.close()
        ring.unlink()


def _add_world_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--curriculum", default=None, choices=CURRICULUM_ORDER,
                        help="named curriculum preset: world config + reward weights + a "
                             "default seed, staged flat-safe -> resource-world -> "
                             "night-survival -> caves -> combat -> crafting (docs/curriculum.md); "
                             "an explicit flag below still overrides its world_config value")
    parser.add_argument("--seed", type=int, default=None,
                        help="base episode seed (default: the curriculum's seed, else 0)")
    parser.add_argument("--episode-ticks", type=int, default=None,
                        help="episode length in ticks (default: the curriculum's, else 6000)")
    parser.add_argument("--difficulty", type=float, default=None,
                        help="default: the curriculum's, else 1.0")
    parser.add_argument("--world-size", type=int, default=None,
                        help="default: the curriculum's, else 64")
    parser.add_argument("--day-length", type=int, default=None,
                        help="full day/night cycle in ticks; night is the second half "
                             "(default: the curriculum's, else 6000)")
    parser.add_argument("--start-time", type=int, default=None,
                        help="time of day at spawn (default: the curriculum's, else 0)")
    parser.add_argument("--max-mobs", type=int, default=None,
                        help="max concurrent hostile mobs (default: the curriculum's, else 3)")
    parser.add_argument("--pixel-source", choices=["viewer", "grid"], default=None,
                        help="remote backend pixel source: 'viewer' requests "
                             "prismarine-viewer first-person snapshots (default); "
                             "'grid' uses the compact colorized semantic-grid fallback")
    parser.add_argument("--model", default=None, help="path to a trained BC model (learned policy)")
    parser.add_argument("--backend", default="simulated", choices=sorted(BACKENDS),
                        help="survival backend: the deterministic simulated world, or "
                             "a real-Minecraft client (remote; not yet implemented)")
    parser.add_argument("--reward-profile", default=None,
                        help="path to a YAML/JSON reward profile (e.g. goals/survival.yaml, "
                             "goals/ender_dragon.yaml); overrides --curriculum's reward weights "
                             "with a profile-driven reward engine (docs/reward_profiles.md). "
                             "Malformed profiles fail immediately with a diagnosis.")
    parser.add_argument("--intrinsic-risk-threshold", type=float, default=0.5,
                        help="risk-gated intrinsic drive (issue #61): the internal.risk level "
                             "at which internal.safe_novelty's gate is cut in half (default: 0.5)")
    parser.add_argument("--intrinsic-risk-temperature", type=float, default=0.15,
                        help="risk-gated intrinsic drive (issue #61): softness of the risk-gate "
                             "sigmoid around --intrinsic-risk-threshold (default: 0.15)")


#: Programs that aren't Minecraft (issue #89). Default stays "minecraft" for
#: back-compat; ``--world crafter`` routes construction through
#: ``_build_program`` instead of the historical hardcoded ``MinecraftSurvivalBox``.
WORLDS = {"minecraft", "crafter"}


def _add_world_selector_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--world", default="minecraft", choices=sorted(WORLDS),
                        help="which Program to run: the deterministic Minecraft-like "
                             "survival sim (default, for back-compat), or the Crafter "
                             "nursery world (issue #89; needs the 'crafter' extra "
                             "installed). Both implement the same streams-v2 seam, so "
                             "the runtime/policy code is unchanged either way.")


def _build_program(args: argparse.Namespace, program_config: Dict[str, Any],
                    reward_config, reward_profile):
    """Construct the selected world's Program plus its stream/action
    registries -- the "small factory" the ``--world`` selector routes
    through (issue #89). ``reward_config``/``reward_profile`` only apply to
    ``--world minecraft``; Crafter has no reward-profile system yet (it uses
    the achievement/health reward the ``crafter`` package computes itself).
    """
    world = getattr(args, "world", "minecraft")
    if world == "minecraft":
        program = MinecraftSurvivalBox(
            config=program_config,
            reward_config=None if reward_profile else reward_config,
            backend=args.backend,
            reward_profile=reward_profile,
        )
        return program, MINECRAFT_STREAM_REGISTRY, MINECRAFT_ACTION_REGISTRY
    if world == "crafter":
        from cognitive_runtime.programs.crafter.action_registry import CRAFTER_ACTION_REGISTRY
        from cognitive_runtime.programs.crafter.adapter import CrafterWorld
        from cognitive_runtime.programs.crafter.stream_registry import CRAFTER_STREAM_REGISTRY

        # CrafterWorld imports the optional 'crafter' package lazily, inside
        # __init__ -- the ImportError (if it's not installed) only surfaces
        # at construction, not at the module import above.
        try:
            program = CrafterWorld(config=program_config)
        except ImportError as exc:
            sys.exit(str(exc))
        return program, CRAFTER_STREAM_REGISTRY, CRAFTER_ACTION_REGISTRY
    sys.exit(f"unknown --world {world!r}; expected one of {sorted(WORLDS)}")


#: Online-learning policies whose model path needs a checkpoint-or-`--fresh`
#: decision for live runs (issue #33).
_CHECKPOINTED_POLICIES = {"online": "online_model", "actor-critic": "actor_critic_model"}


def _enforce_live_run_protocol(args: argparse.Namespace) -> None:
    """Issue #33 Phase F: every live (``--backend remote``) run must start
    from a checkpoint bundle or explicitly opt out with ``--fresh``, and must
    always record the session including frames -- childhood runs are only
    reviewable if they were recorded, and interruption is only survivable if
    training started from (and saves back to) a checkpoint."""
    if args.backend != "remote":
        return
    if args.no_record:
        sys.exit(
            "live (--backend remote) runs must be recorded -- drop --no-record "
            "(issue #33: recordings are how a childhood run gets reviewed)."
        )
    args.record_frames = True
    model_attr = _CHECKPOINTED_POLICIES.get(args.policy)
    if model_attr is None:
        return
    model_path = getattr(args, model_attr)
    if not os.path.exists(model_path) and not args.fresh:
        sys.exit(
            f"live run: no checkpoint found at {model_path!r}. Pass --fresh to start "
            "a new checkpoint there, or point the model flag at an existing one "
            "(issue #33: live runs must start from a checkpoint or explicitly --fresh)."
        )


def cmd_run(args: argparse.Namespace) -> None:
    _resolve_world_args(args)
    world = getattr(args, "world", "minecraft")
    if world != "minecraft" and args.backend != "simulated":
        sys.exit(f"--backend only applies to --world minecraft (got --world {world!r})")
    if world != "minecraft" and args.curriculum is not None:
        sys.exit(f"--curriculum only applies to --world minecraft (got --world {world!r})")
    _enforce_live_run_protocol(args)
    program_config = _program_config(args)
    reward_profile = _reward_profile_for(args)
    if world != "minecraft" and reward_profile is not None:
        sys.exit(f"--reward-profile only applies to --world minecraft (got --world {world!r})")
    program, stream_registry, action_registry = _build_program(
        args, program_config, _reward_config_for(args), reward_profile,
    )
    encoders = _encoders_for_input_profile(args.input_profile, stream_registry)
    action_space = list(program.metadata().action_space)
    learner = None
    learned_fusion = None
    if args.policy == "online":
        policy, learner = _make_online_policy_and_learner(args, program, encoders)
    elif args.policy == "actor-critic":
        policy, learner, learned_fusion = _make_actor_critic_policy_and_learner(
            args, program, encoders, stream_registry,
        )
    else:
        policy = _make_policy(args.policy, args, action_space)
    world_model = _make_world_model(args, program)
    entity_persistence = _make_entity_persistence(args)
    config = RuntimeConfig(
        tick_rate=args.tick_rate,
        realtime=args.realtime,
        max_ticks_per_episode=args.episode_ticks,
        episodes=args.episodes,
        seed=args.seed,
        record=not args.no_record,
        record_dir=args.record_dir,
        record_frames=args.record_frames,
        record_streams=args.record_streams,
        exclude_streams=args.exclude_streams,
        frame_disk_budget_mb=args.frame_disk_budget_mb,
        pin_on_streams=args.pin_on_streams,
        session_id=args.session_id,
        name=args.name,
        program_config=program_config,
        curriculum=args.curriculum,
        attention_mode=getattr(args, "attention", "off"),
        reflex_mode=getattr(args, "reflex", "on"),
        intrinsic_risk_threshold=getattr(args, "intrinsic_risk_threshold", 0.5),
        intrinsic_risk_temperature=getattr(args, "intrinsic_risk_temperature", 0.15),
    )
    runtime = CognitiveRuntime(
        program=program,
        policy=policy,
        config=config,
        learner=learner,
        world_model=world_model,
        entity_persistence=entity_persistence,
        stream_registry=stream_registry,
        encoders=encoders,
        learned_fusion=learned_fusion,
        action_registry=action_registry,
    )
    try:
        summaries = runtime.run()
    finally:
        _shutdown_async_trainer(learner)
    for summary in summaries:
        stats = summary.program_stats
        print(
            f"{summary.episode_id}: policy={summary.policy_name} seed={summary.seed} "
            f"ticks={summary.duration_ticks} reward={summary.total_reward} "
            f"end={summary.termination_reason} items={stats.get('unique_items_collected')} "
            f"placed={stats.get('blocks_placed')} damage={stats.get('damage_taken')}"
        )
    if summaries:
        row = summarize_episodes(summaries)
        print("\naggregate:")
        print(comparison_table([row]))
    if not args.no_record:
        print(f"\nrecorded to {os.path.join(args.record_dir, runtime.recorder.session_id)}")


def cmd_demo(args: argparse.Namespace) -> None:
    args.policy = "human"
    args.realtime = False  # each tick blocks on human input instead
    args.no_record = False
    args.record_frames = True
    args.record_streams = ["*"]
    args.exclude_streams = []
    args.frame_disk_budget_mb = 512.0
    args.pin_on_streams = ["event.died", "event.damage_taken"]
    if args.session_id is None:
        import time as _time
        args.session_id = f"{_time.strftime('%Y%m%d-%H%M%S')}-human-demo"
    cmd_run(args)


def cmd_evaluate(args: argparse.Namespace) -> None:
    _resolve_world_args(args)
    program_config = _program_config(args)
    reward_profile = _reward_profile_for(args)
    reward_config = None if reward_profile else _reward_config_for(args)
    names = [p.strip() for p in args.policies.split(",") if p.strip()]
    factories: Dict[str, Callable[[], Policy]] = {}
    for name in names:
        factories[name] = (lambda n: (lambda: _make_policy(n, args)))(name)
    rows = compare_policies(
        program_factory=lambda: MinecraftSurvivalBox(
            config=program_config, reward_config=reward_config, backend=args.backend,
            reward_profile=reward_profile,
        ),
        policy_factories=factories,
        episodes=args.episodes,
        seed=args.seed,
        max_ticks=args.episode_ticks,
    )
    print(comparison_table(rows))


def cmd_statistical_evaluate(args: argparse.Namespace) -> None:
    """Statistical evaluation harness (issue #44): mean +/- CI over N episodes
    per policy/checkpoint, either freshly run in sim or loaded from already-
    recorded sessions (``--from-sessions``), with regression flagging against
    a named ``--baseline`` policy."""
    from cognitive_runtime.training.statistical_evaluation import (
        compare_statistics, evaluate_recorded_sessions,
        flagged_regressions, format_comparison_report, format_statistics_report,
        run_statistical_evaluation,
    )

    if args.from_sessions:
        by_group = evaluate_recorded_sessions(args.from_sessions, confidence=args.confidence)
        if not by_group:
            sys.exit(f"no recorded episodes found under {args.from_sessions!r}")
        stats: Dict[str, Any] = {
            (f"{policy} [{curriculum}]" if curriculum != "-" else policy): s
            for (curriculum, policy), s in sorted(by_group.items())
        }
    else:
        _resolve_world_args(args)
        program_config = _program_config(args)
        reward_profile = _reward_profile_for(args)
        reward_config = None if reward_profile else _reward_config_for(args)
        names = [p.strip() for p in args.policies.split(",") if p.strip()]
        stats = {}
        for name in names:
            stats[name] = run_statistical_evaluation(
                program_factory=lambda: MinecraftSurvivalBox(
                    config=program_config, reward_config=reward_config, backend=args.backend,
                    reward_profile=reward_profile,
                ),
                policy_factory=(lambda n: (lambda: _make_policy(n, args)))(name),
                episodes=args.episodes,
                seed=args.seed,
                max_ticks=args.episode_ticks,
                record_dir=args.record_dir,
                session_id=f"stat-eval-{name}" if args.record_dir else None,
                confidence=args.confidence,
            )

    print(format_statistics_report(list(stats.values())))

    if args.baseline:
        baseline = stats.get(args.baseline)
        if baseline is None:
            sys.exit(f"--baseline {args.baseline!r} not among evaluated groups: {sorted(stats)}")
        for name, candidate in stats.items():
            if name == args.baseline:
                continue
            comparisons = compare_statistics(baseline, candidate)
            regressions = flagged_regressions(comparisons)
            print(f"\n{name} vs baseline {args.baseline!r}:")
            print(format_comparison_report(comparisons))
            if regressions:
                print(f"  ** {len(regressions)} statistically significant regression(s) **")


def cmd_train(args: argparse.Namespace) -> None:
    if args.model_type == "neural":
        _train_neural(args)
        return
    if args.model_type == "pixel-encoder":
        _train_pixel_encoder(args)
        return
    if args.model_type == "fusion":
        _train_latent_fusion(args)
        return
    if args.model_type == "world-model":
        _train_world_model(args)
        return
    if args.model_type == "multi-horizon-world-model":
        _train_multi_horizon_world_model(args)
        return
    if args.model_type == "entity-persistence":
        _train_entity_persistence(args)
        return
    dataset = build_dataset(
        args.sessions,
        history=args.history,
        max_samples=args.max_samples,
        min_episode_reward=args.min_reward,
        representation=args.features,
    )
    if len(dataset) == 0:
        sys.exit("no training samples found (were the sessions recorded as streams-v2?)")
    print(f"dataset: {len(dataset)} samples from {len(dataset.sources)} episodes "
          f"({dataset.representation} features, dim={len(dataset.feature_names)})")
    model, metrics = train_bc(
        dataset, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model.save(args.out)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"model saved to {args.out}")


def _train_neural(args: argparse.Namespace) -> None:
    """Pixel-vision end-to-end BC.  torch is imported here so the default
    (linear) training path never requires it."""
    try:
        from cognitive_runtime.training.datasets import build_neural_dataset
        from cognitive_runtime.training.neural import train_neural_bc
    except ImportError as exc:  # torch not installed
        sys.exit(
            f"neural training needs PyTorch ({exc}). Install it with "
            "'pip install -e .[neural]'."
        )
    dataset = build_neural_dataset(
        args.sessions,
        history=args.history,
        max_samples=args.max_samples,
        min_episode_reward=args.min_reward,
        stream_profile=args.stream_profile,
    )
    if len(dataset) == 0:
        sys.exit("no pixel training samples found (record sessions with --record-frames)")
    print(f"dataset: {len(dataset)} pixel samples from {len(dataset.sources)} episodes "
          f"(frame={dataset.pixel_shape}, non-vision dim={len(dataset.non_vision_names)}, "
          f"stream_profile={dataset.stream_profile})")
    model, metrics = train_neural_bc(
        dataset,
        epochs=args.epochs,
        lr=args.neural_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        embed_dim=args.latent_width,
        encoder_init_path=args.encoder_init,
    )
    out = args.out if args.out != DEFAULT_MODEL_OUT else "models/vision_bc.pt"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    model.save(out)
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"model saved to {out}")


def _train_pixel_encoder(args: argparse.Namespace) -> None:
    """Offline visual representation pretraining for PixelStreamEncoder."""
    try:
        from cognitive_runtime.training.datasets import build_pixel_sequence_dataset
        from cognitive_runtime.training.visual_representation import (
            VisualPretrainingConfig,
            save_pixel_encoder_pretraining_checkpoint,
            train_pixel_encoder_pretraining,
        )
    except ImportError as exc:  # torch not installed
        sys.exit(
            f"pixel-encoder pretraining needs PyTorch ({exc}). Install it with "
            "'pip install -e .[neural]'."
        )
    dataset = build_pixel_sequence_dataset(
        args.sessions,
        max_samples=args.max_samples,
        min_episode_reward=args.min_reward,
    )
    if len(dataset) == 0:
        sys.exit("no adjacent pixel samples found (record sessions with --record-frames)")
    print(
        f"dataset: {len(dataset)} adjacent pixel pairs from {len(dataset.sources)} episodes "
        f"(frame={dataset.pixel_shape})"
    )
    config = VisualPretrainingConfig(
        epochs=args.epochs,
        lr=args.neural_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        latent_width=args.latent_width,
        hidden_dim=args.hidden_dim,
        reconstruction_size=args.reconstruction_size,
        reconstruction_weight=args.reconstruction_weight,
        next_latent_weight=args.next_latent_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
    )
    model, stats = train_pixel_encoder_pretraining(dataset, config)
    out = args.out if args.out != DEFAULT_MODEL_OUT else "models/pixel_encoder.pt"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_pixel_encoder_pretraining_checkpoint(out, model, dataset, stats, name=args.name)
    for key in (
        "final_total_loss",
        "final_reconstruction_loss",
        "final_next_latent_loss",
        "final_contrastive_loss",
    ):
        print(f"  {key}: {stats[key]}")
    print(f"checkpoint bundle saved to {out}")


def _train_latent_fusion(args: argparse.Namespace) -> None:
    """Offline Phase-C learned latent fusion training."""
    try:
        from cognitive_runtime.training.datasets import build_latent_fusion_dataset
        from cognitive_runtime.training.fusion import (
            FusionTrainingConfig,
            save_latent_fusion_checkpoint,
            train_latent_fusion_model,
        )
    except ImportError as exc:  # torch not installed
        sys.exit(
            f"latent fusion training needs PyTorch ({exc}). Install it with "
            "'pip install -e .[neural]'."
        )
    dataset = build_latent_fusion_dataset(
        args.sessions,
        max_samples=args.max_samples,
        min_episode_reward=args.min_reward,
    )
    if len(dataset) == 0:
        sys.exit("no fusion training samples found (were the sessions recorded as streams-v2?)")
    print(
        f"dataset: {len(dataset)} fusion samples from {len(dataset.sources)} episodes "
        f"(streams={len(dataset.stream_ids)}, dim={len(dataset.feature_names)})"
    )
    config = FusionTrainingConfig(
        epochs=args.epochs,
        lr=args.neural_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        fused_width=args.latent_width,
        hidden_dim=args.hidden_dim,
        depth=args.fusion_depth,
        dropout=args.fusion_dropout,
    )
    model, stats = train_latent_fusion_model(dataset, config)
    out = args.out if args.out != DEFAULT_MODEL_OUT else "models/latent_fusion.pt"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_latent_fusion_checkpoint(out, model, dataset, stats, name=args.name)
    for key in (
        "final_action_loss",
        "final_reward_loss",
        "final_next_latent_loss",
        "final_total_loss",
    ):
        print(f"  {key}: {stats[key]}")
    print(f"checkpoint bundle saved to {out}")


def _train_world_model(args: argparse.Namespace) -> None:
    """Offline Phase-D action-conditioned world-model training (issue #26)."""
    try:
        from cognitive_runtime.training.datasets import build_world_model_dataset
        from cognitive_runtime.training.world_model import (
            WorldModelTrainingConfig,
            death_prediction_auc,
            save_world_model_checkpoint,
            train_world_model,
        )
    except ImportError as exc:  # torch not installed
        sys.exit(
            f"world-model training needs PyTorch ({exc}). Install it with "
            "'pip install -e .[neural]'."
        )
    dataset = build_world_model_dataset(
        args.sessions,
        max_samples=args.max_samples,
        min_episode_reward=args.min_reward,
    )
    if len(dataset) == 0:
        sys.exit("no world-model training samples found (were the sessions recorded as streams-v2?)")
    print(
        f"dataset: {len(dataset)} transitions ({dataset.death_count()} death-preceding) "
        f"from {len(dataset.sources)} episodes (dim={len(dataset.feature_names)})"
    )
    config = WorldModelTrainingConfig(
        epochs=args.epochs,
        lr=args.neural_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        depth=args.fusion_depth,
        dropout=args.fusion_dropout,
    )
    model, stats = train_world_model(dataset, config)
    out = args.out if args.out != DEFAULT_MODEL_OUT else "models/world_model.pt"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_world_model_checkpoint(out, model, dataset, stats, name=args.name)
    for key in (
        "final_next_latent_loss",
        "final_reward_loss",
        "final_death_loss",
        "final_risk_loss",
        "final_prediction_error_loss",
        "final_total_loss",
    ):
        print(f"  {key}: {stats[key]}")
    if dataset.death_count() > 0:
        try:
            auc = death_prediction_auc(model, dataset)
            print(f"  death_prediction_auc (in-sample): {round(auc, 4)}")
        except ValueError as exc:
            print(f"  death_prediction_auc: skipped ({exc})")
    print(f"checkpoint bundle saved to {out}")


def _train_multi_horizon_world_model(args: argparse.Namespace) -> None:
    """Offline multi-horizon, uncertainty-aware world-model training
    (issue #39): predicts next_latent/reward/terminal/risk/prediction_error
    at every ``--horizons`` tick offset, each with a learned uncertainty."""
    try:
        from cognitive_runtime.training.datasets import build_multi_horizon_world_model_dataset
        from cognitive_runtime.training.world_model import (
            MultiHorizonWorldModelTrainingConfig,
            save_multi_horizon_world_model_checkpoint,
            train_multi_horizon_world_model,
        )
    except ImportError as exc:  # torch not installed
        sys.exit(
            f"multi-horizon world-model training needs PyTorch ({exc}). Install it with "
            "'pip install -e .[neural]'."
        )
    dataset = build_multi_horizon_world_model_dataset(
        args.sessions,
        horizons=args.horizons,
        max_samples=args.max_samples,
        min_episode_reward=args.min_reward,
    )
    if len(dataset) == 0:
        sys.exit(
            "no multi-horizon world-model training samples found (were the sessions "
            "recorded as streams-v2, and long enough for the largest --horizons value?)"
        )
    print(
        f"dataset: {len(dataset)} samples at horizons {dataset.horizons} from "
        f"{len(dataset.sources)} episodes (dim={len(dataset.feature_names)})"
    )
    config = MultiHorizonWorldModelTrainingConfig(
        epochs=args.epochs,
        lr=args.neural_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        depth=args.fusion_depth,
        dropout=args.fusion_dropout,
    )
    model, stats = train_multi_horizon_world_model(dataset, config)
    out = args.out if args.out != DEFAULT_MODEL_OUT else "models/multi_horizon_world_model.pt"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_multi_horizon_world_model_checkpoint(out, model, dataset, stats, name=args.name)
    for h, entry in stats["evaluation"].items():
        print(
            f"  horizon t+{h}: model_mse={round(entry['model_mse'], 4)} "
            f"copy_last_mse={round(entry['copy_last_mse'], 4)} "
            f"mean_latent_mse={round(entry['mean_latent_mse'], 4)} "
            f"beats_copy_last={entry['beats_copy_last']} "
            f"beats_mean_latent={entry['beats_mean_latent']} "
            f"uncertainty_error_correlation={round(entry['uncertainty_error_correlation'], 4)}"
        )
    print(f"checkpoint bundle saved to {out}")


def _train_entity_persistence(args: argparse.Namespace) -> None:
    """Offline entity-persistence training (issue #27: object permanence).

    Learns to predict a tracked mob's feature during an occlusion gap from
    every occlusion-then-reappearance recorded sessions went through --
    record with a mix of night/combat episodes so mobs actually go behind
    walls and come back.
    """
    try:
        from cognitive_runtime.training.entity_persistence import (
            EntityPersistenceTrainingConfig,
            build_entity_persistence_dataset,
            save_entity_persistence_checkpoint,
            train_entity_persistence_model,
        )
    except ImportError as exc:  # torch not installed
        sys.exit(
            f"entity-persistence training needs PyTorch ({exc}). Install it with "
            "'pip install -e .[neural]'."
        )
    dataset = build_entity_persistence_dataset(args.sessions, max_samples=args.max_samples)
    if len(dataset) == 0:
        sys.exit(
            "no entity-persistence training samples found: no tracked mob was ever "
            "occluded and then reappeared in these sessions (record night/combat "
            "episodes where mobs walk behind walls)"
        )
    print(
        f"dataset: {len(dataset)} occlusion/reappearance samples from "
        f"{len(dataset.sources)} episodes (baseline_mse={round(dataset.baseline_mse(), 4)})"
    )
    config = EntityPersistenceTrainingConfig(
        epochs=args.epochs,
        lr=args.neural_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
    )
    model, stats = train_entity_persistence_model(dataset, config)
    out = args.out if args.out != DEFAULT_MODEL_OUT else "models/entity_persistence.pt"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_entity_persistence_checkpoint(out, model, dataset, stats, name=args.name)
    for key in ("final_feature_loss", "final_surprise_loss", "final_total_loss",
                "baseline_mse", "model_mse", "beats_forget_baseline"):
        print(f"  {key}: {stats[key]}")
    print(f"checkpoint bundle saved to {out}")


def cmd_ego_motion_canary(args: argparse.Namespace) -> None:
    """``ccr ego-motion-canary`` (issue #39): generate ``walk_forward``
    episodes at multiple seeds via the simulated backend, train a next-frame
    predictor on a train-seed subset only, and evaluate held-out-seed
    next-frame prediction (PSNR/SSIM, iterated rollout to every
    ``--horizons`` tick offset) against copy-last-frame and mean-frame
    baselines.
    """
    try:
        from cognitive_runtime.training.ego_motion_canary import (
            EgoMotionCanaryConfig,
            run_ego_motion_canary,
            save_ego_motion_canary_checkpoint,
        )
    except ImportError as exc:  # torch not installed
        sys.exit(
            f"the ego-motion canary needs PyTorch ({exc}). Install it with "
            "'pip install -e .[neural]'."
        )
    train_seeds = list(range(args.train_seeds))
    holdout_seeds = list(range(args.train_seeds, args.train_seeds + args.holdout_seeds))
    config = EgoMotionCanaryConfig(
        train_seeds=train_seeds,
        holdout_seeds=holdout_seeds,
        episode_ticks=args.episode_ticks,
        world_size=args.world_size,
        action_noise=args.action_noise,
        horizons=args.horizons,
        latent_width=args.latent_width,
        hidden_dim=args.hidden_dim,
        reconstruction_size=args.reconstruction_size,
        epochs=args.epochs,
        lr=args.neural_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        consistency_epochs=args.consistency_epochs,
    )
    print(
        f"recording {len(train_seeds)} train seeds {train_seeds} and "
        f"{len(holdout_seeds)} held-out seeds {holdout_seeds} "
        f"({config.episode_ticks} ticks each, world_size={config.world_size})"
    )
    model, report = run_ego_motion_canary(args.record_dir, config)
    for h, entry in report.horizon_metrics.items():
        print(
            f"  horizon t+{h} (n={entry['n_samples']}): "
            f"psnr model={round(entry['psnr_model'], 2)} "
            f"copy_last={round(entry['psnr_copy_last'], 2)} "
            f"mean_frame={round(entry['psnr_mean_frame'], 2)} | "
            f"ssim model={round(entry['ssim_model'], 4)} "
            f"copy_last={round(entry['ssim_copy_last'], 4)} "
            f"mean_frame={round(entry['ssim_mean_frame'], 4)} | "
            f"beats_copy_last={entry['beats_copy_last']} "
            f"beats_mean_frame={entry['beats_mean_frame']}"
        )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        save_ego_motion_canary_checkpoint(args.out, model, report)
        print(f"checkpoint bundle saved to {args.out}")


def cmd_nursery_list(args: argparse.Namespace) -> None:
    """``ccr nursery list`` (issue #62): print every registered nursery
    scenario -- scripted micro-scenarios that isolate one worldly regularity
    each, feeding checkpoints into the survival curriculum's stage one.
    ``--world crafter`` (issue #90) lists the Crafter ports instead."""
    try:
        from cognitive_runtime.training.nursery import _scenarios_for_world
    except ImportError as exc:  # torch not installed
        sys.exit(f"the nursery suite needs PyTorch ({exc}). Install it with 'pip install -e .[neural]'.")
    scenarios = _scenarios_for_world(getattr(args, "world", "minecraft"))
    for name in sorted(scenarios):
        scenario = scenarios[name]
        tag = " [+entity-persistence metric]" if scenario.entity_persistence_metric else ""
        print(f"{name}{tag}: {scenario.description}")


def cmd_nursery_run(args: argparse.Namespace) -> None:
    """``ccr nursery run <scenario|all>`` (issue #62): record train/holdout
    episodes for one nursery scenario (or every scenario via ``all``),
    pretrain a pixel encoder+decoder+next-latent predictor on the train
    seeds only, and evaluate multi-horizon next-frame prediction on
    held-out seeds against copy-last-frame and mean-frame baselines --
    generalizing ``ego-motion-canary`` (issue #39) into a suite.
    ``object_permanence`` also reports an entity-persistence metric (issue
    #27); every held-out episode gets a rendered dream strip (predicted vs.
    actual frames at each horizon).
    """
    try:
        import json

        from cognitive_runtime.training.nursery import (
            NurseryConfig,
            _scenarios_for_world,
            run_nursery_scenario,
            save_nursery_scenario_checkpoint,
        )
    except ImportError as exc:  # torch not installed
        sys.exit(f"the nursery suite needs PyTorch ({exc}). Install it with 'pip install -e .[neural]'.")

    world = getattr(args, "world", "minecraft")
    scenarios = _scenarios_for_world(world)
    if args.scenario != "all" and args.scenario not in scenarios:
        sys.exit(
            f"unknown nursery scenario {args.scenario!r} for --world {world!r}; choices: "
            f"{sorted(scenarios)} or 'all'"
        )
    scenario_names = sorted(scenarios) if args.scenario == "all" else [args.scenario]

    train_seeds = list(range(args.train_seeds))
    holdout_seeds = list(range(args.train_seeds, args.train_seeds + args.holdout_seeds))
    config = NurseryConfig(
        train_seeds=train_seeds,
        holdout_seeds=holdout_seeds,
        episode_ticks=args.episode_ticks,
        world_size=args.world_size,
        world=world,
        backend=args.backend,
        realtime=args.realtime or args.backend == "remote",
        horizons=args.horizons,
        latent_width=args.latent_width,
        hidden_dim=args.hidden_dim,
        reconstruction_size=args.reconstruction_size,
        epochs=args.epochs,
        lr=args.neural_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        consistency_epochs=args.consistency_epochs,
        entity_persistence_epochs=args.entity_persistence_epochs,
        data_quality_gate=not args.skip_data_quality_gate,
        export_predictions=not args.no_export_predictions,
        name=args.name,
    )
    backend_note = f"backend={config.backend}" if world == "minecraft" else f"world={world}"
    print(
        f"nursery: running {'all scenarios' if args.scenario == 'all' else args.scenario} "
        f"({len(train_seeds)} train seeds, {len(holdout_seeds)} held-out seeds, "
        f"{config.episode_ticks} ticks each, world_size={config.world_size}, {backend_note})"
    )
    if world == "minecraft" and config.backend != "simulated":
        print(
            "nursery: WARNING -- the remote backend plays on the server's persistent "
            "world: seeds do NOT vary terrain, each session starts where the previous "
            "one ended, sim-only scenario setup hooks are skipped, and realtime pacing "
            "records vision at the config's realtime_vision_hz (10 Hz by default), not "
            "the 20 Hz tick rate. The data-quality gate will reject recordings without "
            "the scenario's signal (e.g. a stuck agent)."
        )

    report_payload: Dict[str, Any] = {}
    for name in scenario_names:
        model, report = run_nursery_scenario(args.record_dir, name, config)
        print(f"\n{name}:")
        for h, entry in report.horizon_metrics.items():
            print(
                f"  horizon t+{h} (n={entry['n_samples']}): "
                f"psnr model={round(entry['psnr_model'], 2)} "
                f"copy_last={round(entry['psnr_copy_last'], 2)} "
                f"mean_frame={round(entry['psnr_mean_frame'], 2)} | "
                f"ssim model={round(entry['ssim_model'], 4)} "
                f"copy_last={round(entry['ssim_copy_last'], 4)} "
                f"mean_frame={round(entry['ssim_mean_frame'], 4)} | "
                f"beats_copy_last={entry['beats_copy_last']} "
                f"beats_mean_frame={entry['beats_mean_frame']}"
            )
        if report.ticks_per_frame > 1.05:
            print(
                f"  vision ran at ~1 frame per {round(report.ticks_per_frame, 2)} ticks; "
                f"tick horizons {list(config.horizons)} evaluated as frame steps "
                f"{report.horizon_frames}"
            )
        health = report.rollout_health
        if health.get("frozen_rollout"):
            print(
                "  WARNING: FROZEN ROLLOUT -- predictions barely vary across horizons "
                f"(prediction dispersion {health['prediction_dispersion']:.2e} vs actual "
                f"{health['target_dispersion']:.2e}); the predictor has collapsed to a "
                "fixed point and is not modelling the dynamics"
            )
        if report.entity_persistence_stats is not None:
            eps = report.entity_persistence_stats
            if "beats_forget_baseline" in eps:
                print(
                    f"  entity persistence: model_mse={round(eps['model_mse'], 4)} "
                    f"baseline_mse={round(eps['baseline_mse'], 4)} "
                    f"beats_forget_baseline={eps['beats_forget_baseline']}"
                )
            else:
                print(f"  entity persistence: {eps.get('note', eps)}")
        print(f"  dream strips rendered: {len(report.dream_strips)}")

        if report.prediction_files:
            print(f"  viewer predictions exported: {len(report.prediction_files)} episode(s)")

        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.out_dir, f"{name}.pt")
            save_nursery_scenario_checkpoint(checkpoint_path, model, report)
            print(f"  checkpoint saved to {checkpoint_path}")
            # The unified checkpoint keeps only the encoder; the full bundle
            # (encoder+decoder+next-predictor) lets the prediction exporter
            # re-render predicted frames later without retraining.
            from cognitive_runtime.training.prediction_export import save_full_visual_model

            full_model_path = os.path.join(args.out_dir, f"{name}-full.pt")
            save_full_visual_model(model, full_model_path)
            print(f"  full model bundle saved to {full_model_path}")

        report_payload[name] = {
            "horizon_metrics": {str(h): v for h, v in report.horizon_metrics.items()},
            "horizon_frames": report.horizon_frames,
            "ticks_per_frame": report.ticks_per_frame,
            "rollout_health": report.rollout_health,
            "entity_persistence_stats": report.entity_persistence_stats,
            "dream_strips": report.dream_strips,
            "train_sessions": report.train_sessions,
            "holdout_sessions": report.holdout_sessions,
            "prediction_files": report.prediction_files,
        }

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report_payload, fh, indent=2)
        print(f"\nreport written to {args.report}")


def cmd_nursery_joint(args: argparse.Namespace) -> None:
    """``ccr nursery joint``: record every scenario and train ONE
    action-conditioned recurrent world model across them (phase 3 of
    docs/nursery-turn-in-place-analysis.md), evaluating in-distribution
    generalization (held-out seeds), zero-shot generality (held-out
    scenarios), rollout health (frozen-rollout detector), and a yaw linear
    probe."""
    try:
        import json

        from cognitive_runtime.training.action_world_model import (
            ActionWorldModelConfig,
            save_action_world_model,
        )
        from cognitive_runtime.training.nursery import (
            NURSERY_SCENARIOS,
            NurseryConfig,
            run_nursery_joint,
        )
    except ImportError as exc:  # torch not installed
        sys.exit(f"the nursery suite needs PyTorch ({exc}). Install it with 'pip install -e .[neural]'.")

    holdout_scenarios = args.holdout_scenarios or ["approach_entity"]
    train_scenarios = args.train_scenarios or None
    for name in (train_scenarios or []) + holdout_scenarios:
        if name not in NURSERY_SCENARIOS:
            sys.exit(
                f"unknown nursery scenario {name!r}; choices: {sorted(NURSERY_SCENARIOS)}"
            )

    train_seeds = list(range(args.train_seeds))
    holdout_seeds = list(range(args.train_seeds, args.train_seeds + args.holdout_seeds))
    config = NurseryConfig(
        train_seeds=train_seeds,
        holdout_seeds=holdout_seeds,
        episode_ticks=args.episode_ticks,
        world_size=args.world_size,
        backend=args.backend,
        realtime=args.realtime or args.backend == "remote",
        horizons=args.horizons,
        latent_width=args.latent_width,
        hidden_dim=args.hidden_dim,
        reconstruction_size=args.reconstruction_size,
        epochs=args.epochs,
        lr=args.neural_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        data_quality_gate=not args.skip_data_quality_gate,
    )
    model_config = ActionWorldModelConfig(
        latent_width=args.latent_width,
        hidden_dim=args.hidden_dim,
        reconstruction_size=args.reconstruction_size,
        epochs=args.epochs,
        lr=args.neural_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        warmup_frames=args.warmup_frames,
        rollout_frames=args.rollout_frames,
        backbone=args.backbone,
        context_length=args.context_length,
    )
    print(
        f"nursery joint: training one action-conditioned world model "
        f"(holdout scenarios: {holdout_scenarios}; {len(train_seeds)} train seeds, "
        f"{len(holdout_seeds)} held-out seeds, backend={config.backend})"
    )

    model, report = run_nursery_joint(
        args.record_dir,
        train_scenarios=train_scenarios,
        holdout_scenarios=holdout_scenarios,
        config=config,
        model_config=model_config,
    )

    def _print_metrics(label: str, metrics: Dict[str, Any]) -> None:
        print(f"\n{label}:")
        for h, entry in metrics["horizons"].items():
            oracle = entry["model_over_oracle_mse"]
            print(
                f"  t+{h} frames (n={entry['n_samples']}): "
                f"model_mse={entry['model_mse']:.5f} "
                f"copy_last={entry['copy_last_mse']:.5f} "
                f"model/copy_last={entry['model_over_copy_last_mse']:.2f} "
                f"model/oracle={f'{oracle:.2f}' if oracle is not None else 'n/a'} "
                f"beats_copy_last={entry['beats_copy_last']}"
            )
        health = metrics["rollout_health"]
        if health.get("frozen_rollout"):
            print(
                "  WARNING: FROZEN ROLLOUT (prediction dispersion "
                f"{health['prediction_dispersion']:.2e} vs actual "
                f"{health['target_dispersion']:.2e})"
            )

    if report.ticks_per_frame > 1.05:
        print(
            f"vision ran at ~1 frame per {round(report.ticks_per_frame, 2)} ticks; "
            f"tick horizons {list(config.horizons)} evaluated as frame steps "
            f"{report.horizon_frames}"
        )
    for name, metrics in report.scenario_metrics.items():
        _print_metrics(f"{name} (held-out seeds)", metrics)
    for name, metrics in report.zero_shot_metrics.items():
        _print_metrics(f"{name} (ZERO-SHOT scenario)", metrics)

    probe = report.yaw_probe
    if "latent" in probe:
        print(
            f"\nyaw probe (n={probe['n_samples']}): "
            f"latent r2={probe['latent']['r2']:.3f} "
            f"({probe['latent']['mean_angular_error_deg']:.1f} deg err), "
            f"hidden r2={probe['hidden']['r2']:.3f} "
            f"({probe['hidden']['mean_angular_error_deg']:.1f} deg err)"
        )

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        model_path = os.path.join(args.out_dir, "joint-world-model.pt")
        save_action_world_model(model_path, model, report.training_stats)
        print(f"\njoint world model saved to {model_path}")

    if args.report:
        payload = {
            "train_scenarios": report.train_scenarios,
            "holdout_scenarios": report.holdout_scenarios,
            "horizon_frames": report.horizon_frames,
            "ticks_per_frame": report.ticks_per_frame,
            "training_stats": report.training_stats,
            "scenario_metrics": report.scenario_metrics,
            "zero_shot_metrics": report.zero_shot_metrics,
            "yaw_probe": report.yaw_probe,
            "train_sessions": report.train_sessions,
            "eval_sessions": report.eval_sessions,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"report written to {args.report}")


def cmd_nursery_backbone_benchmark(args: argparse.Namespace) -> None:
    """``ccr nursery backbone-benchmark`` (issue #93): train the cortex once
    per temporal backbone on identical recordings and report GRU vs
    dilated-conv/transformer on the Phase 2 scoring gates (model/copy-last,
    model/oracle, frozen-rollout) per horizon."""
    try:
        import json

        from cognitive_runtime.training.action_world_model import ActionWorldModelConfig
        from cognitive_runtime.training.nursery import (
            NURSERY_SCENARIOS,
            NurseryConfig,
            run_backbone_benchmark,
        )
    except ImportError as exc:  # torch not installed
        sys.exit(f"the nursery suite needs PyTorch ({exc}). Install it with 'pip install -e .[neural]'.")

    for name in args.train_scenarios:
        if name not in NURSERY_SCENARIOS:
            sys.exit(f"unknown nursery scenario {name!r}; choices: {sorted(NURSERY_SCENARIOS)}")
    if args.eval_scenario not in args.train_scenarios:
        sys.exit(
            f"--eval-scenario {args.eval_scenario!r} must be one of --train-scenarios "
            f"{args.train_scenarios!r}"
        )

    train_seeds = list(range(args.train_seeds))
    holdout_seeds = list(range(args.train_seeds, args.train_seeds + args.holdout_seeds))
    config = NurseryConfig(
        train_seeds=train_seeds,
        holdout_seeds=holdout_seeds,
        episode_ticks=args.episode_ticks,
        world_size=args.world_size,
        backend=args.backend,
        realtime=args.realtime or args.backend == "remote",
        horizons=args.horizons,
        latent_width=args.latent_width,
        hidden_dim=args.hidden_dim,
        reconstruction_size=args.reconstruction_size,
        epochs=args.epochs,
        lr=args.neural_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        data_quality_gate=not args.skip_data_quality_gate,
    )
    model_config = ActionWorldModelConfig(
        latent_width=args.latent_width,
        hidden_dim=args.hidden_dim,
        reconstruction_size=args.reconstruction_size,
        epochs=args.epochs,
        lr=args.neural_lr,
        batch_size=args.batch_size,
        seed=args.seed,
        warmup_frames=args.warmup_frames,
        rollout_frames=args.rollout_frames,
        context_length=args.context_length,
    )
    print(
        f"nursery backbone-benchmark: {args.backbones} on {args.eval_scenario!r} "
        f"({len(train_seeds)} train seeds, {len(holdout_seeds)} held-out seeds)"
    )

    report = run_backbone_benchmark(
        args.record_dir,
        train_scenarios=args.train_scenarios,
        eval_scenario=args.eval_scenario,
        backbones=args.backbones,
        baseline_backbone=args.baseline_backbone,
        config=config,
        model_config=model_config,
    )

    for name in report.metrics:
        print(f"\n{name}:")
        for h, entry in report.metrics[name]["horizons"].items():
            oracle = entry["model_over_oracle_mse"]
            print(
                f"  t+{h} frames (n={entry['n_samples']}): "
                f"model_mse={entry['model_mse']:.5f} "
                f"model/copy_last={entry['model_over_copy_last_mse']:.2f} "
                f"model/oracle={f'{oracle:.2f}' if oracle is not None else 'n/a'} "
                f"beats_copy_last={entry['beats_copy_last']}"
            )
        health = report.metrics[name]["rollout_health"]
        if health.get("frozen_rollout"):
            print(f"  WARNING: FROZEN ROLLOUT ({name})")
        if name != report.baseline_backbone:
            for h, comparison in report.comparisons[name].items():
                print(f"  vs {report.baseline_backbone} at t+{h}: {comparison.direction}")

    if args.report:
        payload = {
            "train_scenarios": report.train_scenarios,
            "eval_scenario": report.eval_scenario,
            "baseline_backbone": report.baseline_backbone,
            "metrics": report.metrics,
            "stats": {
                name: {h: s.to_dict() for h, s in horizon_stats.items()}
                for name, horizon_stats in report.stats.items()
            },
            "comparisons": {
                name: {h: c.to_dict() for h, c in horizon_comparisons.items()}
                for name, horizon_comparisons in report.comparisons.items()
            },
            "beats_copy_last": report.beats_copy_last,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"report written to {args.report}")


def cmd_trainer(args: argparse.Namespace) -> None:
    """``ccr trainer`` (issue #37): run an ``AsyncTrainer`` to completion in
    the foreground, pointed only at recorded sessions and no live actor --
    "the same trainer, pointed only at recorded sessions with no live
    actor, performs offline pretraining." The output checkpoint is exactly
    what ``run --async-trainer`` (or the synchronous ``--policy
    actor-critic``) reads back in.
    """
    try:
        import multiprocessing as mp

        from cognitive_runtime.policies.actor_critic import world_feature_width
        from sleep.async_trainer import ActorCriticArch, AsyncTrainer
    except ImportError as exc:  # torch not installed
        sys.exit(f"the trainer needs PyTorch ({exc}); install '.[neural]'.")
    from cognitive_runtime.core.streams import TemporalFusion
    from cognitive_runtime.core.streams.events import StreamSpec
    from cognitive_runtime.runtime.replay import load_session_metadata, require_streams_v2

    metadata = load_session_metadata(args.sessions[0])
    require_streams_v2(metadata)
    catalog = [StreamSpec.from_dict(s) for s in metadata.get("stream_catalog", [])]
    fusion = TemporalFusion(catalog)
    action_keys = tuple(metadata.get("action_space", []))
    if not action_keys:
        sys.exit(f"session {args.sessions[0]!r} has no recorded action_space")

    arch = ActorCriticArch(
        fused_width=fusion.width,
        world_feature_width=world_feature_width(action_keys),
        n_actions=len(action_keys),
        action_keys=action_keys,
        layout_hash=fusion.layout_hash,
        hidden_dim=args.hidden_dim,
        has_world_model=args.world_model_loss,
    )
    trainer = AsyncTrainer(
        arch, args.out,
        lr=args.lr, gamma=args.gamma, entropy_coef=args.entropy_coef,
        grad_clip_norm=args.grad_clip_norm, seed=args.seed,
        session_dirs=args.sessions,
        max_transitions_from_sessions=args.max_transitions,
        min_episode_reward=args.min_episode_reward,
        batch_size=args.batch_size,
        min_buffer_size=1,
        publish_every_steps=args.publish_every,
    )
    resumed = trainer.resume_if_checkpoint_exists()
    loaded = trainer.load_recorded_sessions()
    print(
        f"{'resumed from checkpoint; ' if resumed else ''}"
        f"loaded {loaded} transitions from {len(args.sessions)} session(s)"
    )
    if loaded == 0:
        sys.exit("no training transitions found in the given sessions")

    stats = trainer.run_forever(mp.Event(), max_steps=args.steps)
    print(f"trained {stats['step_count']} steps; checkpoint written to {args.out}")
    print(f"last metrics: {stats['last_metrics']}")


def cmd_replay(args: argparse.Namespace) -> None:
    reward_profile = _reward_profile_for(args)
    try:
        results = replay_session(
            args.session, episode_id=args.episode, verify=not args.no_verify,
            reward_profile=reward_profile,
        )
    except NonDeterministicSessionError as exc:
        sys.exit(f"replay skipped: {exc}")
    except ValueError as exc:
        sys.exit(str(exc))
    print(format_results(results))
    if any(not r.matched for r in results):
        sys.exit(1)


def cmd_view(args: argparse.Namespace) -> None:
    print(view_episode(args.session, args.episode, tail=args.tail))


def cmd_evaluation_gates(args: argparse.Namespace) -> None:
    """The evaluation-gate one-liner (issue #31, docs/neural-stream-agent.md
    Phase E): train actor/critic and linear online-Q, eval both plus
    scripted/random on identical seeds, and report the three deprecation gates.
    Recorded eval sessions are summarizable with ``dashboard --record-dir``."""
    try:
        from cognitive_runtime.training.evaluation_gates import run_evaluation_gates
    except ImportError as exc:  # torch not installed
        sys.exit(f"the evaluation gates need PyTorch ({exc}); install '.[neural]'.")

    result = run_evaluation_gates(
        curriculum=args.curriculum,
        config=None,  # curriculum preset or the default gate config supplies it
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        record_dir=None if args.no_record else args.record_dir,
        checkpoint_path=args.checkpoint,
        check_reproducible=args.reproducible,
    )

    columns = ["policy", "total_reward", "total_ticks", "average_reward"]
    rows = [
        {
            "policy": name,
            "total_reward": s.total_reward,
            "total_ticks": s.total_ticks,
            "average_reward": s.average_reward,
        }
        for name, s in result.summaries.items()
    ]
    print(comparison_table(rows, columns=columns))
    print()
    print(f"metric: {result.metric} (identical eval seeds)")
    print(f"gate 1  actor/critic > random     : {result.gate1_beats_random}")
    print(f"gate 2  actor/critic > linear Q    : {result.gate2_beats_linear_q}")
    print(f"gate 3  reproducible improvement   : {result.gate3_reproducible}")

    from cognitive_runtime.training.statistical_evaluation import format_comparison_report

    print("\nstatistical comparison (issue #44, mean +/- CI over the eval episodes):")
    print("  actor-critic vs random:")
    print("  " + format_comparison_report(result.gate1_comparisons).replace("\n", "\n  "))
    print("  actor-critic vs linear-Q:")
    print("  " + format_comparison_report(result.gate2_comparisons).replace("\n", "\n  "))
    if not args.no_record:
        print(f"\nrecorded eval sessions under {args.record_dir!r}; inspect with:")
        print(f"    python -m cognitive_runtime dashboard --record-dir {args.record_dir}")
    if args.checkpoint:
        print(f"\ngate results written to checkpoint training stats: {args.checkpoint}")


def cmd_curriculum_run(args: argparse.Namespace) -> None:
    """The curriculum runner (issue #43): train/evaluate/promote (or hold) an
    actor/critic checkpoint through an ordered list of staged world/reward
    configs, unattended. See docs/curriculum.md for the definition schema.

    ``--ladder`` (issue #139) runs the built-in Gestation->Foraging ladder
    (``development.ladder.GESTATION_TO_FORAGING``) instead of a
    ``--curriculum-file``, with its real Phase 2-6 milestone gates
    auto-attached via ``development.ladder.ladder_milestone_metrics`` --
    before this, the only way to invoke that ladder with working gates was a
    custom Python call that partially applied the provider itself; nothing
    on the CLI surface exposed the ladder or wired its milestone metrics up.

    A run that reaches Objects or Foraging (``motor_freedom="learned"``)
    stops there with a clear error: those stages need a real Phase 6
    voluntary controller (MPC-over-cortex or an alternative), which needs a
    trained predictive cortex and has no ``--ladder`` flag to supply one yet
    (PR #161 review; tracked separately). Milestone 7's own acceptance bar
    -- Gestation, Babbling, Crawling -- doesn't need one.
    """
    try:
        from development.definitions import CurriculumDefinitionError
        from development.runner import run_curriculum
    except ImportError as exc:  # torch not installed
        sys.exit(f"the curriculum runner needs PyTorch ({exc}); install '.[neural]'.")

    milestone_metrics = None
    world_model_checkpoint_paths: Sequence[str] = ()
    if args.ladder:
        import functools

        from development.ladder import (
            GESTATION_TO_FORAGING,
            ladder_cortex_checkpoint_paths,
            ladder_milestone_metrics,
        )

        if args.no_record:
            sys.exit("--ladder needs session recording for its milestone metrics; drop --no-record.")
        definition = GESTATION_TO_FORAGING
        milestone_metrics = functools.partial(
            ladder_milestone_metrics, record_dir=args.record_dir, cortex_checkpoint_base=args.checkpoint,
        )
        world_model_checkpoint_paths = tuple(ladder_cortex_checkpoint_paths(args.checkpoint).values())
    else:
        from development.definitions import load_curriculum_definition

        try:
            definition = load_curriculum_definition(args.curriculum_file)
        except CurriculumDefinitionError as exc:
            sys.exit(str(exc))

    try:
        result = run_curriculum(
            definition,
            checkpoint_path=args.checkpoint,
            model_seed=args.model_seed,
            train_seed=args.train_seed,
            eval_seed=args.eval_seed,
            start_stage=args.stage,
            force_promote=args.force_promote,
            fresh=args.fresh,
            record_dir=None if args.no_record else args.record_dir,
            name=args.name,
            milestone_metrics=milestone_metrics,
            world_model_checkpoint_paths=world_model_checkpoint_paths,
        )
    except (CurriculumDefinitionError, ValueError) as exc:
        message = str(exc)
        if args.ladder and "voluntary_controller factory" in message:
            # PR #161 review: the ladder's Objects/Foraging stages declare
            # motor_freedom="learned" and need a real Phase 6 voluntary
            # controller (MPC-over-cortex or an alternative), which needs a
            # trained predictive cortex development.runner does not build on
            # its own -- and which --ladder has no flag to supply yet. The
            # underlying error's "pass one via run_curriculum(...)" is
            # Python-call advice that doesn't map to any flag on this CLI
            # surface, so a --ladder user would otherwise be told to do
            # something they have no way to do here.
            message += (
                "\n\n--ladder has no flag yet to supply a voluntary controller for "
                "Objects/Foraging; a run that reaches either of them from the CLI "
                "will stop here until that wiring is added (tracked separately -- "
                "see issue #103's real Phase 6 controllers). Gestation, Babbling, "
                "and Crawling (Milestone 7's own acceptance bar) don't need one."
            )
        sys.exit(message)

    print(f"curriculum: {definition.name}  ({'resumed' if result.resumed else 'fresh start'})")
    print(f"status: {result.status}")
    if result.completed:
        print(f"all {len(definition.stages)} stage(s) promoted through.")
    else:
        print(f"held at stage {result.state.stage_index} ({definition.stages[result.state.stage_index].name!r}):")
        print(f"  {result.state.hold_reason}")
    print("\nattempt history:")
    for entry in result.state.history:
        print(
            f"  stage={entry['stage']!r} attempt={entry['attempt']} "
            f"{entry['metric']}={entry['value']!r} threshold={entry['threshold']!r} "
            f"promoted={entry['promoted']}{' (forced)' if entry['forced'] else ''}"
        )
    print(f"\ncurriculum state written to checkpoint training stats: {args.checkpoint}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    print(dashboard(args.record_dir, statistical=args.statistical, name=args.name))


def cmd_review(args: argparse.Namespace) -> None:
    """Post-run review (issue #33): summarize a session, compare it against
    baseline sessions on the same curriculum, and show per-episode detail --
    the one command to run after a childhood run before deciding whether to
    advance to the next curriculum step."""
    print(review_run(
        args.session, record_dir=args.record_dir, episode=args.episode, tail=args.tail
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cognitive_runtime", description="Continuous Cognitive Runtime (Minecraft MVP)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the runtime with a policy")
    p_run.add_argument("--name", default=None,
                       help="issue #88: organism name, threaded into the session id, "
                            "recorded metadata, checkpoints and exports; default: a "
                            "generated Docker-style name (e.g. vigorous-shannon)")
    p_run.add_argument("--policy", default="scripted",
                       choices=["null", "random", "scripted", "learned", "neural", "online",
                                "actor-critic", "human"])
    p_run.add_argument("--input-profile", default="full", choices=sorted(INPUT_PROFILES),
                       help="issue #32: 'full' (default) fuses every stream the legacy "
                            "encoder registry binds, including hand-computed semantic "
                            "streams (world.front_block, world.sheltered, vision.entities, "
                            "event.* marks); 'raw' restricts the fused policy state to "
                            "streams the stream registry classifies agent_input -- "
                            "semantic streams still publish/record for debugging and aux "
                            "losses, they just stop reaching the policy")
    p_run.add_argument("--episodes", type=int, default=1)
    p_run.add_argument("--tick-rate", type=float, default=20.0)
    p_run.add_argument("--realtime", action="store_true",
                       help="hold the tick rate in wall-clock time (default: fast-forward)")
    p_run.add_argument("--no-record", action="store_true")
    p_run.add_argument("--record-frames", action="store_true")
    p_run.add_argument("--record-streams", nargs="+", default=["*"],
                       help="stream globs to log with full payload (default: all)")
    p_run.add_argument("--exclude-streams", nargs="+", default=[],
                       help="stream globs to log hash-only, e.g. vision.*")
    p_run.add_argument("--frame-disk-budget-mb", type=float, default=512.0,
                       help="rolling binary frame store budget; oldest unpinned "
                            "segments are dropped once exceeded")
    p_run.add_argument("--pin-on-streams", nargs="+",
                       default=["event.died", "event.damage_taken"],
                       help="stream globs that pin the frame store's current "
                            "segment when they fire, e.g. event.died")
    p_run.add_argument("--record-dir", default="sessions")
    p_run.add_argument("--session-id", default=None)
    p_run.add_argument("--online-model", default=DEFAULT_ONLINE_MODEL_OUT,
                       help="online Q checkpoint path")
    p_run.add_argument("--online-save-every", type=int, default=1000,
                       help="save online Q checkpoint every N TD updates")
    p_run.add_argument("--epsilon-start", type=float, default=0.2)
    p_run.add_argument("--epsilon-min", type=float, default=0.05)
    p_run.add_argument("--epsilon-decay-ticks", type=int, default=50000)
    p_run.add_argument("--online-lr", type=float, default=0.02)
    p_run.add_argument("--online-gamma", type=float, default=0.99)
    p_run.add_argument("--online-train", dest="online_train", action="store_true",
                       default=True, help="train the online Q model while running")
    p_run.add_argument("--no-online-train", dest="online_train", action="store_false",
                       help="run online Q in eval mode without mutating the model")
    p_run.add_argument("--fresh", action="store_true",
                       help="initialize online/actor-critic weights fresh even though no "
                            "checkpoint exists yet at the model path; required for "
                            "--backend remote with no existing checkpoint (issue #33)")
    p_run.add_argument("--actor-critic-model", default=DEFAULT_ACTOR_CRITIC_MODEL_OUT,
                       help="actor-critic checkpoint bundle path (.pt)")
    p_run.add_argument("--fusion", choices=sorted(FUSION_MODES), default="fixed",
                       help="actor-critic's fused-state source (issue #57): 'fixed' (default) "
                            "is TemporalFusion's hand-written concatenation; 'learned' runs "
                            "trainable stream encoders + LatentFusionModel in the tick's "
                            "inference path instead. A checkpoint trained under one mode "
                            "fails loudly if resumed under the other.")
    p_run.add_argument("--attention", choices=sorted(ATTENTION_MODES), default="off",
                       help="deterministic attention controller (issue #59): 'off' (default) "
                            "gives every agent-input stream uniform weight 1.0, reproducing "
                            "the pre-#59 fused output exactly; 'budgeted' scores every "
                            "agent-input stream's salience each tick and gates the fused "
                            "state under a hard budget, recording an AttentionState (weights, "
                            "focus stream, reason breakdown) every tick.")
    p_run.add_argument("--reflex", choices=sorted(REFLEX_MODES), default="on",
                       help="scripted orienting reflex (issue #60): 'on' (default) turns "
                            "toward a bottom-up attention capture with a localizable "
                            "direction hint, bounded and vetoed by high internal.risk or a "
                            "survival-critical policy action; 'off' disables it (the "
                            "ablation); 'learned-only' leaves orienting to the policy "
                            "instead. Only fires when --attention=budgeted.")
    p_run.add_argument("--actor-critic-save-every", type=int, default=1000,
                       help="save the actor-critic checkpoint every N gradient steps")
    p_run.add_argument("--actor-critic-lr", type=float, default=1e-3)
    p_run.add_argument("--actor-critic-gamma", type=float, default=0.99)
    p_run.add_argument("--actor-critic-entropy-coef", type=float, default=0.01,
                       help="entropy-bonus weight encouraging exploration")
    p_run.add_argument("--actor-critic-grad-clip-norm", type=float, default=5.0)
    p_run.add_argument("--actor-critic-hidden-dim", type=int, default=128)
    p_run.add_argument("--actor-critic-history", type=int, default=8,
                       help="recent-action window fed into world_features")
    p_run.add_argument("--actor-critic-replay-every", type=int, default=32,
                       help="pull a replay minibatch every N ticks")
    p_run.add_argument("--actor-critic-replay-batch-size", type=int, default=32)
    p_run.add_argument("--actor-critic-world-model-loss", dest="actor_critic_world_model_loss",
                       action="store_true", default=True,
                       help="jointly train an action-conditioned world model from the same "
                            "transitions (default: on)")
    p_run.add_argument("--no-actor-critic-world-model-loss", dest="actor_critic_world_model_loss",
                       action="store_false")
    p_run.add_argument("--actor-critic-train", dest="actor_critic_train", action="store_true",
                       default=True, help="train the actor-critic model while running")
    p_run.add_argument("--no-actor-critic-train", dest="actor_critic_train", action="store_false",
                       help="run actor-critic in eval mode without mutating weights")
    p_run.add_argument("--async-trainer", dest="actor_critic_async", action="store_true",
                       default=False,
                       help="actor/learner split: train in the background instead of "
                            "synchronously in the tick loop; see --async-schedule for "
                            "phasic (default) vs concurrent")
    p_run.add_argument("--async-schedule", choices=("phasic", "concurrent"), default="phasic",
                       help="'phasic' (default): pause acting for bounded sleep "
                            "consolidation, no staleness. 'concurrent' (issue #100): the "
                            "trainer runs continuously in its own process while the actor "
                            "keeps acting, polling for EMA-averaged weight snapshots every "
                            "--async-reload-every-ticks ticks instead of pausing")
    p_run.add_argument("--async-ring-capacity", type=int, default=20_000,
                       help="live-experience ring buffer capacity (transitions); "
                            "drop-oldest once full")
    p_run.add_argument("--async-batch-size", type=int, default=32,
                       help="trainer process minibatch size")
    p_run.add_argument("--async-min-buffer-size", type=int, default=256,
                       help="trainer process waits for this many transitions before its "
                            "first gradient step")
    p_run.add_argument("--async-publish-every", type=int, default=50,
                       help="trainer process publishes a new weight snapshot every N "
                            "gradient steps")
    p_run.add_argument("--async-reload-every-ticks", type=int, default=5,
                       help="ignored by --async-schedule phasic, which reloads exactly "
                            "once after each completed consolidation; for "
                            "--async-schedule concurrent, the actor polls for a newer "
                            "snapshot every N ticks")
    p_run.add_argument("--async-ema-decay", type=float, default=0.999,
                       help="--async-schedule concurrent only: Polyak/EMA decay for "
                            "published weight snapshots (closer to 1 = slower-moving "
                            "target, less tick-to-tick oscillation)")
    p_run.add_argument("--async-wake-ticks", type=int, default=50,
                       help="number of acting ticks between sleep consolidation passes")
    p_run.add_argument("--async-consolidation-steps", type=int, default=50,
                       help="maximum gradient steps in each sleep consolidation pass")
    _add_world_args(p_run)
    _add_world_selector_arg(p_run)
    _add_world_model_arg(p_run)
    _add_entity_persistence_arg(p_run)
    p_run.set_defaults(func=cmd_run)

    p_demo = sub.add_parser("demo", help="play SurvivalBox yourself; recorded as demonstrations")
    p_demo.add_argument("--episodes", type=int, default=1)
    p_demo.add_argument("--tick-rate", type=float, default=20.0)
    p_demo.add_argument("--record-dir", default="sessions")
    p_demo.add_argument("--session-id", default=None)
    p_demo.add_argument("--name", default=None,
                        help="issue #88: organism name (see 'run --name'); default: generated")
    _add_world_args(p_demo)
    _add_world_selector_arg(p_demo)
    _add_world_model_arg(p_demo)
    _add_entity_persistence_arg(p_demo)
    p_demo.set_defaults(func=cmd_demo)

    p_eval = sub.add_parser("evaluate", help="compare policies on identical episodes")
    p_eval.add_argument("--policies", default="null,random,scripted")
    p_eval.add_argument("--episodes", type=int, default=3)
    _add_world_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_stat_eval = sub.add_parser(
        "statistical-evaluate",
        help="statistical evaluation harness (issue #44): mean +/- CI across N "
             "episodes per policy/checkpoint, with regression flagging against "
             "a --baseline",
    )
    p_stat_eval.add_argument("--policies", default="null,random,scripted",
                             help="comma-separated policy names to run fresh in sim "
                                  "(ignored with --from-sessions)")
    p_stat_eval.add_argument("--episodes", type=int, default=10,
                             help="episodes per policy (larger N narrows the CI)")
    p_stat_eval.add_argument("--confidence", type=float, default=0.95,
                             help="confidence level for the reported interval")
    p_stat_eval.add_argument("--baseline", default=None,
                             help="policy/group name to compare every other group "
                                  "against, flagging statistically significant regressions")
    p_stat_eval.add_argument("--record-dir", default=None,
                             help="record each policy's eval episodes here (omit to skip)")
    p_stat_eval.add_argument("--from-sessions", default=None,
                             help="skip running fresh episodes; load recorded "
                                  "EpisodeSummary data from this record_dir instead, "
                                  "grouped by (curriculum, policy)")
    _add_world_args(p_stat_eval)
    p_stat_eval.set_defaults(func=cmd_statistical_evaluate)

    p_gates = sub.add_parser(
        "evaluation-gates",
        help="evaluation gates: actor/critic vs random/scripted/linear-Q "
             "on identical seeds (issue #31)",
    )
    p_gates.add_argument("--curriculum", default=None, choices=CURRICULUM_ORDER,
                         help="curriculum preset supplying world + reward config "
                              "(default: the fixed DEFAULT_GATE_CONFIG)")
    p_gates.add_argument("--train-episodes", type=int, default=20,
                         help="training episodes per learner before eval")
    p_gates.add_argument("--eval-episodes", type=int, default=2,
                         help="no-mutation eval episodes per policy on identical seeds")
    p_gates.add_argument("--reproducible", action="store_true",
                         help="rerun train+eval with the same seeds and report gate 3 "
                              "(reproducible improvement)")
    p_gates.add_argument("--record-dir", default="sessions",
                         help="record eval sessions here for dashboard inspection")
    p_gates.add_argument("--no-record", action="store_true",
                         help="skip recording eval sessions")
    p_gates.add_argument("--checkpoint", default=None,
                         help="write the trained actor/critic bundle here with the gate "
                              "results in its training stats (issue #20)")
    p_gates.set_defaults(func=cmd_evaluation_gates)

    p_curriculum_run = sub.add_parser(
        "curriculum-run",
        help="run/resume a staged curriculum with metric-gated promotion (issue #43)",
    )
    p_curriculum_run.add_argument(
        "--curriculum-file", default="goals/curricula/toy_two_stage.yaml",
        help="curriculum definition YAML/JSON: ordered stages, each a world/reward "
             "config plus promotion criteria (docs/curriculum.md); ignored with "
             "--ladder",
    )
    p_curriculum_run.add_argument(
        "--ladder", action="store_true",
        help="run the built-in Gestation->Foraging ladder (issue #105) instead of "
             "--curriculum-file, with its real Phase 2-6 milestone gates "
             "auto-attached (issue #139); needs session recording (no --no-record) "
             "since the gates train/evaluate against the recorded sessions. Stops "
             "with a clear error on reaching Objects/Foraging: those "
             "motor_freedom='learned' stages need a real Phase 6 voluntary "
             "controller this flag can't supply yet -- Milestone 7's own bar "
             "(Gestation/Babbling/Crawling) doesn't need one",
    )
    p_curriculum_run.add_argument(
        "--checkpoint", required=True,
        help="actor/critic checkpoint bundle carried across stage boundaries; curriculum "
             "progress (stage, attempts, promotion history) lives in its training stats",
    )
    p_curriculum_run.add_argument(
        "--stage", type=int, default=None,
        help="override the stage to (re)start from, by index (default: resume from the "
             "checkpoint's saved progress, or stage 0 with no checkpoint/--fresh)",
    )
    p_curriculum_run.add_argument(
        "--force-promote", action="store_true",
        help="promote past the very next evaluation regardless of its metric value "
             "(manual override for experimentation)",
    )
    p_curriculum_run.add_argument(
        "--fresh", action="store_true",
        help="ignore any existing checkpoint and start stage 0 with fresh weights",
    )
    p_curriculum_run.add_argument("--name", default=None,
                                  help="issue #88: organism name threaded into every stage's "
                                       "recorded session metadata and the actor/critic "
                                       "checkpoint; default: generated per stage run")
    p_curriculum_run.add_argument("--model-seed", type=int, default=1)
    p_curriculum_run.add_argument("--train-seed", type=int, default=100)
    p_curriculum_run.add_argument("--eval-seed", type=int, default=500)
    p_curriculum_run.add_argument("--record-dir", default="sessions",
                                   help="record train/eval sessions here")
    p_curriculum_run.add_argument("--no-record", action="store_true",
                                   help="skip recording sessions")
    p_curriculum_run.set_defaults(func=cmd_curriculum_run)

    p_train = sub.add_parser("train", help="train a behavioral-cloning policy from sessions")
    p_train.add_argument("--name", default=None,
                         help="issue #88: organism name stamped into the trained checkpoint's "
                              "metadata (model-type checkpoints only, not the plain linear BC "
                              "model); default: unstamped")
    p_train.add_argument("--sessions", nargs="+", required=True,
                         help="session directories (e.g. sessions/20260101-...-scripted)")
    p_train.add_argument("--out", default=DEFAULT_MODEL_OUT,
                         help="output path; neural models default to models/vision_bc.pt")
    p_train.add_argument("--model-type",
                         choices=["linear", "neural", "pixel-encoder", "fusion", "world-model",
                                  "multi-horizon-world-model", "entity-persistence"],
                         default="linear",
                         help="linear softmax head (default), pixel BC, pixel encoder pretrain, "
                              "learned latent fusion, the action-conditioned world model, the "
                              "multi-horizon uncertainty-aware world model (issue #39), or "
                              "the entity-persistence (object permanence) model")
    p_train.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 100],
                         help="--model-type multi-horizon-world-model only: tick offsets to "
                              "predict at (action ticks, per build_multi_horizon_world_model_"
                              "dataset; must include 1)")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=0.5, help="linear-model learning rate")
    p_train.add_argument("--neural-lr", type=float, default=1e-3, help="neural-model learning rate")
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--history", type=int, default=8)
    p_train.add_argument("--stream-profile", default="full", choices=["full", "raw"],
                         help="--model-type neural only (issue #32): the non-vision companion "
                              "vector's ablation. 'full' (default) is pixels + semantics "
                              "(every non-vision stream the registry fuses); 'raw' is pixel "
                              "only (restricts the non-vision vector to agent_input-classified "
                              "body/reward/spatial proprioception, dropping hand-computed "
                              "semantic scalars)")
    p_train.add_argument("--encoder-init", default=None,
                         help="pixel-encoder checkpoint bundle used to initialize neural BC")
    p_train.add_argument("--latent-width", type=int, default=64,
                         help="pixel-encoder latent width for pretraining")
    p_train.add_argument("--hidden-dim", type=int, default=128,
                         help="hidden width for neural training heads")
    p_train.add_argument("--fusion-depth", type=int, default=2,
                         help="number of hidden layers for learned fusion")
    p_train.add_argument("--fusion-dropout", type=float, default=0.0,
                         help="dropout for learned fusion hidden layers")
    p_train.add_argument("--reconstruction-size", type=int, default=16,
                         help="max side length for downsampled reconstruction targets")
    p_train.add_argument("--reconstruction-weight", type=float, default=1.0)
    p_train.add_argument("--next-latent-weight", type=float, default=1.0)
    p_train.add_argument("--contrastive-weight", type=float, default=1.0)
    p_train.add_argument("--contrastive-temperature", type=float, default=0.2)
    p_train.add_argument("--features", choices=["latent", "handcrafted"], default="latent",
                         help="linear policy input: fused latent state (default) or hand featurizer")
    p_train.add_argument("--max-samples", type=int, default=None)
    p_train.add_argument("--min-reward", type=float, default=None,
                         help="skip episodes below this total reward")
    p_train.add_argument("--seed", type=int, default=0)
    p_train.set_defaults(func=cmd_train)

    p_trainer = sub.add_parser(
        "trainer",
        help="standalone actor/critic AsyncTrainer (issue #37): pointed only at recorded "
             "sessions with no live actor, this is offline pretraining -- the same trainer "
             "`run --async-trainer` spawns as a background process, run here to completion "
             "in the foreground",
    )
    p_trainer.add_argument("--sessions", nargs="+", required=True,
                           help="session directories to pretrain from (streams-v2)")
    p_trainer.add_argument("--out", default=DEFAULT_ACTOR_CRITIC_MODEL_OUT,
                           help="checkpoint bundle path (.pt); resumed from if it already "
                                "exists")
    p_trainer.add_argument("--steps", type=int, default=2000,
                           help="gradient steps to run (an offline trainer has no live "
                                "stream to keep waiting on, so it stops here)")
    p_trainer.add_argument("--max-transitions", type=int, default=None,
                           help="cap on transitions loaded from the sessions")
    p_trainer.add_argument("--min-episode-reward", type=float, default=None,
                           help="skip episodes below this total reward")
    p_trainer.add_argument("--batch-size", type=int, default=32)
    p_trainer.add_argument("--publish-every", type=int, default=100,
                           help="write a checkpoint every N gradient steps, plus once more "
                                "at the end")
    p_trainer.add_argument("--hidden-dim", type=int, default=128)
    p_trainer.add_argument("--lr", type=float, default=1e-3)
    p_trainer.add_argument("--gamma", type=float, default=0.99)
    p_trainer.add_argument("--entropy-coef", type=float, default=0.01)
    p_trainer.add_argument("--grad-clip-norm", type=float, default=5.0)
    p_trainer.add_argument("--world-model-loss", dest="world_model_loss",
                           action="store_true", default=True)
    p_trainer.add_argument("--no-world-model-loss", dest="world_model_loss",
                           action="store_false")
    p_trainer.add_argument("--seed", type=int, default=0)
    p_trainer.set_defaults(func=cmd_trainer)

    p_canary = sub.add_parser(
        "ego-motion-canary",
        help="issue #39: walk_forward next-frame prediction benchmark on held-out seeds, "
             "vs. copy-last-frame and mean-frame baselines (PSNR/SSIM)",
    )
    p_canary.add_argument("--record-dir", default="sessions",
                          help="directory to record the walk_forward train/holdout episodes into")
    p_canary.add_argument("--train-seeds", type=int, default=6,
                          help="number of train-seed episodes (seeds 0..N-1)")
    p_canary.add_argument("--holdout-seeds", type=int, default=2,
                          help="number of held-out-seed episodes (seeds N..N+M-1, never trained on)")
    p_canary.add_argument("--episode-ticks", type=int, default=120)
    p_canary.add_argument("--world-size", type=int, default=48)
    p_canary.add_argument("--action-noise", type=float, default=0.0,
                          help="probability each tick's action is a random action instead of "
                               "MOVE_FORWARD")
    p_canary.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 100],
                          help="tick offsets to evaluate next-frame prediction at")
    p_canary.add_argument("--latent-width", type=int, default=32)
    p_canary.add_argument("--hidden-dim", type=int, default=64)
    p_canary.add_argument("--reconstruction-size", type=int, default=16,
                          help="max side length for downsampled reconstruction targets")
    p_canary.add_argument("--epochs", type=int, default=15,
                          help="pixel encoder/decoder pretraining epochs")
    p_canary.add_argument("--consistency-epochs", type=int, default=15,
                          help="horizon-consistency fine-tuning epochs (0 skips it)")
    p_canary.add_argument("--neural-lr", type=float, default=1e-3)
    p_canary.add_argument("--batch-size", type=int, default=32)
    p_canary.add_argument("--seed", type=int, default=0)
    p_canary.add_argument("--out", default=None,
                          help="checkpoint bundle path (.pt); omit to skip saving")
    p_canary.set_defaults(func=cmd_ego_motion_canary)

    p_nursery = sub.add_parser(
        "nursery",
        help="issue #62: nursery scenario suite -- scripted micro-scenarios "
             "benchmarking multi-horizon (t+1/t+10/t+100) world-model prediction "
             "against copy-last-frame/mean-frame baselines",
    )
    nursery_sub = p_nursery.add_subparsers(dest="nursery_command", required=True)

    p_nursery_list = nursery_sub.add_parser("list", help="list available nursery scenarios")
    p_nursery_list.add_argument("--world", default="minecraft", choices=sorted(WORLDS),
                                help="issue #90: list Minecraft's scenarios (default) or "
                                     "Crafter's ports (--world crafter)")
    p_nursery_list.set_defaults(func=cmd_nursery_list)

    p_nursery_run = nursery_sub.add_parser(
        "run", help="record + benchmark one scenario (or 'all' for the whole suite)"
    )
    p_nursery_run.add_argument(
        "scenario", help="scenario name (see 'nursery list'), or 'all' to run the full suite"
    )
    p_nursery_run.add_argument("--world", default="minecraft", choices=sorted(WORLDS),
                               help="issue #90: record against Minecraft's simulated/remote "
                                    "backend (default) or the Crafter nursery world "
                                    "(--world crafter; needs the 'crafter' extra installed). "
                                    "--backend/--world-size only apply to --world minecraft.")
    p_nursery_run.add_argument("--name", default=None,
                               help="issue #88: organism name threaded into every recorded "
                                    "episode's session metadata, prediction exports, and the "
                                    "trained encoder checkpoint; default: generated per episode")
    p_nursery_run.add_argument("--record-dir", default="sessions",
                               help="directory to record each scenario's train/holdout episodes into")
    p_nursery_run.add_argument("--train-seeds", type=int, default=6,
                               help="number of train-seed episodes (seeds 0..N-1)")
    p_nursery_run.add_argument("--holdout-seeds", type=int, default=2,
                               help="number of held-out-seed episodes (seeds N..N+M-1, never trained on)")
    p_nursery_run.add_argument("--episode-ticks", type=int, default=400)
    p_nursery_run.add_argument("--world-size", type=int, default=48)
    p_nursery_run.add_argument("--backend", default=_default_nursery_backend(),
                               choices=sorted(BACKENDS),
                               help="backend used to record nursery episodes. Defaults to "
                                    "remote when CCR_MINECRAFT_HOST is set, otherwise "
                                    "simulated; CCR_NURSERY_BACKEND can override this.")
    p_nursery_run.add_argument("--realtime", action="store_true",
                               help="hold wall-clock tick pacing while recording nursery "
                                    "episodes. Remote nursery recordings force realtime.")
    p_nursery_run.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 100],
                               help="tick offsets to evaluate next-frame prediction at")
    p_nursery_run.add_argument("--latent-width", type=int, default=32)
    p_nursery_run.add_argument("--hidden-dim", type=int, default=64)
    p_nursery_run.add_argument("--reconstruction-size", type=int, default=16,
                               help="max side length for downsampled reconstruction targets")
    p_nursery_run.add_argument("--epochs", type=int, default=15,
                               help="pixel encoder/decoder pretraining epochs")
    p_nursery_run.add_argument("--consistency-epochs", type=int, default=15,
                               help="horizon-consistency fine-tuning epochs (0 skips it)")
    p_nursery_run.add_argument("--entity-persistence-epochs", type=int, default=30,
                               help="object_permanence only: entity-persistence model training epochs")
    p_nursery_run.add_argument("--neural-lr", type=float, default=1e-3)
    p_nursery_run.add_argument("--batch-size", type=int, default=32)
    p_nursery_run.add_argument("--seed", type=int, default=0)
    p_nursery_run.add_argument("--out-dir", default=None,
                               help="directory to save one checkpoint bundle per scenario "
                                    "(<out-dir>/<scenario>.pt) plus a full-model bundle "
                                    "(<out-dir>/<scenario>-full.pt) the prediction exporter "
                                    "can reload; omit to skip saving")
    p_nursery_run.add_argument("--report", default=None,
                               help="path to save a JSON report (per-scenario per-horizon "
                                    "metrics + dream strips); omit to skip saving")
    p_nursery_run.add_argument("--no-export-predictions", action="store_true",
                               help="skip writing predictions_<episode>.json (the pixel "
                                    "viewer's 'model' source) next to each recorded episode")
    p_nursery_run.add_argument("--skip-data-quality-gate", action="store_true",
                               help="train even when recordings fail the scenario's "
                                    "data-quality expectations (stuck agent, static view)")
    p_nursery_run.set_defaults(func=cmd_nursery_run)

    p_nursery_joint = nursery_sub.add_parser(
        "joint",
        help="record every scenario and train ONE action-conditioned recurrent "
             "world model across them, with zero-shot held-out-scenario "
             "evaluation, a frozen-rollout detector, and a yaw linear probe",
    )
    p_nursery_joint.add_argument("--record-dir", default="sessions")
    p_nursery_joint.add_argument("--train-scenarios", nargs="+", default=None,
                                 help="scenarios to train on (default: every scenario not held out)")
    p_nursery_joint.add_argument("--holdout-scenarios", nargs="+", default=None,
                                 help="scenarios excluded from training and evaluated zero-shot "
                                      "(default: approach_entity)")
    p_nursery_joint.add_argument("--train-seeds", type=int, default=6)
    p_nursery_joint.add_argument("--holdout-seeds", type=int, default=2)
    p_nursery_joint.add_argument("--episode-ticks", type=int, default=400)
    p_nursery_joint.add_argument("--world-size", type=int, default=48)
    p_nursery_joint.add_argument("--backend", default=_default_nursery_backend(),
                                 choices=sorted(BACKENDS))
    p_nursery_joint.add_argument("--realtime", action="store_true")
    p_nursery_joint.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 100],
                                 help="tick offsets to evaluate at (converted to recorded-frame "
                                      "steps via the measured vision rate)")
    p_nursery_joint.add_argument("--latent-width", type=int, default=32)
    p_nursery_joint.add_argument("--hidden-dim", type=int, default=64)
    p_nursery_joint.add_argument("--reconstruction-size", type=int, default=16)
    p_nursery_joint.add_argument("--epochs", type=int, default=30)
    p_nursery_joint.add_argument("--warmup-frames", type=int, default=3,
                                 help="teacher-forced frames before each training rollout")
    p_nursery_joint.add_argument("--rollout-frames", type=int, default=8,
                                 help="closed-loop steps per training window (short on purpose)")
    p_nursery_joint.add_argument("--backbone", default="gru",
                                 choices=["gru", "dilated_conv", "transformer"],
                                 help="cortex temporal backbone (issue #93): the default recurrent "
                                      "GRU, a WaveNet-style dilated causal conv, or a small causal "
                                      "transformer, both windowed over --context-length")
    p_nursery_joint.add_argument("--context-length", type=int, default=8,
                                 help="window size the dilated_conv/transformer backbones attend "
                                      "over (ignored by gru); ramped 1 -> this value over training "
                                      "via the context-length curriculum")
    p_nursery_joint.add_argument("--neural-lr", type=float, default=1e-3)
    p_nursery_joint.add_argument("--batch-size", type=int, default=32)
    p_nursery_joint.add_argument("--seed", type=int, default=0)
    p_nursery_joint.add_argument("--out-dir", default=None,
                                 help="directory to save the joint model bundle "
                                      "(<out-dir>/joint-world-model.pt)")
    p_nursery_joint.add_argument("--report", default=None,
                                 help="path to save a JSON report of all metrics")
    p_nursery_joint.add_argument("--skip-data-quality-gate", action="store_true")
    p_nursery_joint.set_defaults(func=cmd_nursery_joint)

    p_nursery_bench = nursery_sub.add_parser(
        "backbone-benchmark",
        help="issue #93: train the cortex once per temporal backbone (gru, dilated_conv, "
             "transformer) on identical recordings and report the Phase 2 scoring gates "
             "(model/copy-last, model/oracle, frozen-rollout) per horizon for each",
    )
    p_nursery_bench.add_argument("--record-dir", default="sessions")
    p_nursery_bench.add_argument("--train-scenarios", nargs="+", default=["walk_forward", "turn_in_place"],
                                 help="scenarios recorded and trained on (shared by every backbone)")
    p_nursery_bench.add_argument("--eval-scenario", default="turn_in_place",
                                 help="held-out-seed scenario each backbone is scored on; must be "
                                      "one of --train-scenarios")
    p_nursery_bench.add_argument("--backbones", nargs="+", default=["gru", "dilated_conv", "transformer"],
                                 choices=["gru", "dilated_conv", "transformer"],
                                 help="backbones to benchmark")
    p_nursery_bench.add_argument("--baseline-backbone", default="gru",
                                 choices=["gru", "dilated_conv", "transformer"],
                                 help="backbone every other backbone's comparison is measured against")
    p_nursery_bench.add_argument("--train-seeds", type=int, default=6)
    p_nursery_bench.add_argument("--holdout-seeds", type=int, default=2)
    p_nursery_bench.add_argument("--episode-ticks", type=int, default=400)
    p_nursery_bench.add_argument("--world-size", type=int, default=48)
    p_nursery_bench.add_argument("--backend", default=_default_nursery_backend(),
                                 choices=sorted(BACKENDS))
    p_nursery_bench.add_argument("--realtime", action="store_true")
    p_nursery_bench.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 100],
                                 help="tick offsets to evaluate at (converted to recorded-frame "
                                      "steps via the measured vision rate)")
    p_nursery_bench.add_argument("--latent-width", type=int, default=32)
    p_nursery_bench.add_argument("--hidden-dim", type=int, default=64)
    p_nursery_bench.add_argument("--reconstruction-size", type=int, default=16)
    p_nursery_bench.add_argument("--epochs", type=int, default=30)
    p_nursery_bench.add_argument("--warmup-frames", type=int, default=3,
                                 help="teacher-forced frames before each training rollout")
    p_nursery_bench.add_argument("--rollout-frames", type=int, default=8,
                                 help="closed-loop steps per training window (short on purpose)")
    p_nursery_bench.add_argument("--context-length", type=int, default=8,
                                 help="window size the dilated_conv/transformer backbones attend "
                                      "over; ramped 1 -> this value via the context-length curriculum")
    p_nursery_bench.add_argument("--neural-lr", type=float, default=1e-3)
    p_nursery_bench.add_argument("--batch-size", type=int, default=32)
    p_nursery_bench.add_argument("--seed", type=int, default=0)
    p_nursery_bench.add_argument("--report", default=None,
                                 help="path to save a JSON report of all metrics")
    p_nursery_bench.add_argument("--skip-data-quality-gate", action="store_true")
    p_nursery_bench.set_defaults(func=cmd_nursery_backbone_benchmark)

    p_replay = sub.add_parser("replay", help="re-simulate a session and verify determinism")
    p_replay.add_argument("--session", required=True)
    p_replay.add_argument("--episode", default=None)
    p_replay.add_argument("--no-verify", action="store_true")
    p_replay.add_argument("--reward-profile", default=None,
                          help="the reward profile the session was recorded with (required to "
                               "replay a session recorded with --reward-profile; must match by "
                               "content -- see docs/reward_profiles.md)")
    p_replay.set_defaults(func=cmd_replay)

    p_view = sub.add_parser("view", help="inspect a recorded episode")
    p_view.add_argument("--session", required=True)
    p_view.add_argument("--episode", required=True)
    p_view.add_argument("--tail", type=int, default=10)
    p_view.set_defaults(func=cmd_view)

    p_dash = sub.add_parser("dashboard", help="aggregate metrics across all sessions")
    p_dash.add_argument("--record-dir", default="sessions")
    p_dash.add_argument("--statistical", action="store_true",
                        help="append the statistical evaluation harness's mean +/- CI "
                             "report (issue #44) for the same (curriculum, policy) groups")
    p_dash.add_argument("--name", default=None,
                        help="issue #88: restrict to one organism name (sessions recorded "
                             "before this field existed group as 'legacy')")
    p_dash.set_defaults(func=cmd_dashboard)

    p_review = sub.add_parser(
        "review",
        help="post-run review: summarize a session, compare it against baseline "
             "sessions on the same curriculum, and show per-episode detail (issue #33)",
    )
    p_review.add_argument("--session", required=True,
                          help="the run's session directory, e.g. sessions/<id>")
    p_review.add_argument("--record-dir", default="sessions",
                          help="directory to search for baseline sessions on the same curriculum")
    p_review.add_argument("--episode", default=None,
                          help="specific episode id to show in detail (default: the last "
                               "--tail episodes)")
    p_review.add_argument("--tail", type=int, default=3,
                          help="number of most-recent episodes to show in detail")
    p_review.set_defaults(func=cmd_review)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
