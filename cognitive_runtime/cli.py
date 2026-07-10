"""Command-line interface for the Continuous Cognitive Runtime.

    python -m cognitive_runtime run --policy scripted --episodes 3
    python -m cognitive_runtime demo
    python -m cognitive_runtime evaluate --episodes 3
    python -m cognitive_runtime train --sessions sessions/<id> --out models/bc.json
    python -m cognitive_runtime replay --session sessions/<id> --verify
    python -m cognitive_runtime view --session sessions/<id> --episode episode_00000
    python -m cognitive_runtime dashboard
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import Any, Callable, Dict, Optional

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
from cognitive_runtime.programs.minecraft.rewards import SurvivalRewardConfig
from cognitive_runtime.programs.minecraft.stream_registry import MINECRAFT_STREAM_REGISTRY
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import NonDeterministicSessionError
from cognitive_runtime.tools.episode_viewer import view_episode
from cognitive_runtime.tools.metrics_dashboard import dashboard
from cognitive_runtime.tools.replay_runner import format_results, replay_session
from cognitive_runtime.training.datasets import build_dataset
from cognitive_runtime.training.evaluation import compare_policies
from cognitive_runtime.training.imitation import train_bc

DEFAULT_MODEL_OUT = "models/bc.json"
DEFAULT_ONLINE_MODEL_OUT = "models/online-q.json"


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


def _program_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "episode_ticks": args.episode_ticks,
        "difficulty": args.difficulty,
        "world_size": args.world_size,
        "day_length": args.day_length,
        "start_time": args.start_time,
        "max_mobs": args.max_mobs,
    }


def _make_policy(name: str, args: argparse.Namespace) -> Policy:
    if name == "null":
        return NullPolicy()
    if name == "random":
        return RandomPolicy(ACTION_SPACE, seed=args.seed)
    if name == "scripted":
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
    args: argparse.Namespace, program: MinecraftSurvivalBox
) -> tuple[OnlineQPolicy, OnlineQLearner]:
    action_space = list(program.metadata().action_space)
    action_keys = [action.key() for action in action_space]
    fusion = TemporalFusion(program.stream_catalog(), default_encoder_registry())
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
    parser.add_argument("--model", default=None, help="path to a trained BC model (learned policy)")
    parser.add_argument("--backend", default="simulated", choices=sorted(BACKENDS),
                        help="survival backend: the deterministic simulated world, or "
                             "a real-Minecraft client (remote; not yet implemented)")


def cmd_run(args: argparse.Namespace) -> None:
    _resolve_world_args(args)
    program_config = _program_config(args)
    program = MinecraftSurvivalBox(
        config=program_config, reward_config=_reward_config_for(args), backend=args.backend
    )
    learner = None
    if args.policy == "online":
        policy, learner = _make_online_policy_and_learner(args, program)
    else:
        policy = _make_policy(args.policy, args)
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
        program_config=program_config,
        curriculum=args.curriculum,
    )
    runtime = CognitiveRuntime(
        program=program,
        policy=policy,
        config=config,
        learner=learner,
        world_model=world_model,
        entity_persistence=entity_persistence,
        stream_registry=MINECRAFT_STREAM_REGISTRY,
    )
    summaries = runtime.run()
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
    reward_config = _reward_config_for(args)
    names = [p.strip() for p in args.policies.split(",") if p.strip()]
    factories: Dict[str, Callable[[], Policy]] = {}
    for name in names:
        factories[name] = (lambda n: (lambda: _make_policy(n, args)))(name)
    rows = compare_policies(
        program_factory=lambda: MinecraftSurvivalBox(
            config=program_config, reward_config=reward_config, backend=args.backend
        ),
        policy_factories=factories,
        episodes=args.episodes,
        seed=args.seed,
        max_ticks=args.episode_ticks,
    )
    print(comparison_table(rows))


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
    )
    if len(dataset) == 0:
        sys.exit("no pixel training samples found (record sessions with --record-frames)")
    print(f"dataset: {len(dataset)} pixel samples from {len(dataset.sources)} episodes "
          f"(frame={dataset.pixel_shape}, non-vision dim={len(dataset.non_vision_names)})")
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
    save_pixel_encoder_pretraining_checkpoint(out, model, dataset, stats)
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
    save_latent_fusion_checkpoint(out, model, dataset, stats)
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
    save_world_model_checkpoint(out, model, dataset, stats)
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
    save_entity_persistence_checkpoint(out, model, dataset, stats)
    for key in ("final_feature_loss", "final_surprise_loss", "final_total_loss",
                "baseline_mse", "model_mse", "beats_forget_baseline"):
        print(f"  {key}: {stats[key]}")
    print(f"checkpoint bundle saved to {out}")


def cmd_replay(args: argparse.Namespace) -> None:
    try:
        results = replay_session(args.session, episode_id=args.episode, verify=not args.no_verify)
    except NonDeterministicSessionError as exc:
        sys.exit(f"replay skipped: {exc}")
    print(format_results(results))
    if any(not r.matched for r in results):
        sys.exit(1)


def cmd_view(args: argparse.Namespace) -> None:
    print(view_episode(args.session, args.episode, tail=args.tail))


def cmd_dashboard(args: argparse.Namespace) -> None:
    print(dashboard(args.record_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cognitive_runtime", description="Continuous Cognitive Runtime (Minecraft MVP)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the runtime with a policy")
    p_run.add_argument("--policy", default="scripted",
                       choices=["null", "random", "scripted", "learned", "neural", "online",
                                "human"])
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
    _add_world_args(p_run)
    _add_world_model_arg(p_run)
    _add_entity_persistence_arg(p_run)
    p_run.set_defaults(func=cmd_run)

    p_demo = sub.add_parser("demo", help="play SurvivalBox yourself; recorded as demonstrations")
    p_demo.add_argument("--episodes", type=int, default=1)
    p_demo.add_argument("--tick-rate", type=float, default=20.0)
    p_demo.add_argument("--record-dir", default="sessions")
    p_demo.add_argument("--session-id", default=None)
    _add_world_args(p_demo)
    _add_world_model_arg(p_demo)
    _add_entity_persistence_arg(p_demo)
    p_demo.set_defaults(func=cmd_demo)

    p_eval = sub.add_parser("evaluate", help="compare policies on identical episodes")
    p_eval.add_argument("--policies", default="null,random,scripted")
    p_eval.add_argument("--episodes", type=int, default=3)
    _add_world_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_train = sub.add_parser("train", help="train a behavioral-cloning policy from sessions")
    p_train.add_argument("--sessions", nargs="+", required=True,
                         help="session directories (e.g. sessions/20260101-...-scripted)")
    p_train.add_argument("--out", default=DEFAULT_MODEL_OUT,
                         help="output path; neural models default to models/vision_bc.pt")
    p_train.add_argument("--model-type",
                         choices=["linear", "neural", "pixel-encoder", "fusion", "world-model",
                                  "entity-persistence"],
                         default="linear",
                         help="linear softmax head (default), pixel BC, pixel encoder pretrain, "
                              "learned latent fusion, the action-conditioned world model, or "
                              "the entity-persistence (object permanence) model")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=0.5, help="linear-model learning rate")
    p_train.add_argument("--neural-lr", type=float, default=1e-3, help="neural-model learning rate")
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--history", type=int, default=8)
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

    p_replay = sub.add_parser("replay", help="re-simulate a session and verify determinism")
    p_replay.add_argument("--session", required=True)
    p_replay.add_argument("--episode", default=None)
    p_replay.add_argument("--no-verify", action="store_true")
    p_replay.set_defaults(func=cmd_replay)

    p_view = sub.add_parser("view", help="inspect a recorded episode")
    p_view.add_argument("--session", required=True)
    p_view.add_argument("--episode", required=True)
    p_view.add_argument("--tail", type=int, default=10)
    p_view.set_defaults(func=cmd_view)

    p_dash = sub.add_parser("dashboard", help="aggregate metrics across all sessions")
    p_dash.add_argument("--record-dir", default="sessions")
    p_dash.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
