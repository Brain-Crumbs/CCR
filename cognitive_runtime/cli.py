"""Command-line interface for the Continuous Cognitive Runtime.

    python -m cognitive_runtime run --episodes 3
    python -m cognitive_runtime run --world minecraft --policy scripted --episodes 3
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
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

if TYPE_CHECKING:
    # Issue #176: the CLI's default --world is now "crafter", and the
    # survival-economy world (MinecraftSurvivalBox, its --backend registry,
    # the profile-driven reward system) is only imported, at runtime, from
    # the functions below that actually select --world minecraft or pass
    # --reward-profile -- not at module import time.
    from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
    from cognitive_runtime.programs.minecraft.reward_profile import RewardProfile
    from cognitive_runtime.programs.minecraft.rewards import SurvivalRewardConfig

from brain.hippocampus import Hippocampus
from cognitive_runtime.core.attention import ATTENTION_MODES
from cognitive_runtime.core.orienting_reflex import REFLEX_MODES
from cognitive_runtime.core.policy import Policy
from cognitive_runtime.core.program import Program
from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry
from cognitive_runtime.models.online_q import OnlineQModel
from cognitive_runtime.observability import DEFAULT_TRACE_DIR, configure_logging, start_run
from cognitive_runtime.observability.logs import LEVELS
from cognitive_runtime.policies import (
    HumanDemoPolicy,
    LearnedPolicy,
    NullPolicy,
    OnlineQLearner,
    OnlineQPolicy,
    RandomPolicy,
    ScriptedSurvivalPolicy,
)
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

#: Issue #32 "raw input" ablation: which stream classifications the online
#: policy's fused state is built from. "full" preserves the pre-#32 behavior
#: (encoders=None -> default_encoder_registry()); "raw" restricts fusion to
#: MINECRAFT_STREAM_REGISTRY streams classified agent_input, so hand-computed
#: semantic streams keep publishing/recording but stop reaching the policy.
INPUT_PROFILES = {"full", "raw"}

#: ``--backend`` choices for ``--world minecraft`` (the ``BACKENDS`` registry
#: itself lives in ``programs.minecraft.adapter``; its keys are mirrored here,
#: statically, so building the parser -- which needs these for every
#: subcommand's ``--backend``/``choices=`` metadata, regardless of the world
#: actually selected -- never imports the survival-economy adapter module
#: (issue #176).
MINECRAFT_BACKEND_NAMES = ("remote", "simulated")


def _default_nursery_backend() -> str:
    """Use the live backend by default when live connection env is present."""
    env_default = os.environ.get("CCR_NURSERY_BACKEND")
    if env_default in MINECRAFT_BACKEND_NAMES:
        return env_default
    if os.environ.get("CCR_MINECRAFT_HOST"):
        return "remote"
    return "simulated"


def _encoders_for_input_profile(profile: str, stream_registry=None):
    if profile == "full":
        return None
    if stream_registry is None:
        from cognitive_runtime.programs.minecraft.stream_registry import MINECRAFT_STREAM_REGISTRY
        stream_registry = MINECRAFT_STREAM_REGISTRY
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
    if args.curriculum:
        from cognitive_runtime.programs.minecraft.curriculum import get_curriculum
        preset = get_curriculum(args.curriculum)
    else:
        preset = None
    for key, default in _WORLD_DEFAULTS.items():
        if getattr(args, key, None) is None:
            value = preset.world_config.get(key, default) if preset else default
            setattr(args, key, value)
    if args.seed is None:
        args.seed = preset.seed if preset else 0


def _reward_config_for(args: argparse.Namespace) -> Optional["SurvivalRewardConfig"]:
    """The curriculum's reward-weight bundle applied over the defaults, or
    `None` (default reward config) when no curriculum was chosen."""
    if not args.curriculum:
        return None
    from cognitive_runtime.programs.minecraft.curriculum import get_curriculum
    from cognitive_runtime.programs.minecraft.rewards import SurvivalRewardConfig

    preset = get_curriculum(args.curriculum)
    return dataclasses.replace(SurvivalRewardConfig(), **preset.reward_config)


def _reward_profile_for(args: argparse.Namespace) -> Optional["RewardProfile"]:
    """The loaded `--reward-profile`, or `None` for the legacy hard-coded
    reward path.  Fails the whole invocation immediately (issue #41: "a
    malformed profile fails at startup with a clear message, not mid-run")
    rather than letting a bad profile surface as a mid-episode crash.

    Issue #176: the profile-driven reward economy (reward_profile.py) is
    only imported here, when `--reward-profile` is actually given -- not at
    CLI module import time.
    """
    path = getattr(args, "reward_profile", None)
    if not path:
        return None
    from cognitive_runtime.programs.minecraft.reward_profile import (
        RewardProfileError,
        load_reward_profile,
    )

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
        if not action_space:
            from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
            action_space = ACTION_SPACE
        return RandomPolicy(action_space, seed=args.seed)
    if name == "scripted":
        if getattr(args, "world", "minecraft") != "minecraft":
            sys.exit(
                f"--policy scripted is Minecraft-specific (a hand-authored heuristic over "
                f"Minecraft's own actions); it does not support --world {args.world!r}. "
                "Pick --policy null/random/human/online/learned/neural instead."
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


def _make_world_model(
    args: argparse.Namespace,
    program: Program,
    hippocampus: Optional[Any] = None,
):
    """The heuristic default (`None`, `TrendWorldModel`), or a trained neural
    world-model checkpoint bridged behind the same `world_model` seam.

    Two neural paths, both requiring the `neural` extra:

    - `--world-model cortex:PATH` drives the recurrent, action-conditioned
      `PredictiveCortex` as the live world model (issue #166) -- its backbone
      hidden state persists across ticks and resets on each episode boundary.
      `--async-trainer` (issue #175) additionally wires a live
      `sleep.cortex_consolidation.CortexConsolidator` into it, so the cortex
      itself is the online learner instead of a separate actor/critic stack;
      consolidated weights are persisted back to the same checkpoint path on
      every publish (`CortexWorldModel.checkpoint_path` defaults from the
      loaded path), so the run survives an episode end, crash, or interrupt.
    - `--world-model PATH` bridges the memoryless `MLPWorldModel` (issue #26).

    `--world-model` is unset unless the caller opts in. `hippocampus` is only
    consulted for the cortex path with `--async-trainer` set -- it must be the
    *same* instance the runtime loop is writing seeds into, so the
    consolidator's dream mixer has real seeds to draw from (`cmd_run` builds
    one `Hippocampus()` and passes it to both). Other callers (e.g. `cmd_demo`,
    whose args namespace has no `async_trainer` attribute) can omit it."""
    path = getattr(args, "world_model", None)
    async_trainer = getattr(args, "async_trainer", False)
    if not path:
        if async_trainer:
            sys.exit(
                "--async-trainer (cortex consolidation, issue #175) requires "
                "--world-model cortex:<ckpt>; it repoints the online learner at the "
                "predictive cortex, so a live cortex world model must be selected"
            )
        return None
    action_keys = [action.key() for action in program.metadata().action_space]
    cortex_prefix = "cortex:"
    if path.startswith(cortex_prefix):
        checkpoint = path[len(cortex_prefix):]
        if not checkpoint:
            sys.exit("--world-model cortex: needs a checkpoint path (cortex:PATH)")
        try:
            from cognitive_runtime.policies.cortex_world_model import CortexWorldModel
        except ImportError as exc:  # torch not installed
            sys.exit(f"the predictive cortex needs PyTorch ({exc}); install '.[neural]'.")
        world_model = CortexWorldModel(checkpoint, action_keys=action_keys)
        if async_trainer:
            try:
                from sleep.cortex_consolidation import CortexConsolidator
            except ImportError as exc:  # torch not installed
                sys.exit(f"cortex consolidation needs PyTorch ({exc}); install '.[neural]'.")
            world_model.consolidator = CortexConsolidator(
                cortex=world_model.model, hippocampus=hippocampus,
            )
            world_model.consolidate_every_ticks = args.async_wake_ticks
            world_model.consolidation_steps = args.async_consolidation_steps
        return world_model
    if async_trainer:
        sys.exit(
            "--async-trainer (cortex consolidation, issue #175) requires "
            "--world-model cortex:<ckpt>; it repoints the online learner at the "
            "predictive cortex, so a live cortex world model must be selected"
        )
    try:
        from cognitive_runtime.policies.neural_world_model import NeuralWorldModel
    except ImportError as exc:  # torch not installed
        sys.exit(f"the neural world model needs PyTorch ({exc}); install '.[neural]'.")
    return NeuralWorldModel(path, action_keys=action_keys)


def _add_world_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--world-model", default=None,
                        help="path to a trained neural world-model checkpoint (.pt bundle, "
                             "--model-type world-model); prefix with 'cortex:' to drive a "
                             "trained recurrent PredictiveCortex checkpoint as the live world "
                             "model (issue #166); default: the heuristic TrendWorldModel")


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
    args: argparse.Namespace, program: Program, encoders=None
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


def _add_world_args(parser: argparse.ArgumentParser) -> None:
    from cognitive_runtime.programs.minecraft.curriculum import CURRICULUM_ORDER
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
    parser.add_argument("--backend", default="simulated", choices=sorted(MINECRAFT_BACKEND_NAMES),
                        help="survival backend: the deterministic simulated world, or "
                             "a real-Minecraft client (remote; not yet implemented)")
    parser.add_argument("--reward-profile", default=None,
                        help="path to a YAML/JSON reward profile (e.g. goals/survival.yaml, "
                             "goals/ender_dragon.yaml); overrides --curriculum's reward weights "
                             "with a profile-driven reward engine (docs/history/reward_profiles.md). "
                             "Malformed profiles fail immediately with a diagnosis.")
    parser.add_argument("--intrinsic-risk-threshold", type=float, default=0.5,
                        help="risk-gated intrinsic drive (issue #61): the internal.risk level "
                             "at which internal.safe_novelty's gate is cut in half (default: 0.5)")
    parser.add_argument("--intrinsic-risk-temperature", type=float, default=0.15,
                        help="risk-gated intrinsic drive (issue #61): softness of the risk-gate "
                             "sigmoid around --intrinsic-risk-threshold (default: 0.15)")


#: Programs the ``--world`` selector can build (issue #89). Crafter is the
#: default (issue #176): it's the live V2 nursery world and needs none of
#: Minecraft's survival-economy weight (crafting/inventory/reward-profile
#: system) the predictive objective doesn't use. Minecraft stays fully
#: supported, opt-in via ``--world minecraft`` -- see
#: ``cognitive_runtime/programs/minecraft/__init__.py`` for its quarantine
#: note and the future graduation-world milestone it's kept for.
WORLDS = {"minecraft", "crafter"}


def _add_world_selector_arg(parser: argparse.ArgumentParser, default: str = "crafter") -> None:
    parser.add_argument("--world", default=default, choices=sorted(WORLDS),
                        help="which Program to run: the Crafter nursery world "
                             f"or the legacy Minecraft-like survival sim "
                             f"(default here: {default}). Both implement the "
                             "same streams-v2 seam, so the runtime/policy code "
                             "is unchanged either way.")


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
        # Issue #176: MinecraftSurvivalBox (and the survival-economy reward
        # system it pulls in) is only imported here, when --world minecraft
        # is actually selected -- never for the default (crafter) path.
        from cognitive_runtime.programs.minecraft.action_registry import MINECRAFT_ACTION_REGISTRY
        from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
        from cognitive_runtime.programs.minecraft.stream_registry import MINECRAFT_STREAM_REGISTRY

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

        # CrafterWorld imports the 'crafter' package (a core dependency,
        # issue #176) lazily, inside __init__ -- a partial/dev install
        # missing it only surfaces an ImportError here, at construction, not
        # at the module import above.
        try:
            program = CrafterWorld(config=program_config)
        except ImportError as exc:
            sys.exit(str(exc))
        return program, CRAFTER_STREAM_REGISTRY, CRAFTER_ACTION_REGISTRY
    sys.exit(f"unknown --world {world!r}; expected one of {sorted(WORLDS)}")


#: Online-learning policies whose model path needs a checkpoint-or-`--fresh`
#: decision for live runs (issue #33).
_CHECKPOINTED_POLICIES = {"online": "online_model"}


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
    has_cortex_wm = getattr(args, "world_model", None) and args.world_model.startswith("cortex:")
    if args.policy is None:
        if has_cortex_wm:
            args.policy = "cortex-mpc"
        elif world == "minecraft":
            args.policy = "scripted"
        else:
            args.policy = "random"
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
    # World model must be created before cortex-mpc policy, which reads
    # the cortex's live hidden state every tick.
    hippocampus = Hippocampus()
    world_model = _make_world_model(args, program, hippocampus)
    if args.policy == "online":
        policy, learner = _make_online_policy_and_learner(args, program, encoders)
    elif args.policy == "cortex-mpc":
        from cognitive_runtime.policies.cortex_world_model import CortexWorldModel
        if not isinstance(world_model, CortexWorldModel):
            sys.exit(
                "--policy cortex-mpc requires --world-model cortex:<ckpt>; "
                "the MPC plans over the live cortex's recurrent hidden state"
            )
        from motor.cortex_mpc import build_cortex_mpc
        from motor.organism_policy import MotorFreedomPolicy
        controller = build_cortex_mpc(world_model)
        policy = MotorFreedomPolicy("learned", action_space, voluntary=controller)
    else:
        policy = _make_policy(args.policy, args, action_space)
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
        action_registry=action_registry,
        hippocampus=hippocampus,
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
        from cognitive_runtime.programs.minecraft.evaluation import comparison_table, summarize_episodes
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
    # `evaluate` has no --world selector; it's a Minecraft-only tool (issue
    # #176: imported here, not at CLI module load, since most commands
    # never touch the survival economy).
    from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox

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
    from cognitive_runtime.programs.minecraft.evaluation import comparison_table
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
        # `statistical-evaluate` has no --world selector; it's a
        # Minecraft-only tool (issue #176: imported here, not at CLI module
        # load).
        from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox

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
    for name in scenarios:
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
    scenario_names = list(scenarios) if args.scenario == "all" else [args.scenario]

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
        navigation_random_action_fraction=args.navigation_random_action_fraction,
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
    docs/history/nursery-turn-in-place-analysis.md), evaluating in-distribution
    generalization (held-out seeds), zero-shot generality (held-out
    scenarios), rollout health (frozen-rollout detector), and a yaw linear
    probe."""
    try:
        import json

        from cognitive_runtime.training.action_world_model import (
            ActionWorldModelConfig,
            save_action_world_model,
        )
        from cognitive_runtime.training.prediction_export import ExperimentIdentity, checkpoint_sha256
        from cognitive_runtime.training.statistical_evaluation import build_experiment_report, write_experiment_report
        from cognitive_runtime.training.nursery import (
            CRAFTER_SCENARIOS,
            NURSERY_SCENARIOS,
            NurseryConfig,
            run_nursery_joint,
        )
    except ImportError as exc:  # torch not installed
        sys.exit(f"the nursery suite needs PyTorch ({exc}). Install it with 'pip install -e .[neural]'.")

    holdout_scenarios = args.holdout_scenarios or ["approach_entity"]
    train_scenarios = args.train_scenarios or None
    scenarios = CRAFTER_SCENARIOS if args.world == "crafter" else NURSERY_SCENARIOS
    for name in (train_scenarios or []) + holdout_scenarios:
        if name not in scenarios:
            sys.exit(
                f"unknown nursery scenario {name!r}; choices: {sorted(scenarios)}"
            )

    train_seeds = list(range(args.train_seeds))
    holdout_seeds = list(range(args.train_seeds, args.train_seeds + args.holdout_seeds))
    config = NurseryConfig(
        train_seeds=train_seeds,
        holdout_seeds=holdout_seeds,
        episode_ticks=args.episode_ticks,
        world_size=args.world_size,
        world=args.world,
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
        navigation_random_action_fraction=args.navigation_random_action_fraction,
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
        ema_target_decay=args.ema_target_decay,
        training_objective=args.training_objective,
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
    representation = report.representation_diagnostics
    latent = representation["latent"]
    status = (
        "PASS" if representation["passed"] else
        "N/A" if not representation["gate_evaluable"] else "FAIL"
    )
    print(
        f"representation gate: {status}; latent variance={latent['mean_variance']:.3e}, "
        f"effective rank={latent['effective_rank']:.2f}/{latent['dimensions']}"
    )

    checkpoint = {}
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        model_path = os.path.join(args.out_dir, "joint-world-model.pt")
        save_action_world_model(model_path, model, report.training_stats)
        checkpoint = {"path": os.path.abspath(model_path), "sha256": checkpoint_sha256(model_path),
                      "model_type": "predictive_cortex"}
        print(f"\njoint world model saved to {model_path}")

    if args.report or args.out_dir:
        legacy_payload = {
            "train_scenarios": report.train_scenarios,
            "holdout_scenarios": report.holdout_scenarios,
            "horizon_frames": report.horizon_frames,
            "ticks_per_frame": report.ticks_per_frame,
            "training_stats": report.training_stats,
            "scenario_metrics": report.scenario_metrics,
            "zero_shot_metrics": report.zero_shot_metrics,
            "yaw_probe": report.yaw_probe,
            "orientation_probe": report.orientation_probe,
            "representation_diagnostics": report.representation_diagnostics,
            "train_sessions": report.train_sessions,
            "eval_sessions": report.eval_sessions,
        }
        # Keep the report inspectable without notebook state. The selected
        # metric is the first evaluated scenario; every scenario's direct and
        # rollout reports remain alongside it under ``by_scenario``.
        all_metrics = {**report.scenario_metrics, **report.zero_shot_metrics}
        selected = next(iter(all_metrics.values()), {})
        identity = ExperimentIdentity.create(
            f"nursery-joint-{args.seed}", config.name or "PixelTwo"
        )
        payload = build_experiment_report(
            experiment=identity.__dict__, data_quality={"gate_enabled": config.data_quality_gate},
            split_overlap={"gate_enabled": config.split_overlap_gate}, training_stats=report.training_stats,
            direct_metrics={**(selected.get("direct") or {}), "by_scenario": {name: value.get("direct") for name, value in all_metrics.items()}},
            rollout_metrics={**(selected.get("rollout") or {}), "by_scenario": {name: value.get("rollout") for name, value in all_metrics.items()}},
            checkpoint=checkpoint,
            action_world_model_config=model_config,
            nursery_config=config,
            data_config={
                "train_scenarios": list(report.train_scenarios),
                "holdout_scenarios": list(report.holdout_scenarios),
                "train_sessions": report.train_sessions,
                "evaluation_sessions": report.eval_sessions,
                "horizon_frames": report.horizon_frames,
                "ticks_per_frame": report.ticks_per_frame,
            },
        )
        payload["joint_nursery"] = legacy_payload
        report_path = args.report or os.path.join(args.out_dir, "experiment_report.json")
        write_experiment_report(report_path, payload)
        print(f"experiment report written to {report_path}")


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
    from cognitive_runtime.programs.minecraft.evaluation import comparison_table
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
    configs, unattended. See docs/curriculum.md for the definition schema."""
    try:
        from cognitive_runtime.training.curriculum_runner import (
            CurriculumDefinitionError,
            load_curriculum_definition,
            run_curriculum,
        )
    except ImportError as exc:  # torch not installed
        sys.exit(f"the curriculum runner needs PyTorch ({exc}); install '.[neural]'.")

    try:
        definition = load_curriculum_definition(args.curriculum_file)
    except CurriculumDefinitionError as exc:
        sys.exit(str(exc))

    voluntary_ctrl = None
    cortex_ckpt = getattr(args, "cortex_checkpoint", None)
    if cortex_ckpt:
        try:
            from cognitive_runtime.policies.cortex_world_model import CortexWorldModel
            from motor.cortex_mpc import cortex_mpc_factory
        except ImportError as exc:
            sys.exit(f"cortex-mpc needs PyTorch ({exc}); install '.[neural]'.")
        cortex_wm = CortexWorldModel(cortex_ckpt)
        voluntary_ctrl = cortex_mpc_factory(cortex_wm)

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
            voluntary_controller=voluntary_ctrl,
        )
    except (CurriculumDefinitionError, ValueError) as exc:
        sys.exit(str(exc))

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


def cmd_trace_list(args: argparse.Namespace) -> None:
    """``ccr trace list``: every traced run under --trace-dir, oldest first."""
    from cognitive_runtime.observability import format_run_list, list_runs

    print(format_run_list(list_runs(getattr(args, "trace_dir", None))))


def cmd_trace_show(args: argparse.Namespace) -> None:
    """``ccr trace show [run]``: the phase tree, metric summaries, config and
    identity of one traced run (default: the latest)."""
    from cognitive_runtime.observability import format_trace_summary, load_trace

    try:
        manifest, events = load_trace(args.run, trace_dir=getattr(args, "trace_dir", None))
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    print(format_trace_summary(manifest, events, tail=args.tail))


# --------------------------------------------------------------------------- model factory (ccr factory ...)
#
# issue #228 (MF-C6, epic #212 §20): a thin dispatcher over
# cognitive_runtime.training.model_factory's runner/registry/promotion/
# corpus/confirmation modules -- argument parsing plus one call into the
# domain layer, no orchestration logic here. Every Model Factory submodule
# except the actual neural trainer stays torch-free at import time (each
# defers its own `import torch` to the function that needs it), so most
# factory commands only require the neural extra at the point they actually
# touch a checkpoint or train -- `ccr factory --help` and `ccr factory show`
# work in a core-only install.

_FACTORY_RUNS_ROOT_DEFAULT = "runs"
_FACTORY_CORPORA_ROOT_DEFAULT = "corpora"


def _factory_run_directory(root: str, run_id: str, organism: Optional[str] = None) -> Path:
    """Locate ``<root>/<organism>/<run_id>``, searching every organism under
    ``root`` when ``organism`` is not given (mirrors
    ``model_factory.corpus._corpus_directory``'s own ambiguity handling)."""
    root_path = Path(root)
    if organism:
        directory = root_path / organism / run_id
        if not directory.is_dir():
            sys.exit(f"no run {run_id!r} for organism {organism!r} under {root!r}")
        return directory
    candidates = sorted(path for path in root_path.glob(f"*/{run_id}") if path.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        sys.exit(f"run {run_id!r} was not found under {root!r}")
    organisms = ", ".join(sorted(candidate.parent.name for candidate in candidates))
    sys.exit(f"run id {run_id!r} is ambiguous under {root!r} (organisms: {organisms}); pass --organism")


def _factory_latest_run_directory(root: str) -> Path:
    """The most recently allocated run directory under ``root``, for the
    commands that default to inspecting "the latest run" when none is
    named."""
    root_path = Path(root)
    candidates = sorted(
        root_path.glob("*/*/experiment.json"), key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        sys.exit(f"no Model Factory runs found under {root!r}")
    return candidates[-1].parent


def _factory_resolve_run(args: argparse.Namespace) -> Path:
    root = args.root
    run_id = getattr(args, "run", None)
    if run_id:
        return _factory_run_directory(root, run_id, getattr(args, "organism", None))
    return _factory_latest_run_directory(root)


def _factory_load_validation(directory: Path) -> Tuple[Dict[str, Any], float]:
    """Load one run's ``metrics/validation.json`` plus its data contract's
    ``ticks_per_frame``, restoring the ``rollout``/``direct``/
    ``per_episode_model_mse`` horizon keys from JSON's string coercion back
    to ``int`` so the payload matches the shape
    ``runner._resolve_selection_metric`` expects."""
    with (directory / "metrics" / "validation.json").open(encoding="utf-8") as handle:
        payload: Dict[str, Any] = json.load(handle)
    with (directory / "contracts.json").open(encoding="utf-8") as handle:
        contracts = json.load(handle)
    ticks_per_frame = float(contracts["data_contract"]["ticks_per_frame"])
    for report_name in ("rollout", "direct"):
        report = payload.get(report_name)
        if not isinstance(report, dict):
            continue
        if isinstance(report.get("horizons"), dict):
            report["horizons"] = {int(key): value for key, value in report["horizons"].items()}
        if isinstance(report.get("per_episode_model_mse"), dict):
            report["per_episode_model_mse"] = {
                int(key): value for key, value in report["per_episode_model_mse"].items()
            }
    per_episode = payload.get("per_episode_model_mse")
    if isinstance(per_episode, dict):
        payload["per_episode_model_mse"] = {int(key): value for key, value in per_episode.items()}
    return payload, ticks_per_frame


def _factory_slot_defaults(run_directory: Path, trial_spec: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    """``(family, tier)`` defaults so ``ccr factory promote <run>``/
    ``ccr factory test <run>`` work with no extra flags, matching the epic
    proposal's own terse workflow examples: family defaults to the run's
    organism, tier to whatever ``build_budget_report`` recorded for it."""
    family = str(trial_spec["organism"])
    contracts_path = run_directory / "contracts.json"
    if contracts_path.is_file():
        with contracts_path.open(encoding="utf-8") as handle:
            data_contract = (json.load(handle).get("data_contract") or {})
        retention = data_contract.get("retention_policy") or {}
        behavior_mixture = data_contract.get("behavior_mixture_policy") or {}
        if (
            retention.get("suite") == "generic_action_effects_v1"
            and behavior_mixture.get("expert_policy") == "astar"
        ):
            # Issue #240: a navigation fine-tune defaults into its own
            # champion namespace.  It can never overwrite the generic
            # world-dynamics champion merely because --family was omitted.
            family = "goal_navigation_v1"
    tier: Optional[str] = None
    budget_report_path = run_directory / "metrics" / "budget_report.json"
    if budget_report_path.is_file():
        with budget_report_path.open(encoding="utf-8") as handle:
            tier = json.load(handle).get("tier")
    return family, tier


def _factory_genome_value(training: Mapping[str, Any], dotted_path: str) -> Any:
    """Read one declared genome field from a resolved training block."""
    value: Any = training
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"resolved training block is missing declared genome field {dotted_path!r}")
        value = value[part]
    return value


def _factory_present_genome_fields(
    training: Mapping[str, Any], declared_fields: Sequence[str],
) -> Tuple[str, ...]:
    """Keep only schema-declared paths represented by this spec generation.

    A schema can contain an inactive or not-yet-materialized optional field;
    it is not a diversity dimension for a run that has no resolved value for
    it.  The retained fields are still declared schema fields, never runtime
    state or an inferred implementation detail.
    """
    present = []
    for field in declared_fields:
        try:
            _factory_genome_value(training, field)
        except ValueError:
            continue
        present.append(field)
    if not present:
        raise ValueError("no declared genome fields are present in the resolved training specification")
    return tuple(present)


def _factory_compute_ledger(run_directory: Path, *, seen: Optional[set[str]] = None) -> Any:
    """Recover inclusive compute cost along the one checkpoint-donor lineage."""
    from cognitive_runtime.training.model_factory.population import ComputeLedger

    seen = set() if seen is None else set(seen)
    run_id = run_directory.name
    if run_id in seen:
        raise ValueError(f"cycle while recovering compute lineage for {run_id!r}")
    seen.add(run_id)
    with (run_directory / "metrics" / "budget_report.json").open(encoding="utf-8") as handle:
        budget = json.load(handle)
    trial_compute = float(budget.get("total_trial_seconds", budget.get("measured_training_seconds", 0.0)))
    lineage_path = run_directory / "lineage.json"
    if not lineage_path.is_file():
        return ComputeLedger.fresh(trial_compute)
    with lineage_path.open(encoding="utf-8") as handle:
        lineage = json.load(handle)
    donor_run_id = lineage.get("weight_donor") or (lineage.get("parent") or {}).get("run_id")
    if not donor_run_id:
        return ComputeLedger.fresh(trial_compute)
    return ComputeLedger.child(
        trial_compute,
        _factory_compute_ledger(run_directory.parent / str(donor_run_id), seen=seen),
    )


def _factory_population_candidate(
    run_directory: Path,
    *,
    objective: str,
    declared_genome_fields: Sequence[str],
    gates: Optional[Mapping[str, bool]] = None,
) -> Any:
    """Build a validation-only D5 candidate from persisted run artifacts."""
    from cognitive_runtime.training.model_factory.population import PopulationCandidate
    from cognitive_runtime.training.model_factory.runner import _resolve_selection_metric

    with (run_directory / "trial_spec.json").open(encoding="utf-8") as handle:
        trial_spec = json.load(handle)
    validation, ticks_per_frame = _factory_load_validation(run_directory)
    primary_metric, _ = _resolve_selection_metric(validation, objective, ticks_per_frame)
    retention_metric = float(primary_metric)
    retention_path = run_directory / "metrics" / "retention.json"
    if retention_path.is_file():
        with retention_path.open(encoding="utf-8") as handle:
            retention_payload = json.load(handle)
        retention_metric = float(retention_payload["forgetting_amount"])
    with (run_directory / "metrics" / "budget_report.json").open(encoding="utf-8") as handle:
        budget = json.load(handle)
    return PopulationCandidate(
        run_id=run_directory.name,
        resolved_spec={key: value for key, value in trial_spec.items() if key != "format"},
        genome={
            field: _factory_genome_value(trial_spec["training"], field)
            for field in declared_genome_fields
        },
        primary_metric=float(primary_metric),
        runtime=float(budget.get("total_trial_seconds", budget.get("measured_training_seconds", 0.0))),
        # Generic dynamics has no separate report and retains its validation
        # objective here. Navigation fine-tunes use the CI-refereed forgetting
        # amount written by the runner instead.
        retention_metric=retention_metric,
        ledger=_factory_compute_ledger(run_directory),
        gates=dict(gates or {}),
        completion_status=str(budget.get("completion_status", "completed")),
    )


def _print_trial_result(result: Any) -> None:
    print(f"run_id: {result.run_id}")
    print(f"display_name: {result.display_name}")
    print(f"mode: {result.mode}")
    print(f"state: {result.state}")
    print(f"directory: {result.directory}")
    print(f"architecture_hash: {result.architecture_hash}")
    print(f"data_contract_hash: {result.data_contract_hash}")
    print(f"training_contract_hash: {result.training_contract_hash}")
    print(f"checkpoint: {result.checkpoint_path} (sha256={result.checkpoint_sha256})")
    if result.comparison is not None:
        print(
            f"comparison: decision={result.comparison.get('decision')} "
            f"mean_delta={result.comparison.get('mean_delta')}"
        )


def cmd_factory_baseline(args: argparse.Namespace) -> None:
    """``ccr factory baseline <spec>`` (issue #228, epic #212 §20 workflow
    step 1): resolve an experiment spec and launch one fresh/clone/resume/
    fine_tune trial through ``run_trial``, establishing a fully recorded,
    immutable run with a unique run ID."""
    from cognitive_runtime.training.model_factory.spec import SpecError, apply_overrides, load_spec, resolve

    raw = load_spec(args.spec)
    if args.set:
        raw = apply_overrides(raw, args.set)
    try:
        resolved = resolve(raw)
    except SpecError as exc:
        sys.exit(f"invalid spec: {exc}")

    try:
        from cognitive_runtime.training.model_factory.runner import run_trial

        result = run_trial(
            resolved, root=args.root, corpus_root=args.corpus_root, naming_seed=args.naming_seed,
            export_predictions=not args.no_export_predictions,
            export_predictions_max_episodes=args.export_predictions_max,
        )
    except ImportError as exc:
        sys.exit(f"'ccr factory baseline' needs PyTorch ({exc}). Install it with 'pip install -e .[neural]'.")
    _print_trial_result(result)


def cmd_factory_clone(args: argparse.Namespace) -> None:
    """``ccr factory clone <run> --set path=value`` (issue #228): build a
    clone/fine_tune child spec from an existing run's persisted
    ``trial_spec.json``, apply dotted-path ``--set`` overrides, and launch it
    as a new controlled sibling via ``run_trial``.

    An unknown dotted path (a typo'd field, or one nested a level too deep)
    is rejected by the same ``resolve()``/``validate()`` machinery every
    other spec goes through: :class:`SpecError` already names the nearest
    valid field via ``difflib``, so this command does not duplicate that
    logic -- it only lets the error surface.
    """
    from cognitive_runtime.training.model_factory.checkpoint import read_factory_checkpoint_metadata
    from cognitive_runtime.training.model_factory.spec import SpecError, apply_overrides, resolve

    parent_directory = _factory_run_directory(args.root, args.run, args.organism)
    with (parent_directory / "trial_spec.json").open(encoding="utf-8") as handle:
        parent_spec = json.load(handle)

    checkpoint_path = parent_directory / "checkpoints" / args.checkpoint
    try:
        checkpoint_meta = read_factory_checkpoint_metadata(str(checkpoint_path))
    except OSError as exc:
        sys.exit(f"cannot read parent checkpoint metadata at {checkpoint_path}: {exc}")

    child_doc = {
        "organism": parent_spec["organism"],
        "mode": args.mode,
        "data": parent_spec["data"],
        "model": parent_spec["model"],
        "training": parent_spec["training"],
        "evaluation": parent_spec["evaluation"],
        "parent": {
            "run_id": args.run,
            "checkpoint": args.checkpoint,
            "sha256": checkpoint_meta.get("checkpoint_sha256"),
        },
    }
    if args.set:
        child_doc = apply_overrides(child_doc, args.set)

    try:
        resolved = resolve(child_doc)
    except SpecError as exc:
        sys.exit(f"invalid clone: {exc}")

    try:
        from cognitive_runtime.training.model_factory.runner import run_trial

        result = run_trial(
            resolved, root=args.root, corpus_root=args.corpus_root, naming_seed=args.naming_seed,
            export_predictions=not args.no_export_predictions,
            export_predictions_max_episodes=args.export_predictions_max,
        )
    except ImportError as exc:
        sys.exit(f"'ccr factory clone' needs PyTorch ({exc}). Install it with 'pip install -e .[neural]'.")
    _print_trial_result(result)


def cmd_factory_compare(args: argparse.Namespace) -> None:
    """``ccr factory compare <run> <run> [<run> ...]`` (issue #228): pair
    every candidate's frozen validation evidence against the first named run
    (the baseline) with the same paired bootstrap/permutation rule (MF-C1)
    promotion itself uses. Read-only: no trial is launched, nothing is
    written."""
    from cognitive_runtime.training.model_factory.comparison import compare_paired_episodes
    from cognitive_runtime.training.model_factory.runner import _resolve_selection_metric

    baseline_id, *candidate_ids = args.runs
    if not candidate_ids:
        sys.exit("ccr factory compare needs a baseline run and at least one candidate run")

    def series_for(run_id: str) -> Tuple[str, Dict[str, float]]:
        directory = _factory_run_directory(args.root, run_id, args.organism)
        payload, ticks_per_frame = _factory_load_validation(directory)
        selection_metric = payload["selection_metric"]
        _, per_episode = _resolve_selection_metric(payload, selection_metric, ticks_per_frame)
        return selection_metric, dict(zip(payload["episode_ids"], per_episode))

    baseline_metric, baseline_series = series_for(baseline_id)
    print(f"baseline: {baseline_id} (selection_metric={baseline_metric})")
    for candidate_id in candidate_ids:
        candidate_metric, candidate_series = series_for(candidate_id)
        if candidate_metric != baseline_metric:
            sys.exit(
                f"cannot compare {candidate_id!r} (selection_metric={candidate_metric!r}) against "
                f"{baseline_id!r} (selection_metric={baseline_metric!r}): selection metrics differ"
            )
        # minimum_episode_count=1 matches runner._build_comparison's own override of
        # compare_paired_episodes's default (5): whether a handful of paired episodes is
        # *statistically* enough to trust is a promotion-policy question, not a reason for
        # this read-only command to contradict the comparison persisted for the same runs.
        comparison = compare_paired_episodes(
            candidate_series, baseline_series, primary_metric=baseline_metric, minimum_episode_count=1,
        )
        decision = (
            "candidate_improves"
            if comparison.status == "evaluable" and comparison.mean_delta is not None and comparison.mean_delta < 0
            else "hold"
        )
        detail = "" if comparison.status == "evaluable" else f" ({comparison.reason})"
        print(
            f"{candidate_id}: status={comparison.status} mean_delta={comparison.mean_delta} "
            f"win_rate={comparison.win_rate} decision={decision}{detail}"
        )


def cmd_factory_promote(args: argparse.Namespace) -> None:
    """``ccr factory promote <run>`` (issue #228): gate a run against
    ``promotion.evaluate_promotion``'s promotion conditions (data
    quality, split overlap, rollout health, primary-metric margin, safety,
    navigation retention, training-time budget, and -- with ``--durable`` -- sealed-test
    confirmation), then record it as a champion-population member of one
    ``(family, tier, objective)`` registry slot only if every gate passes.
    A candidate that fails any gate is refused and recorded as a hold
    instead, exactly like the workflow's own "promote only if gates pass"
    (``--hold`` records an evaluated-but-not-promoted decision directly,
    skipping gate evaluation). ``--family``/``--tier``/``--objective``
    default from the run's own organism, recorded budget tier, and declared
    selection metric."""
    from cognitive_runtime.training.model_factory.checkpoint import read_factory_checkpoint_metadata
    from cognitive_runtime.training.model_factory.corpus import resolve_corpus
    from cognitive_runtime.training.model_factory.genome import GENERIC_ACTION_EFFECTS_V1
    from cognitive_runtime.training.model_factory.population import PopulationPolicy
    from cognitive_runtime.training.model_factory.promotion import (
        PromotionPolicy,
        TestConfirmation,
        evaluate_promotion,
    )
    from cognitive_runtime.training.model_factory.registry import (
        GOAL_NAVIGATION_V1,
        RegistryError,
        hold,
        leading_champion,
        population,
        promote,
    )
    from cognitive_runtime.training.model_factory.runner import _resolve_selection_metric

    run_directory = _factory_run_directory(args.root, args.run, args.organism)
    with (run_directory / "trial_spec.json").open(encoding="utf-8") as handle:
        trial_spec = json.load(handle)

    default_family, default_tier = _factory_slot_defaults(run_directory, trial_spec)
    family = args.family or default_family
    tier = args.tier or default_tier
    objective = args.objective or trial_spec["evaluation"]["selection_metric"]
    if not tier:
        sys.exit(
            "ccr factory promote: no --tier given and none could be inferred from "
            "metrics/budget_report.json; pass --tier explicitly"
        )

    registry_root = run_directory.parent

    if args.hold:
        try:
            slot = hold(registry_root, family=family, tier=tier, objective=objective,
                        run_id=args.run, reason=args.reason)
        except RegistryError as exc:
            sys.exit(f"promotion refused: {exc}")
        print(f"family={family} tier={tier} objective={objective}")
        print(f"leading_champion: {slot.leading_champion}")
        print(f"population: {[entry.run_id for entry in slot.population]}")
        return

    experiment_report_path = run_directory / "experiment_report.json"
    if not experiment_report_path.is_file():
        sys.exit(
            f"ccr factory promote: no experiment_report.json for {args.run!r}; "
            "the run is not gate-evaluable"
        )
    with experiment_report_path.open(encoding="utf-8") as handle:
        experiment_report = json.load(handle)

    corpus = resolve_corpus(
        trial_spec["data"]["corpus_id"], root=args.corpus_root, organism=trial_spec["organism"],
    )
    with (corpus.directory / "quality_report.json").open(encoding="utf-8") as handle:
        data_quality = json.load(handle)
    with (corpus.directory / "split_overlap_report.json").open(encoding="utf-8") as handle:
        split_overlap = json.load(handle)

    comparison = None
    comparison_path = run_directory / "metrics" / "comparison.json"
    if comparison_path.is_file():
        with comparison_path.open(encoding="utf-8") as handle:
            comparison = json.load(handle)

    has_champion = leading_champion(registry_root, family=family, tier=tier, objective=objective) is not None

    test_confirmation = None
    if args.durable:
        test_metrics_path = run_directory / "metrics" / "test.json"
        if not test_metrics_path.is_file():
            sys.exit(
                f"ccr factory promote --durable: no metrics/test.json for {args.run!r}; "
                "run 'ccr factory test' first"
            )
        with test_metrics_path.open(encoding="utf-8") as handle:
            test_payload = json.load(handle)
        test_confirmation = TestConfirmation(
            run_id=test_payload["run_id"], performed=True, passed=test_payload["passed"],
            test_uses_after=test_payload["test_uses_after"],
            max_sealed_test_uses=test_payload["max_sealed_test_uses"], reason=test_payload.get("reason"),
        )

    policy = PromotionPolicy(minimum_practical_margin=args.minimum_practical_margin)
    retention_report = None
    if family == GOAL_NAVIGATION_V1.name:
        retention_path = run_directory / "metrics" / "retention.json"
        if retention_path.is_file():
            with retention_path.open(encoding="utf-8") as handle:
                retention_report = json.load(handle)
    verdict = evaluate_promotion(
        policy=policy, candidate_run_id=args.run, experiment_report=experiment_report,
        data_quality=data_quality, split_overlap=split_overlap, has_champion=has_champion,
        comparison=comparison, durable=args.durable, test_confirmation=test_confirmation,
        target_family=family,
        retention_report=retention_report,
        retention_policy=dict(corpus.data_contract.retention_policy),
    )
    for gate in verdict.gates:
        status = "PASS" if gate.passed else ("N/A" if not gate.applicable else "FAIL")
        print(f"[{status}] {gate.name}: {gate.reason}")

    if not verdict.promoted:
        gate_reason = "; ".join(verdict.reasons)
        hold_reason = f"{args.reason}; {gate_reason}" if args.reason else gate_reason
        try:
            hold(registry_root, family=family, tier=tier, objective=objective, run_id=args.run, reason=hold_reason)
        except RegistryError:
            pass  # best-effort audit trail; the refusal below is what matters
        sys.exit(f"promotion refused (gates failed): {gate_reason}")

    try:
        checkpoint_path = run_directory / "checkpoints" / "best-validation.pt"
        checkpoint_meta = read_factory_checkpoint_metadata(str(checkpoint_path))
        validation_payload, ticks_per_frame = _factory_load_validation(run_directory)
        metric_value, _ = _resolve_selection_metric(validation_payload, objective, ticks_per_frame)
        population_policy = PopulationPolicy(declared_genome_fields=_factory_present_genome_fields(
            trial_spec["training"], GENERIC_ACTION_EFFECTS_V1.gene_names,
        ))
        existing_members = population(
            registry_root, family=family, tier=tier, objective=objective,
        )
        candidates = [
            _factory_population_candidate(
                registry_root / entry.run_id,
                objective=objective,
                declared_genome_fields=population_policy.declared_genome_fields,
            )
            for entry in existing_members if entry.run_id != args.run
        ]
        candidates.append(_factory_population_candidate(
            run_directory,
            objective=objective,
            declared_genome_fields=population_policy.declared_genome_fields,
            gates={gate.name: gate.passed for gate in verdict.gates},
        ))
        candidate_ledger = candidates[-1].ledger
        slot = promote(
            registry_root, family=family, tier=tier, objective=objective, run_id=args.run,
            checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_meta.get("checkpoint_sha256"),
            metrics={
                "selection_metric": objective,
                "selection_metric_value": metric_value,
                "compute_ledger": candidate_ledger.to_dict(),
            },
            as_leading=not args.no_as_leading, reason=args.reason,
            population_policy=population_policy, population_candidates=tuple(candidates),
        )
    except RegistryError as exc:
        sys.exit(f"promotion refused: {exc}")

    print(f"family={family} tier={tier} objective={objective}")
    print(f"leading_champion: {slot.leading_champion}")
    print(f"population: {[entry.run_id for entry in slot.population]}")


def cmd_factory_show(args: argparse.Namespace) -> None:
    """``ccr factory show [run]`` (issue #228, default: the latest run):
    print the run's complete effective config, all three contract hashes,
    its display name, and its completion status -- epic #212 success
    criterion 5. Reads only persisted JSON manifests, so this works with
    torch not installed."""
    from cognitive_runtime.training.model_factory.naming import load_display_name
    from cognitive_runtime.training.model_factory.state import load_state, state_path

    directory = _factory_resolve_run(args)
    run_id = directory.name
    organism = directory.parent.name

    with (directory / "trial_spec.json").open(encoding="utf-8") as handle:
        trial_spec = json.load(handle)
    with (directory / "contracts.json").open(encoding="utf-8") as handle:
        contracts = json.load(handle)
    display_name = load_display_name(directory / "display_name.json").display_name
    state_file = state_path(directory)
    completion_status = load_state(state_file).state if state_file.exists() else "unknown"

    print(f"run_id: {run_id}")
    print(f"organism: {organism}")
    print(f"display_name: {display_name}")
    print(f"mode: {trial_spec.get('mode')}")
    print(f"completion_status: {completion_status}")
    print(f"architecture_hash: {contracts['architecture_hash']}")
    print(f"data_contract_hash: {contracts['data_contract_hash']}")
    print(f"training_contract_hash: {contracts['training_contract_hash']}")

    validation_path = directory / "metrics" / "validation.json"
    if validation_path.is_file():
        with validation_path.open(encoding="utf-8") as handle:
            print(f"selection_metric: {json.load(handle).get('selection_metric')}")
    comparison_path = directory / "metrics" / "comparison.json"
    if comparison_path.is_file():
        with comparison_path.open(encoding="utf-8") as handle:
            print(f"comparison decision: {json.load(handle).get('decision')}")

    print("resolved spec (complete effective config):")
    print(json.dumps(trial_spec, indent=2, sort_keys=True))


def _print_lineage_node(node: Any, *, indent: int) -> None:
    prefix = "  " * indent
    print(f"{prefix}{node.run_id} (mode={node.mode})")
    if node.parent_run_id:
        print(f"{prefix}  parent: {node.parent_run_id}")
    if node.configuration_parents:
        print(f"{prefix}  configuration_parents: {', '.join(node.configuration_parents)}")
    if node.weight_donor:
        print(f"{prefix}  weight_donor: {node.weight_donor}")


def cmd_factory_lineage(args: argparse.Namespace) -> None:
    """``ccr factory lineage [run]`` (issue #228, default: the latest run):
    render the ancestor chain walked from ``lineage.json`` -- both
    configuration parents and the weight donor for a bred child. Reads only
    persisted JSON manifests, so this works with torch not installed."""
    from cognitive_runtime.training.model_factory.registry import RegistryError, lineage_graph

    directory = _factory_resolve_run(args)
    run_id = directory.name
    organism_directory = directory.parent

    try:
        graph = lineage_graph(organism_directory, run_id)
    except RegistryError as exc:
        sys.exit(str(exc))

    print(f"lineage for {run_id} (organism={organism_directory.name}):")
    _print_lineage_node(graph.nodes[run_id], indent=0)
    for ancestor_id in graph.ancestor_ids:
        _print_lineage_node(graph.nodes[ancestor_id], indent=1)


def cmd_factory_corpus_build(args: argparse.Namespace) -> None:
    """``ccr factory corpus build <spec>`` (issue #228): record/reuse every
    declared episode, run the corpus's quality and split-overlap gates, and
    freeze its session hashes via ``build_corpus``."""
    from cognitive_runtime.training.model_factory.spec import load_spec

    raw = load_spec(args.spec)
    raw.setdefault("root", args.root)

    try:
        from cognitive_runtime.training.model_factory.corpus import build_corpus

        corpus = build_corpus(raw)
    except ImportError as exc:
        sys.exit(f"'ccr factory corpus build' needs PyTorch ({exc}). Install it with 'pip install -e .[neural]'.")

    sessions = corpus.manifest.get("sessions", {})
    print(f"corpus_id: {corpus.corpus_id}")
    print(f"directory: {corpus.directory}")
    print(f"data_contract_hash: {corpus.data_contract_hash}")
    for split in ("train", "validation", "test"):
        print(f"{split}: {len(sessions.get(split, []))} sessions")


def cmd_factory_test(args: argparse.Namespace) -> None:
    """``ccr factory test <run>`` (issue #228, MF-C5): the sealed-test final
    action -- evaluate a candidate already confirmed across seeds against the
    sealed test split exactly once, charging its sealed-test-use budget."""
    run_directory = _factory_run_directory(args.root, args.run, args.organism)
    organism = run_directory.parent.name
    with (run_directory / "trial_spec.json").open(encoding="utf-8") as handle:
        trial_spec = json.load(handle)

    default_family, default_tier = _factory_slot_defaults(run_directory, trial_spec)
    family = args.family or default_family
    tier = args.tier or default_tier
    objective = args.objective or trial_spec["evaluation"]["selection_metric"]
    if not tier:
        sys.exit(
            "ccr factory test: no --tier given and none could be inferred from "
            "metrics/budget_report.json; pass --tier explicitly"
        )

    try:
        from cognitive_runtime.training.model_factory.confirmation import ConfirmationError, final_test
        from cognitive_runtime.training.model_factory.promotion import (
            PromotionPolicy,
            SealedTestBudgetExhaustedError,
        )

        policy = PromotionPolicy(
            minimum_practical_margin=args.minimum_practical_margin,
            max_sealed_test_uses=args.max_sealed_test_uses,
        )
        result = final_test(
            args.run, organism=organism, family=family, tier=tier, objective=objective,
            policy=policy, root=args.root, corpus_root=args.corpus_root, reason=args.reason,
        )
    except ImportError as exc:
        sys.exit(f"'ccr factory test' needs PyTorch ({exc}). Install it with 'pip install -e .[neural]'.")
    except (ConfirmationError, SealedTestBudgetExhaustedError) as exc:
        sys.exit(str(exc))

    print(f"run_id: {result.run_id}")
    print(f"passed: {result.passed}")
    print(f"reason: {result.reason}")
    print(f"selection_metric: {result.selection_metric} = {result.selection_metric_value}")
    print(
        f"sealed_test_uses: {result.test_confirmation.test_uses_after}/"
        f"{result.test_confirmation.max_sealed_test_uses}"
    )
    print(f"metrics: {result.test_metrics_path}")


# --------------------------------------------------------------------------- observability wiring


def _add_observability_args(parser: argparse.ArgumentParser) -> None:
    """Logging/tracing flags, added to the top-level parser *and* to every
    subparser so they work in either position (``ccr --log-level debug
    nursery joint`` and ``ccr nursery joint --log-level debug``).

    Every option defaults to ``SUPPRESS`` so an unset subparser copy never
    clobbers a value given before the subcommand.
    """
    group = parser.add_argument_group("logging & tracing")
    group.add_argument("--log-level", choices=list(LEVELS), default=argparse.SUPPRESS,
                       help="console log level (default: $CCR_LOG_LEVEL or 'info'). "
                            "'debug' adds per-span/per-metric detail")
    group.add_argument("--log-file", default=argparse.SUPPRESS,
                       help="also write every log record at DEBUG to this file, "
                            "whatever the console level is")
    group.add_argument("--log-format", choices=["human", "json"], default=argparse.SUPPRESS,
                       help="'human' (default) or 'json' (one object per line, for CI)")
    group.add_argument("--trace-dir", default=argparse.SUPPRESS,
                       help=f"root directory for run traces (default: $CCR_TRACE_DIR or "
                            f"{DEFAULT_TRACE_DIR}); each run writes "
                            f"<trace-dir>/<run-id>/{{manifest.json,trace.jsonl}}")
    group.add_argument("--run-id", default=argparse.SUPPRESS,
                       help="name this run's trace directory instead of generating an id")
    group.add_argument("--no-trace", action="store_true", default=argparse.SUPPRESS,
                       help="disable run tracing (logging still works)")


def _add_observability_args_everywhere(parser: argparse.ArgumentParser) -> None:
    """Recursively attach the observability flags to every subparser."""
    _add_observability_args(parser)
    for action in parser._actions:  # noqa: SLF001 - argparse exposes no public walk
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            for subparser in set(action.choices.values()):
                _add_observability_args_everywhere(subparser)


def _run_name(args: argparse.Namespace) -> str:
    """``nursery joint`` -> ``nursery.joint``: the trace's name and the stem
    of its generated run id."""
    parts = [args.command]
    for attr in ("nursery_command", "trace_command", "factory_command", "factory_corpus_command"):
        value = getattr(args, attr, None)
        if value:
            parts.append(str(value))
    scenario = getattr(args, "scenario", None)
    if scenario:
        parts.append(str(scenario))
    return ".".join(parts)


#: Read-only inspection commands.  They produce no result worth reproducing
#: and get run often, so tracing them would bury the actual training runs in
#: ``ccr trace list``.
_UNTRACED_COMMANDS = frozenset({
    "trace", "view", "dashboard", "review", "nursery.list",
    "factory.show", "factory.lineage", "factory.compare",
})


def _should_trace(args: argparse.Namespace, name: str) -> bool:
    if getattr(args, "no_trace", False):
        return False
    return args.command not in _UNTRACED_COMMANDS and name not in _UNTRACED_COMMANDS


def _trace_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Snapshot the parsed arguments into the trace manifest: the point of a
    trace is being able to answer "what exactly produced this number?"."""
    skip = {"func", "log_level", "log_file", "log_format", "trace_dir", "run_id", "no_trace"}
    return {
        key: value for key, value in sorted(vars(args).items())
        if key not in skip and not key.startswith("_")
    }


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
    p_run.add_argument("--policy", default=None,
                       choices=["null", "random", "scripted", "learned", "neural", "online",
                                "human", "cortex-mpc"],
                       help="default: 'cortex-mpc' when --world-model cortex:* is given "
                            "(one-step MPC over the live cortex), 'scripted' for --world "
                            "minecraft, 'random' otherwise")
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
                       help="initialize online weights fresh even though no checkpoint "
                            "exists yet at the model path; required for --backend remote "
                            "with no existing checkpoint (issue #33)")
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
    p_run.add_argument("--async-trainer", dest="async_trainer", action="store_true",
                       default=False,
                       help="issue #175: consolidate the live predictive cortex online -- "
                            "requires --world-model cortex:<ckpt>. Every --async-wake-ticks "
                            "ticks, the tick blocks for a bounded sleep pass (a "
                            "sleep.cortex_consolidation.CortexConsolidator draws quality-"
                            "gated replay batches -- real transitions recorded from every "
                            "tick, mixed with guardrailed generative dreams -- and takes "
                            "--async-consolidation-steps gradient steps), then publishes "
                            "the result back into the cortex. The world model is the "
                            "online learner; nothing in the motor path trains online.")
    p_run.add_argument("--async-wake-ticks", type=int, default=50,
                       help="number of acting ticks between cortex consolidation passes "
                            "(only with --async-trainer)")
    p_run.add_argument("--async-consolidation-steps", type=int, default=50,
                       help="gradient steps taken in each consolidation pass "
                            "(only with --async-trainer)")
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
    # `demo` forces HumanDemoPolicy (cmd_demo below), a Minecraft-specific
    # terminal keymap/status display (world.front_block, body.hunger, ...);
    # it doesn't understand Crafter's action space or streams, so unlike
    # `run`/`nursery`, this default stays "minecraft" (issue #193 review).
    _add_world_selector_arg(p_demo, default="minecraft")
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
    from cognitive_runtime.programs.minecraft.curriculum import CURRICULUM_ORDER as _CURRICULUM_ORDER
    p_gates.add_argument("--curriculum", default=None, choices=_CURRICULUM_ORDER,
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
             "config plus promotion criteria (docs/curriculum.md)",
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
    p_curriculum_run.add_argument("--cortex-checkpoint", default=None,
                                   help="path to a PredictiveCortex checkpoint; enables "
                                        "cortex-MPC as the voluntary controller for "
                                        "'learned' motor-freedom stages")
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
    p_nursery_list.add_argument("--world", default="crafter", choices=sorted(WORLDS),
                                help="issue #90/#176: list Crafter's scenarios (default) or "
                                     "Minecraft's legacy ports (--world minecraft)")
    p_nursery_list.set_defaults(func=cmd_nursery_list)

    p_nursery_run = nursery_sub.add_parser(
        "run", help="record + benchmark one scenario (or 'all' for the whole suite)"
    )
    p_nursery_run.add_argument(
        "scenario", help="scenario name (see 'nursery list'), or 'all' to run the full suite"
    )
    p_nursery_run.add_argument("--world", default="crafter", choices=sorted(WORLDS),
                               help="issue #90/#176: record against the Crafter nursery world "
                                    "(default) or Minecraft's legacy simulated/remote backend "
                                    "(--world minecraft). --backend/--world-size only apply to "
                                    "--world minecraft.")
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
    p_nursery_run.add_argument(
        "--navigation-random-action-fraction", type=float, default=0.25,
        help="navigate_* only: seeded random-cardinal injection probability; "
             "the goal_navigation_v1 default is 0.25 (epic starting band 0.20-0.30)",
    )
    p_nursery_run.add_argument("--world-size", type=int, default=48)
    p_nursery_run.add_argument("--backend", default=_default_nursery_backend(),
                               choices=sorted(MINECRAFT_BACKEND_NAMES),
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
    p_nursery_joint.add_argument("--world", default="crafter", choices=sorted(WORLDS),
                                 help="world providing pixel frames and optional semantic grids")
    p_nursery_joint.add_argument("--record-dir", default="sessions")
    p_nursery_joint.add_argument("--train-scenarios", nargs="+", default=None,
                                 help="scenarios to train on (default: every scenario not held out)")
    p_nursery_joint.add_argument("--holdout-scenarios", nargs="+", default=None,
                                 help="scenarios excluded from training and evaluated zero-shot "
                                      "(default: approach_entity)")
    p_nursery_joint.add_argument("--train-seeds", type=int, default=6)
    p_nursery_joint.add_argument("--holdout-seeds", type=int, default=2)
    p_nursery_joint.add_argument("--episode-ticks", type=int, default=400)
    p_nursery_joint.add_argument(
        "--navigation-random-action-fraction", type=float, default=0.25,
        help="navigate_* only: seeded random-cardinal injection probability; "
             "the goal_navigation_v1 default is 0.25 (epic starting band 0.20-0.30)",
    )
    p_nursery_joint.add_argument("--world-size", type=int, default=48)
    p_nursery_joint.add_argument("--backend", default=_default_nursery_backend(),
                                 choices=sorted(MINECRAFT_BACKEND_NAMES))
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
    p_nursery_joint.add_argument(
        "--ema-target-decay", type=float, default=None,
        help="enable a Polyak target encoder for latent targets (e.g. 0.99); off by default",
    )
    p_nursery_joint.add_argument(
        "--training-objective", choices=["windowed_rollout", "autoregressive"],
        default="windowed_rollout", help="cortex training objective",
    )
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
                                 choices=sorted(MINECRAFT_BACKEND_NAMES))
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
                               "content -- see docs/history/reward_profiles.md)")
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

    p_trace = sub.add_parser(
        "trace",
        help="inspect run traces written by any other command "
             "(phase timings, metric curves, config, git commit)",
    )
    trace_sub = p_trace.add_subparsers(dest="trace_command", required=True)

    p_trace_list = trace_sub.add_parser("list", help="list traced runs, oldest first")
    p_trace_list.set_defaults(func=cmd_trace_list)

    p_trace_show = trace_sub.add_parser(
        "show", help="show one run's phase tree, metrics, config and identity"
    )
    p_trace_show.add_argument("run", nargs="?", default=None,
                              help="run id, run directory, or trace.jsonl path "
                                   "(default: the latest run)")
    p_trace_show.add_argument("--tail", type=int, default=0,
                              help="also print the last N raw trace events")
    p_trace_show.set_defaults(func=cmd_trace_show)

    p_factory = sub.add_parser(
        "factory",
        help="issue #228 (Model Factory, epic #212): baseline/clone/compare/promote "
             "reproducible checkpoint lineage, plus lineage/config inspection and the "
             "sealed final-test action",
    )
    factory_sub = p_factory.add_subparsers(dest="factory_command", required=True)

    p_factory_baseline = factory_sub.add_parser(
        "baseline", help="resolve a spec and launch one fresh/clone/resume/fine_tune trial"
    )
    p_factory_baseline.add_argument("spec", help="experiment spec file (.yaml/.yml/.json)")
    p_factory_baseline.add_argument("--root", default=_FACTORY_RUNS_ROOT_DEFAULT,
                                    help=f"runs root directory (default: {_FACTORY_RUNS_ROOT_DEFAULT!r})")
    p_factory_baseline.add_argument("--corpus-root", default=None,
                                    help=f"corpora root directory (default: {_FACTORY_CORPORA_ROOT_DEFAULT!r})")
    p_factory_baseline.add_argument("--set", action="append", metavar="dotted.path=value", default=None,
                                    help="override a resolved spec field before launching, e.g. "
                                         "--set training.batch_size=64 (repeatable)")
    p_factory_baseline.add_argument("--naming-seed", default=None,
                                    help="seed the run's cosmetic display name deterministically "
                                         "(default: a fresh random seed)")
    p_factory_baseline.add_argument("--export-predictions-max", type=int, default=3, metavar="N",
                                    help="clinic (viewer/) prediction export: up to N validation episodes "
                                         "from the promoted checkpoint (default: 3)")
    p_factory_baseline.add_argument("--no-export-predictions", action="store_true",
                                    help="skip the clinic prediction export")
    p_factory_baseline.set_defaults(func=cmd_factory_baseline)

    p_factory_clone = factory_sub.add_parser(
        "clone", help="build a clone/fine_tune child spec from an existing run and launch it"
    )
    p_factory_clone.add_argument("run", help="parent run id to clone from")
    p_factory_clone.add_argument("--set", action="append", metavar="dotted.path=value", default=None,
                                 help="override the parent's resolved spec field, e.g. "
                                      "--set training.loss_weights.closed_loop_pixel=0.125 (repeatable)")
    p_factory_clone.add_argument("--mode", choices=["clone", "fine_tune"], default="clone",
                                 help="child trial mode (default: clone)")
    p_factory_clone.add_argument("--checkpoint", default="best-validation.pt",
                                 help="parent checkpoint file name under its checkpoints/ directory "
                                      "(default: best-validation.pt)")
    p_factory_clone.add_argument("--organism", default=None,
                                 help="parent run's organism, only needed if the run id is "
                                      "ambiguous across organisms under --root")
    p_factory_clone.add_argument("--root", default=_FACTORY_RUNS_ROOT_DEFAULT,
                                 help=f"runs root directory (default: {_FACTORY_RUNS_ROOT_DEFAULT!r})")
    p_factory_clone.add_argument("--corpus-root", default=None,
                                 help=f"corpora root directory (default: {_FACTORY_CORPORA_ROOT_DEFAULT!r})")
    p_factory_clone.add_argument("--naming-seed", default=None,
                                 help="seed the child's cosmetic display name deterministically")
    p_factory_clone.add_argument("--export-predictions-max", type=int, default=3, metavar="N",
                                 help="clinic (viewer/) prediction export: up to N validation episodes "
                                      "from the promoted checkpoint (default: 3)")
    p_factory_clone.add_argument("--no-export-predictions", action="store_true",
                                 help="skip the clinic prediction export")
    p_factory_clone.set_defaults(func=cmd_factory_clone)

    p_factory_compare = factory_sub.add_parser(
        "compare", help="pair each candidate's validation evidence against a baseline run (MF-C1)"
    )
    p_factory_compare.add_argument("runs", nargs="+", metavar="run",
                                   help="baseline run id, followed by one or more candidate run ids")
    p_factory_compare.add_argument("--organism", default=None,
                                   help="organism shared by every named run, only needed if a run id "
                                        "is ambiguous across organisms under --root")
    p_factory_compare.add_argument("--root", default=_FACTORY_RUNS_ROOT_DEFAULT,
                                   help=f"runs root directory (default: {_FACTORY_RUNS_ROOT_DEFAULT!r})")
    p_factory_compare.set_defaults(func=cmd_factory_compare)

    p_factory_promote = factory_sub.add_parser(
        "promote", help="gate a run against evaluate_promotion and, if every gate passes, record it "
                        "as a champion-population member of one registry slot"
    )
    p_factory_promote.add_argument("run", help="run id to promote")
    p_factory_promote.add_argument("--family", default=None,
                                   help="registry slot family (default: the run's organism)")
    p_factory_promote.add_argument("--tier", default=None,
                                   help="registry slot budget tier, e.g. fast/scale "
                                        "(default: this run's recorded budget_report.json tier)")
    p_factory_promote.add_argument("--objective", default=None,
                                   help="registry slot objective (default: the run's declared "
                                        "evaluation.selection_metric)")
    p_factory_promote.add_argument("--hold", action="store_true",
                                   help="record an evaluated-but-not-promoted decision directly, "
                                        "skipping gate evaluation")
    p_factory_promote.add_argument("--no-as-leading", action="store_true",
                                   help="add to the champion population without becoming the slot's "
                                        "new leading champion")
    p_factory_promote.add_argument("--durable", action="store_true",
                                   help="also require the durable_test_confirmation gate: a prior "
                                        "'ccr factory test' pass recorded in metrics/test.json")
    p_factory_promote.add_argument("--minimum-practical-margin", type=float, default=0.0,
                                   help="required primary-metric improvement magnitude (default: 0.0)")
    p_factory_promote.add_argument("--reason", default=None, help="free-form audit note")
    p_factory_promote.add_argument("--organism", default=None,
                                   help="run's organism, only needed if the run id is ambiguous "
                                        "across organisms under --root")
    p_factory_promote.add_argument("--root", default=_FACTORY_RUNS_ROOT_DEFAULT,
                                   help=f"runs root directory (default: {_FACTORY_RUNS_ROOT_DEFAULT!r})")
    p_factory_promote.add_argument("--corpus-root", default=None,
                                   help=f"corpora root directory, for the data-quality/split-overlap "
                                        f"gates (default: {_FACTORY_CORPORA_ROOT_DEFAULT!r})")
    p_factory_promote.set_defaults(func=cmd_factory_promote)

    p_factory_show = factory_sub.add_parser(
        "show", help="print one run's complete effective config, contract hashes, display name, "
                     "and completion status"
    )
    p_factory_show.add_argument("run", nargs="?", default=None,
                                help="run id (default: the latest run under --root)")
    p_factory_show.add_argument("--organism", default=None,
                                help="run's organism, only needed if the run id is ambiguous "
                                     "across organisms under --root")
    p_factory_show.add_argument("--root", default=_FACTORY_RUNS_ROOT_DEFAULT,
                                help=f"runs root directory (default: {_FACTORY_RUNS_ROOT_DEFAULT!r})")
    p_factory_show.set_defaults(func=cmd_factory_show)

    p_factory_lineage = factory_sub.add_parser(
        "lineage", help="render one run's ancestor graph (configuration parents + weight donor)"
    )
    p_factory_lineage.add_argument("run", nargs="?", default=None,
                                   help="run id (default: the latest run under --root)")
    p_factory_lineage.add_argument("--organism", default=None,
                                   help="run's organism, only needed if the run id is ambiguous "
                                        "across organisms under --root")
    p_factory_lineage.add_argument("--root", default=_FACTORY_RUNS_ROOT_DEFAULT,
                                   help=f"runs root directory (default: {_FACTORY_RUNS_ROOT_DEFAULT!r})")
    p_factory_lineage.set_defaults(func=cmd_factory_lineage)

    p_factory_corpus = factory_sub.add_parser("corpus", help="build/inspect frozen nursery corpora")
    factory_corpus_sub = p_factory_corpus.add_subparsers(dest="factory_corpus_command", required=True)

    p_factory_corpus_build = factory_corpus_sub.add_parser(
        "build", help="record/reuse every declared episode, gate it, and freeze its session hashes"
    )
    p_factory_corpus_build.add_argument("spec", help="corpus spec file (.yaml/.yml/.json)")
    p_factory_corpus_build.add_argument("--root", default=_FACTORY_CORPORA_ROOT_DEFAULT,
                                        help=f"corpora root directory (default: "
                                             f"{_FACTORY_CORPORA_ROOT_DEFAULT!r})")
    p_factory_corpus_build.set_defaults(func=cmd_factory_corpus_build)

    p_factory_test = factory_sub.add_parser(
        "test", help="the sealed-test final action (MF-C5): evaluate a seed-confirmed candidate "
                     "against the sealed test split exactly once"
    )
    p_factory_test.add_argument("run", help="run id to sealed-test")
    p_factory_test.add_argument("--family", default=None,
                                help="registry slot family (default: the run's organism)")
    p_factory_test.add_argument("--tier", default=None,
                                help="registry slot budget tier (default: this run's recorded "
                                     "budget_report.json tier)")
    p_factory_test.add_argument("--objective", default=None,
                                help="registry slot objective (default: the run's declared "
                                     "evaluation.selection_metric)")
    p_factory_test.add_argument("--minimum-practical-margin", type=float, default=0.0,
                                help="required primary-metric improvement magnitude (default: 0.0)")
    p_factory_test.add_argument("--max-sealed-test-uses", type=int, default=3,
                                help="sealed-test-use budget cap for this run (default: 3)")
    p_factory_test.add_argument("--reason", default=None, help="free-form audit note")
    p_factory_test.add_argument("--organism", default=None,
                                help="run's organism, only needed if the run id is ambiguous "
                                     "across organisms under --root")
    p_factory_test.add_argument("--root", default=_FACTORY_RUNS_ROOT_DEFAULT,
                                help=f"runs root directory (default: {_FACTORY_RUNS_ROOT_DEFAULT!r})")
    p_factory_test.add_argument("--corpus-root", default=None,
                                help=f"corpora root directory (default: {_FACTORY_CORPORA_ROOT_DEFAULT!r})")
    p_factory_test.set_defaults(func=cmd_factory_test)

    _add_observability_args_everywhere(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    configure_logging(
        getattr(args, "log_level", None),
        log_file=getattr(args, "log_file", None),
        log_format=getattr(args, "log_format", "human"),
    )

    name = _run_name(args)
    with start_run(
        name,
        trace_dir=getattr(args, "trace_dir", None),
        run_id=getattr(args, "run_id", None),
        config=_trace_config(args),
        enabled=_should_trace(args, name),
    ):
        args.func(args)


if __name__ == "__main__":
    main()
