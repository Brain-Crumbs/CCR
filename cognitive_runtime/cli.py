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
import os
import sys
from typing import Any, Callable, Dict

from cognitive_runtime.core.policy import Policy
from cognitive_runtime.policies import (
    HumanDemoPolicy,
    LearnedPolicy,
    NullPolicy,
    RandomPolicy,
    ScriptedSurvivalPolicy,
)
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
from cognitive_runtime.programs.minecraft.adapter import BACKENDS, MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.evaluation import comparison_table, summarize_episodes
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import NonDeterministicSessionError
from cognitive_runtime.tools.episode_viewer import view_episode
from cognitive_runtime.tools.metrics_dashboard import dashboard
from cognitive_runtime.tools.replay_runner import format_results, replay_session
from cognitive_runtime.training.datasets import build_dataset
from cognitive_runtime.training.evaluation import compare_policies
from cognitive_runtime.training.imitation import train_bc


def _program_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "episode_ticks": args.episode_ticks,
        "difficulty": args.difficulty,
        "world_size": args.world_size,
        "day_length": args.day_length,
        "start_time": args.start_time,
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
    sys.exit(f"unknown policy: {name}")


def _add_world_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0, help="base episode seed")
    parser.add_argument("--episode-ticks", type=int, default=6000,
                        help="episode length in ticks (5 min at 20 tps)")
    parser.add_argument("--difficulty", type=float, default=1.0)
    parser.add_argument("--world-size", type=int, default=64)
    parser.add_argument("--day-length", type=int, default=6000,
                        help="full day/night cycle in ticks; night is the second half")
    parser.add_argument("--start-time", type=int, default=0, help="time of day at spawn")
    parser.add_argument("--model", default=None, help="path to a trained BC model (learned policy)")
    parser.add_argument("--backend", default="simulated", choices=sorted(BACKENDS),
                        help="survival backend: the deterministic simulated world, or "
                             "a real-Minecraft client (remote; not yet implemented)")


def cmd_run(args: argparse.Namespace) -> None:
    program_config = _program_config(args)
    policy = _make_policy(args.policy, args)
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
        session_id=args.session_id,
        program_config=program_config,
    )
    runtime = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=program_config, backend=args.backend),
        policy=policy,
        config=config,
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
    if args.session_id is None:
        import time as _time
        args.session_id = f"{_time.strftime('%Y%m%d-%H%M%S')}-human-demo"
    cmd_run(args)


def cmd_evaluate(args: argparse.Namespace) -> None:
    program_config = _program_config(args)
    names = [p.strip() for p in args.policies.split(",") if p.strip()]
    factories: Dict[str, Callable[[], Policy]] = {}
    for name in names:
        factories[name] = (lambda n: (lambda: _make_policy(n, args)))(name)
    rows = compare_policies(
        program_factory=lambda: MinecraftSurvivalBox(config=program_config, backend=args.backend),
        policy_factories=factories,
        episodes=args.episodes,
        seed=args.seed,
        max_ticks=args.episode_ticks,
    )
    print(comparison_table(rows))


def cmd_train(args: argparse.Namespace) -> None:
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
                       choices=["null", "random", "scripted", "learned", "human"])
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
    p_run.add_argument("--record-dir", default="sessions")
    p_run.add_argument("--session-id", default=None)
    _add_world_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_demo = sub.add_parser("demo", help="play SurvivalBox yourself; recorded as demonstrations")
    p_demo.add_argument("--episodes", type=int, default=1)
    p_demo.add_argument("--tick-rate", type=float, default=20.0)
    p_demo.add_argument("--record-dir", default="sessions")
    p_demo.add_argument("--session-id", default=None)
    _add_world_args(p_demo)
    p_demo.set_defaults(func=cmd_demo)

    p_eval = sub.add_parser("evaluate", help="compare policies on identical episodes")
    p_eval.add_argument("--policies", default="null,random,scripted")
    p_eval.add_argument("--episodes", type=int, default=3)
    _add_world_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_train = sub.add_parser("train", help="train a behavioral-cloning policy from sessions")
    p_train.add_argument("--sessions", nargs="+", required=True,
                         help="session directories (e.g. sessions/20260101-...-scripted)")
    p_train.add_argument("--out", default="models/bc.json")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=0.5)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--history", type=int, default=8)
    p_train.add_argument("--features", choices=["latent", "handcrafted"], default="latent",
                         help="policy input: fused latent state (default) or hand featurizer")
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
