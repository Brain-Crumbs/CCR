"""Live pathfinder nursery for action-conditioned visual world models.

The nursery is intentionally narrow now: collect first-person Minecraft
recordings from a pathfinder teacher on a live server, then train the shared
action-conditioned recurrent world model to predict future pixels over
multiple horizons.  Simulated fallback exists only for fast tests; production
nursery runs should use ``--backend remote`` with first-person viewer pixels.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import SingleActionPolicy
from cognitive_runtime.core.world_model import Prediction
from cognitive_runtime.programs.minecraft.adapter import BACKENDS, MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import iter_cognitive_ticks, list_episodes
from cognitive_runtime.training.action_world_model import (
    ActionWorldModelConfig,
    build_action_sequence_dataset,
    evaluate_action_world_model,
    horizons_ticks_to_frames,
    train_action_world_model,
)


MOVE_FORWARD = Action("MOVE_FORWARD")
LOOK_LEFT = Action("LOOK_LEFT")
LOOK_RIGHT = Action("LOOK_RIGHT")
SPRINT = Action("SPRINT")


@dataclass(frozen=True)
class PathfinderGoal:
    x: float
    y: Optional[float]
    z: float
    radius: float = 1.5

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"x": self.x, "z": self.z, "radius": self.radius}
        if self.y is not None:
            payload["y"] = self.y
        return payload


@dataclass
class NurseryConfig:
    """Configuration for one live pathfinder nursery run."""

    train_episodes: int = 6
    holdout_episodes: int = 2
    episode_ticks: int = 500
    backend: str = "remote"
    realtime: bool = True
    world_size: int = 128
    seed: int = 0
    horizons: Sequence[int] = (1, 4, 8)
    latent_width: int = 32
    hidden_dim: int = 64
    reconstruction_size: int = 16
    epochs: int = 30
    lr: float = 1e-3
    batch_size: int = 32
    warmup_frames: int = 3
    rollout_frames: int = 8
    target_radius: float = 1.5
    arena_origin_x: int = 0
    arena_origin_y: Optional[int] = None
    arena_origin_z: int = 0
    arena_spacing: int = 64
    arena_radius: int = 128
    setup_live_arena: bool = True
    require_first_person: bool = True
    min_unique_frame_fraction: float = 0.05
    min_blocks_per_tick: float = 0.005
    min_action_kinds: int = 2
    data_quality_gate: bool = True
    require_learning: bool = False


def nursery_recorded_ticks(cfg: NurseryConfig) -> int:
    """Runtime ticks to record, including target frames beyond active ticks."""
    return int(cfg.episode_ticks) + max([int(h) for h in cfg.horizons] or [0])


@dataclass
class PathfinderRecordingQuality:
    session_dir: str
    episode_id: str
    n_frames: int
    unique_frames: int
    net_displacement: float = 0.0
    max_displacement: float = 0.0
    duration_ticks: int = 0
    action_counts: Dict[str, int] = field(default_factory=dict)
    completed: Optional[bool] = None
    termination_reason: str = ""
    pixel_sources: List[str] = field(default_factory=list)
    setup_requested: bool = False
    setup_reached_start: Optional[bool] = None
    setup_distance_from_start: Optional[float] = None

    @property
    def unique_frame_fraction(self) -> float:
        return self.unique_frames / self.n_frames if self.n_frames else 0.0

    @property
    def non_null_action_counts(self) -> Dict[str, int]:
        return {k: v for k, v in self.action_counts.items() if k != NULL_ACTION.name}

    @property
    def blocks_per_tick(self) -> float:
        return self.net_displacement / self.duration_ticks if self.duration_ticks else 0.0


@dataclass
class LivePathfinderNurseryReport:
    config: NurseryConfig
    train_sessions: List[str]
    holdout_sessions: List[str]
    training_stats: Dict[str, Any]
    metrics: Dict[str, Any]
    horizon_ticks: List[int]
    horizon_frames: List[int]
    horizon_frame_mapping: Dict[str, int]
    ticks_per_frame: float
    quality: List[PathfinderRecordingQuality]
    learning_check: Dict[str, Any]


class PathfinderTeacherPolicy(SingleActionPolicy):
    """Ask the backend for a pathfinder teacher action each tick.

    The remote backend delegates to the mineflayer bridge.  Simulated tests use
    the local steering fallback, which still emits ordinary SurvivalBox actions
    so the action-world-model dataset has the same shape.
    """

    name = "pathfinder-teacher"

    def __init__(
        self,
        goal: PathfinderGoal,
        *,
        backend_provider: Optional[Callable[[], Any]] = None,
        prefer_sprint: bool = False,
    ):
        self.goal = goal
        self.backend_provider = backend_provider
        self.prefer_sprint = prefer_sprint

    def decide(
        self, state: State, memory: Memory, prediction: Optional[Prediction]
    ) -> Action:
        backend = self.backend_provider() if self.backend_provider is not None else None
        if backend is not None and hasattr(backend, "suggest_pathfinder_action"):
            try:
                return backend.suggest_pathfinder_action(self.goal.as_dict())
            except Exception:
                # The quality gate will catch compromised live recordings; keep
                # the episode moving so a failed teacher can be diagnosed from
                # the logs instead of disappearing before recording starts.
                pass
        return self._steer_from_state(state)

    def _steer_from_state(self, state: State) -> Action:
        data = state.observation.data if state and state.observation else {}
        position = data.get("spatial.position") if isinstance(data, dict) else None
        rotation = data.get("spatial.rotation") if isinstance(data, dict) else None
        x = float(
            (position or {}).get("x", data.get("x", 0.0))
            if isinstance(data, dict)
            else 0.0
        )
        z = float(
            (position or {}).get("z", data.get("z", 0.0))
            if isinstance(data, dict)
            else 0.0
        )
        yaw = float(
            (rotation or {}).get("yaw", data.get("yaw", 0.0))
            if isinstance(data, dict)
            else 0.0
        )
        dx = self.goal.x - x
        dz = self.goal.z - z
        if math.hypot(dx, dz) <= self.goal.radius:
            return NULL_ACTION
        # SurvivalBox yaw 0 faces negative Z; match the bridge/action math.
        desired = math.degrees(math.atan2(-dx, dz)) % 360.0
        delta = (desired - yaw + 180.0) % 360.0 - 180.0
        if abs(delta) > 18.0:
            return LOOK_LEFT if delta < 0.0 else LOOK_RIGHT
        return SPRINT if self.prefer_sprint else MOVE_FORWARD


def _episode_goal(cfg: NurseryConfig, index: int) -> Tuple[Tuple[float, Optional[float], float], PathfinderGoal]:
    col = index % 4
    row = index // 4
    start_x = cfg.arena_origin_x + col * cfg.arena_spacing
    start_z = cfg.arena_origin_z + row * cfg.arena_spacing
    # Vary target offsets so the teacher must turn and move, not memorize a
    # single constant action trace.
    angle = (index * 137.5 + cfg.seed * 19.0) * math.pi / 180.0
    distance = max(4.0, cfg.arena_radius - 2 )
    target_x = start_x + math.cos(angle) * distance
    target_z = start_z + math.sin(angle) * distance
    return (
        (float(start_x), float(cfg.arena_origin_y) if cfg.arena_origin_y is not None else None, float(start_z)),
        PathfinderGoal(target_x, float(cfg.arena_origin_y) if cfg.arena_origin_y is not None else None, target_z, cfg.target_radius),
    )


def _program_config(cfg: NurseryConfig, seed: int, index: int, goal: PathfinderGoal) -> Dict[str, Any]:
    start, _ = _episode_goal(cfg, index)
    tail_ticks = max([int(h) for h in cfg.horizons] or [0])
    nursery = {
        "mode": "pathfinder",
        "seed": seed,
        "setup": cfg.setup_live_arena,
        "arena_radius": cfg.arena_radius,
        "active_episode_ticks": int(cfg.episode_ticks),
        "horizon_tail_ticks": tail_ticks,
        "playback_horizons": [int(h) for h in cfg.horizons],
        "start": {"x": start[0], "z": start[2]},
        "target": goal.as_dict(),
    }
    if start[1] is not None:
        nursery["start"]["y"] = start[1]
    return {
        "episode_ticks": int(cfg.episode_ticks) + tail_ticks,
        "world_size": cfg.world_size,
        "difficulty": 0.0,
        "max_mobs": 0,
        "pixel_source": "viewer",
        "nursery": nursery,
    }


def _assert_session_targets_unused(record_dir: str, session_ids: Sequence[str]) -> None:
    existing = [
        os.path.join(record_dir, session_id)
        for session_id in session_ids
        if os.path.exists(os.path.join(record_dir, session_id))
    ]
    if existing:
        raise FileExistsError(
            "pathfinder nursery refuses to overwrite or reuse existing session "
            "directories; remove them or choose a fresh --record-dir:\n  - "
            + "\n  - ".join(existing)
        )


def _allocate_unused_session_ids(record_dir: str, prefix: str, count: int) -> List[str]:
    session_ids: List[str] = []
    index = 0
    while len(session_ids) < count:
        session_id = f"{prefix}-{index}"
        if not os.path.exists(os.path.join(record_dir, session_id)):
            session_ids.append(session_id)
        index += 1
    return session_ids


def record_pathfinder_episode(
    record_dir: str,
    session_id: str,
    seed: int,
    index: int,
    cfg: NurseryConfig,
) -> str:
    if cfg.backend not in BACKENDS:
        raise ValueError(f"unknown nursery backend {cfg.backend!r}; choices: {sorted(BACKENDS)}")
    _assert_session_targets_unused(record_dir, [session_id])
    _start, goal = _episode_goal(cfg, index)
    program_config = _program_config(cfg, seed, index, goal)
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=seed,
        max_ticks_per_episode=nursery_recorded_ticks(cfg),
        record_dir=record_dir,
        session_id=session_id,
        program_config=program_config,
        realtime=cfg.realtime or cfg.backend == "remote",
        record_frames=True,
        curriculum="nursery/pathfinder",
        reflex_mode="off",
    )
    program = MinecraftSurvivalBox(config=program_config, backend=cfg.backend)
    policy = PathfinderTeacherPolicy(
        goal,
        backend_provider=lambda: program._backend,  # narrow recording helper; not runtime API.
        prefer_sprint=False,
    )
    try:
        CognitiveRuntime(program=program, policy=policy, config=runtime_config).run()
    finally:
        program.close()
    return os.path.join(record_dir, session_id)


def measure_pathfinder_recording(session_dir: str, episode_id: str) -> PathfinderRecordingQuality:
    frames = 0
    frame_refs: set[str] = set()
    action_counts: Dict[str, int] = {}
    first_pos: Optional[Tuple[float, float]] = None
    last_pos: Optional[Tuple[float, float]] = None
    max_displacement = 0.0
    streams_path = os.path.join(session_dir, f"{episode_id}.streams.jsonl")
    with open(streams_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("stream_id") == "vision.frame.pixels":
                frames += 1
                ref = record.get("frame_ref") or record.get("hash")
                if ref:
                    frame_refs.add(str(ref))
            elif record.get("stream_id") == "motor.command":
                payload = record.get("payload") or {}
                action = payload.get("action")
                if isinstance(action, str) and action:
                    action_counts[action] = action_counts.get(action, 0) + 1
            elif record.get("stream_id") == "spatial.position":
                payload = record.get("payload") or {}
                pos = (float(payload.get("x", 0.0)), float(payload.get("z", 0.0)))
                if first_pos is None:
                    first_pos = pos
                last_pos = pos
                if first_pos is not None:
                    max_displacement = max(
                        max_displacement,
                        math.hypot(pos[0] - first_pos[0], pos[1] - first_pos[1]),
                    )

    summary_path = os.path.join(session_dir, f"{episode_id}.summary.json")
    completed: Optional[bool] = None
    termination_reason = ""
    pixel_sources: List[str] = []
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        completed = summary.get("success")
        termination_reason = str(summary.get("termination_reason") or "")
        stats = summary.get("program_stats") or {}
        pixel_sources = [str(s) for s in stats.get("pixel_sources", [])]
        setup_requested = bool(stats.get("nursery_setup_requested", False))
        reached = stats.get("nursery_setup_reached_start")
        setup_reached_start = reached if isinstance(reached, bool) else None
        distance = stats.get("nursery_setup_distance_from_start")
        setup_distance_from_start = (
            float(distance) if isinstance(distance, (int, float)) else None
        )
        duration_ticks = int(summary.get("duration_ticks") or stats.get("final_tick") or 0)
    else:
        duration_ticks = 0
        setup_requested = False
        setup_reached_start = None
        setup_distance_from_start = None
    net_displacement = 0.0
    if first_pos is not None and last_pos is not None:
        net_displacement = math.hypot(last_pos[0] - first_pos[0], last_pos[1] - first_pos[1])

    return PathfinderRecordingQuality(
        session_dir=session_dir,
        episode_id=episode_id,
        n_frames=frames,
        unique_frames=len(frame_refs),
        net_displacement=net_displacement,
        max_displacement=max_displacement,
        duration_ticks=duration_ticks,
        action_counts=action_counts,
        completed=completed if isinstance(completed, bool) else None,
        termination_reason=termination_reason,
        pixel_sources=sorted(pixel_sources),
        setup_requested=setup_requested,
        setup_reached_start=setup_reached_start,
        setup_distance_from_start=setup_distance_from_start,
    )


def validate_pathfinder_recordings(
    session_dirs: Sequence[str], cfg: NurseryConfig
) -> Tuple[List[PathfinderRecordingQuality], List[str]]:
    qualities: List[PathfinderRecordingQuality] = []
    issues: List[str] = []
    aggregate_actions: Dict[str, int] = {}
    for session_dir in session_dirs:
        for episode_id in list_episodes(session_dir):
            quality = measure_pathfinder_recording(session_dir, episode_id)
            qualities.append(quality)
            for name, count in quality.non_null_action_counts.items():
                aggregate_actions[name] = aggregate_actions.get(name, 0) + count
            label = f"{session_dir}/{episode_id}"
            if quality.completed is False:
                issues.append(
                    f"{label}: episode ended early ({quality.termination_reason or 'unknown reason'})"
                )
            if quality.n_frames < 2:
                issues.append(f"{label}: only {quality.n_frames} pixel frames were recorded")
            if quality.unique_frame_fraction < cfg.min_unique_frame_fraction:
                issues.append(
                    f"{label}: only {quality.unique_frames}/{quality.n_frames} unique pixel "
                    f"frames ({quality.unique_frame_fraction:.1%} < "
                    f"{cfg.min_unique_frame_fraction:.1%})"
                )
            if quality.blocks_per_tick < cfg.min_blocks_per_tick:
                issues.append(
                    f"{label}: net displacement {quality.net_displacement:.2f} blocks over "
                    f"{quality.duration_ticks} ticks "
                    f"({quality.blocks_per_tick:.4f}/tick < {cfg.min_blocks_per_tick}/tick)"
                )
            if cfg.require_first_person and cfg.backend == "remote":
                if quality.pixel_sources != ["viewer"]:
                    issues.append(
                        f"{label}: pixel source {quality.pixel_sources or 'unknown'} is not "
                        "first-person viewer; refusing to train on grid fallback"
                    )
            if cfg.backend == "remote" and cfg.setup_live_arena:
                if quality.setup_requested and quality.setup_reached_start is False:
                    distance = (
                        f"{quality.setup_distance_from_start:.2f} blocks"
                        if quality.setup_distance_from_start is not None
                        else "unknown distance"
                    )
                    issues.append(
                        f"{label}: live nursery arena setup did not move the bot to "
                        f"the requested start ({distance}); op the bot or pass "
                        "--no-setup-live-arena"
                    )
    if len(aggregate_actions) < cfg.min_action_kinds:
        issues.append(
            "pathfinder nursery: teacher emitted only "
            f"{sorted(aggregate_actions)} non-null action kind(s) across all recordings; "
            "the world model needs action-diverse traces"
        )
    return qualities, issues


def _learning_check(metrics: Dict[str, Any]) -> Dict[str, Any]:
    horizons = metrics.get("tick_horizons") or metrics.get("horizons", {})
    ratios = {
        str(h): entry.get("model_over_copy_last_mse")
        for h, entry in horizons.items()
        if entry.get("model_over_copy_last_mse") is not None
    }
    beating = {
        h: ratio for h, ratio in ratios.items() if isinstance(ratio, (int, float)) and ratio < 1.0
    }
    return {
        "beats_copy_last_any_horizon": bool(beating),
        "model_over_copy_last_mse": ratios,
        "best_model_over_copy_last_mse": min(beating.values()) if beating else None,
    }


def run_live_pathfinder_nursery(
    record_dir: str,
    config: Optional[NurseryConfig] = None,
    model_config: Optional[ActionWorldModelConfig] = None,
) -> Tuple[Any, LivePathfinderNurseryReport]:
    cfg = config or NurseryConfig()
    if cfg.train_episodes <= 0:
        raise ValueError("train_episodes must be positive")
    if cfg.holdout_episodes <= 0:
        raise ValueError("holdout_episodes must be positive")
    if not cfg.horizons or any(h <= 0 for h in cfg.horizons):
        raise ValueError(f"horizons must be positive tick offsets, got {cfg.horizons!r}")

    train_ids = _allocate_unused_session_ids(
        record_dir, "nursery-pathfinder-train", cfg.train_episodes
    )
    holdout_ids = _allocate_unused_session_ids(
        record_dir, "nursery-pathfinder-holdout", cfg.holdout_episodes
    )

    train_sessions: List[str] = []
    holdout_sessions: List[str] = []
    for i in range(cfg.train_episodes):
        seed = cfg.seed + i
        train_sessions.append(
            record_pathfinder_episode(
                record_dir, train_ids[i], seed, i, cfg
            )
        )
    for i in range(cfg.holdout_episodes):
        index = cfg.train_episodes + i
        seed = cfg.seed + index
        holdout_sessions.append(
            record_pathfinder_episode(
                record_dir, holdout_ids[i], seed, index, cfg
            )
        )

    qualities, issues = validate_pathfinder_recordings(
        train_sessions + holdout_sessions, cfg
    )
    if cfg.data_quality_gate and issues:
        raise ValueError(
            "pathfinder nursery recordings fail the quality gate:\n  - "
            + "\n  - ".join(issues)
        )

    from cognitive_runtime.training.features import ACTION_KEYS

    action_keys = list(ACTION_KEYS)
    if NULL_ACTION.name not in action_keys:
        action_keys.append(NULL_ACTION.name)
    dataset = build_action_sequence_dataset(train_sessions, action_keys=action_keys)
    if len(dataset) == 0:
        raise ValueError("pathfinder nursery: no frame/action transitions in training sessions")

    model_cfg = model_config or ActionWorldModelConfig(
        latent_width=cfg.latent_width,
        hidden_dim=cfg.hidden_dim,
        reconstruction_size=cfg.reconstruction_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        warmup_frames=cfg.warmup_frames,
        rollout_frames=cfg.rollout_frames,
    )
    model, training_stats = train_action_world_model(dataset, model_cfg)
    ticks_per_frame = dataset.ticks_per_frame
    horizon_frames = horizons_ticks_to_frames(cfg.horizons, ticks_per_frame)
    unique_horizon_frames = sorted(set(horizon_frames))
    holdout_dataset = build_action_sequence_dataset(holdout_sessions, action_keys=model.action_keys)
    metrics = evaluate_action_world_model(
        model,
        holdout_dataset,
        unique_horizon_frames,
        warmup_frames=model_cfg.warmup_frames,
    )
    frame_metrics = metrics.get("horizons", {})
    metrics["tick_horizons"] = {
        str(int(tick_h)): {
            "frame_horizon": int(frame_h),
            **dict(frame_metrics.get(frame_h, frame_metrics.get(str(frame_h), {}))),
        }
        for tick_h, frame_h in zip(cfg.horizons, horizon_frames)
    }
    learning = _learning_check(metrics)
    if cfg.require_learning and not learning["beats_copy_last_any_horizon"]:
        raise ValueError(
            "pathfinder nursery: trained model did not beat copy-last on any "
            f"holdout horizon ({learning['model_over_copy_last_mse']})"
        )

    return model, LivePathfinderNurseryReport(
        config=cfg,
        train_sessions=train_sessions,
        holdout_sessions=holdout_sessions,
        training_stats=training_stats,
        metrics=metrics,
        horizon_ticks=[int(h) for h in cfg.horizons],
        horizon_frames=horizon_frames,
        horizon_frame_mapping={
            str(int(tick_h)): int(frame_h)
            for tick_h, frame_h in zip(cfg.horizons, horizon_frames)
        },
        ticks_per_frame=ticks_per_frame,
        quality=qualities,
        learning_check=learning,
    )
