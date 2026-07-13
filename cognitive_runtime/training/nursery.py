"""Nursery scenario suite (issue #62): scripted micro-scenarios that each
isolate one worldly regularity -- ego-motion, view rotation, motion
onset/offset, object permanence, day/night, approach -- generate clean
recorded sessions, and benchmark multi-horizon world-model prediction
(t+1, t+10, t+100 by default) on held-out seeds against copy-last-frame and
mean-frame baselines.

Stage zero below the survival curriculum (issue #43): before the agent
learns to survive, the world model needs to learn that the world is
*lawful*. Generalizes issue #39's ``walk_forward`` ego-motion canary
(``training.ego_motion_canary``) into a suite -- this module reuses that
canary's recording shape, its scenario-agnostic
``evaluate_ego_motion_holdout``/``train_horizon_consistency`` helpers, and
``training.visual_representation``'s encoder/decoder/next-latent predictor,
rather than reinventing the multi-horizon pixel-prediction harness per
scenario.

Every scenario is recorded through ``MinecraftSurvivalBox``'s simulated
backend with ``curriculum=f"nursery/{scenario_name}"`` in session metadata,
so ``dashboard``/``review``/the statistical evaluation harness (issue #44)
group nursery runs the same way they already group curriculum-preset runs
-- no new grouping mechanism needed.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.neural.pixel_stream_encoder import pixels_to_chw
from cognitive_runtime.policies.constant_action import ConstantActionPolicy
from cognitive_runtime.policies.null_policy import NullPolicy
from cognitive_runtime.policies.scripted_sequence import ScriptedSequencePolicy
from cognitive_runtime.programs.minecraft.adapter import BACKENDS, MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import list_episodes
from cognitive_runtime.training.datasets import (
    build_pixel_sequence_dataset,
    load_episode_pixel_frames,
)
from cognitive_runtime.training.ego_motion_canary import (
    evaluate_ego_motion_holdout,
    train_horizon_consistency,
)
from cognitive_runtime.training.entity_persistence import (
    EntityPersistenceTrainingConfig,
    build_entity_persistence_dataset,
    train_entity_persistence_model,
)
from cognitive_runtime.training.prediction_export import export_session_predictions
from cognitive_runtime.training.visual_representation import (
    VisualPretrainingConfig,
    VisualRepresentationModel,
    reconstruction_target,
    save_pixel_encoder_pretraining_checkpoint,
    train_pixel_encoder_pretraining,
)

MOVE_FORWARD = Action("MOVE_FORWARD")
LOOK_LEFT = Action("LOOK_LEFT")
MOVE_LEFT = Action("MOVE_LEFT")

#: Fixed x-offset (agent-relative) where the occluding wall sits for
#: ``object_permanence`` -- every scripted mob offset used by that scenario
#: must clear it so the wall sits strictly between the agent and the mob
#: during the occluded phase (see ``programs.minecraft.world
#: .SurvivalWorld._has_line_of_sight``, and the same technique proven in
#: ``tests/test_entity_persistence_phase_d.py``).
_WALL_OFFSET = 3


@dataclass
class NurseryConfig:
    train_seeds: Sequence[int] = (0, 1, 2, 3)
    holdout_seeds: Sequence[int] = (1000, 1001)
    episode_ticks: int = 400
    world_size: int = 48
    backend: str = "simulated"
    realtime: bool = False
    horizons: Sequence[int] = (1, 10, 100)
    latent_width: int = 32
    hidden_dim: int = 64
    reconstruction_size: int = 16
    epochs: int = 15
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 0
    max_train_samples: Optional[int] = None
    ssim_window: int = 3
    #: Fine-tuning epochs for the horizon-consistency loss (0 skips it and
    #: evaluates the raw single-step predictor rolled out iteratively; see
    #: ``ego_motion_canary.train_horizon_consistency``).
    consistency_epochs: int = 15
    consistency_lr: float = 1e-3
    #: ``object_permanence`` only: training epochs for the entity-persistence
    #: model that distinguishes "trained on occlusion/reappearance" from the
    #: forget-immediately baseline (issue #27).
    entity_persistence_epochs: int = 30
    #: Refuse to train on recordings that don't contain the regularity the
    #: scenario exists to capture (each scenario declares its own
    #: expectations -- see ``NurseryScenario.min_blocks_per_tick`` /
    #: ``min_unique_frame_fraction``).  The first real walk_forward run
    #: recorded an agent that was stuck against an obstacle for ~95% of its
    #: frames; this gate fails such sessions before any training happens.
    data_quality_gate: bool = True
    #: Write ``predictions_<episode>.json`` (viewer "model" source) for every
    #: recorded session after training.  The nursery checkpoint persists only
    #: the pixel encoder, so predicted frames are unrecoverable later unless
    #: exported now (or the full model is saved --
    #: ``training.prediction_export.save_full_visual_model``).
    export_predictions: bool = True


@dataclass
class ScenarioRecording:
    """What one seed's episode needs, as produced by a scenario's ``build``."""

    policy: Any
    program_config_extra: Dict[str, Any] = field(default_factory=dict)
    #: Optional one-shot world-scripting hook, run on the constructed
    #: ``MinecraftSurvivalBox`` before the episode plays -- for scenarios
    #: that need scripted entities/terrain beyond what a policy can express
    #: (``object_permanence``, ``approach_entity``).
    scene_setup: Optional[Callable[[MinecraftSurvivalBox], None]] = None
    #: Overrides ``NurseryConfig.episode_ticks`` when a scenario needs a
    #: specific length (e.g. an occlusion cycle with fixed phase lengths).
    episode_ticks: Optional[int] = None


@dataclass
class NurseryScenario:
    name: str
    description: str
    build: Callable[[int, NurseryConfig], ScenarioRecording]
    #: Whether this scenario also reports the entity-persistence metric
    #: (only ``object_permanence`` today).
    entity_persistence_metric: bool = False
    #: Data-quality expectations for the gate (0.0 = no expectation).  Only
    #: movement scenarios expect displacement; the top-down render doesn't
    #: rotate with yaw or shade with daylight, so ``turn_in_place`` /
    #: ``day_night`` legitimately record near-static pixels and declare
    #: nothing.  Thresholds sit well below a healthy simulated recording
    #: (a walking agent covers ~0.09 blocks/tick with ~12% unique frames)
    #: but well above the pathological runs they exist to catch (the stuck
    #: remote agent: 0.009 blocks/tick, ~2% unique frames).
    min_blocks_per_tick: float = 0.0
    min_unique_frame_fraction: float = 0.0


REMOTE_TURN_IN_PLACE_MIN_UNIQUE_FRAME_FRACTION = 0.05


@dataclass
class NurseryScenarioReport:
    scenario: str
    config: NurseryConfig
    train_sessions: List[str] = field(default_factory=list)
    holdout_sessions: List[str] = field(default_factory=list)
    pretraining_stats: Dict[str, Any] = field(default_factory=dict)
    consistency_stats: Dict[str, List[float]] = field(default_factory=dict)
    #: Per horizon: model/copy-last/mean-frame PSNR + SSIM (see
    #: ``ego_motion_canary.evaluate_ego_motion_holdout``).
    horizon_metrics: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    #: ``object_permanence`` only: entity-persistence training stats,
    #: including ``beats_forget_baseline`` -- the metric that distinguishes a
    #: model trained on occlusion/reappearance from one without (issue #27).
    entity_persistence_stats: Optional[Dict[str, Any]] = None
    #: One rendered dream strip (predicted vs. actual frame per horizon) per
    #: held-out ``session_dir/episode_id``.
    dream_strips: Dict[str, str] = field(default_factory=dict)
    #: ``{f"{session_dir}/{episode_id}": predictions_<episode>.json path}``
    #: written after training (``NurseryConfig.export_predictions``) for the
    #: pixel viewer's "model" source.
    prediction_files: Dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- scenario builders


def _walk_forward(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    return ScenarioRecording(policy=ConstantActionPolicy(MOVE_FORWARD, seed=seed))


def _turn_in_place(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    return ScenarioRecording(policy=ConstantActionPolicy(LOOK_LEFT, seed=seed))


def _strafe_and_stop(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    phase = max(1, cfg.episode_ticks // 8)
    policy = ScriptedSequencePolicy([(MOVE_LEFT, phase), (NULL_ACTION, phase)])
    return ScenarioRecording(policy=policy)


def _day_night(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    day_length = max(cfg.episode_ticks * 2, 40)
    return ScenarioRecording(
        policy=ConstantActionPolicy(NULL_ACTION),
        program_config_extra={"day_length": day_length, "start_time": 0, "max_mobs": 0},
    )


def _approach_entity(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    distance = 6 + (seed % 6)

    def scene_setup(program: MinecraftSurvivalBox) -> None:
        world = program._backend.world
        world.reset(0)
        ax, az = int(world.x), int(world.z)
        _clear_terrain(world, ax, az, radius=distance + 4)
        _freeze_mobs(world, [{"id": 1, "x": ax, "z": az + distance, "hp": 10, "cooldown": 0}])
        world.reset = lambda seed: None  # type: ignore[method-assign]

    return ScenarioRecording(
        policy=ConstantActionPolicy(MOVE_FORWARD),
        program_config_extra={"max_mobs": 0},
        scene_setup=scene_setup,
    )


def _object_permanence(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    phase_ticks = max(5, cfg.episode_ticks // 3)
    offset = float(_WALL_OFFSET + 2 + (seed % 8))

    def scene_setup(program: MinecraftSurvivalBox) -> None:
        world = program._backend.world
        world.reset(0)
        ax, az = int(world.x), int(world.z)
        _clear_terrain(world, ax, az, radius=int(offset) + 3)
        world.terrain[ax + _WALL_OFFSET][az] = "stone"
        path = [(ax + offset, az + dz) for dz in _occlusion_dz_sequence(offset, phase_ticks)]
        _install_scripted_mob_path(world, path)
        world.reset = lambda seed: None  # type: ignore[method-assign]

    return ScenarioRecording(
        policy=NullPolicy(),
        program_config_extra={"max_mobs": 0},
        scene_setup=scene_setup,
        episode_ticks=phase_ticks * 3 + 5,
    )


NURSERY_SCENARIOS: Dict[str, NurseryScenario] = {
    "walk_forward": NurseryScenario(
        "walk_forward",
        "constant MOVE_FORWARD over varied terrain seeds -- ego-motion/optical-flow "
        "regularities (subsumes issue #39's canary).",
        _walk_forward,
        min_blocks_per_tick=0.02,
        min_unique_frame_fraction=0.05,
    ),
    "turn_in_place": NurseryScenario(
        "turn_in_place",
        "constant turn -- view rotation consistency.",
        _turn_in_place,
    ),
    "strafe_and_stop": NurseryScenario(
        "strafe_and_stop",
        "alternating movement/stillness -- motion onset/offset dynamics.",
        _strafe_and_stop,
        min_blocks_per_tick=0.01,
        min_unique_frame_fraction=0.05,
    ),
    "object_permanence": NurseryScenario(
        "object_permanence",
        "a mob passes behind an occluder -- predicted latent should retain the "
        "hidden object (issue #27).",
        _object_permanence,
        entity_persistence_metric=True,
    ),
    "day_night": NurseryScenario(
        "day_night",
        "stand still through light transitions -- slow global dynamics.",
        _day_night,
    ),
    "approach_entity": NurseryScenario(
        "approach_entity",
        "scripted approach to a passive entity -- scale change with distance.",
        _approach_entity,
        min_blocks_per_tick=0.02,
    ),
}


# --------------------------------------------------------------------------- data-quality gate


@dataclass
class EpisodeRecordingQuality:
    """What the gate measures from one recorded episode's stream log."""

    session_dir: str
    episode_id: str
    n_frames: int
    unique_frames: int
    net_displacement: float
    duration_ticks: int

    @property
    def unique_frame_fraction(self) -> float:
        return self.unique_frames / self.n_frames if self.n_frames else 0.0

    @property
    def blocks_per_tick(self) -> float:
        return self.net_displacement / self.duration_ticks if self.duration_ticks else 0.0


def measure_recording_quality(session_dir: str, episode_id: str) -> EpisodeRecordingQuality:
    """Scan one episode's stream log for the gate's signals: unique pixel
    frames (via content-hash ``frame_ref``) and net x/z displacement."""

    first_pos: Optional[Tuple[float, float]] = None
    last_pos: Optional[Tuple[float, float]] = None
    n_frames = 0
    frame_refs: set = set()
    streams_path = os.path.join(session_dir, f"{episode_id}.streams.jsonl")
    with open(streams_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            stream_id = record.get("stream_id")
            if stream_id == "vision.frame.pixels":
                n_frames += 1
                ref = record.get("frame_ref") or record.get("hash")
                if ref:
                    frame_refs.add(ref)
            elif stream_id == "spatial.position":
                payload = record.get("payload") or {}
                pos = (float(payload.get("x", 0.0)), float(payload.get("z", 0.0)))
                if first_pos is None:
                    first_pos = pos
                last_pos = pos

    displacement = (
        math.hypot(last_pos[0] - first_pos[0], last_pos[1] - first_pos[1])
        if first_pos is not None and last_pos is not None
        else 0.0
    )
    duration_ticks = 0
    summary_path = os.path.join(session_dir, f"{episode_id}.summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            duration_ticks = int(json.load(fh).get("duration_ticks", 0))
    return EpisodeRecordingQuality(
        session_dir=session_dir,
        episode_id=episode_id,
        n_frames=n_frames,
        unique_frames=len(frame_refs),
        net_displacement=displacement,
        duration_ticks=duration_ticks,
    )


def _session_backend(session_dir: str) -> str:
    path = os.path.join(session_dir, "session.json")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        metadata = json.load(fh)
    tags = metadata.get("program_tags") or []
    if "remote" in tags:
        return "remote"
    if "simulated" in tags:
        return "simulated"
    return ""


def _min_unique_frame_fraction(scenario: NurseryScenario, session_dir: str) -> float:
    if scenario.name == "turn_in_place" and _session_backend(session_dir) == "remote":
        return max(
            scenario.min_unique_frame_fraction,
            REMOTE_TURN_IN_PLACE_MIN_UNIQUE_FRAME_FRACTION,
        )
    return scenario.min_unique_frame_fraction


def validate_nursery_recordings(
    session_dirs: Sequence[str], scenario: NurseryScenario
) -> List[str]:
    """Check every recorded episode against the scenario's data-quality
    expectations; returns human-readable issue strings (empty = healthy).

    Exists because of the first real ``walk_forward`` run: recorded against
    the remote backend's persistent world, the agent was stuck against an
    obstacle with ~2% unique frames and near-zero displacement -- data with
    none of the ego-motion regularity the scenario exists to capture, which
    no amount of training can fix.
    """
    issues: List[str] = []
    for session_dir in session_dirs:
        for episode_id in list_episodes(session_dir):
            quality = measure_recording_quality(session_dir, episode_id)
            where = f"{session_dir}/{episode_id}"
            if quality.n_frames == 0:
                issues.append(f"{where}: no pixel frames recorded (record_frames off?)")
                continue
            min_unique = _min_unique_frame_fraction(scenario, session_dir)
            if (
                min_unique > 0.0
                and quality.unique_frame_fraction < min_unique
            ):
                issues.append(
                    f"{where}: only {quality.unique_frames}/{quality.n_frames} unique pixel "
                    f"frames ({quality.unique_frame_fraction:.1%} < "
                    f"{min_unique:.1%}) -- a near-static view has "
                    f"no {scenario.name!r} signal to learn"
                )
            if (
                scenario.min_blocks_per_tick > 0.0
                and quality.duration_ticks > 0
                and quality.blocks_per_tick < scenario.min_blocks_per_tick
            ):
                issues.append(
                    f"{where}: net displacement {quality.net_displacement:.2f} blocks over "
                    f"{quality.duration_ticks} ticks ({quality.blocks_per_tick:.4f}/tick < "
                    f"{scenario.min_blocks_per_tick}/tick) -- the agent barely moved "
                    f"(stuck against an obstacle?)"
                )
    return issues


# --------------------------------------------------------------------------- recording


def _clear_terrain(world: Any, cx: int, cz: int, radius: int) -> None:
    lo_x, hi_x = max(0, cx - radius), min(world.size - 1, cx + radius)
    lo_z, hi_z = max(0, cz - radius), min(world.size - 1, cz + radius)
    for x in range(lo_x, hi_x + 1):
        for z in range(lo_z, hi_z + 1):
            world.terrain[x][z] = "grass"


def _install_scripted_mob_path(world: Any, path: List[Tuple[float, float]]) -> None:
    """Replace one world's mob AI with a scripted per-tick position list --
    ``path[i]`` is the mob's ``(x, z)`` at ``world.tick == i + 1``; once
    exhausted, the mob is removed. Same technique
    ``tests/test_entity_persistence_phase_d.py`` proved for deterministic
    occlusion/reappearance sequences."""

    def scripted_update_mobs(self: Any, events: List[str]) -> None:
        idx = self.tick - 1
        if idx < len(path):
            x, z = path[idx]
            if not self.mobs:
                self._mob_serial += 1
                self.mobs = [{"id": self._mob_serial, "x": x, "z": z, "hp": 10, "cooldown": 0}]
            else:
                self.mobs[0]["x"] = x
                self.mobs[0]["z"] = z
        else:
            self.mobs = []

    world._update_mobs = scripted_update_mobs.__get__(world)


def _freeze_mobs(world: Any, mobs: List[Dict[str, Any]]) -> None:
    """Place ``mobs`` once and disable further mob AI updates -- a
    stationary entity for scenarios (``approach_entity``) that don't need
    the mob itself to move."""

    world.mobs = list(mobs)

    def noop_update_mobs(self: Any, events: List[str]) -> None:
        pass

    world._update_mobs = noop_update_mobs.__get__(world)


def _occlusion_dz_sequence(offset: float, phase_ticks: int) -> List[float]:
    """Visible off-axis, occluded directly behind the wall, visible again
    off-axis on the far side -- three phases, held ``phase_ticks`` each."""
    return [offset] * phase_ticks + [0.0] * phase_ticks + [-offset] * phase_ticks


def _record_scenario_episode(
    record_dir: str, session_id: str, seed: int, scenario: NurseryScenario, cfg: NurseryConfig
) -> str:
    recording = scenario.build(seed, cfg)
    episode_ticks = recording.episode_ticks or cfg.episode_ticks
    program_config: Dict[str, Any] = {"episode_ticks": episode_ticks, "world_size": cfg.world_size}
    program_config.update(recording.program_config_extra)
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=seed,
        max_ticks_per_episode=episode_ticks,
        record_dir=record_dir,
        session_id=session_id,
        program_config=program_config,
        realtime=cfg.realtime,
        record_frames=True,
        curriculum=f"nursery/{scenario.name}",
    )
    program = MinecraftSurvivalBox(config=program_config, backend=cfg.backend)
    if recording.scene_setup is not None and cfg.backend == "simulated":
        recording.scene_setup(program)
    try:
        CognitiveRuntime(program=program, policy=recording.policy, config=runtime_config).run()
    finally:
        program.close()
    return os.path.join(record_dir, session_id)


# --------------------------------------------------------------------------- benchmark harness


def run_nursery_scenario(
    record_dir: str,
    scenario_name: str,
    config: Optional[NurseryConfig] = None,
) -> Tuple[VisualRepresentationModel, NurseryScenarioReport]:
    """Record train/holdout episodes for one nursery scenario, pretrain a
    pixel encoder+decoder+next-latent predictor on the train seeds only,
    then evaluate multi-horizon next-frame prediction on held-out seeds
    against copy-last-frame and mean-frame baselines. ``object_permanence``
    additionally reports an entity-persistence metric."""

    if scenario_name not in NURSERY_SCENARIOS:
        raise ValueError(
            f"unknown nursery scenario {scenario_name!r}; choices: {sorted(NURSERY_SCENARIOS)}"
        )
    scenario = NURSERY_SCENARIOS[scenario_name]
    cfg = config or NurseryConfig()
    if cfg.backend not in BACKENDS:
        raise ValueError(f"unknown nursery backend {cfg.backend!r}; choices: {sorted(BACKENDS)}")
    if not cfg.horizons:
        raise ValueError("horizons must be non-empty")
    if any(h <= 0 for h in cfg.horizons):
        raise ValueError(f"horizons must be positive tick offsets, got {cfg.horizons!r}")
    if set(cfg.train_seeds) & set(cfg.holdout_seeds):
        raise ValueError("train_seeds and holdout_seeds must not overlap")

    train_sessions = [
        _record_scenario_episode(record_dir, f"nursery-{scenario_name}-train-{seed}", seed, scenario, cfg)
        for seed in cfg.train_seeds
    ]
    holdout_sessions = [
        _record_scenario_episode(record_dir, f"nursery-{scenario_name}-holdout-{seed}", seed, scenario, cfg)
        for seed in cfg.holdout_seeds
    ]

    if cfg.data_quality_gate:
        issues = validate_nursery_recordings(train_sessions + holdout_sessions, scenario)
        if issues:
            hint = (
                " Hint: the remote backend plays on the server's persistent world -- "
                "seeds do not vary terrain, sessions inherit the previous session's "
                "agent position, and a stuck agent records a static view. Re-record "
                "on the simulated backend, or reposition the agent per session. "
                "(data_quality_gate=False skips this check.)"
                if cfg.backend != "simulated"
                else " (data_quality_gate=False skips this check.)"
            )
            raise ValueError(
                f"nursery scenario {scenario_name!r}: recorded data fails the quality "
                "gate:\n  - " + "\n  - ".join(issues) + hint
            )

    train_dataset = build_pixel_sequence_dataset(train_sessions, max_samples=cfg.max_train_samples)
    if len(train_dataset) == 0:
        raise ValueError(
            f"nursery scenario {scenario_name!r}: no adjacent pixel pairs in the training "
            "sessions (episode_ticks too small?)"
        )

    visual_config = VisualPretrainingConfig(
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        latent_width=cfg.latent_width,
        hidden_dim=cfg.hidden_dim,
        reconstruction_size=cfg.reconstruction_size,
    )
    model, pretraining_stats = train_pixel_encoder_pretraining(train_dataset, visual_config)

    consistency_stats: Dict[str, List[float]] = {}
    if cfg.consistency_epochs > 0:
        consistency_stats = train_horizon_consistency(
            model,
            train_sessions,
            cfg.horizons,
            epochs=cfg.consistency_epochs,
            lr=cfg.consistency_lr,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
        )

    max_horizon = max(cfg.horizons)
    for session_dir in holdout_sessions:
        for episode_id in list_episodes(session_dir):
            if len(load_episode_pixel_frames(session_dir, episode_id)) <= max_horizon:
                raise ValueError(
                    f"{session_dir}/{episode_id} is too short for the largest horizon "
                    f"({max_horizon}); increase episode_ticks"
                )

    horizon_metrics = evaluate_ego_motion_holdout(
        model, holdout_sessions, cfg.horizons, ssim_window=cfg.ssim_window
    )

    entity_persistence_stats: Optional[Dict[str, Any]] = None
    if scenario.entity_persistence_metric:
        entity_persistence_stats = _run_entity_persistence_metric(
            train_sessions, holdout_sessions, cfg
        )

    dream_strips: Dict[str, str] = {}
    for session_dir in holdout_sessions:
        for episode_id in list_episodes(session_dir):
            dream_strips[f"{session_dir}/{episode_id}"] = render_dream_strip(
                model, session_dir, episode_id, cfg.horizons
            )

    # The checkpoint persists only the encoder, so predicted frames are
    # unrecoverable once this process exits -- export them now for the pixel
    # viewer's "model" source (viewer/README.md).
    prediction_files: Dict[str, str] = {}
    if cfg.export_predictions:
        prediction_files = export_session_predictions(
            model, train_sessions + holdout_sessions, cfg.horizons
        )

    return model, NurseryScenarioReport(
        scenario=scenario_name,
        config=cfg,
        train_sessions=train_sessions,
        holdout_sessions=holdout_sessions,
        pretraining_stats=pretraining_stats,
        consistency_stats=consistency_stats,
        horizon_metrics=horizon_metrics,
        entity_persistence_stats=entity_persistence_stats,
        dream_strips=dream_strips,
        prediction_files=prediction_files,
    )


def run_nursery_suite(
    record_dir: str,
    scenario_names: Optional[Sequence[str]] = None,
    config: Optional[NurseryConfig] = None,
) -> Dict[str, NurseryScenarioReport]:
    """``nursery run all``: run every named scenario (default: every
    registered scenario) unattended, returning one report per scenario."""

    names = list(scenario_names) if scenario_names is not None else sorted(NURSERY_SCENARIOS)
    reports: Dict[str, NurseryScenarioReport] = {}
    for name in names:
        _model, report = run_nursery_scenario(record_dir, name, config)
        reports[name] = report
    return reports


def _run_entity_persistence_metric(
    train_sessions: Sequence[str], holdout_sessions: Sequence[str], cfg: NurseryConfig
) -> Dict[str, Any]:
    """Train an entity-persistence model on the recorded occlusion sessions
    and report whether it beats the "forget immediately" baseline --
    the metric that distinguishes a model with entity-persistence training
    (issue #27) from one without."""

    dataset = build_entity_persistence_dataset(list(train_sessions) + list(holdout_sessions))
    if len(dataset) == 0:
        return {
            "samples": 0,
            "note": "no occlusion/reappearance events recorded; check the wall/mob scene setup",
        }
    ep_config = EntityPersistenceTrainingConfig(epochs=cfg.entity_persistence_epochs, seed=cfg.seed)
    _model, stats = train_entity_persistence_model(dataset, ep_config)
    return stats


# --------------------------------------------------------------------------- dream strips


_ASCII_RAMP = " .:-=+*#%@"


def render_dream_strip(
    model: VisualRepresentationModel,
    session_dir: str,
    episode_id: str,
    horizons: Sequence[int],
    *,
    start_tick: int = 0,
    thumb_size: Tuple[int, int] = (6, 24),
) -> str:
    """Render an ASCII "dream strip": predicted vs. actual frame at each
    horizon, rolled out from ``start_tick``. No image-rendering dependency
    is available in this project, so luminance is quantized onto
    ``_ASCII_RAMP`` -- coarse, but enough to see the model's rollout track
    (or fail to track) the real episode."""

    frames = load_episode_pixel_frames(session_dir, episode_id)
    horizons_sorted = sorted(set(int(h) for h in horizons))
    max_horizon = horizons_sorted[-1]
    if len(frames) <= max_horizon:
        raise ValueError(
            f"{session_dir}/{episode_id} has {len(frames)} frames, too short for "
            f"horizon {max_horizon}"
        )
    if start_tick + max_horizon >= len(frames):
        start_tick = max(0, len(frames) - max_horizon - 1)

    pixel_tensors = torch.stack([pixels_to_chw(f) for f in frames])
    targets = reconstruction_target(pixel_tensors, model.reconstruction_shape)

    was_training = model.training
    model.eval()
    lines = [f"dream strip: {session_dir}/{episode_id} (start tick {start_tick})"]
    with torch.no_grad():
        rolled = model.encoder(pixel_tensors[start_tick : start_tick + 1])
        for step in range(1, max_horizon + 1):
            rolled = model.next_predictor(rolled)
            if step not in horizons_sorted:
                continue
            predicted = model.decoder(rolled).squeeze(0)
            actual = targets[start_tick + step]
            lines.append(f"  t+{step}: predicted | actual")
            for p_row, a_row in zip(
                _ascii_thumbnail(predicted, thumb_size), _ascii_thumbnail(actual, thumb_size)
            ):
                lines.append(f"    {p_row} | {a_row}")
    if was_training:
        model.train()
    return "\n".join(lines)


def _ascii_thumbnail(frame: torch.Tensor, size: Tuple[int, int]) -> List[str]:
    """``frame``: ``Tensor[C, H, W]`` in ``[0, 1]`` -> list of ASCII rows,
    one row per thumbnail pixel row."""
    th, tw = size
    gray = frame.mean(dim=0, keepdim=True).unsqueeze(0)  # 1, 1, H, W
    thumb = F.adaptive_avg_pool2d(gray, output_size=(th, tw))[0, 0].clamp(0.0, 1.0)
    ramp = _ASCII_RAMP
    last = len(ramp) - 1
    rows = []
    for r in range(th):
        rows.append("".join(ramp[min(last, int(thumb[r, c].item() * len(ramp)))] for c in range(tw)))
    return rows


# --------------------------------------------------------------------------- checkpointing


@dataclass
class _StubPixelDataset:
    """Just enough of ``PixelSequenceDataset`` for
    ``save_pixel_encoder_pretraining_checkpoint``'s metadata."""

    layout_hash: Optional[str]
    sources: List[str]
    pixel_shape: Tuple[int, int, int]
    representation: str

    def __len__(self) -> int:
        return len(self.sources)


def save_nursery_scenario_checkpoint(
    path: str,
    model: VisualRepresentationModel,
    report: NurseryScenarioReport,
) -> Dict[str, Any]:
    """Save the trained encoder in the unified checkpoint format, with the
    scenario's holdout metrics folded into training stats."""

    stats = dict(report.pretraining_stats)
    stats["nursery"] = {
        "scenario": report.scenario,
        "horizons": list(report.config.horizons),
        "train_sessions": report.train_sessions,
        "holdout_sessions": report.holdout_sessions,
        "horizon_metrics": report.horizon_metrics,
        "entity_persistence_stats": report.entity_persistence_stats,
    }
    dataset_stub = _StubPixelDataset(
        layout_hash=None,
        sources=report.train_sessions + report.holdout_sessions,
        pixel_shape=model.pixel_shape,
        representation=f"nursery-{report.scenario}",
    )
    return save_pixel_encoder_pretraining_checkpoint(path, model, dataset_stub, stats)
