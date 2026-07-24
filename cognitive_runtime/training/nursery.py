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
import logging
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

log = logging.getLogger("ccr.training.nursery")
import torch.nn.functional as F

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.neural.pixel_stream_encoder import pixels_to_chw
from cognitive_runtime.policies.constant_action import ConstantActionPolicy
from cognitive_runtime.policies.null_policy import NullPolicy
from cognitive_runtime.policies.scripted_sequence import ScriptedSequencePolicy
from cognitive_runtime.programs.crafter.config import CrafterConfig
from cognitive_runtime.programs.minecraft.adapter import BACKENDS, MinecraftSurvivalBox
#: Re-exported for back-compat: this gate moved to ``record.quality`` (issue
#: #90) so it can run world-agnostically; existing imports of
#: ``EpisodeRecordingQuality``/``measure_recording_quality`` from here still
#: work unchanged.
from cognitive_runtime.record.quality import (  # noqa: F401
    EpisodeRecordingQuality,
    measure_recording_quality,
    validate_recordings,
)
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import list_episodes
from cognitive_runtime.training.datasets import (
    build_pixel_sequence_dataset,
    load_episode_pixel_frames,
)
from cognitive_runtime.training.action_world_model import (
    ActionWorldModelConfig,
    build_action_sequence_dataset,
    evaluate_action_world_model,
    horizons_ticks_to_frames,
    linear_probe_yaw,
    linear_probe_orientation,
    representation_collapse_diagnostics,
    load_action_world_model,
    save_action_world_model,
    train_action_world_model,
)
from cognitive_runtime.training.ego_motion_canary import (
    evaluate_ego_motion_holdout,
    evaluate_rollout_health,
    train_horizon_consistency,
)
from cognitive_runtime.training.entity_persistence import (
    EntityPersistenceTrainingConfig,
    build_entity_persistence_dataset,
    train_entity_persistence_model,
)
from cognitive_runtime.training.prediction_export import (
    export_session_predictions,
    load_full_visual_model,
    save_full_visual_model,
)
from cognitive_runtime.training.statistical_evaluation import (
    MetricComparison,
    MetricStats,
    cortex_horizon_statistics,
    compare_cortex_horizon_statistics,
)
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
    #: Which Program records the scenario (issue #90): ``"minecraft"`` looks
    #: scenario names up in ``NURSERY_SCENARIOS``; ``"crafter"`` looks them
    #: up in ``CRAFTER_SCENARIOS`` and ignores ``backend`` (Crafter has no
    #: backend choice -- the ``crafter`` package *is* the backend).
    world: str = "minecraft"
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
    #: When set ("viewer"/"grid"), the data-quality gate fails any recording
    #: whose backend-reported pixel provenance differs -- the first real runs
    #: requested the first-person viewer and silently trained on the grid
    #: fallback instead.  ``None`` accepts either, but still refuses to mix
    #: sources within one training run.
    expected_pixel_source: Optional[str] = None
    #: Organism identity (issue #88): threaded into every recorded episode's
    #: `RuntimeConfig.name`, so its session metadata, prediction exports, and
    #: the encoder checkpoint all carry it. ``None`` lets each recorded
    #: episode resolve its own generated name (cosmetic only).
    name: Optional[str] = None


@dataclass
class ScenarioRecording:
    """What one seed's episode needs, as produced by a scenario's ``build``."""

    policy: Any
    program_config_extra: Dict[str, Any] = field(default_factory=dict)
    #: Optional one-shot world-scripting hook, run on the constructed
    #: Program before the episode plays -- for scenarios that need scripted
    #: entities/terrain beyond what a policy can express
    #: (``object_permanence``, ``approach_entity``). Takes a
    #: ``MinecraftSurvivalBox`` for scenarios registered in
    #: ``NURSERY_SCENARIOS``, a ``CrafterWorld`` for ones in
    #: ``CRAFTER_SCENARIOS`` -- never both, since each scenario is only ever
    #: built for the world it's registered under.
    scene_setup: Optional[Callable[[Any], None]] = None
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
    #: movement scenarios expect displacement.  Thresholds sit well below a
    #: healthy simulated recording (a walking agent covers ~0.09 blocks/tick
    #: with ~12% unique frames) but well above the pathological runs they
    #: exist to catch (the stuck remote agent: 0.009 blocks/tick, ~2% unique
    #: frames).  The pixel render rotates with yaw (``_orient_frame_grid``),
    #: so ``turn_in_place`` produces changing pixels and declares a
    #: unique-frame floor like the movement scenarios.
    min_blocks_per_tick: float = 0.0
    min_unique_frame_fraction: float = 0.0
    #: Upper displacement bound (None = no expectation): scenarios whose
    #: premise is a stationary agent (``turn_in_place``) fail their purpose
    #: when live-server physics (knockback, water, mobs) drags the agent
    #: around -- the first real turn_in_place run drifted up to 24 blocks
    #: while only ever issuing LOOK_LEFT.
    max_blocks_per_tick: Optional[float] = None
    #: Minimum total |yaw delta| over the episode in degrees (0.0 = no
    #: expectation): ``turn_in_place`` requires at least one full revolution,
    #: otherwise there is no view-rotation regularity to learn.
    min_yaw_sweep_degrees: float = 0.0
    #: Discrete-facing equivalent of ``min_yaw_sweep_degrees`` (0 = no
    #: expectation): Crafter has no continuous view to rotate, so its
    #: ``turn`` scenario instead requires visiting this many distinct facing
    #: directions (``spatial.facing``; max 4 on a grid).
    min_unique_facings: int = 0
    #: Nursery episodes are scripted micro-scenarios; one that terminated
    #: early (the first real turn_in_place train-0 was beaten to death by
    #: mobs at tick 167/400) is not the scenario it claims to be.
    require_completed: bool = True


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
    #: ``config.horizons`` is declared in *ticks*; recorded vision may run
    #: below the tick rate (the first remote runs paced ~10 Hz against 20 Hz
    #: ticks), so evaluation converts to recorded-frame steps via the
    #: measured rate.  ``horizon_metrics``/``dream_strips`` are keyed by
    #: these frame horizons.
    horizon_frames: List[int] = field(default_factory=list)
    ticks_per_frame: float = 1.0
    #: Frozen-rollout detector (``ego_motion_canary.evaluate_rollout_health``):
    #: flags predictions that do not vary across horizons while the actual
    #: frames do -- the collapse signature of the first real turn_in_place
    #: run.
    rollout_health: Dict[str, Any] = field(default_factory=dict)


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
        min_unique_frame_fraction=0.05,
        max_blocks_per_tick=0.02,
        min_yaw_sweep_degrees=360.0,
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


# --------------------------------------------------------------------------- crafter scenario builders
#
# Crafter ports of walk_forward/turn/object_permanence/approach_entity
# (issue #90), registered in the parallel ``CRAFTER_SCENARIOS`` below rather
# than folded into ``NURSERY_SCENARIOS``: Crafter is a different Program
# (``programs.crafter.adapter.CrafterWorld``), so its scene-setup hooks take
# a ``CrafterWorld``, not a ``MinecraftSurvivalBox``.
#
# Crafter has no first-person view to rotate, so ``turn_in_place`` isn't
# ported as-is (Crafter is 2-D top-down; see docs/v2/phases
# /phase-1-nursery-world.md's "do not smuggle ego-motion back in").  Its
# discrete analogue, ``turn``, boxes the agent in with stone on all four
# sides and cycles the four directional actions -- every move is blocked,
# so only the discrete ``facing`` changes (``crafter.objects.Player._move``
# sets ``self.facing`` before checking collision, so a blocked move still
# turns the agent).
#
# Crafter's renderer draws objects over terrain regardless of what's
# underneath them (no line-of-sight occlusion), so ``object_permanence``
# isn't ported via a literal wall either -- a mob standing on a "wall" tile
# would still render on top of it.  Its real occlusion is the bounded
# egocentric view (``CrafterConfig.grid_radius``): a scripted mob walks
# out past the view radius, holds there, then walks back -- object
# permanence via genuine limited perceptual range.  Because that relies on
# view-radius geometry rather than Minecraft's ``vision.entities``/
# ``EntityTracker`` semantics, Crafter's port does not report the
# entity-persistence metric (``NurseryScenario.entity_persistence_metric``
# stays ``False``); only the recording itself is ported here.


def _crafter_env(program: Any) -> Any:
    return program._env


def _crafter_clear_terrain(world: Any, cx: int, cy: int, radius: int) -> None:
    area = world.area
    for x in range(max(0, cx - radius), min(area[0], cx + radius + 1)):
        for y in range(max(0, cy - radius), min(area[1], cy + radius + 1)):
            world[(x, y)] = "grass"


def _crafter_neutralize_wildlife(env: Any) -> None:
    """Stop every non-player creature in ``env`` from moving, disable
    Crafter's own periodic spawn/despawn balancing (``Env._balance_chunk``,
    no Minecraft-style ``max_mobs=0`` knob exists), and stop the player's own
    neglect-driven survival decay (hunger/thirst/energy depletion and the
    health regen/degen it drives -- ``Player._update_life_stats``/
    ``_degen_or_regen_health``). A short scripted micro-scenario isn't
    testing "can the agent feed itself"; over a few hundred ticks of pure
    ``MOVE_UP`` (issue #90's ``walk_forward``) neglect alone starves the
    agent to death well before that, which isn't the ego-motion/facing
    regularity these scenarios exist to capture."""
    for obj in list(env._world.objects):
        if obj is not env._player:
            obj.update = lambda: None
    env._balance_chunk = lambda chunk, objs: None
    env._player._update_life_stats = lambda: None
    env._player._degen_or_regen_health = lambda: None


def _crafter_freeze_wildlife(program: Any) -> None:
    """One-shot hazard neutralization for scenarios that also freeze the
    env entirely (``_crafter_box_in_player``, ``object_permanence``,
    ``approach_entity``): those already pin the world to its as-constructed
    state via ``freeze_reset()`` (ignoring the per-episode seed, like
    Minecraft's own scripted scenarios hardcoding ``world.reset(0)``), so a
    single neutralization pass is enough."""
    env = _crafter_env(program)
    _crafter_neutralize_wildlife(env)
    program.freeze_reset()


def _crafter_neutralize_wildlife_every_reset(program: Any) -> None:
    """Like ``_crafter_freeze_wildlife``, but re-applied after every
    ``reset(seed)`` instead of freezing the env -- for scenarios
    (``walk_forward``) that still need each seed's own generated terrain,
    just without wildlife/neglect able to kill the episode outright."""
    original_reset = program.reset

    def reset_and_neutralize(seed: Optional[int] = None) -> None:
        original_reset(seed)
        _crafter_neutralize_wildlife(_crafter_env(program))

    program.reset = reset_and_neutralize
    _crafter_neutralize_wildlife(_crafter_env(program))  # the env built by __init__


def _crafter_clear_walk_corridor(program: Any) -> None:
    """Clear a corridor of terrain in the MOVE_UP direction (negative y) so
    the walk_forward agent doesn't get stuck on trees/stone. Re-applied after
    every ``reset(seed)`` so each seed still generates its own terrain outside
    the corridor -- the agent sees varied surroundings while having a
    guaranteed walkable path."""
    original_reset = program.reset

    def reset_clear_and_neutralize(seed: Optional[int] = None) -> None:
        original_reset(seed)
        env = _crafter_env(program)
        _crafter_neutralize_wildlife(env)
        px, py = int(env._player.pos[0]), int(env._player.pos[1])
        # Clear a 3-wide corridor from the player to the top edge (y=0).
        # Width of 3 (player column +/- 1) gives margin for the player's
        # collision box without flattening the whole world.
        world = env._world
        area = world.area
        for x in range(max(0, px - 1), min(area[0], px + 2)):
            for y in range(0, py + 1):
                world[(x, y)] = "grass"

    program.reset = reset_clear_and_neutralize
    # Apply to the env built by __init__ too
    env = _crafter_env(program)
    _crafter_neutralize_wildlife(env)
    px, py = int(env._player.pos[0]), int(env._player.pos[1])
    world = env._world
    area = world.area
    for x in range(max(0, px - 1), min(area[0], px + 2)):
        for y in range(0, py + 1):
            world[(x, y)] = "grass"


def _crafter_box_in_player(program: Any) -> None:
    """Wall the player in on all four sides with stone -- every MOVE_*
    attempt is blocked, so only ``facing`` changes (issue #90's discrete
    ``turn``)."""
    _crafter_freeze_wildlife(program)
    env = _crafter_env(program)
    x, y = int(env._player.pos[0]), int(env._player.pos[1])
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        env._world[(x + dx, y + dy)] = "stone"


def _crafter_script_mob_path(mob: Any, path: List[Tuple[int, int]]) -> None:
    """Replace one Crafter object's per-tick ``update()`` with a scripted
    position list -- ``path[i]`` is the mob's ``(x, y)`` at update call
    index ``i``; holds its final position once the path is exhausted.
    Skips (holds) a step whose target cell is occupied rather than raising
    -- ``_crafter_freeze_wildlife`` should already prevent collisions, but
    this stays robust to any it doesn't.  Same shape as
    ``_install_scripted_mob_path`` below, adapted to ``crafter.World``'s
    ``move``/``_obj_map``."""
    import numpy as np

    state = {"i": 0}

    def scripted_update(self: Any) -> None:
        i = min(state["i"], len(path) - 1)
        state["i"] += 1
        target = path[i]
        if tuple(int(v) for v in self.pos) == target:
            return
        if self.world._obj_map[target] == 0:
            self.world.move(self, np.array(target))

    mob.update = scripted_update.__get__(mob)


def _crafter_occlusion_distances(close: int, hidden: int, phase_ticks: int) -> List[int]:
    """Distances (agent-relative, along one axis) for the three-phase
    excursion: visible near -> ramps out past the view radius (occluded) ->
    ramps back to visible near. Mirrors ``_occlusion_dz_sequence`` below."""
    import numpy as np

    outbound = np.linspace(close, hidden, phase_ticks).round().astype(int).tolist()
    inbound = np.linspace(hidden, close, phase_ticks).round().astype(int).tolist()
    return outbound + [hidden] * phase_ticks + inbound


_CRAFTER_MOVE_UP = Action("MOVE_UP")


def _crafter_walk_forward(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    return ScenarioRecording(
        policy=ConstantActionPolicy(_CRAFTER_MOVE_UP, seed=seed),
        scene_setup=_crafter_clear_walk_corridor,
    )


def _crafter_turn(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    phase = max(1, cfg.episode_ticks // 4)
    policy = ScriptedSequencePolicy(
        [
            (Action("MOVE_UP"), phase), (Action("MOVE_RIGHT"), phase),
            (Action("MOVE_DOWN"), phase), (Action("MOVE_LEFT"), phase),
        ]
    )
    return ScenarioRecording(policy=policy, scene_setup=_crafter_box_in_player)


def _crafter_approach_entity(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    distance = 6 + (seed % 6)

    def _setup_approach(program: Any) -> None:
        import crafter as crafter_pkg

        env = _crafter_env(program)
        _crafter_neutralize_wildlife(env)
        x, y = int(env._player.pos[0]), int(env._player.pos[1])
        target = (x, y - distance)
        _crafter_clear_terrain(env._world, x, y - distance // 2, radius=distance + 3)
        cow = crafter_pkg.objects.Cow(env._world, target)
        env._world.add(cow)
        cow.update = lambda: None

    def scene_setup(program: Any) -> None:
        original_reset = program.reset

        def reset_and_setup(seed: Optional[int] = None) -> None:
            original_reset(seed)
            _setup_approach(program)

        program.reset = reset_and_setup
        _setup_approach(program)

    return ScenarioRecording(policy=ConstantActionPolicy(_CRAFTER_MOVE_UP), scene_setup=scene_setup)


def _crafter_object_permanence(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    phase_ticks = max(5, cfg.episode_ticks // 3)
    close = 2
    hidden = CrafterConfig().grid_radius + 3 + (seed % 4)

    def scene_setup(program: Any) -> None:
        import crafter as crafter_pkg

        env = _crafter_env(program)
        _crafter_freeze_wildlife(program)
        x, y = int(env._player.pos[0]), int(env._player.pos[1])
        _crafter_clear_terrain(env._world, x, y, radius=hidden + 3)
        cow = crafter_pkg.objects.Cow(env._world, (x + close, y))
        env._world.add(cow)
        path = [(x + d, y) for d in _crafter_occlusion_distances(close, hidden, phase_ticks)]
        _crafter_script_mob_path(cow, path)

    return ScenarioRecording(
        policy=NullPolicy(),
        scene_setup=scene_setup,
        episode_ticks=phase_ticks * 3,
    )


CRAFTER_SCENARIOS: Dict[str, NurseryScenario] = {
    "walk_forward": NurseryScenario(
        "walk_forward",
        "constant MOVE_UP over varied terrain seeds -- ego-motion/optical-flow "
        "regularities (Crafter port of the Minecraft scenario of the same name).",
        _crafter_walk_forward,
        # Crafter's world is a small (default 64x64) bounded grid -- a
        # constant single-direction walk plateaus once it hits the map edge
        # or an obstacle, diluting the average the longer the episode runs
        # past that point (mirrors why Minecraft's own thresholds sit well
        # below a healthy run's rate; this floor just accounts for a smaller
        # world reaching its plateau sooner).
        min_blocks_per_tick=0.01,
        min_unique_frame_fraction=0.05,
    ),
    "turn": NurseryScenario(
        "turn",
        "boxed in on all four sides, cycling the four directional actions -- "
        "discrete facing changes with zero displacement (Crafter's re-scoped "
        "port of turn_in_place: a discrete flip, not continuous rotation).",
        _crafter_turn,
        max_blocks_per_tick=0.0,
        min_unique_facings=4,
    ),
    "object_permanence": NurseryScenario(
        "object_permanence",
        "a scripted mob walks out past the egocentric view radius and back -- "
        "genuine limited perceptual range, not a synthetic occluder (Crafter "
        "port; does not report the entity-persistence metric -- see module "
        "docstring above).",
        _crafter_object_permanence,
    ),
    "approach_entity": NurseryScenario(
        "approach_entity",
        "scripted approach to a frozen entity -- scale change with distance.",
        _crafter_approach_entity,
        # The approach distance is 6-11 blocks (seed-dependent) and the
        # agent stops once blocked by the entity, so with 400 ticks the
        # per-tick rate is only 6/400=0.015 at the shortest distance.
        min_blocks_per_tick=0.01,
    ),
}


# --------------------------------------------------------------------------- data-quality gate
#
# The gate itself moved to ``cognitive_runtime.record.quality`` (issue #90):
# a world-agnostic module that reads any Program's stream log the same way.
# ``EpisodeRecordingQuality``/``measure_recording_quality`` are re-exported
# here unchanged for back-compat; ``validate_nursery_recordings`` adapts a
# ``NurseryScenario``'s threshold fields to the generic gate.


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


def validate_nursery_recordings(
    session_dirs: Sequence[str],
    scenario: NurseryScenario,
    *,
    expected_pixel_source: Optional[str] = None,
) -> List[str]:
    """Check every recorded episode against the scenario's data-quality
    expectations; returns human-readable issue strings (empty = healthy).
    Adapts ``NurseryScenario``'s threshold fields to the world-agnostic gate
    in ``record.quality``.

    Exists because of the first real ``walk_forward`` run: recorded against
    the remote backend's persistent world, the agent was stuck against an
    obstacle with ~2% unique frames and near-zero displacement -- data with
    none of the ego-motion regularity the scenario exists to capture, which
    no amount of training can fix.  Extended after the first real
    ``turn_in_place`` run: the agent drifted up to 24 blocks, one session
    was killed by mobs mid-episode, and the requested first-person viewer
    had silently fallen back to the grid render -- so the gate now also
    checks displacement ceilings, yaw sweep, episode completion, and pixel
    provenance (no mixing, and matching ``expected_pixel_source`` when set).
    """
    return validate_recordings(
        session_dirs,
        name=scenario.name,
        min_blocks_per_tick=scenario.min_blocks_per_tick,
        min_unique_frame_fraction=scenario.min_unique_frame_fraction,
        max_blocks_per_tick=scenario.max_blocks_per_tick,
        min_yaw_sweep_degrees=scenario.min_yaw_sweep_degrees,
        min_unique_facings=scenario.min_unique_facings,
        require_completed=scenario.require_completed,
        expected_pixel_source=expected_pixel_source,
    )


def _measured_ticks_per_frame(session_dirs: Sequence[str]) -> float:
    """Median cognitive ticks per recorded vision frame across the given
    sessions -- ~1.0 on the simulated backend, ~2.0 on the first paced remote
    runs.  Falls back to 1.0 when nothing is measurable."""
    values: List[float] = []
    for session_dir in session_dirs:
        for episode_id in list_episodes(session_dir):
            quality = measure_recording_quality(session_dir, episode_id)
            if quality.n_frames > 1 and quality.duration_ticks > 0:
                values.append(quality.duration_ticks / quality.n_frames)
    if not values:
        return 1.0
    values.sort()
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


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


def _build_scenario_program(cfg: NurseryConfig, program_config: Dict[str, Any]) -> Any:
    """Construct the Program a scenario records against (issue #90's
    ``--world`` selector): ``MinecraftSurvivalBox`` (default, back-compat) or
    ``CrafterWorld``. Mirrors ``cli.py``'s ``_build_program`` factory, minus
    the stream/action-registry return value nursery recording doesn't need."""
    if cfg.world == "crafter":
        from cognitive_runtime.programs.crafter.adapter import CrafterWorld

        return CrafterWorld(config=program_config)
    return MinecraftSurvivalBox(config=program_config, backend=cfg.backend)


def _record_scenario_episode(
    record_dir: str, session_id: str, seed: int, scenario: NurseryScenario, cfg: NurseryConfig
) -> str:
    log.info("recording %s  seed=%d  ticks=%d", session_id, seed, cfg.episode_ticks)
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
        name=cfg.name,
    )
    program = _build_scenario_program(cfg, program_config)
    # Crafter's scripted scenarios always run against the (only) live env;
    # Minecraft's scene-setup only makes sense against the simulated
    # backend's in-process world (a remote server has no scriptable world to
    # reach into).
    if recording.scene_setup is not None and (cfg.world == "crafter" or cfg.backend == "simulated"):
        recording.scene_setup(program)
    try:
        CognitiveRuntime(program=program, policy=recording.policy, config=runtime_config).run()
    finally:
        close = getattr(program, "close", None)
        if callable(close):
            close()
    return os.path.join(record_dir, session_id)


# --------------------------------------------------------------------------- benchmark harness


def _scenarios_for_world(world: str) -> Dict[str, NurseryScenario]:
    """``--world`` selector (issue #90): which scenario registry a
    ``NurseryConfig.world`` records against."""
    if world == "minecraft":
        return NURSERY_SCENARIOS
    if world == "crafter":
        return CRAFTER_SCENARIOS
    raise ValueError(f"unknown nursery world {world!r}; choices: ['crafter', 'minecraft']")


def _action_keys_for_world(world: str) -> List[str]:
    """Return the selected World's complete, stable action vocabulary.

    Stage recordings often exercise only a subset. Pinning checkpoints to
    that incidental subset makes warm-starting across nursery/development
    stages unsafe, so every dataset for a World uses the registry's full
    ordered action space from the outset.
    """
    if world == "crafter":
        from cognitive_runtime.programs.crafter.actions import ACTION_SPACE
    elif world == "minecraft":
        from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
    else:
        raise ValueError(f"unknown nursery world {world!r}; choices: ['crafter', 'minecraft']")
    return [action.key() for action in ACTION_SPACE]


def run_nursery_scenario(
    record_dir: str,
    scenario_name: str,
    config: Optional[NurseryConfig] = None,
    *,
    cortex_checkpoint_path: Optional[str] = None,
) -> Tuple[VisualRepresentationModel, NurseryScenarioReport]:
    """Record train/holdout episodes for one nursery scenario, pretrain a
    pixel encoder+decoder+next-latent predictor on the train seeds only,
    then evaluate multi-horizon next-frame prediction on held-out seeds
    against copy-last-frame and mean-frame baselines. ``object_permanence``
    additionally reports an entity-persistence metric (Minecraft only --
    Crafter's port doesn't set ``entity_persistence_metric``).

    ``cortex_checkpoint_path`` (issue #134), when given, warm-starts
    pretraining from that path's previously-saved
    ``prediction_export.save_full_visual_model`` bundle (if it exists)
    instead of a fresh random model, and saves the result back to it
    afterward -- so a caller that reuses the same path across repeated
    calls (e.g. one milestone-gate attempt per stage attempt) is actually
    continuing to train *the same* model, not discarding it and measuring a
    disposable one every time.
    """

    log.info("=== nursery scenario: %s ===", scenario_name)
    cfg = config or NurseryConfig()
    scenarios = _scenarios_for_world(cfg.world)
    if scenario_name not in scenarios:
        raise ValueError(
            f"unknown nursery scenario {scenario_name!r} for --world {cfg.world!r}; "
            f"choices: {sorted(scenarios)}"
        )
    scenario = scenarios[scenario_name]
    if cfg.world == "minecraft" and cfg.backend not in BACKENDS:
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
    log.info("recorded %d train + %d holdout sessions", len(train_sessions), len(holdout_sessions))

    if cfg.data_quality_gate:
        issues = validate_nursery_recordings(
            train_sessions + holdout_sessions,
            scenario,
            expected_pixel_source=cfg.expected_pixel_source,
        )
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

    log.info("quality gate passed")
    ticks_per_frame = _measured_ticks_per_frame(train_sessions + holdout_sessions)
    horizon_frames = horizons_ticks_to_frames(cfg.horizons, ticks_per_frame)
    log.info("horizons (ticks): %s -> frames: %s  (%.2f ticks/frame)",
             cfg.horizons, horizon_frames, ticks_per_frame)

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
    initial_model = None
    if cortex_checkpoint_path is not None and os.path.exists(cortex_checkpoint_path):
        log.info("warm-starting from checkpoint %s", cortex_checkpoint_path)
        initial_model = load_full_visual_model(cortex_checkpoint_path)
    log.info("training pixel encoder  samples=%d  epochs=%d  lr=%s",
             len(train_dataset), cfg.epochs, cfg.lr)
    model, pretraining_stats = train_pixel_encoder_pretraining(
        train_dataset, visual_config, initial_model=initial_model,
    )
    log.info("pixel encoder training complete")

    consistency_stats: Dict[str, List[float]] = {}
    if cfg.consistency_epochs > 0:
        consistency_stats = train_horizon_consistency(
            model,
            train_sessions,
            horizon_frames,
            epochs=cfg.consistency_epochs,
            lr=cfg.consistency_lr,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
        )
    if cortex_checkpoint_path is not None:
        save_full_visual_model(model, cortex_checkpoint_path)

    max_horizon = max(horizon_frames)
    for session_dir in holdout_sessions:
        for episode_id in list_episodes(session_dir):
            if len(load_episode_pixel_frames(session_dir, episode_id)) <= max_horizon:
                raise ValueError(
                    f"{session_dir}/{episode_id} is too short for the largest horizon "
                    f"({max_horizon} frames); increase episode_ticks"
                )

    log.info("evaluating on %d holdout sessions", len(holdout_sessions))
    horizon_metrics = evaluate_ego_motion_holdout(
        model, holdout_sessions, horizon_frames, ssim_window=cfg.ssim_window
    )
    rollout_health = evaluate_rollout_health(model, holdout_sessions, horizon_frames)
    for h, metrics in horizon_metrics.items():
        log.info("  t+%d: model_mse=%.4f  copy_last_mse=%.4f  beats=%s",
                 h, metrics.get("model_mse", 0), metrics.get("copy_last_mse", 0),
                 metrics.get("beats_copy_last", "?"))

    entity_persistence_stats: Optional[Dict[str, Any]] = None
    if scenario.entity_persistence_metric:
        entity_persistence_stats = _run_entity_persistence_metric(
            train_sessions, holdout_sessions, cfg
        )

    dream_strips: Dict[str, str] = {}
    for session_dir in holdout_sessions:
        for episode_id in list_episodes(session_dir):
            dream_strips[f"{session_dir}/{episode_id}"] = render_dream_strip(
                model, session_dir, episode_id, horizon_frames
            )

    # The checkpoint persists only the encoder, so predicted frames are
    # unrecoverable once this process exits -- export them now for the pixel
    # viewer's "model" source (viewer/README.md).
    prediction_files: Dict[str, str] = {}
    if cfg.export_predictions:
        prediction_files = export_session_predictions(
            model, train_sessions + holdout_sessions, horizon_frames
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
        horizon_frames=horizon_frames,
        ticks_per_frame=ticks_per_frame,
        rollout_health=rollout_health,
    )


def run_nursery_suite(
    record_dir: str,
    scenario_names: Optional[Sequence[str]] = None,
    config: Optional[NurseryConfig] = None,
) -> Dict[str, NurseryScenarioReport]:
    """``nursery run all``: run every named scenario (default: every
    scenario registered for ``config.world``) unattended, returning one
    report per scenario."""

    cfg = config or NurseryConfig()
    names = (
        list(scenario_names)
        if scenario_names is not None
        else sorted(_scenarios_for_world(cfg.world))
    )
    reports: Dict[str, NurseryScenarioReport] = {}
    for name in names:
        _model, report = run_nursery_scenario(record_dir, name, cfg)
        reports[name] = report
    return reports


# --------------------------------------------------------------------------- joint world model


@dataclass
class JointNurseryReport:
    """One action-conditioned world model trained across scenarios
    (phase 3 of docs/history/nursery-turn-in-place-analysis.md)."""

    train_scenarios: List[str]
    holdout_scenarios: List[str]
    config: NurseryConfig
    model_config: ActionWorldModelConfig
    #: scenario -> its recorded train-seed session dirs (training pool).
    train_sessions: Dict[str, List[str]] = field(default_factory=dict)
    #: scenario -> its recorded holdout-seed session dirs (evaluation).
    eval_sessions: Dict[str, List[str]] = field(default_factory=dict)
    training_stats: Dict[str, Any] = field(default_factory=dict)
    #: In-distribution generalization: per train scenario, evaluated on that
    #: scenario's held-out seeds ({"horizons": ..., "rollout_health": ...}).
    scenario_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: Zero-shot generality: per held-out scenario the model never trained
    #: on, same metric shape.
    zero_shot_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: Does the representation linearly decode the agent's heading?
    yaw_probe: Dict[str, Any] = field(default_factory=dict)
    #: Heading probe using yaw or Crafter's discrete facing stream.
    orientation_probe: Dict[str, Any] = field(default_factory=dict)
    #: Promotion-grade yaw + latent variance/effective-rank collapse gate.
    representation_diagnostics: Dict[str, Any] = field(default_factory=dict)
    horizon_frames: List[int] = field(default_factory=list)
    ticks_per_frame: float = 1.0


def run_nursery_joint(
    record_dir: str,
    train_scenarios: Optional[Sequence[str]] = None,
    holdout_scenarios: Sequence[str] = ("approach_entity",),
    config: Optional[NurseryConfig] = None,
    model_config: Optional[ActionWorldModelConfig] = None,
) -> Tuple[Any, JointNurseryReport]:
    """Record every scenario, train ONE action-conditioned recurrent world
    model on the train scenarios' train seeds, then evaluate:

    - per train scenario on its held-out seeds (in-distribution
      generalization),
    - per held-out scenario the model never saw (zero-shot generality --
      the metric that separates "memorized six scripted policies" from
      "learned how actions move the view"),
    - a yaw linear probe (does the latent/hidden state carry heading?).

    This is the "general model" counterpart of ``run_nursery_suite``'s
    per-scenario canaries: the same recordings, one shared model, and the
    action stream (already in every log as ``motor.command``) finally used
    as a model input instead of being baked into per-scenario dynamics.
    """
    cfg = config or NurseryConfig()
    if cfg.backend not in BACKENDS:
        raise ValueError(f"unknown nursery backend {cfg.backend!r}; choices: {sorted(BACKENDS)}")
    if not cfg.horizons or any(h <= 0 for h in cfg.horizons):
        raise ValueError(f"horizons must be positive tick offsets, got {cfg.horizons!r}")
    if set(cfg.train_seeds) & set(cfg.holdout_seeds):
        raise ValueError("train_seeds and holdout_seeds must not overlap")

    holdout_names = list(holdout_scenarios)
    train_names = (
        list(train_scenarios)
        if train_scenarios is not None
        else [n for n in sorted(NURSERY_SCENARIOS) if n not in holdout_names]
    )
    for name in list(train_names) + holdout_names:
        if name not in NURSERY_SCENARIOS:
            raise ValueError(
                f"unknown nursery scenario {name!r}; choices: {sorted(NURSERY_SCENARIOS)}"
            )
    overlap = set(train_names) & set(holdout_names)
    if overlap:
        raise ValueError(f"scenarios cannot be both trained and held out: {sorted(overlap)}")
    if not train_names:
        raise ValueError("no training scenarios left after excluding holdouts")

    model_cfg = model_config or ActionWorldModelConfig(
        latent_width=cfg.latent_width,
        hidden_dim=cfg.hidden_dim,
        reconstruction_size=cfg.reconstruction_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    train_sessions: Dict[str, List[str]] = {}
    eval_sessions: Dict[str, List[str]] = {}
    for name in train_names:
        scenario = NURSERY_SCENARIOS[name]
        train_sessions[name] = [
            _record_scenario_episode(
                record_dir, f"nursery-{name}-train-{seed}", seed, scenario, cfg
            )
            for seed in cfg.train_seeds
        ]
        eval_sessions[name] = [
            _record_scenario_episode(
                record_dir, f"nursery-{name}-holdout-{seed}", seed, scenario, cfg
            )
            for seed in cfg.holdout_seeds
        ]
    for name in holdout_names:
        scenario = NURSERY_SCENARIOS[name]
        eval_sessions[name] = [
            _record_scenario_episode(
                record_dir, f"nursery-{name}-holdout-{seed}", seed, scenario, cfg
            )
            for seed in cfg.holdout_seeds
        ]

    if cfg.data_quality_gate:
        issues: List[str] = []
        for name in train_names:
            issues += validate_nursery_recordings(
                train_sessions[name] + eval_sessions[name],
                NURSERY_SCENARIOS[name],
                expected_pixel_source=cfg.expected_pixel_source,
            )
        for name in holdout_names:
            issues += validate_nursery_recordings(
                eval_sessions[name],
                NURSERY_SCENARIOS[name],
                expected_pixel_source=cfg.expected_pixel_source,
            )
        if issues:
            raise ValueError(
                "nursery joint run: recorded data fails the quality gate:\n  - "
                + "\n  - ".join(issues)
            )

    # Pin the vocabulary to the full action space (plus NULL): a held-out
    # scenario may issue actions no training scenario used, and zero-shot
    # evaluation must be able to encode them (their embeddings are simply
    # untrained).
    vocabulary = _action_keys_for_world(cfg.world)

    all_train_dirs = [d for name in train_names for d in train_sessions[name]]
    dataset = build_action_sequence_dataset(all_train_dirs, action_keys=vocabulary)
    if len(dataset) == 0:
        raise ValueError("nursery joint run: no frame transitions in the training sessions")
    ticks_per_frame = dataset.ticks_per_frame
    horizon_frames = horizons_ticks_to_frames(cfg.horizons, ticks_per_frame)

    model, training_stats = train_action_world_model(dataset, model_cfg)

    scenario_metrics: Dict[str, Dict[str, Any]] = {}
    for name in train_names:
        holdout_dataset = build_action_sequence_dataset(
            eval_sessions[name], action_keys=model.action_keys
        )
        scenario_metrics[name] = evaluate_action_world_model(
            model, holdout_dataset, horizon_frames, warmup_frames=model_cfg.warmup_frames
        )
    zero_shot_metrics: Dict[str, Dict[str, Any]] = {}
    for name in holdout_names:
        holdout_dataset = build_action_sequence_dataset(
            eval_sessions[name], action_keys=model.action_keys
        )
        zero_shot_metrics[name] = evaluate_action_world_model(
            model, holdout_dataset, horizon_frames, warmup_frames=model_cfg.warmup_frames
        )

    probe_dataset = build_action_sequence_dataset(
        [d for dirs in eval_sessions.values() for d in dirs], action_keys=model.action_keys
    )
    yaw_probe = linear_probe_yaw(model, probe_dataset)
    orientation_probe = linear_probe_orientation(model, probe_dataset)
    representation_diagnostics = representation_collapse_diagnostics(
        model, probe_dataset, config=model_cfg
    )

    return model, JointNurseryReport(
        train_scenarios=train_names,
        holdout_scenarios=holdout_names,
        config=cfg,
        model_config=model_cfg,
        train_sessions=train_sessions,
        eval_sessions=eval_sessions,
        training_stats=training_stats,
        scenario_metrics=scenario_metrics,
        zero_shot_metrics=zero_shot_metrics,
        yaw_probe=yaw_probe,
        orientation_probe=orientation_probe,
        representation_diagnostics=representation_diagnostics,
        horizon_frames=horizon_frames,
        ticks_per_frame=ticks_per_frame,
    )


# --------------------------------------------------------------------------- action-ablation (issue #92)


@dataclass
class ActionAblationReport:
    """Milestone 2's action-ablation proof
    (docs/v2/phases/phase-2-predictive-cortex.md): the same joint cortex,
    trained twice on byte-identical recordings -- once seeing the real
    ``motor.command`` stream, once with every action index overwritten by a
    constant -- to show action-conditioning is load-bearing rather than
    decorative. "A predictor that never sees its action can't tell 'kept
    turning' from 'stopped'." """

    train_scenarios: List[str]
    eval_scenario: str
    #: Full ``evaluate_action_world_model`` report for the model trained
    #: with the real action stream.
    with_actions_metrics: Dict[str, Any]
    #: Same, for the model trained with actions withheld.
    without_actions_metrics: Dict[str, Any]
    #: Per-horizon mean +/- CI over held-out seeds for each model
    #: (``statistical_evaluation.cortex_horizon_statistics``).
    with_actions_stats: Dict[int, MetricStats]
    without_actions_stats: Dict[int, MetricStats]
    #: Per-horizon regression comparison, ablated-vs-baseline
    #: (``statistical_evaluation.compare_cortex_horizon_statistics``).
    comparisons: Dict[int, MetricComparison]
    #: True when withholding actions raises ``eval_scenario``'s held-out
    #: model MSE at every evaluated horizon -- the Milestone 2 assertion.
    action_withholding_degrades: bool
    #: Held-out representation gate for the promoted with-actions cortex.
    representation_diagnostics: Dict[str, Any] = field(default_factory=dict)


def run_action_ablation_eval(
    record_dir: str,
    train_scenarios: Sequence[str] = ("walk_forward", "turn_in_place"),
    eval_scenario: str = "turn_in_place",
    config: Optional[NurseryConfig] = None,
    model_config: Optional[ActionWorldModelConfig] = None,
    *,
    cortex_checkpoint_path: Optional[str] = None,
) -> ActionAblationReport:
    """Train the joint cortex twice on identical recorded data -- with and
    without the action stream reaching the model during training -- and
    compare held-out ``eval_scenario`` performance.

    Both runs share the same recordings, architecture, and training
    hyperparameters (only ``ActionWorldModelConfig.withhold_actions``
    differs), so a regression on ``eval_scenario`` when withheld is direct
    evidence the baseline model actually uses its action input, not an
    artifact of noise or a different random init. ``eval_scenario`` must be
    one of ``train_scenarios`` -- the claim is "harder to predict the
    scenario it trained on", not zero-shot generality.

    ``cortex_checkpoint_path`` (issue #134), when given, warm-starts the
    *with-actions* run from that path's previously-saved
    :func:`~cognitive_runtime.training.action_world_model.save_action_world_model`
    bundle (if it exists) and saves the result back to it afterward, so a
    caller reusing the same path across calls actually keeps improving the
    same cortex instead of measuring a fresh, disposable one every attempt.

    The *without-actions* control warm-starts from (and saves to) its own
    sibling path (``cortex_checkpoint_path + ".control"``) rather than the
    with-actions path -- it must never see the with-actions run's weights or
    actions, but it does need the *same accumulated training budget* across
    repeated calls (PR #155 review): warm-starting only the with-actions run
    would let it accumulate strictly more total training than a
    freshly-initialized control every attempt, so ``action_ablation_margin``
    could turn positive from more training alone rather than from access to
    actions -- exactly the confound this ablation exists to rule out.
    """
    cfg = config or NurseryConfig()
    log.info("=== action ablation eval  train=%s  eval=%s ===", list(train_scenarios), eval_scenario)
    if eval_scenario not in train_scenarios:
        raise ValueError(
            f"eval_scenario {eval_scenario!r} must be one of the trained scenarios "
            f"{list(train_scenarios)!r}: the ablation proves training-time action-"
            "conditioning matters for a scenario the model trained on, not zero-shot "
            "generality to one it never saw"
        )
    if set(cfg.train_seeds) & set(cfg.holdout_seeds):
        raise ValueError("train_seeds and holdout_seeds must not overlap")

    scenarios = _scenarios_for_world(cfg.world)
    for name in train_scenarios:
        if name not in scenarios:
            raise ValueError(
                f"unknown nursery scenario {name!r} for --world {cfg.world!r}; "
                f"choices: {sorted(scenarios)}"
            )

    base_model_cfg = model_config or ActionWorldModelConfig(
        latent_width=cfg.latent_width,
        hidden_dim=cfg.hidden_dim,
        reconstruction_size=cfg.reconstruction_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    train_sessions: Dict[str, List[str]] = {}
    eval_sessions: Dict[str, List[str]] = {}
    for name in train_scenarios:
        scenario = scenarios[name]
        train_sessions[name] = [
            _record_scenario_episode(record_dir, f"ablation-{name}-train-{seed}", seed, scenario, cfg)
            for seed in cfg.train_seeds
        ]
        eval_sessions[name] = [
            _record_scenario_episode(record_dir, f"ablation-{name}-holdout-{seed}", seed, scenario, cfg)
            for seed in cfg.holdout_seeds
        ]

    if cfg.data_quality_gate:
        issues: List[str] = []
        for name in train_scenarios:
            issues += validate_nursery_recordings(
                train_sessions[name] + eval_sessions[name], scenarios[name],
                expected_pixel_source=cfg.expected_pixel_source,
            )
        if issues:
            raise ValueError(
                "action-ablation eval: recorded data fails the quality gate:\n  - "
                + "\n  - ".join(issues)
            )

    # Pin the vocabulary to the full action space (issue #91's joint-training
    # convention) so both runs' models share one embedding table even though
    # only one is trained to actually read it.
    vocabulary = _action_keys_for_world(cfg.world)

    all_train_dirs = [d for name in train_scenarios for d in train_sessions[name]]
    dataset = build_action_sequence_dataset(all_train_dirs, action_keys=vocabulary)
    if len(dataset) == 0:
        raise ValueError("action-ablation eval: no frame transitions in the training sessions")
    horizon_frames = horizons_ticks_to_frames(cfg.horizons, dataset.ticks_per_frame)

    with_actions_cfg = replace(base_model_cfg, withhold_actions=False)
    # The intentionally-deprived control must remain measurable even if its
    # representation collapses; promotion gates the real with-actions model.
    without_actions_cfg = replace(
        base_model_cfg, withhold_actions=True, collapse_gate_enabled=False
    )

    without_actions_checkpoint_path = (
        cortex_checkpoint_path + ".control" if cortex_checkpoint_path is not None else None
    )

    with_initial = None
    if cortex_checkpoint_path is not None and os.path.exists(cortex_checkpoint_path):
        with_initial, _ = load_action_world_model(cortex_checkpoint_path)
    model_with, stats_with = train_action_world_model(
        dataset, with_actions_cfg, initial_model=with_initial,
    )
    if cortex_checkpoint_path is not None:
        save_action_world_model(cortex_checkpoint_path, model_with, stats_with)

    # The control warm-starts from its own sibling checkpoint (never the
    # with-actions one) so it accumulates the *same* total training budget
    # across repeated calls while still never seeing actions (see docstring).
    without_initial = None
    if without_actions_checkpoint_path is not None and os.path.exists(without_actions_checkpoint_path):
        without_initial, _ = load_action_world_model(without_actions_checkpoint_path)
    model_without, stats_without = train_action_world_model(
        dataset, without_actions_cfg, initial_model=without_initial,
    )
    if without_actions_checkpoint_path is not None:
        save_action_world_model(without_actions_checkpoint_path, model_without, stats_without)

    holdout_dataset = build_action_sequence_dataset(
        eval_sessions[eval_scenario], action_keys=model_with.action_keys
    )
    with_actions_metrics = evaluate_action_world_model(
        model_with, holdout_dataset, horizon_frames, warmup_frames=base_model_cfg.warmup_frames
    )
    without_actions_metrics = evaluate_action_world_model(
        model_without, holdout_dataset, horizon_frames, warmup_frames=base_model_cfg.warmup_frames
    )

    with_actions_stats = cortex_horizon_statistics(with_actions_metrics["per_episode_model_mse"])
    without_actions_stats = cortex_horizon_statistics(without_actions_metrics["per_episode_model_mse"])
    comparisons = compare_cortex_horizon_statistics(with_actions_stats, without_actions_stats)

    degrades = bool(horizon_frames) and all(
        without_actions_metrics["horizons"][h]["model_mse"]
        > with_actions_metrics["horizons"][h]["model_mse"]
        for h in horizon_frames
    )
    representation_diagnostics = representation_collapse_diagnostics(
        model_with, holdout_dataset, config=with_actions_cfg
    )

    log.info("ablation result  degrades_without_actions=%s", degrades)
    return ActionAblationReport(
        train_scenarios=list(train_scenarios),
        eval_scenario=eval_scenario,
        with_actions_metrics=with_actions_metrics,
        without_actions_metrics=without_actions_metrics,
        with_actions_stats=with_actions_stats,
        without_actions_stats=without_actions_stats,
        comparisons=comparisons,
        action_withholding_degrades=degrades,
        representation_diagnostics=representation_diagnostics,
    )


# --------------------------------------------------------------------------- backbone benchmark (issue #93)


@dataclass
class BackboneBenchmarkReport:
    """A/B benchmark of the cortex's temporal backbones
    (docs/v2/phases/phase-2-predictive-cortex.md task 5): the same
    recordings, the same architecture and training budget otherwise, only
    ``ActionWorldModelConfig.backbone`` differs -- so a difference in the
    Phase 2 scoring gates (``model/copy-last``, ``model/oracle``,
    frozen-rollout) is attributable to the backbone choice, not noise."""

    train_scenarios: List[str]
    eval_scenario: str
    baseline_backbone: str
    #: backbone name -> full ``evaluate_action_world_model`` report (the
    #: Phase 2 scoring gates per horizon, plus rollout health).
    metrics: Dict[str, Dict[str, Any]]
    #: backbone name -> per-horizon mean +/- CI over held-out seeds.
    stats: Dict[str, Dict[int, MetricStats]]
    #: backbone name (excluding ``baseline_backbone``) -> per-horizon
    #: regression comparison against the baseline backbone.
    comparisons: Dict[str, Dict[int, MetricComparison]]
    #: backbone name -> {horizon: beats_copy_last} (the Milestone 2 gate
    #: each backbone must clear on its own to be a credible alternative).
    beats_copy_last: Dict[str, Dict[int, bool]]


def run_backbone_benchmark(
    record_dir: str,
    train_scenarios: Sequence[str] = ("walk_forward", "turn_in_place"),
    eval_scenario: str = "turn_in_place",
    backbones: Sequence[str] = ("gru", "dilated_conv", "transformer"),
    baseline_backbone: str = "gru",
    config: Optional[NurseryConfig] = None,
    model_config: Optional[ActionWorldModelConfig] = None,
) -> BackboneBenchmarkReport:
    """Train the cortex once per backbone on identical recorded data and
    compare held-out ``eval_scenario`` performance -- the "benchmark harness
    reports GRU vs temporal-conv/transformer on the Phase 2 scoring gates"
    exit criterion (issue #93).

    Mirrors :func:`run_action_ablation_eval`'s shape (same recordings, same
    dataset, only the varied field of ``ActionWorldModelConfig`` differs
    across runs) so the two harnesses read the same way.
    """
    cfg = config or NurseryConfig()
    log.info("=== backbone benchmark  backbones=%s  baseline=%s  eval=%s ===",
             list(backbones), baseline_backbone, eval_scenario)
    if eval_scenario not in train_scenarios:
        raise ValueError(
            f"eval_scenario {eval_scenario!r} must be one of the trained scenarios "
            f"{list(train_scenarios)!r}"
        )
    if baseline_backbone not in backbones:
        raise ValueError(
            f"baseline_backbone {baseline_backbone!r} must be one of backbones {list(backbones)!r}"
        )
    if set(cfg.train_seeds) & set(cfg.holdout_seeds):
        raise ValueError("train_seeds and holdout_seeds must not overlap")

    scenarios = _scenarios_for_world(cfg.world)
    for name in train_scenarios:
        if name not in scenarios:
            raise ValueError(
                f"unknown nursery scenario {name!r} for --world {cfg.world!r}; "
                f"choices: {sorted(scenarios)}"
            )

    base_model_cfg = model_config or ActionWorldModelConfig(
        latent_width=cfg.latent_width,
        hidden_dim=cfg.hidden_dim,
        reconstruction_size=cfg.reconstruction_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    train_sessions: Dict[str, List[str]] = {}
    eval_sessions: Dict[str, List[str]] = {}
    for name in train_scenarios:
        scenario = scenarios[name]
        train_sessions[name] = [
            _record_scenario_episode(record_dir, f"backbone-bench-{name}-train-{seed}", seed, scenario, cfg)
            for seed in cfg.train_seeds
        ]
        eval_sessions[name] = [
            _record_scenario_episode(record_dir, f"backbone-bench-{name}-holdout-{seed}", seed, scenario, cfg)
            for seed in cfg.holdout_seeds
        ]

    if cfg.data_quality_gate:
        issues: List[str] = []
        for name in train_scenarios:
            issues += validate_nursery_recordings(
                train_sessions[name] + eval_sessions[name], scenarios[name],
                expected_pixel_source=cfg.expected_pixel_source,
            )
        if issues:
            raise ValueError(
                "backbone benchmark: recorded data fails the quality gate:\n  - "
                + "\n  - ".join(issues)
            )

    vocabulary = _action_keys_for_world(cfg.world)

    all_train_dirs = [d for name in train_scenarios for d in train_sessions[name]]
    dataset = build_action_sequence_dataset(all_train_dirs, action_keys=vocabulary)
    if len(dataset) == 0:
        raise ValueError("backbone benchmark: no frame transitions in the training sessions")
    horizon_frames = horizons_ticks_to_frames(cfg.horizons, dataset.ticks_per_frame)
    holdout_dataset = build_action_sequence_dataset(
        eval_sessions[eval_scenario], action_keys=vocabulary
    )

    metrics: Dict[str, Dict[str, Any]] = {}
    stats: Dict[str, Dict[int, MetricStats]] = {}
    beats_copy_last: Dict[str, Dict[int, bool]] = {}
    for name in backbones:
        backbone_cfg = replace(base_model_cfg, backbone=name)
        model, _train_stats = train_action_world_model(dataset, backbone_cfg)
        report = evaluate_action_world_model(
            model, holdout_dataset, horizon_frames, warmup_frames=base_model_cfg.warmup_frames
        )
        metrics[name] = report
        stats[name] = cortex_horizon_statistics(report["per_episode_model_mse"])
        beats_copy_last[name] = {h: entry["beats_copy_last"] for h, entry in report["horizons"].items()}
        log.info("backbone %s  beats_copy_last=%s", name, beats_copy_last[name])

    comparisons: Dict[str, Dict[int, MetricComparison]] = {
        name: compare_cortex_horizon_statistics(stats[baseline_backbone], stats[name])
        for name in backbones
        if name != baseline_backbone
    }

    return BackboneBenchmarkReport(
        train_scenarios=list(train_scenarios),
        eval_scenario=eval_scenario,
        baseline_backbone=baseline_backbone,
        metrics=metrics,
        stats=stats,
        comparisons=comparisons,
        beats_copy_last=beats_copy_last,
    )


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
    thumb = F.adaptive_avg_pool2d(gray, output_size=(th, tw))[0, 0].nan_to_num(0.0).clamp(0.0, 1.0)
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
        "horizon_frames": list(report.horizon_frames),
        "ticks_per_frame": report.ticks_per_frame,
        "train_sessions": report.train_sessions,
        "holdout_sessions": report.holdout_sessions,
        "horizon_metrics": report.horizon_metrics,
        "rollout_health": report.rollout_health,
        "entity_persistence_stats": report.entity_persistence_stats,
    }
    dataset_stub = _StubPixelDataset(
        layout_hash=None,
        sources=report.train_sessions + report.holdout_sessions,
        pixel_shape=model.pixel_shape,
        representation=f"nursery-{report.scenario}",
    )
    return save_pixel_encoder_pretraining_checkpoint(
        path, model, dataset_stub, stats, name=report.config.name
    )
