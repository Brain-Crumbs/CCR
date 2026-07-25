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
from cognitive_runtime.observability import span, trace_counter, trace_event, trace_metrics
from cognitive_runtime.policies.constant_action import ConstantActionPolicy
from cognitive_runtime.policies.null_policy import NullPolicy
from cognitive_runtime.policies.scripted_sequence import ScriptedSequencePolicy
from cognitive_runtime.programs.crafter.config import CrafterConfig, area_from_world_size
from cognitive_runtime.programs.crafter.streams import FRAME_LEGEND
from cognitive_runtime.programs.minecraft.adapter import BACKENDS, MinecraftSurvivalBox
#: Re-exported for back-compat: this gate moved to ``record.quality`` (issue
#: #90) so it can run world-agnostically; existing imports of
#: ``EpisodeRecordingQuality``/``measure_recording_quality`` from here still
#: work unchanged.
from cognitive_runtime.record.quality import (  # noqa: F401
    EpisodeRecordingQuality,
    SplitOverlapReport,
    audit_split_overlap,
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
    #: Refuse to train when a recorded holdout episode is an exact or
    #: near-exact duplicate of a recorded training episode (issue #202:
    #: ``record.quality.audit_split_overlap``) -- a "held-out" evaluation
    #: that leaky cannot support the generalization claim it exists to make.
    #: Defaults off (unlike ``data_quality_gate``), for two independent
    #: reasons:
    #:
    #: - Minecraft's simulated backend does not re-render
    #:   ``vision.frame.pixels`` for a scripted non-player entity's own
    #:   movement (only for the player's own position/facing changing), so
    #:   ``object_permanence``/``approach_entity`` recordings there are
    #:   pixel-frozen regardless of scripted parameter or seed -- a
    #:   pre-existing rendering gap this issue's Crafter-focused fixes do
    #:   not reach.
    #: - Any scenario (either world) that places its scripted entity/wall
    #:   relative to the player's spawn point -- always the exact world
    #:   center in Crafter, regardless of seed -- legitimately shares much
    #:   of its early "approach" background across every episode; only the
    #:   scripted parameter (distance/depth) varies. That is by design (the
    #:   issue's own guidance: hold out a new distance/depth, not a new
    #:   world), not evidence of a leaked recording, but it also means the
    #:   default ``max_corresponding_frame_fraction`` is far looser than
    #:   these scenarios would need to pass cleanly.
    #:
    #: Turn this on explicitly for scenarios with no such convergence point
    #: (``turn``, ``walk_forward_short`` -- verified leak-free at the
    #: default threshold in ``tests/test_crafter_scenarios.py``), and treat
    #: ``approach_entity``/``object_permanence``/``blocked_forward`` as
    #: needing their own, much looser (or disabled) threshold if enabled at
    #: all.
    split_overlap_gate: bool = False
    #: Hard-fail threshold (fraction of index-aligned matching frames) for
    #: ``split_overlap_gate`` -- see ``audit_split_overlap``'s docstring for
    #: why this defaults looser than the <1% acceptance target: an exact
    #: whole-episode duplicate always hard-fails regardless of this value.
    max_corresponding_frame_fraction: float = 0.35
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
    #: Minimum fraction of frame-to-frame transitions where the agent's
    #: forward-filled grid position actually changed (0.0 = no expectation;
    #: issue #202). Unlike ``min_blocks_per_tick`` (net displacement over
    #: the whole episode, diluted by a long stationary tail),
    #: ``moving_transition_fraction`` looks at every transition, so it
    #: still catches an agent that walked, then got stuck for the back half
    #: of the episode.
    min_moving_transition_fraction: float = 0.0
    #: Minimum fraction of frame-to-frame transitions where the semantic
    #: grid actually changed (0.0 = no expectation; issue #202). Pixel-hash
    #: uniqueness alone cannot distinguish genuine world motion from a
    #: static scene whose pixels merely animate (HUD flicker, render
    #: noise); this floor requires the *semantic* scene to actually change.
    min_semantic_change_fraction: float = 0.0
    #: Upper bound (None = no expectation) on the longest run of
    #: consecutive "position unchanged" transitions ending at the very last
    #: frame -- an accidental long stationary tail (the agent walked into
    #: an obstacle/boundary and idled for the rest of the episode), as
    #: opposed to an intentional, explicitly bounded blocked phase
    #: (``blocked_forward``'s few-tick collision hold, which this bound
    #: comfortably covers).
    max_longest_stationary_tail: Optional[int] = None
    #: Upper bound (None = no expectation) on the longest "position
    #: unchanged" run *anywhere* in the episode -- unlike
    #: ``max_longest_stationary_tail``, this also catches a mid-episode
    #: blocked phase that never actually unblocks (``blocked_forward``'s
    #: recovery action failing to move the agent).
    max_longest_stationary_run: Optional[int] = None


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
    #: The actual ``program_config`` used to construct the recording Program
    #: (post ``world_size``->``area`` translation for Crafter) -- so a test
    #: (or a human comparing runs) can assert the requested world size
    #: actually took effect, without re-deriving it from ``config`` (issue
    #: #202).
    resolved_program_config: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- explicit split parameters (issue #202)
#
# Scripted scenarios that derive a scene parameter (approach distance,
# occlusion depth) from ``seed`` used to do it via a small modulus
# (``6 + seed % 6``): with only 2-4 holdout seeds, a small modulus makes a
# holdout seed land on the exact same parameter value as a training seed --
# a holdout that silently repeats a training variant instead of testing a
# new one. ``_split_parameter`` instead draws from an explicit, disjoint
# tuple of values per split, indexed by the seed's position within its own
# pool (``NurseryConfig.train_seeds``/``holdout_seeds``) -- so disjointness
# is a property of the two tuples, not an accident of the moduli involved.


def _seed_split_index(seed: int, cfg: NurseryConfig) -> Tuple[str, int]:
    """Classify ``seed`` as belonging to ``cfg.train_seeds`` or
    ``cfg.holdout_seeds`` and return its position within that pool."""
    train_seeds = list(cfg.train_seeds)
    holdout_seeds = list(cfg.holdout_seeds)
    if seed in train_seeds:
        return "train", train_seeds.index(seed)
    if seed in holdout_seeds:
        return "holdout", holdout_seeds.index(seed)
    raise ValueError(
        f"seed {seed} is neither a train nor a holdout seed in this NurseryConfig "
        f"(train_seeds={cfg.train_seeds!r}, holdout_seeds={cfg.holdout_seeds!r})"
    )


def _split_parameter(
    seed: int, cfg: NurseryConfig, train_values: Sequence[Any], holdout_values: Sequence[Any]
) -> Any:
    """Pick this seed's scripted scenario parameter from its split's
    explicit value tuple, cycling by position within the pool if there are
    more seeds than declared values. ``train_values``/``holdout_values``
    must be disjoint (see ``tests/test_nursery.py``'s split-parameter
    disjointness checks) -- that, not the seed arithmetic, is what makes a
    holdout episode's scripted parameter always novel relative to training.
    """
    split, index = _seed_split_index(seed, cfg)
    values = train_values if split == "train" else holdout_values
    return values[index % len(values)]


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


#: Disjoint by construction (issue #202) -- see ``_split_parameter``.
_APPROACH_ENTITY_TRAIN_DISTANCES = (6, 8, 10, 12)
_APPROACH_ENTITY_HOLDOUT_DISTANCES = (7, 9, 11, 13)


def _approach_entity(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    distance = _split_parameter(
        seed, cfg, _APPROACH_ENTITY_TRAIN_DISTANCES, _APPROACH_ENTITY_HOLDOUT_DISTANCES
    )

    def scene_setup(program: MinecraftSurvivalBox) -> None:
        world = program._backend.world
        # Regenerate the world from this episode's own seed (issue #202),
        # not a hardcoded ``reset(0)`` -- freezing every seed to the exact
        # same generated background world made train and holdout episodes
        # share most of their approach-phase frames regardless of the
        # (disjoint) scripted distance.
        world.reset(seed)
        ax, az = int(world.x), int(world.z)
        _clear_terrain(world, ax, az, radius=distance + 4)
        _freeze_mobs(world, [{"id": 1, "x": ax, "z": az + distance, "hp": 10, "cooldown": 0}])
        world.reset = lambda seed: None  # type: ignore[method-assign]

    return ScenarioRecording(
        policy=ConstantActionPolicy(MOVE_FORWARD),
        program_config_extra={"max_mobs": 0},
        scene_setup=scene_setup,
    )


#: Disjoint by construction (issue #202) -- occlusion depth (offset past the
#: wall) for the object-permanence excursion, per split.
_OBJECT_PERMANENCE_TRAIN_OFFSETS = tuple(float(_WALL_OFFSET + 2 + d) for d in (0, 2, 4, 6))
_OBJECT_PERMANENCE_HOLDOUT_OFFSETS = tuple(float(_WALL_OFFSET + 2 + d) for d in (1, 3, 5, 7))


def _object_permanence(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    phase_ticks = max(5, cfg.episode_ticks // 3)
    offset = _split_parameter(
        seed, cfg, _OBJECT_PERMANENCE_TRAIN_OFFSETS, _OBJECT_PERMANENCE_HOLDOUT_OFFSETS
    )

    def scene_setup(program: MinecraftSurvivalBox) -> None:
        world = program._backend.world
        # Regenerate the world from this episode's own seed (issue #202) --
        # see ``_approach_entity``'s matching comment.
        world.reset(seed)
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


def _crafter_disable_neglect_and_spawn_balance(env: Any) -> None:
    """Disable Crafter's own periodic spawn/despawn chunk balancing
    (``Env._balance_chunk``, no Minecraft-style ``max_mobs=0`` knob exists)
    and the player's own neglect-driven survival decay (hunger/thirst/energy
    depletion and the health regen/degen it drives --
    ``Player._update_life_stats``/``_degen_or_regen_health``). A short
    scripted micro-scenario isn't testing "can the agent feed itself"; over
    a few hundred ticks of pure ``MOVE_UP`` (issue #90's ``walk_forward``)
    neglect alone starves the agent to death well before that, which isn't
    the regularity these scenarios exist to capture. Orthogonal to whether
    *other* creatures are removed or kept -- every crafter scenario wants
    this, entity scenarios included."""
    env._balance_chunk = lambda chunk, objs: None
    env._player._update_life_stats = lambda: None
    env._player._degen_or_regen_health = lambda: None


def _crafter_remove_wildlife(env: Any) -> None:
    """Remove every non-player creature from ``env`` outright (issue #202).

    The previous approach only froze wildlife's ``update()`` to a no-op,
    which stopped it moving but left it *rendering* in every frame --
    world generation's incidental cow became a permanent feature of every
    non-entity training episode instead of the absence it should have been.
    Non-entity scenarios (``walk_forward_short``, ``blocked_forward``,
    ``turn``) call this so their recorded semantic grid never contains a
    creature id at all."""
    _crafter_disable_neglect_and_spawn_balance(env)
    world = env._world
    for obj in list(world.objects):
        if obj is not env._player:
            world.remove(obj)


def _crafter_freeze_scripted_entities(env: Any, keep: Sequence[Any] = ()) -> None:
    """Remove every non-player creature except ``keep``, and freeze each
    kept entity's own default AI (no-op ``update``) -- issue #202's entity
    scenarios (``approach_entity``, ``object_permanence``) call this right
    after creating their own scripted entity/entities so the only creature
    left in the world is the one they asked for, nothing world generation
    happened to spawn alongside it. Equivalent to removing existing
    wildlife before adding the scripted entity (net result: player + kept
    entities only), just expressed the other way around so both entity
    scenarios can share one call. A caller whose entity needs to actually
    move overrides the no-op afterward (``_crafter_script_mob_path``); one
    that wants it frozen in place (``approach_entity``) leaves it as-is."""
    _crafter_disable_neglect_and_spawn_balance(env)
    world = env._world
    keep_set = set(keep)
    for obj in list(world.objects):
        if obj is env._player or obj in keep_set:
            continue
        world.remove(obj)
    for obj in keep_set:
        obj.update = lambda: None


def _crafter_clear_and_neutralize(env: Any) -> None:
    """Shared body of ``_crafter_clear_walk_corridor``'s per-reset setup:
    remove wildlife, then clear a 3-wide corridor of terrain in the
    MOVE_UP direction (negative y) from the player to the top edge so the
    agent doesn't get stuck on trees/stone. Width of 3 (player column +/- 1)
    gives margin for the player's collision box without flattening the
    whole world."""
    _crafter_remove_wildlife(env)
    px, py = int(env._player.pos[0]), int(env._player.pos[1])
    world = env._world
    area = world.area
    for x in range(max(0, px - 1), min(area[0], px + 2)):
        for y in range(0, py + 1):
            world[(x, y)] = "grass"


def _crafter_clear_walk_corridor(program: Any) -> None:
    """Clear a corridor of terrain in the MOVE_UP direction (negative y) so
    the walk_forward agent doesn't get stuck on trees/stone, and remove
    wildlife. Registered as a post-reset hook (issue #202,
    ``CrafterWorld.set_post_reset_hook``) rather than applied after calling
    ``program.reset()``: each seed generates its own fresh world (this isn't
    a frozen scenario), so the edit must land *before* that seed's reset
    ever publishes its first frame -- applying it afterward left the
    unedited world-generation state (incidental wildlife included) as the
    episode's already-published first recorded frame."""
    program.set_post_reset_hook(_crafter_clear_and_neutralize)


def _crafter_box_in(env: Any) -> None:
    """Wall the player in on all four sides with stone -- every MOVE_*
    attempt is blocked, so only ``facing`` changes (issue #90's discrete
    ``turn``)."""
    _crafter_remove_wildlife(env)
    x, y = int(env._player.pos[0]), int(env._player.pos[1])
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        env._world[(x + dx, y + dy)] = "stone"


def _crafter_box_in_player(program: Any) -> None:
    """Registered as a post-reset hook (issue #202) so each seed still
    generates its own fresh surrounding world, boxed in on all four sides --
    unlike the previous ``freeze_reset()`` approach, which pinned every
    train *and* holdout episode to the exact same single generated world
    (only the box itself, not the background, needs to be seed-invariant)."""
    program.set_post_reset_hook(_crafter_box_in)


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
_CRAFTER_MOVE_LEFT = Action("MOVE_LEFT")


def _crafter_safe_corridor_length(world_size: int, margin: int = 2) -> int:
    """Ticks a constant-``MOVE_UP`` walk can run before the player -- always
    spawned at the exact world center (``crafter.Env.reset``) -- would reach
    the world's top edge (``y=0``), minus a small safety margin. Bounds
    ``walk_forward_short`` (issue #202) so it ends *before* the agent piles
    up an accidental stationary tail against the boundary -- the bug the
    original unbounded ``walk_forward`` had (a constant walk plateaus at the
    edge/an obstacle and records a long stationary tail for the remainder
    of the episode)."""
    return max(1, world_size // 2 - margin)


def _crafter_walk_forward_short(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    return ScenarioRecording(
        policy=ConstantActionPolicy(_CRAFTER_MOVE_UP, seed=seed),
        scene_setup=_crafter_clear_walk_corridor,
        episode_ticks=min(cfg.episode_ticks, _crafter_safe_corridor_length(cfg.world_size)),
    )


#: ``blocked_forward``'s fixed phase lengths (issue #202): continuing to
#: push into the wall for ``_BLOCKED_FORWARD_BLOCKED_HOLD`` ticks past
#: collision is the explicitly bounded blocked phase (well within the
#: "useful training data" 4-8 frame range the issue calls out, as opposed to
#: an accidental 70-frame stationary tail); the final
#: ``_BLOCKED_FORWARD_RECOVER_TICKS`` is the direction-change/recovery
#: action. Collision distance itself is *not* fixed -- see
#: ``_BLOCKED_FORWARD_TRAIN_COLLISION_DISTANCES`` below: the player always
#: spawns at the exact world center regardless of seed, so a *fixed*
#: distance made every train and holdout episode collide at the identical
#: point and record byte-identical episodes.
_BLOCKED_FORWARD_BLOCKED_HOLD = 6
_BLOCKED_FORWARD_RECOVER_TICKS = 6
#: Disjoint by construction (issue #202) -- see ``_split_parameter``.
_BLOCKED_FORWARD_TRAIN_COLLISION_DISTANCES = (4, 6, 8, 10)
_BLOCKED_FORWARD_HOLDOUT_COLLISION_DISTANCES = (5, 7, 9, 11)


def _crafter_setup_blocked_corridor(collision_distance: int, env: Any) -> None:
    _crafter_remove_wildlife(env)
    x, y = int(env._player.pos[0]), int(env._player.pos[1])
    collision_y = y - collision_distance
    # Clear enough room around the collision point for the recovery phase's
    # sideways move too, not just the forward approach -- the recovery
    # action must reliably succeed regardless of what world generation put
    # there for this seed.
    clear_radius = max(collision_distance, _BLOCKED_FORWARD_RECOVER_TICKS) + 3
    _crafter_clear_terrain(env._world, x, collision_y, radius=clear_radius)
    env._world[(x, collision_y)] = "stone"


def _crafter_blocked_forward(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    """Movement onset, continuation, and intentional collision (walking
    ``MOVE_UP`` into a wall placed ``collision_distance`` tiles ahead), then
    a bounded blocked phase (still pushing ``MOVE_UP``), then a direction
    change / recovery action (``MOVE_LEFT`` into cleared space) -- issue
    #202's motion-transition scenario, distinguishing a short, explicitly
    labelled blocked phase (useful training data) from an accidental long
    stationary tail (not). ``collision_distance`` is drawn from a
    train/holdout-disjoint tuple, not fixed: the player always spawns at
    the exact world center, so a fixed distance would make every episode
    collide at the identical point and record byte-identical episodes
    regardless of seed."""
    collision_distance = _split_parameter(
        seed, cfg,
        _BLOCKED_FORWARD_TRAIN_COLLISION_DISTANCES, _BLOCKED_FORWARD_HOLDOUT_COLLISION_DISTANCES,
    )

    def scene_setup(program: Any) -> None:
        # Registered as a post-reset hook (issue #202, mirrors
        # ``_crafter_clear_walk_corridor``) so each seed's fresh world is
        # edited before its first frame is ever published, not after.
        program.set_post_reset_hook(
            lambda env: _crafter_setup_blocked_corridor(collision_distance, env)
        )

    policy = ScriptedSequencePolicy(
        [
            (_CRAFTER_MOVE_UP, collision_distance + _BLOCKED_FORWARD_BLOCKED_HOLD),
            (_CRAFTER_MOVE_LEFT, _BLOCKED_FORWARD_RECOVER_TICKS),
        ]
    )
    episode_ticks = collision_distance + _BLOCKED_FORWARD_BLOCKED_HOLD + _BLOCKED_FORWARD_RECOVER_TICKS
    return ScenarioRecording(policy=policy, scene_setup=scene_setup, episode_ticks=episode_ticks)


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
    distance = _split_parameter(
        seed, cfg, _APPROACH_ENTITY_TRAIN_DISTANCES, _APPROACH_ENTITY_HOLDOUT_DISTANCES
    )

    def _setup_approach(env: Any) -> None:
        import crafter as crafter_pkg

        x, y = int(env._player.pos[0]), int(env._player.pos[1])
        target = (x, y - distance)
        _crafter_clear_terrain(env._world, x, y - distance // 2, radius=distance + 3)
        cow = crafter_pkg.objects.Cow(env._world, target)
        env._world.add(cow)
        # Remove whatever wildlife world generation happened to spawn and
        # freeze the scripted cow in place -- an entity scenario's recorded
        # population must be exactly its own scripted entity, nothing else
        # (issue #202).
        _crafter_freeze_scripted_entities(env, keep=[cow])

    def scene_setup(program: Any) -> None:
        # Registered as a post-reset hook (issue #202) so each seed's fresh
        # world is edited before its first frame is ever published: applying
        # the edit only after ``program.reset()`` returned left whatever
        # wildlife world generation happened to spawn as the episode's
        # already-published first recorded frame.
        program.set_post_reset_hook(_setup_approach)

    return ScenarioRecording(policy=ConstantActionPolicy(_CRAFTER_MOVE_UP), scene_setup=scene_setup)


#: Disjoint by construction (issue #202) -- how many cells past the
#: egocentric view radius the scripted mob walks before returning, per
#: split (added to ``grid_radius + 3`` at use).
_CRAFTER_OBJECT_PERMANENCE_TRAIN_HIDDEN_DELTAS = (0, 2, 4, 6)
_CRAFTER_OBJECT_PERMANENCE_HOLDOUT_HIDDEN_DELTAS = (1, 3, 5, 7)


def _crafter_object_permanence(seed: int, cfg: NurseryConfig) -> ScenarioRecording:
    phase_ticks = max(5, cfg.episode_ticks // 3)
    close = 2
    hidden_delta = _split_parameter(
        seed, cfg,
        _CRAFTER_OBJECT_PERMANENCE_TRAIN_HIDDEN_DELTAS,
        _CRAFTER_OBJECT_PERMANENCE_HOLDOUT_HIDDEN_DELTAS,
    )
    hidden = CrafterConfig().grid_radius + 3 + hidden_delta

    def _setup_occlusion(env: Any) -> None:
        import crafter as crafter_pkg

        x, y = int(env._player.pos[0]), int(env._player.pos[1])
        _crafter_clear_terrain(env._world, x, y, radius=hidden + 3)
        cow = crafter_pkg.objects.Cow(env._world, (x + close, y))
        env._world.add(cow)
        _crafter_freeze_scripted_entities(env, keep=[cow])
        path = [(x + d, y) for d in _crafter_occlusion_distances(close, hidden, phase_ticks)]
        _crafter_script_mob_path(cow, path)

    def scene_setup(program: Any) -> None:
        # Registered as a post-reset hook (issue #202), like
        # ``approach_entity``, rather than ``freeze_reset()``: freezing
        # pinned every seed's object_permanence episode to the exact same
        # generated background world (only the occlusion depth varied),
        # which is exactly the "one generated world across all seeds" bug
        # this issue calls out.
        program.set_post_reset_hook(_setup_occlusion)

    return ScenarioRecording(
        policy=NullPolicy(),
        scene_setup=scene_setup,
        episode_ticks=phase_ticks * 3,
    )


CRAFTER_SCENARIOS: Dict[str, NurseryScenario] = {
    "walk_forward_short": NurseryScenario(
        "walk_forward_short",
        "constant MOVE_UP over varied terrain seeds, bounded to end before the "
        "player reaches the world boundary -- a short optical-flow canary "
        "(issue #202; Crafter port of Minecraft's walk_forward, re-scoped "
        "after the unbounded version was found to plateau at the boundary and "
        "record a long accidental stationary tail).",
        _crafter_walk_forward_short,
        min_blocks_per_tick=0.01,
        min_unique_frame_fraction=0.05,
        # Bounded to the safe corridor length precisely so it should almost
        # never collide; a handful of trailing stationary transitions would
        # still be an early regression signal (an off-by-one in the corridor
        # bound, or a seed with terrain the corridor-clear didn't cover).
        min_moving_transition_fraction=0.5,
        max_longest_stationary_tail=2,
    ),
    "blocked_forward": NurseryScenario(
        "blocked_forward",
        "movement onset, continuation and intentional collision with a fixed "
        "wall, a bounded blocked phase, then a direction-change recovery -- "
        "issue #202's motion-transition scenario: an explicitly labelled few-"
        "tick blocked phase is useful training data, unlike an accidental "
        "long stationary tail.",
        _crafter_blocked_forward,
        # Roughly a third of transitions are the forward approach + the
        # recovery move; the rest is the deliberately bounded blocked hold.
        min_moving_transition_fraction=0.2,
        max_longest_stationary_run=_BLOCKED_FORWARD_BLOCKED_HOLD + 2,
        # The recovery phase should move the agent almost immediately; a
        # long trailing stationary run means the recovery action failed to
        # actually unblock it.
        max_longest_stationary_tail=2,
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
        min_moving_transition_fraction=scenario.min_moving_transition_fraction,
        min_semantic_change_fraction=scenario.min_semantic_change_fraction,
        max_longest_stationary_tail=scenario.max_longest_stationary_tail,
        max_longest_stationary_run=scenario.max_longest_stationary_run,
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


# --------------------------------------------------------------------------- sample balance (issue #202)
#
# The joint training pool spans several scripted scenarios whose class
# balance is an accident of how many ticks each scenario happens to record,
# not a deliberate choice: the audit that produced this issue found 505/700
# walk-forward transitions semantically unchanged and ~84% of joint frames
# containing a visible cow. This section computes per-transition labels
# (semantic change, coarse movement state, entity presence) straight from
# the recorded stream logs and a balanced sampling weight from them --
# reported in the run report (acceptance criterion: "the run report prints
# per-scenario sample counts and event balance"), not silently folded into
# training by duplicating frames ("prefer additional seeds and scenario
# parameter variants" -- the issue is explicit that balance must not come
# from exact-frame duplication).


@dataclass
class TransitionLabel:
    """One frame-to-frame transition's sampling labels."""

    scenario: str
    semantic_changed: bool
    #: "continuing" (position changed) / "turning" (facing changed, position
    #: didn't) / "blocked" (neither changed).
    movement_state: str
    #: "absent" / "present" / "entering" / "leaving", based on whether any
    #: cell classified ``"entity"`` in ``programs.crafter.streams.FRAME_LEGEND``
    #: appears in the semantic grid (Crafter only -- worlds without a
    #: semantic-grid stream report "absent" for every transition, a
    #: documented simplification, not a claim that no entity exists).
    entity_state: str


def _session_is_crafter(session_dir: str) -> bool:
    """Whether ``session_dir``'s recording came from ``CrafterWorld`` --
    ``vision.frame.grid`` exists on Minecraft too, but with a completely
    different (block-type) id vocabulary, so entity detection via
    ``programs.crafter.streams.FRAME_LEGEND`` must not run against it."""
    path = os.path.join(session_dir, "session.json")
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as fh:
        metadata = json.load(fh)
    return "crafter" in (metadata.get("program_tags") or [])


def compute_scenario_transition_labels(
    session_dir: str, episode_id: str, scenario: str
) -> List[TransitionLabel]:
    """Per-transition sampling labels for one recorded episode, read
    straight from its stream log the same tick-grouped way
    ``measure_recording_quality`` does. Entity presence is detected via
    ``vision.frame.grid``'s semantic ids, so it is only meaningful for
    Crafter recordings (see ``_session_is_crafter``); non-Crafter
    transitions always report ``entity_state="absent"``, a documented
    simplification rather than a claim that no entity exists."""
    detect_entities = _session_is_crafter(session_dir)
    position_at_frame: List[Optional[tuple]] = []
    facing_at_frame: List[Optional[tuple]] = []
    entity_present_at_frame: List[bool] = []
    grid_key_at_frame: List[Optional[tuple]] = []
    known_position: Optional[tuple] = None
    known_facing: Optional[tuple] = None
    known_entity_present = False
    known_grid_key: Optional[tuple] = None

    def flush_tick(records: List[Dict[str, Any]]) -> None:
        nonlocal known_position, known_facing, known_entity_present, known_grid_key
        has_pixel = False
        for record in records:
            stream_id = record.get("stream_id")
            if stream_id == "vision.frame.pixels":
                has_pixel = True
            elif stream_id == "vision.frame.grid":
                grid = record.get("payload")
                if isinstance(grid, list):
                    known_grid_key = tuple(tuple(row) for row in grid)
                    if detect_entities:
                        known_entity_present = any(
                            FRAME_LEGEND.get(cell) == "entity" for row in grid for cell in row
                        )
            elif stream_id == "spatial.position":
                payload = record.get("payload") or {}
                known_position = (payload.get("x"), payload.get("z", payload.get("y")))
            elif stream_id == "spatial.facing":
                payload = record.get("payload") or {}
                known_facing = (payload.get("x"), payload.get("y"))
        if has_pixel:
            position_at_frame.append(known_position)
            facing_at_frame.append(known_facing)
            entity_present_at_frame.append(known_entity_present)
            grid_key_at_frame.append(known_grid_key)

    streams_path = os.path.join(session_dir, f"{episode_id}.streams.jsonl")
    tick_timestamp: Any = object()
    tick_records: List[Dict[str, Any]] = []
    with open(streams_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            timestamp = record.get("timestamp")
            if tick_records and timestamp != tick_timestamp:
                flush_tick(tick_records)
                tick_records = []
            tick_timestamp = timestamp
            tick_records.append(record)
    if tick_records:
        flush_tick(tick_records)

    labels: List[TransitionLabel] = []
    for i in range(len(position_at_frame) - 1):
        moved = (
            position_at_frame[i] is not None
            and position_at_frame[i + 1] is not None
            and position_at_frame[i] != position_at_frame[i + 1]
        )
        turned = (
            not moved
            and facing_at_frame[i] is not None
            and facing_at_frame[i + 1] is not None
            and facing_at_frame[i] != facing_at_frame[i + 1]
        )
        movement_state = "continuing" if moved else ("turning" if turned else "blocked")

        was_present, now_present = entity_present_at_frame[i], entity_present_at_frame[i + 1]
        if was_present and now_present:
            entity_state = "present"
        elif was_present and not now_present:
            entity_state = "leaving"
        elif not was_present and now_present:
            entity_state = "entering"
        else:
            entity_state = "absent"

        grid_a, grid_b = grid_key_at_frame[i], grid_key_at_frame[i + 1]
        semantic_changed = grid_a is not None and grid_b is not None and grid_a != grid_b

        labels.append(
            TransitionLabel(
                scenario=scenario, semantic_changed=semantic_changed,
                movement_state=movement_state, entity_state=entity_state,
            )
        )
    return labels


def summarize_sample_balance(
    labels_by_scenario: Dict[str, Sequence[TransitionLabel]]
) -> Dict[str, Any]:
    """Per-scenario sample counts and event balance -- printed in the
    nursery run report (issue #202's acceptance criterion)."""
    summary: Dict[str, Any] = {}
    for scenario, labels in labels_by_scenario.items():
        total = len(labels)
        summary[scenario] = {
            "samples": total,
            "semantic_changed_fraction": (
                sum(1 for l in labels if l.semantic_changed) / total if total else 0.0
            ),
            "movement_state_counts": {
                state: sum(1 for l in labels if l.movement_state == state)
                for state in ("continuing", "blocked", "turning")
            },
            "entity_state_counts": {
                state: sum(1 for l in labels if l.entity_state == state)
                for state in ("absent", "present", "entering", "leaving")
            },
        }
    return summary


def log_sample_balance(summary: Dict[str, Any]) -> None:
    """Print ``summarize_sample_balance``'s result -- the run report's
    per-scenario sample counts and event balance."""
    for scenario, stats in summary.items():
        log.info(
            "sample balance  scenario=%s  samples=%d  semantic_changed=%.1f%%  "
            "movement=%s  entity=%s",
            scenario, stats["samples"], 100.0 * stats["semantic_changed_fraction"],
            stats["movement_state_counts"], stats["entity_state_counts"],
        )


def balanced_transition_weights(
    labels: Sequence[TransitionLabel],
    *,
    max_stationary_fraction: float = 0.25,
    entity_balance_target: float = 0.5,
) -> List[float]:
    """Per-transition sampling weight implementing the recommended starting
    balance policy (issue #202): no more than ``max_stationary_fraction``
    blocked/stationary transitions, ~``entity_balance_target``/
    ``1 - entity_balance_target`` entity-present/-absent, oversample
    entry/exit events, and weight every scenario equally regardless of how
    many transitions it happens to contribute (sample scenarios uniformly
    before sampling a transition).

    Returns per-transition weights (not probabilities), suitable for a
    weighted sampler (e.g. ``torch.utils.data.WeightedRandomSampler``) drawn
    *without* replacement up to ``len(labels)`` or fewer -- this function
    does not duplicate frames; achieving genuine balance beyond what a
    reweighted draw can reach means recording additional seeds/scenario
    parameter variants, not resampling exact duplicates.
    """
    n = len(labels)
    if n == 0:
        return []

    #: Applied as a sequence of rebalancing passes, each targeting a
    #: fraction of the *current* running total (not a fraction of the raw
    #: transition count) -- composing stages this way means each stage's
    #: guarantee holds regardless of what an earlier stage already did to
    #: the weights, rather than every stage independently targeting the raw
    #: count and the results only being correct when combined by luck.
    weights: List[float] = [1.0] * n

    #: Entry/exit events are individually rare; oversample them relative to
    #: a flat entity-state split rather than splitting the "present" budget
    #: with "entering"/"leaving" too -- a class this rare would otherwise be
    #: swamped by the majority "absent"/"present" states.
    entry_exit_bonus = 3.0
    for i, label in enumerate(labels):
        if label.entity_state in ("entering", "leaving"):
            weights[i] *= entry_exit_bonus

    # Entity present/absent balance: scale each class's current total to
    # its target share of the (present + absent) total -- entering/leaving
    # already got their own bonus above and sit outside this split.
    present_idx = [i for i, l in enumerate(labels) if l.entity_state == "present"]
    absent_idx = [i for i, l in enumerate(labels) if l.entity_state == "absent"]
    for idx, target_fraction in ((present_idx, entity_balance_target),
                                  (absent_idx, 1.0 - entity_balance_target)):
        other_idx = (absent_idx if idx is present_idx else present_idx)
        current = sum(weights[i] for i in idx)
        other = sum(weights[i] for i in other_idx)
        if current <= 0 or not other_idx:
            continue
        target_total = (target_fraction / (1.0 - target_fraction)) * other if target_fraction < 1.0 else current
        scale = target_total / current
        for i in idx:
            weights[i] *= scale

    # Stationary/blocked cap: scale the blocked class's current total down
    # so it is at most `max_stationary_fraction` of the (blocked + other)
    # total -- never scaled up, since being *under* the cap is fine.
    blocked_idx = [i for i, l in enumerate(labels) if l.movement_state == "blocked"]
    if blocked_idx:
        blocked_total = sum(weights[i] for i in blocked_idx)
        other_total = sum(weights) - blocked_total
        if blocked_total > 0 and other_total > 0:
            target_blocked_total = (
                max_stationary_fraction / (1.0 - max_stationary_fraction)
            ) * other_total
            scale = min(1.0, target_blocked_total / blocked_total)
            for i in blocked_idx:
                weights[i] *= scale

    # Scenario-uniform (final pass): rescale each scenario's current total
    # to be equal, preserving every within-scenario ratio set above (every
    # label in a scenario is divided by that same scenario's current total).
    scenario_idx: Dict[str, List[int]] = {}
    for i, label in enumerate(labels):
        scenario_idx.setdefault(label.scenario, []).append(i)
    n_scenarios = len(scenario_idx)
    final_weights = [0.0] * n
    for idx in scenario_idx.values():
        total = sum(weights[i] for i in idx)
        if total <= 0:
            continue
        for i in idx:
            final_weights[i] = weights[i] / total / n_scenarios
    return final_weights


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


def _scenario_program_config(
    cfg: NurseryConfig, episode_ticks: int, extra: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve one scenario recording's base ``program_config``, translating
    ``NurseryConfig.world_size`` to whatever key the selected World actually
    understands (issue #202): Minecraft's ``SurvivalBoxConfig`` has its own
    ``world_size`` field, but Crafter has none -- ``CrafterConfig.from_dict``
    silently drops it, so a Crafter program must instead receive ``area``
    (see ``programs.crafter.config.area_from_world_size``). A notebook's
    requested 128-unit world previously recorded, un-erroring, against
    Crafter's default 64x64 ``area``."""
    program_config: Dict[str, Any] = {"episode_ticks": episode_ticks}
    if cfg.world == "crafter":
        program_config["area"] = area_from_world_size(cfg.world_size)
    else:
        program_config["world_size"] = cfg.world_size
    program_config.update(extra)
    return program_config


def _record_scenario_episode(
    record_dir: str, session_id: str, seed: int, scenario: NurseryScenario, cfg: NurseryConfig
) -> str:
    with span("nursery.record.episode", session=session_id, scenario=scenario.name,
              seed=seed, ticks=cfg.episode_ticks):
        log.info("recording %s  seed=%d  ticks=%d", session_id, seed, cfg.episode_ticks)
        recording = scenario.build(seed, cfg)
        episode_ticks = recording.episode_ticks or cfg.episode_ticks
        program_config = _scenario_program_config(
            cfg, episode_ticks, recording.program_config_extra
        )
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
        if recording.scene_setup is not None and (
            cfg.world == "crafter" or cfg.backend == "simulated"
        ):
            recording.scene_setup(program)
        try:
            CognitiveRuntime(program=program, policy=recording.policy, config=runtime_config).run()
        finally:
            close = getattr(program, "close", None)
            if callable(close):
                close()
        # Counted only once the episode is actually on disk: in a failed run
        # -- the trace where this counter is most worth reading -- an
        # increment before the runtime would overstate the usable recordings
        # by the very episode that killed the run.
        trace_counter("episodes_recorded")
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

    with span("nursery.record", scenario=scenario_name,
              train_seeds=len(cfg.train_seeds), holdout_seeds=len(cfg.holdout_seeds)):
        train_sessions = [
            _record_scenario_episode(
                record_dir, f"nursery-{scenario_name}-train-{seed}", seed, scenario, cfg
            )
            for seed in cfg.train_seeds
        ]
        holdout_sessions = [
            _record_scenario_episode(
                record_dir, f"nursery-{scenario_name}-holdout-{seed}", seed, scenario, cfg
            )
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
            trace_event("nursery.quality_gate.failed",
                        scenario=scenario_name, issues=issues)
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

    if cfg.split_overlap_gate:
        overlap_report = audit_split_overlap(
            train_sessions, holdout_sessions,
            max_corresponding_frame_fraction=cfg.max_corresponding_frame_fraction,
        )
        if overlap_report.issues:
            trace_event("nursery.split_overlap_gate.failed",
                        scenario=scenario_name, issues=overlap_report.issues)
            raise ValueError(
                f"nursery scenario {scenario_name!r}: recorded holdout data leaks into "
                "training:\n  - " + "\n  - ".join(overlap_report.issues)
                + " (split_overlap_gate=False skips this check.)"
            )
        log.info("split-overlap gate passed")

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
    with span("nursery.train.pixel_encoder",
              samples=len(train_dataset), epochs=cfg.epochs, lr=cfg.lr,
              warm_start=initial_model is not None):
        model, pretraining_stats = train_pixel_encoder_pretraining(
            train_dataset, visual_config, initial_model=initial_model,
        )
    log.info("pixel encoder training complete")

    consistency_stats: Dict[str, List[float]] = {}
    if cfg.consistency_epochs > 0:
        with span("nursery.train.horizon_consistency", epochs=cfg.consistency_epochs):
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
    with span("nursery.evaluate", scenario=scenario_name, sessions=len(holdout_sessions)):
        horizon_metrics = evaluate_ego_motion_holdout(
            model, holdout_sessions, horizon_frames, ssim_window=cfg.ssim_window
        )
        rollout_health = evaluate_rollout_health(model, holdout_sessions, horizon_frames)
    for h, metrics in horizon_metrics.items():
        log.info("  t+%d: model_mse=%.4f  copy_last_mse=%.4f  beats=%s",
                 h, metrics.get("model_mse", 0), metrics.get("copy_last_mse", 0),
                 metrics.get("beats_copy_last", "?"))
    _trace_horizon_metrics("scenario", scenario_name, {"horizons": horizon_metrics,
                                                       "rollout_health": rollout_health})

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

    canonical_recording = scenario.build(cfg.train_seeds[0], cfg)
    resolved_program_config = _scenario_program_config(
        cfg, canonical_recording.episode_ticks or cfg.episode_ticks,
        canonical_recording.program_config_extra,
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
        resolved_program_config=resolved_program_config,
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
    #: Per-scenario sample counts and event balance over the training pool
    #: (issue #202's acceptance criterion) -- see ``summarize_sample_balance``.
    sample_balance: Dict[str, Any] = field(default_factory=dict)


def _final_loss(training_stats: Dict[str, Any]) -> Optional[float]:
    final = training_stats.get("final_total_loss")
    if final is not None:
        return float(final)
    totals = (training_stats.get("loss_curves") or {}).get("total_loss") or []
    return float(totals[-1]) if totals else None


def _trace_horizon_metrics(split: str, scenario: str, metrics: Dict[str, Any]) -> None:
    """Push one evaluation's per-horizon numbers into the trace as metric
    series, so ``ccr trace show`` can answer "did it beat copy-last-frame?"
    without re-reading the report JSON -- and so a run that dies during a
    later scenario still has the earlier scenarios' verdicts on disk."""
    for horizon, entry in (metrics.get("horizons") or {}).items():
        trace_metrics(
            **{
                f"eval/{split}/{scenario}/t+{horizon}/model_mse": entry.get("model_mse"),
                f"eval/{split}/{scenario}/t+{horizon}/copy_last_mse": entry.get("copy_last_mse"),
                f"eval/{split}/{scenario}/t+{horizon}/beats_copy_last": float(
                    bool(entry.get("beats_copy_last"))
                ),
            }
        )
    health = metrics.get("rollout_health") or {}
    trace_event(
        "nursery.evaluate.result", split=split, scenario=scenario,
        horizons={
            str(h): {
                "model_mse": e.get("model_mse"),
                "copy_last_mse": e.get("copy_last_mse"),
                "beats_copy_last": e.get("beats_copy_last"),
            }
            for h, e in (metrics.get("horizons") or {}).items()
        },
        frozen_rollout=health.get("frozen_rollout"),
    )


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

    log.info("=== nursery joint: train=%s  holdout=%s  world=%s ===",
             train_names, holdout_names, cfg.world)
    trace_event(
        "nursery.joint.plan",
        train_scenarios=train_names, holdout_scenarios=holdout_names, world=cfg.world,
        train_seeds=list(cfg.train_seeds), holdout_seeds=list(cfg.holdout_seeds),
        horizons=list(cfg.horizons), epochs=model_cfg.epochs, backbone=model_cfg.backbone,
        training_objective=model_cfg.training_objective,
    )

    train_sessions: Dict[str, List[str]] = {}
    eval_sessions: Dict[str, List[str]] = {}
    with span("nursery.record",
              scenarios=len(train_names) + len(holdout_names),
              episodes=(len(train_names) * (len(cfg.train_seeds) + len(cfg.holdout_seeds))
                        + len(holdout_names) * len(cfg.holdout_seeds))) as recording_span:
        for name in train_names:
            scenario = NURSERY_SCENARIOS[name]
            with span("nursery.record.scenario", scenario=name, split="train+holdout"):
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
            with span("nursery.record.scenario", scenario=name, split="zero-shot"):
                eval_sessions[name] = [
                    _record_scenario_episode(
                        record_dir, f"nursery-{name}-holdout-{seed}", seed, scenario, cfg
                    )
                    for seed in cfg.holdout_seeds
                ]
        recording_span.set(
            train_sessions=sum(len(v) for v in train_sessions.values()),
            eval_sessions=sum(len(v) for v in eval_sessions.values()),
        )

    if cfg.data_quality_gate:
        with span("nursery.quality_gate") as gate_span:
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
            gate_span.set(issues=len(issues))
            if issues:
                # The gate's verdict belongs in the trace: a failed run whose
                # only record is a traceback can't be compared against the
                # run that passed.
                trace_event("nursery.quality_gate.failed", issues=issues)
                raise ValueError(
                    "nursery joint run: recorded data fails the quality gate:\n  - "
                    + "\n  - ".join(issues)
                )
            log.info("quality gate passed")

    if cfg.split_overlap_gate:
        with span("nursery.split_overlap_gate") as overlap_span:
            overlap_issues: List[str] = []
            for name in train_names:
                overlap_issues += audit_split_overlap(
                    train_sessions[name], eval_sessions[name],
                    max_corresponding_frame_fraction=cfg.max_corresponding_frame_fraction,
                ).issues
            overlap_span.set(issues=len(overlap_issues))
            if overlap_issues:
                trace_event("nursery.split_overlap_gate.failed", issues=overlap_issues)
                raise ValueError(
                    "nursery joint run: recorded holdout data leaks into training:\n  - "
                    + "\n  - ".join(overlap_issues)
                )
            log.info("split-overlap gate passed")

    with span("nursery.sample_balance"):
        labels_by_scenario: Dict[str, List[TransitionLabel]] = {}
        for name in train_names:
            labels_by_scenario[name] = [
                label
                for session_dir in train_sessions[name]
                for episode_id in list_episodes(session_dir)
                for label in compute_scenario_transition_labels(session_dir, episode_id, name)
            ]
        sample_balance = summarize_sample_balance(labels_by_scenario)
        log_sample_balance(sample_balance)
        trace_event("nursery.sample_balance", balance=sample_balance)

    # Pin the vocabulary to the full action space (plus NULL): a held-out
    # scenario may issue actions no training scenario used, and zero-shot
    # evaluation must be able to encode them (their embeddings are simply
    # untrained).
    vocabulary = _action_keys_for_world(cfg.world)

    all_train_dirs = [d for name in train_names for d in train_sessions[name]]
    with span("nursery.dataset", sessions=len(all_train_dirs)) as dataset_span:
        dataset = build_action_sequence_dataset(all_train_dirs, action_keys=vocabulary)
        if len(dataset) == 0:
            raise ValueError("nursery joint run: no frame transitions in the training sessions")
        ticks_per_frame = dataset.ticks_per_frame
        horizon_frames = horizons_ticks_to_frames(cfg.horizons, ticks_per_frame)
        dataset_span.set(
            transitions=len(dataset), ticks_per_frame=ticks_per_frame,
            horizon_frames=horizon_frames, action_vocabulary=len(vocabulary),
        )
    log.info("horizons (ticks): %s -> frames: %s  (%.2f ticks/frame)",
             list(cfg.horizons), horizon_frames, ticks_per_frame)

    with span("nursery.train", epochs=model_cfg.epochs, backbone=model_cfg.backbone,
              objective=model_cfg.training_objective) as train_span:
        model, training_stats = train_action_world_model(dataset, model_cfg)
        train_span.set(final_total_loss=_final_loss(training_stats))

    scenario_metrics: Dict[str, Dict[str, Any]] = {}
    with span("nursery.evaluate.in_distribution", scenarios=len(train_names)):
        for name in train_names:
            with span("nursery.evaluate.scenario", scenario=name, split="in_distribution"):
                holdout_dataset = build_action_sequence_dataset(
                    eval_sessions[name], action_keys=model.action_keys
                )
                scenario_metrics[name] = evaluate_action_world_model(
                    model, holdout_dataset, horizon_frames,
                    warmup_frames=model_cfg.warmup_frames,
                )
                _trace_horizon_metrics("in_distribution", name, scenario_metrics[name])
    zero_shot_metrics: Dict[str, Dict[str, Any]] = {}
    with span("nursery.evaluate.zero_shot", scenarios=len(holdout_names)):
        for name in holdout_names:
            with span("nursery.evaluate.scenario", scenario=name, split="zero_shot"):
                holdout_dataset = build_action_sequence_dataset(
                    eval_sessions[name], action_keys=model.action_keys
                )
                zero_shot_metrics[name] = evaluate_action_world_model(
                    model, holdout_dataset, horizon_frames,
                    warmup_frames=model_cfg.warmup_frames,
                )
                _trace_horizon_metrics("zero_shot", name, zero_shot_metrics[name])

    with span("nursery.probes"):
        probe_dataset = build_action_sequence_dataset(
            [d for dirs in eval_sessions.values() for d in dirs], action_keys=model.action_keys
        )
        yaw_probe = linear_probe_yaw(model, probe_dataset)
        orientation_probe = linear_probe_orientation(model, probe_dataset)
        representation_diagnostics = representation_collapse_diagnostics(
            model, probe_dataset, config=model_cfg
        )
        trace_event("nursery.probes.result",
                    yaw=yaw_probe, orientation=orientation_probe,
                    representation=representation_diagnostics)

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
        sample_balance=sample_balance,
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
    with span("ablation.train", arm="with_actions", epochs=with_actions_cfg.epochs):
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
    with span("ablation.train", arm="without_actions", epochs=without_actions_cfg.epochs):
        model_without, stats_without = train_action_world_model(
            dataset, without_actions_cfg, initial_model=without_initial,
        )
    if without_actions_checkpoint_path is not None:
        save_action_world_model(without_actions_checkpoint_path, model_without, stats_without)

    holdout_dataset = build_action_sequence_dataset(
        eval_sessions[eval_scenario], action_keys=model_with.action_keys
    )
    with span("ablation.evaluate", scenario=eval_scenario):
        with_actions_metrics = evaluate_action_world_model(
            model_with, holdout_dataset, horizon_frames,
            warmup_frames=base_model_cfg.warmup_frames,
        )
        without_actions_metrics = evaluate_action_world_model(
            model_without, holdout_dataset, horizon_frames,
            warmup_frames=base_model_cfg.warmup_frames,
        )
        _trace_horizon_metrics("ablation_with_actions", eval_scenario, with_actions_metrics)
        _trace_horizon_metrics("ablation_without_actions", eval_scenario, without_actions_metrics)

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
    trace_event("ablation.result", degrades_without_actions=degrades,
                eval_scenario=eval_scenario, comparisons=comparisons)
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
        with span("benchmark.backbone", backbone=name, epochs=base_model_cfg.epochs):
            backbone_cfg = replace(base_model_cfg, backbone=name)
            model, _train_stats = train_action_world_model(dataset, backbone_cfg)
            report = evaluate_action_world_model(
                model, holdout_dataset, horizon_frames,
                warmup_frames=base_model_cfg.warmup_frames,
            )
            metrics[name] = report
            stats[name] = cortex_horizon_statistics(report["per_episode_model_mse"])
            beats_copy_last[name] = {
                h: entry["beats_copy_last"] for h, entry in report["horizons"].items()
            }
            _trace_horizon_metrics(f"backbone/{name}", eval_scenario, report)
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
