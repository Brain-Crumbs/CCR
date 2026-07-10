"""Phase D: entity persistence and the combined novelty stream (issue #27)."""

from __future__ import annotations

import math
import os

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.core.entity_features import (  # noqa: E402
    NEUTRAL_ENTITY_FEATURE,
    entity_feature_vector,
)
from cognitive_runtime.core.entity_tracker import EntityTracker  # noqa: E402
from cognitive_runtime.neural.entity_persistence import (  # noqa: E402
    EntityPersistenceModel,
    EntityPersistenceOutput,
    normalize_gap,
)
from cognitive_runtime.neural import CheckpointCompatibilityError  # noqa: E402
from cognitive_runtime.policies import NullPolicy, ScriptedSurvivalPolicy  # noqa: E402
from cognitive_runtime.policies.neural_entity_persistence import (  # noqa: E402
    NeuralEntityPersistence,
)
from cognitive_runtime.policies.neural_world_model import NeuralWorldModel  # noqa: E402
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox  # noqa: E402
from cognitive_runtime.runtime.config import RuntimeConfig  # noqa: E402
from cognitive_runtime.runtime.loop import NOVELTY_STREAM, CognitiveRuntime  # noqa: E402
from cognitive_runtime.runtime.replay import iter_cognitive_ticks  # noqa: E402
from cognitive_runtime.tools.episode_viewer import view_episode  # noqa: E402
from cognitive_runtime.training.datasets import build_world_model_dataset  # noqa: E402
from cognitive_runtime.training.entity_persistence import (  # noqa: E402
    EntityPersistenceTrainingConfig,
    build_entity_persistence_dataset,
    load_entity_persistence_checkpoint,
    save_entity_persistence_checkpoint,
    train_entity_persistence_model,
)
from cognitive_runtime.training.world_model import (  # noqa: E402
    WorldModelTrainingConfig,
    save_world_model_checkpoint,
    train_world_model,
)


def _install_scripted_mob_path(world, path):
    """Replace one world's mob AI with a scripted per-tick position list, so
    an occlusion/reappearance sequence is fully controlled instead of left to
    (undeterministic-for-our-purposes) zombie pathing.  ``path[i]`` is the
    mob's ``(x, z)`` at ``world.tick == i + 1``; once exhausted, the mob is
    removed.
    """

    def scripted_update_mobs(self, events):
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


def _clear_terrain(world, cx, cz, radius):
    lo_x, hi_x = max(0, cx - radius), min(world.size - 1, cx + radius)
    lo_z, hi_z = max(0, cz - radius), min(world.size - 1, cz + radius)
    for x in range(lo_x, hi_x + 1):
        for z in range(lo_z, hi_z + 1):
            world.terrain[x][z] = "grass"


#: Fixed distance from the agent, along its own row, where the wall sits.
#: Every ``offset`` this test uses is greater than this, so the wall always
#: sits strictly *between* the agent and the mob during the occluded phase
#: rather than coinciding with the mob's own cell (which the line-of-sight
#: raycast never samples -- see ``SurvivalWorld._has_line_of_sight``).
_WALL_OFFSET = 3


def _occlusion_path(offset: float, phase_ticks: int = 10):
    """A "walks behind a block" cycle: visible off-axis, occluded directly
    behind the wall (same z as the agent, beyond it), visible again off-axis
    on the far side -- three ``(x, z)`` waypoints, held for ``phase_ticks``
    each."""
    visible_1 = (offset, offset)
    occluded = (offset, 0.0)
    visible_2 = (offset, -offset)
    return [visible_1] * phase_ticks + [occluded] * phase_ticks + [visible_2] * phase_ticks


def _record_occlusion_session(tmp_path, session_id, *, offset, phase_ticks=10, world_size=48):
    assert offset > _WALL_OFFSET, "offset must clear the wall so the mob truly passes it"
    config = {"episode_ticks": phase_ticks * 3 + 5, "world_size": world_size, "max_mobs": 0}
    runtime_config = RuntimeConfig(
        episodes=1, seed=0, max_ticks_per_episode=phase_ticks * 3 + 5,
        record_dir=str(tmp_path), session_id=session_id, program_config=config,
    )
    program = MinecraftSurvivalBox(config=config)
    world = program._backend.world
    # `CognitiveRuntime._run_episode` calls `program.reset(seed=...)` itself
    # (in-place: terrain/spawn/mobs regenerate on the *same* world instance,
    # see `SurvivalWorld.reset`), which would wipe our wall/mob setup below.
    # Do our own reset now, then make the world's own reset a no-op so the
    # runtime's later call leaves our scripted scene alone; the rest of
    # `program.reset` (buses, reward fn, initial-state republish) still runs.
    world.reset(0)
    ax, az = int(world.x), int(world.z)
    _clear_terrain(world, ax, az, radius=int(offset) + 3)
    wall_x = ax + _WALL_OFFSET
    world.terrain[wall_x][az] = "stone"
    path = _occlusion_path(float(int(offset)), phase_ticks)
    # Positions are agent-relative offsets; shift onto the actual agent spot.
    path = [(ax + dx, az + dz) for dx, dz in path]
    _install_scripted_mob_path(world, path)
    world.reset = lambda seed: None

    runtime = CognitiveRuntime(program=program, policy=NullPolicy(), config=runtime_config)
    runtime.run()
    return os.path.join(str(tmp_path), session_id)


def test_entity_feature_vector_and_neutral_shape():
    vec = entity_feature_vector({"distance": 8.0, "angle": 90.0})
    assert len(vec) == 3
    assert vec[0] == pytest.approx(0.5)
    assert vec[1] == pytest.approx(math.sin(math.radians(90.0)))
    assert len(NEUTRAL_ENTITY_FEATURE) == 3


def test_entity_tracker_detects_occlusion_and_reappearance():
    tracker = EntityTracker(max_gap_ticks=50)
    assert tracker.update([{"id": 1, "distance": 5.0, "angle": 0.0}]) == []
    assert tracker.occluded() == []

    for _ in range(4):
        assert tracker.update([]) == []  # occluded, no reappearance yet
    assert tracker.occluded() == [1]
    assert tracker.state(1).gap_ticks == 4

    reappearances = tracker.update([{"id": 1, "distance": 6.0, "angle": 10.0}])
    assert len(reappearances) == 1
    event = reappearances[0]
    assert event.entity_id == 1
    assert event.gap_ticks == 4
    assert tracker.occluded() == []


def test_entity_persistence_model_forward_shapes_and_checkpoint_metadata():
    torch.manual_seed(0)
    model = EntityPersistenceModel(hidden_dim=8, depth=2)
    out = model(torch.randn(5, 3), torch.rand(5))

    assert isinstance(out, EntityPersistenceOutput)
    assert out.predicted_feature.shape == (5, 3)
    assert out.surprise.shape == (5,)
    assert bool((out.surprise >= 0).all())

    meta = model.checkpoint_metadata()
    assert meta["feature_width"] == 3

    with pytest.raises(ValueError):
        model(torch.randn(5, 2), torch.rand(5))
    with pytest.raises(ValueError):
        model(torch.randn(5, 3), torch.rand(4))


def test_build_entity_persistence_dataset_from_recorded_session(tmp_path):
    session_dir = _record_occlusion_session(tmp_path, "ep-dataset", offset=6)
    dataset = build_entity_persistence_dataset([session_dir])

    assert len(dataset) >= 1
    assert len(dataset.last_features[0]) == 3
    assert len(dataset.target_features[0]) == 3
    assert dataset.gaps[0] > 0
    assert dataset.baseline_mse() > 0.0


def test_entity_persistence_training_beats_baseline_and_checkpoints(tmp_path):
    session_dirs = [
        _record_occlusion_session(tmp_path, f"ep-train-{offset}", offset=offset)
        for offset in (5, 6, 7, 8, 10, 11)
    ]
    dataset = build_entity_persistence_dataset(session_dirs)
    assert len(dataset) >= len(session_dirs)

    model, stats = train_entity_persistence_model(
        dataset,
        EntityPersistenceTrainingConfig(epochs=60, lr=5e-3, batch_size=8, hidden_dim=32, seed=1),
    )

    assert stats["feature_loss_decreased"] is True
    assert stats["beats_forget_baseline"] is True
    assert stats["model_mse"] < stats["baseline_mse"]

    path = os.path.join(str(tmp_path), "entity_persistence.pt")
    metadata = save_entity_persistence_checkpoint(path, model, dataset, stats)
    loaded, loaded_metadata = load_entity_persistence_checkpoint(path)

    assert metadata["modules"]["encoders"]["entity_persistence"]["checkpoint_metadata"][
        "feature_width"
    ] == 3
    assert loaded_metadata["training_stats"]["beats_forget_baseline"] is True
    assert loaded.feature_width == model.feature_width

    with pytest.raises(CheckpointCompatibilityError):
        load_entity_persistence_checkpoint(path, expected_layout_hash="different-layout")


def test_synthetic_occlusion_predicted_latent_beats_forget_immediately_baseline(tmp_path):
    """Acceptance criterion #1: entity walks behind a block; the model's
    predicted entity latent *during* the occlusion gap (not just at
    reappearance) should be closer to the true reappearance state than a
    "forget immediately" baseline that just assumes nothing is there."""
    session_dirs = [
        _record_occlusion_session(tmp_path, f"ep-train2-{offset}", offset=offset)
        for offset in (5, 6, 7, 8, 10, 11, 12)
    ]
    dataset = build_entity_persistence_dataset(session_dirs)
    model, _stats = train_entity_persistence_model(
        dataset,
        EntityPersistenceTrainingConfig(epochs=80, lr=5e-3, batch_size=8, hidden_dim=32, seed=2),
    )
    model.eval()

    holdout_offset = 9
    phase_ticks = 10
    holdout_dir = _record_occlusion_session(
        tmp_path, "ep-holdout", offset=holdout_offset, phase_ticks=phase_ticks
    )

    tracker = EntityTracker(max_gap_ticks=phase_ticks * 3)
    last_feature_before_gap = None
    mid_gap_feature = None
    mid_gap_ticks = None
    true_reappearance_feature = None
    tick = 0
    for _decision, sensory, _motor in iter_cognitive_ticks(holdout_dir, "episode_00000"):
        tick += 1
        entities = None
        for record in sensory:
            if record.get("stream_id") == "vision.entities" and not record.get("elided"):
                entities = record["payload"]
        if entities is None:
            entities = []
        occluded_before = set(tracker.occluded())
        reappearances = tracker.update(entities)
        if tick <= phase_ticks and tracker.state(1) is not None:
            last_feature_before_gap = tracker.state(1).last_feature
        # Halfway through the occlusion phase: capture the tracker's state so
        # we can ask the model "what do you think its feature is right now".
        if tick == phase_ticks + phase_ticks // 2 and 1 in occluded_before | set(tracker.occluded()):
            tracked = tracker.state(1)
            mid_gap_feature = list(tracked.last_feature)
            mid_gap_ticks = tracked.gap_ticks
        for event in reappearances:
            true_reappearance_feature = event.feature_now

    assert last_feature_before_gap is not None
    assert mid_gap_feature is not None
    assert true_reappearance_feature is not None

    with torch.no_grad():
        out = model(
            torch.tensor([mid_gap_feature], dtype=torch.float32),
            torch.tensor([normalize_gap(mid_gap_ticks, model.gap_cap)], dtype=torch.float32),
        )
    predicted = out.predicted_feature[0].tolist()

    def _sq_dist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    model_distance = _sq_dist(predicted, true_reappearance_feature)
    baseline_distance = _sq_dist(NEUTRAL_ENTITY_FEATURE, true_reappearance_feature)
    assert model_distance < baseline_distance


def test_novelty_stream_recorded_and_visible_in_episode_viewer(tmp_path):
    """Acceptance criterion #2: the combined novelty stream is recorded and
    shows up in the episode viewer, backed by a live NeuralEntityPersistence
    + NeuralWorldModel bridge (not just the offline dataset stats)."""
    seed_dirs = [
        _record_occlusion_session(tmp_path, f"ep-seed-{offset}", offset=offset)
        for offset in (5, 6, 7, 8)
    ]
    ep_dataset = build_entity_persistence_dataset(seed_dirs)
    ep_model, ep_stats = train_entity_persistence_model(
        ep_dataset, EntityPersistenceTrainingConfig(epochs=10, lr=5e-3, batch_size=8)
    )
    ep_path = os.path.join(str(tmp_path), "ep_model.pt")
    save_entity_persistence_checkpoint(ep_path, ep_model, ep_dataset, ep_stats)

    scripted_dir_config = {"episode_ticks": 200, "world_size": 24, "max_mobs": 3}
    scripted_runtime_config = RuntimeConfig(
        episodes=1, seed=1, max_ticks_per_episode=200,
        record_dir=str(tmp_path), session_id="wm-seed", program_config=scripted_dir_config,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=scripted_dir_config),
        policy=ScriptedSurvivalPolicy(seed=1),
        config=scripted_runtime_config,
    ).run()
    wm_dataset = build_world_model_dataset(
        [os.path.join(str(tmp_path), "wm-seed")], max_samples=32
    )
    wm_model, wm_stats = train_world_model(
        wm_dataset, WorldModelTrainingConfig(epochs=2, lr=1e-3, batch_size=16, hidden_dim=16, depth=1)
    )
    wm_path = os.path.join(str(tmp_path), "wm_model.pt")
    save_world_model_checkpoint(wm_path, wm_model, wm_dataset, wm_stats)

    world_model = NeuralWorldModel(wm_path, action_keys=wm_dataset.action_keys)
    entity_persistence = NeuralEntityPersistence(ep_path)

    run_config = {"episode_ticks": 300, "world_size": 24, "max_mobs": 3,
                  "day_length": 400, "start_time": 200}
    run_runtime_config = RuntimeConfig(
        episodes=1, seed=42, max_ticks_per_episode=300,
        record_dir=str(tmp_path), session_id="novelty-run", program_config=run_config,
    )
    summaries = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=run_config),
        policy=ScriptedSurvivalPolicy(seed=42),
        config=run_runtime_config,
        world_model=world_model,
        entity_persistence=entity_persistence,
    ).run()

    assert summaries[0].avg_novelty is not None

    session_dir = os.path.join(str(tmp_path), "novelty-run")
    novelty_events = 0
    for _decision, sensory, _motor in iter_cognitive_ticks(session_dir, summaries[0].episode_id):
        for record in sensory:
            if record.get("stream_id") == NOVELTY_STREAM and not record.get("elided"):
                novelty_events += 1
    assert novelty_events > 0

    rendered = view_episode(session_dir, summaries[0].episode_id, tail=5)
    assert "avg_novelty" in rendered
    assert NOVELTY_STREAM in rendered
    assert "novelty=" in rendered
