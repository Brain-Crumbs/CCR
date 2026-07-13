"""Action-conditioned recurrent world model (phases 1-3 of
docs/nursery-turn-in-place-analysis.md)."""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.training.action_world_model import (  # noqa: E402
    ActionWorldModelConfig,
    build_action_sequence_dataset,
    evaluate_action_world_model,
    horizons_ticks_to_frames,
    linear_probe_yaw,
    load_action_world_model,
    save_action_world_model,
    train_action_world_model,
)
from cognitive_runtime.training.nursery import (  # noqa: E402
    NurseryConfig,
    record_pathfinder_episode,
)
from cognitive_runtime.training.prediction_export import export_action_prediction_file  # noqa: E402
from cognitive_runtime.training.action_world_model import ActionWorldModelConfig  # noqa: E402


def _small_nursery_config(**overrides) -> NurseryConfig:
    base = dict(
        backend="simulated",
        realtime=False,
        train_episodes=2,
        holdout_episodes=1,
        episode_ticks=26,
        world_size=16,
        horizons=(1, 5),
        latent_width=16,
        hidden_dim=32,
        reconstruction_size=8,
        epochs=2,
        batch_size=16,
        setup_live_arena=False,
        require_first_person=False,
        require_learning=False,
    )
    base.update(overrides)
    return NurseryConfig(**base)


def _small_model_config(**overrides) -> ActionWorldModelConfig:
    base = dict(
        latent_width=16,
        hidden_dim=32,
        reconstruction_size=8,
        epochs=2,
        batch_size=16,
        warmup_frames=2,
        rollout_frames=3,
    )
    base.update(overrides)
    return ActionWorldModelConfig(**base)


@pytest.fixture(scope="module")
def turn_session(tmp_path_factory):
    root = tmp_path_factory.mktemp("awm-sessions")
    cfg = _small_nursery_config()
    return record_pathfinder_episode(str(root), "awm-pathfinder", 0, 0, cfg)


def test_horizons_ticks_to_frames_converts_and_preserves_collisions():
    # Simulated backend: 1 tick per frame -- identity.
    assert horizons_ticks_to_frames([1, 4, 8], 1.0) == [1, 4, 8]
    # Remote first-person capture can run at ~2 ticks/frame, so the default
    # nursery horizons stay distinct in recorded frame space.
    assert horizons_ticks_to_frames([1, 4, 8], 2.0) == [1, 2, 4]
    # t+1 and t+2 still collide at 2 ticks/frame if explicitly requested.
    # both label the same recorded frame step and must remain visible.
    assert horizons_ticks_to_frames([1, 2, 4], 2.0) == [1, 1, 2]
    assert horizons_ticks_to_frames([1, 2, 10, 100], 2.0) == [1, 1, 5, 50]
    with pytest.raises(ValueError):
        horizons_ticks_to_frames([1], 0.0)


def test_build_action_sequence_dataset_aligns_frames_actions_and_yaw(turn_session):
    dataset = build_action_sequence_dataset([turn_session])
    assert len(dataset.episodes) == 1
    episode = dataset.episodes[0]
    # One action per frame transition, from ordinary low-level motor commands.
    assert len(episode.actions) == len(episode.frames) - 1
    assert dataset.action_keys
    assert dataset.pixel_shape is not None and dataset.pixel_shape[2] == 3
    # spatial.rotation publishes every tick, so yaw labels ride along.
    assert any(y is not None for y in episode.yaw)
    # Simulated backend records ~one frame per tick.
    assert 0.5 < dataset.ticks_per_frame < 1.5


def test_build_action_sequence_dataset_pins_and_extends_vocabulary(turn_session):
    dataset = build_action_sequence_dataset(
        [turn_session], action_keys=["MOVE_FORWARD", "LOOK_LEFT"]
    )
    assert dataset.action_keys[:2] == ["MOVE_FORWARD", "LOOK_LEFT"]
    dataset = build_action_sequence_dataset([turn_session], action_keys=["SOMETHING"])
    assert "SOMETHING" in dataset.action_keys
    assert len(dataset.action_keys) > 1


def test_train_evaluate_probe_and_round_trip(turn_session, tmp_path):
    dataset = build_action_sequence_dataset([turn_session])
    model, stats = train_action_world_model(dataset, _small_model_config())
    assert stats["final_total_loss"] > 0.0
    assert stats["action_keys"] == dataset.action_keys

    report = evaluate_action_world_model(model, dataset, [1, 3], warmup_frames=2)
    assert set(report["horizons"]) == {1, 3}
    for entry in report["horizons"].values():
        assert entry["n_samples"] > 0
        assert entry["model_mse"] > 0.0
        assert entry["model_over_copy_last_mse"] is not None
    health = report["rollout_health"]
    assert set(health) >= {"prediction_dispersion", "target_dispersion", "frozen_rollout"}

    probe = linear_probe_yaw(model, dataset)
    assert probe["n_samples"] > 0
    if "latent" in probe:
        assert -1.5 <= probe["latent"]["r2"] <= 1.0

    pred_path = export_action_prediction_file(
        model,
        turn_session,
        "episode_00000",
        horizons=[1, 2, 4],
        horizon_labels=[1, 4, 8],
        out_path=os.path.join(str(tmp_path), "predictions.json"),
    )
    with open(pred_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["horizons"] == [1, 4, 8]
    assert payload["horizon_frames"] == {"1": 1, "4": 2, "8": 4}
    assert set(payload["predictions"]) == {"1", "4", "8"}

    path = os.path.join(str(tmp_path), "awm.pt")
    save_action_world_model(path, model, stats)
    reloaded, reloaded_stats = load_action_world_model(path)
    assert reloaded.action_keys == model.action_keys
    assert reloaded_stats["action_keys"] == dataset.action_keys
    with torch.no_grad():
        frames = torch.stack(
            [torch.rand(3, *model.pixel_shape[:2]) for _ in range(2)]
        )
        assert torch.allclose(model.encoder(frames), reloaded.encoder(frames))


def test_evaluate_rejects_actions_outside_the_model_vocabulary(turn_session):
    dataset = build_action_sequence_dataset([turn_session])
    model, _stats = train_action_world_model(dataset, _small_model_config())
    foreign = build_action_sequence_dataset([turn_session])
    foreign.action_keys = ["SOMETHING_ELSE"]
    with pytest.raises(ValueError, match="vocabulary"):
        evaluate_action_world_model(model, foreign, [1])


def test_frozen_rollout_detector_flags_constant_predictions(turn_session):
    """A model whose decoder ignores its input decodes the same frame at
    every horizon -- the detector must flag it when reality is moving."""
    dataset = build_action_sequence_dataset([turn_session])
    model, _stats = train_action_world_model(dataset, _small_model_config())
    with torch.no_grad():
        for p in model.decoder.parameters():
            p.zero_()
    report = evaluate_action_world_model(model, dataset, [1, 3], warmup_frames=2)
    health = report["rollout_health"]
    assert health["prediction_dispersion"] == pytest.approx(0.0, abs=1e-12)
    assert health["target_dispersion"] > 0.0
    assert health["frozen_rollout"] is True


