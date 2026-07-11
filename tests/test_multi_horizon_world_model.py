"""Multi-horizon, uncertainty-aware world model (issue #39)."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.neural import (  # noqa: E402
    CheckpointCompatibilityError,
    HorizonPrediction,
    MultiHorizonMLPWorldModel,
    MultiHorizonWorldModelOutput,
    WorldModelOutput,
)
from cognitive_runtime.policies import ScriptedSurvivalPolicy  # noqa: E402
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox  # noqa: E402
from cognitive_runtime.runtime.config import RuntimeConfig  # noqa: E402
from cognitive_runtime.runtime.loop import CognitiveRuntime  # noqa: E402
from cognitive_runtime.training.datasets import build_multi_horizon_world_model_dataset  # noqa: E402
from cognitive_runtime.training.world_model import (  # noqa: E402
    MultiHorizonWorldModelTrainingConfig,
    evaluate_multi_horizon_model,
    load_multi_horizon_world_model_checkpoint,
    multi_horizon_baseline_mse,
    save_multi_horizon_world_model_checkpoint,
    train_multi_horizon_world_model,
    uncertainty_calibration,
)


def _record_session(tmp_path, session_id, *, policy, ticks, seed=0, world_size=16):
    config = {"episode_ticks": ticks, "world_size": world_size}
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=seed,
        max_ticks_per_episode=ticks,
        record_dir=str(tmp_path),
        session_id=session_id,
        program_config=config,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config), policy=policy, config=runtime_config
    ).run()
    return os.path.join(str(tmp_path), session_id)


def test_multi_horizon_model_forward_is_backward_compatible_worldmodel():
    torch.manual_seed(0)
    model = MultiHorizonMLPWorldModel(
        fused_width=6, n_actions=3, horizons=(1, 5, 20), hidden_dim=8, depth=2,
        layout_hash="layout-x", action_keys=["a", "b", "c"],
    )
    fused = torch.randn(4, 6)
    actions = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 0]), num_classes=3).float()

    out = model(fused, actions)
    assert isinstance(out, WorldModelOutput)
    assert out.next_latent.shape == (4, 6)

    horizon_out = model.forward_horizons(fused, actions)
    assert isinstance(horizon_out, MultiHorizonWorldModelOutput)
    assert set(horizon_out.horizons) == {1, 5, 20}
    for h, pred in horizon_out.horizons.items():
        assert isinstance(pred, HorizonPrediction)
        assert pred.next_latent.shape == (4, 6)
        assert pred.reward.shape == (4,)
        assert pred.uncertainty.shape == (4,)
        assert bool((pred.uncertainty >= 0).all())
        assert bool((pred.prediction_error >= 0).all())

    # horizon 1 matches forward()'s own heads exactly (shared parameters).
    assert torch.allclose(out.next_latent, horizon_out[1].next_latent)
    assert torch.allclose(out.reward, horizon_out[1].reward)

    meta = model.checkpoint_metadata()
    assert meta["horizons"] == [1, 5, 20]


def test_horizons_must_include_one():
    with pytest.raises(ValueError, match="horizons must include 1"):
        MultiHorizonMLPWorldModel(fused_width=4, n_actions=2, horizons=(5, 20))


def test_build_multi_horizon_dataset_from_recorded_session(tmp_path):
    session_dir = _record_session(
        tmp_path, "mh-dataset", policy=ScriptedSurvivalPolicy(seed=1), ticks=200
    )
    dataset = build_multi_horizon_world_model_dataset([session_dir], horizons=(1, 5, 20))

    assert len(dataset) > 0
    assert dataset.horizons == [1, 5, 20]
    for h in dataset.horizons:
        assert len(dataset.future_latents[h]) == len(dataset)
        assert len(dataset.future_rewards[h]) == len(dataset)
        assert all(d in (0.0, 1.0) for d in dataset.future_dones[h])
        assert all(r in (0.0, 1.0) for r in dataset.future_risks[h])
    assert dataset.layout_hash is not None


def test_multi_horizon_baseline_mse_reports_every_horizon(tmp_path):
    session_dir = _record_session(
        tmp_path, "mh-baseline", policy=ScriptedSurvivalPolicy(seed=2), ticks=200
    )
    dataset = build_multi_horizon_world_model_dataset([session_dir], horizons=(1, 5, 20))
    report = multi_horizon_baseline_mse(dataset)
    assert set(report) == {1, 5, 20}
    for h, entry in report.items():
        assert entry["copy_last_mse"] >= 0.0
        assert entry["mean_latent_mse"] >= 0.0


def test_multi_horizon_training_beats_baselines_and_checkpoints(tmp_path):
    session_dir = _record_session(
        tmp_path, "mh-train", policy=ScriptedSurvivalPolicy(seed=3), ticks=400
    )
    dataset = build_multi_horizon_world_model_dataset([session_dir], horizons=(1, 5, 20))
    assert len(dataset) > 16

    model, stats = train_multi_horizon_world_model(
        dataset,
        MultiHorizonWorldModelTrainingConfig(
            epochs=25, lr=5e-3, batch_size=16, seed=4, hidden_dim=32, depth=2,
        ),
    )

    for h in dataset.horizons:
        curve = stats["loss_curves"][f"h{h}_next_latent_nll"]
        assert curve[-1] < curve[0], h

    evaluation = stats["evaluation"]
    assert set(evaluation) == {1, 5, 20}
    for entry in evaluation.values():
        assert entry["model_mse"] >= 0.0
        assert entry["copy_last_mse"] >= 0.0
        assert entry["mean_latent_mse"] >= 0.0

    path = os.path.join(str(tmp_path), "multi_horizon_world_model.pt")
    metadata = save_multi_horizon_world_model_checkpoint(path, model, dataset, stats)
    loaded, loaded_metadata = load_multi_horizon_world_model_checkpoint(
        path, expected_layout_hash=dataset.layout_hash
    )

    assert (
        metadata["modules"]["world_model"]["checkpoint_metadata"]["horizons"] == [1, 5, 20]
    )
    assert loaded_metadata["training_stats"]["evaluation"]
    assert loaded.fused_width() == model.fused_width()
    assert loaded.horizons_list == model.horizons_list

    with pytest.raises(CheckpointCompatibilityError, match="layout"):
        load_multi_horizon_world_model_checkpoint(path, expected_layout_hash="different-layout")


def test_uncertainty_calibration_returns_finite_correlation_per_horizon(tmp_path):
    session_dir = _record_session(
        tmp_path, "mh-calib", policy=ScriptedSurvivalPolicy(seed=5), ticks=400
    )
    dataset = build_multi_horizon_world_model_dataset([session_dir], horizons=(1, 5, 20))
    model, _stats = train_multi_horizon_world_model(
        dataset,
        MultiHorizonWorldModelTrainingConfig(
            epochs=10, lr=5e-3, batch_size=16, seed=6, hidden_dim=32, depth=2,
        ),
    )
    calibration = uncertainty_calibration(model, dataset)
    assert set(calibration) == {1, 5, 20}
    for value in calibration.values():
        assert -1.0 <= value <= 1.0


def test_evaluate_multi_horizon_model_on_held_out_episode(tmp_path):
    train_dir = _record_session(
        tmp_path, "mh-holdout-train", policy=ScriptedSurvivalPolicy(seed=7), ticks=400
    )
    holdout_dir = _record_session(
        tmp_path, "mh-holdout-eval", policy=ScriptedSurvivalPolicy(seed=8), ticks=400, seed=42
    )
    train_dataset = build_multi_horizon_world_model_dataset([train_dir], horizons=(1, 5))
    holdout_dataset = build_multi_horizon_world_model_dataset([holdout_dir], horizons=(1, 5))

    model, _stats = train_multi_horizon_world_model(
        train_dataset,
        MultiHorizonWorldModelTrainingConfig(
            epochs=15, lr=5e-3, batch_size=16, seed=9, hidden_dim=32, depth=2,
        ),
    )
    report = evaluate_multi_horizon_model(model, holdout_dataset)
    assert set(report) == {1, 5}
    for entry in report.values():
        assert entry["model_mse"] >= 0.0
