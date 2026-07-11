"""Ego-motion canary benchmark (issue #39)."""

from __future__ import annotations

import math
import os

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.policies.constant_action import ConstantActionPolicy  # noqa: E402
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE  # noqa: E402
from cognitive_runtime.training.ego_motion_canary import (  # noqa: E402
    EgoMotionCanaryConfig,
    run_ego_motion_canary,
    save_ego_motion_canary_checkpoint,
)
from cognitive_runtime.training.visual_representation import load_pretrained_pixel_encoder  # noqa: E402


def test_constant_action_policy_repeats_and_can_add_noise():
    from cognitive_runtime.core.action import Action

    policy = ConstantActionPolicy(Action("MOVE_FORWARD"))
    for _ in range(10):
        assert policy.decide(None, None, None) == Action("MOVE_FORWARD")

    noisy = ConstantActionPolicy(
        Action("MOVE_FORWARD"), noise=1.0, action_space=ACTION_SPACE, seed=0
    )
    decisions = {noisy.decide(None, None, None) for _ in range(20)}
    assert len(decisions) > 1  # noise=1.0 always samples randomly

    with pytest.raises(ValueError):
        ConstantActionPolicy(Action("MOVE_FORWARD"), noise=0.5)  # needs action_space


def test_ego_motion_canary_runs_end_to_end_and_reports_every_horizon(tmp_path):
    cfg = EgoMotionCanaryConfig(
        train_seeds=(0, 1),
        holdout_seeds=(1000,),
        episode_ticks=40,
        world_size=16,
        horizons=(1, 3, 8),
        latent_width=16,
        hidden_dim=32,
        reconstruction_size=8,
        epochs=3,
        consistency_epochs=2,
        batch_size=16,
    )
    model, report = run_ego_motion_canary(str(tmp_path), cfg)

    assert len(report.train_sessions) == 2
    assert len(report.holdout_sessions) == 1
    for session_dir in report.train_sessions + report.holdout_sessions:
        assert os.path.isdir(session_dir)

    assert set(report.horizon_metrics) == {1, 3, 8}
    for h, entry in report.horizon_metrics.items():
        assert entry["n_samples"] > 0
        assert math.isfinite(entry["psnr_model"])
        assert math.isfinite(entry["psnr_copy_last"])
        assert math.isfinite(entry["psnr_mean_frame"])
        assert 0.0 <= entry["ssim_model"] <= 1.0 + 1e-6
        assert 0.0 <= entry["ssim_copy_last"] <= 1.0 + 1e-6
        assert 0.0 <= entry["ssim_mean_frame"] <= 1.0 + 1e-6
        assert isinstance(entry["beats_copy_last"], bool)
        assert isinstance(entry["beats_mean_frame"], bool)

    # Pretraining and horizon-consistency losses both make progress.
    assert (
        report.pretraining_stats["loss_curves"]["reconstruction_loss"][-1]
        <= report.pretraining_stats["loss_curves"]["reconstruction_loss"][0]
    )
    assert report.consistency_stats["total_loss"][-1] <= report.consistency_stats["total_loss"][0]

    path = os.path.join(str(tmp_path), "ego_motion_canary.pt")
    metadata = save_ego_motion_canary_checkpoint(path, model, report)
    assert metadata["training_stats"]["ego_motion_canary"]["horizon_metrics"]
    loaded_encoder = load_pretrained_pixel_encoder(
        path, pixel_shape=model.pixel_shape, latent_width=cfg.latent_width
    )
    assert loaded_encoder.latent_width == cfg.latent_width


def test_ego_motion_canary_rejects_overlapping_seeds():
    cfg = EgoMotionCanaryConfig(train_seeds=(0, 1), holdout_seeds=(1,))
    with pytest.raises(ValueError, match="overlap"):
        run_ego_motion_canary("unused", cfg)


def test_evaluate_ego_motion_holdout_needs_frames_past_the_largest_horizon(tmp_path):
    cfg = EgoMotionCanaryConfig(
        train_seeds=(0,),
        holdout_seeds=(1000,),
        episode_ticks=10,
        world_size=16,
        horizons=(1, 5, 50),
        latent_width=8,
        hidden_dim=16,
        reconstruction_size=8,
        epochs=1,
        consistency_epochs=0,
        batch_size=8,
    )
    with pytest.raises(ValueError, match="too short"):
        run_ego_motion_canary(str(tmp_path), cfg)
