"""Live pathfinder nursery."""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.cli import build_parser, main  # noqa: E402
from cognitive_runtime.training.action_world_model import ActionWorldModelConfig  # noqa: E402
from cognitive_runtime.training.nursery import (  # noqa: E402
    NurseryConfig,
    PathfinderGoal,
    PathfinderTeacherPolicy,
    measure_pathfinder_recording,
    nursery_recorded_ticks,
    record_pathfinder_episode,
    run_live_pathfinder_nursery,
    validate_pathfinder_recordings,
)
from cognitive_runtime.training.nursery import _learning_check  # noqa: E402
from cognitive_runtime.runtime.replay import list_episodes, load_decisions, load_session_metadata  # noqa: E402


def _small_config(**overrides) -> NurseryConfig:
    base = dict(
        backend="simulated",
        realtime=False,
        train_episodes=2,
        holdout_episodes=1,
        episode_ticks=34,
        world_size=18,
        horizons=(1, 3),
        latent_width=16,
        hidden_dim=32,
        reconstruction_size=8,
        epochs=2,
        batch_size=16,
        warmup_frames=2,
        rollout_frames=3,
        setup_live_arena=False,
        require_first_person=False,
        require_learning=False,
    )
    base.update(overrides)
    return NurseryConfig(**base)


def _small_model_config() -> ActionWorldModelConfig:
    return ActionWorldModelConfig(
        latent_width=16,
        hidden_dim=32,
        reconstruction_size=8,
        epochs=2,
        batch_size=16,
        warmup_frames=2,
        rollout_frames=3,
    )


def test_pathfinder_teacher_emits_turns_then_forward():
    policy = PathfinderTeacherPolicy(PathfinderGoal(0.0, None, 8.0, 1.0))
    action = policy.decide(
        state=type("StateLike", (), {
            "observation": type("Obs", (), {"data": {"x": 0.0, "z": 0.0, "yaw": 90.0}})()
        })(),
        memory=None,
        prediction=None,
    )
    assert action.name in {"LOOK_LEFT", "LOOK_RIGHT"}

    action = policy.decide(
        state=type("StateLike", (), {
            "observation": type("Obs", (), {"data": {"x": 0.0, "z": 0.0, "yaw": 0.0}})()
        })(),
        memory=None,
        prediction=None,
    )
    assert action.name == "MOVE_FORWARD"


def test_record_pathfinder_episode_records_nursery_metadata_and_actions(tmp_path):
    cfg = _small_config(episode_ticks=24, horizons=(1, 4, 8))
    session_dir = record_pathfinder_episode(str(tmp_path), "pathfinder-one", 0, 0, cfg)
    metadata = load_session_metadata(session_dir)
    assert metadata["curriculum"] == "nursery/pathfinder"
    assert metadata["program_config"]["episode_ticks"] == nursery_recorded_ticks(cfg)
    assert metadata["program_config"]["nursery"]["active_episode_ticks"] == cfg.episode_ticks
    assert metadata["program_config"]["nursery"]["horizon_tail_ticks"] == 8
    assert len(load_decisions(session_dir, "episode_00000")) == cfg.episode_ticks + 8

    episode_id = list_episodes(session_dir)[0]
    quality = measure_pathfinder_recording(session_dir, episode_id)
    assert quality.n_frames > 1
    assert quality.non_null_action_counts


def test_record_pathfinder_episode_refuses_existing_session_dir(tmp_path):
    cfg = _small_config(episode_ticks=12)
    record_pathfinder_episode(str(tmp_path), "pathfinder-one", 0, 0, cfg)

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        record_pathfinder_episode(str(tmp_path), "pathfinder-one", 0, 0, cfg)


def test_validate_pathfinder_recordings_checks_first_person_for_remote(tmp_path):
    cfg = _small_config(episode_ticks=24)
    session_dir = record_pathfinder_episode(str(tmp_path), "pathfinder-grid", 0, 0, cfg)
    episode_id = list_episodes(session_dir)[0]
    summary_path = os.path.join(session_dir, f"{episode_id}.summary.json")
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    summary.setdefault("program_stats", {})["pixel_sources"] = ["grid"]
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh)

    remote_cfg = _small_config(backend="remote", require_first_person=True)
    _qualities, issues = validate_pathfinder_recordings([session_dir], remote_cfg)
    assert any("first-person viewer" in issue for issue in issues)


def test_validate_pathfinder_recordings_checks_failed_live_setup(tmp_path):
    cfg = _small_config(episode_ticks=24)
    session_dir = record_pathfinder_episode(str(tmp_path), "pathfinder-setup", 0, 0, cfg)
    episode_id = list_episodes(session_dir)[0]
    summary_path = os.path.join(session_dir, f"{episode_id}.summary.json")
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    summary.setdefault("program_stats", {}).update(
        {
            "pixel_sources": ["viewer"],
            "nursery_setup_requested": True,
            "nursery_setup_reached_start": False,
            "nursery_setup_distance_from_start": 12.5,
        }
    )
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh)

    remote_cfg = _small_config(
        backend="remote",
        setup_live_arena=True,
        require_first_person=True,
    )
    _qualities, issues = validate_pathfinder_recordings([session_dir], remote_cfg)
    assert any("arena setup did not move the bot" in issue for issue in issues)


def test_run_live_pathfinder_nursery_trains_action_world_model(tmp_path):
    cfg = _small_config()
    model, report = run_live_pathfinder_nursery(
        str(tmp_path), config=cfg, model_config=_small_model_config()
    )
    assert model.action_keys
    assert len(report.train_sessions) == cfg.train_episodes
    assert len(report.holdout_sessions) == cfg.holdout_episodes
    assert set(report.metrics["horizons"]) == set(report.horizon_frames)
    assert report.horizon_ticks == list(cfg.horizons)
    assert report.horizon_frame_mapping == {"1": 1, "3": 3}
    assert set(report.metrics["tick_horizons"]) == {"1", "3"}
    assert report.metrics["tick_horizons"]["1"]["frame_horizon"] == 1
    assert "beats_copy_last_any_horizon" in report.learning_check
    assert report.quality


def test_run_live_pathfinder_nursery_skips_existing_session_dirs(tmp_path):
    cfg = _small_config()
    existing = tmp_path / "nursery-pathfinder-train-1"
    existing.mkdir()

    _model, report = run_live_pathfinder_nursery(
        str(tmp_path), config=cfg, model_config=_small_model_config()
    )

    train_names = {os.path.basename(path) for path in report.train_sessions}
    assert train_names == {"nursery-pathfinder-train-0", "nursery-pathfinder-train-2"}
    assert existing.exists()


def test_learning_check_prefers_tick_horizon_labels_when_frames_collide():
    metrics = {
        "horizons": {
            1: {"model_over_copy_last_mse": 10.0},
            2: {"model_over_copy_last_mse": 20.0},
        },
        "tick_horizons": {
            "1": {"frame_horizon": 1, "model_over_copy_last_mse": 10.0},
            "2": {"frame_horizon": 1, "model_over_copy_last_mse": 10.0},
            "8": {"frame_horizon": 2, "model_over_copy_last_mse": 20.0},
        },
    }

    check = _learning_check(metrics)

    assert check["model_over_copy_last_mse"] == {"1": 10.0, "2": 10.0, "8": 20.0}


def test_nursery_cli_list_and_simulated_smoke(tmp_path, capsys):
    main(["nursery", "list"])
    assert "pathfinder" in capsys.readouterr().out

    report_path = tmp_path / "report.json"
    main([
        "nursery", "run",
        "--backend", "simulated",
        "--record-dir", str(tmp_path / "sessions"),
        "--train-episodes", "2",
        "--holdout-episodes", "1",
        "--episode-ticks", "24",
        "--world-size", "18",
        "--horizons", "1", "3",
        "--latent-width", "16",
        "--hidden-dim", "32",
        "--reconstruction-size", "8",
        "--epochs", "2",
        "--warmup-frames", "2",
        "--rollout-frames", "3",
        "--batch-size", "16",
        "--no-setup-live-arena",
        "--allow-grid-pixels",
        "--out-dir", str(tmp_path / "models"),
        "--report", str(report_path),
    ])
    out = capsys.readouterr().out
    assert "holdout prediction metrics" in out
    assert report_path.exists()
    with open(report_path, encoding="utf-8") as fh:
        report = json.load(fh)
    pred_path = os.path.join(
        report["holdout_sessions"][0], "predictions_episode_00000.json"
    )
    assert os.path.exists(pred_path)
    with open(pred_path, encoding="utf-8") as fh:
        predictions = json.load(fh)
    assert predictions["format"] == "pixel-predictions-v1"
    assert predictions["playback_frame_count"] == 24
    assert report["horizon_ticks"] == [1, 3]
    assert report["horizon_frame_mapping"] == {"1": 1, "3": 3}
    assert set(predictions["predictions"]) == set(str(h) for h in report["horizon_ticks"])
    assert predictions["horizon_frames"] == {"1": 1, "3": 3}
    assert {
        len(entry["frames"]) for entry in predictions["predictions"].values()
    } == {predictions["playback_frame_count"]}


def test_nursery_cli_backend_default_tracks_live_env(monkeypatch):
    monkeypatch.delenv("CCR_NURSERY_BACKEND", raising=False)
    monkeypatch.delenv("CCR_MINECRAFT_HOST", raising=False)
    parser = build_parser()
    args = parser.parse_args(["nursery", "run"])
    assert args.backend == "simulated"
    assert args.require_learning is False

    monkeypatch.setenv("CCR_MINECRAFT_HOST", "localhost")
    parser = build_parser()
    args = parser.parse_args(["nursery", "run"])
    assert args.backend == "remote"

    monkeypatch.setenv("CCR_NURSERY_BACKEND", "simulated")
    parser = build_parser()
    args = parser.parse_args(["nursery", "run"])
    assert args.backend == "simulated"

    parser = build_parser()
    args = parser.parse_args(["nursery", "run", "--require-learning"])
    assert args.require_learning is True
