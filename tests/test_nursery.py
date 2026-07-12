"""Nursery scenario suite (issue #62)."""

from __future__ import annotations

import json
import math
import os

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.cli import main  # noqa: E402
from cognitive_runtime.cli import build_parser  # noqa: E402
from cognitive_runtime.policies.scripted_sequence import ScriptedSequencePolicy  # noqa: E402
from cognitive_runtime.core.action import Action  # noqa: E402
from cognitive_runtime.training.nursery import (  # noqa: E402
    NURSERY_SCENARIOS,
    NurseryConfig,
    render_dream_strip,
    run_nursery_scenario,
    run_nursery_suite,
    save_nursery_scenario_checkpoint,
)
from cognitive_runtime.training.visual_representation import load_pretrained_pixel_encoder  # noqa: E402
from cognitive_runtime.runtime.replay import list_episodes, load_session_metadata  # noqa: E402


def test_scripted_sequence_policy_cycles_and_validates():
    policy = ScriptedSequencePolicy([(Action("MOVE_LEFT"), 2), (Action("NULL"), 3)])
    actions = [policy.decide(None, None, None) for _ in range(10)]
    assert actions == [
        Action("MOVE_LEFT"), Action("MOVE_LEFT"), Action("NULL"), Action("NULL"), Action("NULL"),
        Action("MOVE_LEFT"), Action("MOVE_LEFT"), Action("NULL"), Action("NULL"), Action("NULL"),
    ]
    policy.reset()
    assert policy.decide(None, None, None) == Action("MOVE_LEFT")

    with pytest.raises(ValueError):
        ScriptedSequencePolicy([])
    with pytest.raises(ValueError):
        ScriptedSequencePolicy([(Action("MOVE_LEFT"), 0)])


def test_registry_has_every_scenario_from_the_issue():
    expected = {
        "walk_forward", "turn_in_place", "strafe_and_stop",
        "object_permanence", "day_night", "approach_entity",
    }
    assert set(NURSERY_SCENARIOS) == expected
    assert NURSERY_SCENARIOS["object_permanence"].entity_persistence_metric is True
    assert NURSERY_SCENARIOS["walk_forward"].entity_persistence_metric is False


def _small_config(**overrides) -> NurseryConfig:
    base = dict(
        train_seeds=(0, 1),
        holdout_seeds=(1000,),
        episode_ticks=24,
        world_size=16,
        horizons=(1, 5),
        latent_width=16,
        hidden_dim=32,
        reconstruction_size=8,
        epochs=3,
        consistency_epochs=2,
        batch_size=16,
        entity_persistence_epochs=30,
    )
    base.update(overrides)
    return NurseryConfig(**base)


def test_walk_forward_scenario_matches_the_ego_motion_canary_shape(tmp_path):
    """`walk_forward` reuses `ego_motion_canary`'s recording/evaluation
    helpers verbatim, so its report has the same per-horizon shape as issue
    #39's canary (`test_ego_motion_canary.py` doesn't hard-assert
    `beats_copy_last`/`beats_mean_frame` either -- whether the model beats
    the baselines is a hyperparameter/training-budget-scale property,
    validated by running `nursery run walk_forward` at production
    hyperparameters, not by this fast unit test)."""
    cfg = _small_config()
    model, report = run_nursery_scenario(str(tmp_path), "walk_forward", cfg)

    assert report.scenario == "walk_forward"
    assert len(report.train_sessions) == 2
    assert len(report.holdout_sessions) == 1
    for session_dir in report.train_sessions + report.holdout_sessions:
        assert os.path.isdir(session_dir)
        metadata = load_session_metadata(session_dir)
        assert metadata["curriculum"] == "nursery/walk_forward"

    assert set(report.horizon_metrics) == {1, 5}
    for entry in report.horizon_metrics.values():
        assert entry["n_samples"] > 0
        assert math.isfinite(entry["psnr_model"])
        assert math.isfinite(entry["psnr_copy_last"])
        assert math.isfinite(entry["psnr_mean_frame"])
        assert isinstance(entry["beats_copy_last"], bool)
        assert isinstance(entry["beats_mean_frame"], bool)
    assert report.entity_persistence_stats is None
    assert report.dream_strips  # one per holdout episode


def test_object_permanence_scenario_records_occlusion_and_beats_forget_baseline(tmp_path):
    """Acceptance criterion: object_permanence's metric distinguishes a model
    with entity-persistence training from one without (issue #27) -- here,
    the trained model's MSE against the "forget immediately" baseline."""
    cfg = _small_config(train_seeds=(0, 1, 2), holdout_seeds=(1000,), horizons=(1, 3))
    model, report = run_nursery_scenario(str(tmp_path), "object_permanence", cfg)

    assert report.scenario == "object_permanence"
    assert report.entity_persistence_stats is not None
    stats = report.entity_persistence_stats
    assert stats.get("samples", 1) != 0  # not the "no occlusion events" fallback
    assert stats["beats_forget_baseline"] is True
    assert stats["model_mse"] < stats["baseline_mse"]

    # The occlusion/reappearance cycle also produced pixel frames long enough
    # for the configured horizons.
    assert set(report.horizon_metrics) == {1, 3}


def test_run_nursery_scenario_rejects_overlapping_seeds():
    cfg = NurseryConfig(train_seeds=(0, 1), holdout_seeds=(1,))
    with pytest.raises(ValueError, match="overlap"):
        run_nursery_scenario("unused", "walk_forward", cfg)


def test_run_nursery_scenario_rejects_unknown_backend():
    cfg = NurseryConfig(backend="not-a-backend")
    with pytest.raises(ValueError, match="unknown nursery backend"):
        run_nursery_scenario("unused", "walk_forward", cfg)


def test_run_nursery_scenario_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="unknown nursery scenario"):
        run_nursery_scenario(str(tmp_path), "not_a_real_scenario", NurseryConfig())


def test_run_nursery_suite_runs_every_scenario_unattended(tmp_path):
    """Acceptance criterion: `nursery run all` runs unattended in the
    simulated backend, producing recordings + a per-scenario, per-horizon
    report."""
    cfg = _small_config(train_seeds=(0, 1), holdout_seeds=(1000,), horizons=(1, 2))
    reports = run_nursery_suite(str(tmp_path), config=cfg)

    assert set(reports) == set(NURSERY_SCENARIOS)
    for name, report in reports.items():
        assert set(report.horizon_metrics) == {1, 2}
        for entry in report.horizon_metrics.values():
            assert math.isfinite(entry["psnr_model"])
            assert isinstance(entry["beats_copy_last"], bool)
        assert report.dream_strips
        if NURSERY_SCENARIOS[name].entity_persistence_metric:
            assert report.entity_persistence_stats is not None
        else:
            assert report.entity_persistence_stats is None


def test_render_dream_strip_has_a_line_per_horizon(tmp_path):
    cfg = _small_config(horizons=(1, 5))
    model, report = run_nursery_scenario(str(tmp_path), "walk_forward", cfg)
    session_dir = report.holdout_sessions[0]
    episode_id = list_episodes(session_dir)[0]

    strip = render_dream_strip(model, session_dir, episode_id, cfg.horizons)
    assert "t+1:" in strip
    assert "t+5:" in strip
    assert "predicted | actual" in strip


def test_save_nursery_scenario_checkpoint_round_trips(tmp_path):
    cfg = _small_config()
    model, report = run_nursery_scenario(str(tmp_path), "walk_forward", cfg)
    path = os.path.join(str(tmp_path), "nursery_walk_forward.pt")
    metadata = save_nursery_scenario_checkpoint(path, model, report)
    assert metadata["training_stats"]["nursery"]["scenario"] == "walk_forward"
    assert metadata["training_stats"]["nursery"]["horizon_metrics"]

    loaded_encoder = load_pretrained_pixel_encoder(
        path, pixel_shape=model.pixel_shape, latent_width=cfg.latent_width
    )
    assert loaded_encoder.latent_width == cfg.latent_width


def test_nursery_cli_list_and_run(tmp_path, capsys):
    main(["nursery", "list"])
    listed = capsys.readouterr().out
    assert "walk_forward" in listed
    assert "object_permanence" in listed

    record_dir = str(tmp_path / "sessions")
    report_path = str(tmp_path / "report.json")
    main([
        "nursery", "run", "walk_forward",
        "--record-dir", record_dir,
        "--train-seeds", "2", "--holdout-seeds", "1",
        "--episode-ticks", "24", "--world-size", "16",
        "--horizons", "1", "3",
        "--latent-width", "16", "--hidden-dim", "32", "--reconstruction-size", "8",
        "--epochs", "3", "--consistency-epochs", "2", "--batch-size", "16",
        "--report", report_path,
    ])
    out = capsys.readouterr().out
    assert "walk_forward:" in out
    assert "horizon t+1" in out
    assert os.path.exists(report_path)
    with open(report_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert "walk_forward" in payload
    assert "1" in payload["walk_forward"]["horizon_metrics"]

    with pytest.raises(SystemExit, match="unknown nursery scenario"):
        main(["nursery", "run", "not_a_scenario", "--record-dir", record_dir])


def test_nursery_cli_backend_default_tracks_live_env(monkeypatch):
    monkeypatch.delenv("CCR_NURSERY_BACKEND", raising=False)
    monkeypatch.delenv("CCR_MINECRAFT_HOST", raising=False)
    parser = build_parser()
    args = parser.parse_args(["nursery", "run", "walk_forward"])
    assert args.backend == "simulated"

    monkeypatch.setenv("CCR_MINECRAFT_HOST", "localhost")
    parser = build_parser()
    args = parser.parse_args(["nursery", "run", "walk_forward"])
    assert args.backend == "remote"

    monkeypatch.setenv("CCR_NURSERY_BACKEND", "simulated")
    parser = build_parser()
    args = parser.parse_args(["nursery", "run", "walk_forward"])
    assert args.backend == "simulated"
