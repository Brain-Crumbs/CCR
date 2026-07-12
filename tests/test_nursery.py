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
    NurseryScenario,
    ScenarioRecording,
    _record_scenario_episode,
    measure_recording_quality,
    render_dream_strip,
    run_nursery_scenario,
    run_nursery_suite,
    save_nursery_scenario_checkpoint,
    validate_nursery_recordings,
)
from cognitive_runtime.training.prediction_export import (  # noqa: E402
    export_prediction_file,
    load_full_visual_model,
    save_full_visual_model,
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


# --------------------------------------------------------------------------- data-quality gate


def _record_static_walk_session(tmp_path, session_id="static-walk"):
    """Record a session whose agent never moves -- the recorded shape of the
    first real walk_forward run (stuck agent on the remote backend's
    persistent world), reproduced on the simulated backend with a null
    policy."""
    from cognitive_runtime.policies.null_policy import NullPolicy

    stuck = NurseryScenario(
        "walk_forward", "stuck-agent surrogate",
        lambda seed, cfg: ScenarioRecording(policy=NullPolicy()),
    )
    cfg = _small_config()
    return _record_scenario_episode(str(tmp_path), session_id, 0, stuck, cfg)


def test_measure_recording_quality_sees_motion_and_unique_frames(tmp_path):
    cfg = _small_config()
    session_dir = _record_scenario_episode(
        str(tmp_path), "healthy-walk", 0, NURSERY_SCENARIOS["walk_forward"], cfg
    )
    quality = measure_recording_quality(session_dir, list_episodes(session_dir)[0])
    assert quality.n_frames > 0
    assert quality.duration_ticks == cfg.episode_ticks
    assert quality.blocks_per_tick >= NURSERY_SCENARIOS["walk_forward"].min_blocks_per_tick
    assert (
        quality.unique_frame_fraction
        >= NURSERY_SCENARIOS["walk_forward"].min_unique_frame_fraction
    )


def test_validate_nursery_recordings_flags_a_static_session(tmp_path):
    session_dir = _record_static_walk_session(tmp_path)
    issues = validate_nursery_recordings([session_dir], NURSERY_SCENARIOS["walk_forward"])
    assert issues
    assert any("barely moved" in issue for issue in issues)


def test_validate_nursery_recordings_passes_a_healthy_walk(tmp_path):
    cfg = _small_config()
    session_dir = _record_scenario_episode(
        str(tmp_path), "healthy-walk", 0, NURSERY_SCENARIOS["walk_forward"], cfg
    )
    assert validate_nursery_recordings([session_dir], NURSERY_SCENARIOS["walk_forward"]) == []


def test_run_nursery_scenario_gate_rejects_motionless_recordings(tmp_path, monkeypatch):
    from cognitive_runtime.policies.null_policy import NullPolicy

    stuck = NurseryScenario(
        "walk_forward",
        NURSERY_SCENARIOS["walk_forward"].description,
        lambda seed, cfg: ScenarioRecording(policy=NullPolicy()),
        min_blocks_per_tick=NURSERY_SCENARIOS["walk_forward"].min_blocks_per_tick,
        min_unique_frame_fraction=NURSERY_SCENARIOS["walk_forward"].min_unique_frame_fraction,
    )
    monkeypatch.setitem(NURSERY_SCENARIOS, "walk_forward", stuck)
    with pytest.raises(ValueError, match="quality gate"):
        run_nursery_scenario(str(tmp_path), "walk_forward", _small_config())

    # data_quality_gate=False trains on the same recordings without raising.
    model, report = run_nursery_scenario(
        str(tmp_path / "ungated"), "walk_forward", _small_config(data_quality_gate=False)
    )
    assert report.horizon_metrics


# --------------------------------------------------------------------------- prediction export


def test_run_nursery_scenario_exports_viewer_predictions(tmp_path):
    cfg = _small_config()
    _model, report = run_nursery_scenario(str(tmp_path), "walk_forward", cfg)

    assert report.prediction_files
    all_sessions = set(report.train_sessions + report.holdout_sessions)
    for key, path in report.prediction_files.items():
        session_dir = key.rsplit("/", 1)[0]
        assert session_dir in all_sessions
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["format"] == "pixel-predictions-v1"
        assert payload["horizons"] == sorted(cfg.horizons)
        assert len(payload["targets"]) == payload["n_frames"]
        for h in cfg.horizons:
            assert len(payload["predictions"][str(h)]["frames"]) == payload["n_frames"] - h


def test_run_nursery_scenario_can_skip_prediction_export(tmp_path):
    cfg = _small_config(export_predictions=False)
    _model, report = run_nursery_scenario(str(tmp_path), "walk_forward", cfg)
    assert report.prediction_files == {}
    for session_dir in report.train_sessions + report.holdout_sessions:
        assert not [f for f in os.listdir(session_dir) if f.startswith("predictions_")]


def test_full_visual_model_round_trips_and_re_exports(tmp_path):
    import base64

    cfg = _small_config()
    model, report = run_nursery_scenario(str(tmp_path), "walk_forward", cfg)

    bundle = tmp_path / "walk-full.pt"
    save_full_visual_model(model, str(bundle))
    reloaded = load_full_visual_model(str(bundle))
    assert reloaded.pixel_shape == model.pixel_shape
    assert reloaded.latent_width == model.latent_width
    assert reloaded.reconstruction_shape == model.reconstruction_shape

    session_dir = report.holdout_sessions[0]
    episode_id = list_episodes(session_dir)[0]
    out = export_prediction_file(
        reloaded, session_dir, episode_id, cfg.horizons,
        out_path=str(tmp_path / "re-export.json"),
    )
    with open(out, encoding="utf-8") as fh:
        payload = json.load(fh)
    h, w, c = payload["prediction_shape"]
    frame = base64.b64decode(payload["predictions"]["1"]["frames"][0])
    assert len(frame) == h * w * c


# --------------------------------------------------------------------------- catalog honesty


def test_stream_catalog_reports_paced_rates_in_realtime():
    from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox

    program = MinecraftSurvivalBox(config={"episode_ticks": 8, "world_size": 16})
    try:
        specs = {s.stream_id: s for s in program.stream_catalog()}
        assert specs["vision.frame.pixels"].nominal_rate_hz == 20.0
        assert specs["body.health"].nominal_rate_hz == 1.0
        assert specs["spatial.position"].range == (0.0, 16.0)

        program.set_realtime(True)
        specs = {s.stream_id: s for s in program.stream_catalog()}
        assert specs["vision.frame.pixels"].nominal_rate_hz == program._config.realtime_vision_hz
        assert specs["vision.frame.grid"].nominal_rate_hz == program._config.realtime_vision_hz
        assert specs["body.health"].nominal_rate_hz == program._config.realtime_body_heartbeat_hz

        program.set_realtime(False)
        specs = {s.stream_id: s for s in program.stream_catalog()}
        assert specs["vision.frame.pixels"].nominal_rate_hz == 20.0
    finally:
        program.close()


def test_stream_catalog_drops_position_bounds_for_remote_backend():
    from cognitive_runtime.programs.minecraft.streams import build_survival_stream_specs

    specs = {s.stream_id: s for s in build_survival_stream_specs(48, bounded_position=False)}
    assert specs["spatial.position"].range is None
    assert specs["spatial.distance_from_spawn"].range is None
