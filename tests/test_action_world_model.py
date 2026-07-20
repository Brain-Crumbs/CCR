"""Action-conditioned recurrent world model (phases 1-3 of
docs/nursery-turn-in-place-analysis.md)."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.training.action_world_model import (  # noqa: E402
    ActionWorldModelConfig,
    _episode_tensors,
    build_action_sequence_dataset,
    build_action_world_model,
    evaluate_action_world_model,
    evaluate_cortex_heads,
    horizons_ticks_to_frames,
    linear_probe_yaw,
    load_action_world_model,
    save_action_world_model,
    train_action_world_model,
)
from cognitive_runtime.training.nursery import (  # noqa: E402
    NURSERY_SCENARIOS,
    NurseryConfig,
    _record_scenario_episode,
    run_nursery_joint,
)
from cognitive_runtime.training.action_world_model import ActionWorldModelConfig  # noqa: E402
from brain.cortex.predictive import PredictiveCortex  # noqa: E402


def _small_nursery_config(**overrides) -> NurseryConfig:
    base = dict(
        train_seeds=(0, 1),
        holdout_seeds=(1000,),
        episode_ticks=26,
        world_size=16,
        horizons=(1, 5),
        latent_width=16,
        hidden_dim=32,
        reconstruction_size=8,
        epochs=2,
        batch_size=16,
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
    return _record_scenario_episode(
        str(root), "awm-turn", 0, NURSERY_SCENARIOS["turn_in_place"], cfg
    )


def test_horizons_ticks_to_frames_converts_and_dedupes():
    # Simulated backend: 1 tick per frame -- identity.
    assert horizons_ticks_to_frames([1, 10, 100], 1.0) == [1, 10, 100]
    # First remote runs: ~2 ticks per frame -- t+1 and t+2 collide at 1 frame.
    assert horizons_ticks_to_frames([1, 2, 10, 100], 2.0) == [1, 5, 50]
    with pytest.raises(ValueError):
        horizons_ticks_to_frames([1], 0.0)


def test_build_action_sequence_dataset_aligns_frames_actions_and_yaw(turn_session):
    dataset = build_action_sequence_dataset([turn_session])
    assert len(dataset.episodes) == 1
    episode = dataset.episodes[0]
    # One action per frame transition, and the scripted policy is constant.
    assert len(episode.actions) == len(episode.frames) - 1
    assert dataset.action_keys == ["LOOK_LEFT"]
    assert dataset.pixel_shape is not None and dataset.pixel_shape[2] == 3
    # spatial.rotation publishes every tick, so yaw labels ride along.
    assert any(y is not None for y in episode.yaw)
    # Simulated backend records ~one frame per tick.
    assert 0.5 < dataset.ticks_per_frame < 1.5


def test_build_action_sequence_dataset_aligns_reward_terminal_risk(turn_session):
    """issue #169: reward/terminal/risk ride along per-frame like yaw does,
    read off the decision record's ``reward_window_total`` and the tick's
    ``internal.risk``/``event.died`` streams."""
    dataset = build_action_sequence_dataset([turn_session])
    episode = dataset.episodes[0]
    assert len(episode.reward) == len(episode.frames)
    assert len(episode.terminal) == len(episode.frames)
    assert len(episode.risk) == len(episode.frames)
    assert all(isinstance(r, float) for r in episode.reward)
    assert all(isinstance(t, bool) for t in episode.terminal)
    # turn_in_place never dies; risk should be a recorded, mostly-small float,
    # not a placeholder constant across the whole episode.
    assert all(0.0 <= r <= 1.0 for r in episode.risk)
    assert not any(episode.terminal)


def test_build_action_sequence_dataset_pins_and_extends_vocabulary(turn_session):
    dataset = build_action_sequence_dataset(
        [turn_session], action_keys=["MOVE_FORWARD", "LOOK_LEFT"]
    )
    assert dataset.action_keys == ["MOVE_FORWARD", "LOOK_LEFT"]
    dataset = build_action_sequence_dataset([turn_session], action_keys=["MOVE_FORWARD"])
    assert dataset.action_keys == ["MOVE_FORWARD", "LOOK_LEFT"]


def test_train_evaluate_probe_and_round_trip(turn_session, tmp_path):
    dataset = build_action_sequence_dataset([turn_session])
    model, stats = train_action_world_model(dataset, _small_model_config())
    assert stats["final_total_loss"] > 0.0
    assert stats["action_keys"] == ["LOOK_LEFT"]

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

    path = os.path.join(str(tmp_path), "awm.pt")
    save_action_world_model(path, model, stats)
    reloaded, reloaded_stats = load_action_world_model(path)
    assert reloaded.action_keys == model.action_keys
    assert reloaded_stats["action_keys"] == ["LOOK_LEFT"]
    with torch.no_grad():
        frames = torch.stack(
            [torch.rand(3, *model.pixel_shape[:2]) for _ in range(2)]
        )
        assert torch.allclose(model.encoder(frames), reloaded.encoder(frames))


def test_train_action_world_model_trains_reward_terminal_risk_uncertainty_heads(turn_session):
    """issue #169: the previously-untrained heads now get a loss curve each,
    and ``evaluate_cortex_heads`` reports a well-formed diagnostic against
    held-out data."""
    dataset = build_action_sequence_dataset([turn_session])
    model, stats = train_action_world_model(dataset, _small_model_config())

    for key in ("reward_loss", "terminal_loss", "risk_loss", "uncertainty_loss"):
        assert key in stats["loss_curves"]
        assert len(stats["loss_curves"][key]) == stats["epochs"]
        assert all(v >= 0.0 for v in stats["loss_curves"][key])
        assert stats[f"final_{key}"] == stats["loss_curves"][key][-1]

    report = evaluate_cortex_heads(model, dataset, [1, 3], warmup_frames=2)
    assert set(report) == {1, 3}
    for row in report.values():
        assert row["n_samples"] > 0
        for head in ("reward", "terminal", "risk"):
            assert row[f"{head}_mse"] >= 0.0
            assert row[f"{head}_constant_mse"] >= 0.0
            assert isinstance(row[f"{head}_beats_constant"], bool)
        correlation = row["uncertainty_error_correlation"]
        ci_low, ci_high = row["uncertainty_error_correlation_ci"]
        assert -1.0 <= correlation <= 1.0
        assert -1.0 <= ci_low <= ci_high <= 1.0


def test_evaluate_cortex_heads_rejects_actions_outside_the_model_vocabulary(turn_session):
    dataset = build_action_sequence_dataset([turn_session])
    model, _stats = train_action_world_model(dataset, _small_model_config())
    foreign = build_action_sequence_dataset([turn_session])
    foreign.action_keys = ["SOMETHING_ELSE"]
    with pytest.raises(ValueError, match="vocabulary"):
        evaluate_cortex_heads(model, foreign, [1])


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


def test_run_nursery_joint_trains_one_model_across_scenarios(tmp_path):
    cfg = _small_nursery_config(episode_ticks=30, horizons=(1, 3))
    model, report = run_nursery_joint(
        str(tmp_path),
        train_scenarios=["walk_forward", "turn_in_place"],
        holdout_scenarios=["strafe_and_stop"],
        config=cfg,
        model_config=_small_model_config(),
    )
    assert report.train_scenarios == ["walk_forward", "turn_in_place"]
    assert report.holdout_scenarios == ["strafe_and_stop"]
    # The vocabulary is pinned to the full action space so the zero-shot
    # scenario's actions are encodable.
    assert "MOVE_LEFT" in model.action_keys
    assert set(report.scenario_metrics) == {"walk_forward", "turn_in_place"}
    assert set(report.zero_shot_metrics) == {"strafe_and_stop"}
    for metrics in list(report.scenario_metrics.values()) + list(
        report.zero_shot_metrics.values()
    ):
        assert set(metrics["horizons"]) == set(report.horizon_frames)
        assert "frozen_rollout" in metrics["rollout_health"]
    assert report.yaw_probe.get("n_samples", 0) > 0


def test_run_nursery_joint_rejects_overlapping_scenarios(tmp_path):
    with pytest.raises(ValueError, match="both trained and held out"):
        run_nursery_joint(
            str(tmp_path),
            train_scenarios=["turn_in_place"],
            holdout_scenarios=["turn_in_place"],
            config=_small_nursery_config(),
        )


# ---------------------------------------------------------------- issue #91
# Predictive Cortex: promotion to brain/cortex/predictive.py, merged
# multi-horizon + uncertainty heads, ticks-denominated horizons.


def test_build_action_world_model_returns_predictive_cortex(turn_session):
    dataset = build_action_sequence_dataset([turn_session])
    model = build_action_world_model(
        dataset.pixel_shape, dataset.action_keys, _small_model_config()
    )
    assert isinstance(model, PredictiveCortex)


def test_forward_horizons_yields_decoded_frame_and_heads_at_every_horizon(turn_session):
    dataset = build_action_sequence_dataset([turn_session])
    model, _stats = train_action_world_model(dataset, _small_model_config())
    episodes = _episode_tensors(dataset, model.reconstruction_shape)
    _episode, pixels, _targets, actions = episodes[0]
    assert actions.shape[0] >= 3

    with torch.no_grad():
        latents = model.encoder(pixels)
        hidden = model.initial_state(1)
        out = model.forward_horizons(
            latents[:1], actions[:3].unsqueeze(0), hidden, horizon_frames=[1, 3]
        )

    h, w, c = model.reconstruction_shape
    assert set(out.horizons) == {1, 3}
    for horizon, pred in out.horizons.items():
        # Decoder output is CHW (matches the pixel/reconstruction-target
        # convention elsewhere in this module), same input shape at every
        # horizon -- the "decoded-frame shape equals input shape" criterion.
        assert pred.decoded.shape == (1, c, h, w)
        assert pred.latent.shape == (1, model.latent_width)
        assert pred.reward.shape == (1,)
        assert pred.terminal_logit.shape == (1,)
        assert pred.risk.shape == (1,)
        assert pred.uncertainty.shape == (1,)
        assert bool((pred.uncertainty >= 0).all())
        assert bool((pred.risk >= 0).all())
    # Horizon 3's decoded frame is not just a repeat of horizon 1's: the
    # heads read a state that actually advanced through the rollout.
    assert not torch.allclose(out[1].latent, out[3].latent)


def test_forward_horizons_rejects_non_positive_horizon(turn_session):
    dataset = build_action_sequence_dataset([turn_session])
    model, _stats = train_action_world_model(dataset, _small_model_config())
    start_latent = torch.zeros(1, model.latent_width)
    actions = torch.zeros(1, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="positive"):
        model.forward_horizons(start_latent, actions, model.initial_state(1), horizon_frames=[0])


def test_forward_horizons_rejects_insufficient_actions(turn_session):
    dataset = build_action_sequence_dataset([turn_session])
    model, _stats = train_action_world_model(dataset, _small_model_config())
    start_latent = torch.zeros(1, model.latent_width)
    actions = torch.zeros(1, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="actions must cover"):
        model.forward_horizons(start_latent, actions, model.initial_state(1), horizon_frames=[5])


def test_default_horizons_ticks_is_1_4_8(turn_session):
    dataset = build_action_sequence_dataset([turn_session])
    model, _stats = train_action_world_model(dataset, _small_model_config())
    assert model.horizons_ticks == (1, 4, 8)


def test_horizons_ticks_persist_through_checkpoint_round_trip(turn_session, tmp_path):
    dataset = build_action_sequence_dataset([turn_session])
    cfg = _small_model_config(horizons_ticks=(1, 2, 5))
    model, stats = train_action_world_model(dataset, cfg)
    assert model.horizons_ticks == (1, 2, 5)

    path = os.path.join(str(tmp_path), "cortex.pt")
    save_action_world_model(path, model, stats)
    reloaded, _stats = load_action_world_model(path)
    assert reloaded.horizons_ticks == (1, 2, 5)
