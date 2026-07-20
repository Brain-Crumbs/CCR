"""Temporal-backbone A/B (issue #93, docs/v2/phases/phase-2-predictive-cortex.md
task 5): the dilated-conv/transformer backbones satisfy the same cortex
interface as the GRU default, the context-length curriculum actually ramps
the window, checkpoints round-trip the backbone choice, and the benchmark
harness reports the Phase 2 scoring gates for each backbone.

``tests/test_predictive_cortex.py``/``test_action_world_model.py`` cover the
GRU backbone's forward shape and the Milestone 2 scoring-gate correctness;
this file is deliberately backbone-parametrized so a regression that only
shows up for the windowed backbones (e.g. a shape mismatch in the ring
buffer) cannot hide behind GRU-only coverage.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from brain.cortex.backbones import (  # noqa: E402
    DilatedConvBackbone,
    GRUBackbone,
    TransformerBackbone,
    build_backbone,
)
from brain.cortex.predictive import PredictiveCortex, PredictiveCortexConfig  # noqa: E402
from cognitive_runtime.training.action_world_model import (  # noqa: E402
    ActionWorldModelConfig,
    build_action_sequence_dataset,
    evaluate_action_world_model,
    load_action_world_model,
    save_action_world_model,
    train_action_world_model,
)
from cognitive_runtime.training.nursery import (  # noqa: E402
    NURSERY_SCENARIOS,
    NurseryConfig,
    _record_scenario_episode,
    run_backbone_benchmark,
)

BACKBONES = ["gru", "dilated_conv", "transformer"]
PIXEL_SHAPE = (4, 4, 3)
ACTION_KEYS = ["NULL", "LEFT_TURN", "RIGHT_TURN"]


def _small_nursery_config(**overrides) -> NurseryConfig:
    base = dict(
        train_seeds=(0, 1),
        holdout_seeds=(1000,),
        episode_ticks=26,
        world_size=16,
        horizons=(1, 3),
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
        latent_width=16, hidden_dim=32, reconstruction_size=8,
        epochs=2, batch_size=16, warmup_frames=2, rollout_frames=3, context_length=4,
    )
    base.update(overrides)
    return ActionWorldModelConfig(**base)


@pytest.fixture(scope="module")
def turn_session(tmp_path_factory):
    root = tmp_path_factory.mktemp("backbone-sessions")
    cfg = _small_nursery_config()
    return _record_scenario_episode(
        str(root), "backbone-turn", 0, NURSERY_SCENARIOS["turn_in_place"], cfg
    )


# --------------------------------------------------------------- backbone unit contract


@pytest.mark.parametrize("name", BACKBONES)
def test_backbone_satisfies_step_readout_contract(name):
    """Every backbone's ``step`` returns a ``(hidden, state)`` pair where
    ``readout(state)`` recovers exactly that hidden -- the contract
    :class:`PredictiveCortex` relies on to stay backbone-agnostic."""
    backbone = build_backbone(name, input_dim=10, hidden_dim=12, context_length=5)
    state = backbone.initial_state(3)
    x = torch.randn(3, 10)
    hidden, next_state = backbone.step(x, state)
    assert hidden.shape == (3, 12)
    assert torch.equal(backbone.readout(next_state), hidden)


@pytest.mark.parametrize("name", BACKBONES)
def test_backbone_forward_sequence_returns_all_causal_positions(name):
    backbone = build_backbone(name, input_dim=10, hidden_dim=12, context_length=5)
    backbone.eval()
    inputs = torch.randn(2, 7, 10)
    with torch.no_grad():
        output = backbone.forward_sequence(inputs)
        changed_future = inputs.clone()
        changed_future[:, 5:] += 100.0
        output_changed = backbone.forward_sequence(changed_future)
    assert output.shape == (2, 7, 12)
    # Causality: changing tokens after position 4 cannot alter its prefix.
    assert torch.allclose(output[:, :5], output_changed[:, :5])


@pytest.mark.parametrize("name", BACKBONES)
def test_backbone_context_length_max_matches_windowed_or_unbounded(name):
    backbone = build_backbone(name, input_dim=6, hidden_dim=8, context_length=5)
    if name == "gru":
        assert backbone.context_length_max is None
        assert isinstance(backbone, GRUBackbone)
    else:
        assert backbone.context_length_max == 5
        assert isinstance(backbone, (DilatedConvBackbone, TransformerBackbone))


def test_build_backbone_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown cortex backbone"):
        build_backbone("lstm", input_dim=4, hidden_dim=4)


@pytest.mark.parametrize("name", ["dilated_conv", "transformer"])
def test_windowed_backbone_set_context_length_clamps_to_max(name):
    backbone = build_backbone(name, input_dim=6, hidden_dim=8, context_length=4)
    backbone.set_context_length(100)
    assert backbone._current_context == 4
    backbone.set_context_length(1)
    assert backbone._current_context == 1
    backbone.set_context_length(0)
    assert backbone._current_context == 1


# --------------------------------------------------------------- cortex-level contract


@pytest.mark.parametrize("name", BACKBONES)
def test_predictive_cortex_forward_horizons_identical_shape_across_backbones(name):
    """Task 5's acceptance criterion in miniature: the backbone is
    selectable by config, and one forward pass yields the same decoded +
    reward/terminal/risk/uncertainty structure at every horizon regardless
    of which backbone produced it."""
    cfg = PredictiveCortexConfig(
        latent_width=8, hidden_dim=12, reconstruction_size=4, backbone=name, context_length=4,
    )
    model = PredictiveCortex(PIXEL_SHAPE, ACTION_KEYS, cfg)
    batch = 2
    hidden = model.initial_state(batch)
    latent = torch.randn(batch, 8)
    actions = torch.randint(0, len(ACTION_KEYS), (batch, 6))

    out = model.forward_horizons(latent, actions, hidden, horizon_frames=[1, 3, 6])
    assert set(out.horizons) == {1, 3, 6}
    decoded_shape = (batch, model.reconstruction_shape[2], *model.reconstruction_shape[:2])
    for pred in out.horizons.values():
        assert pred.latent.shape == (batch, 8)
        assert tuple(pred.decoded.shape) == decoded_shape
        assert pred.reward.shape == (batch,)
        assert pred.terminal_logit.shape == (batch,)
        assert pred.risk.shape == (batch,)
        assert pred.uncertainty.shape == (batch,)
        assert bool((pred.risk >= 0).all())
        assert bool((pred.uncertainty >= 0).all())

    meta = model.checkpoint_metadata()
    assert meta["backbone"] == name
    assert meta["context_length"] == 4


@pytest.mark.parametrize("name", BACKBONES)
def test_predictive_cortex_sequence_heads_predict_every_position(name):
    cfg = PredictiveCortexConfig(
        latent_width=8, hidden_dim=12, reconstruction_size=4,
        horizons_ticks=(1, 3), backbone=name, context_length=4,
    )
    model = PredictiveCortex(PIXEL_SHAPE, ACTION_KEYS, cfg)
    hidden = model.forward_sequence(
        torch.randn(2, 6, 8), torch.randint(0, len(ACTION_KEYS), (2, 6))
    )
    assert hidden.shape == (2, 6, 12)
    for horizon in (1, 3):
        prediction = model.sequence_prediction(hidden, horizon)
        assert prediction.latent.shape == (2, 6, 8)
        assert prediction.decoded.shape[:2] == (2, 6)
        assert prediction.reward.shape == (2, 6)


@pytest.mark.parametrize("name", BACKBONES)
def test_predictive_cortex_rollout_and_step_are_the_same_state_object(name):
    """``rollout`` and repeated ``step`` calls must agree: rollout is just
    ``step`` called in a loop, for any backbone."""
    cfg = PredictiveCortexConfig(
        latent_width=6, hidden_dim=8, reconstruction_size=4, backbone=name, context_length=3,
    )
    model = PredictiveCortex(PIXEL_SHAPE, ACTION_KEYS, cfg)
    model.eval()
    batch = 2
    hidden = model.initial_state(batch)
    latent = torch.randn(batch, 6)
    actions = torch.randint(0, len(ACTION_KEYS), (batch, 4))

    with torch.no_grad():
        rolled, rollout_hidden = model.rollout(latent, actions, hidden)

        stepped = []
        step_latent, step_hidden = latent, hidden
        for i in range(actions.shape[1]):
            step_latent, step_hidden = model.step(step_latent, actions[:, i], step_hidden)
            stepped.append(step_latent)
        stepped = torch.stack(stepped, dim=1)

    assert torch.allclose(rolled, stepped)
    assert torch.allclose(
        model.transition_backbone.readout(rollout_hidden),
        model.transition_backbone.readout(step_hidden),
    )


# --------------------------------------------------------------- train/evaluate parity


@pytest.mark.parametrize("name", BACKBONES)
def test_train_and_evaluate_report_identical_structure_across_backbones(turn_session, name):
    """Both backbones train and evaluate through the identical cortex
    interface (task 5's acceptance criterion): ``evaluate_action_world_model``
    returns the same structured-report shape -- the Phase 2 scoring gates --
    no matter which backbone produced the model."""
    dataset = build_action_sequence_dataset([turn_session])
    model_cfg = _small_model_config(backbone=name)
    model, stats = train_action_world_model(dataset, model_cfg)
    assert stats["final_total_loss"] >= 0.0

    report = evaluate_action_world_model(model, dataset, [1, 3], warmup_frames=2)
    assert set(report["horizons"]) == {1, 3}
    for entry in report["horizons"].values():
        assert set(entry) >= {
            "n_samples", "model_mse", "copy_last_mse", "mean_frame_mse", "oracle_mse",
            "psnr_model", "model_over_copy_last_mse", "model_over_oracle_mse",
            "beats_copy_last", "beats_mean_frame",
        }
        assert entry["n_samples"] > 0
    assert set(report["rollout_health"]) >= {
        "prediction_dispersion", "target_dispersion", "frozen_rollout",
    }


@pytest.mark.parametrize("name", BACKBONES)
def test_autoregressive_objective_trains_every_backbone(turn_session, name):
    dataset = build_action_sequence_dataset([turn_session])
    cfg = _small_model_config(
        backbone=name, training_objective="autoregressive", horizons_ticks=(1, 3),
    )
    _model, stats = train_action_world_model(dataset, cfg)
    assert stats["training_objective"] == "autoregressive"
    assert stats["autoregressive_horizons"] == [1, 3]
    assert len(stats["loss_curves"]["closed_loop_loss"]) == cfg.epochs


# --------------------------------------------------------------- context-length curriculum


@pytest.mark.parametrize("name", ["dilated_conv", "transformer"])
def test_context_length_curriculum_ramps_from_one_to_max(turn_session, name, monkeypatch):
    calls = []
    original = PredictiveCortex.set_context_length

    def spy(self, n):
        calls.append(n)
        return original(self, n)

    monkeypatch.setattr(PredictiveCortex, "set_context_length", spy)

    dataset = build_action_sequence_dataset([turn_session])
    cfg = _small_model_config(backbone=name, context_length=6, epochs=4)
    train_action_world_model(dataset, cfg)

    assert calls, "curriculum should call set_context_length at least once per epoch"
    assert calls[0] == 1
    assert calls[-1] == 6
    assert calls == sorted(calls)


def test_context_length_curriculum_is_a_no_op_for_gru(turn_session, monkeypatch):
    calls = []
    original = PredictiveCortex.set_context_length

    def spy(self, n):
        calls.append(n)
        return original(self, n)

    monkeypatch.setattr(PredictiveCortex, "set_context_length", spy)

    dataset = build_action_sequence_dataset([turn_session])
    cfg = _small_model_config(backbone="gru", epochs=4)
    train_action_world_model(dataset, cfg)

    assert calls == []


@pytest.mark.parametrize("name", ["dilated_conv", "transformer"])
def test_context_length_curriculum_disabled_leaves_full_window(turn_session, name, monkeypatch):
    calls = []
    original = PredictiveCortex.set_context_length

    def spy(self, n):
        calls.append(n)
        return original(self, n)

    monkeypatch.setattr(PredictiveCortex, "set_context_length", spy)

    dataset = build_action_sequence_dataset([turn_session])
    cfg = _small_model_config(backbone=name, context_length=6, epochs=3, context_length_curriculum=False)
    train_action_world_model(dataset, cfg)

    assert calls == []


# --------------------------------------------------------------- checkpoint round-trip


@pytest.mark.parametrize("name", ["dilated_conv", "transformer"])
def test_checkpoint_round_trips_backbone_choice(turn_session, name, tmp_path):
    dataset = build_action_sequence_dataset([turn_session])
    model, stats = train_action_world_model(dataset, _small_model_config(backbone=name))
    path = str(tmp_path / "model.pt")
    save_action_world_model(path, model, stats)

    loaded, _stats = load_action_world_model(path)
    assert loaded.config.backbone == name
    assert loaded.config.context_length == model.config.context_length
    assert type(loaded.transition_backbone) is type(model.transition_backbone)

    loaded.eval()
    model.eval()
    report_before = evaluate_action_world_model(model, dataset, [1], warmup_frames=2)
    report_after = evaluate_action_world_model(loaded, dataset, [1], warmup_frames=2)
    assert report_before["horizons"][1]["model_mse"] == pytest.approx(
        report_after["horizons"][1]["model_mse"], rel=1e-5
    )


# --------------------------------------------------------------- benchmark harness


def test_run_backbone_benchmark_reports_phase_2_gates_for_each_backbone(tmp_path):
    cfg = _small_nursery_config(episode_ticks=26, horizons=(1,))
    report = run_backbone_benchmark(
        str(tmp_path),
        train_scenarios=["walk_forward", "turn_in_place"],
        eval_scenario="turn_in_place",
        backbones=("gru", "dilated_conv"),
        baseline_backbone="gru",
        config=cfg,
        model_config=_small_model_config(),
    )
    assert set(report.metrics) == {"gru", "dilated_conv"}
    for name, metrics in report.metrics.items():
        assert set(metrics["horizons"]) == {1}
        entry = metrics["horizons"][1]
        assert "model_over_copy_last_mse" in entry
        assert "model_over_oracle_mse" in entry
        assert isinstance(report.beats_copy_last[name][1], bool)
    assert set(report.comparisons) == {"dilated_conv"}
    for comparison in report.comparisons["dilated_conv"].values():
        assert comparison.direction in {"improved", "regressed", "no_significant_difference"}


def test_run_backbone_benchmark_rejects_eval_scenario_outside_train_scenarios(tmp_path):
    with pytest.raises(ValueError, match="must be one of the trained scenarios"):
        run_backbone_benchmark(
            str(tmp_path),
            train_scenarios=["walk_forward"],
            eval_scenario="turn_in_place",
            config=_small_nursery_config(),
        )


def test_run_backbone_benchmark_rejects_baseline_outside_backbones(tmp_path):
    with pytest.raises(ValueError, match="baseline_backbone"):
        run_backbone_benchmark(
            str(tmp_path),
            train_scenarios=["walk_forward"],
            eval_scenario="walk_forward",
            backbones=("gru",),
            baseline_backbone="transformer",
            config=_small_nursery_config(),
        )
