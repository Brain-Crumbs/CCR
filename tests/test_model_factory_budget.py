"""Regression coverage for Model Factory graceful budget enforcement (#221)."""

from __future__ import annotations

import pytest

from cognitive_runtime.training.model_factory.budget import (
    BUDGET_REPORT_FORMAT,
    STATUS_BUDGET_EXCEEDED,
    STATUS_COMPLETED,
    STATUS_TIMEOUT_UNRECOVERABLE,
    TIMING_POLICY_VERSION,
    BudgetDecision,
    EpochTimer,
    TrainingBudget,
    budget_seconds_for_tier,
    build_budget_report,
    watchdog_expired,
    write_budget_report,
)
from cognitive_runtime.training.statistical_evaluation import build_experiment_report


def test_budget_tiers_match_epic_15_1():
    assert budget_seconds_for_tier("fast") == 600.0
    assert budget_seconds_for_tier("scale") == 1200.0
    with pytest.raises(ValueError, match="unknown runtime budget tier"):
        budget_seconds_for_tier("nope")


def test_slow_warmup_epoch_does_not_trigger_premature_stop():
    """A pessimistic first-epoch ETA must not stop a run that is actually fine."""
    budget = TrainingBudget(
        max_training_seconds=budget_seconds_for_tier("fast"), total_epochs=21,
        warmup_epochs=1, checkpoint_cadence_epochs=5,
    )

    warmup_decision = budget.record_epoch(1, 55.0)  # CUDA warm-up + data priming
    assert warmup_decision.median_epoch_seconds is None  # excluded, not yet enough data
    assert not warmup_decision.stop_requested
    assert not warmup_decision.should_stop_now

    # A naive "first-epoch ETA" projection would already have blown the budget,
    # even though the run's true total (below) stays comfortably within it.
    naive_projection = 55.0 * 21
    assert naive_projection > budget.max_training_seconds

    decisions = [budget.record_epoch(epoch, 10.0) for epoch in range(2, 22)]
    assert all(not d.stop_requested for d in decisions)
    assert all(not d.should_stop_now for d in decisions)
    assert budget.elapsed_seconds == 55.0 + 10.0 * 20
    assert budget.median_epoch_seconds() == 10.0
    # The full-run projection (not just a one-epoch lookahead) confirms the
    # steady-state trajectory finishes comfortably inside the budget.
    assert decisions[-1].projected_seconds == budget.elapsed_seconds


def test_over_budget_run_requests_stop_as_soon_as_the_full_run_projection_overruns():
    """The projection covers every remaining epoch, so a genuine overrun is
    caught immediately after warm-up -- not only once elapsed time itself
    approaches the deadline (which a one-epoch lookahead would delay)."""
    budget = TrainingBudget(
        max_training_seconds=100.0, total_epochs=10, warmup_epochs=1, checkpoint_cadence_epochs=4,
    )

    budget.record_epoch(1, 5.0)  # warm-up, excluded from the ETA
    d2 = budget.record_epoch(2, 40.0)  # elapsed=45, median=40, 8 epochs remain: projected=365 > 100
    assert d2.projected_seconds == 45.0 + 40.0 * 8
    assert d2.stop_requested
    assert not d2.at_checkpoint_boundary  # epoch 2 is not a multiple of the cadence
    assert not d2.should_stop_now  # requested, but not yet at a safe boundary
    assert not d2.hard_deadline_exceeded

    d3 = budget.record_epoch(3, 40.0)  # elapsed=85, still not a checkpoint boundary
    assert d3.stop_requested
    assert not d3.at_checkpoint_boundary
    assert not d3.should_stop_now
    assert not d3.hard_deadline_exceeded

    d4 = budget.record_epoch(4, 5.0)  # elapsed=90, epoch 4 is a checkpoint boundary
    assert d4.stop_requested  # sticky
    assert d4.at_checkpoint_boundary
    assert d4.should_stop_now
    assert not d4.hard_deadline_exceeded

    assert isinstance(d4, BudgetDecision)


def test_projection_is_none_without_a_declared_total_epochs():
    """Without ``total_epochs`` there is no honest full-run projection, so no
    graceful stop is ever requested -- only the hard deadline still applies."""
    budget = TrainingBudget(max_training_seconds=100.0, warmup_epochs=1, checkpoint_cadence_epochs=4)

    budget.record_epoch(1, 5.0)
    d2 = budget.record_epoch(2, 40.0)

    assert d2.median_epoch_seconds == 40.0
    assert d2.projected_seconds is None
    assert not d2.stop_requested


def test_hard_deadline_enforced_even_off_checkpoint_cadence():
    """The hard deadline stops training even when no cadence boundary is due."""
    budget = TrainingBudget(max_training_seconds=50.0, warmup_epochs=0, checkpoint_cadence_epochs=10)

    decision = budget.record_epoch(1, 60.0)

    assert decision.hard_deadline_exceeded
    assert decision.should_stop_now
    assert not decision.at_checkpoint_boundary


def test_record_epoch_rejects_negative_duration():
    budget = TrainingBudget(max_training_seconds=10.0)
    with pytest.raises(ValueError, match="must not be negative"):
        budget.record_epoch(1, -1.0)


def test_epoch_timer_and_elapsed_seconds_exclude_untimed_gaps():
    """``measured_training_seconds`` only ever sums timed epoch blocks --
    time spent resolving the corpus or running validation between them, which
    the caller never wraps in :class:`EpochTimer`, must not be counted."""
    clock_values = iter([0.0, 5.0, 100.0, 103.0])

    def fake_clock() -> float:
        return next(clock_values)

    budget = TrainingBudget(max_training_seconds=600.0, warmup_epochs=0)

    with EpochTimer(clock=fake_clock) as timer:
        pass  # simulated epoch-1 training work
    budget.record_epoch(1, timer.duration_seconds)

    # An untimed corpus/validation gap of 95s (in fake-clock ticks) happens
    # here and is never passed to record_epoch.

    with EpochTimer(clock=fake_clock) as timer:
        pass  # simulated epoch-2 training work
    budget.record_epoch(2, timer.duration_seconds)

    assert budget.elapsed_seconds == 5.0 + 3.0


def test_watchdog_timeout_is_distinct_from_budget_exceeded():
    assert watchdog_expired(seconds_since_heartbeat=301.0, grace_period_seconds=300.0)
    assert not watchdog_expired(seconds_since_heartbeat=100.0, grace_period_seconds=300.0)
    with pytest.raises(ValueError, match="grace_period_seconds"):
        watchdog_expired(seconds_since_heartbeat=1.0, grace_period_seconds=0.0)

    # The two statuses are distinct strings recorded by different mechanisms:
    # a live-but-slow run's TrainingBudget never produces "timeout_unrecoverable",
    # and a watchdog's stall detection never produces "budget_exceeded".
    assert STATUS_BUDGET_EXCEEDED != STATUS_TIMEOUT_UNRECOVERABLE


def test_build_budget_report_records_required_15_2_fields():
    budget = TrainingBudget(max_training_seconds=600.0, warmup_epochs=1, checkpoint_cadence_epochs=4)
    for epoch, duration in [(1, 90.0), (2, 200.0), (3, 200.0), (4, 200.0)]:
        budget.record_epoch(epoch, duration)

    report = build_budget_report(
        budget, completion_status=STATUS_BUDGET_EXCEEDED, total_trial_seconds=750.0,
        precision="fp32", tier="fast", hardware_profile={"device": "cpu"},
    )

    assert report["format"] == BUDGET_REPORT_FORMAT
    assert report["tier"] == "fast"
    assert report["max_training_seconds"] == 600.0
    assert report["timing_policy_version"] == TIMING_POLICY_VERSION
    assert report["measured_training_seconds"] == budget.elapsed_seconds
    assert report["total_trial_seconds"] == 750.0
    assert report["hardware_profile"] == {"device": "cpu"}
    assert report["precision"] == "fp32"
    assert report["completion_status"] == STATUS_BUDGET_EXCEEDED
    # measured_training_seconds excludes corpus resolution and validation:
    # total_trial_seconds (elapsed process time) must be strictly larger.
    assert report["total_trial_seconds"] > report["measured_training_seconds"]


def test_build_budget_report_rejects_unknown_status_and_impossible_totals():
    budget = TrainingBudget(max_training_seconds=600.0)
    budget.record_epoch(1, 10.0)
    with pytest.raises(ValueError, match="unknown completion status"):
        build_budget_report(
            budget, completion_status="not-a-status", total_trial_seconds=20.0, precision="fp32",
        )
    with pytest.raises(ValueError, match="total_trial_seconds"):
        build_budget_report(
            budget, completion_status=STATUS_COMPLETED, total_trial_seconds=1.0, precision="fp32",
        )


def test_build_budget_report_rejects_a_completed_report_that_ran_past_its_cap():
    """A report cannot claim completed while measured time exceeds the cap --
    that combination is self-contradictory and would slip past a promotion
    gate that only checks the literal completion_status string."""
    budget = TrainingBudget(max_training_seconds=100.0)
    budget.record_epoch(1, 60.0)
    budget.record_epoch(2, 60.0)  # elapsed=120 > max_training_seconds=100

    with pytest.raises(ValueError, match="completion_status='completed'"):
        build_budget_report(
            budget, completion_status=STATUS_COMPLETED, total_trial_seconds=150.0, precision="fp32",
        )

    # The honest status for the same timings is accepted without complaint.
    report = build_budget_report(
        budget, completion_status=STATUS_BUDGET_EXCEEDED, total_trial_seconds=150.0, precision="fp32",
    )
    assert report["completion_status"] == STATUS_BUDGET_EXCEEDED


def test_write_budget_report_round_trips(tmp_path):
    budget = TrainingBudget(max_training_seconds=600.0)
    budget.record_epoch(1, 10.0)
    report = build_budget_report(
        budget, completion_status=STATUS_COMPLETED, total_trial_seconds=15.0, precision="fp32",
    )
    path = write_budget_report(tmp_path / "budget_report.json", report)

    import json

    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_budget_exceeded_trial_writes_a_complete_inspectable_partial_report(tmp_path):
    """AC: a budget_exceeded trial still writes a complete partial report."""
    budget = TrainingBudget(max_training_seconds=100.0, warmup_epochs=1, checkpoint_cadence_epochs=2)
    budget.record_epoch(1, 20.0)
    decision = budget.record_epoch(2, 90.0)
    assert decision.should_stop_now

    budget_report = build_budget_report(
        budget, completion_status=STATUS_BUDGET_EXCEEDED, total_trial_seconds=130.0,
        precision="fp32", tier="fast",
    )
    report = build_experiment_report(
        experiment={"experiment_id": "budget-exceeded-run"},
        checkpoint={"path": "/models/last.pt", "sha256": "deadbeef"},
        training_stats={**budget_report, "epoch": 2},
        # No rollout metrics: validation never ran before the hard deadline.
        rollout_metrics={},
    )
    path = write_budget_report(tmp_path / "trial_budget_report.json", budget_report)

    from cognitive_runtime.training.statistical_evaluation import (
        load_experiment_report, write_experiment_report,
    )

    experiment_report_path = write_experiment_report(str(tmp_path / "experiment_report.json"), report)
    loaded = load_experiment_report(experiment_report_path)

    assert path.is_file()
    assert loaded["checkpoint"]["sha256"] == "deadbeef"
    assert loaded["training_stats"]["completion_status"] == STATUS_BUDGET_EXCEEDED
    assert loaded["training_stats"]["measured_training_seconds"] == budget.elapsed_seconds


def test_budget_exceeded_experiment_is_never_eligible_for_promotion():
    """AC: assert against the promotion engine's input contract."""
    report = build_experiment_report(
        experiment={"experiment_id": "budget-exceeded-run"},
        checkpoint={"path": "/models/last.pt", "sha256": "deadbeef"},
        training_stats={"completion_status": STATUS_BUDGET_EXCEEDED},
        rollout_metrics={"horizons": {1: {"beats_copy_last": True}}, "event_metrics": {}},
    )

    assert report["promotion_verdict"]["promoted"] is False
    assert "budget_exceeded trial cannot support promotion" in report["promotion_verdict"]["reasons"]


def test_completed_experiment_with_same_shape_is_still_promotable():
    """The gate blocks non-"completed" statuses, not every completion_status."""
    report = build_experiment_report(
        experiment={"experiment_id": "completed-run"},
        checkpoint={"path": "/models/last.pt", "sha256": "deadbeef"},
        training_stats={"completion_status": STATUS_COMPLETED},
        rollout_metrics={"horizons": {1: {"beats_copy_last": True}}, "event_metrics": {}},
    )

    assert report["promotion_verdict"]["promoted"] is True


@pytest.mark.parametrize("completion_status", [STATUS_TIMEOUT_UNRECOVERABLE, "failed", "cancelled"])
def test_every_incomplete_completion_status_is_refused_promotion(completion_status):
    """Not just budget_exceeded: any declared non-"completed" terminal status
    (a watchdog kill, a crash, an operator cancellation, ...) must be refused
    even if it happens to carry a valid checkpoint and passing metrics."""
    report = build_experiment_report(
        experiment={"experiment_id": "incomplete-run"},
        checkpoint={"path": "/models/last.pt", "sha256": "deadbeef"},
        training_stats={"completion_status": completion_status},
        rollout_metrics={"horizons": {1: {"beats_copy_last": True}}, "event_metrics": {}},
    )

    assert report["promotion_verdict"]["promoted"] is False
    assert f"{completion_status} trial cannot support promotion" in report["promotion_verdict"]["reasons"]


@pytest.mark.parametrize("torch_available", [True])
def test_budget_exceeded_checkpoint_is_valid_and_loadable(tmp_path, torch_available):
    """AC: a genuinely over-budget run saves a valid, loadable last.pt with
    reason budget_exceeded at a checkpoint boundary."""
    torch = pytest.importorskip("torch")
    from brain.cortex.predictive import PredictiveCortex, PredictiveCortexConfig
    from cognitive_runtime.training.model_factory.checkpoint import (
        load_factory_checkpoint, save_factory_checkpoint,
    )
    from cognitive_runtime.training.model_factory.contracts import (
        ArchitectureContract, DataContract, TrainingContract,
    )

    model = PredictiveCortex(
        (4, 4, 3), ["NULL", "LEFT"], PredictiveCortexConfig(
            latent_width=4, hidden_dim=8, action_embed_dim=3, reconstruction_size=4,
            horizons_ticks=(1, 2), backbone="transformer", context_length=3,
            backbone_kwargs={"n_heads": 2, "n_layers": 1},
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    architecture = ArchitectureContract(
        pixel_shape=tuple(model.pixel_shape), rgb_preprocessing_version="v1",
        action_vocabulary=tuple(model.action_keys), workspace_modalities={},
        workspace_layout_hash=None, latent_width=model.latent_width,
        hidden_dim=model.hidden_dim, action_embed_dim=model.config.action_embed_dim,
        reconstruction_shape=tuple(model.reconstruction_shape),
        visual_architecture=model.visual_architecture,
        semantic_classes=model.config.semantic_classes,
        horizons_ticks=tuple(model.horizons_ticks),
        direct_horizon_topology=tuple(model.horizons_ticks),
        backbone=model.config.backbone, context_length=model.config.context_length,
        backbone_kwargs=dict(model.config.backbone_kwargs),
    )
    data = DataContract(
        world="test", backend="test", program_config={}, scenario_names=(),
        scenario_code_version="v1", train_session_ids=(), train_session_hashes=(),
        validation_session_ids=(), validation_session_hashes=(), test_session_ids=(),
        test_session_hashes=(), seed_assignments={}, pixel_provenance="test",
        semantic_vocabulary_version="v1", preprocessing_version="v1",
        horizons_ticks=tuple(model.horizons_ticks), ticks_per_frame=1.0,
    )
    training = TrainingContract(
        objective="test", optimizer={"name": "AdamW", "lr": 0.01}, batch_size=2, seed=7,
        rollout_frames=2, warmup_frames=1, scheduled_sampling_p=0.0, loss_weights={},
        checkpoint_selection_policy={}, device="cpu", precision="fp32",
        determinism_policy={"seed": 7}, max_training_seconds=100.0,
        checkpoint_cadence_epochs=2,
    )

    budget = TrainingBudget(
        max_training_seconds=training.max_training_seconds, warmup_epochs=1,
        checkpoint_cadence_epochs=training.checkpoint_cadence_epochs,
    )
    budget.record_epoch(1, 20.0)
    decision = budget.record_epoch(2, 90.0)
    assert decision.should_stop_now

    budget_report = build_budget_report(
        budget, completion_status=STATUS_BUDGET_EXCEEDED, total_trial_seconds=130.0,
        precision="fp32", tier="fast",
    )

    path = tmp_path / "last.pt"
    save_factory_checkpoint(
        str(path), model, optimizer,
        trainer_state={"epoch": 2, "global_step": 2, "best_validation_metric": None},
        architecture_contract=architecture, data_contract_hash=data, training_contract=training,
        training_stats=budget_report,
    )

    loaded = load_factory_checkpoint(str(path))

    assert loaded.trainer_state["epoch"] == 2
    assert loaded.training_stats["completion_status"] == STATUS_BUDGET_EXCEEDED
    assert loaded.training_stats["measured_training_seconds"] == budget.elapsed_seconds
    assert loaded.model is not None
