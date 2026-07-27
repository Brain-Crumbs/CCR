from cognitive_runtime.training.event_evaluation import (
    FrameEventLabels, evaluate_entity_predictions, extract_frame_event_labels,
    motion_stratified_metrics, rollout_health,
)
from cognitive_runtime.training.statistical_evaluation import (
    build_experiment_report, load_experiment_report, write_experiment_report,
)


def test_semantic_events_detect_cow_entry_exit_and_blocked_not_idle():
    empty = [[0, 0], [0, 0]]
    cow = [[0, 14], [0, 0]]
    labels = extract_frame_event_labels(
        [empty, cow, empty, empty], positions=[(1, 1)] * 4,
        actions=["MOVE_UP", "MOVE_UP", "MOVE_UP"],
    )
    assert labels[1].entity_entered and labels[1].cow_cells == ((0, 1),)
    assert labels[2].entity_left
    assert labels[3].blocked_forward
    idle = extract_frame_event_labels([empty, empty], positions=[(1, 1), (1, 1)], actions=["NULL"])
    assert not idle[1].blocked_forward


def test_entity_metrics_count_hallucination_and_entry_latency():
    labels = [
        FrameEventLabels(False, (), False, False, False, False, False),
        FrameEventLabels(True, ((1, 1),), True, False, True, False, True),
        FrameEventLabels(False, (), False, True, True, False, True),
    ]
    report = evaluate_entity_predictions(labels, [.9, .1, .8], [[], [], []])
    assert report["false_positive"] == 2
    assert report["entity_entry_incorporation_latency"] == 1
    assert report["entity_disappearance_persistence_duration"] == 1


def test_motion_counts_and_rollout_health_states():
    metrics = motion_stratified_metrics({"static": [{"model_mse": 1, "copy_last_mse": 2}], "moving": [{"model_mse": 4, "copy_last_mse": 2}]})
    assert metrics["static"]["n_samples"] + metrics["moving"]["n_samples"] == 2
    assert rollout_health(.5, 1)["state"] == "healthy"
    assert rollout_health(.1, 1)["state"] == "underdynamic"
    assert rollout_health(.01, 1)["state"] == "frozen"
    assert rollout_health(.01, 0)["state"] == "not_evaluable"


def test_experiment_report_round_trips_with_checkpoint_identity(tmp_path):
    report = build_experiment_report(
        experiment={"experiment_id": "smoke"}, checkpoint={"sha256": "abc"},
        rollout_metrics={"horizons": {1: {"beats_copy_last": True}}, "event_metrics": {}},
    )
    path = write_experiment_report(str(tmp_path / "experiment_report.json"), report)
    assert load_experiment_report(path)["checkpoint"]["sha256"] == "abc"


def test_training_replay_experiment_cannot_be_promoted():
    report = build_experiment_report(
        experiment={"experiment_id": "overfit"},
        training_stats={"evaluation_mode": "training_replay"},
        rollout_metrics={"horizons": {1: {"beats_copy_last": True}}, "event_metrics": {}},
    )
    assert report["promotion_verdict"]["promoted"] is False
    assert "training-replay evaluation cannot support promotion" in report["promotion_verdict"]["reasons"]
