from cognitive_runtime.training.action_effect_taxonomy import ACTION_EFFECT_CLASSES
from cognitive_runtime.training.event_evaluation import (
    FrameEventLabels, effect_class_stratified_metrics, evaluate_entity_predictions,
    extract_frame_event_labels, motion_stratified_metrics, rollout_health,
)
from cognitive_runtime.programs.crafter.streams import SEMANTIC_CLASS_IDS
from cognitive_runtime.training.statistical_evaluation import (
    build_experiment_report, load_experiment_report, write_experiment_report,
)


def test_semantic_events_detect_cow_entry_exit_and_blocked_not_idle():
    empty = [[0, 0], [0, 0]]
    cow = [[0, SEMANTIC_CLASS_IDS["entity"]], [0, 0]]
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


def test_action_effect_class_derivation_matches_mf_e1_precedence():
    """Issue #237: ``extract_frame_event_labels`` derives the same five-class
    MF-E1 taxonomy (epic #212 Sec 12.3) ``action_effects.py`` derives from
    raw session data, but world-agnostically from already-loaded frame
    signals -- one transition of each class, in classify_action_effect's own
    precedence order."""
    grid = [[0, 0], [0, 0]]
    grids = [grid] * 6
    positions = [(0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    actions = ["MOVE_UP", "MOVE_UP", "DO", "NULL", "NULL"]
    facings = [(0, 1), (0, 1), (0, 1), (0, 1), (1, 0), (1, 0)]
    labels = extract_frame_event_labels(
        grids, positions=positions, actions=actions, facings=facings,
    )
    assert labels[1].action_effect_class == "moved"
    assert labels[2].action_effect_class == "blocked"
    assert labels[3].action_effect_class == "interacted"
    assert labels[4].action_effect_class == "turned_only"
    assert labels[5].action_effect_class == "no_op"
    assert {label.action_effect_class for label in labels[1:]} == set(ACTION_EFFECT_CLASSES)


def test_effect_class_stratified_report_flags_blocked_regression_hidden_by_healthy_aggregate():
    """Issue #237's direct acceptance test: a synthetic model accurate on
    ``no_op`` frames and inaccurate on ``blocked`` frames must be visibly
    flagged by the per-effect-class report even though its pooled aggregate
    beats copy-last on average -- exactly the failure mode event-stratified
    evaluation exists to catch."""
    no_op_rows = [{"model_mse": 0.001, "copy_last_mse": 0.01} for _ in range(90)]
    blocked_rows = [{"model_mse": 0.05, "copy_last_mse": 0.01} for _ in range(10)]
    labels = (
        [FrameEventLabels(False, (), False, False, False, False, False, False, "no_op") for _ in no_op_rows]
        + [FrameEventLabels(False, (), False, False, False, True, False, False, "blocked") for _ in blocked_rows]
    )
    rows = no_op_rows + blocked_rows

    aggregate = motion_stratified_metrics({"all": rows})["all"]
    assert aggregate["model_over_copy_last"] < 1.0, "aggregate must look healthy for this to be a real regression test"

    stratified = effect_class_stratified_metrics(labels, rows)
    assert set(stratified) == set(ACTION_EFFECT_CLASSES)
    assert stratified["no_op"]["model_over_copy_last"] < 1.0
    assert stratified["blocked"]["model_over_copy_last"] > 1.0
    assert stratified["blocked"]["n_samples"] == 10
    assert stratified["no_op"]["n_samples"] == 90
    for other in ("moved", "turned_only", "interacted"):
        assert stratified[other]["n_samples"] == 0


def test_motion_counts_and_rollout_health_states():
    metrics = motion_stratified_metrics({"static": [{"model_mse": 1, "copy_last_mse": 2}], "moving": [{"model_mse": 4, "copy_last_mse": 2}]})
    assert metrics["static"]["n_samples"] + metrics["moving"]["n_samples"] == 2
    assert rollout_health(.5, 1)["state"] == "healthy"
    assert rollout_health(.1, 1)["state"] == "underdynamic"
    assert rollout_health(.01, 1)["state"] == "frozen"
    assert rollout_health(.01, 0)["state"] == "not_evaluable"


def test_experiment_report_round_trips_with_checkpoint_identity(tmp_path):
    report = build_experiment_report(
        experiment={"experiment_id": "smoke"},
        checkpoint={"path": "/models/smoke.pt", "sha256": "abc"},
        rollout_metrics={"horizons": {1: {"beats_copy_last": True}}, "event_metrics": {}},
    )
    path = write_experiment_report(str(tmp_path / "experiment_report.json"), report)
    assert load_experiment_report(path)["checkpoint"]["sha256"] == "abc"


def test_experiment_report_without_checkpoint_cannot_be_promoted():
    report = build_experiment_report(
        experiment={"experiment_id": "untracked"},
        rollout_metrics={"horizons": {1: {"beats_copy_last": True}}, "event_metrics": {}},
    )
    assert report["promotion_verdict"]["promoted"] is False
    assert "checkpoint identity requires path and sha256" in report["promotion_verdict"]["reasons"]


def test_training_replay_experiment_cannot_be_promoted():
    report = build_experiment_report(
        experiment={"experiment_id": "overfit"},
        training_stats={"evaluation_mode": "training_replay"},
        rollout_metrics={"horizons": {1: {"beats_copy_last": True}}, "event_metrics": {}},
    )
    assert report["promotion_verdict"]["promoted"] is False
    assert "training-replay evaluation cannot support promotion" in report["promotion_verdict"]["reasons"]
