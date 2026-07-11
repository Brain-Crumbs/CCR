"""Curriculum runner (issue #43): staged goals with metric-gated promotion.

The definition loader/validation tests are torch-free (mirroring
tests/test_curriculum.py and tests/test_reward_profile.py); the actual
train/evaluate/promote runner is torch-gated like tests/test_evaluation_gates.py.
"""

from __future__ import annotations

import pytest

from cognitive_runtime.training.curriculum_runner import (
    CurriculumDefinition,
    CurriculumDefinitionError,
    CurriculumStageSpec,
    PromotionCriteria,
    curriculum_definition_from_dict,
    load_curriculum_definition,
)

TOY_CURRICULUM_PATH = "goals/curricula/toy_two_stage.yaml"

_BASE_WORLD = {"world_size": 32, "episode_ticks": 100, "difficulty": 0.0, "max_mobs": 0}


def _stage(name: str, **overrides) -> dict:
    stage = {
        "name": name,
        "world_config": dict(_BASE_WORLD),
        "train_episodes": 1,
        "promotion": {"metric": "average_ticks", "threshold": 0.0, "sample_size": 1},
        "max_attempts": 1,
    }
    stage.update(overrides)
    return stage


def test_promotion_criteria_rejects_unknown_metric():
    with pytest.raises(ValueError, match="unknown promotion metric"):
        PromotionCriteria(metric="not-a-metric")


def test_promotion_criteria_rejects_non_positive_sample_size():
    with pytest.raises(ValueError, match="sample_size"):
        PromotionCriteria(sample_size=0)


def test_stage_spec_rejects_reward_config_and_profile_together():
    with pytest.raises(ValueError, match="mutually exclusive"):
        CurriculumStageSpec(
            name="s", reward_config={"tick_alive": 0.1}, reward_profile_path="goals/survival.yaml",
        )


def test_curriculum_definition_from_dict_builds_ordered_stages():
    definition = curriculum_definition_from_dict(
        {"name": "toy", "stages": [_stage("a"), _stage("b")]}
    )
    assert definition.name == "toy"
    assert [s.name for s in definition.stages] == ["a", "b"]
    assert definition.index_of("b") == 1


def test_curriculum_definition_rejects_no_stages():
    with pytest.raises(CurriculumDefinitionError, match="non-empty list"):
        curriculum_definition_from_dict({"name": "toy", "stages": []})


def test_curriculum_definition_rejects_duplicate_stage_names():
    with pytest.raises(CurriculumDefinitionError, match="duplicate stage names"):
        curriculum_definition_from_dict({"name": "toy", "stages": [_stage("a"), _stage("a")]})


def test_curriculum_definition_rejects_unknown_top_level_field():
    with pytest.raises(CurriculumDefinitionError, match="unknown top-level field"):
        curriculum_definition_from_dict({"name": "toy", "stages": [_stage("a")], "bogus": 1})


def test_curriculum_definition_rejects_mismatched_world_size():
    """A curriculum runner checkpoint can't carry across a stream-layout
    change, so a differing world_size across stages must fail at load time."""
    stage_a = _stage("a")
    stage_b = _stage("b", world_config={**_BASE_WORLD, "world_size": 64})
    with pytest.raises(CurriculumDefinitionError, match="disagree on stream layout"):
        curriculum_definition_from_dict({"name": "toy", "stages": [stage_a, stage_b]})


def test_load_curriculum_definition_toy_file():
    definition = load_curriculum_definition(TOY_CURRICULUM_PATH)
    assert definition.name == "toy-two-stage"
    assert [s.name for s in definition.stages] == ["flat-safe-toy", "night-survival-toy"]
    assert definition.stages[0].promotion.metric == "average_ticks"
    assert definition.stages[1].promotion.metric == "survival_rate"


def test_load_curriculum_definition_rejects_bad_extension(tmp_path):
    bad = tmp_path / "curriculum.txt"
    bad.write_text("name: toy\nstages: []\n")
    with pytest.raises(CurriculumDefinitionError, match="unsupported extension"):
        load_curriculum_definition(str(bad))


# --------------------------------------------------------------------------
# Torch-gated: the actual train/evaluate/promote/hold/resume runner.

pytest.importorskip("torch")

from cognitive_runtime.neural import read_checkpoint_metadata  # noqa: E402
from cognitive_runtime.training.curriculum_runner import run_curriculum  # noqa: E402


def _impossible_stage(name: str, **overrides) -> dict:
    """A stage whose promotion criterion can never be met, for hold tests."""
    stage = _stage(
        name,
        promotion={"metric": "average_ticks", "threshold": 1_000_000.0, "sample_size": 1},
    )
    stage.update(overrides)
    return stage


def test_two_stage_toy_curriculum_promotes_through_both_stages(tmp_path):
    definition = load_curriculum_definition(TOY_CURRICULUM_PATH)
    checkpoint_path = str(tmp_path / "curriculum.pt")

    result = run_curriculum(
        definition, checkpoint_path=checkpoint_path,
        record_dir=str(tmp_path / "sessions"),
    )

    assert result.status == "completed"
    assert result.resumed is False
    assert result.state.stage_index == len(definition.stages)
    stages_seen = {entry["stage"] for entry in result.state.history}
    assert stages_seen == {"flat-safe-toy", "night-survival-toy"}
    assert all(entry["promoted"] for entry in result.state.history)

    # Curriculum state lives in the checkpoint bundle's training stats (#20).
    meta = read_checkpoint_metadata(checkpoint_path)
    curriculum_stats = meta["training_stats"]["curriculum"]
    assert curriculum_stats["completed"] is True
    assert curriculum_stats["definition_name"] == "toy-two-stage"


def test_hold_when_promotion_criteria_unreachable(tmp_path):
    definition = curriculum_definition_from_dict(
        {"name": "unreachable", "stages": [_impossible_stage("stuck")]}
    )
    checkpoint_path = str(tmp_path / "curriculum.pt")

    result = run_curriculum(definition, checkpoint_path=checkpoint_path, record_dir=None)

    assert result.status == "held"
    assert result.state.held is True
    assert "stuck" in result.state.hold_reason
    assert "average_ticks" in result.state.hold_reason
    # Bounded retries -- exactly max_attempts (1), not an infinite spin.
    assert result.state.attempts_at_stage == 1
    assert len(result.state.history) == 1
    assert result.state.history[0]["promoted"] is False

    meta = read_checkpoint_metadata(checkpoint_path)
    assert meta["training_stats"]["curriculum"]["held"] is True


def test_resume_continues_at_held_stage_without_redoing_earlier_stages(tmp_path):
    definition = curriculum_definition_from_dict(
        {"name": "two-stage", "stages": [_stage("easy"), _impossible_stage("stuck")]}
    )
    checkpoint_path = str(tmp_path / "curriculum.pt")

    first = run_curriculum(definition, checkpoint_path=checkpoint_path, record_dir=None)
    assert first.status == "held"
    assert first.state.stage_index == 1  # "easy" promoted, "stuck" holds
    first_history_len = len(first.state.history)

    second = run_curriculum(definition, checkpoint_path=checkpoint_path, record_dir=None)
    assert second.resumed is True
    assert second.status == "held"
    # Resumed at the held stage, not restarted from stage 0: no new "easy"
    # attempts were appended, only more "stuck" ones.
    new_entries = second.state.history[first_history_len:]
    assert new_entries
    assert all(entry["stage"] == "stuck" for entry in new_entries)
    stage_zero_entries = [e for e in second.state.history if e["stage"] == "easy"]
    assert len(stage_zero_entries) == 1  # unchanged from the first run


def test_resuming_a_completed_curriculum_is_a_noop(tmp_path):
    definition = load_curriculum_definition(TOY_CURRICULUM_PATH)
    checkpoint_path = str(tmp_path / "curriculum.pt")

    first = run_curriculum(definition, checkpoint_path=checkpoint_path, record_dir=None)
    assert first.status == "completed"
    history_len = len(first.state.history)

    second = run_curriculum(definition, checkpoint_path=checkpoint_path, record_dir=None)
    assert second.resumed is True
    assert second.status == "completed"
    assert len(second.state.history) == history_len  # no re-training


def test_force_promote_overrides_unmet_metric(tmp_path):
    definition = curriculum_definition_from_dict(
        {"name": "forced", "stages": [_impossible_stage("stuck", max_attempts=5)]}
    )
    checkpoint_path = str(tmp_path / "curriculum.pt")

    result = run_curriculum(
        definition, checkpoint_path=checkpoint_path, record_dir=None, force_promote=True,
    )

    assert result.status == "completed"
    assert len(result.state.history) == 1
    entry = result.state.history[0]
    assert entry["promoted"] is True
    assert entry["forced"] is True
    assert entry["value"] < entry["threshold"]


def test_stage_override_restarts_at_given_index(tmp_path):
    definition = load_curriculum_definition(TOY_CURRICULUM_PATH)
    checkpoint_path = str(tmp_path / "curriculum.pt")

    run_curriculum(definition, checkpoint_path=checkpoint_path, record_dir=None)
    result = run_curriculum(
        definition, checkpoint_path=checkpoint_path, record_dir=None, start_stage=0,
    )

    assert result.status == "completed"
    assert result.state.stage_index == len(definition.stages)


def test_stage_override_out_of_range_raises(tmp_path):
    definition = load_curriculum_definition(TOY_CURRICULUM_PATH)
    checkpoint_path = str(tmp_path / "curriculum.pt")
    with pytest.raises(ValueError, match="out of range"):
        run_curriculum(definition, checkpoint_path=checkpoint_path, record_dir=None, start_stage=5)
