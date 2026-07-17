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
# Phase 7 (issue #104): staged-ontogeny stage fields (world/senses/
# motor-freedom/losses/gates), generalised from `training/curriculum_runner.py`
# into `development/` behind this module's shim.

def test_stage_spec_defaults_keep_pre_phase7_shape():
    """A stage built with no Phase 7 fields is exactly the pre-#104 shape --
    old curriculum defs still load through the shim unchanged."""
    stage = CurriculumStageSpec(name="legacy")
    assert stage.world is None
    assert stage.scenario is None
    assert stage.senses == ()
    assert stage.motor_freedom is None
    assert stage.losses == ()
    assert stage.gates == ()
    assert stage.is_staged_ontogeny is False


def test_stage_spec_accepts_phase7_fields_and_validates():
    stage = CurriculumStageSpec(
        name="crawling",
        world="crafter",
        scenario="walk_forward",
        senses=("vision", "proprioception"),
        motor_freedom="overridden",
        losses=("prediction", "action_conditioning"),
        gates=(
            PromotionCriteria(metric="cortex_beats_copy_last", threshold=1.0),
            PromotionCriteria(metric="action_ablation_margin", threshold=0.1),
        ),
    )
    assert stage.world == "crafter"
    assert stage.scenario == "walk_forward"
    assert stage.senses == ("vision", "proprioception")
    assert stage.motor_freedom == "overridden"
    assert stage.losses == ("prediction", "action_conditioning")
    assert [g.metric for g in stage.gates] == ["cortex_beats_copy_last", "action_ablation_margin"]
    assert stage.is_staged_ontogeny is True


def test_stage_spec_rejects_unknown_world():
    with pytest.raises(ValueError, match="unknown world"):
        CurriculumStageSpec(name="s", world="atari")


def test_stage_spec_rejects_unknown_motor_freedom():
    with pytest.raises(ValueError, match="unknown motor_freedom"):
        CurriculumStageSpec(name="s", motor_freedom="wandering")


def test_stage_spec_rejects_duplicate_senses():
    with pytest.raises(ValueError, match="duplicate senses"):
        CurriculumStageSpec(name="s", senses=("vision", "vision"))


def test_stage_spec_rejects_duplicate_losses():
    with pytest.raises(ValueError, match="duplicate losses"):
        CurriculumStageSpec(name="s", losses=("prediction", "prediction"))


def test_stage_spec_rejects_duplicate_gate_metrics():
    with pytest.raises(ValueError, match="duplicate milestone gate metrics"):
        CurriculumStageSpec(
            name="s",
            gates=(
                PromotionCriteria(metric="average_ticks", threshold=1.0),
                PromotionCriteria(metric="average_ticks", threshold=2.0),
            ),
        )


def test_evaluate_gates_requires_all_gates_to_pass():
    stage = CurriculumStageSpec(
        name="crawling",
        gates=(
            PromotionCriteria(metric="cortex_beats_copy_last", threshold=1.0),
            PromotionCriteria(metric="action_ablation_margin", threshold=0.1),
        ),
    )
    all_pass = stage.evaluate_gates({"cortex_beats_copy_last": 1.5, "action_ablation_margin": 0.2})
    assert all_pass == {"cortex_beats_copy_last": True, "action_ablation_margin": True}

    one_fails = stage.evaluate_gates({"cortex_beats_copy_last": 1.5, "action_ablation_margin": 0.05})
    assert one_fails == {"cortex_beats_copy_last": True, "action_ablation_margin": False}
    assert not all(one_fails.values())


def test_promotion_criteria_value_of_reads_a_metrics_mapping():
    gate = PromotionCriteria(metric="forgetting_score", threshold=0.5)
    assert gate.evaluate({"forgetting_score": 0.9}) is True
    assert gate.evaluate({"forgetting_score": 0.1}) is False


def test_promotion_criteria_value_of_raises_on_missing_metric():
    gate = PromotionCriteria(metric="forgetting_score", threshold=0.5)
    with pytest.raises(CurriculumDefinitionError, match="not present"):
        gate.value_of({"some_other_metric": 1.0})


def test_curriculum_definition_from_dict_parses_phase7_fields():
    stage = _stage(
        "gestation",
        world="minecraft",
        scenario="habituate",
        senses=["vision", "proprioception"],
        motor_freedom="frozen",
        losses=["prediction"],
        gates=[{"metric": "average_ticks", "threshold": 1.0, "sample_size": 1}],
    )
    definition = curriculum_definition_from_dict({"name": "ladder", "stages": [stage]})
    spec = definition.stages[0]
    assert spec.world == "minecraft"
    assert spec.scenario == "habituate"
    assert spec.senses == ("vision", "proprioception")
    assert spec.motor_freedom == "frozen"
    assert spec.losses == ("prediction",)
    assert [g.metric for g in spec.gates] == ["average_ticks"]


def test_old_curriculum_defs_still_load_through_the_shim():
    """The exact back-compat contract of the shim (issue #104): a
    pre-Phase-7 definition, with no world/senses/motor-freedom/losses/gates
    fields anywhere, still loads and every stage keeps the legacy shape."""
    definition = load_curriculum_definition(TOY_CURRICULUM_PATH)
    for stage in definition.stages:
        assert stage.is_staged_ontogeny is False
        assert stage.gates == ()


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


# --------------------------------------------------------------------------
# Phase 7 (issue #104): milestone-gated promotion, torch-gated (exercises
# the real train/evaluate loop). ``promotion.sample_size`` still controls
# how many eval episodes run each attempt; the *gate* decision comes from
# ``gates`` instead of ``promotion`` whenever a stage declares any.

def _gated_stage(name: str, gates, **overrides) -> dict:
    stage = _stage(name, promotion={"metric": "average_ticks", "threshold": 0.0, "sample_size": 1})
    stage["gates"] = gates
    stage.update(overrides)
    return stage


def test_promotion_fires_only_when_every_milestone_gate_passes(tmp_path):
    """"Not a single scalar": two gates on the same easy, mob-free world both
    pass, so the stage promotes even though neither individually is the old
    ``promotion`` field."""
    definition = curriculum_definition_from_dict({
        "name": "multi-gate",
        "stages": [_gated_stage("both-easy", gates=[
            {"metric": "average_ticks", "threshold": 0.0, "sample_size": 1},
            {"metric": "survival_rate", "threshold": 0.0, "sample_size": 1},
        ])],
    })
    checkpoint_path = str(tmp_path / "curriculum.pt")

    result = run_curriculum(definition, checkpoint_path=checkpoint_path, record_dir=None)

    assert result.status == "completed"
    entry = result.state.history[0]
    assert entry["metric"] == ["average_ticks", "survival_rate"]
    assert entry["promoted"] is True
    assert set(entry["value"]) == {"average_ticks", "survival_rate"}


def test_a_failing_milestone_gate_holds_even_if_the_others_pass(tmp_path):
    definition = curriculum_definition_from_dict({
        "name": "multi-gate-holds",
        "stages": [_gated_stage("one-impossible", gates=[
            {"metric": "average_ticks", "threshold": 0.0, "sample_size": 1},
            # survival_rate is a fraction in [0, 1]; > 1.0 can never pass.
            {"metric": "survival_rate", "threshold": 1_000_000.0, "sample_size": 1},
        ], max_attempts=1)],
    })
    checkpoint_path = str(tmp_path / "curriculum.pt")

    result = run_curriculum(definition, checkpoint_path=checkpoint_path, record_dir=None)

    assert result.status == "held"
    entry = result.state.history[0]
    assert entry["promoted"] is False
    assert entry["value"]["average_ticks"] >= entry["threshold"]["average_ticks"]
    assert entry["value"]["survival_rate"] < entry["threshold"]["survival_rate"]


def test_milestone_metrics_provider_supplies_phase2to6_metrics_to_gates(tmp_path):
    """The seam issue #105's ladder wiring will use: a stage's gate can name
    a Phase 2-6 milestone metric (not just the plain eval-episode ones), and
    the runner asks a caller-supplied provider for it."""
    definition = curriculum_definition_from_dict({
        "name": "milestone-metric",
        "stages": [_gated_stage("crawling", gates=[
            {"metric": "cortex_beats_copy_last", "threshold": 1.0, "sample_size": 1},
        ])],
    })
    checkpoint_path = str(tmp_path / "curriculum.pt")

    def milestone_metrics(stage, summary):
        assert stage.name == "crawling"
        return {"cortex_beats_copy_last": 1.5}

    result = run_curriculum(
        definition, checkpoint_path=checkpoint_path, record_dir=None,
        milestone_metrics=milestone_metrics,
    )

    assert result.status == "completed"
    assert result.state.history[0]["value"] == {"cortex_beats_copy_last": 1.5}


def test_milestone_gate_without_a_metrics_provider_raises_a_clear_error(tmp_path):
    """A gate referencing a milestone metric with no way to compute it is a
    definition/wiring bug, not a hold -- it should fail loudly."""
    definition = curriculum_definition_from_dict({
        "name": "unwired-milestone",
        "stages": [_gated_stage("crawling", gates=[
            {"metric": "cortex_beats_copy_last", "threshold": 1.0, "sample_size": 1},
        ])],
    })
    checkpoint_path = str(tmp_path / "curriculum.pt")

    with pytest.raises(CurriculumDefinitionError, match="cortex_beats_copy_last"):
        run_curriculum(definition, checkpoint_path=checkpoint_path, record_dir=None)
