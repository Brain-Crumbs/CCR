"""Issue #240: navigation curriculum, corpus contract, family, and retention gate."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cognitive_runtime.cli import _factory_slot_defaults, cmd_nursery_list
from cognitive_runtime.training import nursery as nursery_module
from cognitive_runtime.training.model_factory import corpus as corpus_module
from cognitive_runtime.training.model_factory.promotion import PromotionPolicy, evaluate_promotion
from cognitive_runtime.training.model_factory.registry import (
    BENCHMARK_FAMILIES,
    GOAL_NAVIGATION_V1,
    leading_champion,
    promote,
)
from cognitive_runtime.training.model_factory.state import (
    STATE_COMPLETED,
    STATE_RUNNING,
    create_state,
    state_path,
    transition,
)
from sleep.forgetting import compute_forgetting_metric


NAVIGATION_SCENARIOS = (
    "navigate_open_goal",
    "navigate_single_wall",
    "navigate_two_wall",
    "replan_after_block",
)


def _config(**overrides):
    fields = dict(
        train_seeds=(0, 1, 2, 3),
        holdout_seeds=(1000, 1001, 2000, 2001),
        episode_ticks=24,
        world_size=48,
        world="crafter",
        backend="crafter",
        expected_pixel_source="crafter",
        export_predictions=False,
    )
    fields.update(overrides)
    return nursery_module.NurseryConfig(**fields)


def _decision_rows(session_dir: Path) -> list[dict]:
    path = next(session_dir.glob("episode_*.decisions.jsonl"))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _stable_navigation_trace(rows: list[dict]) -> list[dict]:
    return [row["oracle_labels"] for row in rows if row.get("oracle_labels") is not None]


def test_navigation_scenarios_are_registered_in_curriculum_order_and_listed(capsys):
    keys = tuple(nursery_module.CRAFTER_SCENARIOS)
    positions = [keys.index(name) for name in NAVIGATION_SCENARIOS]
    assert positions == sorted(positions)

    cmd_nursery_list(SimpleNamespace(world="crafter"))
    listed = capsys.readouterr().out
    for name in NAVIGATION_SCENARIOS:
        assert f"{name}:" in listed
    assert [listed.index(name) for name in NAVIGATION_SCENARIOS] == sorted(
        listed.index(name) for name in NAVIGATION_SCENARIOS
    )


def test_seeded_navigation_layouts_and_behavior_are_reproducible(tmp_path):
    cfg = _config()
    for name in NAVIGATION_SCENARIOS:
        first = nursery_module.CRAFTER_SCENARIOS[name].build(0, cfg)
        second = nursery_module.CRAFTER_SCENARIOS[name].build(0, cfg)
        assert first.program_config_extra == second.program_config_extra

    scenario = nursery_module.CRAFTER_SCENARIOS["navigate_single_wall"]
    first_path = Path(nursery_module._record_scenario_episode(
        str(tmp_path / "a"), "first", 0, scenario, cfg,
    ))
    second_path = Path(nursery_module._record_scenario_episode(
        str(tmp_path / "b"), "second", 0, scenario, cfg,
    ))
    assert _stable_navigation_trace(_decision_rows(first_path)) == _stable_navigation_trace(
        _decision_rows(second_path)
    )


def test_unsolvable_generated_layout_is_rejected_before_session_is_recorded(tmp_path, monkeypatch):
    cfg = _config(train_seeds=(0,), holdout_seeds=(1000,))
    original = nursery_module._navigation_layout

    def sealed_layout(seed, config, scenario_name):
        layout = original(seed, config, scenario_name)
        # A full-height wall one cell east of spawn separates the positive-x
        # caregiver goal from the player; there is no route around it.
        layout["walls"] = tuple((1, y) for y in range(-24, 24))
        layout["layout_id"] = "deliberately-unsolvable"
        return layout

    monkeypatch.setattr(nursery_module, "_navigation_layout", sealed_layout)
    with pytest.raises(ValueError, match="unsolvable.*before recording"):
        nursery_module._record_scenario_episode(
            str(tmp_path), "rejected", 0,
            nursery_module.CRAFTER_SCENARIOS["navigate_open_goal"], cfg,
        )
    assert not (tmp_path / "rejected").exists()


def test_two_wall_holdout_layout_distribution_is_disjoint_from_training():
    cfg = _config()
    training = [nursery_module._navigation_layout(seed, cfg, "navigate_two_wall") for seed in cfg.train_seeds]
    holdout = [nursery_module._navigation_layout(seed, cfg, "navigate_two_wall") for seed in cfg.holdout_seeds]

    assert {layout["layout_family"] for layout in training} == {"two-wall-train-zigzag-v1"}
    assert {layout["layout_family"] for layout in holdout} == {"two-wall-holdout-long-chicane-v1"}
    assert {tuple(layout["walls"]) for layout in training}.isdisjoint(
        {tuple(layout["walls"]) for layout in holdout}
    )


def test_replan_after_block_records_a_real_mid_episode_route_invalidation(tmp_path):
    cfg = _config(train_seeds=(0,), holdout_seeds=(1000,))
    session = Path(nursery_module._record_scenario_episode(
        str(tmp_path), "replan", 0,
        nursery_module.CRAFTER_SCENARIOS["replan_after_block"], cfg,
    ))
    rows = _decision_rows(session)
    invalidation = next(row for row in rows if row["tick_index"] == 3)
    labels = invalidation["oracle_labels"]

    assert labels["replan_required"] is True
    assert labels["path_progress_delta"] < 0
    assert labels["geodesic_distance"] is not None
    assert labels["behavior_mixture"]["expert_action"] is not None
    active = [
        row["oracle_labels"]["behavior_mixture"]
        for row in rows
        if row.get("oracle_labels")
        and row["oracle_labels"]["behavior_mixture"]["expert_action"] is not None
    ]
    assert any(step["injected"] for step in active)
    assert sum(step["injected"] for step in active) / len(active) == pytest.approx(0.25, abs=0.07)


def _install_fake_navigation_corpus(monkeypatch, cache: Path) -> None:
    scenario = SimpleNamespace(name="navigate_open_goal")
    monkeypatch.setattr(corpus_module, "_scenarios_for_world", lambda world: {scenario.name: scenario})
    monkeypatch.setattr(corpus_module, "validate_nursery_recordings", lambda *args, **kwargs: [])

    def record(_record_dir, _session_id, seed, _scenario, _cfg):
        session = cache / f"nav-{seed}"
        session.mkdir(parents=True, exist_ok=True)
        navigation = {
            "generator_name": "navigate_open_goal",
            "generator_version": "goal-navigation-layouts-v1",
            "generator_seed": seed,
            "layout_distribution": {"layout_id": f"layout-{seed}", "solvability_check": "astar_before_recording"},
            "goal_distribution_policy": {"generator_version": "goal_distribution_v1"},
            "oracle_planner_policy": {"planner_version": "crafter-astar-v1"},
            "goal_reward_policy": {"potential": "linear"},
        }
        (session / "session.json").write_text(json.dumps({
            "session_id": session.name,
            "program_config": {"navigation": navigation},
        }), encoding="utf-8")
        (session / "episode_00000.decisions.jsonl").write_text("{}\n", encoding="utf-8")
        (session / "episode_00000.streams.jsonl").write_text(json.dumps({
            "stream_id": "vision.frame.pixels", "frame_ref": f"frame-{seed}",
        }) + "\n", encoding="utf-8")
        return str(session)

    monkeypatch.setattr(corpus_module, "_record_or_reuse_scenario_episode", record)


def test_configurable_mixture_and_retention_are_frozen_in_navigation_data_contract(tmp_path, monkeypatch):
    _install_fake_navigation_corpus(monkeypatch, tmp_path / "recordings")
    spec = {
        "root": str(tmp_path / "corpora"),
        "organism": "test",
        "corpus_id": "navigation-contract-v1",
        "generator": {
            "world": "minecraft", "backend": "simulated",
            "navigation_random_action_fraction": 0.30,
        },
        "splits": {
            "train": {"navigate_open_goal": [1]},
            "validation": {"navigate_open_goal": [2]},
            "test": {"navigate_open_goal": [3]},
        },
        "behavior_mixture": {"random_action_fraction": 0.30},
        "retention_policy": {
            "suite": "generic_action_effects_v1",
            "max_regression": 0.05,
            "replay_mixture": {"generic_action_effects_v1": 0.25, "goal_navigation_v1": 0.75},
        },
    }
    corpus = corpus_module.build_corpus(spec)

    assert corpus.data_contract.behavior_mixture_policy["random_action_fraction"] == 0.30
    assert corpus.data_contract.behavior_mixture_policy["expert_action_fraction"] == 0.70
    assert corpus.data_contract.retention_policy["suite"] == "generic_action_effects_v1"
    assert corpus.data_contract.oracle_planner_policy["planner_version"] == "crafter-astar-v1"
    assert nursery_module.NurseryConfig().navigation_random_action_fraction == 0.25


def _mark_completed(root: Path, run_id: str) -> None:
    path = state_path(root / run_id)
    create_state(path, run_id)
    transition(path, STATE_RUNNING)
    transition(path, STATE_COMPLETED)


def test_goal_navigation_family_has_own_metrics_and_never_overwrites_generic_champion(tmp_path):
    assert GOAL_NAVIGATION_V1.promotion_metrics == (
        "success_rate", "geodesic_efficiency", "collision_rate", "replan_recovery",
    )
    assert set(BENCHMARK_FAMILIES) >= {"generic_action_effects_v1", "goal_navigation_v1"}
    _mark_completed(tmp_path, "generic")
    _mark_completed(tmp_path, "navigation")
    common = dict(tier="fast", objective="success_rate", checkpoint_path="checkpoint.pt")
    promote(
        tmp_path, family="generic_action_effects_v1", run_id="generic",
        checkpoint_sha256="a" * 64, metrics={}, **common,
    )
    promote(
        tmp_path, family="goal_navigation_v1", run_id="navigation",
        checkpoint_sha256="b" * 64, metrics={}, **common,
    )
    assert leading_champion(
        tmp_path, family="generic_action_effects_v1", tier="fast", objective="success_rate",
    ).run_id == "generic"
    assert leading_champion(
        tmp_path, family="goal_navigation_v1", tier="fast", objective="success_rate",
    ).run_id == "navigation"


def test_navigation_contract_defaults_factory_promotion_to_navigation_family(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "contracts.json").write_text(json.dumps({
        "data_contract": {
            "behavior_mixture_policy": {"expert_policy": "astar"},
            "retention_policy": {"suite": "generic_action_effects_v1"},
        },
    }), encoding="utf-8")

    family, tier = _factory_slot_defaults(run, {"organism": "Crafter"})
    assert family == "goal_navigation_v1"
    assert tier is None


def _passing_promotion_inputs() -> dict:
    return {
        "policy": PromotionPolicy(minimum_practical_margin=0.0),
        "candidate_run_id": "navigation",
        "experiment_report": {
            "checkpoint": {"path": "checkpoint.pt", "sha256": "a" * 64},
            "training_stats": {"evaluation_mode": "heldout", "completion_status": "completed"},
            "metrics": {"rollout": {
                "horizons": {1: {
                    "beats_copy_last": True,
                    "model_mse": 0.5,
                    "copy_last_mse": 1.0,
                    "model_over_copy_last_mse": 0.5,
                }},
                "rollout_health": {"state": "healthy"},
            }},
            "event_stratified_metrics": {1: {"entity": {
                "cow_false_positive_rate": 0.0,
                "false_positive": 0,
                "true_negative": 8,
            }}},
        },
        "data_quality": {"passed": True, "issues": []},
        "split_overlap": {"disjoint": True, "issues": []},
        "has_champion": False,
        "target_family": "goal_navigation_v1",
        "retention_policy": {
            "suite": "generic_action_effects_v1",
            "max_regression": 0.05,
            "replay_mixture": {"generic_action_effects_v1": 0.25, "goal_navigation_v1": 0.75},
        },
    }


def test_navigation_promotion_reuses_forgetting_metric_and_fails_beyond_declared_allowance():
    passing = compute_forgetting_metric(
        [1.0, 1.0], [1.04, 1.04],
        old_scenario="generic_action_effects_v1",
        new_scenario="goal_navigation_v1",
        tolerance=0.05,
    )
    verdict = evaluate_promotion(**_passing_promotion_inputs(), retention_report=passing)
    assert verdict.promoted

    failing = compute_forgetting_metric(
        [1.0, 1.0], [1.06, 1.06],
        old_scenario="generic_action_effects_v1",
        new_scenario="goal_navigation_v1",
        tolerance=0.05,
    )
    verdict = evaluate_promotion(**_passing_promotion_inputs(), retention_report=failing)
    assert not verdict.promoted
    gate = next(gate for gate in verdict.gates if gate.name == "generic_retention")
    assert not gate.passed
    assert "beyond the declared allowance" in gate.reason
