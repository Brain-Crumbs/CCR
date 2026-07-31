"""Crafter nursery scenario ports (issue #90; extended by issue #202):
walk_forward_short, blocked_forward, turn, object_permanence, approach_entity
registered in ``CRAFTER_SCENARIOS`` alongside Minecraft's
``NURSERY_SCENARIOS``. Milestone 1 exit gate: these record deterministically,
with genuine frame-to-frame motion, and pass the data-quality gates.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

pytest.importorskip("crafter")
torch = pytest.importorskip("torch")

from cognitive_runtime.programs.crafter.streams import SEMANTIC_CLASS_IDS  # noqa: E402
from cognitive_runtime.runtime.replay import list_episodes  # noqa: E402
from cognitive_runtime.training.nursery import (  # noqa: E402
    CRAFTER_SCENARIOS,
    NurseryScenario,
    NurseryConfig,
    _record_scenario_episode,
    _crafter_box_in,
    _scenarios_for_world,
    measure_recording_quality,
    run_nursery_scenario,
    validate_nursery_recordings,
)
from cognitive_runtime.training.action_world_model import ActionWorldModelConfig  # noqa: E402
import cognitive_runtime.training.nursery as nursery_module  # noqa: E402

_ENTITY_ID = SEMANTIC_CLASS_IDS["entity"]


def _crafter_config(**overrides) -> NurseryConfig:
    base = dict(world="crafter", episode_ticks=40, train_seeds=(0, 1), holdout_seeds=(1000,))
    base.update(overrides)
    return NurseryConfig(**base)


def _semantic_ids_seen(session_dir: str, episode_id: str) -> set:
    ids: set = set()
    with open(os.path.join(session_dir, f"{episode_id}.streams.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("stream_id") != "vision.frame.grid":
                continue
            grid = record.get("payload") or []
            for row in grid:
                ids.update(row)
    return ids


def test_registry_has_every_crafter_scenario():
    assert set(CRAFTER_SCENARIOS) == {
        "walk_forward_short", "blocked_forward", "turn", "object_permanence",
        "approach_entity", "motor_babbling_open", "motor_babbling_walls",
    }


def test_scenarios_for_world_selects_the_right_registry():
    from cognitive_runtime.training.nursery import NURSERY_SCENARIOS

    assert _scenarios_for_world("crafter") is CRAFTER_SCENARIOS
    assert _scenarios_for_world("minecraft") is NURSERY_SCENARIOS
    with pytest.raises(ValueError, match="unknown nursery world"):
        _scenarios_for_world("not-a-world")


def test_episode_cache_reuses_a_compatible_scenario_seed(monkeypatch, tmp_path):
    """Hyperparameter reruns must reuse recording data, not replay Crafter."""
    calls = []

    def fake_record(record_dir, session_id, seed, scenario, cfg, *, recording=None):
        calls.append((record_dir, session_id, seed))
        session_dir = os.path.join(record_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        with open(os.path.join(session_dir, "session.json"), "w", encoding="utf-8") as fh:
            json.dump({"session_id": session_id}, fh)
        with open(os.path.join(session_dir, "episode_00000.decisions.jsonl"), "w", encoding="utf-8"):
            pass
        return session_dir

    monkeypatch.setattr(nursery_module, "_record_scenario_episode", fake_record)
    cfg = _crafter_config(episode_cache_dir=str(tmp_path / "episode-cache"))
    scenario = CRAFTER_SCENARIOS["turn"]
    first = nursery_module._record_or_reuse_scenario_episode(
        str(tmp_path / "run-a"), "turn-train-0", 0, scenario, cfg,
    )
    second = nursery_module._record_or_reuse_scenario_episode(
        str(tmp_path / "run-b"), "turn-train-0", 0, scenario, cfg,
    )

    assert first == second
    assert len(calls) == 1
    assert os.path.isfile(os.path.join(first, "nursery_cache.json"))


@pytest.mark.parametrize("scenario_name", sorted(CRAFTER_SCENARIOS))
def test_each_crafter_scenario_passes_its_own_quality_gate(tmp_path, scenario_name):
    scenario = CRAFTER_SCENARIOS[scenario_name]
    cfg = _crafter_config()
    session_dir = _record_scenario_episode(
        str(tmp_path), f"crafter-{scenario_name}", 0, scenario, cfg
    )
    episode_id = list_episodes(session_dir)[0]
    quality = measure_recording_quality(session_dir, episode_id)

    assert quality.n_frames > 0
    assert quality.completed is True
    issues = validate_nursery_recordings([session_dir], scenario)
    assert issues == [], issues


def test_walk_forward_short_has_genuine_frame_to_frame_motion(tmp_path):
    scenario = CRAFTER_SCENARIOS["walk_forward_short"]
    cfg = _crafter_config()
    session_dir = _record_scenario_episode(str(tmp_path), "crafter-walk", 0, scenario, cfg)
    quality = measure_recording_quality(session_dir, list_episodes(session_dir)[0])

    assert quality.blocks_per_tick >= scenario.min_blocks_per_tick
    assert quality.unique_frame_fraction >= scenario.min_unique_frame_fraction
    assert quality.pixel_sources  # real pixel provenance is recorded


def test_walk_forward_short_ends_before_the_boundary_with_no_accidental_tail(tmp_path):
    """Issue #202: the old unbounded ``walk_forward`` plateaued at the world
    boundary/an obstacle and recorded a long accidental stationary tail.
    ``walk_forward_short`` is bounded to the safe corridor length and should
    contain sustained movement with (at most) a trivial trailing hold."""
    scenario = CRAFTER_SCENARIOS["walk_forward_short"]
    cfg = _crafter_config(episode_ticks=200)  # far longer than any safe corridor
    session_dir = _record_scenario_episode(str(tmp_path), "crafter-walk-long", 0, scenario, cfg)
    quality = measure_recording_quality(session_dir, list_episodes(session_dir)[0])

    assert quality.moving_transition_fraction >= 0.5
    assert quality.longest_stationary_tail <= 2


def test_blocked_forward_contains_movement_and_a_bounded_blocked_phase(tmp_path):
    """Issue #202: an explicitly labelled few-tick blocked phase (movement,
    then a bounded hold against a wall, then a recovery move) is useful
    training data, unlike an accidental long stationary tail."""
    scenario = CRAFTER_SCENARIOS["blocked_forward"]
    cfg = _crafter_config()
    session_dir = _record_scenario_episode(str(tmp_path), "crafter-blocked", 0, scenario, cfg)
    quality = measure_recording_quality(session_dir, list_episodes(session_dir)[0])

    # Movement happened (the approach + the recovery move)...
    assert quality.position_change_count > 0
    # ...and the blocked hold is bounded, not an open-ended stall.
    assert 0 < quality.longest_stationary_run <= scenario.max_longest_stationary_run
    # ...and the episode does not end mid-stall: the recovery action moves
    # the agent again before the episode is over.
    assert quality.longest_stationary_tail <= scenario.max_longest_stationary_tail


def test_turn_sweeps_every_facing_with_zero_displacement(tmp_path):
    scenario = CRAFTER_SCENARIOS["turn"]
    cfg = _crafter_config()
    session_dir = _record_scenario_episode(str(tmp_path), "crafter-turn", 0, scenario, cfg)
    quality = measure_recording_quality(session_dir, list_episodes(session_dir)[0])

    assert quality.unique_facings == 4
    assert quality.max_displacement == 0.0


def test_object_permanence_player_is_stationary_while_mob_moves(tmp_path):
    """Crafter's re-scoped occlusion: the player (NullPolicy) never moves --
    only the scripted mob does, walking out past the egocentric view radius
    and back (no literal wall occluder; see the module docstring in
    ``training.nursery``)."""
    scenario = CRAFTER_SCENARIOS["object_permanence"]
    cfg = _crafter_config()
    session_dir = _record_scenario_episode(str(tmp_path), "crafter-occlusion", 0, scenario, cfg)
    episode_id = list_episodes(session_dir)[0]

    quality = measure_recording_quality(session_dir, episode_id)
    assert quality.net_displacement == 0.0
    assert quality.max_displacement == 0.0
    assert quality.completed is True


def test_approach_entity_stays_dynamic_after_nearing_the_cow(tmp_path):
    """The entity approach must not spend most of a short episode blocked
    against the cow; it approaches, retreats, and preserves useful motion
    supervision throughout the configured recording length."""
    scenario = CRAFTER_SCENARIOS["approach_entity"]
    cfg = _crafter_config()
    session_dir = _record_scenario_episode(
        str(tmp_path), "crafter-approach-dynamic", 0, scenario, cfg
    )
    quality = measure_recording_quality(session_dir, list_episodes(session_dir)[0])

    assert quality.moving_transition_fraction >= scenario.min_moving_transition_fraction
    assert quality.longest_stationary_tail <= scenario.max_longest_stationary_tail


def _motor_commands_seen(session_dir: str, episode_id: str) -> set:
    actions: set = set()
    with open(os.path.join(session_dir, f"{episode_id}.streams.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("stream_id") == "motor.command":
                actions.add(record["payload"]["action"])
    return actions


@pytest.mark.parametrize("scenario_name", ["motor_babbling_open", "motor_babbling_walls"])
def test_motor_babbling_scenarios_only_emit_their_declared_action_subset(tmp_path, scenario_name):
    """Issue #235: the generator policy declares MOVE_UP/MOVE_DOWN/MOVE_LEFT/
    MOVE_RIGHT/NULL and nothing else. NULL never appears on the
    ``motor.command`` stream (a NULL decision emits no motor event -- see
    ``SingleActionPolicy.emit``), so this checks the recording only ever
    contains the four movement actions."""
    scenario = CRAFTER_SCENARIOS[scenario_name]
    cfg = _crafter_config(episode_ticks=200)
    for seed in (0, 1, 2, 3, 1000):
        session_dir = _record_scenario_episode(str(tmp_path), f"crafter-{scenario_name}-{seed}", seed, scenario, cfg)
        episode_id = list_episodes(session_dir)[0]
        actions = _motor_commands_seen(session_dir, episode_id)
        assert actions, (seed, actions)
        assert actions <= {"MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"}, (seed, actions)


def test_motor_babbling_open_records_its_generator_metadata(tmp_path):
    """Issue #235's acceptance criterion: generator name/version, action
    subset, burst-length distribution and generator seed must be recorded
    (not hardcoded silently), so a ``DataContract`` can read them straight
    out of the recorded evidence (epic #212 §12.6)."""
    scenario = CRAFTER_SCENARIOS["motor_babbling_open"]
    cfg = _crafter_config()
    session_dir = _record_scenario_episode(str(tmp_path), "crafter-babbling-meta", 3, scenario, cfg)

    with open(os.path.join(session_dir, "session.json"), encoding="utf-8") as fh:
        metadata = json.load(fh)
    generator = metadata["program_config"]["motor_babbling"]
    assert generator["generator_name"] == "motor_babbling_open"
    assert generator["generator_version"]
    assert set(generator["action_subset"]) == {
        "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "NULL",
    }
    assert generator["burst_ticks_distribution"]["min_ticks"] == 1
    assert generator["burst_ticks_distribution"]["max_ticks"] == 4
    assert generator["generator_seed"] == 3


def test_motor_babbling_walls_reports_seeded_layout_and_realised_outcome_mix(tmp_path):
    """Issue #236: the contract carries both the declared wall distribution
    and the *realised* MF-E1 outcome mix; requested action samples alone are
    not evidence that the world supplied useful transitions."""
    scenario = CRAFTER_SCENARIOS["motor_babbling_walls"]
    cfg = _crafter_config(episode_ticks=80)
    session_dir = _record_scenario_episode(str(tmp_path), "crafter-babbling-walls", 3, scenario, cfg)

    with open(os.path.join(session_dir, "session.json"), encoding="utf-8") as fh:
        metadata = json.load(fh)
    generator = metadata["program_config"]["motor_babbling"]
    report = metadata["quality_report"]
    mix = report["episodes"]["episode_00000"]

    assert generator["generator_name"] == "motor_babbling_walls"
    assert generator["layout_distribution"]["layout_seed"] == 3
    assert generator["layout_distribution"]["interior_barrier"]["guaranteed_recovery_gap"] is True
    assert generator["outcome_mix_bounds"] == {
        outcome: {"min_fraction": lower, "max_fraction": upper}
        for outcome, (lower, upper) in scenario.action_effect_mix_bounds.items()
    }
    assert report["accepted"] is True
    assert set(mix["fractions"]) >= {"moved", "blocked", "turned_only", "no_op"}
    for outcome, (lower, upper) in scenario.action_effect_mix_bounds.items():
        assert lower <= mix["fractions"][outcome] <= upper


def test_motor_babbling_walls_rejects_a_boxed_in_stationary_tail(tmp_path):
    """A direct regression fixture for the failure mode in #236: all action
    bursts hit walls, producing a long blocked/stationary tail.  The recorder
    writes rejected quality evidence and raises before such a session can be
    returned to a corpus builder."""
    walls = CRAFTER_SCENARIOS["motor_babbling_walls"]

    def boxed_build(seed, cfg):
        recording = walls.build(seed, cfg)

        def scene_setup(program):
            program.set_post_reset_hook(_crafter_box_in)

        return replace(recording, scene_setup=scene_setup)

    boxed = NurseryScenario(
        "motor_babbling_walls_boxed_fixture", "test-only boxed wall fixture", boxed_build,
        max_longest_stationary_tail=walls.max_longest_stationary_tail,
        max_longest_stationary_run=walls.max_longest_stationary_run,
        action_effect_mix_bounds=walls.action_effect_mix_bounds,
    )
    cfg = _crafter_config(episode_ticks=40)
    with pytest.raises(ValueError, match="rejected recorded episode"):
        _record_scenario_episode(str(tmp_path), "crafter-babbling-boxed", 0, boxed, cfg)

    with open(tmp_path / "crafter-babbling-boxed" / "session.json", encoding="utf-8") as fh:
        report = json.load(fh)["quality_report"]
    assert report["accepted"] is False
    assert report["episodes"]["episode_00000"]["fractions"]["blocked"] > 0.7
    assert any("stationary tail" in issue for issue in report["issues"])


def test_motor_babbling_open_starting_facing_varies_by_seed(tmp_path):
    scenario = CRAFTER_SCENARIOS["motor_babbling_open"]
    cfg = _crafter_config(episode_ticks=4)
    facings = set()
    for seed in (0, 1, 2, 3):
        session_dir = _record_scenario_episode(
            str(tmp_path), f"crafter-babbling-facing-{seed}", seed, scenario, cfg
        )
        episode_id = list_episodes(session_dir)[0]
        with open(os.path.join(session_dir, f"{episode_id}.streams.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                record = json.loads(line)
                if record.get("stream_id") == "spatial.facing":
                    facings.add((record["payload"].get("x"), record["payload"].get("y")))
                    break
    assert len(facings) > 1, facings


@pytest.mark.parametrize(
    "scenario_name", ["walk_forward_short", "blocked_forward", "turn", "motor_babbling_open", "motor_babbling_walls"]
)
def test_non_entity_scenarios_contain_no_entity_semantic_id(tmp_path, scenario_name):
    """Issue #202: wildlife must be *removed*, not merely frozen in place --
    a frozen-but-rendered cow was silently becoming a permanent training
    feature of scenarios that never asked for one."""
    scenario = CRAFTER_SCENARIOS[scenario_name]
    cfg = _crafter_config(train_seeds=(0, 1, 2, 3), holdout_seeds=(1000,))
    for seed in (0, 1, 2, 3, 1000):
        session_dir = _record_scenario_episode(
            str(tmp_path), f"crafter-{scenario_name}-{seed}", seed, scenario, cfg
        )
        episode_id = list_episodes(session_dir)[0]
        ids = _semantic_ids_seen(session_dir, episode_id)
        assert _ENTITY_ID not in ids, (scenario_name, seed, ids)


@pytest.mark.parametrize("scenario_name", ["approach_entity", "object_permanence"])
def test_entity_scenarios_contain_the_scripted_entity_class(tmp_path, scenario_name):
    """The compact recording exposes the action-relevant entity class only."""
    scenario = CRAFTER_SCENARIOS[scenario_name]
    cfg = _crafter_config(train_seeds=(0, 1, 2, 3), holdout_seeds=(1000,))
    for seed in (0, 1, 2, 3, 1000):
        session_dir = _record_scenario_episode(
            str(tmp_path), f"crafter-{scenario_name}-{seed}", seed, scenario, cfg
        )
        episode_id = list_episodes(session_dir)[0]
        ids = _semantic_ids_seen(session_dir, episode_id)
        assert _ENTITY_ID in ids, (scenario_name, seed, ids)


def test_crafter_scenarios_are_deterministic(tmp_path):
    cfg = _crafter_config()
    for name, scenario in CRAFTER_SCENARIOS.items():
        first = _record_scenario_episode(str(tmp_path / "a"), f"{name}-a", 0, scenario, cfg)
        second = _record_scenario_episode(str(tmp_path / "b"), f"{name}-b", 0, scenario, cfg)
        q1 = measure_recording_quality(first, list_episodes(first)[0])
        q2 = measure_recording_quality(second, list_episodes(second)[0])
        assert q1.net_displacement == q2.net_displacement, name
        assert q1.n_frames == q2.n_frames, name
        assert q1.unique_frames == q2.unique_frames, name


def test_requested_crafter_world_size_resolves_to_the_requested_area(tmp_path):
    """Issue #202: a notebook's requested ``world_size`` must actually
    resolve to Crafter's ``area`` -- ``CrafterConfig.from_dict`` silently
    ignores an unknown ``world_size`` key, so the nursery must translate it."""
    scenario = CRAFTER_SCENARIOS["turn"]
    cfg = _crafter_config(world_size=96, episode_ticks=8)
    session_dir = _record_scenario_episode(str(tmp_path), "crafter-worldsize", 0, scenario, cfg)

    with open(os.path.join(session_dir, "session.json"), encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert list(metadata["program_config"]["area"]) == [96, 96]


def test_run_nursery_scenario_reports_the_resolved_program_config(tmp_path):
    cfg = _crafter_config(
        world_size=80, horizons=(1,), latent_width=16, hidden_dim=32, reconstruction_size=8,
        epochs=1, consistency_epochs=0, batch_size=16,
    )
    _model, report = run_nursery_scenario(str(tmp_path), "turn", cfg)
    assert list(report.resolved_program_config["area"]) == [80, 80]


@pytest.mark.parametrize("scenario_name", sorted(CRAFTER_SCENARIOS))
def test_split_overlap_gate_passes_a_genuine_crafter_recording(tmp_path, scenario_name):
    """Every Crafter scenario retains enough seeded visual context for the
    default train/holdout overlap gate to catch a real duplicated recording
    without rejecting the intentionally scripted route or entity."""
    cfg = _crafter_config(
        train_seeds=(0, 1, 2), holdout_seeds=(1000, 1001),
        horizons=(1,), latent_width=16, hidden_dim=32, reconstruction_size=8,
        epochs=1, consistency_epochs=0, batch_size=16,
        split_overlap_gate=True,
    )
    _model, report = run_nursery_scenario(str(tmp_path), scenario_name, cfg)
    assert report.scenario == scenario_name


def test_joint_overfit_evaluation_replays_exact_training_sessions(tmp_path):
    """The overfit canary must not create a fake held-out recording: it
    evaluates the exact paths that supplied training frames and labels the
    result as a training replay."""
    from cognitive_runtime.training.nursery import run_nursery_joint

    cfg = _crafter_config(
        train_seeds=(0, 1), holdout_seeds=(1000,), horizons=(1,),
        split_overlap_gate=True, overfit_evaluation=True,
    )
    model_cfg = ActionWorldModelConfig(
        latent_width=16, hidden_dim=32, reconstruction_size=8,
        horizons_ticks=(1,), epochs=1, batch_size=16,
        warmup_frames=2, rollout_frames=2,
    )
    _model, report = run_nursery_joint(
        str(tmp_path), train_scenarios=("approach_entity",), holdout_scenarios=(),
        config=cfg, model_config=model_cfg,
    )

    assert report.evaluation_mode == "training_replay"
    assert report.eval_sessions["approach_entity"] == report.train_sessions["approach_entity"]
    assert report.zero_shot_metrics == {}
    assert report.training_stats["evaluation_mode"] == "training_replay"


def test_run_nursery_scenario_end_to_end_against_crafter(tmp_path):
    """``ccr nursery run --world crafter walk_forward_short`` (phase-1's
    acceptance criterion): records train/holdout episodes and
    trains/evaluates the same pixel-prediction pipeline Minecraft uses --
    world-agnostic downstream of the recorded session dir."""
    cfg = _crafter_config(
        horizons=(1, 3), latent_width=16, hidden_dim=32, reconstruction_size=8,
        epochs=2, consistency_epochs=1, batch_size=16,
    )
    model, report = run_nursery_scenario(str(tmp_path), "walk_forward_short", cfg)

    assert report.scenario == "walk_forward_short"
    assert len(report.train_sessions) == 2
    assert len(report.holdout_sessions) == 1
    for session_dir in report.train_sessions + report.holdout_sessions:
        assert os.path.isdir(session_dir)
    assert set(report.horizon_metrics) == {1, 3}
    assert report.dream_strips
