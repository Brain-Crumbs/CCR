"""Crafter nursery scenario ports (issue #90; extended by issue #202):
walk_forward_short, blocked_forward, turn, object_permanence, approach_entity
registered in ``CRAFTER_SCENARIOS`` alongside Minecraft's
``NURSERY_SCENARIOS``. Milestone 1 exit gate: these record deterministically,
with genuine frame-to-frame motion, and pass the data-quality gates.
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("crafter")
torch = pytest.importorskip("torch")

from cognitive_runtime.programs.crafter.streams import SEMANTIC_LEGEND_NAMES  # noqa: E402
from cognitive_runtime.runtime.replay import list_episodes  # noqa: E402
from cognitive_runtime.training.nursery import (  # noqa: E402
    CRAFTER_SCENARIOS,
    NurseryConfig,
    _record_scenario_episode,
    _scenarios_for_world,
    measure_recording_quality,
    run_nursery_scenario,
    validate_nursery_recordings,
)

_COW_ID = next(i for i, n in SEMANTIC_LEGEND_NAMES.items() if n == "cow")
_ENTITY_IDS = {
    i for i, n in SEMANTIC_LEGEND_NAMES.items() if n in ("cow", "zombie", "skeleton")
}


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
        "approach_entity",
    }


def test_scenarios_for_world_selects_the_right_registry():
    from cognitive_runtime.training.nursery import NURSERY_SCENARIOS

    assert _scenarios_for_world("crafter") is CRAFTER_SCENARIOS
    assert _scenarios_for_world("minecraft") is NURSERY_SCENARIOS
    with pytest.raises(ValueError, match="unknown nursery world"):
        _scenarios_for_world("not-a-world")


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


@pytest.mark.parametrize("scenario_name", ["walk_forward_short", "blocked_forward", "turn"])
def test_non_entity_scenarios_contain_no_cow_semantic_id(tmp_path, scenario_name):
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
        assert not (ids & _ENTITY_IDS), (scenario_name, seed, ids)


@pytest.mark.parametrize("scenario_name", ["approach_entity", "object_permanence"])
def test_entity_scenarios_contain_only_the_scripted_entity_population(tmp_path, scenario_name):
    """Issue #202: an entity scenario's recorded population must be exactly
    its own scripted entity (one cow), nothing else world generation
    happened to spawn alongside it."""
    scenario = CRAFTER_SCENARIOS[scenario_name]
    cfg = _crafter_config(train_seeds=(0, 1, 2, 3), holdout_seeds=(1000,))
    for seed in (0, 1, 2, 3, 1000):
        session_dir = _record_scenario_episode(
            str(tmp_path), f"crafter-{scenario_name}-{seed}", seed, scenario, cfg
        )
        episode_id = list_episodes(session_dir)[0]
        ids = _semantic_ids_seen(session_dir, episode_id)
        assert ids & _ENTITY_IDS == {_COW_ID}, (scenario_name, seed, ids)


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


@pytest.mark.parametrize("scenario_name", ["turn", "walk_forward_short"])
def test_split_overlap_gate_passes_a_genuine_crafter_recording(tmp_path, scenario_name):
    """Issue #202: scenarios with no deterministic convergence point (no
    scripted entity/wall placed relative to the player's fixed spawn) leave
    genuinely different, non-overlapping recordings per seed -- the gate
    should be safe to enable outright for these at its default threshold.
    (``approach_entity``/``object_permanence``/``blocked_forward`` all place
    their scripted target relative to the player's spawn, which Crafter
    always puts at the exact world center regardless of seed -- their early
    approach frames legitimately share much of their background across
    episodes by design, per ``NurseryConfig.split_overlap_gate``'s
    docstring, and need a looser or scenario-specific threshold.)"""
    cfg = _crafter_config(
        train_seeds=(0, 1, 2), holdout_seeds=(1000, 1001),
        horizons=(1,), latent_width=16, hidden_dim=32, reconstruction_size=8,
        epochs=1, consistency_epochs=0, batch_size=16,
        split_overlap_gate=True,
    )
    _model, report = run_nursery_scenario(str(tmp_path), scenario_name, cfg)
    assert report.scenario == scenario_name


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
