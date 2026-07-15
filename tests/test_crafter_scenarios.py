"""Crafter nursery scenario ports (issue #90): walk_forward, turn,
object_permanence, approach_entity registered in ``CRAFTER_SCENARIOS``
alongside Minecraft's ``NURSERY_SCENARIOS``. Milestone 1 exit gate: these
record deterministically, with genuine frame-to-frame motion, and pass the
data-quality gates.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("crafter")
torch = pytest.importorskip("torch")

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


def _crafter_config(**overrides) -> NurseryConfig:
    base = dict(world="crafter", episode_ticks=40, train_seeds=(0, 1), holdout_seeds=(1000,))
    base.update(overrides)
    return NurseryConfig(**base)


def test_registry_has_every_crafter_scenario():
    assert set(CRAFTER_SCENARIOS) == {
        "walk_forward", "turn", "object_permanence", "approach_entity",
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


def test_walk_forward_has_genuine_frame_to_frame_motion(tmp_path):
    scenario = CRAFTER_SCENARIOS["walk_forward"]
    cfg = _crafter_config()
    session_dir = _record_scenario_episode(str(tmp_path), "crafter-walk", 0, scenario, cfg)
    quality = measure_recording_quality(session_dir, list_episodes(session_dir)[0])

    assert quality.blocks_per_tick >= scenario.min_blocks_per_tick
    assert quality.unique_frame_fraction >= scenario.min_unique_frame_fraction
    assert quality.pixel_sources  # real pixel provenance is recorded


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


def test_run_nursery_scenario_end_to_end_against_crafter(tmp_path):
    """``ccr nursery run --world crafter walk_forward`` (phase-1's acceptance
    criterion): records train/holdout episodes and trains/evaluates the same
    pixel-prediction pipeline Minecraft uses -- world-agnostic downstream of
    the recorded session dir."""
    cfg = _crafter_config(
        horizons=(1, 3), latent_width=16, hidden_dim=32, reconstruction_size=8,
        epochs=2, consistency_epochs=1, batch_size=16,
    )
    model, report = run_nursery_scenario(str(tmp_path), "walk_forward", cfg)

    assert report.scenario == "walk_forward"
    assert len(report.train_sessions) == 2
    assert len(report.holdout_sessions) == 1
    for session_dir in report.train_sessions + report.holdout_sessions:
        assert os.path.isdir(session_dir)
    assert set(report.horizon_metrics) == {1, 3}
    assert report.dream_strips
