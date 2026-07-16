"""World-agnostic recording-quality gates (issue #90): ``record.quality``
reads any Program's stream log the same way -- pixel provenance, motion
floor, completed-episode, and a facing-sweep check that covers both
continuous yaw (Minecraft) and discrete facing (Crafter). Exercises the
gates directly (not through ``training.nursery``'s ``NurseryScenario``
adapter) against a recording from each world, plus the green/amber/red
verdict.
"""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.record.quality import (  # noqa: E402
    validate_recording_quality,
    validate_recordings,
    verdict_for_session,
)
from cognitive_runtime.runtime.replay import list_episodes  # noqa: E402
from cognitive_runtime.training.nursery import (  # noqa: E402
    CRAFTER_SCENARIOS,
    NURSERY_SCENARIOS,
    NurseryConfig,
    _record_scenario_episode,
    measure_recording_quality,
)


def _record_minecraft_walk(tmp_path, session_id="mc-walk"):
    cfg = NurseryConfig(episode_ticks=26, world_size=16)
    return _record_scenario_episode(
        str(tmp_path), session_id, 0, NURSERY_SCENARIOS["walk_forward"], cfg
    )


def _record_crafter_walk(tmp_path, session_id="crafter-walk"):
    pytest.importorskip("crafter")
    cfg = NurseryConfig(world="crafter", episode_ticks=40)
    return _record_scenario_episode(
        str(tmp_path), session_id, 0, CRAFTER_SCENARIOS["walk_forward"], cfg
    )


def _record_crafter_stationary(tmp_path, session_id="crafter-still"):
    """Object_permanence's player never moves -- a genuinely frozen
    recording from the Crafter side, for the "flag a static session" gate
    checks below."""
    pytest.importorskip("crafter")
    cfg = NurseryConfig(world="crafter", episode_ticks=40)
    return _record_scenario_episode(
        str(tmp_path), session_id, 0, CRAFTER_SCENARIOS["object_permanence"], cfg
    )


# --------------------------------------------------------------------------- measurement


def test_measure_recording_quality_reads_a_minecraft_recording(tmp_path):
    session_dir = _record_minecraft_walk(tmp_path)
    quality = measure_recording_quality(session_dir, list_episodes(session_dir)[0])
    assert quality.n_frames > 0
    assert quality.blocks_per_tick > 0.0
    assert quality.pixel_sources == ["grid"]


def test_measure_recording_quality_reads_a_crafter_recording(tmp_path):
    session_dir = _record_crafter_walk(tmp_path)
    quality = measure_recording_quality(session_dir, list_episodes(session_dir)[0])
    assert quality.n_frames > 0
    assert quality.blocks_per_tick > 0.0
    assert quality.pixel_sources == ["crafter"]


# ------------------------------------------------------------------- validate_recording_quality


def test_validate_recording_quality_passes_a_healthy_recording_from_either_world(tmp_path):
    mc = _record_minecraft_walk(tmp_path, "mc-clean")
    crafter = _record_crafter_walk(tmp_path, "crafter-clean")
    for session_dir in (mc, crafter):
        quality = measure_recording_quality(session_dir, list_episodes(session_dir)[0])
        issues = validate_recording_quality(
            quality, name="walk", min_blocks_per_tick=0.005, min_unique_frame_fraction=0.05,
        )
        assert issues == [], (session_dir, issues)


def test_validate_recording_quality_flags_a_frozen_recording_from_either_world(tmp_path):
    mc_still = _record_scenario_episode(
        str(tmp_path), "mc-still", 0, NURSERY_SCENARIOS["turn_in_place"],
        NurseryConfig(episode_ticks=26, world_size=16),
    )
    crafter_still = _record_crafter_stationary(tmp_path)
    for session_dir in (mc_still, crafter_still):
        quality = measure_recording_quality(session_dir, list_episodes(session_dir)[0])
        issues = validate_recording_quality(
            quality, name="walk", min_blocks_per_tick=0.05,
        )
        assert any("barely moved" in issue for issue in issues), (session_dir, issues)


def test_validate_recordings_checks_pixel_provenance_across_worlds(tmp_path):
    crafter = _record_crafter_walk(tmp_path)
    assert validate_recordings([crafter], name="walk", expected_pixel_source="crafter") == []
    issues = validate_recordings([crafter], name="walk", expected_pixel_source="grid")
    assert any("pixel source" in issue for issue in issues)


# --------------------------------------------------------------------------- discrete facing


def test_validate_recording_quality_flags_missing_unique_facings(tmp_path):
    pytest.importorskip("crafter")
    cfg = NurseryConfig(world="crafter", episode_ticks=40)
    session_dir = _record_scenario_episode(
        str(tmp_path), "crafter-turn", 0, CRAFTER_SCENARIOS["turn"], cfg
    )
    quality = measure_recording_quality(session_dir, list_episodes(session_dir)[0])
    assert quality.unique_facings == 4
    assert validate_recording_quality(quality, name="turn", min_unique_facings=4) == []
    issues = validate_recording_quality(quality, name="turn", min_unique_facings=5)
    assert any("unique facing" in issue for issue in issues)


# --------------------------------------------------------------------------- verdict


def test_verdict_for_session_is_green_for_a_healthy_recording(tmp_path):
    session_dir = _record_crafter_walk(tmp_path)
    verdict = verdict_for_session(
        session_dir, name="walk", min_blocks_per_tick=0.01, min_unique_frame_fraction=0.05,
    )
    assert verdict.verdict == "green"
    assert verdict.issues == []
    assert verdict.warnings == []


def test_verdict_for_session_is_red_for_a_frozen_recording(tmp_path):
    session_dir = _record_crafter_stationary(tmp_path)
    verdict = verdict_for_session(session_dir, name="walk", min_blocks_per_tick=0.05)
    assert verdict.verdict == "red"
    assert verdict.issues


def test_world_agnostic_unique_frame_count_gate_rejects_frozen_pixels(tmp_path):
    session_dir = _record_crafter_stationary(tmp_path)
    verdict = verdict_for_session(session_dir, min_unique_frames=2)
    assert verdict.verdict == "red"
    assert any("recording appears frozen" in issue for issue in verdict.issues)


def test_verdict_for_session_is_amber_when_provenance_predates_tracking(tmp_path):
    session_dir = _record_crafter_walk(tmp_path)
    episode_id = list_episodes(session_dir)[0]
    summary_path = os.path.join(session_dir, f"{episode_id}.summary.json")
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    summary["program_stats"].pop("pixel_sources", None)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh)

    verdict = verdict_for_session(
        session_dir, name="walk", min_blocks_per_tick=0.01, min_unique_frame_fraction=0.05,
    )
    assert verdict.verdict == "amber"
    assert verdict.issues == []
    assert any("provenance" in w for w in verdict.warnings)


def test_verdict_for_session_is_amber_when_motion_is_close_to_the_floor(tmp_path):
    session_dir = _record_crafter_walk(tmp_path)
    quality = measure_recording_quality(session_dir, list_episodes(session_dir)[0])
    # A floor just under the measured rate clears the hard gate but sits
    # within AMBER_MARGIN of it.
    close_floor = quality.blocks_per_tick / 1.2

    verdict = verdict_for_session(session_dir, name="walk", min_blocks_per_tick=close_floor)
    assert verdict.verdict == "amber"
    assert any("motion" in w for w in verdict.warnings)
