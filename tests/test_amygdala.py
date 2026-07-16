"""Fast threat appraisal -> adrenaline release (issue #94, task 2):
rising predicted-pain produces an adrenaline spike; a calm scene keeps it
near zero; reuses `safe_gate`."""

from __future__ import annotations

import pytest

from brain.amygdala import ADRENALINE_STREAM, Amygdala, AmygdalaConfig
from brain.neuromod.modulation import safe_gate


def test_calm_scene_keeps_adrenaline_near_zero():
    amygdala = Amygdala()
    level = 0.0
    for _ in range(30):
        level = amygdala.appraise(risk=0.05)
    # Settles at the (small but nonzero) threat reading for a risk well
    # below threat_threshold, not literally 0.0 -- quiet relative to the
    # >0.5 spike a real threat produces (below), not exactly zero.
    assert level < 0.1


def test_rising_predicted_pain_produces_an_adrenaline_spike():
    amygdala = Amygdala()
    for _ in range(10):
        amygdala.appraise(risk=0.05)
    calm_level = amygdala.level
    assert calm_level < 0.05

    levels = [amygdala.appraise(risk=0.95) for _ in range(4)]
    assert levels[-1] > 0.5  # a real spike, not a rounding blip
    assert levels[-1] > calm_level
    assert levels == sorted(levels)  # monotonically rising into the threat


def test_adrenaline_release_is_faster_than_its_clearance():
    """Real adrenaline clears slower than it releases (module docstring):
    the same number of ticks spent rising from quiet to danger should move
    the level further than the same number of ticks spent falling back."""
    amygdala = Amygdala()
    for _ in range(20):
        amygdala.appraise(risk=0.05)
    baseline = amygdala.level

    rise_levels = [amygdala.appraise(risk=0.95) for _ in range(3)]
    peak = rise_levels[-1]
    rise_distance = peak - baseline

    fall_levels = [amygdala.appraise(risk=0.05) for _ in range(3)]
    fall_distance = peak - fall_levels[-1]

    assert rise_distance > fall_distance


def test_reset_clears_the_tracked_level():
    amygdala = Amygdala()
    for _ in range(5):
        amygdala.appraise(risk=0.9)
    assert amygdala.level > 0.3
    amygdala.reset()
    assert amygdala.level == 0.0


def test_appraise_reuses_safe_gate_directly():
    config = AmygdalaConfig(threat_threshold=0.5, threat_temperature=0.15)
    amygdala = Amygdala(config)
    expected = 1.0 - safe_gate(0.8, config.threat_threshold, config.threat_temperature)
    level = amygdala.appraise(risk=0.8)
    # First tick: the tracked level starts at 0.0 < expected threat, so the
    # rise EMA applies -- not a full jump to `expected`, but strictly
    # between the starting point and the raw appraisal.
    assert 0.0 < level < expected + 1e-9


def test_predicted_risk_aversion_drives_the_appraisal_when_more_alarming():
    """`predicted_risk_aversion` is `-risk` under the current modulation
    math, but the amygdala takes whichever reading indicates more danger
    (module docstring) -- feeding a more-aversive aversion term than the
    raw risk value still raises the appraisal."""
    amygdala = Amygdala()
    level_risk_only = amygdala.appraise(risk=0.1)
    amygdala.reset()
    level_with_aversion = amygdala.appraise(risk=0.1, predicted_risk_aversion=-0.9)
    assert level_with_aversion > level_risk_only


def test_threat_threshold_and_temperature_are_configurable():
    permissive = Amygdala(AmygdalaConfig(threat_threshold=0.95, threat_temperature=0.15))
    strict = Amygdala(AmygdalaConfig(threat_threshold=0.1, threat_temperature=0.15))
    for _ in range(10):
        permissive_level = permissive.appraise(risk=0.5)
        strict_level = strict.appraise(risk=0.5)
    assert strict_level > permissive_level


def test_rise_alpha_must_be_positive_and_at_most_one():
    with pytest.raises(ValueError, match="rise_alpha"):
        AmygdalaConfig(rise_alpha=0.0)


def test_fall_alpha_must_be_positive_and_at_most_one():
    with pytest.raises(ValueError, match="fall_alpha"):
        AmygdalaConfig(fall_alpha=1.5)


def test_as_payload_matches_uniform_internal_stream_shape():
    amygdala = Amygdala()
    amygdala.appraise(risk=0.6)
    payload = amygdala.as_payload()
    assert set(payload) == {"value"}
    assert payload["value"] == pytest.approx(amygdala.level, abs=1e-6)


def test_adrenaline_stream_id_is_the_named_internal_stream():
    assert ADRENALINE_STREAM == "internal.adrenaline"
