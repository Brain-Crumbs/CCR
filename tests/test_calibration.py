"""Calibration metrics for the arbiter's surprise input (issue #95, task 4):
reliability-diagram bucketing, Expected Calibration Error (ECE), and
temperature-scaling correction on a synthetic mis-calibrated head."""

from __future__ import annotations

import random

import pytest

from brain.calibration import (
    expected_calibration_error,
    fit_temperature,
    logit,
    reliability_diagram,
    sigmoid,
)


def _synthetic_forecast(rng: random.Random, n: int):
    """`n` (true_probability, realized_outcome) pairs -- a well-specified
    forecaster's raw material: draw a probability, then draw the outcome it
    actually predicts."""
    true_p = [rng.random() for _ in range(n)]
    outcomes = [rng.random() < p for p in true_p]
    return true_p, outcomes


# ------------------------------------------------------------ reliability_diagram


def test_reliability_diagram_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        reliability_diagram([0.1, 0.2], [True], n_bins=5)


def test_reliability_diagram_rejects_non_positive_bins():
    with pytest.raises(ValueError, match="n_bins"):
        reliability_diagram([0.1], [True], n_bins=0)


def test_reliability_diagram_covers_every_bin_including_empty_ones():
    bins = reliability_diagram([0.05, 0.05], [True, False], n_bins=10)
    assert len(bins) == 10
    assert bins[0].count == 2
    assert all(b.count == 0 for b in bins[1:])


def test_reliability_diagram_last_bin_is_closed_on_both_ends():
    """A confidence of exactly 1.0 must land in the last bin, not be
    dropped by a half-open `[lower, upper)` convention."""
    bins = reliability_diagram([1.0], [True], n_bins=10)
    assert bins[-1].count == 1
    assert bins[-1].avg_confidence == pytest.approx(1.0)


def test_reliability_diagram_bin_confidence_and_rate_average_correctly():
    confidences = [0.15, 0.19, 0.16]  # all in bin [0.1, 0.2)
    outcomes = [True, False, True]
    bins = reliability_diagram(confidences, outcomes, n_bins=10)
    target = bins[1]
    assert target.count == 3
    assert target.avg_confidence == pytest.approx(sum(confidences) / 3)
    assert target.empirical_rate == pytest.approx(2 / 3)


# ------------------------------------------------------------ expected_calibration_error


def test_ece_is_zero_with_no_observations():
    assert expected_calibration_error([], [], n_bins=10) == 0.0


def test_perfectly_calibrated_forecaster_has_near_zero_ece():
    rng = random.Random(0)
    true_p, outcomes = _synthetic_forecast(rng, 4000)
    ece = expected_calibration_error(true_p, outcomes, n_bins=10)
    assert ece < 0.05


def test_always_wrong_confidence_has_high_ece():
    confidences = [0.9] * 100
    outcomes = [False] * 100  # claims 90% confident, right 0% of the time
    ece = expected_calibration_error(confidences, outcomes, n_bins=10)
    assert ece == pytest.approx(0.9, abs=1e-9)


# ------------------------------------------------------------ fit_temperature


def test_fit_temperature_requires_at_least_one_observation():
    with pytest.raises(ValueError, match="at least one observation"):
        fit_temperature([], [])


def test_overconfident_synthetic_head_is_flagged_and_temperature_scaling_corrects_it():
    """The phase doc's acceptance line: 'an uncalibrated head is visibly
    flagged.' A head that squashes every true probability toward the
    extremes (classic overconfidence) must show measurably higher ECE than
    the underlying well-calibrated probabilities, and temperature scaling
    (`T > 1`, softening) must measurably reduce it."""
    rng = random.Random(1)
    true_p, outcomes = _synthetic_forecast(rng, 4000)
    # Overconfident: stretches every probability away from 0.5.
    overconfident = [sigmoid(logit(p) * 4.0) for p in true_p]

    raw_ece = expected_calibration_error(overconfident, outcomes, n_bins=10)
    fit = fit_temperature(overconfident, outcomes, n_bins=10)

    assert raw_ece > 0.15, f"expected the distorted head to be visibly miscalibrated, got {raw_ece}"
    assert fit.ece_before == pytest.approx(raw_ece)
    assert fit.ece_after < raw_ece / 2
    assert fit.temperature > 1.0  # softens an overconfident head


def test_underconfident_synthetic_head_is_flagged_and_temperature_scaling_corrects_it():
    rng = random.Random(2)
    true_p, outcomes = _synthetic_forecast(rng, 4000)
    # Underconfident: pulls every probability toward 0.5.
    underconfident = [sigmoid(logit(p) * 0.25) for p in true_p]

    raw_ece = expected_calibration_error(underconfident, outcomes, n_bins=10)
    fit = fit_temperature(underconfident, outcomes, n_bins=10)

    assert raw_ece > 0.05
    assert fit.ece_after < raw_ece
    assert fit.temperature < 1.0  # sharpens an underconfident head


def test_fit_temperature_never_reports_worse_than_raw_ece():
    rng = random.Random(3)
    true_p, outcomes = _synthetic_forecast(rng, 500)
    fit = fit_temperature(true_p, outcomes, n_bins=10)
    assert fit.ece_after <= fit.ece_before + 1e-9


# ------------------------------------------------------------ sigmoid/logit

def test_sigmoid_and_logit_are_inverses():
    for p in (0.01, 0.25, 0.5, 0.75, 0.99):
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-6)


def test_logit_clamps_saturated_probabilities_instead_of_raising():
    assert logit(0.0) < 0  # would be -inf unclamped
    assert logit(1.0) > 0  # would be +inf unclamped
