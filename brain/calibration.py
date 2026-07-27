"""Calibration: reliability diagrams, Expected Calibration Error (ECE), and
temperature scaling for a binary "is this reading surprising" forecast
(docs/v2/phases/phase-3-neuromodulators-arbiter.md, issue #95, task 4).

The arbiter's 2x2 lookup (`brain.arbiter`) is only as good as its inputs: a
raw surprise reading (cortex sigma / prediction-error stand-in) is a
*confidence* that this tick is surprising, and a confidence is only useful
if it means what it says -- a reading of 0.8 should turn out to be "this
tick was surprising" roughly 80% of the time it fires, not some other rate
the raw scale happens to produce. This module makes that measurable
(:func:`expected_calibration_error`, :func:`reliability_diagram`) and
correctable (:func:`fit_temperature`) rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

__all__ = [
    "ReliabilityBin",
    "reliability_diagram",
    "expected_calibration_error",
    "TemperatureFit",
    "fit_temperature",
    "sigmoid",
    "logit",
]

#: The grid `fit_temperature` searches by default: `T < 1` sharpens an
#: underconfident head, `T > 1` softens an overconfident one, `T == 1` is a
#: no-op (the raw reading was already well-calibrated).
DEFAULT_TEMPERATURE_GRID: Tuple[float, ...] = tuple(round(0.1 + 0.05 * i, 2) for i in range(1, 120))


def sigmoid(x: float) -> float:
    """Overflow-safe logistic sigmoid (mirrors `brain.neuromod.safe_gate`'s
    branch-on-sign idiom)."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def logit(p: float, eps: float = 1e-6) -> float:
    """Inverse sigmoid, clamped away from 0/1 so a saturated confidence
    never produces +/-inf."""
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class ReliabilityBin:
    """One confidence bucket: how many readings fell in it, what they
    claimed on average, and how often the claim actually held."""

    lower: float
    upper: float
    count: int
    avg_confidence: float
    empirical_rate: float


def reliability_diagram(
    confidences: Sequence[float], outcomes: Sequence[bool], n_bins: int = 10
) -> List[ReliabilityBin]:
    """Bucket `confidences` into `n_bins` equal-width `[0, 1]` bins and
    compare each bin's average confidence against its empirical outcome
    rate -- the standard reliability-diagram construction. Empty bins are
    kept (count=0) so callers get one entry per bin, not a sparse list."""
    if len(confidences) != len(outcomes):
        raise ValueError(
            f"confidences and outcomes must be the same length, got "
            f"{len(confidences)} and {len(outcomes)}"
        )
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins!r}")
    edges = [i / n_bins for i in range(n_bins + 1)]
    # One pass over the data (not one pass per bin): each point's bin index
    # is `floor(c * n_bins)`, clamped into range so `c == 1.0` (or a
    # slightly-out-of-range float) lands in the last bin rather than
    # overflowing it -- equivalent to the half-open `[lower, upper)`
    # convention with the last bin closed on both ends.
    counts = [0] * n_bins
    confidence_sums = [0.0] * n_bins
    outcome_sums = [0.0] * n_bins
    for c, o in zip(confidences, outcomes):
        index = min(n_bins - 1, max(0, int(c * n_bins)))
        counts[index] += 1
        confidence_sums[index] += c
        outcome_sums[index] += 1.0 if o else 0.0
    bins: List[ReliabilityBin] = []
    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        count = counts[i]
        if count == 0:
            bins.append(ReliabilityBin(lower, upper, 0, 0.0, 0.0))
        else:
            bins.append(
                ReliabilityBin(lower, upper, count, confidence_sums[i] / count, outcome_sums[i] / count)
            )
    return bins


def expected_calibration_error(
    confidences: Sequence[float], outcomes: Sequence[bool], n_bins: int = 10
) -> float:
    """ECE: the count-weighted average gap between each bin's average
    confidence and its empirical outcome rate. `0.0` when there is nothing
    to score (no observations) -- quiet, not undefined, matching this
    codebase's convention for "no signal yet" (e.g. `Amygdala.level` at
    rest)."""
    bins = reliability_diagram(confidences, outcomes, n_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        return 0.0
    return sum(b.count * abs(b.avg_confidence - b.empirical_rate) for b in bins) / total


@dataclass(frozen=True)
class TemperatureFit:
    """A `fit_temperature` result: the chosen temperature and the ECE it
    achieves, next to the raw (unscaled) ECE it improved on."""

    temperature: float
    ece_before: float
    ece_after: float


def fit_temperature(
    confidences: Sequence[float],
    outcomes: Sequence[bool],
    n_bins: int = 10,
    candidates: Sequence[float] = DEFAULT_TEMPERATURE_GRID,
) -> TemperatureFit:
    """Grid-search the scalar temperature `T` that minimizes ECE when raw
    confidences are remapped through `sigmoid(logit(c) / T)` (temperature
    scaling, Guo et al. 2017) -- `T > 1` softens an overconfident head
    (pulls extreme confidences back toward 0.5), `T < 1` sharpens an
    underconfident one. Falls back to `T = 1.0` (no-op) if nothing in the
    grid beats the raw ECE, so a caller never applies a "fit" that makes
    things worse."""
    if not confidences:
        raise ValueError("need at least one observation to fit a temperature")
    ece_before = expected_calibration_error(confidences, outcomes, n_bins)
    best_temperature, best_ece = 1.0, ece_before
    for temperature in candidates:
        scaled = [sigmoid(logit(c) / temperature) for c in confidences]
        ece = expected_calibration_error(scaled, outcomes, n_bins)
        if ece < best_ece:
            best_temperature, best_ece = temperature, ece
    return TemperatureFit(temperature=best_temperature, ece_before=ece_before, ece_after=best_ece)
