"""The forgetting metric (docs/v2/phases/phase-5-sleep-consolidation.md task
5, issue #99) -- Milestone 5's falsifiable result.

Measures whether an old, already-mastered scenario's held-out prediction
loss survives training on a new scenario -- generally, "old-scenario
accuracy after training a new one" -- and routes the before/after comparison
through ``cognitive_runtime.training.statistical_evaluation`` so a regression
is flagged with confidence intervals rather than off a single noisy sample,
the same discipline every other regression gate in this codebase uses.

This module only *reports* the metric from caller-supplied loss samples; it
does not run training itself. The staged+replay-vs-flat-training comparison
the milestone's claim rests on lives in the test that exercises this metric
against both conditions (``tests/test_forgetting_metric.py``) using
``sleep.replay_mix.GenerativeReplayMixer`` for the staged+replay condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from cognitive_runtime.training.statistical_evaluation import (
    DEFAULT_CONFIDENCE,
    MetricComparison,
    MetricStats,
    compare_metric,
    metric_statistics,
)

__all__ = ["ForgettingReport", "compute_forgetting_metric"]


@dataclass(frozen=True)
class ForgettingReport:
    """One old-scenario retention check, before vs. after training a new
    scenario.

    ``before``/``after`` are held-out loss statistics on the old scenario
    (lower is better); ``comparison`` is the CI-refereed regression verdict
    ``statistical_evaluation`` reports; ``retained`` is the tolerance-based
    pass/fail the milestone's "within tolerance" acceptance line asks for --
    a coarser, always-decidable check that does not depend on having enough
    episodes for a non-trivial CI.
    """

    old_scenario: str
    new_scenario: str
    before: MetricStats
    after: MetricStats
    comparison: MetricComparison
    tolerance: float
    retained: bool

    @property
    def forgetting_amount(self) -> float:
        """How much worse (higher-loss) the old scenario got; negative means
        it improved."""
        return self.after.mean - self.before.mean


def compute_forgetting_metric(
    old_scenario_losses_before: Sequence[float],
    old_scenario_losses_after: Sequence[float],
    *,
    old_scenario: str,
    new_scenario: str,
    tolerance: float,
    confidence: float = DEFAULT_CONFIDENCE,
) -> ForgettingReport:
    """Report ``old_scenario``'s retention after training ``new_scenario``.

    ``tolerance`` bounds how much the old scenario's mean held-out loss may
    rise (``after.mean - before.mean``) and still count as "retained" -- the
    phase doc's "forgetting metric stays within tolerance". Separately,
    ``comparison`` flags a statistically-significant regression (non-
    overlapping confidence intervals on the worse side) whenever there is
    enough data to support one; a caller wanting the strict statistical gate
    instead of/in addition to the tolerance check can inspect
    ``comparison.regressed``.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance!r}")
    before = metric_statistics(old_scenario_losses_before, confidence)
    after = metric_statistics(old_scenario_losses_after, confidence)
    comparison = compare_metric(
        before, after, metric=f"{old_scenario}_retention", higher_is_better=False,
    )
    retained = (after.mean - before.mean) <= tolerance
    return ForgettingReport(
        old_scenario=old_scenario,
        new_scenario=new_scenario,
        before=before,
        after=after,
        comparison=comparison,
        tolerance=tolerance,
        retained=retained,
    )
