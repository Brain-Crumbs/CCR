"""Forgetting metric (issue #99): does an old scenario's accuracy survive
learning a new one?

Retention is judged the same way the rest of ``statistical_evaluation``
referees regressions: a candidate ("after learning the new scenario") is
compared against a baseline ("before") via non-overlapping confidence
intervals over held-out samples, not a single point estimate -- so an
incidental dip is not mistaken for forgetting, and a real one cannot hide
behind a lucky rerun.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from cognitive_runtime.training.statistical_evaluation import (
    MetricComparison,
    MetricStats,
    compare_scalar_metrics,
)

__all__ = ["ForgettingReport", "compute_forgetting_metric", "compare_forgetting_conditions"]


@dataclass(frozen=True)
class ForgettingReport:
    """Retention of ``old_scenario`` after training ``new_scenario`` under
    one ``condition`` (e.g. ``"staged+replay"`` or ``"flat"``).

    ``comparison`` is the before-vs-after :class:`MetricComparison` (CI
    overlap against the model's *own* pre-new-training mastery) -- useful
    context, but too strict a bar for "retained": any nonzero interference
    between the two scenarios' learned mappings nudges ``after`` away from
    ``before``'s often near-zero-variance point estimate even when the
    model still works. ``retained`` instead asks the same question
    ``beats_copy_last`` asks everywhere else in this codebase: is the
    model, after learning something new, still meaningfully better at the
    old scenario than predicting no change at all?
    """

    old_scenario: str
    new_scenario: str
    condition: str
    before: MetricStats
    after: MetricStats
    copy_last_mse: float
    comparison: MetricComparison

    @property
    def retained(self) -> bool:
        """True iff ``after``'s confidence interval sits confidently on the
        useful side of the copy-last baseline on ``old_scenario`` -- not
        just its point estimate, which sampling noise could otherwise put
        on the wrong side of the bar. Honors the metric's own direction
        (``comparison.higher_is_better``), so an accuracy-style metric
        (higher is better) and an MSE-style one (lower is better) are each
        judged correctly instead of always applying the MSE convention."""
        if self.comparison.higher_is_better:
            return self.after.ci_low > self.copy_last_mse
        return self.after.ci_high < self.copy_last_mse

    def to_dict(self) -> Dict[str, object]:
        return {
            "old_scenario": self.old_scenario,
            "new_scenario": self.new_scenario,
            "condition": self.condition,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "copy_last_mse": self.copy_last_mse,
            "comparison": self.comparison.to_dict(),
        }


def compute_forgetting_metric(
    old_scenario: str,
    new_scenario: str,
    condition: str,
    before: MetricStats,
    after: MetricStats,
    copy_last_mse: float,
    *,
    metric_name: str = "old_scenario_mse",
    higher_is_better: bool = False,
) -> ForgettingReport:
    """Compare ``old_scenario``'s held-out quality before vs. after training
    ``new_scenario`` under ``condition``, against ``copy_last_mse`` (the
    old scenario's copy-last-frame/latent baseline, e.g.
    ``sleep.replay.evaluate_next_latent_quality(..., old_holdout, ...)``'s
    ``"copy_last_mse"``) for the ``retained`` bar. Defaults to an MSE-style
    metric (lower is better, matching ``action_world_model``/
    ``sleep.replay``'s baseline-relative convention); pass
    ``higher_is_better=True`` for an accuracy-style metric instead."""
    comparison = compare_scalar_metrics(metric_name, before, after, higher_is_better)
    return ForgettingReport(
        old_scenario=old_scenario, new_scenario=new_scenario, condition=condition,
        before=before, after=after, copy_last_mse=copy_last_mse, comparison=comparison,
    )


def compare_forgetting_conditions(staged: ForgettingReport, flat: ForgettingReport) -> bool:
    """Milestone 5's measured claim: staged+replay retains the previously
    mastered scenario while flat training on the same new data does not.

    Requires both (a) each condition's own CI-backed ``retained`` verdict
    against the copy-last bar and (b) a statistically significant
    separation between the two conditions' ``after`` scores themselves
    (non-overlapping confidence intervals, not just two point estimates
    straddling copy-last) -- otherwise sampling noise near the bar could
    manufacture the claim even when staged and flat are not meaningfully
    different from each other.
    """
    if staged.old_scenario != flat.old_scenario or staged.new_scenario != flat.new_scenario:
        raise ValueError("staged and flat reports must cover the same scenario pair")
    if staged.copy_last_mse != flat.copy_last_mse:
        raise ValueError("staged and flat reports must share the same copy-last baseline")
    higher_is_better = staged.comparison.higher_is_better
    separation = compare_scalar_metrics("staged_vs_flat_after", flat.after, staged.after, higher_is_better)
    return staged.retained and not flat.retained and separation.direction == "improved"
