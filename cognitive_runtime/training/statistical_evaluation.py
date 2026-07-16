"""Statistical evaluation harness (issue #44).

Deterministic byte-identical replay (``runtime/replay.py``) and the
deterministic acceptance harnesses (``online_q_acceptance.py``,
``actor_critic_acceptance.py``) proved plumbing correctness by re-simulating
one episode bit-for-bit. That guarantee does not survive neural online
training (torch/GPU nondeterminism, weights changing mid-episode) or the
live remote backend, so it can no longer be the regression story for
learning runs.

What replaces it: run N episodes per policy/checkpoint on matched conditions
(same curriculum stage/world config; same seed set in sim, or the same
server/time budget live), and report **mean +/- confidence interval** on the
metric families that matter -- survival, reward by tier (issue #41),
exploration coverage, world-model prediction error (issue #39), and death
causes -- instead of a single run. Two such reports (a baseline and a
candidate checkpoint) are compared metric-by-metric; a regression is flagged
only when the candidate's confidence interval no longer overlaps the
baseline's *and* sits on the worse side -- an incidental one-episode dip
should not fail a gate, but a checkpoint that consistently and significantly
survives fewer ticks or explores less should.

The confidence interval uses a normal approximation
(``statistics.NormalDist``) rather than a t-distribution, to avoid adding a
scipy dependency; with the episode counts these harnesses actually use
(tens, not few), the approximation is adequate for regression flagging.

``cortex_horizon_statistics``/``compare_cortex_horizon_statistics`` (issue
#92) reuse the same ``_mean_ci``/``MetricStats``/``MetricComparison``
machinery for the predictive cortex's per-horizon held-out MSE, the metric
family ``action_world_model.evaluate_action_world_model`` reports.
"""

from __future__ import annotations

import os
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from cognitive_runtime.core.policy import Policy
from cognitive_runtime.core.program import Program
from cognitive_runtime.runtime.recorder import EpisodeSummary
from cognitive_runtime.training.evaluation import run_policy
from cognitive_runtime.tools.metrics_dashboard import load_summaries

DEFAULT_CONFIDENCE = 0.95


# --------------------------------------------------------------- statistics


@dataclass(frozen=True)
class MetricStats:
    """Mean +/- confidence interval for one metric over N episodes."""

    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n, "mean": self.mean, "std": self.std,
            "ci_low": self.ci_low, "ci_high": self.ci_high, "confidence": self.confidence,
        }

    def overlaps(self, other: "MetricStats") -> bool:
        return self.ci_low <= other.ci_high and other.ci_low <= self.ci_high


def _mean_ci(values: Sequence[float], confidence: float = DEFAULT_CONFIDENCE) -> MetricStats:
    """Mean +/- CI over ``values``; a single sample has zero-width CI (not
    enough data to bound uncertainty, not "certainty")."""
    n = len(values)
    if n == 0:
        return MetricStats(n=0, mean=0.0, std=0.0, ci_low=0.0, ci_high=0.0, confidence=confidence)
    mean = statistics.fmean(values)
    if n < 2:
        return MetricStats(n=n, mean=mean, std=0.0, ci_low=mean, ci_high=mean, confidence=confidence)
    std = statistics.stdev(values)
    z = statistics.NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    margin = z * std / (n ** 0.5)
    return MetricStats(
        n=n, mean=round(mean, 6), std=round(std, 6),
        ci_low=round(mean - margin, 6), ci_high=round(mean + margin, 6), confidence=confidence,
    )


@dataclass(frozen=True)
class PolicyStatistics:
    """Statistical summary for one policy/checkpoint over N episodes."""

    policy: str
    episodes: int
    survival_ticks: MetricStats
    total_reward: MetricStats
    death_rate: float
    death_causes: Dict[str, float]
    exploration_coverage: Optional[MetricStats] = None
    prediction_error: Optional[MetricStats] = None
    novelty: Optional[MetricStats] = None
    reward_by_tier: Dict[str, MetricStats] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "episodes": self.episodes,
            "survival_ticks": self.survival_ticks.to_dict(),
            "total_reward": self.total_reward.to_dict(),
            "death_rate": self.death_rate,
            "death_causes": dict(self.death_causes),
            "exploration_coverage": (
                self.exploration_coverage.to_dict() if self.exploration_coverage else None
            ),
            "prediction_error": self.prediction_error.to_dict() if self.prediction_error else None,
            "novelty": self.novelty.to_dict() if self.novelty else None,
            "reward_by_tier": {k: v.to_dict() for k, v in self.reward_by_tier.items()},
        }


def compute_statistics(
    policy: str, summaries: List[EpisodeSummary], confidence: float = DEFAULT_CONFIDENCE,
) -> PolicyStatistics:
    """Aggregate ``summaries`` (one policy/checkpoint's episodes, matched
    conditions) into a :class:`PolicyStatistics` report."""
    n = len(summaries)
    ticks = [float(s.duration_ticks) for s in summaries]
    rewards = [s.total_reward for s in summaries]
    deaths = [s for s in summaries if s.termination_reason.startswith("death")]

    coverage = [
        float(s.program_stats["exploration_coverage"]) for s in summaries
        if "exploration_coverage" in s.program_stats
    ]
    pred_error = [s.avg_prediction_error for s in summaries if s.avg_prediction_error is not None]
    novelty = [s.avg_novelty for s in summaries if s.avg_novelty is not None]

    tiers: Dict[str, List[float]] = {}
    for s in summaries:
        by_tier = s.program_stats.get("reward_by_tier")
        if not isinstance(by_tier, dict):
            continue
        for tier in by_tier:
            tiers.setdefault(tier, [])
    for tier, values in tiers.items():
        for s in summaries:
            by_tier = s.program_stats.get("reward_by_tier") or {}
            values.append(float(by_tier.get(tier, 0.0)))

    death_causes = Counter(
        s.program_stats.get("death_reason") for s in deaths if s.program_stats.get("death_reason")
    )

    return PolicyStatistics(
        policy=policy,
        episodes=n,
        survival_ticks=_mean_ci(ticks, confidence),
        total_reward=_mean_ci(rewards, confidence),
        death_rate=round(len(deaths) / n, 3) if n else 0.0,
        death_causes={cause: round(count / n, 3) for cause, count in death_causes.items()} if n else {},
        exploration_coverage=_mean_ci(coverage, confidence) if coverage else None,
        prediction_error=_mean_ci(pred_error, confidence) if pred_error else None,
        novelty=_mean_ci(novelty, confidence) if novelty else None,
        reward_by_tier={tier: _mean_ci(values, confidence) for tier, values in tiers.items()},
    )


# --------------------------------------------------------------- comparison


#: Metric name -> True if higher is better, False if lower is better.
_CORE_METRIC_DIRECTIONS: Dict[str, bool] = {
    "survival_ticks": True,
    "total_reward": True,
    "exploration_coverage": True,
    "prediction_error": False,
}


@dataclass(frozen=True)
class MetricComparison:
    """One metric's statistical comparison between a baseline and a candidate."""

    metric: str
    baseline: MetricStats
    candidate: MetricStats
    higher_is_better: bool
    direction: str  # "improved" | "regressed" | "no_significant_difference"

    @property
    def significant(self) -> bool:
        return self.direction != "no_significant_difference"

    @property
    def regressed(self) -> bool:
        return self.direction == "regressed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "higher_is_better": self.higher_is_better,
            "direction": self.direction,
        }

    def __str__(self) -> str:
        return (
            f"{self.metric}: baseline={self.baseline.mean:.4f} "
            f"[{self.baseline.ci_low:.4f}, {self.baseline.ci_high:.4f}]  "
            f"candidate={self.candidate.mean:.4f} "
            f"[{self.candidate.ci_low:.4f}, {self.candidate.ci_high:.4f}]  "
            f"-> {self.direction}"
        )


def _direction(baseline: MetricStats, candidate: MetricStats, higher_is_better: bool) -> str:
    if baseline.n < 2 or candidate.n < 2 or baseline.overlaps(candidate):
        return "no_significant_difference"
    candidate_better = candidate.mean > baseline.mean if higher_is_better else candidate.mean < baseline.mean
    return "improved" if candidate_better else "regressed"


def compare_statistics(
    baseline: PolicyStatistics,
    candidate: PolicyStatistics,
    metrics: Optional[Sequence[str]] = None,
) -> List[MetricComparison]:
    """Compare ``candidate`` against ``baseline`` metric-by-metric.

    Non-overlapping confidence intervals on the worse side flag a
    "regressed" direction; non-overlapping on the better side flags
    "improved"; overlapping (or too few episodes to compute a CI) is
    "no_significant_difference" -- the harness deliberately does not flag a
    regression it cannot statistically support.
    """
    core = dict(_CORE_METRIC_DIRECTIONS)
    tier_names = sorted(set(baseline.reward_by_tier) & set(candidate.reward_by_tier))
    available: Dict[str, Tuple[MetricStats, MetricStats, bool]] = {}
    for name, higher_is_better in core.items():
        b, c = getattr(baseline, name), getattr(candidate, name)
        if b is not None and c is not None:
            available[name] = (b, c, higher_is_better)
    for tier in tier_names:
        available[f"reward_by_tier.{tier}"] = (
            baseline.reward_by_tier[tier], candidate.reward_by_tier[tier], True,
        )

    names = list(metrics) if metrics is not None else list(available)
    comparisons = []
    for name in names:
        if name not in available:
            continue
        b, c, higher_is_better = available[name]
        comparisons.append(
            MetricComparison(
                metric=name, baseline=b, candidate=c, higher_is_better=higher_is_better,
                direction=_direction(b, c, higher_is_better),
            )
        )
    return comparisons


def flagged_regressions(comparisons: Sequence[MetricComparison]) -> List[MetricComparison]:
    return [c for c in comparisons if c.regressed]


# ------------------------------------------------------------- cortex scoring


def cortex_horizon_statistics(
    per_episode_mse: Dict[int, Sequence[float]], confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[int, MetricStats]:
    """Mean +/- CI over held-out episodes/seeds for each horizon's cortex
    model MSE (issue #92) -- the per-horizon counterpart of
    :func:`compute_statistics`, built on
    ``action_world_model.evaluate_action_world_model``'s
    ``per_episode_model_mse`` (one independent sample per held-out
    episode/seed, not the many overlapping rollout-window samples pooled
    into its ``horizons[h]["model_mse"]`` point estimate)."""
    return {h: _mean_ci(values, confidence) for h, values in per_episode_mse.items()}


def compare_cortex_horizon_statistics(
    baseline: Dict[int, MetricStats], candidate: Dict[int, MetricStats],
) -> Dict[int, MetricComparison]:
    """Per-horizon regression check for cortex MSE (lower is better): a
    ``candidate`` (e.g. an action-ablated model) whose CI sits entirely above
    ``baseline``'s at some horizon is flagged "regressed" there -- the
    action-ablation harness's "measurably hurts" claim, refereed the same
    way whole-episode metrics are."""
    common = sorted(set(baseline) & set(candidate))
    return {
        h: MetricComparison(
            metric=f"horizon_{h}_model_mse",
            baseline=baseline[h],
            candidate=candidate[h],
            higher_is_better=False,
            direction=_direction(baseline[h], candidate[h], False),
        )
        for h in common
    }


# ------------------------------------------------------------------ running


def run_statistical_evaluation(
    program_factory: Callable[[], Program],
    policy_factory: Callable[[], Policy],
    episodes: int,
    seed: int,
    max_ticks: int,
    *,
    record_dir: Optional[str] = None,
    session_id: Optional[str] = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PolicyStatistics:
    """Run ``episodes`` matched-seed episodes of one policy and report
    :class:`PolicyStatistics` -- the harness's live/sim runner counterpart to
    :func:`evaluate_recorded_sessions` (recorded sessions)."""
    summaries = run_policy(
        program=program_factory(), policy=policy_factory(), episodes=episodes,
        seed=seed, max_ticks=max_ticks, record_dir=record_dir, session_id=session_id,
    )
    name = summaries[0].policy_name if summaries else "unknown"
    return compute_statistics(name, summaries, confidence=confidence)


def evaluate_recorded_sessions(
    record_dir: str, confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[Tuple[str, str], PolicyStatistics]:
    """Statistical report from every already-recorded session under
    ``record_dir``, grouped by ``(curriculum, policy)`` -- the "produces the
    statistical report from recorded sessions" acceptance criterion.

    Mirrors :func:`cognitive_runtime.tools.metrics_dashboard.dashboard`'s
    grouping so the same recordings back both the plain-mean dashboard and
    this statistical view.
    """
    by_group: Dict[Tuple[str, str], List[EpisodeSummary]] = {}
    if not os.path.isdir(record_dir):
        return {}
    for session_id in sorted(os.listdir(record_dir)):
        session_dir = os.path.join(record_dir, session_id)
        if not os.path.isdir(session_dir):
            continue
        for summary in load_summaries(session_dir):
            key = (summary.curriculum or "-", summary.policy_name)
            by_group.setdefault(key, []).append(summary)
    return {
        key: compute_statistics(key[1], group, confidence=confidence)
        for key, group in by_group.items()
    }


# ------------------------------------------------------------------ reports


def format_statistics_report(stats: Sequence[PolicyStatistics]) -> str:
    """Plain-text mean +/- CI table, one row per policy/checkpoint."""
    if not stats:
        return "(no results)"
    lines = []
    header = (
        f"{'policy':<20} {'n':>4} {'survival_ticks':>28} {'total_reward':>26} "
        f"{'death_rate':>10} {'coverage':>22}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for s in stats:
        ticks = s.survival_ticks
        reward = s.total_reward
        cov = s.exploration_coverage
        cov_str = f"{cov.mean:.2f} [{cov.ci_low:.2f}, {cov.ci_high:.2f}]" if cov else "-"
        lines.append(
            f"{s.policy:<20} {s.episodes:>4} "
            f"{ticks.mean:>10.1f} [{ticks.ci_low:>7.1f}, {ticks.ci_high:>7.1f}] "
            f"{reward.mean:>8.3f} [{reward.ci_low:>7.3f}, {reward.ci_high:>7.3f}] "
            f"{s.death_rate:>10.3f} {cov_str:>22}"
        )
        if s.reward_by_tier:
            tier_str = ", ".join(
                f"{tier}={m.mean:.3f} [{m.ci_low:.3f}, {m.ci_high:.3f}]"
                for tier, m in sorted(s.reward_by_tier.items())
            )
            lines.append(f"{'':<20}   reward by tier: {tier_str}")
        if s.death_causes:
            causes_str = ", ".join(
                f"{cause}={rate:.3f}" for cause, rate in sorted(s.death_causes.items())
            )
            lines.append(f"{'':<20}   death causes: {causes_str}")
    return "\n".join(lines)


def format_comparison_report(comparisons: Sequence[MetricComparison]) -> str:
    if not comparisons:
        return "(no comparable metrics)"
    return "\n".join(f"  {c}" for c in comparisons)
