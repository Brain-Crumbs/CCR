"""Paired per-episode comparison statistics for Model Factory trials (#223).

Candidate and baseline values are keyed by the same frozen validation episode
identifier.  This module deliberately refuses positional or aggregate-only
inputs: a matching count is not evidence that two models saw matching
episodes.  It is pure Python (and specifically never imports torch), so the
comparison report can be produced wherever a validation result is available.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union


COMPARISON_FORMAT = "model-factory-paired-comparison-v1"
DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 10_000
DEFAULT_MIN_EPISODE_COUNT = 5


def _validate_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")


def _validate_resamples(resamples: int) -> None:
    if resamples <= 0:
        raise ValueError("resamples must be positive")


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Linearly interpolated percentile without a numpy/scipy dependency."""
    if not values:
        raise ValueError("cannot compute a percentile of no values")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _validate_deltas(deltas: Sequence[float]) -> Tuple[float, ...]:
    values = tuple(float(value) for value in deltas)
    if not values:
        raise ValueError("at least one paired delta is required")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("paired deltas must all be finite")
    return values


def paired_bootstrap_interval(
    deltas: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile interval for the mean of already paired differences.

    Each resample draws complete pair differences, preserving the candidate /
    baseline pairing.  A local ``Random`` instance makes results independent
    of global RNG state and exactly reproducible for a fixed seed.
    """
    values = _validate_deltas(deltas)
    _validate_confidence(confidence)
    _validate_resamples(resamples)
    generator = random.Random(seed)
    count = len(values)
    means = [
        sum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    ]
    tail = (1.0 - confidence) / 2.0
    return _percentile(means, tail), _percentile(means, 1.0 - tail)


def paired_permutation_interval(
    deltas: Sequence[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Tuple[float, float]:
    """Sign-flip randomization interval for a paired mean difference.

    Under the paired null, each difference's sign is exchangeable.  We draw
    that null distribution and invert its central interval about the observed
    mean.  This produces an interval for the observed paired effect (rather
    than the misleading null interval centred at zero).
    """
    values = _validate_deltas(deltas)
    _validate_confidence(confidence)
    _validate_resamples(resamples)
    generator = random.Random(seed)
    count = len(values)
    observed_mean = statistics.fmean(values)
    null_means = [
        sum(value if generator.getrandbits(1) else -value for value in values) / count
        for _ in range(resamples)
    ]
    tail = (1.0 - confidence) / 2.0
    # Invert the sign-flip null distribution.  For symmetric null samples this
    # is equivalent to observed_mean +/- the central randomization radius.
    return (
        observed_mean - _percentile(null_means, 1.0 - tail),
        observed_mean - _percentile(null_means, tail),
    )


@dataclass(frozen=True)
class PairedComparison:
    """A report-ready paired comparison of a lower-is-better error metric."""

    primary_metric: str
    episode_ids: Tuple[Hashable, ...]
    candidate_errors: Tuple[float, ...]
    baseline_errors: Tuple[float, ...]
    deltas: Tuple[float, ...]
    mean_delta: Optional[float]
    win_rate: Optional[float]
    bootstrap_interval: Optional[Tuple[float, float]]
    permutation_interval: Optional[Tuple[float, float]]
    confidence: float
    status: str
    reason: Optional[str] = None

    @property
    def episode_count(self) -> int:
        return len(self.episode_ids)

    @property
    def not_evaluable(self) -> bool:
        return self.status == "not_evaluable"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready report, retaining raw errors beside ratios."""
        raw_candidate = dict(zip(self.episode_ids, self.candidate_errors))
        raw_baseline = dict(zip(self.episode_ids, self.baseline_errors))
        ratios = [
            candidate / baseline if baseline != 0.0 else None
            for candidate, baseline in zip(self.candidate_errors, self.baseline_errors)
        ]
        return {
            "format": COMPARISON_FORMAT,
            "primary_metric": self.primary_metric,
            "status": self.status,
            "reason": self.reason,
            "episode_count": self.episode_count,
            "episode_ids": list(self.episode_ids),
            "mean_delta": self.mean_delta,
            "win_rate": self.win_rate,
            "confidence": self.confidence,
            "intervals": {
                "paired_bootstrap": _interval_dict(self.bootstrap_interval),
                "paired_permutation": _interval_dict(self.permutation_interval),
            },
            "per_episode": {
                "candidate_raw_errors": raw_candidate,
                "baseline_raw_errors": raw_baseline,
                "deltas": list(self.deltas),
                "candidate_over_baseline_ratio": ratios,
            },
        }


def _interval_dict(interval: Optional[Tuple[float, float]]) -> Optional[Dict[str, float]]:
    if interval is None:
        return None
    return {"low": interval[0], "high": interval[1]}


EpisodeErrors = Union[Mapping[Hashable, float], Sequence[float]]


def _episode_mapping(
    errors: EpisodeErrors,
    episode_ids: Optional[Sequence[Hashable]],
    *,
    side: str,
) -> Mapping[Hashable, float]:
    """Give the evaluator's list output explicit frozen-corpus identities.

    ``evaluate_action_world_model_milestone`` exposes
    ``per_episode_model_mse`` as a list per horizon.  A caller may pass that
    list directly with its frozen corpus's episode IDs; mappings already carry
    those IDs and need no separate argument.  Unlabelled lists are rejected so
    matching positions can never be mistaken for a verified pairing.
    """
    if isinstance(errors, Mapping):
        if episode_ids is not None:
            raise ValueError(f"{side}_episode_ids is redundant when {side} errors are a mapping")
        return errors
    if episode_ids is None:
        raise ValueError(f"{side} episode errors need explicit {side}_episode_ids for paired comparison")
    if len(errors) != len(episode_ids):
        raise ValueError(
            f"{side} errors has {len(errors)} values but {side}_episode_ids has {len(episode_ids)} ids"
        )
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError(f"{side}_episode_ids contains duplicate episode ids")
    return dict(zip(episode_ids, errors))


def _validated_pairs(
    candidate: Mapping[Hashable, float], baseline: Mapping[Hashable, float],
) -> Tuple[Tuple[Hashable, ...], Tuple[float, ...], Tuple[float, ...]]:
    candidate_ids = set(candidate)
    baseline_ids = set(baseline)
    if candidate_ids != baseline_ids:
        candidate_only = sorted(candidate_ids - baseline_ids, key=repr)
        baseline_only = sorted(baseline_ids - candidate_ids, key=repr)
        raise ValueError(
            "unpaired episode ids: "
            f"candidate_only={candidate_only!r}, baseline_only={baseline_only!r}"
        )
    # repr is stable for the intended str/int episode IDs and lets mixed, but
    # still valid, hashable ID types remain deterministic without requiring
    # them to be mutually orderable.
    episode_ids = tuple(sorted(candidate_ids, key=repr))
    candidate_values = tuple(float(candidate[episode_id]) for episode_id in episode_ids)
    baseline_values = tuple(float(baseline[episode_id]) for episode_id in episode_ids)
    if any(not math.isfinite(value) for value in candidate_values + baseline_values):
        raise ValueError("candidate and baseline errors must all be finite")
    return episode_ids, candidate_values, baseline_values


def compare_paired_episodes(
    candidate: EpisodeErrors,
    baseline: EpisodeErrors,
    *,
    candidate_episode_ids: Optional[Sequence[Hashable]] = None,
    baseline_episode_ids: Optional[Sequence[Hashable]] = None,
    primary_metric: str = "rollout_t4_mse",
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    minimum_episode_count: int = DEFAULT_MIN_EPISODE_COUNT,
) -> PairedComparison:
    """Compare matched per-episode errors (candidate minus baseline).

    Lower error wins, so a negative delta and ``candidate < baseline`` count
    as a win.  The raw errors are retained in the returned report even though
    a per-episode candidate/baseline ratio is also emitted.  For the existing
    evaluator's list-valued ``per_episode_model_mse`` reports, pass the frozen
    validation IDs in ``candidate_episode_ids`` and ``baseline_episode_ids``.
    """
    _validate_confidence(confidence)
    _validate_resamples(resamples)
    if minimum_episode_count <= 0:
        raise ValueError("minimum_episode_count must be positive")
    candidate_by_episode = _episode_mapping(
        candidate, candidate_episode_ids, side="candidate",
    )
    baseline_by_episode = _episode_mapping(
        baseline, baseline_episode_ids, side="baseline",
    )
    episode_ids, candidate_errors, baseline_errors = _validated_pairs(
        candidate_by_episode, baseline_by_episode,
    )
    deltas = tuple(
        candidate_error - baseline_error
        for candidate_error, baseline_error in zip(candidate_errors, baseline_errors)
    )
    if len(episode_ids) < minimum_episode_count:
        return PairedComparison(
            primary_metric=primary_metric,
            episode_ids=episode_ids,
            candidate_errors=candidate_errors,
            baseline_errors=baseline_errors,
            deltas=deltas,
            mean_delta=None,
            win_rate=None,
            bootstrap_interval=None,
            permutation_interval=None,
            confidence=confidence,
            status="not_evaluable",
            reason=(
                f"requires at least {minimum_episode_count} paired episodes; "
                f"received {len(episode_ids)}"
            ),
        )
    return PairedComparison(
        primary_metric=primary_metric,
        episode_ids=episode_ids,
        candidate_errors=candidate_errors,
        baseline_errors=baseline_errors,
        deltas=deltas,
        mean_delta=statistics.fmean(deltas),
        win_rate=sum(delta < 0.0 for delta in deltas) / len(deltas),
        bootstrap_interval=paired_bootstrap_interval(
            deltas, confidence=confidence, resamples=resamples, seed=seed,
        ),
        permutation_interval=paired_permutation_interval(
            deltas, confidence=confidence, resamples=resamples, seed=seed,
        ),
        confidence=confidence,
        status="evaluable",
    )


# Concise public alias for callers that do not need to spell out "episodes".
compare_paired = compare_paired_episodes
