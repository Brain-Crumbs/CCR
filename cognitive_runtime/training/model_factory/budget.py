"""Graceful training-time budget enforcement (epic #212, Phase B step 5, §15).

A trial runner must not abruptly kill a training process merely because a
single early epoch had a pessimistic ETA -- CUDA warm-up and data-loader
priming make the first epoch(s) unrepresentative of steady-state cost. This
module tracks per-epoch training durations, projects remaining time from a
**median** of the representative (post-warm-up) durations, and distinguishes
two enforcement points:

- a **graceful** stop request, honored at the next checkpoint boundary
  (:func:`cognitive_runtime.training.model_factory.checkpoint.checkpoint_due`,
  MF-B3); and
- the **hard** deadline, which is always enforced at the current epoch
  boundary regardless of cadence.

Either path ends a trial with completion status ``budget_exceeded``, which is
recorded distinctly from a watchdog's ``timeout_unrecoverable`` (a stalled
worker killed after a grace period -- a safety fallback, not budget
enforcement) and is never eligible for champion promotion (see
:func:`cognitive_runtime.training.statistical_evaluation.build_experiment_report`).

Pure Python -- no torch import -- so this module is usable in the core-only
install, before a checkpoint ever touches a GPU.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from cognitive_runtime.observability.trace import _device_info
from cognitive_runtime.training.model_factory.artifacts import atomic_write_json
from cognitive_runtime.training.model_factory.checkpoint import checkpoint_due

#: Versioned identity for the ETA/deadline policy implemented here (epic
#: §15.2). Bump this whenever the estimation rule (warm-up handling, median
#: window, boundary semantics) changes, so an older resolved spec's timing
#: fields are never misread under a newer policy.
TIMING_POLICY_VERSION = "model-factory-budget-timing-v1"

#: epic #212 §15.1 -- the factory's initial runtime budget portfolio.
BUDGET_TIERS: Dict[str, float] = {"fast": 600.0, "scale": 1200.0}

STATUS_COMPLETED = "completed"
STATUS_BUDGET_EXCEEDED = "budget_exceeded"
STATUS_TIMEOUT_UNRECOVERABLE = "timeout_unrecoverable"
STATUS_CANCELLED = "cancelled"

#: The full §16 completion vocabulary a resolved spec may record.
COMPLETION_STATUSES: Tuple[str, ...] = (
    STATUS_COMPLETED,
    STATUS_BUDGET_EXCEEDED,
    STATUS_TIMEOUT_UNRECOVERABLE,
    "failed",
    STATUS_CANCELLED,
)

BUDGET_REPORT_FORMAT = "model-factory-budget-report-v1"


def budget_seconds_for_tier(tier: str) -> float:
    """Resolve a declared §15.1 tier name to its ``max_training_seconds`` cap."""
    try:
        return BUDGET_TIERS[tier]
    except KeyError:
        raise ValueError(
            f"unknown runtime budget tier {tier!r}; expected one of {sorted(BUDGET_TIERS)}"
        ) from None


@dataclass(frozen=True)
class BudgetDecision:
    """The monitor's verdict immediately after one recorded epoch."""

    epoch: int
    epochs_recorded: int
    elapsed_seconds: float
    median_epoch_seconds: Optional[float]
    #: Full-run completion estimate (elapsed + median * remaining epochs);
    #: ``None`` unless both a median and ``TrainingBudget.total_epochs`` are
    #: known -- never a misleading one-epoch-ahead number.
    projected_seconds: Optional[float]
    stop_requested: bool
    hard_deadline_exceeded: bool
    at_checkpoint_boundary: bool

    @property
    def should_stop_now(self) -> bool:
        """Stop this trial now.

        Either the hard deadline has already elapsed (enforced immediately,
        at whatever epoch boundary just completed), or a graceful stop was
        requested by the ETA projection and training has reached a safe
        checkpoint boundary to act on it.
        """
        return self.hard_deadline_exceeded or (self.stop_requested and self.at_checkpoint_boundary)


@dataclass
class TrainingBudget:
    """Tracks per-epoch training time against a declared deadline.

    ``warmup_epochs`` completed epochs are recorded (so ``elapsed_seconds``
    stays exact) but excluded from the ETA estimate: CUDA warm-up and data
    preparation make early timing unrepresentative of steady-state cost
    (epic §15.2). The ETA itself uses the **median**, not the mean or the
    first sample, of the remaining recorded durations, so one further
    slow/fast outlier epoch cannot swing the projection.

    ``total_epochs``, when declared (``TrainingContract.epoch_budget``),
    lets the projection cover *every* remaining epoch rather than just the
    next one: ``elapsed + median * (total_epochs - epoch)``. Without it,
    there is no honest way to project full-run completion, so no graceful
    stop is ever requested -- only the hard deadline below still protects
    the budget.

    Call :meth:`record_epoch` with the *absolute* epoch number (not a count
    relative to this monitor instance) so the checkpoint-boundary check
    stays correct across a resumed run that starts measuring partway through
    training.
    """

    max_training_seconds: float
    total_epochs: Optional[int] = None
    warmup_epochs: int = 1
    checkpoint_cadence_epochs: Optional[int] = None
    _durations: List[float] = field(default_factory=list, repr=False)
    _stop_requested: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_training_seconds <= 0:
            raise ValueError("max_training_seconds must be positive")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must not be negative")
        if self.total_epochs is not None and self.total_epochs <= 0:
            raise ValueError("total_epochs must be positive when declared")

    @property
    def elapsed_seconds(self) -> float:
        return sum(self._durations)

    @property
    def epochs_recorded(self) -> int:
        return len(self._durations)

    def _representative_durations(self) -> Sequence[float]:
        """Completed-epoch durations robust to an unrepresentative warm-up."""
        if len(self._durations) <= self.warmup_epochs:
            return ()
        return self._durations[self.warmup_epochs:]

    def median_epoch_seconds(self) -> Optional[float]:
        """Median duration of the representative (post-warm-up) epochs.

        ``None`` while too few representative epochs have completed to
        estimate anything -- never falls back to the (possibly pessimistic)
        warm-up epoch's own duration.
        """
        representative = self._representative_durations()
        if not representative:
            return None
        return statistics.median(representative)

    def _projected_total_seconds(self, epoch: int, elapsed: float, median: Optional[float]) -> Optional[float]:
        """Full-run completion projection: elapsed plus every remaining epoch
        at the median rate -- not just the next one, or a real overrun many
        epochs out would only surface once ``elapsed`` alone had already
        blown the deadline (defeating the point of a *graceful* stop)."""
        if median is None or self.total_epochs is None:
            return None
        remaining_epochs = max(self.total_epochs - epoch, 0)
        return elapsed + median * remaining_epochs

    def record_epoch(self, epoch: int, duration_seconds: float) -> BudgetDecision:
        """Record one completed epoch's training-only duration.

        Returns a :class:`BudgetDecision` describing whether the caller
        should stop -- gracefully at the next checkpoint boundary, or
        immediately if the hard deadline has already passed.
        """
        if duration_seconds < 0:
            raise ValueError("duration_seconds must not be negative")
        self._durations.append(duration_seconds)
        elapsed = self.elapsed_seconds
        median = self.median_epoch_seconds()
        projected = self._projected_total_seconds(epoch, elapsed, median)

        hard_exceeded = elapsed >= self.max_training_seconds
        if projected is not None and projected > self.max_training_seconds:
            # Sticky: once a graceful stop is requested it remains requested
            # until acted on at the next checkpoint boundary, rather than
            # flapping if a later epoch's median briefly dips back down.
            self._stop_requested = True
        at_boundary = checkpoint_due(epoch, self.checkpoint_cadence_epochs)

        return BudgetDecision(
            epoch=epoch,
            epochs_recorded=self.epochs_recorded,
            elapsed_seconds=elapsed,
            median_epoch_seconds=median,
            projected_seconds=projected,
            stop_requested=self._stop_requested,
            hard_deadline_exceeded=hard_exceeded,
            at_checkpoint_boundary=at_boundary,
        )


class EpochTimer:
    """Measures one epoch's training-only wall-clock duration.

    Start it only after the frozen corpus has been resolved, and exit it
    before validation begins (epic §15's budget-clock boundary) -- run
    validation, if any, outside this context so its time never enters
    ``measured_training_seconds``.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._start: Optional[float] = None
        self.duration_seconds: Optional[float] = None

    def __enter__(self) -> "EpochTimer":
        self._start = self._clock()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._start is not None
        self.duration_seconds = self._clock() - self._start
        return None


def watchdog_expired(seconds_since_heartbeat: float, grace_period_seconds: float) -> bool:
    """True once a stalled worker's heartbeat gap exceeds the watchdog's
    grace period.

    This is the ``timeout_unrecoverable`` safety fallback (epic §15.2): a
    hung worker killed here is recorded distinctly from :class:`TrainingBudget`'s
    graceful/hard ``budget_exceeded`` enforcement above, which always stops a
    live, responsive process at a safe boundary instead.
    """
    if grace_period_seconds <= 0:
        raise ValueError("grace_period_seconds must be positive")
    if seconds_since_heartbeat < 0:
        raise ValueError("seconds_since_heartbeat must not be negative")
    return seconds_since_heartbeat >= grace_period_seconds


def build_budget_report(
    budget: TrainingBudget,
    *,
    completion_status: str,
    total_trial_seconds: float,
    precision: str,
    tier: Optional[str] = None,
    hardware_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the §15.2 timing/completion record for a trial's resolved spec.

    Records the budget, timing-policy version, measured training seconds,
    total trial seconds, hardware/precision profile, and completion status --
    the exact fields §15.2 requires so a claimed tier eligibility (for
    example "completed within 10 minutes") stays interpretable later.
    """
    if completion_status not in COMPLETION_STATUSES:
        raise ValueError(
            f"unknown completion status {completion_status!r}; expected one of {COMPLETION_STATUSES}"
        )
    if total_trial_seconds < budget.elapsed_seconds:
        raise ValueError(
            "total_trial_seconds must be at least measured_training_seconds "
            f"({total_trial_seconds!r} < {budget.elapsed_seconds!r})"
        )
    if completion_status == STATUS_COMPLETED and budget.elapsed_seconds > budget.max_training_seconds:
        # A report cannot claim a clean tier completion while also reporting
        # measured training time past the declared cap -- that contradiction
        # is exactly the "claimed 10-minute eligibility" §15.1 promises stays
        # interpretable. The caller should report budget_exceeded instead.
        raise ValueError(
            "completion_status='completed' but measured_training_seconds "
            f"({budget.elapsed_seconds!r}) exceeds max_training_seconds "
            f"({budget.max_training_seconds!r}); report completion_status="
            f"{STATUS_BUDGET_EXCEEDED!r} for a trial that ran past its cap"
        )
    profile = dict(hardware_profile) if hardware_profile is not None else _device_info()
    return {
        "format": BUDGET_REPORT_FORMAT,
        "tier": tier,
        "max_training_seconds": budget.max_training_seconds,
        "timing_policy_version": TIMING_POLICY_VERSION,
        "measured_training_seconds": budget.elapsed_seconds,
        "total_trial_seconds": float(total_trial_seconds),
        "hardware_profile": profile,
        "precision": precision,
        "completion_status": completion_status,
        "epochs_recorded": budget.epochs_recorded,
        "warmup_epochs": budget.warmup_epochs,
        "total_epochs": budget.total_epochs,
    }


def write_budget_report(path: Union[str, Path], report: Mapping[str, Any]) -> Path:
    """Persist a budget report atomically, validating its format first."""
    if report.get("format") != BUDGET_REPORT_FORMAT:
        raise ValueError(f"expected {BUDGET_REPORT_FORMAT}, got {report.get('format')!r}")
    return atomic_write_json(path, report)


__all__ = [
    "TIMING_POLICY_VERSION",
    "BUDGET_TIERS",
    "STATUS_COMPLETED",
    "STATUS_BUDGET_EXCEEDED",
    "STATUS_TIMEOUT_UNRECOVERABLE",
    "STATUS_CANCELLED",
    "COMPLETION_STATUSES",
    "BUDGET_REPORT_FORMAT",
    "budget_seconds_for_tier",
    "BudgetDecision",
    "TrainingBudget",
    "EpochTimer",
    "watchdog_expired",
    "build_budget_report",
    "write_budget_report",
]
