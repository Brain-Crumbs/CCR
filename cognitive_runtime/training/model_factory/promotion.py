"""Promotion rule engine, gate wiring, and sealed-test-use budget (epic #212
Phase C step 4, §10.3, issue #226).

A candidate is promoted only when **all** of the following independently
testable, independently blocking gates pass:

1. ``data_quality`` and ``split_overlap`` -- the corpus's own quality and
   disjointness gates (:mod:`cognitive_runtime.training.model_factory.corpus`).
2. ``representation`` and ``rollout_health`` -- the yaw/variance collapse
   probe (:func:`cognitive_runtime.training.action_world_model.representation_collapse_diagnostics`)
   and the rollout dispersion health check.
3. ``primary_metric_margin`` -- the primary validation metric improves over
   the champion by a configured **minimum practical margin** *and* passes
   the paired-comparison rule (MF-C1,
   :mod:`cognitive_runtime.training.model_factory.comparison`). A candidate
   whose improvement is statistically significant but smaller than the
   margin is held, not promoted.
4. ``safety_metrics`` -- no configured safety metric regresses beyond its
   own allowed margin.
5. ``generic_retention`` -- a ``goal_navigation_v1`` fine-tune supplies the
   existing CI-refereed forgetting report for ``generic_action_effects_v1``
   and stays within its DataContract's declared regression allowance.
6. ``training_time_budget`` -- the run completed within its stage's
   declared training-time budget (never ``budget_exceeded``, ``failed``,
   ``cancelled``, or ``timeout_unrecoverable``).
7. ``metric_schema`` -- raw and stratified metrics satisfy their versioned
   schema: the copy-last ratio (``model_over_copy_last_mse``) and the
   event-stratified rate (``cow_false_positive_rate``) are never present
   without the raw errors/counts they are computed from.
8. ``durable_test_confirmation`` -- only checked when the caller declares
   ``durable=True``: a final sealed-test action (MF-C5, issue #227) must
   have confirmed the candidate before it may become a *durable* champion.

Two rules are enforced mechanically rather than by convention:

* **Training loss is never a promotion metric.** ``evaluate_promotion``
  never reads a training-loss field for any gate; only the held-out
  ``comparison`` (paired validation-metric deltas) and the persisted
  ``experiment_report`` drive the decision. A candidate with a better
  training loss and a worse validation metric is rejected exactly like any
  other regression (see ``tests/test_model_factory_promotion.py``).
* **Test-use accounting.** :func:`perform_sealed_test_action` is the only
  sanctioned way to charge a sealed-test-set action against a candidate's
  ledger. It refuses once :class:`PromotionPolicy`'s
  ``max_sealed_test_uses`` is already spent, and every accepted use is
  recorded in the champion registry
  (:func:`cognitive_runtime.training.model_factory.registry.record_test_use`)
  -- routine search and population updates never call it.

This module extends ``build_experiment_report``'s existing
``promotion_verdict`` shape (``cognitive_runtime.training.statistical_evaluation``:
already refuses promotion without checkpoint identity, and already refuses
``training_replay`` evaluation) rather than inventing a parallel one: gates
1-2's checkpoint/evaluation-mode/rollout/event-stratified checks re-derive
the same predicates from the same report fields, just itemized into
independently named :class:`GateResult`\\ s instead of one flat reasons
list. The result is always a structured :class:`PromotionVerdict` naming
every gate and why it passed or failed -- never a bare boolean.

Pure Python -- no torch import -- so this module is usable in the
core-only install, before a checkpoint ever touches a GPU. Every input is
a plain JSON-safe mapping (or a small frozen dataclass); callers already
holding richer objects (a :class:`~.comparison.PairedComparison`, an
``ActionWorldModelConfig``-produced diagnostics dict) pass their
``.to_dict()``/plain-mapping form.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from cognitive_runtime.training.model_factory.registry import (
    DECISION_HOLD,
    DECISION_PROMOTE,
    TestBudgetExhaustedError,
    GOAL_NAVIGATION_V1,
    record_test_use,
)

PROMOTION_VERDICT_FORMAT = "model-factory-promotion-verdict-v1"

#: Versioned identity for gate 6's raw-errors-alongside-ratio invariant.
#: Bump only if the schema this gate checks (which fields must accompany a
#: derived ratio/rate) actually changes.
METRIC_SCHEMA_VERSION = "model-factory-promotion-metric-schema-v1"


class PromotionPolicyError(ValueError):
    """An invalid :class:`PromotionPolicy` declaration."""


class SealedTestBudgetExhaustedError(ValueError):
    """Raised when a sealed-test action is requested after the policy's
    ``max_sealed_test_uses`` budget for ``run_id`` is already spent."""

    def __init__(self, run_id: str, max_sealed_test_uses: int, current_uses: int) -> None:
        self.run_id = run_id
        self.max_sealed_test_uses = max_sealed_test_uses
        self.current_uses = current_uses
        super().__init__(
            f"sealed-test-use budget exhausted for run {run_id!r}: "
            f"{current_uses} of {max_sealed_test_uses} action(s) already recorded; "
            "refusing this test action"
        )


@dataclass(frozen=True)
class PromotionPolicy:
    """The stage-declared thresholds :func:`evaluate_promotion` gates against.

    ``minimum_practical_margin`` is on the same scale as the paired
    comparison's ``mean_delta`` (candidate minus baseline; lower is better
    per :mod:`.comparison`) -- a required improvement magnitude, never
    negative. ``max_sealed_test_uses`` is the small explicit budget epic
    §10.3 requires: how many sealed-test actions one candidate may consume
    before :func:`perform_sealed_test_action` refuses further ones.
    """

    minimum_practical_margin: float
    max_sealed_test_uses: int = 3

    def __post_init__(self) -> None:
        if self.minimum_practical_margin < 0:
            raise PromotionPolicyError("minimum_practical_margin must not be negative")
        if self.max_sealed_test_uses <= 0:
            raise PromotionPolicyError("max_sealed_test_uses must be positive")


@dataclass(frozen=True)
class SafetyMetricObservation:
    """One configured safety metric's champion/candidate values and its own
    allowed regression margin (gate 4).

    ``regression`` is positive when the candidate is worse than the
    champion, whatever the metric's direction; ``max_regression`` is always
    a non-negative magnitude on that same scale.
    """

    name: str
    baseline_value: float
    candidate_value: float
    higher_is_better: bool
    max_regression: float

    def __post_init__(self) -> None:
        if self.max_regression < 0:
            raise PromotionPolicyError(f"safety metric {self.name!r}: max_regression must not be negative")

    @property
    def regression(self) -> float:
        if self.higher_is_better:
            return self.baseline_value - self.candidate_value
        return self.candidate_value - self.baseline_value

    @property
    def regressed_beyond_margin(self) -> bool:
        return self.regression > self.max_regression


@dataclass(frozen=True)
class TestConfirmation:
    """The outcome of one :func:`perform_sealed_test_action` call -- the
    evidence gate 7 requires before a candidate may become a *durable*
    champion (MF-C5, issue #227)."""

    #: Not a pytest test case -- silences pytest's name-based collection
    #: heuristic (the class name starts with "Test").
    __test__ = False

    run_id: str
    performed: bool
    passed: bool
    test_uses_after: int
    max_sealed_test_uses: int
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class GateResult:
    """One named gate's verdict: whether it passed, and why.

    ``applicable=False`` marks a gate that was skipped for a structural
    reason (no champion yet to compare against; durable promotion not
    requested) rather than evaluated and passed -- it never blocks
    promotion, but stays visible in the verdict so "why wasn't this gate
    checked" is answerable from the report itself.
    """

    name: str
    passed: bool
    reason: str
    applicable: bool = True

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "applicable": self.applicable, "reason": self.reason}


@dataclass(frozen=True)
class PromotionVerdict:
    """Every gate's pass/fail and reason, plus the overall decision --
    never a bare boolean (epic #212 §10.3)."""

    decision: str
    gates: Tuple[GateResult, ...]
    reasons: Tuple[str, ...]

    @property
    def promoted(self) -> bool:
        return self.decision == DECISION_PROMOTE

    def to_dict(self) -> dict:
        return {
            "format": PROMOTION_VERDICT_FORMAT,
            "decision": self.decision,
            "promoted": self.promoted,
            "gates": [gate.to_dict() for gate in self.gates],
            "reasons": list(self.reasons) or ["all configured gates passed"],
        }


# --------------------------------------------------------------------- gates


def _checkpoint_identity_gate(checkpoint: Mapping[str, Any]) -> GateResult:
    ok = bool(checkpoint.get("path")) and bool(checkpoint.get("sha256"))
    reason = "checkpoint has a path and sha256" if ok else "checkpoint identity requires path and sha256"
    return GateResult("checkpoint_identity", ok, reason)


def _evaluation_mode_gate(training_stats: Mapping[str, Any]) -> GateResult:
    mode = training_stats.get("evaluation_mode")
    ok = mode != "training_replay"
    reason = (
        "evaluation mode supports promotion" if ok
        else "training-replay evaluation cannot support promotion"
    )
    return GateResult("evaluation_mode", ok, reason)


def _training_time_budget_gate(training_stats: Mapping[str, Any]) -> GateResult:
    status = training_stats.get("completion_status")
    ok = status in (None, "completed")
    reason = (
        "the run completed within its declared training-time budget" if ok
        else f"{status} trial cannot be promoted regardless of its metrics"
    )
    return GateResult("training_time_budget", ok, reason)


def _rollout_beats_copy_last_gate(rollout_metrics: Mapping[str, Any]) -> GateResult:
    horizons = rollout_metrics.get("horizons") or {}
    failing = sorted(
        str(horizon) for horizon, values in horizons.items() if not (values or {}).get("beats_copy_last", False)
    )
    ok = not failing
    reason = (
        "model beats copy-last at every evaluated horizon" if ok
        else f"rollout does not beat copy-last at horizon(s) {failing!r}"
    )
    return GateResult("rollout_beats_copy_last", ok, reason)


def _rollout_health_gate(rollout_metrics: Mapping[str, Any]) -> GateResult:
    health = rollout_metrics.get("rollout_health") or {}
    state = health.get("state")
    ok = state in (None, "healthy", "not_evaluable")
    reason = "rollout health is healthy" if ok else f"rollout health is {state}"
    return GateResult("rollout_health", ok, reason)


def _event_stratified_evaluability_gate(event_stratified_metrics: Mapping[str, Any]) -> GateResult:
    failing = sorted(
        str(horizon) for horizon, values in (event_stratified_metrics or {}).items()
        if ((values or {}).get("entity") or {}).get("cow_false_positive_rate") is None
    )
    ok = not failing
    reason = (
        "every stratified horizon has evaluable cow-absent samples" if ok
        else f"horizon(s) {failing!r} have no evaluable cow-absent samples"
    )
    return GateResult("event_stratified_evaluability", ok, reason)


def _data_quality_gate(data_quality: Mapping[str, Any]) -> GateResult:
    ok = bool(data_quality.get("passed"))
    issues = list(data_quality.get("issues") or [])
    reason = (
        "data-quality gate passed" if ok
        else f"data-quality gate failed: {'; '.join(issues) if issues else 'not passed'}"
    )
    return GateResult("data_quality", ok, reason)


def _split_overlap_gate(split_overlap: Mapping[str, Any]) -> GateResult:
    ok = bool(split_overlap.get("disjoint"))
    issues = list(split_overlap.get("issues") or [])
    reason = (
        "split-overlap gate passed (train/validation/test disjoint)" if ok
        else f"split-overlap gate failed: {'; '.join(issues) if issues else 'not disjoint'}"
    )
    return GateResult("split_overlap", ok, reason)


def _representation_gate(representation: Optional[Mapping[str, Any]]) -> GateResult:
    if representation is None:
        return GateResult(
            "representation", True,
            "representation-collapse gate not configured for this stage", applicable=False,
        )
    ok = bool(representation.get("passed"))
    reason = (
        "representation-collapse gate passed" if ok
        else (
            "representation collapse gate failed: hidden orientation R^2="
            f"{representation.get('hidden_orientation_r2', representation.get('hidden_yaw_r2'))}, "
            f"latent_collapsed={representation.get('latent_collapsed')}"
        )
    )
    return GateResult("representation", ok, reason)


def _metric_schema_gate(
    rollout_metrics: Mapping[str, Any], event_stratified_metrics: Mapping[str, Any],
) -> GateResult:
    """Gate 6: a derived ratio/rate is never reported without its raw errors."""
    problems = []
    for horizon, values in (rollout_metrics.get("horizons") or {}).items():
        values = values or {}
        ratio = values.get("model_over_copy_last_mse")
        if ratio is not None and (values.get("model_mse") is None or values.get("copy_last_mse") is None):
            problems.append(
                f"rollout horizon {horizon!r} carries model_over_copy_last_mse without its raw "
                "model_mse/copy_last_mse"
            )
    for horizon, values in (event_stratified_metrics or {}).items():
        entity = ((values or {}).get("entity")) or {}
        rate = entity.get("cow_false_positive_rate")
        if rate is not None and (entity.get("false_positive") is None or entity.get("true_negative") is None):
            problems.append(
                f"event-stratified horizon {horizon!r} carries cow_false_positive_rate without its "
                "raw false_positive/true_negative counts"
            )
    ok = not problems
    reason = f"versioned metric schema {METRIC_SCHEMA_VERSION} satisfied" if ok else "; ".join(problems)
    return GateResult("metric_schema", ok, reason)


def _comparison_mapping(comparison: Optional[Union[Mapping[str, Any], Any]]) -> Optional[Mapping[str, Any]]:
    if comparison is None:
        return None
    if isinstance(comparison, Mapping):
        return comparison
    to_dict = getattr(comparison, "to_dict", None)
    if to_dict is None:
        raise PromotionPolicyError(
            "comparison must be a mapping or an object with a to_dict() method "
            f"(e.g. comparison.PairedComparison); got {type(comparison).__name__}"
        )
    return to_dict()


def _primary_metric_margin_gate(
    comparison: Optional[Union[Mapping[str, Any], Any]],
    *,
    has_champion: bool,
    minimum_practical_margin: float,
) -> GateResult:
    """Gate 3: significant paired improvement (MF-C1) *and* at least the
    configured minimum practical margin -- a significant-but-small
    improvement is held, not promoted."""
    if not has_champion:
        return GateResult(
            "primary_metric_margin", True,
            "no existing champion for this slot; first candidate is accepted without a paired comparison",
            applicable=False,
        )
    payload = _comparison_mapping(comparison)
    if payload is None:
        return GateResult(
            "primary_metric_margin", False,
            "a champion exists for this slot but no paired comparison against it was supplied",
        )
    if payload.get("status") != "evaluable":
        return GateResult(
            "primary_metric_margin", False,
            f"paired comparison against the champion is not evaluable: {payload.get('reason')}",
        )
    intervals = payload.get("intervals") or {}
    bootstrap = intervals.get("paired_bootstrap")
    permutation = intervals.get("paired_permutation")
    if not bootstrap or not permutation:
        return GateResult(
            "primary_metric_margin", False,
            "paired comparison is missing its bootstrap/permutation intervals",
        )
    mean_delta = payload.get("mean_delta")
    significant_improvement = bootstrap["high"] < 0 and permutation["high"] < 0
    if not significant_improvement:
        significant_regression = bootstrap["low"] > 0 and permutation["low"] > 0
        if significant_regression:
            return GateResult(
                "primary_metric_margin", False,
                f"candidate regressed on the primary metric relative to the champion (mean_delta={mean_delta})",
            )
        return GateResult(
            "primary_metric_margin", False,
            "improvement over the champion is not statistically significant under the paired-comparison "
            "rule (MF-C1)",
        )
    improvement = -mean_delta if mean_delta is not None else None
    if improvement is None or improvement < minimum_practical_margin:
        return GateResult(
            "primary_metric_margin", False,
            f"primary metric improvement ({improvement}) is inside the minimum practical margin "
            f"({minimum_practical_margin}); held, not promoted",
        )
    return GateResult(
        "primary_metric_margin", True,
        f"primary metric improved by {improvement} (>= minimum practical margin {minimum_practical_margin}), "
        "significant under the paired-comparison rule (MF-C1)",
    )


def _safety_metrics_gate(safety_metrics: Sequence[SafetyMetricObservation]) -> GateResult:
    regressed = [observation for observation in safety_metrics if observation.regressed_beyond_margin]
    ok = not regressed
    if ok:
        reason = (
            "no configured safety metric regressed beyond its allowed margin" if safety_metrics
            else "no safety metrics were configured for this stage"
        )
    else:
        names = ", ".join(sorted(observation.name for observation in regressed))
        details = "; ".join(
            f"{observation.name} regressed by {observation.regression} "
            f"(allowed margin {observation.max_regression})"
            for observation in regressed
        )
        reason = f"safety metric(s) regressed beyond their allowed margin: {names} ({details})"
    return GateResult("safety_metrics", ok, reason)


def _generic_retention_gate(
    *,
    target_family: Optional[str],
    retention_report: Optional[Union[Mapping[str, Any], Any]],
    retention_policy: Optional[Mapping[str, Any]],
) -> GateResult:
    """Fail closed for navigation fine-tunes that forget generic dynamics.

    The evidence is ``sleep.forgetting.compute_forgetting_metric``'s report,
    not a second promotion-specific forgetting calculation.  We only verify
    that its tolerance matches the immutable DataContract allowance and that
    the report retained the declared generic suite.
    """
    if target_family != GOAL_NAVIGATION_V1.name:
        return GateResult(
            "generic_retention", True,
            "generic retention is required only for goal_navigation_v1 promotion",
            applicable=False,
        )
    policy = dict(retention_policy or {})
    suite = policy.get("suite")
    allowance = policy.get("max_regression")
    replay_mixture = policy.get("replay_mixture")
    if suite != "generic_action_effects_v1" or allowance is None or not replay_mixture:
        return GateResult(
            "generic_retention", False,
            "goal_navigation_v1 requires a DataContract retention policy naming "
            "generic_action_effects_v1, its replay mixture, and max_regression",
        )
    if retention_report is None:
        return GateResult(
            "generic_retention", False,
            "goal_navigation_v1 promotion requires generic retention-suite evidence from "
            "sleep.forgetting.compute_forgetting_metric",
        )
    if isinstance(retention_report, Mapping):
        report = retention_report
    else:
        to_dict = getattr(retention_report, "to_dict", None)
        if not callable(to_dict):
            raise PromotionPolicyError(
                "retention_report must be a mapping or ForgettingReport-like object with to_dict()"
            )
        report = to_dict()
    if report.get("old_scenario") != suite:
        return GateResult(
            "generic_retention", False,
            f"retention evidence covers {report.get('old_scenario')!r}, not declared suite {suite!r}",
        )
    tolerance = report.get("tolerance")
    if tolerance is None or abs(float(tolerance) - float(allowance)) > 1e-12:
        return GateResult(
            "generic_retention", False,
            f"forgetting report tolerance {tolerance!r} does not match the DataContract "
            f"max_regression {allowance!r}",
        )
    forgetting_amount = report.get("forgetting_amount")
    if forgetting_amount is None or float(forgetting_amount) > float(allowance) or not bool(
        report.get("retained")
    ):
        return GateResult(
            "generic_retention", False,
            f"generic retention regressed by {forgetting_amount} beyond the "
            f"declared allowance {allowance}",
        )
    return GateResult(
        "generic_retention", True,
        f"generic_action_effects_v1 retained within the declared regression allowance {allowance}",
    )


def _durable_test_confirmation_gate(
    durable: bool, test_confirmation: Optional[TestConfirmation], candidate_run_id: str,
) -> GateResult:
    if not durable:
        return GateResult(
            "durable_test_confirmation", True,
            "durable promotion not requested for this candidate", applicable=False,
        )
    if test_confirmation is None:
        return GateResult(
            "durable_test_confirmation", False,
            "a durable champion requires a confirmed final sealed-test action (MF-C5) which was not supplied",
        )
    if test_confirmation.run_id != candidate_run_id:
        return GateResult(
            "durable_test_confirmation", False,
            f"final sealed-test action confirmed run_id={test_confirmation.run_id!r}, not the candidate "
            f"being evaluated (run_id={candidate_run_id!r}); a confirmation may not be reused across candidates",
        )
    if not (test_confirmation.performed and test_confirmation.passed):
        return GateResult(
            "durable_test_confirmation", False,
            f"final sealed-test action did not confirm the candidate: {test_confirmation.reason}",
        )
    return GateResult(
        "durable_test_confirmation", True,
        f"final sealed-test action confirmed the candidate (test_uses_after={test_confirmation.test_uses_after})",
    )


# ---------------------------------------------------------------- the engine


def evaluate_promotion(
    *,
    policy: PromotionPolicy,
    candidate_run_id: str,
    experiment_report: Mapping[str, Any],
    data_quality: Mapping[str, Any],
    split_overlap: Mapping[str, Any],
    has_champion: bool,
    representation: Optional[Mapping[str, Any]] = None,
    comparison: Optional[Union[Mapping[str, Any], Any]] = None,
    safety_metrics: Sequence[SafetyMetricObservation] = (),
    durable: bool = False,
    test_confirmation: Optional[TestConfirmation] = None,
    target_family: Optional[str] = None,
    retention_report: Optional[Union[Mapping[str, Any], Any]] = None,
    retention_policy: Optional[Mapping[str, Any]] = None,
) -> PromotionVerdict:
    """Gate one candidate against all configured conditions, returning a
    structured :class:`PromotionVerdict`.

    ``candidate_run_id`` is the run actually being evaluated. Gate 7 checks
    it against ``test_confirmation.run_id``: a sealed-test confirmation
    earned by one candidate must never be reusable to durably promote a
    different one.

    ``experiment_report`` is the artifact
    ``cognitive_runtime.training.statistical_evaluation.build_experiment_report``
    already produces (``checkpoint``, ``training_stats``,
    ``metrics.rollout``, ``event_stratified_metrics``); this function reuses
    those fields for gates 2, 5, and 6 rather than re-deriving them from a
    parallel source. ``data_quality``/``split_overlap`` are the corpus's own
    ``quality_report.json``/``split_overlap_report.json`` payloads
    (:mod:`.corpus`) and are required explicitly -- never defaulted to
    "passed" -- because a real trial's ``experiment_report`` does not yet
    carry them. ``representation`` is
    ``action_world_model.representation_collapse_diagnostics``'s report, or
    ``None`` when that gate is not configured for this stage.

    ``has_champion`` disambiguates "no champion exists yet for this slot"
    (gate 3 is trivially satisfied -- there is nothing to beat) from
    "a champion exists but no comparison was supplied" (gate 3 fails
    closed): a bare ``comparison=None`` is ambiguous between those two
    cases, so callers must say which one applies.

    Training loss is never read here: only ``comparison`` (a held-out
    paired validation-metric comparison) and ``experiment_report`` drive
    the decision, so a candidate cannot buy promotion with a better
    training loss no matter what ``training_stats`` also records.
    """
    checkpoint = experiment_report.get("checkpoint") or {}
    training_stats = experiment_report.get("training_stats") or {}
    rollout_metrics = (experiment_report.get("metrics") or {}).get("rollout") or {}
    event_stratified_metrics = experiment_report.get("event_stratified_metrics") or {}

    gates = (
        _checkpoint_identity_gate(checkpoint),
        _evaluation_mode_gate(training_stats),
        _training_time_budget_gate(training_stats),
        _data_quality_gate(data_quality),
        _split_overlap_gate(split_overlap),
        _representation_gate(representation),
        _rollout_beats_copy_last_gate(rollout_metrics),
        _rollout_health_gate(rollout_metrics),
        _event_stratified_evaluability_gate(event_stratified_metrics),
        _metric_schema_gate(rollout_metrics, event_stratified_metrics),
        _primary_metric_margin_gate(
            comparison, has_champion=has_champion, minimum_practical_margin=policy.minimum_practical_margin,
        ),
        _generic_retention_gate(
            target_family=target_family,
            retention_report=retention_report,
            retention_policy=retention_policy,
        ),
        _safety_metrics_gate(safety_metrics),
        _durable_test_confirmation_gate(durable, test_confirmation, candidate_run_id),
    )
    failing_reasons = tuple(gate.reason for gate in gates if not gate.passed)
    decision = DECISION_HOLD if failing_reasons else DECISION_PROMOTE
    return PromotionVerdict(decision=decision, gates=gates, reasons=failing_reasons)


# ------------------------------------------------------------ test-use budget


def perform_sealed_test_action(
    root: Union[str, Path],
    *,
    family: str,
    tier: str,
    objective: str,
    run_id: str,
    passed: bool,
    policy: PromotionPolicy,
    reason: Optional[str] = None,
) -> TestConfirmation:
    """Charge one sealed-test-set action against ``run_id``'s ledger.

    Refuses (:class:`SealedTestBudgetExhaustedError`) once ``policy``'s
    ``max_sealed_test_uses`` is already spent for this run. The cap check
    and the increment happen inside
    :func:`cognitive_runtime.training.model_factory.registry.record_test_use`'s
    single locked critical section (not a separate read here followed by a
    separate locked write), so two callers racing to charge the last
    remaining use can never both observe the pre-increment count and both
    succeed. ``run_id`` must already be a population member of the named
    slot: a test action confirms an already-evaluated candidate, never an
    unrelated identifier.

    This is the *only* sanctioned way to query the sealed test split in the
    Model Factory (epic §10.3): routine search and population updates must
    never call it, only an explicit final-test decision (MF-C5) may.
    """
    try:
        updated = record_test_use(
            root, family=family, tier=tier, objective=objective, run_id=run_id,
            reason=reason, max_uses=policy.max_sealed_test_uses,
        )
    except TestBudgetExhaustedError as exc:
        raise SealedTestBudgetExhaustedError(exc.run_id, exc.max_uses, exc.current_uses) from exc
    return TestConfirmation(
        run_id=run_id, performed=True, passed=passed, test_uses_after=updated.test_uses,
        max_sealed_test_uses=policy.max_sealed_test_uses, reason=reason,
    )


__all__ = [
    "PROMOTION_VERDICT_FORMAT",
    "METRIC_SCHEMA_VERSION",
    "PromotionPolicyError",
    "SealedTestBudgetExhaustedError",
    "PromotionPolicy",
    "SafetyMetricObservation",
    "TestConfirmation",
    "GateResult",
    "PromotionVerdict",
    "evaluate_promotion",
    "perform_sealed_test_action",
]
