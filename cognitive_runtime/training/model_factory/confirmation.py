"""Durable multi-training-seed confirmation and the sealed-test final action
(epic #212 Phase C step 5, §10.2, §10.3, issue #227).

Two execution paths live here to keep "won an A/B" separate from "is
actually better":

* :func:`confirm_across_seeds` reruns the **leading configuration** --
  one that already won its paired validation A/B against a baseline
  champion (a ``clone``/``fine_tune`` run whose ``metrics/comparison.json``
  records ``decision == "candidate_improves"``) -- on a small declared set
  of independent **training** seeds, each as its own fresh ``run_trial``.
  Paired validation comparison (MF-C1, :mod:`.comparison`) controls
  *evaluation* noise across episodes; it says nothing about *optimization*
  noise across training seeds -- one lucky initialization/data-order draw.
  The aggregate report this function writes distinguishes the two
  explicitly: **within-seed spread** is the dispersion of one seed's own
  per-episode validation errors (population dispersion over the complete
  frozen validation set); **across-seed spread** is the sample dispersion
  of the different seeds' mean metric values (optimization noise). A
  configuration whose multi-seed aggregate no longer beats the baseline it
  paired-beat is reported ``confirmed=False`` -- it must not be treated as
  durably promoted.

* :func:`final_test` is the single explicit action that evaluates an
  already-promoted candidate against the sealed test split and writes
  ``metrics/test.json``. It is a thin caller around
  :func:`~.promotion.perform_sealed_test_action` (the only sanctioned way
  to charge the sealed-test-use budget) plus the same
  ``rollout_beats_copy_last``/``rollout_health`` gates
  :mod:`.promotion` already uses for ordinary promotion, now checked
  against the sealed test data instead of validation. It writes
  ``metrics/test.json`` exactly once per run: a second call for the same
  run is refused before either the corpus or the sealed-test budget is
  touched, and a call that loses the race for the budget's last remaining
  use is refused without writing anything.

Neither function is reachable from ordinary search: ``run_trial`` (MF-C3)
only ever resolves ``train``/``validation`` corpus entries, and this
module's own two functions are the only Model Factory code that resolves
``test`` entries at all.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from cognitive_runtime.training.model_factory.artifacts import atomic_write_json
from cognitive_runtime.training.model_factory.checkpoint import (
    load_factory_checkpoint,
    read_factory_checkpoint_metadata,
)
from cognitive_runtime.training.model_factory.corpus import resolve_corpus
from cognitive_runtime.training.model_factory.navigation_metrics import evaluate_navigation_sessions
from cognitive_runtime.training.model_factory.promotion import (
    PromotionPolicy,
    SealedTestBudgetExhaustedError,
    TestConfirmation,
    _rollout_beats_copy_last_gate,
    _rollout_health_gate,
    perform_sealed_test_action,
)
from cognitive_runtime.training.model_factory.registry import RegistryError, population as _population
from cognitive_runtime.training.model_factory.runner import (
    RUNS_ROOT_DEFAULT,
    _action_world_model_config,
    _action_world_model_module,
    _evaluate,
    _nursery_module,
    _resolve_selection_metric,
    run_trial,
)
from cognitive_runtime.training.model_factory.spec import ExperimentSpec, _thaw
from cognitive_runtime.training.model_factory.spec import resolve as resolve_spec
from cognitive_runtime.training.model_factory.state import (
    STATE_COMPLETED,
    _lock_path_for,
    _locked,
    load_state,
    require_completed_for_promotion,
    state_path,
)

SEED_CONFIRMATION_FORMAT = "model-factory-seed-confirmation-v1"
SEED_CONFIRMATION_CHILD_FORMAT = "model-factory-seed-confirmation-child-v1"
TEST_METRICS_FORMAT = "model-factory-test-metrics-v1"


class ConfirmationError(ValueError):
    """A malformed or ineligible ``confirm_across_seeds``/``final_test`` request."""


class TestAlreadyPerformedError(ConfirmationError):
    """Raised when ``final_test`` is called a second time for the same run."""

    #: Not a pytest test case -- silences pytest's name-based collection
    #: heuristic (the class name starts with "Test").
    __test__ = False

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(
            f"final_test already performed for run {run_id!r}: metrics/test.json already exists; "
            "a run may be sealed-tested exactly once"
        )


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _run_directory(root: Union[str, Path], organism: str, run_id: str) -> Path:
    return Path(root) / organism / run_id


def _load_run_spec(run_directory: Path) -> ExperimentSpec:
    path = run_directory / "trial_spec.json"
    if not path.is_file():
        raise ConfirmationError(f"no trial_spec.json for run at {run_directory}")
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return resolve_spec(raw)


def _require_completed(run_directory: Path) -> None:
    require_completed_for_promotion(load_state(state_path(run_directory)))


def _require_confirmed_across_seeds(run_directory: Path, run_id: str) -> None:
    """``final_test`` may only confirm a candidate that already passed
    :func:`confirm_across_seeds` -- a durable champion needs both kinds of
    evidence (epic §10.2/§10.3), never a sealed-test pass standing in for a
    multi-seed rerun that was never performed."""
    path = run_directory / "metrics" / "seed_confirmation.json"
    if not path.is_file():
        raise ConfirmationError(
            f"run {run_id!r} has no metrics/seed_confirmation.json; final_test requires a prior "
            "confirm_across_seeds call that confirmed this exact run before it may sealed-test it"
        )
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("confirmed_run_id") != run_id:
        raise ConfirmationError(
            f"run {run_id!r}'s metrics/seed_confirmation.json names a different confirmed run "
            f"({payload.get('confirmed_run_id')!r}); a seed confirmation may not be reused across runs"
        )
    if not payload.get("confirmed"):
        raise ConfirmationError(
            f"run {run_id!r} was not confirmed across seeds ({payload.get('reason')}); "
            "final_test refuses to sealed-test a configuration whose multi-seed aggregate did not hold"
        )


def _check_test_budget_eligibility(
    organism_directory: Path, *, family: str, tier: str, objective: str, run_id: str, max_uses: int,
) -> None:
    """Best-effort pre-check so an ineligible or already-exhausted call never
    pays for a sealed-data evaluation it cannot record.

    Not itself the authoritative guard -- ``record_test_use``'s single
    locked critical section (invoked via ``perform_sealed_test_action``
    after evaluation) remains the one atomic check-then-act two racing
    callers are resolved against; this unlocked read only short-circuits the
    ordinary (non-racing) case of a run that was never promoted into this
    slot, or whose budget this call already knows is spent.
    """
    entries = _population(organism_directory, family=family, tier=tier, objective=objective)
    entry = next((item for item in entries if item.run_id == run_id), None)
    if entry is None:
        raise RegistryError(
            f"run_id {run_id!r} is not a population member of family={family!r} tier={tier!r} objective={objective!r}"
        )
    if entry.test_uses >= max_uses:
        raise SealedTestBudgetExhaustedError(run_id, max_uses, entry.test_uses)


# --------------------------------------------------------------- confirm_across_seeds


@dataclass(frozen=True)
class SeedConfirmationChild:
    """One training-seed rerun of the confirmed configuration.

    ``metric_value`` and ``within_seed_spread`` are both the mean and
    population standard deviation of the *raw* per-episode model MSE at the
    selection metric's declared report/horizon -- never a derived ratio/rate
    -- so they stay directly comparable to ``baseline_metric_value``.
    """

    seed: int
    run_id: str
    display_name: str
    metric_value: float
    within_seed_spread: float
    episode_count: int

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SeedConfirmationReport:
    """The full multi-seed confirmation report for one leading configuration."""

    format: str
    confirmed_run_id: str
    organism: str
    selection_metric: str
    baseline_metric_value: float
    seeds: Tuple[int, ...]
    children: Tuple[SeedConfirmationChild, ...]
    aggregate_metric_mean: float
    across_seed_spread: float
    within_seed_spread_mean: float
    minimum_practical_margin: float
    confirmed: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "confirmed_run_id": self.confirmed_run_id,
            "organism": self.organism,
            "selection_metric": self.selection_metric,
            "baseline_metric_value": self.baseline_metric_value,
            "seeds": list(self.seeds),
            "children": [child.to_dict() for child in self.children],
            "aggregate_metric_mean": self.aggregate_metric_mean,
            "across_seed_spread": self.across_seed_spread,
            "within_seed_spread_mean": self.within_seed_spread_mean,
            "minimum_practical_margin": self.minimum_practical_margin,
            "confirmed": self.confirmed,
            "reason": self.reason,
        }


def _child_spec_doc(confirmed_spec: ExperimentSpec, seed: int) -> Dict[str, Any]:
    """A fresh (not clone/resume) spec document for one training-seed rerun.

    Full independent retraining -- not a checkpoint continuation -- is what
    tests optimization noise: a clone/fine_tune child would inherit the
    confirmed run's already-trained weights and only vary what happens
    downstream of that shared starting point, understating exactly the
    "one lucky optimization trajectory" risk this function exists to catch.
    """
    doc = _thaw(confirmed_spec.to_dict())
    doc["mode"] = "fresh"
    doc.pop("parent", None)
    doc["training"]["seed"] = int(seed)
    return doc


def confirm_across_seeds(
    run_id: str,
    seeds: Sequence[int],
    *,
    organism: str,
    root: Union[str, Path] = RUNS_ROOT_DEFAULT,
    corpus_root: Optional[Union[str, Path]] = None,
    minimum_practical_margin: float = 0.0,
    naming_seed: Optional[Union[str, int]] = None,
    trace_dir: Optional[Union[str, Path]] = None,
    heartbeat_timeout_seconds: Optional[float] = None,
) -> SeedConfirmationReport:
    """Rerun ``run_id``'s configuration on each declared training seed.

    ``run_id`` must already be a completed ``clone``/``fine_tune`` trial
    whose ``metrics/comparison.json`` records
    ``decision == "candidate_improves"`` -- confirmation reruns only a
    configuration that already won its own paired A/B, per epic §10.2.
    ``seeds`` must declare at least two distinct training seeds: a single
    seed is ordinary fast search, not a durable confirmation.

    Writes one child run per seed (each with its own ``run_id``, a
    ``confirmation_source.json`` in its run directory naming ``run_id`` as
    the confirmed configuration it reruns) plus an aggregate
    ``metrics/seed_confirmation.json`` under ``run_id``'s own directory.
    Never mutates the champion registry or the sealed-test-use budget --
    those stay ``final_test``'s and the registry's own concerns.
    """
    if minimum_practical_margin < 0:
        raise ConfirmationError("minimum_practical_margin must not be negative")
    seeds = list(seeds)
    unique_seeds = list(dict.fromkeys(int(seed) for seed in seeds))
    if len(unique_seeds) != len(seeds):
        raise ConfirmationError("seeds must not contain duplicates")
    if len(unique_seeds) < 2:
        raise ConfirmationError(
            "confirm_across_seeds requires at least two independent training seeds; "
            "a single seed is ordinary fast search, not a durable confirmation"
        )

    confirmed_directory = _run_directory(root, organism, run_id)
    _require_completed(confirmed_directory)
    confirmed_spec = _load_run_spec(confirmed_directory)

    comparison_path = confirmed_directory / "metrics" / "comparison.json"
    if not comparison_path.is_file():
        raise ConfirmationError(
            f"run {run_id!r} has no metrics/comparison.json; confirm_across_seeds reruns only a "
            "configuration that already won its own paired A/B against a baseline champion"
        )
    with comparison_path.open(encoding="utf-8") as handle:
        comparison_payload = json.load(handle)
    if comparison_payload.get("status") != "evaluable" or comparison_payload.get("decision") != "candidate_improves":
        raise ConfirmationError(
            f"run {run_id!r} did not win its paired A/B (status={comparison_payload.get('status')!r}, "
            f"decision={comparison_payload.get('decision')!r}); confirm_across_seeds reruns only the "
            "leading configuration"
        )
    baseline_errors = comparison_payload["per_episode"]["baseline_raw_errors"]
    baseline_metric_value = statistics.fmean(float(value) for value in baseline_errors.values())

    selection_metric = confirmed_spec.evaluation["selection_metric"]
    corpus = resolve_corpus(
        confirmed_spec.data["corpus_id"], allow_record=False, root=corpus_root, organism=confirmed_spec.organism,
    )
    ticks_per_frame = float(corpus.data_contract.ticks_per_frame)

    children = []
    for seed in unique_seeds:
        child_doc = _child_spec_doc(confirmed_spec, seed)
        child_spec = resolve_spec(child_doc)
        result = run_trial(
            child_spec, root=root, corpus_root=corpus_root,
            naming_seed=naming_seed, trace_dir=trace_dir,
            heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        )
        if result.state != STATE_COMPLETED:
            raise ConfirmationError(
                f"seed {seed} rerun {result.run_id!r} did not complete within its training-time budget "
                f"(state={result.state!r}); a durable confirmation requires every seed rerun to complete, "
                "exactly like the training_time_budget promotion gate"
            )
        # `_resolve_selection_metric`'s per-episode values are normalized
        # lower-is-better comparison errors. For prediction paths they are
        # raw model MSE; higher-is-better navigation scores become 1-score.
        # This stays in the same units/direction as comparison.json's
        # baseline_raw_errors.
        _, per_episode = _resolve_selection_metric(result.evaluation, selection_metric, ticks_per_frame)
        metric_value = statistics.fmean(per_episode)
        within_seed_spread = statistics.pstdev(per_episode)
        atomic_write_json(
            _run_directory(root, organism, result.run_id) / "confirmation_source.json",
            {
                "format": SEED_CONFIRMATION_CHILD_FORMAT,
                "confirmed_run_id": run_id,
                "seed": seed,
            },
        )
        children.append(SeedConfirmationChild(
            seed=seed, run_id=result.run_id, display_name=result.display_name,
            metric_value=metric_value, within_seed_spread=within_seed_spread,
            episode_count=len(per_episode),
        ))

    seed_values = [child.metric_value for child in children]
    aggregate_mean = statistics.fmean(seed_values)
    across_seed_spread = statistics.stdev(seed_values)
    within_seed_spread_mean = statistics.fmean(child.within_seed_spread for child in children)
    improvement = baseline_metric_value - aggregate_mean
    confirmed = improvement >= minimum_practical_margin
    reason = (
        f"multi-seed aggregate ({aggregate_mean!r}) improves over the paired-A/B baseline "
        f"({baseline_metric_value!r}) by {improvement!r}, clearing the minimum practical margin "
        f"({minimum_practical_margin!r})"
        if confirmed else
        f"multi-seed aggregate ({aggregate_mean!r}) does not improve over the paired-A/B baseline "
        f"({baseline_metric_value!r}) by at least the minimum practical margin "
        f"({minimum_practical_margin!r}); a paired A/B win is not durable across independent "
        "training seeds and must not be treated as durably promoted"
    )

    report = SeedConfirmationReport(
        format=SEED_CONFIRMATION_FORMAT, confirmed_run_id=run_id, organism=organism,
        selection_metric=selection_metric, baseline_metric_value=baseline_metric_value,
        seeds=tuple(unique_seeds), children=tuple(children),
        aggregate_metric_mean=aggregate_mean, across_seed_spread=across_seed_spread,
        within_seed_spread_mean=within_seed_spread_mean,
        minimum_practical_margin=minimum_practical_margin, confirmed=confirmed, reason=reason,
    )
    atomic_write_json(confirmed_directory / "metrics" / "seed_confirmation.json", report.to_dict())
    return report


# --------------------------------------------------------------------------- final_test


@dataclass(frozen=True)
class FinalTestResult:
    """The outcome of one ``final_test`` call."""

    run_id: str
    passed: bool
    reason: str
    selection_metric: str
    selection_metric_value: float
    test_metrics_path: str
    test_confirmation: TestConfirmation

    def to_dict(self) -> Dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["test_confirmation"] = self.test_confirmation.to_dict()
        return payload


def final_test(
    run_id: str,
    *,
    organism: str,
    family: str,
    tier: str,
    objective: str,
    policy: PromotionPolicy,
    root: Union[str, Path] = RUNS_ROOT_DEFAULT,
    corpus_root: Optional[Union[str, Path]] = None,
    reason: Optional[str] = None,
) -> FinalTestResult:
    """Evaluate ``run_id``'s best-validation checkpoint against the sealed
    test split, exactly once.

    ``run_id`` must already be: a completed trial; confirmed by a prior
    :func:`confirm_across_seeds` call recorded in its own
    ``metrics/seed_confirmation.json`` with ``confirmed=True`` (durable
    promotion requires both the multi-seed and the sealed-test evidence,
    never sealed-test evidence alone); and a population member of the
    ``(family, tier, objective)`` registry slot (i.e. already promoted
    non-durably) -- ``perform_sealed_test_action`` (the sole sanctioned
    sealed-test-budget charge, :mod:`.promotion`) enforces the latter.
    Refuses immediately, before touching the corpus or the checkpoint, if
    ``metrics/test.json`` already exists for this run, or if the sealed-test
    budget is already known to be exhausted for ``run_id`` -- so an
    ineligible or already-exhausted call never pays for a sealed-data
    evaluation it cannot record. (This eligibility pre-check is a best-effort
    optimization, not the authoritative guard: the same
    ``perform_sealed_test_action`` call every ``final_test`` still makes
    after evaluating remains the one atomic, lock-guarded budget charge that
    two concurrent callers racing for the same last remaining use are
    resolved against.) A call refused this way, or refused later by the
    atomic charge itself, writes nothing.

    ``passed`` reuses the same ``rollout_beats_copy_last``/``rollout_health``
    gates :mod:`.promotion` already applies for ordinary (validation-based)
    promotion, now checked against the sealed test evaluation instead.
    """
    run_directory = _run_directory(root, organism, run_id)
    test_metrics_path = run_directory / "metrics" / "test.json"

    with _locked(_lock_path_for(test_metrics_path)):
        if test_metrics_path.exists():
            raise TestAlreadyPerformedError(run_id)

        _require_completed(run_directory)
        _require_confirmed_across_seeds(run_directory, run_id)
        spec = _load_run_spec(run_directory)
        selection_metric = spec.evaluation["selection_metric"]

        _check_test_budget_eligibility(
            Path(root) / organism, family=family, tier=tier, objective=objective,
            run_id=run_id, max_uses=policy.max_sealed_test_uses,
        )

        corpus = resolve_corpus(
            spec.data["corpus_id"], allow_record=False, root=corpus_root, organism=spec.organism,
        )
        test_entries = corpus.manifest["sessions"]["test"]
        if not test_entries:
            raise ConfirmationError(f"corpus {corpus.corpus_id!r} has no sealed test sessions")
        test_episode_ids = [str(entry["session_id"]) for entry in test_entries]

        awm = _action_world_model_module()
        vocabulary = _nursery_module()._action_keys_for_world(spec.data["world"])
        test_dataset = awm.build_action_sequence_dataset(
            [entry["session_path"] for entry in test_entries], action_keys=vocabulary,
        )

        checkpoint_path = run_directory / "checkpoints" / "best-validation.pt"
        loaded = load_factory_checkpoint(str(checkpoint_path))
        cfg = _action_world_model_config(spec)
        navigation_metrics = (
            evaluate_navigation_sessions(test_entries)
            if corpus.data_contract.behavior_mixture_policy else None
        )
        eval_result = _evaluate(
            loaded.model, test_dataset, spec, cfg,
            navigation_metrics=navigation_metrics,
        )

        rollout_metrics = eval_result["rollout"]
        beats_copy_last_gate = _rollout_beats_copy_last_gate(rollout_metrics)
        health_gate = _rollout_health_gate(rollout_metrics)
        passed = beats_copy_last_gate.passed and health_gate.passed
        gate_reason = "; ".join(
            gate.reason for gate in (beats_copy_last_gate, health_gate) if not gate.passed
        ) or "sealed test evaluation beats copy-last and reports healthy rollout dispersion at every horizon"

        metric_value, _ = _resolve_selection_metric(eval_result, selection_metric, test_dataset.ticks_per_frame)
        checkpoint_meta = read_factory_checkpoint_metadata(str(checkpoint_path))

        # Charge the sealed-test-use budget before persisting anything: a
        # refusal here (exhausted budget, or run_id not a population member)
        # must leave no metrics/test.json behind to retry against.
        confirmation = perform_sealed_test_action(
            Path(root) / organism, family=family, tier=tier, objective=objective,
            run_id=run_id, passed=passed, policy=policy, reason=reason,
        )

        payload = {
            "format": TEST_METRICS_FORMAT,
            "run_id": run_id,
            "selection_metric": selection_metric,
            "selection_metric_value": metric_value,
            "rollout": eval_result["rollout"],
            "direct": eval_result["direct"],
            "rollout_health": eval_result["rollout_health"],
            "per_episode_model_mse": eval_result["per_episode_model_mse"],
            "episode_ids": test_episode_ids,
            "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_meta.get("checkpoint_sha256")},
            "passed": passed,
            "reason": gate_reason,
            "test_uses_after": confirmation.test_uses_after,
            "max_sealed_test_uses": confirmation.max_sealed_test_uses,
            "performed_at": _now_iso(),
        }
        if "goal_navigation" in eval_result:
            payload["goal_navigation"] = eval_result["goal_navigation"]
        atomic_write_json(test_metrics_path, payload)

    return FinalTestResult(
        run_id=run_id, passed=passed, reason=gate_reason, selection_metric=selection_metric,
        selection_metric_value=metric_value, test_metrics_path=str(test_metrics_path),
        test_confirmation=confirmation,
    )


__all__ = [
    "SEED_CONFIRMATION_FORMAT",
    "SEED_CONFIRMATION_CHILD_FORMAT",
    "TEST_METRICS_FORMAT",
    "ConfirmationError",
    "TestAlreadyPerformedError",
    "SeedConfirmationChild",
    "SeedConfirmationReport",
    "FinalTestResult",
    "confirm_across_seeds",
    "final_test",
]
