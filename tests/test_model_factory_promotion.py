"""Regression coverage for the promotion rule engine, gate wiring, and
sealed-test-use budget (epic #212 Phase C step 4, §10.3, issue #226).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from cognitive_runtime.training.model_factory.comparison import compare_paired_episodes
from cognitive_runtime.training.model_factory.promotion import (
    METRIC_SCHEMA_VERSION,
    GateResult,
    PromotionPolicy,
    PromotionPolicyError,
    PromotionVerdict,
    SafetyMetricObservation,
    SealedTestBudgetExhaustedError,
    TestConfirmation,
    evaluate_promotion,
    perform_sealed_test_action,
)
from cognitive_runtime.training.model_factory.registry import (
    RegistryError,
    population,
    promote,
    registry_path,
)
from cognitive_runtime.training.model_factory.state import (
    STATE_COMPLETED,
    STATE_RUNNING,
    create_state,
    state_path,
    transition,
)

ALL_GATE_NAMES = {
    "checkpoint_identity",
    "evaluation_mode",
    "training_time_budget",
    "data_quality",
    "split_overlap",
    "representation",
    "rollout_beats_copy_last",
    "rollout_health",
    "event_stratified_evaluability",
    "metric_schema",
    "primary_metric_margin",
    "safety_metrics",
    "durable_test_confirmation",
}


# --------------------------------------------------------------------- fixtures


def _episode_ids(n=8):
    return [f"ep{i}" for i in range(n)]


def _comparison(*, delta, n=8, margin_confidence=0.95):
    """A paired comparison with an identical delta on every episode -- both
    the bootstrap (degenerate at the constant value) and permutation
    intervals are then deterministic for a fixed seed."""
    ids = _episode_ids(n)
    baseline = {episode: 1.0 for episode in ids}
    candidate = {episode: 1.0 + delta for episode in ids}
    return compare_paired_episodes(candidate, baseline, confidence=margin_confidence)


def _rollout_metrics(*, model_mse=0.5, copy_last_mse=1.0, health_state="healthy", include_raw=True):
    horizon_entry = {
        "beats_copy_last": model_mse < copy_last_mse,
    }
    if include_raw:
        horizon_entry["model_mse"] = model_mse
        horizon_entry["copy_last_mse"] = copy_last_mse
        horizon_entry["model_over_copy_last_mse"] = model_mse / copy_last_mse
    else:
        horizon_entry["model_over_copy_last_mse"] = model_mse / copy_last_mse
    return {
        "horizons": {4: horizon_entry},
        "rollout_health": {"state": health_state},
    }


def _event_stratified_metrics(*, include_raw=True):
    entity = {"cow_false_positive_rate": 0.1}
    if include_raw:
        entity["false_positive"] = 1
        entity["true_negative"] = 9
    return {4: {"entity": entity}}


def _experiment_report(**overrides):
    report = {
        "checkpoint": {"path": "checkpoints/best-validation.pt", "sha256": "a" * 64},
        "training_stats": {"completion_status": "completed", "evaluation_mode": "held_out"},
        "metrics": {"rollout": _rollout_metrics()},
        "event_stratified_metrics": _event_stratified_metrics(),
    }
    report.update(overrides)
    return report


def _data_quality(*, passed=True, issues=()):
    return {"passed": passed, "issues": list(issues)}


def _split_overlap(*, disjoint=True, issues=()):
    return {"disjoint": disjoint, "issues": list(issues)}


def _base_kwargs(**overrides):
    """A fully-passing call to evaluate_promotion; tests flip one field at
    a time to isolate exactly one gate's failure."""
    kwargs = dict(
        policy=PromotionPolicy(minimum_practical_margin=0.1),
        candidate_run_id="candidate",
        experiment_report=_experiment_report(),
        data_quality=_data_quality(),
        split_overlap=_split_overlap(),
        has_champion=True,
        representation=None,
        comparison=_comparison(delta=-0.5),
        safety_metrics=(),
        durable=False,
        test_confirmation=None,
    )
    kwargs.update(overrides)
    return kwargs


def _gate(verdict: PromotionVerdict, name: str) -> GateResult:
    matches = [gate for gate in verdict.gates if gate.name == name]
    assert len(matches) == 1, f"expected exactly one {name!r} gate, found {len(matches)}"
    return matches[0]


# --------------------------------------------------------------------- happy path


def test_all_gates_pass_promotes(tmp_path):
    verdict = evaluate_promotion(**_base_kwargs())
    assert verdict.promoted
    assert verdict.decision == "promote"
    assert verdict.reasons == ()
    assert {gate.name for gate in verdict.gates} == ALL_GATE_NAMES
    assert all(gate.passed for gate in verdict.gates)


def test_no_existing_champion_promotes_trivially_without_a_comparison():
    verdict = evaluate_promotion(**_base_kwargs(has_champion=False, comparison=None))
    assert verdict.promoted
    gate = _gate(verdict, "primary_metric_margin")
    assert gate.passed and not gate.applicable


def test_comparison_object_with_to_dict_is_accepted_directly():
    """comparison may be the PairedComparison dataclass itself, not just
    its to_dict() mapping."""
    comparison = _comparison(delta=-0.5)
    assert not isinstance(comparison, dict)
    verdict = evaluate_promotion(**_base_kwargs(comparison=comparison))
    assert verdict.promoted


# --------------------------------------------------------------------- AC: verdict enumerates all gates


def test_verdict_enumerates_every_gate_with_pass_fail_and_reason():
    verdict = evaluate_promotion(**_base_kwargs())
    payload = verdict.to_dict()
    assert payload["format"]
    assert payload["decision"] == "promote"
    assert {gate["name"] for gate in payload["gates"]} == ALL_GATE_NAMES
    for gate in payload["gates"]:
        assert isinstance(gate["passed"], bool)
        assert isinstance(gate["reason"], str) and gate["reason"]


def test_a_failing_verdict_names_the_reason_in_the_flat_reasons_list():
    verdict = evaluate_promotion(**_base_kwargs(data_quality=_data_quality(passed=False, issues=["bad frames"])))
    assert not verdict.promoted
    assert verdict.decision == "hold"
    assert any("bad frames" in reason for reason in verdict.reasons)


# --------------------------------------------------------------------- AC: safety metric regression


def test_safety_metric_regression_beyond_margin_is_rejected_naming_the_metric():
    safety_metrics = (
        SafetyMetricObservation(
            name="death_rate", baseline_value=0.1, candidate_value=0.4,
            higher_is_better=False, max_regression=0.05,
        ),
    )
    verdict = evaluate_promotion(**_base_kwargs(safety_metrics=safety_metrics))
    assert not verdict.promoted
    gate = _gate(verdict, "safety_metrics")
    assert not gate.passed
    assert "death_rate" in gate.reason
    # Independently blocking: every other gate still passes.
    assert all(g.passed for g in verdict.gates if g.name != "safety_metrics")


def test_safety_metric_within_its_margin_does_not_block():
    safety_metrics = (
        SafetyMetricObservation(
            name="death_rate", baseline_value=0.1, candidate_value=0.12,
            higher_is_better=False, max_regression=0.05,
        ),
    )
    verdict = evaluate_promotion(**_base_kwargs(safety_metrics=safety_metrics))
    assert verdict.promoted


def test_safety_metric_observation_rejects_a_negative_margin():
    with pytest.raises(PromotionPolicyError):
        SafetyMetricObservation(
            name="m", baseline_value=0.0, candidate_value=0.0, higher_is_better=True, max_regression=-0.1,
        )


# --------------------------------------------------------------------- AC: minimum practical margin


def test_improvement_inside_minimum_practical_margin_is_held_not_promoted():
    # A tiny but statistically significant improvement (zero-variance deltas).
    comparison = _comparison(delta=-0.02)
    verdict = evaluate_promotion(**_base_kwargs(
        comparison=comparison, policy=PromotionPolicy(minimum_practical_margin=0.5),
    ))
    assert not verdict.promoted
    assert verdict.decision == "hold"
    gate = _gate(verdict, "primary_metric_margin")
    assert not gate.passed
    assert "minimum practical margin" in gate.reason


def test_improvement_clearing_the_margin_promotes():
    comparison = _comparison(delta=-0.5)
    verdict = evaluate_promotion(**_base_kwargs(
        comparison=comparison, policy=PromotionPolicy(minimum_practical_margin=0.1),
    ))
    assert verdict.promoted


def test_not_statistically_significant_improvement_does_not_promote():
    # Small, noisy deltas straddling zero -> intervals should not exclude 0.
    ids = _episode_ids(6)
    baseline = {e: 1.0 for e in ids}
    candidate = {"ep0": 0.9, "ep1": 1.1, "ep2": 0.95, "ep3": 1.05, "ep4": 1.0, "ep5": 0.99}
    comparison = compare_paired_episodes(candidate, baseline)
    verdict = evaluate_promotion(**_base_kwargs(comparison=comparison))
    assert not verdict.promoted
    gate = _gate(verdict, "primary_metric_margin")
    assert not gate.passed


def test_champion_exists_but_no_comparison_supplied_fails_closed():
    verdict = evaluate_promotion(**_base_kwargs(has_champion=True, comparison=None))
    assert not verdict.promoted
    gate = _gate(verdict, "primary_metric_margin")
    assert not gate.passed
    assert "no paired comparison" in gate.reason


# --------------------------------------------------------------------- AC: budget_exceeded


def test_budget_exceeded_candidate_is_rejected_regardless_of_its_metrics():
    verdict = evaluate_promotion(**_base_kwargs(
        experiment_report=_experiment_report(training_stats={"completion_status": "budget_exceeded"}),
    ))
    assert not verdict.promoted
    gate = _gate(verdict, "training_time_budget")
    assert not gate.passed
    assert "budget_exceeded" in gate.reason
    # Independently blocking: the strong primary-metric improvement is still recognised.
    assert _gate(verdict, "primary_metric_margin").passed


@pytest.mark.parametrize("status", ["failed", "cancelled", "timeout_unrecoverable"])
def test_other_non_completed_statuses_also_block(status):
    verdict = evaluate_promotion(**_base_kwargs(
        experiment_report=_experiment_report(training_stats={"completion_status": status}),
    ))
    assert not verdict.promoted
    assert not _gate(verdict, "training_time_budget").passed


# --------------------------------------------------------------------- AC: training loss is never a promotion metric


def test_better_training_loss_and_worse_validation_metric_is_rejected():
    """A candidate that improved training loss but regressed on the held-out
    validation comparison must still be held -- evaluate_promotion never
    reads a training-loss field for its decision."""
    regressed_comparison = _comparison(delta=+0.2)  # candidate worse than baseline
    report = _experiment_report(training_stats={
        "completion_status": "completed",
        "evaluation_mode": "held_out",
        # A claimed training-loss improvement must have zero effect on the verdict.
        "loss_curves": {"total_loss": [1.0, 0.5, 0.1]},
        "final_total_loss": 0.1,
        "training_loss_improved_over_champion": True,
    })
    verdict = evaluate_promotion(**_base_kwargs(experiment_report=report, comparison=regressed_comparison))
    assert not verdict.promoted
    gate = _gate(verdict, "primary_metric_margin")
    assert not gate.passed
    assert "regressed" in gate.reason


# --------------------------------------------------------------------- AC: data-quality / split-overlap / representation / rollout-health


def test_data_quality_gate_blocks_independently():
    verdict = evaluate_promotion(**_base_kwargs(data_quality=_data_quality(passed=False, issues=["bad pixels"])))
    assert not verdict.promoted
    gate = _gate(verdict, "data_quality")
    assert not gate.passed and "bad pixels" in gate.reason
    assert all(g.passed for g in verdict.gates if g.name != "data_quality")


def test_split_overlap_gate_blocks_independently():
    verdict = evaluate_promotion(**_base_kwargs(
        split_overlap=_split_overlap(disjoint=False, issues=["train and test share session IDs: ['s1']"]),
    ))
    assert not verdict.promoted
    gate = _gate(verdict, "split_overlap")
    assert not gate.passed and "share session IDs" in gate.reason
    assert all(g.passed for g in verdict.gates if g.name != "split_overlap")


def test_representation_gate_skipped_when_not_configured():
    verdict = evaluate_promotion(**_base_kwargs(representation=None))
    gate = _gate(verdict, "representation")
    assert gate.passed and not gate.applicable


def test_representation_gate_blocks_on_collapse():
    representation = {
        "passed": False, "gate_evaluable": True,
        "hidden_orientation_r2": 0.01, "latent_collapsed": True,
    }
    verdict = evaluate_promotion(**_base_kwargs(representation=representation))
    assert not verdict.promoted
    gate = _gate(verdict, "representation")
    assert not gate.passed
    assert "collapse" in gate.reason.lower()


def test_rollout_health_gate_blocks_when_unhealthy():
    report = _experiment_report(metrics={"rollout": _rollout_metrics(health_state="frozen")})
    verdict = evaluate_promotion(**_base_kwargs(experiment_report=report))
    assert not verdict.promoted
    gate = _gate(verdict, "rollout_health")
    assert not gate.passed and "frozen" in gate.reason


def test_rollout_beats_copy_last_gate_blocks_when_worse_than_copy_last():
    report = _experiment_report(metrics={"rollout": _rollout_metrics(model_mse=1.5, copy_last_mse=1.0)})
    verdict = evaluate_promotion(**_base_kwargs(experiment_report=report))
    assert not verdict.promoted
    assert not _gate(verdict, "rollout_beats_copy_last").passed


# --------------------------------------------------------------------- AC: versioned metric schema


def test_metric_schema_gate_blocks_a_copy_last_ratio_without_raw_errors():
    report = _experiment_report(metrics={"rollout": _rollout_metrics(include_raw=False)})
    verdict = evaluate_promotion(**_base_kwargs(experiment_report=report))
    assert not verdict.promoted
    gate = _gate(verdict, "metric_schema")
    assert not gate.passed
    assert "model_over_copy_last_mse" in gate.reason


def test_metric_schema_gate_blocks_an_event_rate_without_raw_counts():
    report = _experiment_report(event_stratified_metrics=_event_stratified_metrics(include_raw=False))
    verdict = evaluate_promotion(**_base_kwargs(experiment_report=report))
    assert not verdict.promoted
    gate = _gate(verdict, "metric_schema")
    assert not gate.passed
    assert "cow_false_positive_rate" in gate.reason


def test_metric_schema_version_is_a_stable_identifier():
    assert METRIC_SCHEMA_VERSION.startswith("model-factory-promotion-metric-schema-v")


# --------------------------------------------------------------------- AC: checkpoint identity / training-replay (existing verdict shape)


def test_checkpoint_identity_gate_blocks_without_path_and_sha256():
    report = _experiment_report(checkpoint={"path": None, "sha256": None})
    verdict = evaluate_promotion(**_base_kwargs(experiment_report=report))
    assert not verdict.promoted
    assert not _gate(verdict, "checkpoint_identity").passed


def test_training_replay_evaluation_mode_blocks_promotion():
    report = _experiment_report(training_stats={
        "completion_status": "completed", "evaluation_mode": "training_replay",
    })
    verdict = evaluate_promotion(**_base_kwargs(experiment_report=report))
    assert not verdict.promoted
    assert not _gate(verdict, "evaluation_mode").passed


# --------------------------------------------------------------------- AC: durable champion needs a final test action


def test_durable_promotion_without_a_test_confirmation_is_held():
    verdict = evaluate_promotion(**_base_kwargs(durable=True, test_confirmation=None))
    assert not verdict.promoted
    assert not _gate(verdict, "durable_test_confirmation").passed


def test_durable_promotion_with_a_failed_confirmation_is_held():
    confirmation = TestConfirmation(
        run_id="candidate", performed=True, passed=False, test_uses_after=1,
        max_sealed_test_uses=3, reason="regressed on the sealed split",
    )
    verdict = evaluate_promotion(**_base_kwargs(durable=True, test_confirmation=confirmation))
    assert not verdict.promoted
    gate = _gate(verdict, "durable_test_confirmation")
    assert not gate.passed
    assert "regressed on the sealed split" in gate.reason


def test_durable_promotion_with_a_passing_confirmation_promotes():
    confirmation = TestConfirmation(
        run_id="candidate", performed=True, passed=True, test_uses_after=1, max_sealed_test_uses=3,
    )
    verdict = evaluate_promotion(**_base_kwargs(durable=True, test_confirmation=confirmation))
    assert verdict.promoted


def test_non_durable_promotion_ignores_test_confirmation_entirely():
    verdict = evaluate_promotion(**_base_kwargs(durable=False, test_confirmation=None))
    gate = _gate(verdict, "durable_test_confirmation")
    assert gate.passed and not gate.applicable


def test_durable_promotion_rejects_a_confirmation_earned_by_a_different_candidate():
    """A sealed-test confirmation is bound to the run it was performed for --
    reusing candidate A's passing confirmation to durably promote candidate
    B must fail, not silently pass through."""
    confirmation_for_a = TestConfirmation(
        run_id="candidate-a", performed=True, passed=True, test_uses_after=1, max_sealed_test_uses=3,
    )
    verdict = evaluate_promotion(**_base_kwargs(
        candidate_run_id="candidate-b", durable=True, test_confirmation=confirmation_for_a,
    ))
    assert not verdict.promoted
    gate = _gate(verdict, "durable_test_confirmation")
    assert not gate.passed
    assert "candidate-a" in gate.reason and "candidate-b" in gate.reason


# --------------------------------------------------------------------- PromotionPolicy validation


def test_promotion_policy_rejects_a_negative_margin():
    with pytest.raises(PromotionPolicyError):
        PromotionPolicy(minimum_practical_margin=-0.1)


def test_promotion_policy_rejects_a_non_positive_test_use_budget():
    with pytest.raises(PromotionPolicyError):
        PromotionPolicy(minimum_practical_margin=0.0, max_sealed_test_uses=0)


# --------------------------------------------------------------------- AC: sealed-test-use budget


def _mark_completed(root, run_id):
    path = state_path(root / run_id)
    create_state(path, run_id)
    transition(path, STATE_RUNNING)
    transition(path, STATE_COMPLETED)


def _promote_population_member(root, run_id, *, family="f", tier="fast", objective="obj"):
    _mark_completed(root, run_id)
    promote(
        root, family=family, tier=tier, objective=objective, run_id=run_id,
        checkpoint_path="checkpoints/best-validation.pt", checkpoint_sha256="c" * 64,
        metrics={"m": 1.0}, as_leading=False,
    )


def test_perform_sealed_test_action_records_each_use_and_refuses_once_exhausted(tmp_path):
    _promote_population_member(tmp_path, "candidate")
    policy = PromotionPolicy(minimum_practical_margin=0.1, max_sealed_test_uses=2)

    first = perform_sealed_test_action(
        tmp_path, family="f", tier="fast", objective="obj", run_id="candidate",
        passed=True, policy=policy, reason="first sealed check",
    )
    assert first.performed and first.test_uses_after == 1

    second = perform_sealed_test_action(
        tmp_path, family="f", tier="fast", objective="obj", run_id="candidate",
        passed=True, policy=policy, reason="second sealed check",
    )
    assert second.test_uses_after == 2

    with pytest.raises(SealedTestBudgetExhaustedError):
        perform_sealed_test_action(
            tmp_path, family="f", tier="fast", objective="obj", run_id="candidate",
            passed=True, policy=policy, reason="third sealed check -- must be refused",
        )

    # The refused action must not have recorded a third use.
    members = {entry.run_id: entry for entry in population(tmp_path, family="f", tier="fast", objective="obj")}
    assert members["candidate"].test_uses == 2

    # Every prior (accepted) use is recorded in the registry's own history.
    document = json.loads(registry_path(tmp_path).read_text(encoding="utf-8"))
    history = document["slots"]["f"]["fast"]["obj"]["history"]
    test_use_entries = [entry for entry in history if entry["action"] == "record_test_use"]
    assert len(test_use_entries) == 2
    assert [entry["reason"] for entry in test_use_entries] == ["first sealed check", "second sealed check"]


def test_perform_sealed_test_action_requires_an_existing_population_member(tmp_path):
    policy = PromotionPolicy(minimum_practical_margin=0.1, max_sealed_test_uses=3)
    with pytest.raises(RegistryError, match="no registry slot"):
        perform_sealed_test_action(
            tmp_path, family="f", tier="fast", objective="obj", run_id="ghost", passed=True, policy=policy,
        )

    # A slot that exists but does not contain this run_id is refused for the
    # same reason `record_test_use` refuses it: a test action is attributed
    # to an already-evaluated candidate, never an unrelated identifier.
    _promote_population_member(tmp_path, "someone-else")
    with pytest.raises(RegistryError, match="not a population member"):
        perform_sealed_test_action(
            tmp_path, family="f", tier="fast", objective="obj", run_id="ghost", passed=True, policy=policy,
        )


def test_sealed_test_budget_exhausted_error_names_the_run_and_counts(tmp_path):
    _promote_population_member(tmp_path, "candidate")
    policy = PromotionPolicy(minimum_practical_margin=0.1, max_sealed_test_uses=1)
    perform_sealed_test_action(
        tmp_path, family="f", tier="fast", objective="obj", run_id="candidate", passed=True, policy=policy,
    )
    with pytest.raises(SealedTestBudgetExhaustedError) as excinfo:
        perform_sealed_test_action(
            tmp_path, family="f", tier="fast", objective="obj", run_id="candidate", passed=True, policy=policy,
        )
    assert excinfo.value.run_id == "candidate"
    assert excinfo.value.max_sealed_test_uses == 1
    assert excinfo.value.current_uses == 1


_SEALED_TEST_WORKER_SCRIPT = """
import sys, os, time, json
sys.path.insert(0, {path!r})
from cognitive_runtime.training.model_factory.promotion import (
    PromotionPolicy, SealedTestBudgetExhaustedError, perform_sealed_test_action,
)

ready_file = {ready_file!r}
go_file = {go_file!r}
out_file = {out_file!r}

with open(ready_file, "w") as handle:
    handle.write("ready")
deadline = time.monotonic() + 30.0
while not os.path.exists(go_file):
    if time.monotonic() > deadline:
        raise SystemExit("timed out waiting for go file")

policy = PromotionPolicy(minimum_practical_margin=0.1, max_sealed_test_uses={max_uses!r})
result = {{}}
try:
    confirmation = perform_sealed_test_action(
        {root!r}, family="f", tier="fast", objective="obj", run_id="candidate",
        passed=True, policy=policy,
    )
    result["ok"] = True
    result["test_uses_after"] = confirmation.test_uses_after
except SealedTestBudgetExhaustedError as exc:
    result["ok"] = False
    result["error"] = str(exc)
with open(out_file, "w") as handle:
    json.dump(result, handle)
"""


def test_perform_sealed_test_action_serializes_two_racing_workers_at_the_cap_boundary(tmp_path):
    """AC: exhausting the sealed-test budget refuses the *next* action. Two
    processes racing for the one remaining slot must never both succeed --
    the check-then-act must be atomic under the registry's lock, not a
    separate read here followed by a separate locked write."""
    _promote_population_member(tmp_path, "candidate")
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    max_uses = 1

    procs, outcomes = [], []
    for label in ("a", "b"):
        ready_file = tmp_path / f"ready-{label}"
        go_file = tmp_path / "go"
        out_file = tmp_path / f"out-{label}.json"
        script = _SEALED_TEST_WORKER_SCRIPT.format(
            path=repo_root, root=str(tmp_path), max_uses=max_uses,
            ready_file=str(ready_file), go_file=str(go_file), out_file=str(out_file),
        )
        proc = subprocess.Popen([sys.executable, "-c", script])
        procs.append(proc)
        outcomes.append((ready_file, out_file))

    deadline = __import__("time").monotonic() + 30.0
    for ready_file, _out_file in outcomes:
        while not ready_file.exists():
            assert __import__("time").monotonic() < deadline
            __import__("time").sleep(0.01)
    (tmp_path / "go").write_text("go", encoding="utf-8")

    for proc in procs:
        assert proc.wait(timeout=30) == 0

    results = [json.loads(out_file.read_text(encoding="utf-8")) for _, out_file in outcomes]
    successes = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    assert len(successes) == 1, results
    assert len(failures) == 1, results
    assert successes[0]["test_uses_after"] == max_uses

    members = {entry.run_id: entry for entry in population(tmp_path, family="f", tier="fast", objective="obj")}
    assert members["candidate"].test_uses == max_uses


# --------------------------------------------------------------------- end-to-end: perform then confirm durable


def test_end_to_end_sealed_test_action_feeds_the_durable_gate(tmp_path):
    _promote_population_member(tmp_path, "candidate")
    policy = PromotionPolicy(minimum_practical_margin=0.1, max_sealed_test_uses=3)
    confirmation = perform_sealed_test_action(
        tmp_path, family="f", tier="fast", objective="obj", run_id="candidate",
        passed=True, policy=policy, reason="final sealed confirmation",
    )
    verdict = evaluate_promotion(**_base_kwargs(policy=policy, durable=True, test_confirmation=confirmation))
    assert verdict.promoted
    assert _gate(verdict, "durable_test_confirmation").passed


# --------------------------------------------------------------------- torch-free import


_NO_TORCH_SCRIPT = """
import sys

class _BlockTorch:
    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is deliberately blocked for this test")
        return None

sys.meta_path.insert(0, _BlockTorch())
sys.path.insert(0, {path!r})
import cognitive_runtime.training.model_factory.promotion  # noqa: F401
print("OK")
"""


def test_promotion_module_imports_cleanly_without_torch():
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH_SCRIPT.format(path=repo_root)],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
