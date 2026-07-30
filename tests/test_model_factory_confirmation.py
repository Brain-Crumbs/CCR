"""Regression coverage for multi-training-seed durable confirmation and the
sealed-test final action (epic #212 Phase C step 5, §10.2, §10.3, issue #227).

Every scenario trains a real (tiny) cortex against real (tiny) recorded
Crafter sessions, mirroring ``tests/test_model_factory_runner.py`` -- the
seed reruns and the sealed-test evaluation are the actual integration this
issue is about, not a stand-in for it. Where a specific pass/fail outcome
needs to be deterministic rather than dependent on how well a few epochs of
toy training happen to go, a ``metrics/comparison.json`` baseline is set to
a value real training could not plausibly cross (very large to guarantee a
win, effectively zero to guarantee a loss) rather than mocking the trainer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.training.model_factory.confirmation import (  # noqa: E402
    ConfirmationError,
    FinalTestResult,
    SeedConfirmationReport,
    TestAlreadyPerformedError,
    confirm_across_seeds,
    final_test,
)
from cognitive_runtime.training.model_factory.corpus import build_corpus  # noqa: E402
from cognitive_runtime.training.model_factory.promotion import (  # noqa: E402
    PromotionPolicy,
    SealedTestBudgetExhaustedError,
    _rollout_beats_copy_last_gate,
    _rollout_health_gate,
    perform_sealed_test_action,
)
from cognitive_runtime.training.model_factory.registry import (  # noqa: E402
    RegistryError,
    population,
    promote,
)
from cognitive_runtime.training.model_factory.runner import run_trial  # noqa: E402
from cognitive_runtime.training.model_factory.state import (  # noqa: E402
    STATE_COMPLETED,
    STATE_RUNNING,
    IncompleteRunError,
    create_state,
    state_path,
    transition,
)

ORGANISM = "Test"


def _build_corpus(root: Path, corpus_id: str = "crafter-tiny-v1", *, with_test_split: bool = True):
    splits = {
        "train": {"walk_forward_short": [0, 1]},
        "validation": {"walk_forward_short": [100, 101]},
        "test": {"walk_forward_short": [200, 201]} if with_test_split else {},
    }
    spec = {
        "root": str(root),
        "organism": ORGANISM,
        "corpus_id": corpus_id,
        "generator": {
            "world": "crafter",
            "backend": "crafter",
            "episode_ticks": 24,
            "world_size": 16,
            "data_quality_gate": False,
        },
        "splits": splits,
    }
    return build_corpus(spec)


def _spec_dict(corpus_id, *, mode="fresh", parent=None, training_overrides=None):
    training = {
        "objective": "windowed_rollout",
        "optimizer": {"name": "adamw", "lr": 0.02},
        "batch_size": 4,
        "seed": 0,
        "rollout_frames": 2,
        "warmup_frames": 2,
        "scheduled_sampling_p": 0.0,
        "epoch_budget": 2,
        "checkpoint_cadence_epochs": 1,
        "device": "cpu",
        "precision": "fp32",
    }
    if training_overrides:
        training.update(training_overrides)
    doc = {
        "organism": ORGANISM,
        "mode": mode,
        "data": {"corpus_id": corpus_id, "world": "crafter", "horizons_ticks": [1, 2]},
        "model": {
            "backbone": "transformer", "latent_width": 8, "hidden_dim": 16,
            "action_embed_dim": 4, "reconstruction_size": 4, "context_length": 3,
            "backbone_kwargs": {"n_heads": 2, "n_layers": 1},
        },
        "training": training,
        "evaluation": {"selection_metric": "rollout.t+1.model_over_copy_last_mse"},
    }
    if parent is not None:
        doc["parent"] = parent
    return doc


def _parent_block(result):
    return {"run_id": result.run_id, "checkpoint": "best-validation.pt", "sha256": result.checkpoint_sha256}


def _run(spec_doc, tmp_path, *, run_id=None):
    return run_trial(
        spec_doc, root=str(tmp_path / "runs"), corpus_root=str(tmp_path / "corpora"), run_id=run_id,
    )


def _write_comparison(directory: Path, *, decision: str, baseline_value: float, status: str = "evaluable"):
    """Overwrite a real clone/fine_tune run's comparison.json with a
    deterministic baseline, so confirm_across_seeds's win/lose decision does
    not depend on how well a couple of toy-training epochs happen to go."""
    path = directory / "metrics" / "comparison.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode_ids = payload["episode_ids"]
    payload["status"] = status
    payload["decision"] = decision
    payload["per_episode"]["baseline_raw_errors"] = {episode: baseline_value for episode in episode_ids}
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def corpus(tmp_path):
    return _build_corpus(tmp_path / "corpora")


def _winning_candidate(tmp_path, corpus, *, baseline_value=1_000.0):
    """A real completed clone run whose comparison.json is forced to record
    a decisive paired-A/B win against an (unrealistically generous) baseline."""
    baseline = _run(_spec_dict(corpus.corpus_id), tmp_path)
    candidate = _run(
        _spec_dict(
            corpus.corpus_id, mode="clone", parent=_parent_block(baseline),
            training_overrides={"scheduled_sampling_p": 0.1},
        ),
        tmp_path,
    )
    _write_comparison(candidate.directory, decision="candidate_improves", baseline_value=baseline_value)
    return baseline, candidate


# --------------------------------------------------------------------- confirm_across_seeds


def test_confirm_across_seeds_requires_at_least_two_distinct_seeds(tmp_path, corpus):
    _, candidate = _winning_candidate(tmp_path, corpus)

    with pytest.raises(ConfirmationError, match="at least two"):
        confirm_across_seeds(
            candidate.run_id, [7], organism=ORGANISM,
            root=str(tmp_path / "runs"), corpus_root=str(tmp_path / "corpora"),
        )

    with pytest.raises(ConfirmationError, match="duplicate"):
        confirm_across_seeds(
            candidate.run_id, [7, 7], organism=ORGANISM,
            root=str(tmp_path / "runs"), corpus_root=str(tmp_path / "corpora"),
        )


def test_confirm_across_seeds_requires_a_winning_paired_comparison(tmp_path, corpus):
    # A fresh baseline has no metrics/comparison.json at all.
    baseline = _run(_spec_dict(corpus.corpus_id), tmp_path)
    with pytest.raises(ConfirmationError, match="no metrics/comparison.json"):
        confirm_across_seeds(
            baseline.run_id, [1, 2], organism=ORGANISM,
            root=str(tmp_path / "runs"), corpus_root=str(tmp_path / "corpora"),
        )

    # A clone whose comparison recorded "hold" (did not win its A/B) is
    # also not eligible for a durable multi-seed rerun.
    held = _run(
        _spec_dict(corpus.corpus_id, mode="clone", parent=_parent_block(baseline)),
        tmp_path,
    )
    _write_comparison(held.directory, decision="hold", baseline_value=0.0)
    with pytest.raises(ConfirmationError, match="did not win its paired A/B"):
        confirm_across_seeds(
            held.run_id, [1, 2], organism=ORGANISM,
            root=str(tmp_path / "runs"), corpus_root=str(tmp_path / "corpora"),
        )


def test_confirm_across_seeds_produces_one_child_per_seed_with_lineage_naming_the_confirmed_configuration(
    tmp_path, corpus,
):
    _, candidate = _winning_candidate(tmp_path, corpus, baseline_value=1_000.0)

    report = confirm_across_seeds(
        candidate.run_id, [11, 22, 33], organism=ORGANISM,
        root=str(tmp_path / "runs"), corpus_root=str(tmp_path / "corpora"),
    )

    assert isinstance(report, SeedConfirmationReport)
    assert report.confirmed_run_id == candidate.run_id
    assert report.seeds == (11, 22, 33)
    assert [child.seed for child in report.children] == [11, 22, 33]

    run_ids = {child.run_id for child in report.children}
    assert len(run_ids) == 3, "each seed must get its own run_id"
    assert candidate.run_id not in run_ids

    for child in report.children:
        child_directory = tmp_path / "runs" / ORGANISM / child.run_id
        assert (child_directory / "checkpoints" / "best-validation.pt").is_file()
        source_path = child_directory / "confirmation_source.json"
        assert source_path.is_file()
        source = json.loads(source_path.read_text(encoding="utf-8"))
        assert source["confirmed_run_id"] == candidate.run_id
        assert source["seed"] == child.seed
        assert child.episode_count == 2  # two frozen validation sessions
        assert child.within_seed_spread >= 0.0

    assert report.across_seed_spread >= 0.0
    # A generous (1000.0) baseline: real toy-model MSE must be far below it.
    assert report.confirmed is True
    assert "improves over" in report.reason

    confirmation_path = candidate.directory / "metrics" / "seed_confirmation.json"
    persisted = json.loads(confirmation_path.read_text(encoding="utf-8"))
    assert persisted == report.to_dict()


def test_confirm_across_seeds_win_losing_the_multi_seed_aggregate_is_not_confirmed(tmp_path, corpus):
    """AC: a configuration that wins its paired A/B but loses on the
    multi-seed aggregate must not be reported as durably confirmed."""
    _, candidate = _winning_candidate(tmp_path, corpus, baseline_value=1e-9)

    report = confirm_across_seeds(
        candidate.run_id, [1, 2], organism=ORGANISM,
        root=str(tmp_path / "runs"), corpus_root=str(tmp_path / "corpora"),
    )

    assert report.confirmed is False
    assert "not durable across independent training seeds" in report.reason
    assert report.aggregate_metric_mean > report.baseline_metric_value


def test_confirm_across_seeds_module_imports_cleanly_without_torch():
    repo_root = str(Path(__file__).resolve().parents[1])
    script = """
import sys

class _BlockTorch:
    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is deliberately blocked for this test")
        return None

sys.meta_path.insert(0, _BlockTorch())
sys.path.insert(0, {path!r})
import cognitive_runtime.training.model_factory.confirmation  # noqa: F401
print("OK")
""".format(path=repo_root)
    result = subprocess.run(
        [sys.executable, "-c", script], env=dict(os.environ), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


# --------------------------------------------------------------------------- final_test


def _mark_completed(directory, run_id):
    path = state_path(directory)
    create_state(path, run_id)
    transition(path, STATE_RUNNING)
    transition(path, STATE_COMPLETED)


def _promote(root, organism, run_id, *, family="fam", tier="fast", objective="obj"):
    promote(
        Path(root) / organism, family=family, tier=tier, objective=objective, run_id=run_id,
        checkpoint_path="checkpoints/best-validation.pt", checkpoint_sha256="c" * 64,
        metrics={}, as_leading=False,
    )


def _population(root, organism, *, family="fam", tier="fast", objective="obj"):
    entries = population(Path(root) / organism, family=family, tier=tier, objective=objective)
    return {entry.run_id: entry for entry in entries}


def test_final_test_writes_metrics_test_json_matching_the_promotion_gates(tmp_path, corpus):
    result = _run(_spec_dict(corpus.corpus_id), tmp_path)
    root = str(tmp_path / "runs")
    _promote(root, ORGANISM, result.run_id)
    policy = PromotionPolicy(minimum_practical_margin=0.1, max_sealed_test_uses=3)

    outcome = final_test(
        result.run_id, organism=ORGANISM, family="fam", tier="fast", objective="obj",
        policy=policy, root=root, corpus_root=str(tmp_path / "corpora"),
        reason="MF-C5 sealed confirmation",
    )

    assert isinstance(outcome, FinalTestResult)
    test_metrics_path = result.directory / "metrics" / "test.json"
    assert test_metrics_path.is_file()
    payload = json.loads(test_metrics_path.read_text(encoding="utf-8"))
    assert payload["format"] == "model-factory-test-metrics-v1"
    assert payload["episode_ids"], "sealed test episode ids must be recorded"
    assert payload["passed"] == outcome.passed
    assert payload["test_uses_after"] == 1

    # `passed` must reuse the same gates ordinary promotion applies, now
    # against the sealed test evaluation.
    expected = (
        _rollout_beats_copy_last_gate(payload["rollout"]).passed
        and _rollout_health_gate(payload["rollout"]).passed
    )
    assert payload["passed"] == expected

    members = _population(root, ORGANISM)
    assert members[result.run_id].test_uses == 1


def test_final_test_refuses_a_second_call_for_the_same_run(tmp_path, corpus):
    result = _run(_spec_dict(corpus.corpus_id), tmp_path)
    root = str(tmp_path / "runs")
    _promote(root, ORGANISM, result.run_id)
    policy = PromotionPolicy(minimum_practical_margin=0.1, max_sealed_test_uses=3)

    final_test(
        result.run_id, organism=ORGANISM, family="fam", tier="fast", objective="obj",
        policy=policy, root=root, corpus_root=str(tmp_path / "corpora"),
    )
    with pytest.raises(TestAlreadyPerformedError):
        final_test(
            result.run_id, organism=ORGANISM, family="fam", tier="fast", objective="obj",
            policy=policy, root=root, corpus_root=str(tmp_path / "corpora"),
        )

    # The refused second call must not have charged a second budget use.
    members = _population(root, ORGANISM)
    assert members[result.run_id].test_uses == 1


def test_final_test_refused_when_budget_already_exhausted_writes_nothing(tmp_path, corpus):
    result = _run(_spec_dict(corpus.corpus_id), tmp_path)
    root = str(tmp_path / "runs")
    _promote(root, ORGANISM, result.run_id)
    policy = PromotionPolicy(minimum_practical_margin=0.1, max_sealed_test_uses=1)

    # Exhaust the budget directly (not through final_test), simulating an
    # earlier sealed-test action already recorded against this run_id.
    perform_sealed_test_action(
        Path(root) / ORGANISM, family="fam", tier="fast", objective="obj",
        run_id=result.run_id, passed=True, policy=policy,
    )

    with pytest.raises(SealedTestBudgetExhaustedError):
        final_test(
            result.run_id, organism=ORGANISM, family="fam", tier="fast", objective="obj",
            policy=policy, root=root, corpus_root=str(tmp_path / "corpora"),
        )

    assert not (result.directory / "metrics" / "test.json").exists()


def test_final_test_requires_a_completed_run(tmp_path, corpus):
    result = _run(_spec_dict(corpus.corpus_id), tmp_path)
    root = str(tmp_path / "runs")
    _promote(root, ORGANISM, result.run_id)
    # Regress the persisted state to something non-terminal, as if a bug
    # elsewhere had left it that way.
    state_file = state_path(result.directory)
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    payload["state"] = "running"
    state_file.write_text(json.dumps(payload), encoding="utf-8")

    policy = PromotionPolicy(minimum_practical_margin=0.1, max_sealed_test_uses=3)
    with pytest.raises(IncompleteRunError):
        final_test(
            result.run_id, organism=ORGANISM, family="fam", tier="fast", objective="obj",
            policy=policy, root=root, corpus_root=str(tmp_path / "corpora"),
        )
    assert not (result.directory / "metrics" / "test.json").exists()


def test_final_test_requires_an_existing_registry_population_member(tmp_path, corpus):
    result = _run(_spec_dict(corpus.corpus_id), tmp_path)
    root = str(tmp_path / "runs")
    # Deliberately never promoted into the registry.
    policy = PromotionPolicy(minimum_practical_margin=0.1, max_sealed_test_uses=3)
    with pytest.raises(RegistryError):
        final_test(
            result.run_id, organism=ORGANISM, family="fam", tier="fast", objective="obj",
            policy=policy, root=root, corpus_root=str(tmp_path / "corpora"),
        )
    assert not (result.directory / "metrics" / "test.json").exists()


def test_final_test_requires_the_corpus_to_declare_test_sessions(tmp_path):
    corpus_without_test = _build_corpus(tmp_path / "corpora", corpus_id="no-test-split", with_test_split=False)
    result = _run(_spec_dict(corpus_without_test.corpus_id), tmp_path)
    root = str(tmp_path / "runs")
    _promote(root, ORGANISM, result.run_id)
    policy = PromotionPolicy(minimum_practical_margin=0.1, max_sealed_test_uses=3)
    with pytest.raises(ConfirmationError, match="no sealed test sessions"):
        final_test(
            result.run_id, organism=ORGANISM, family="fam", tier="fast", objective="obj",
            policy=policy, root=root, corpus_root=str(tmp_path / "corpora"),
        )


def test_final_test_module_imports_cleanly_without_torch():
    # Covered together with confirm_across_seeds's import in the same module;
    # kept as a separate, explicitly named test for the acceptance criterion.
    test_confirm_across_seeds_module_imports_cleanly_without_torch()


# --------------------------------------------------------------------------- AC: sealed split isolation


def test_no_search_step_ever_opens_the_sealed_test_partition(tmp_path, corpus, monkeypatch):
    """AC: no routine trial, search step, or population update can reach the
    test partition. Runs a small "full search" (fresh baseline, a clone A/B,
    a fine_tune branch) and asserts the sealed test sessions were never
    opened by any of them -- then, as a sanity check that this assertion
    mechanism actually catches misuse, confirms final_test *does* open them.
    """
    from cognitive_runtime.training import action_world_model as awm_module

    test_paths = {entry["session_path"] for entry in corpus.manifest["sessions"]["test"]}
    assert test_paths, "fixture must actually declare sealed test sessions"

    opened_paths: list[str] = []
    original = awm_module.build_action_sequence_dataset

    def _tracking(session_paths, *args, **kwargs):
        opened_paths.extend(session_paths)
        return original(session_paths, *args, **kwargs)

    monkeypatch.setattr(awm_module, "build_action_sequence_dataset", _tracking)

    baseline = _run(_spec_dict(corpus.corpus_id), tmp_path)
    parent = _parent_block(baseline)
    candidate = _run(
        _spec_dict(corpus.corpus_id, mode="clone", parent=parent,
                    training_overrides={"scheduled_sampling_p": 0.1}),
        tmp_path,
    )
    _run(
        _spec_dict(corpus.corpus_id, mode="fine_tune", parent=parent,
                    training_overrides={"optimizer": {"name": "adamw", "lr": 0.05}}),
        tmp_path,
    )

    assert not (set(opened_paths) & test_paths), (
        "routine search must never open the sealed test partition"
    )

    # Sanity check: the tracking hook does catch a real sealed-test read.
    root = str(tmp_path / "runs")
    _promote(root, ORGANISM, candidate.run_id)
    policy = PromotionPolicy(minimum_practical_margin=0.1, max_sealed_test_uses=3)
    final_test(
        candidate.run_id, organism=ORGANISM, family="fam", tier="fast", objective="obj",
        policy=policy, root=root, corpus_root=str(tmp_path / "corpora"),
    )
    assert set(opened_paths) & test_paths, "final_test must be the path that reaches the sealed split"
