"""Checkpoint-based successive halving tests (epic #212, Phase D step 2,
§13.2, issue #231).

Most scenarios train a real (tiny) cortex against real (tiny) recorded
Crafter sessions -- seconds, not minutes -- mirroring
``tests/test_model_factory_runner.py``/``test_model_factory_confirmation.py``:
the checkpoint continuation across rungs is the actual integration this
issue is about, not a stand-in for it. Where a specific rank/gate/elimination
outcome needs to be deterministic rather than dependent on how well a couple
of epochs of toy training happen to go, ``run_trial``/``_resolve_selection_metric``
are wrapped to report controlled values for named ``run_id``s -- the real
training and checkpointing still happen underneath -- rather than mocking
the trainer outright.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

torch = pytest.importorskip("torch")

import cognitive_runtime.training.model_factory.search as search_module  # noqa: E402
from cognitive_runtime.training.model_factory.corpus import build_corpus  # noqa: E402
from cognitive_runtime.training.model_factory.genome import build_schema  # noqa: E402
from cognitive_runtime.training.model_factory.promotion import GateResult  # noqa: E402
from cognitive_runtime.training.model_factory.runner import run_trial  # noqa: E402
from cognitive_runtime.training.model_factory.search import (  # noqa: E402
    SearchError,
    SuccessiveHalvingReport,
    run_successive_halving,
)
from cognitive_runtime.training.model_factory.spec import resolve  # noqa: E402

ORGANISM = "Test"
REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA = build_schema("successive-halving-smoke-v1", {
    "optimizer.lr": {
        "type": "float", "bounds": (1e-3, 5e-2), "log_scale": True, "default": 0.02,
        "mutation": {"distribution": "log_normal_perturb", "sigma": 0.3},
    },
    "scheduled_sampling_p": {
        "type": "float", "bounds": (0.0, 0.3), "default": 0.0,
        "mutation": {"distribution": "normal_perturb", "sigma": 0.05},
    },
})


def _build_corpus(root: Path, corpus_id: str = "crafter-tiny-v1"):
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
        "splits": {
            "train": {"walk_forward_short": [0, 1]},
            "validation": {"walk_forward_short": [100, 101]},
            "test": {"walk_forward_short": [200, 201]},
        },
    }
    return build_corpus(spec)


def _base_spec(corpus_id, *, epoch_budget=None):
    raw = {
        "organism": ORGANISM,
        "mode": "fresh",
        "data": {"corpus_id": corpus_id, "world": "crafter", "horizons_ticks": [1, 2]},
        "model": {
            "backbone": "transformer", "latent_width": 8, "hidden_dim": 16,
            "action_embed_dim": 4, "reconstruction_size": 4, "context_length": 3,
            "backbone_kwargs": {"n_heads": 2, "n_layers": 1},
        },
        "training": {
            "objective": "windowed_rollout",
            "optimizer": {"name": "adamw", "lr": 0.02},
            "batch_size": 4,
            "seed": 0,
            "rollout_frames": 2,
            "warmup_frames": 2,
            "scheduled_sampling_p": 0.0,
            "epoch_budget": epoch_budget,
            "device": "cpu",
            "precision": "fp32",
        },
        "evaluation": {"selection_metric": "rollout.t+1.model_over_copy_last_mse"},
    }
    return resolve(raw)


@pytest.fixture()
def corpus(tmp_path):
    return _build_corpus(tmp_path / "corpora")


def _run_halving(base_spec, tmp_path, *, budgets=(1, 2, 3), n=4, seed=1, **kwargs):
    kwargs.setdefault("run_id_prefix", "sh")
    return run_successive_halving(
        base_spec, SCHEMA, budgets, n, seed,
        root=str(tmp_path / "runs"), corpus_root=str(tmp_path / "corpora"), **kwargs,
    )


def _install_controls(
    monkeypatch,
    *,
    metric_overrides: Optional[Dict[str, float]] = None,
    gate_overrides: Optional[Dict[str, bool]] = None,
    state_overrides: Optional[Dict[str, str]] = None,
):
    """Wrap ``search_module.run_trial`` so specific ``run_id``s report
    controlled primary-metric/gate/state outcomes, while every actual
    training/checkpointing call underneath is the real thing.

    Keyed by object identity of the real (per-call) ``evaluation``/
    ``evaluation["rollout"]`` dicts rather than by re-deriving them, so the
    fakes never have to know the real horizon/stat keys the real evaluator
    used.
    """
    metric_overrides = metric_overrides or {}
    gate_overrides = gate_overrides or {}
    state_overrides = state_overrides or {}

    real_run_trial = search_module.run_trial
    real_beats_copy_last, real_health = search_module._HALVING_GATES
    metric_by_eval: Dict[int, float] = {}
    gate_by_rollout: Dict[int, bool] = {}

    def _wrapped_run_trial(spec, *, run_id=None, **kwargs):
        result = real_run_trial(spec, run_id=run_id, **kwargs)
        if run_id in metric_overrides and result.evaluation is not None:
            metric_by_eval[id(result.evaluation)] = metric_overrides[run_id]
        if run_id in gate_overrides and result.evaluation is not None:
            gate_by_rollout[id(result.evaluation["rollout"])] = gate_overrides[run_id]
        if run_id in state_overrides:
            result = dataclasses.replace(result, state=state_overrides[run_id])
        return result

    def _wrapped_resolve(eval_result, metric, ticks_per_frame):
        if id(eval_result) in metric_by_eval:
            value = metric_by_eval[id(eval_result)]
            return value, [value]
        return real_resolve(eval_result, metric, ticks_per_frame)

    def _fake_beats_copy_last(rollout_metrics):
        forced = gate_by_rollout.get(id(rollout_metrics))
        if forced is None:
            return real_beats_copy_last(rollout_metrics)
        return GateResult("rollout_beats_copy_last", True, "stub: forced pass")

    def _fake_health(rollout_metrics):
        forced = gate_by_rollout.get(id(rollout_metrics))
        if forced is None:
            return real_health(rollout_metrics)
        reason = "stub: forced pass" if forced else "stub: forced failure"
        return GateResult("rollout_health", forced, reason)

    real_resolve = search_module._resolve_selection_metric
    monkeypatch.setattr(search_module, "run_trial", _wrapped_run_trial)
    monkeypatch.setattr(search_module, "_resolve_selection_metric", _wrapped_resolve)
    monkeypatch.setattr(search_module, "_HALVING_GATES", (_fake_beats_copy_last, _fake_health))


def _permissive_controls(monkeypatch, run_ids):
    """Force every gate to pass for ``run_ids`` -- used by tests that only
    care about continuation/epoch bookkeeping, not real gate outcomes (a
    couple of epochs of tiny toy training frequently trips the rollout
    health gate, which is not what these tests are checking)."""
    _install_controls(monkeypatch, gate_overrides={run_id: True for run_id in run_ids})


def _expected_run_ids(budgets, n):
    return [f"sh-r{rung}-c{index}" for rung in range(len(budgets)) for index in range(n)]


# --------------------------------------------------------------------------- validation


def test_budgets_must_declare_at_least_one_rung(tmp_path, corpus):
    with pytest.raises(SearchError, match="at least one"):
        _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=())


def test_budgets_must_be_positive(tmp_path, corpus):
    with pytest.raises(SearchError, match="positive"):
        _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=(0, 2))


def test_budgets_must_be_strictly_increasing(tmp_path, corpus):
    with pytest.raises(SearchError, match="strictly increasing"):
        _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=(2, 2, 4))
    with pytest.raises(SearchError, match="strictly increasing"):
        _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=(4, 2))


def test_halving_factor_must_be_at_least_two(tmp_path, corpus):
    with pytest.raises(SearchError, match="halving_factor"):
        _run_halving(_base_spec(corpus.corpus_id), tmp_path, halving_factor=1)


# --------------------------------------------------------------------------- AC: resumes, not from scratch


def test_survivor_resumes_from_checkpoint_and_epochs_match_the_halving_schedule(tmp_path, corpus, monkeypatch):
    """AC: a survivor advancing between rungs resumes from its checkpoint;
    total executed epochs equals the halving schedule's theoretical cost,
    not the from-scratch cost."""
    budgets = (1, 2, 4)
    n = 4
    _permissive_controls(monkeypatch, _expected_run_ids(budgets, n))

    report = _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=1)

    assert isinstance(report, SuccessiveHalvingReport)
    assert report.budgets == budgets
    # 1*4 (rung0) + 1*2 (rung1: delta=1, 2 survivors) + 2*1 (rung2: delta=2, 1 survivor)
    assert report.theoretical_halving_epochs == 4 * 1 + 2 * 1 + 1 * 2
    assert report.total_epochs_executed == report.theoretical_halving_epochs
    # From-scratch would retrain every surviving lineage to the *full* rung
    # budget every time: 4*1 + 2*2 + 1*4 -- strictly more than the halving cost.
    assert report.from_scratch_epochs == 4 * 1 + 2 * 2 + 1 * 4
    assert report.total_epochs_executed < report.from_scratch_epochs

    rung0, rung1, rung2 = report.rungs
    assert [result.mode for result in rung0.results] == ["fresh"] * 4
    assert {result.mode for result in rung1.results} == {"fine_tune"}
    assert {result.mode for result in rung2.results} == {"fine_tune"}
    assert [result.epochs_executed for result in rung0.results] == [1, 1, 1, 1]
    assert [result.epochs_executed for result in rung1.results] == [1, 1]
    assert [result.epochs_executed for result in rung2.results] == [2]

    # The strongest possible proof of genuine continuation (not a
    # re-trained-from-scratch lookalike): each fine_tune rung's own
    # comparison.json names the *specific* prior rung's run_id as its
    # parent, and its checkpoint lineage differs from a scratch run's.
    surviving_index = rung2.results[0].candidate_index
    rung1_run_id = next(r.run_id for r in rung1.results if r.candidate_index == surviving_index)
    rung2_run_id = rung2.results[0].run_id
    comparison_path = tmp_path / "runs" / ORGANISM / rung2_run_id / "metrics" / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["parent_run_id"] == rung1_run_id

    rung0_run_id = next(r.run_id for r in rung0.results if r.candidate_index == surviving_index)
    rung1_comparison = json.loads(
        (tmp_path / "runs" / ORGANISM / rung1_run_id / "metrics" / "comparison.json").read_text(encoding="utf-8")
    )
    assert rung1_comparison["parent_run_id"] == rung0_run_id


def test_total_epochs_never_exceed_the_declared_budget(tmp_path, corpus, monkeypatch):
    """AC: total epochs and total wall-clock never exceed the declared budget."""
    budgets = (1, 2, 3)
    n = 4
    _permissive_controls(monkeypatch, _expected_run_ids(budgets, n))

    report = _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=2)

    assert report.total_epochs_executed <= report.theoretical_halving_epochs
    for rung in report.rungs:
        for result in rung.results:
            assert result.epochs_executed <= rung.delta_epochs
            budget_report_path = (
                tmp_path / "runs" / ORGANISM / result.run_id / "metrics" / "budget_report.json"
            )
            budget_report = json.loads(budget_report_path.read_text(encoding="utf-8"))
            assert budget_report["measured_training_seconds"] <= budget_report["max_training_seconds"]


# --------------------------------------------------------------------------- AC: gate elimination


def test_gate_failure_eliminates_a_candidate_regardless_of_its_primary_metric(tmp_path, corpus, monkeypatch):
    """AC: a candidate failing a safety gate at a rung is eliminated even
    with the best primary metric."""
    budgets = (1, 2)
    n = 4
    rung0_ids = [f"sh-r0-c{i}" for i in range(n)]
    # Candidate 0 has the best (lowest) metric at rung 0 but fails the
    # health gate; every other candidate passes both gates with a worse
    # metric. Candidate 0 must not survive despite winning on the metric.
    _install_controls(
        monkeypatch,
        metric_overrides={rung0_ids[0]: 0.01, rung0_ids[1]: 5.0, rung0_ids[2]: 6.0, rung0_ids[3]: 7.0},
        gate_overrides={rung0_ids[0]: False, rung0_ids[1]: True, rung0_ids[2]: True, rung0_ids[3]: True},
    )

    report = _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=3)

    rung0 = report.rungs[0]
    eliminated = next(r for r in rung0.results if r.run_id == rung0_ids[0])
    assert eliminated.gate_passed is False
    assert eliminated.advanced is False
    assert eliminated.metric_value == pytest.approx(0.01)
    # The best-metric-but-gate-failing candidate is not among the survivors,
    # even though (n // 2 == 2) survivor slots were available.
    assert eliminated.candidate_index not in rung0.survivor_indices
    survivors_by_metric = sorted(
        (r for r in rung0.results if r.advanced), key=lambda r: r.metric_value,
    )
    assert [r.run_id for r in survivors_by_metric] == [rung0_ids[1], rung0_ids[2]]


def test_a_candidate_that_exceeds_its_training_time_budget_is_eliminated(tmp_path, corpus, monkeypatch):
    budgets = (1, 2)
    n = 4
    rung0_ids = [f"sh-r0-c{i}" for i in range(n)]
    _install_controls(
        monkeypatch,
        state_overrides={rung0_ids[0]: "budget_exceeded"},
        gate_overrides={run_id: True for run_id in _expected_run_ids(budgets, n)},
    )

    report = _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=4)

    rung0 = report.rungs[0]
    truncated = next(r for r in rung0.results if r.run_id == rung0_ids[0])
    assert truncated.state == "budget_exceeded"
    assert truncated.gate_passed is False
    assert truncated.advanced is False
    assert truncated.metric_value is None
    assert "did not complete" in truncated.reason


# --------------------------------------------------------------------------- AC: deterministic elimination order


def test_ties_are_broken_by_run_id_deterministically(tmp_path, corpus, monkeypatch):
    budgets = (1, 2)
    n = 4
    rung0_ids = [f"sh-r0-c{i}" for i in range(n)]
    # A three-way exact tie among candidates 1, 2, 3; candidate 0 is
    # decisively worse. Only two of the tied three can survive
    # (n // 2 == 2), so the tie-break must be exercised.
    _install_controls(
        monkeypatch,
        metric_overrides={rung0_ids[0]: 9.0, rung0_ids[1]: 1.0, rung0_ids[2]: 1.0, rung0_ids[3]: 1.0},
        gate_overrides={run_id: True for run_id in _expected_run_ids(budgets, n)},
    )

    report = _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=5)

    rung0 = report.rungs[0]
    # Lexicographically smallest run_ids among the tied candidates win --
    # "...-c1" < "...-c2" < "...-c3".
    assert sorted(rung0.survivor_indices) == [1, 2]


def test_elimination_order_is_reproducible_for_a_fixed_seed(tmp_path, corpus, monkeypatch):
    budgets = (1, 2)
    n = 4
    _permissive_controls(monkeypatch, _expected_run_ids(budgets, n))
    corpus_root = str(tmp_path / "corpora")

    report_a = run_successive_halving(
        _base_spec(corpus.corpus_id), SCHEMA, budgets, n, 6,
        root=str(tmp_path / "runs_a"), corpus_root=corpus_root, run_id_prefix="sh",
    )
    report_b = run_successive_halving(
        _base_spec(corpus.corpus_id), SCHEMA, budgets, n, 6,
        root=str(tmp_path / "runs_b"), corpus_root=corpus_root, run_id_prefix="sh",
    )

    ids_a = [[result.run_id for result in rung.results] for rung in report_a.rungs]
    ids_b = [[result.run_id for result in rung.results] for rung in report_b.rungs]
    assert ids_a == ids_b
    survivors_a = [rung.survivor_indices for rung in report_a.rungs]
    survivors_b = [rung.survivor_indices for rung in report_b.rungs]
    assert survivors_a == survivors_b
    assert report_a.final_survivor_run_id == report_b.final_survivor_run_id


# --------------------------------------------------------------------------- AC: identical validation set


def test_every_rung_evaluates_the_identical_validation_episode_set(tmp_path, corpus, monkeypatch):
    budgets = (1, 2, 3)
    n = 4
    _permissive_controls(monkeypatch, _expected_run_ids(budgets, n))

    report = _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=7)

    episode_sets = set()
    for rung in report.rungs:
        for result in rung.results:
            validation_path = tmp_path / "runs" / ORGANISM / result.run_id / "metrics" / "validation.json"
            payload = json.loads(validation_path.read_text(encoding="utf-8"))
            episode_sets.add(tuple(payload["episode_ids"]))
    assert len(episode_sets) == 1
    (only_set,) = episode_sets
    assert len(only_set) == 2  # two frozen validation sessions in the fixture


def test_a_mismatched_validation_episode_set_is_rejected(tmp_path, corpus, monkeypatch):
    """A defensive regression guard: if a later rung's validation.json ever
    named a different episode set (a corpus-resolution bug, a caller mixing
    corpora across rungs), successive halving must refuse to compare
    apples to oranges rather than silently ranking on mismatched evidence."""
    budgets = (1, 2)
    n = 2
    run_ids = _expected_run_ids(budgets, n)
    _permissive_controls(monkeypatch, run_ids)

    real_run_trial = search_module.run_trial
    rung1_run_id = "sh-r1-c0"

    def _tampering_run_trial(spec, *, run_id=None, **kwargs):
        result = real_run_trial(spec, run_id=run_id, **kwargs)
        if run_id == rung1_run_id:
            validation_path = Path(result.directory) / "metrics" / "validation.json"
            payload = json.loads(validation_path.read_text(encoding="utf-8"))
            payload["episode_ids"] = ["not-a-real-episode"]
            validation_path.write_text(json.dumps(payload), encoding="utf-8")
        return result

    monkeypatch.setattr(search_module, "run_trial", _tampering_run_trial)

    with pytest.raises(SearchError, match="identical validation episode set"):
        _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=8)


# --------------------------------------------------------------------------- AC: sealed test isolation


def test_sealed_test_split_is_never_queried_during_halving(tmp_path, corpus, monkeypatch):
    from cognitive_runtime.training import action_world_model as awm_module

    test_paths = {entry["session_path"] for entry in corpus.manifest["sessions"]["test"]}
    assert test_paths, "fixture must actually declare sealed test sessions"

    budgets = (1, 2)
    n = 3
    _permissive_controls(monkeypatch, _expected_run_ids(budgets, n))

    opened_paths: list[str] = []
    original = awm_module.build_action_sequence_dataset

    def _tracking(session_paths, *args, **kwargs):
        opened_paths.extend(session_paths)
        return original(session_paths, *args, **kwargs)

    monkeypatch.setattr(awm_module, "build_action_sequence_dataset", _tracking)

    _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=9)

    assert not (set(opened_paths) & test_paths), "successive halving must never open the sealed test partition"


# --------------------------------------------------------------------------- champion comparison


def test_final_survivor_is_compared_to_a_supplied_champion(tmp_path, corpus, monkeypatch):
    champion = run_trial(
        _base_spec(corpus.corpus_id, epoch_budget=1),
        root=str(tmp_path / "runs"), corpus_root=str(tmp_path / "corpora"), run_id="champion-run",
    )

    budgets = (1, 2)
    n = 2
    _permissive_controls(monkeypatch, _expected_run_ids(budgets, n))

    report = _run_halving(
        _base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=10,
        champion_run_id=champion.run_id,
    )

    assert report.final_survivor_run_id is not None
    assert report.champion_comparison is not None
    assert report.champion_comparison["champion_run_id"] == champion.run_id
    assert report.champion_comparison["decision"] in ("candidate_improves", "hold")
    assert report.champion_comparison["episode_count"] == 2


def test_champion_comparison_is_none_when_no_champion_is_supplied(tmp_path, corpus, monkeypatch):
    budgets = (1, 2)
    n = 2
    _permissive_controls(monkeypatch, _expected_run_ids(budgets, n))

    report = _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=11)

    assert report.champion_comparison is None


def test_no_survivor_when_every_candidate_is_gate_eliminated(tmp_path, corpus, monkeypatch):
    budgets = (1, 2)
    n = 2
    run_ids = _expected_run_ids(budgets, n)
    _install_controls(monkeypatch, gate_overrides={run_id: False for run_id in run_ids[:n]})

    report = _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=12)

    assert report.final_survivor_run_id is None
    assert report.final_survivor_metric_value is None
    assert len(report.rungs) == 1  # nobody survived to be trained at rung 1
    assert report.champion_comparison is None


def test_last_rung_never_marks_multiple_candidates_as_advanced(tmp_path, corpus, monkeypatch):
    """Regression (Codex review): the last rung has no next budget to
    advance to, so RungReport.survivor_indices/advanced must stay empty
    there even when several candidates pass both gates -- final_survivor_run_id
    (picked by primary metric, ties broken by run_id) is the only record of
    which one is the campaign's winner."""
    budgets = (1, 2)
    n = 4
    rung0_ids = [f"sh-r0-c{i}" for i in range(n)]
    rung1_ids = [f"sh-r1-c{i}" for i in range(n)]
    _install_controls(
        monkeypatch,
        # Candidates 1 and 2 have the best rung-0 metric (guaranteeing both
        # are among the top (4 // halving_factor == 2) kept) and both pass
        # rung 1's gates too, with candidate 1 the best there.
        gate_overrides={run_id: True for run_id in rung0_ids + rung1_ids},
        metric_overrides={
            rung0_ids[0]: 10.0, rung0_ids[1]: 1.0, rung0_ids[2]: 2.0, rung0_ids[3]: 11.0,
            rung1_ids[1]: 1.0, rung1_ids[2]: 2.0,
        },
    )

    report = _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=14)

    rung1 = report.rungs[1]
    assert rung1.rung_index == 1
    assert rung1.survivor_indices == ()
    assert all(result.advanced is False for result in rung1.results)
    gate_passing_indices = {result.candidate_index for result in rung1.results if result.gate_passed}
    assert gate_passing_indices == {1, 2}
    assert report.final_survivor_run_id == rung1_ids[1]
    assert report.final_survivor_metric_value == pytest.approx(1.0)


def test_stage_budget_seconds_caps_the_whole_campaign(tmp_path, corpus, monkeypatch):
    """AC/Codex review: stage_budget_seconds is a campaign-wide wall-clock
    cap, not merely a per-genome cost-projection hint passed to propose() --
    once it is spent, no further run_trial call is ever launched, in this
    rung or any later one."""
    real_run_trial = search_module.run_trial
    calls = []

    def _counting_run_trial(spec, *, run_id=None, **kwargs):
        calls.append(run_id)
        return real_run_trial(spec, run_id=run_id, **kwargs)

    monkeypatch.setattr(search_module, "run_trial", _counting_run_trial)

    budgets = (1, 2)
    n = 3
    report = _run_halving(
        _base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=15,
        stage_budget_seconds=0.0,
    )

    assert calls == []  # the cap was already spent before the very first candidate
    assert report.stage_budget_exceeded is True
    assert len(report.rungs) == 1
    assert [result.state for result in report.rungs[0].results] == ["not_attempted"] * n
    assert all(result.epochs_executed == 0 for result in report.rungs[0].results)
    assert all(result.gate_passed is False for result in report.rungs[0].results)
    assert report.total_epochs_executed == 0
    assert report.final_survivor_run_id is None


# --------------------------------------------------------------------------- report shape


def test_report_to_dict_round_trips_json(tmp_path, corpus, monkeypatch):
    budgets = (1, 2)
    n = 2
    _permissive_controls(monkeypatch, _expected_run_ids(budgets, n))

    report = _run_halving(_base_spec(corpus.corpus_id), tmp_path, budgets=budgets, n=n, seed=13)

    payload = json.dumps(report.to_dict())
    reloaded = json.loads(payload)
    assert reloaded["format"] == search_module.SUCCESSIVE_HALVING_FORMAT
    assert reloaded["budgets"] == list(budgets)
    assert len(reloaded["rungs"]) == 2


# --------------------------------------------------------------------------- torch-free import


def test_search_module_imports_cleanly_without_torch():
    script = """
import sys

class _BlockTorch:
    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is deliberately blocked for this test")
        return None

sys.meta_path.insert(0, _BlockTorch())
sys.path.insert(0, {path!r})
import cognitive_runtime.training.model_factory.search  # noqa: F401
print("OK")
""".format(path=str(REPO_ROOT))
    result = subprocess.run(
        [sys.executable, "-c", script], env=dict(os.environ), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
