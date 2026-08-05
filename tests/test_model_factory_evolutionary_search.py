"""Evolutionary candidate/filter/select/breed workflow tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cognitive_runtime.training.model_factory.search as search_module
from cognitive_runtime.training.model_factory.genome import GENERIC_ACTION_EFFECTS_V1, build_schema
from cognitive_runtime.training.model_factory.promotion import GateResult
from cognitive_runtime.training.model_factory.search import SearchError, run_evolutionary_search
from cognitive_runtime.training.model_factory.spec import resolve


SCHEMA = build_schema("evolutionary-search-test-v1", {
    "optimizer.lr": {
        "type": "float",
        "bounds": (0.001, 0.1),
        "default": 0.01,
        "mutation": {"distribution": "normal_perturb", "sigma": 0.01},
    },
})


def _base_spec():
    return resolve({
        "organism": "EvolutionTest",
        "mode": "fresh",
        "data": {"corpus_id": "fixed-v1", "world": "crafter", "horizons_ticks": [1]},
        "model": {
            "backbone": "transformer",
            "latent_width": 8,
            "hidden_dim": 16,
            "action_embed_dim": 4,
            "reconstruction_size": 4,
            "context_length": 3,
            "backbone_kwargs": {"n_heads": 2, "n_layers": 1},
        },
        "training": {
            "objective": "windowed_rollout",
            "optimizer": {"name": "adamw", "lr": 0.01},
            "batch_size": 4,
            "seed": 0,
            "rollout_frames": 1,
            "warmup_frames": 1,
            "scheduled_sampling_p": 0.0,
            "epoch_budget": 2,
            "device": "cpu",
            "precision": "fp32",
        },
        "evaluation": {"selection_metric": "rollout.t+1.model_over_copy_last_mse"},
    })


def _install_fake_trials(monkeypatch, tmp_path: Path, *, all_fail: bool = False):
    metrics = {
        "evo-p0-c0": 0.10,
        "evo-p0-c1": 0.20,
        "evo-p0-c2": 0.30,
        "evo-p0-c3": 0.40,
        "evo-p1-c1": 0.15,
        "evo-p1-c2": 0.25,
        "evo-p1-c3": 0.35,
    }
    failed_quality = set(metrics) if all_fail else {"evo-p0-c2"}
    recorded_lineages = []

    monkeypatch.setattr(
        search_module,
        "resolve_corpus",
        lambda *args, **kwargs: SimpleNamespace(
            data_contract=SimpleNamespace(ticks_per_frame=1.0, hash="data")
        ),
    )
    monkeypatch.setattr(
        search_module,
        "_resolve_selection_metric",
        lambda evaluation, metric, ticks_per_frame: (evaluation["score"], [evaluation["score"]]),
    )

    def gate(rollout):
        passed = bool(rollout["quality_passed"])
        return GateResult("quality", passed, "quality passed" if passed else "quality failed")

    monkeypatch.setattr(search_module, "_HALVING_GATES", (gate,))
    monkeypatch.setattr(
        search_module,
        "record_breeding_lineage",
        lambda directory, lineage: recorded_lineages.append(lineage),
    )

    def fake_run_trial(spec, *, run_id, root, **kwargs):
        directory = Path(root) / spec.organism / run_id
        (directory / "metrics").mkdir(parents=True)
        (directory / "metrics" / "validation.json").write_text(
            json.dumps({"episode_ids": ["validation-1"]}), encoding="utf-8",
        )
        score = metrics[run_id]
        evaluation = {
            "score": score,
            "rollout": {"quality_passed": run_id not in failed_quality},
        }
        return SimpleNamespace(
            run_id=run_id,
            state="completed",
            directory=directory,
            architecture_hash="architecture",
            data_contract_hash="data",
            checkpoint_path=str(directory / "checkpoints" / "best-validation.pt"),
            checkpoint_sha256=f"sha-{run_id}",
            evaluation=evaluation,
            budget_report={"epochs_recorded": 2},
        )

    monkeypatch.setattr(search_module, "run_trial", fake_run_trial)
    return recorded_lineages


def test_filters_then_keeps_best_half_and_breeds_back_to_population_size(tmp_path, monkeypatch):
    recorded_lineages = _install_fake_trials(monkeypatch, tmp_path)

    report = run_evolutionary_search(
        _base_spec(),
        SCHEMA,
        population_size=4,
        population_count=2,
        seed=9,
        mutation_rate=1.0,
        method="random",
        root=tmp_path / "runs",
        run_id_prefix="evo",
    )

    assert len(report.populations) == 2
    first, second = report.populations
    assert first.survivor_run_ids == ("evo-p0-c0",)
    assert first.offspring_run_ids == ("evo-p1-c1", "evo-p1-c2", "evo-p1-c3")
    assert next(item for item in first.results if item.run_id == "evo-p0-c2").quality_passed is False

    assert len(second.results) == 4
    assert second.results[0].run_id == "evo-p0-c0"
    assert second.results[0].carried is True
    assert second.results[0].epochs_executed == 0
    assert second.survivor_run_ids == ("evo-p0-c0", "evo-p1-c1")
    assert report.final_survivor_run_ids == second.survivor_run_ids
    assert report.best_run_id == "evo-p0-c0"
    assert report.total_trials_started == 7
    assert report.total_epochs_executed == 14
    assert len(recorded_lineages) == 3
    assert all(lineage.parent_a_run_id == "evo-p0-c0" for lineage in recorded_lineages)
    assert all(lineage.parent_b_run_id == "evo-p0-c0" for lineage in recorded_lineages)


def test_zero_quality_survivors_ends_without_breeding(tmp_path, monkeypatch):
    recorded_lineages = _install_fake_trials(monkeypatch, tmp_path, all_fail=True)

    report = run_evolutionary_search(
        _base_spec(), SCHEMA, population_size=4, population_count=3, seed=9,
        method="random", root=tmp_path / "runs", run_id_prefix="evo",
    )

    assert len(report.populations) == 1
    assert report.populations[0].survivor_run_ids == ()
    assert report.populations[0].offspring_run_ids == ()
    assert report.best_run_id is None
    assert recorded_lineages == []


def test_completed_seed_run_enters_population_zero_without_retraining(tmp_path, monkeypatch):
    recorded_lineages = _install_fake_trials(monkeypatch, tmp_path)
    seed_spec = resolve({
        **_base_spec().to_dict(),
        "evolution": {
            "generation": 3,
            "configuration_parents": ["old-a", "old-b"],
            "weight_donor": "old-a",
            "genome_operator": "uniform_crossover_v1",
            "mutations": [],
        },
    })
    seed_parent = search_module.ParentRecord(
        run_id="prior-champion",
        genome_schema_version=SCHEMA.version,
        architecture_hash="architecture",
        data_contract_hash="data",
        corpus_id="fixed-v1",
        tier="evolutionary-search",
        evaluation_contract=dict(seed_spec.evaluation),
        genome={"optimizer.lr": seed_spec.training["optimizer"]["lr"]},
        checkpoint_name="best-validation.pt",
        checkpoint_sha256="seed-sha",
        spec=seed_spec,
    )
    seed_result = search_module.EvolutionCandidateResult(
        candidate_index=0,
        run_id="prior-champion",
        origin_population=0,
        mode="clone",
        state="completed",
        epochs_executed=0,
        metric_value=0.05,
        gates=({"name": "quality", "passed": True, "reason": "quality passed"},),
        quality_passed=True,
        retained=False,
        carried=True,
        configuration_parents=("old-a", "old-b"),
        reason="seeded",
    )
    monkeypatch.setattr(
        search_module,
        "_load_seed_member",
        lambda run_id, **kwargs: search_module._EvolutionMember(
            spec=seed_spec,
            origin_population=0,
            evaluation=seed_result,
            parent_record=seed_parent,
            validation_episode_ids=("validation-1",),
            seeded=True,
        ),
    )

    report = run_evolutionary_search(
        _base_spec(), SCHEMA, population_size=4, population_count=2, seed=9,
        method="random", mutation_rate=1.0, root=tmp_path / "runs", run_id_prefix="evo",
        seed_run_ids=("prior-champion",),
    )

    assert report.seed_run_ids == ("prior-champion",)
    assert report.populations[0].results[0].run_id == "prior-champion"
    assert report.populations[0].results[0].carried is True
    assert report.populations[0].results[0].epochs_executed == 0
    assert report.populations[0].survivor_run_ids == ("prior-champion",)
    assert report.total_trials_started == 6
    assert all(lineage.parent_a_run_id == "prior-champion" for lineage in recorded_lineages)
    assert all(lineage.parent_b_run_id == "prior-champion" for lineage in recorded_lineages)
    assert all(lineage.generation == 4 for lineage in recorded_lineages)


def test_evolutionary_search_rejects_more_seed_runs_than_population_slots():
    with pytest.raises(SearchError, match="cannot exceed population_size"):
        run_evolutionary_search(
            _base_spec(), SCHEMA, population_size=2, population_count=1, seed=1,
            seed_run_ids=("a", "b", "c"),
        )


def test_initial_population_resamples_windows_that_do_not_fit_shortest_episode():
    candidates = search_module._propose_valid_population(
        _base_spec(),
        GENERIC_ACTION_EFFECTS_V1,
        population_size=8,
        seed=7,
        method="lhs",
        min_episode_frames=17,
        stage_budget_seconds=None,
        cost_model=search_module.project_cost_seconds,
    )

    assert len(candidates) == 8
    assert all(
        candidate.training["warmup_frames"] + candidate.training["rollout_frames"] < 17
        for candidate in candidates
    )


@pytest.mark.parametrize("population_size,population_count", [(1, 1), (4, 0)])
def test_evolutionary_search_rejects_invalid_population_configuration(
    population_size, population_count,
):
    with pytest.raises(SearchError):
        run_evolutionary_search(
            _base_spec(), SCHEMA, population_size, population_count, seed=1,
        )
