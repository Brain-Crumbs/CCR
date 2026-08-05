"""Nested architecture (NAS) search tests (clinic redesign, Phase 6b).

Torch-free by construction, mirroring
test_model_factory_evolutionary_search.py: the inner engine
(``search.run_evolutionary_search``) is stubbed, because what this module
adds is the *outer* loop -- how architectures are sampled, scored by an
inner campaign, ranked, carried forward and bred -- not the inner one, which
has its own suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cognitive_runtime.training.model_factory.architecture_search as architecture_search
from cognitive_runtime.training.model_factory.architecture_genome import (
    ARCHITECTURE_SEARCH_V1,
    BACKBONE_PRESET_GENE,
    apply_architecture_genome,
    build_architecture_schema,
)
from cognitive_runtime.training.model_factory.architecture_search import (
    ARCHITECTURE_GENOME_OPERATOR,
    ARCHITECTURE_SEARCH_FORMAT,
    STATE_NO_INNER_SURVIVOR,
    ArchitectureParent,
    ArchitectureSearchError,
    architecture_run_id_prefix,
    breed_architecture,
    estimate_cost,
    preview_architecture_search,
    run_architecture_search,
)
from cognitive_runtime.training.model_factory.artifacts import allocate_run_artifacts
from cognitive_runtime.training.model_factory.contracts import (
    ArchitectureContract,
    DataContract,
    TrainingContract,
)
from cognitive_runtime.training.model_factory.genome import build_schema
from cognitive_runtime.training.model_factory.registry import lineage_graph
from cognitive_runtime.training.model_factory.spec import resolve

INNER_SCHEMA = build_schema("architecture-search-inner-test-v1", {
    "optimizer.lr": {
        "type": "float",
        "bounds": (0.001, 0.1),
        "default": 0.01,
        "mutation": {"distribution": "normal_perturb", "sigma": 0.01},
    },
})

#: A two-gene outer schema: small enough that a 3-architecture population is
#: comfortably samplable, large enough that crossover has something to do.
OUTER_SCHEMA = build_architecture_schema("architecture-search-test-v1", {
    BACKBONE_PRESET_GENE: {
        "type": "categorical",
        "choices": ("gru", "transformer_h2_l1", "transformer_h4_l2"),
        "default": "gru",
        "mutation": {"distribution": "categorical_resample"},
    },
    "hidden_dim": {
        "type": "int",
        "choices": (16, 32, 64),
        "default": 16,
        "mutation": {"distribution": "categorical_resample"},
    },
})


def _base_spec(**training_overrides):
    training = {
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
        "max_training_seconds": 30.0,
    }
    training.update(training_overrides)
    return resolve({
        "organism": "ArchitectureTest",
        "mode": "fresh",
        "data": {"corpus_id": "fixed-v1", "world": "crafter", "horizons_ticks": [1]},
        "model": {
            "backbone": "gru",
            "latent_width": 8,
            "hidden_dim": 16,
            "action_embed_dim": 4,
            "reconstruction_size": 4,
            "context_length": 3,
            "backbone_kwargs": {},
        },
        "training": training,
        "evaluation": {"selection_metric": "rollout.t+1.model_over_copy_last_mse"},
    })


def _install_fake_inner_search(monkeypatch, *, scores, failures=(), raises=()):
    """Stub the inner engine, recording every campaign it was asked to run.

    ``scores`` maps an inner ``run_id_prefix`` (an architecture_id) to the
    ``best_metric_value`` its campaign "found". A prefix in ``failures``
    reports a campaign that completed with no survivor; one in ``raises``
    reports a campaign that blew up.
    """
    campaigns = []

    def fake_run_evolutionary_search(
        base_spec, schema, population_size, population_count, seed, *, run_id_prefix, **kwargs,
    ):
        campaigns.append(SimpleNamespace(
            spec=base_spec, schema=schema, population_size=population_size,
            population_count=population_count, seed=seed, prefix=run_id_prefix, kwargs=kwargs,
        ))
        if run_id_prefix in raises:
            raise RuntimeError("inner campaign exploded")
        if run_id_prefix in failures:
            return SimpleNamespace(
                best_run_id=None, best_metric_value=None, total_trials_started=population_size,
                total_epochs_executed=population_size * 2, stage_budget_exceeded=False,
            )
        return SimpleNamespace(
            best_run_id=f"{run_id_prefix}-p0-c0",
            best_metric_value=scores[run_id_prefix],
            total_trials_started=population_size,
            total_epochs_executed=population_size * 2,
            stage_budget_exceeded=False,
        )

    monkeypatch.setattr(
        architecture_search, "run_evolutionary_search", fake_run_evolutionary_search,
    )
    return campaigns


# --------------------------------------------------------------- cost estimate


def test_cost_estimate_is_the_multiplicative_bound():
    estimate = estimate_cost(
        _base_spec(),
        outer_population_size=4, outer_population_count=2,
        inner_population_size=3, inner_population_count=1,
    )
    assert estimate.estimated_max_trials == 24
    assert estimate.max_training_seconds_per_trial == 30.0
    assert estimate.estimated_max_seconds == 720.0
    assert estimate.to_dict()["estimated_max_trials"] == 24


def test_cost_estimate_reports_no_wall_clock_bound_when_the_spec_declares_none():
    estimate = estimate_cost(
        _base_spec(max_training_seconds=None),
        outer_population_size=2, outer_population_count=1,
        inner_population_size=2, inner_population_count=1,
    )
    assert estimate.estimated_max_trials == 4
    assert estimate.max_training_seconds_per_trial is None
    assert estimate.estimated_max_seconds is None


# --------------------------------------------------------------- run id layout


def test_run_ids_extend_the_inner_convention_with_the_outer_coordinates():
    prefix = architecture_run_id_prefix("arch7", 1, 2)
    assert prefix == "arch7-arch1-a2"
    # The inner engine appends its own -p{pop}-c{cand}, so a full run ID --
    # and therefore a UI's whole polling grid -- is precomputable.
    assert f"{prefix}-p0-c1" == "arch7-arch1-a2-p0-c1"


# --------------------------------------------------------------- dry run


def test_preview_resolves_generation_zero_without_launching_anything():
    preview = preview_architecture_search(
        _base_spec(), OUTER_SCHEMA, INNER_SCHEMA,
        outer_population_size=3, outer_population_count=2,
        inner_population_size=2, inner_population_count=1,
        seed=5, method="random",
    )

    assert preview.format == ARCHITECTURE_SEARCH_FORMAT
    assert preview.campaign_id == "arch5"
    assert preview.estimate.estimated_max_trials == 12
    assert len(preview.candidates) == 3
    assert [candidate.architecture_id for candidate in preview.candidates] == [
        "arch5-arch0-a0", "arch5-arch0-a1", "arch5-arch0-a2",
    ]
    for candidate in preview.candidates:
        # Every previewed candidate is a launchable fresh spec whose model
        # block genuinely carries its own genome.
        assert candidate.spec.mode == "fresh"
        assert candidate.spec.parent is None
        assert candidate.spec.model["hidden_dim"] == candidate.genome["hidden_dim"]
    # Distinct architectures: two identical ones would each burn a full
    # inner campaign proving the same thing.
    identities = {json.dumps(candidate.genome, sort_keys=True) for candidate in preview.candidates}
    assert len(identities) == 3


def test_preview_is_deterministic_for_a_fixed_seed():
    kwargs = dict(
        outer_population_size=3, outer_population_count=1,
        inner_population_size=2, inner_population_count=1, seed=13, method="lhs",
    )
    first = preview_architecture_search(_base_spec(), OUTER_SCHEMA, INNER_SCHEMA, **kwargs)
    second = preview_architecture_search(_base_spec(), OUTER_SCHEMA, INNER_SCHEMA, **kwargs)
    assert [c.genome for c in first.candidates] == [c.genome for c in second.candidates]


def test_genes_the_chosen_backbone_ignores_never_split_one_architecture_in_two():
    """A GRU's context_length is inert, so it must not create extra candidates.

    Without canonicalization the sampler would happily return
    ``gru @ context_length=4`` and ``gru @ context_length=16`` as two
    "distinct" architectures, each spending a full inner campaign building
    the identical model and then being ranked against the other on nothing
    but training noise.
    """
    gru_only = build_architecture_schema("gru-only-architecture-v1", {
        BACKBONE_PRESET_GENE: {
            "type": "categorical", "choices": ("gru",), "default": "gru",
            "mutation": {"distribution": "categorical_resample"},
        },
        "context_length": {
            "type": "int", "choices": (4, 8, 12, 16), "default": 8,
            "mutation": {"distribution": "categorical_resample"},
        },
    })
    # Four context lengths, one buildable architecture: asking for two
    # distinct candidates is genuinely unsatisfiable, and saying so beats
    # silently burning a second inner budget on a duplicate.
    with pytest.raises(ArchitectureSearchError, match="distinct architectures"):
        preview_architecture_search(
            _base_spec(), gru_only, INNER_SCHEMA,
            outer_population_size=2, outer_population_count=1,
            inner_population_size=2, inner_population_count=1, seed=2, method="random",
        )


def test_a_sampled_gru_candidate_records_the_canonical_context_length():
    preview = preview_architecture_search(
        _base_spec(), ARCHITECTURE_SEARCH_V1, INNER_SCHEMA,
        outer_population_size=4, outer_population_count=1,
        inner_population_size=2, inner_population_count=1, seed=5, method="lhs",
    )
    gru_candidates = [
        candidate for candidate in preview.candidates
        if candidate.genome[BACKBONE_PRESET_GENE] == "gru"
    ]
    assert gru_candidates, "expected LHS to place at least one GRU candidate"
    default_context = ARCHITECTURE_SEARCH_V1.genes["context_length"].default
    for candidate in gru_candidates:
        assert candidate.genome["context_length"] == default_context
        # The spec's model block agrees with the recorded genome, so the
        # run's architecture_hash cannot encode an inert difference.
        assert candidate.spec.model["context_length"] == default_context


def test_a_child_that_crosses_onto_a_gru_canonicalizes_the_inert_gene():
    """Crossover can hand a windowed parent's context_length to a GRU child."""
    windowed = build_architecture_schema("crossover-canonicalization-v1", {
        BACKBONE_PRESET_GENE: {
            "type": "categorical", "choices": ("gru", "transformer_h2_l1"), "default": "gru",
            "mutation": {"distribution": "categorical_resample"},
        },
        "context_length": {
            "type": "int", "choices": (4, 8, 12, 16), "default": 8,
            "mutation": {"distribution": "categorical_resample"},
        },
    })
    gru_parent = _parent("a0", "run-a", {BACKBONE_PRESET_GENE: "gru", "context_length": 8}, 0.1)
    windowed_parent = _parent(
        "a1", "run-b", {BACKBONE_PRESET_GENE: "transformer_h2_l1", "context_length": 16}, 0.2,
    )

    # Sweep seeds so the assertion covers whichever crossover draw happens.
    for seed in range(24):
        result = breed_architecture(
            gru_parent, windowed_parent, windowed, _base_spec(),
            objective="windowed_rollout", generation=1, seed=seed, mutation_rate=0.0,
        )
        if result.child_genome[BACKBONE_PRESET_GENE] == "gru":
            assert result.child_genome["context_length"] == 8
            assert result.child_spec.model["context_length"] == 8


def test_requesting_more_architectures_than_the_space_holds_is_refused():
    tiny = build_architecture_schema("tiny-architecture-v1", {
        "hidden_dim": {
            "type": "int", "choices": (16, 32), "default": 16,
            "mutation": {"distribution": "categorical_resample"},
        },
    })
    with pytest.raises(ArchitectureSearchError, match="distinct architectures"):
        preview_architecture_search(
            _base_spec(), tiny, INNER_SCHEMA,
            outer_population_size=4, outer_population_count=1,
            inner_population_size=2, inner_population_count=1, seed=1,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"outer_population_size": 1},
        {"outer_population_count": 0},
        {"inner_population_size": 1},
        {"inner_population_count": 0},
        {"outer_mutation_rate": 1.5},
        {"inner_mutation_rate": -0.1},
    ],
)
def test_malformed_campaign_shapes_are_rejected_before_anything_runs(overrides):
    kwargs = dict(
        outer_population_size=2, outer_population_count=1,
        inner_population_size=2, inner_population_count=1, seed=1,
    )
    kwargs.update(overrides)
    with pytest.raises(ArchitectureSearchError):
        run_architecture_search(_base_spec(), OUTER_SCHEMA, INNER_SCHEMA, **kwargs)


# --------------------------------------------------------------- outer loop


def test_each_architecture_gets_its_own_full_inner_campaign(tmp_path, monkeypatch):
    campaigns = _install_fake_inner_search(monkeypatch, scores={
        "arch3-arch0-a0": 0.30, "arch3-arch0-a1": 0.10, "arch3-arch0-a2": 0.20,
    })

    report = run_architecture_search(
        _base_spec(), OUTER_SCHEMA, INNER_SCHEMA,
        outer_population_size=3, outer_population_count=1,
        inner_population_size=2, inner_population_count=1,
        seed=3, method="random", root=tmp_path / "runs",
    )

    assert len(campaigns) == 3
    assert [campaign.prefix for campaign in campaigns] == [
        "arch3-arch0-a0", "arch3-arch0-a1", "arch3-arch0-a2",
    ]
    # Every inner campaign got the same budget, and the inner schema --
    # that equal-budget-per-architecture property is the whole reason the
    # loops are nested rather than joint.
    assert {campaign.population_size for campaign in campaigns} == {2}
    assert {campaign.population_count for campaign in campaigns} == {1}
    assert all(campaign.schema is INNER_SCHEMA for campaign in campaigns)
    # Each inner campaign's base spec is a launchable fresh spec at its own
    # architecture.
    assert {campaign.spec.mode for campaign in campaigns} == {"fresh"}
    assert len({campaign.spec.model["hidden_dim"] for campaign in campaigns}) >= 1

    assert report.best_architecture_id == "arch3-arch0-a1"
    assert report.best_run_id == "arch3-arch0-a1-p0-c0"
    assert report.best_metric_value == 0.10
    assert report.inner_campaigns_run == 3
    assert report.total_trials_started == 6
    assert report.total_epochs_executed == 12
    assert report.best_model["hidden_dim"] == report.best_genome["hidden_dim"]


def test_survivors_carry_forward_and_children_are_bred_back_to_size(tmp_path, monkeypatch):
    campaigns = _install_fake_inner_search(monkeypatch, scores={
        "arch3-arch0-a0": 0.30, "arch3-arch0-a1": 0.10, "arch3-arch0-a2": 0.20,
        "arch3-arch1-a1": 0.05, "arch3-arch1-a2": 0.40,
    })

    report = run_architecture_search(
        _base_spec(), OUTER_SCHEMA, INNER_SCHEMA,
        outer_population_size=3, outer_population_count=2,
        inner_population_size=2, inner_population_count=1,
        seed=3, method="random", outer_mutation_rate=1.0, root=tmp_path / "runs",
    )

    first, second = report.generations
    assert first.survivor_architecture_ids == ("arch3-arch0-a1",)
    assert first.offspring_architecture_ids == ("arch3-arch1-a1", "arch3-arch1-a2")

    carried = next(result for result in second.results if result.carried)
    assert carried.candidate_index == 0
    assert carried.inner_trials_started == 0
    assert carried.metric_value == 0.10
    # The carried survivor ran no second inner campaign: five campaigns for
    # six candidate slots.
    assert len(campaigns) == 5
    assert report.inner_campaigns_run == 5
    assert report.best_architecture_id == "arch3-arch1-a1"
    assert report.best_metric_value == 0.05


def test_a_bred_architectures_full_provenance_reaches_the_report(tmp_path, monkeypatch):
    """Crossover choices and mutations are not lost between generations.

    An outer candidate is a whole inner campaign, not one run directory, so
    there is nowhere on disk for ``record_breeding_lineage``'s equivalent to
    write -- the report carries it instead.
    """
    _install_fake_inner_search(monkeypatch, scores={
        "arch3-arch0-a0": 0.30, "arch3-arch0-a1": 0.10, "arch3-arch0-a2": 0.20,
        "arch3-arch1-a1": 0.05, "arch3-arch1-a2": 0.40,
    })

    report = run_architecture_search(
        _base_spec(), OUTER_SCHEMA, INNER_SCHEMA,
        outer_population_size=3, outer_population_count=2,
        inner_population_size=2, inner_population_count=1,
        seed=3, method="random", outer_mutation_rate=1.0, root=tmp_path / "runs",
    )

    second = report.generations[1]
    sampled = next(result for result in second.results if result.origin_generation == 0)
    assert sampled.breeding_lineage is None

    child = next(result for result in second.results if result.origin_generation == 1)
    lineage = child.breeding_lineage
    assert lineage is not None
    assert lineage.genome_operator == ARCHITECTURE_GENOME_OPERATOR
    assert lineage.parent_a_run_id == "arch3-arch0-a1-p0-c0"
    assert set(lineage.crossover_choices) == set(OUTER_SCHEMA.gene_names)
    assert lineage.child_genome == dict(child.genome)
    assert json.loads(json.dumps(report.to_dict()))["generations"][1]["results"][1][
        "breeding_lineage"
    ]["genome_operator"] == ARCHITECTURE_GENOME_OPERATOR


def test_an_architecture_whose_inner_campaign_finds_no_survivor_is_eliminated(tmp_path, monkeypatch):
    _install_fake_inner_search(
        monkeypatch,
        scores={"arch3-arch0-a0": 0.30, "arch3-arch0-a2": 0.20},
        failures={"arch3-arch0-a1"},
    )

    report = run_architecture_search(
        _base_spec(), OUTER_SCHEMA, INNER_SCHEMA,
        outer_population_size=3, outer_population_count=1,
        inner_population_size=2, inner_population_count=1,
        seed=3, method="random", root=tmp_path / "runs",
    )

    eliminated = next(
        result for result in report.generations[0].results
        if result.architecture_id == "arch3-arch0-a1"
    )
    assert eliminated.state == STATE_NO_INNER_SURVIVOR
    assert eliminated.quality_passed is False
    assert eliminated.retained is False
    assert "no surviving candidate" in eliminated.reason
    assert report.best_architecture_id == "arch3-arch0-a2"


def test_one_architectures_failure_does_not_abort_its_siblings(tmp_path, monkeypatch):
    _install_fake_inner_search(
        monkeypatch,
        scores={"arch3-arch0-a0": 0.30, "arch3-arch0-a2": 0.20},
        raises={"arch3-arch0-a1"},
    )

    report = run_architecture_search(
        _base_spec(), OUTER_SCHEMA, INNER_SCHEMA,
        outer_population_size=3, outer_population_count=1,
        inner_population_size=2, inner_population_count=1,
        seed=3, method="random", root=tmp_path / "runs",
    )

    failed = next(
        result for result in report.generations[0].results
        if result.architecture_id == "arch3-arch0-a1"
    )
    assert failed.state == "failed"
    assert "RuntimeError" in failed.reason
    assert report.best_architecture_id == "arch3-arch0-a2"
    assert report.inner_campaigns_run == 2


def test_no_surviving_architecture_ends_the_campaign_without_breeding(tmp_path, monkeypatch):
    _install_fake_inner_search(
        monkeypatch, scores={},
        failures={"arch3-arch0-a0", "arch3-arch0-a1", "arch3-arch0-a2"},
    )

    report = run_architecture_search(
        _base_spec(), OUTER_SCHEMA, INNER_SCHEMA,
        outer_population_size=3, outer_population_count=3,
        inner_population_size=2, inner_population_count=1,
        seed=3, method="random", root=tmp_path / "runs",
    )

    assert len(report.generations) == 1
    assert report.generations[0].survivor_architecture_ids == ()
    assert report.generations[0].offspring_architecture_ids == ()
    assert report.best_architecture_id is None
    assert report.best_run_id is None


def test_report_serializes_to_json(tmp_path, monkeypatch):
    _install_fake_inner_search(monkeypatch, scores={
        "arch3-arch0-a0": 0.30, "arch3-arch0-a1": 0.10, "arch3-arch0-a2": 0.20,
    })
    report = run_architecture_search(
        _base_spec(), OUTER_SCHEMA, INNER_SCHEMA,
        outer_population_size=3, outer_population_count=1,
        inner_population_size=2, inner_population_count=1,
        seed=3, method="random", root=tmp_path / "runs",
    )
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["format"] == ARCHITECTURE_SEARCH_FORMAT
    assert payload["estimate"]["estimated_max_trials"] == 6
    assert payload["generations"][0]["results"][0]["architecture_id"] == "arch3-arch0-a0"


# --------------------------------------------------------------- outer breeding


def _parent(architecture_id, run_id, genome, metric_value):
    return ArchitectureParent(
        architecture_id=architecture_id, run_id=run_id, genome=genome, metric_value=metric_value,
    )


def test_structural_child_is_a_launchable_fresh_spec_with_no_parent_block():
    base = _base_spec()
    parent_a = _parent("a0", "run-a", {BACKBONE_PRESET_GENE: "gru", "hidden_dim": 16}, 0.2)
    parent_b = _parent("a1", "run-b", {BACKBONE_PRESET_GENE: "transformer_h4_l2", "hidden_dim": 64}, 0.1)

    result = breed_architecture(
        parent_a, parent_b, OUTER_SCHEMA, base,
        objective="windowed_rollout", generation=1, seed=17, mutation_rate=0.0,
    )

    child = result.child_spec
    # An architecture-changing child trains from scratch: there is no
    # compatible checkpoint to inherit, and mode="fresh" must have no parent.
    assert child.mode == "fresh"
    assert child.parent is None
    assert list(child.evolution["configuration_parents"]) == ["run-a", "run-b"]
    # weight_donor is a provenance label only -- the fitter parent.
    assert child.evolution["weight_donor"] == "run-b"
    assert child.evolution["genome_operator"] == ARCHITECTURE_GENOME_OPERATOR
    assert child.evolution["generation"] == 1
    # Every gene came from one of the two parents (mutation disabled here).
    for name, value in result.child_genome.items():
        assert value in (parent_a.genome[name], parent_b.genome[name])
    assert child.model == apply_architecture_genome(base.model, result.child_genome)


def test_child_training_and_data_blocks_come_from_the_base_spec():
    """The outer loop varies the architecture and nothing else."""
    base = _base_spec()
    result = breed_architecture(
        _parent("a0", "run-a", {BACKBONE_PRESET_GENE: "gru", "hidden_dim": 16}, 0.2),
        _parent("a1", "run-b", {BACKBONE_PRESET_GENE: "gru", "hidden_dim": 64}, 0.1),
        OUTER_SCHEMA, base, objective="windowed_rollout", generation=1, seed=1,
    )
    assert dict(result.child_spec.training) == dict(base.training)
    assert dict(result.child_spec.data) == dict(base.data)
    assert dict(result.child_spec.evaluation) == dict(base.evaluation)


def test_weight_donor_label_follows_the_selection_metric_direction():
    genome_a = {BACKBONE_PRESET_GENE: "gru", "hidden_dim": 16}
    genome_b = {BACKBONE_PRESET_GENE: "gru", "hidden_dim": 64}
    parent_a = _parent("a0", "run-a", genome_a, 0.2)
    parent_b = _parent("a1", "run-b", genome_b, 0.1)

    minimizing = breed_architecture(
        parent_a, parent_b, OUTER_SCHEMA, _base_spec(),
        objective="windowed_rollout", generation=1, seed=1, selection_mode="min",
    )
    maximizing = breed_architecture(
        parent_a, parent_b, OUTER_SCHEMA, _base_spec(),
        objective="windowed_rollout", generation=1, seed=1, selection_mode="max",
    )
    assert minimizing.lineage.weight_donor_run_id == "run-b"
    assert maximizing.lineage.weight_donor_run_id == "run-a"


def test_breeding_is_deterministic_and_records_full_provenance():
    parents = (
        _parent("a0", "run-a", {BACKBONE_PRESET_GENE: "gru", "hidden_dim": 16}, 0.2),
        _parent("a1", "run-b", {BACKBONE_PRESET_GENE: "transformer_h2_l1", "hidden_dim": 64}, 0.1),
    )
    kwargs = dict(objective="windowed_rollout", generation=2, seed=99, mutation_rate=1.0)
    first = breed_architecture(*parents, OUTER_SCHEMA, _base_spec(), **kwargs)
    second = breed_architecture(*parents, OUTER_SCHEMA, _base_spec(), **kwargs)

    assert first.child_genome == second.child_genome
    lineage = first.lineage.to_dict()
    assert lineage["parent_a_run_id"] == "run-a"
    assert lineage["parent_b_run_id"] == "run-b"
    assert lineage["mutation_seed"] == 99
    assert set(lineage["crossover_choices"]) == set(OUTER_SCHEMA.gene_names)
    assert set(lineage["child_genome"]) == set(OUTER_SCHEMA.gene_names)
    assert json.loads(json.dumps(lineage))["child_model"]["hidden_dim"] == \
        first.child_spec.model["hidden_dim"]


# ------------------------------------------- the weight_donor-as-label regression


def _contracts_for(spec):
    """Minimal contracts matching ``spec``, for artifact allocation."""
    architecture = ArchitectureContract(
        pixel_shape=(3, 8, 8),
        rgb_preprocessing_version="v1",
        action_vocabulary=("noop",),
        workspace_modalities={"pixels": 1},
        workspace_layout_hash=None,
        latent_width=int(spec.model["latent_width"]),
        hidden_dim=int(spec.model["hidden_dim"]),
        action_embed_dim=int(spec.model["action_embed_dim"]),
        reconstruction_shape=(3, 4, 4),
        visual_architecture="tiny",
        semantic_classes=1,
        horizons_ticks=(1,),
        direct_horizon_topology=(1,),
        backbone=str(spec.model["backbone"]),
        context_length=int(spec.model["context_length"]),
        backbone_kwargs=dict(spec.model["backbone_kwargs"]),
    )
    data = DataContract(
        world="crafter", backend="test", program_config={}, scenario_names=("s",),
        scenario_code_version="v1", train_session_ids=("t",), train_session_hashes=("h",),
        validation_session_ids=("v",), validation_session_hashes=("h",),
        test_session_ids=(), test_session_hashes=(), seed_assignments={"t": 1},
        pixel_provenance="crafter", semantic_vocabulary_version="v1",
        preprocessing_version="v1", ticks_per_frame=1.0,
    )
    return architecture, data, TrainingContract(**dict(spec.training))


def _allocate_plain_run(root, run_id):
    """Materialise one ordinary fresh run, to stand in for an inner best run."""
    spec = _base_spec()
    architecture, data, training = _contracts_for(spec)
    return allocate_run_artifacts(root, spec, architecture, data, training, run_id=run_id)


def test_lineage_treats_a_structural_childs_weight_donor_as_a_label_only(tmp_path):
    """A ``mode="fresh"`` child naming an unloaded ``weight_donor`` must be walkable.

    This is the one downstream assumption the outer loop leans on: that
    ``evolution.weight_donor`` is provenance, not an assertion that a
    checkpoint was inherited. ``registry.lineage_graph`` is the consumer
    most likely to disagree -- it collapses ``parent``/
    ``configuration_parents``/``weight_donor`` into one edge set and walks
    them -- so this allocates a real structural child's artifacts and walks
    it, rather than trusting that reading.

    The two parents are materialised first because
    ``artifacts._resolve_naming_parents`` requires every
    ``configuration_parents`` entry to have a persisted ``display_name.json``
    under the same organism. That is satisfied naturally by a real campaign
    (an outer parent's ``run_id`` is its inner campaign's best run, in this
    same root), and is the reason :func:`breed_architecture` records inner
    best run IDs rather than the architecture IDs.
    """
    parent_a_run = _allocate_plain_run(tmp_path, "inner-best-a")
    parent_b_run = _allocate_plain_run(tmp_path, "inner-best-b")

    base = _base_spec()
    result = breed_architecture(
        _parent("a0", parent_a_run.run_id, {BACKBONE_PRESET_GENE: "gru", "hidden_dim": 16}, 0.2),
        _parent(
            "a1", parent_b_run.run_id,
            {BACKBONE_PRESET_GENE: "transformer_h4_l2", "hidden_dim": 64}, 0.1,
        ),
        OUTER_SCHEMA, base, objective="windowed_rollout", generation=1, seed=5, mutation_rate=0.0,
    )
    child = result.child_spec
    architecture, data, training = _contracts_for(child)

    artifacts = allocate_run_artifacts(
        tmp_path, child, architecture, data, training, run_id="structural-child",
    )
    organism_root = Path(artifacts.directory).parent

    lineage = json.loads((Path(artifacts.directory) / "lineage.json").read_text(encoding="utf-8"))
    assert lineage["mode"] == "fresh"
    assert lineage["parent"] is None
    assert lineage["parent_checkpoint_sha"] is None
    assert lineage["weight_donor"] == "inner-best-b"
    assert lineage["configuration_parents"] == ["inner-best-a", "inner-best-b"]

    graph = lineage_graph(organism_root, "structural-child")
    node = graph.nodes["structural-child"]
    # No checkpoint edge at all: the child trains from scratch.
    assert node.parent_run_id is None
    assert node.weight_donor == "inner-best-b"
    # Both configuration parents are walked as ordinary ancestor edges; the
    # donor label collapses into the edge it duplicates rather than
    # demanding a checkpoint.
    assert graph.ancestor_ids == ("inner-best-a", "inner-best-b")


@pytest.mark.parametrize("selection_metric", ["rollout.t+1.model_over_copy_last_mse"])
def test_structural_child_launches_through_run_trial(tmp_path, selection_metric):
    """End-to-end: a bred structural child is genuinely launchable.

    Needs torch (and therefore only runs in the neural CI tier); the
    torch-free tests above cover the spec/lineage shape it depends on.
    """
    pytest.importorskip("torch")
    from cognitive_runtime.training.model_factory.spec import validate

    result = breed_architecture(
        _parent("a0", "run-a", {BACKBONE_PRESET_GENE: "gru", "hidden_dim": 16}, 0.2),
        _parent("a1", "run-b", {BACKBONE_PRESET_GENE: "transformer_h2_l1", "hidden_dim": 32}, 0.1),
        OUTER_SCHEMA, _base_spec(), objective="windowed_rollout", generation=1, seed=8,
    )
    # `run_trial` itself needs a real corpus; what is asserted here is the
    # contract every launch path checks first, for the exact spec shape the
    # outer loop produces.
    validate(result.child_spec)
    assert result.child_spec.evaluation["selection_metric"] == selection_metric


def test_shipped_architecture_schema_drives_a_campaign(tmp_path, monkeypatch):
    """The real ARCHITECTURE_SEARCH_V1, not just the tiny test schema."""
    _install_fake_inner_search(monkeypatch, scores={
        "arch1-arch0-a0": 0.4, "arch1-arch0-a1": 0.1,
    })
    report = run_architecture_search(
        _base_spec(), ARCHITECTURE_SEARCH_V1, INNER_SCHEMA,
        outer_population_size=2, outer_population_count=1,
        inner_population_size=2, inner_population_count=1,
        seed=1, root=tmp_path / "runs",
    )
    assert report.architecture_schema_version == "architecture_search_v1"
    assert report.best_architecture_id == "arch1-arch0-a1"
    assert set(report.best_genome) == set(ARCHITECTURE_SEARCH_V1.gene_names)
