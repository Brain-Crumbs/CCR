"""Architecture genome schema tests (Model Factory clinic redesign, Phase 6a).

Deliberately torch-free, like test_model_factory_genome.py: the whole point
of :mod:`~cognitive_runtime.training.model_factory.architecture_genome` is
that a campaign's architecture search space is declarable, hashable and
mergeable before anything touches a GPU.
"""

from __future__ import annotations

import json
import random
from dataclasses import fields as dataclass_fields

import pytest

from cognitive_runtime.training.model_factory.architecture_genome import (
    ARCHITECTURE_GENE_ROOTS,
    ARCHITECTURE_GENOME_SCHEMAS,
    ARCHITECTURE_SEARCH_V1,
    BACKBONE_PRESET_GENE,
    BACKBONE_PRESETS,
    INERT_GENES_BY_BACKBONE,
    ArchitectureGenomeError,
    apply_architecture_genome,
    backbone_preset_for,
    build_architecture_schema,
    canonicalize_architecture_genome,
    default_architecture_genome,
    inert_genes_for,
    extract_architecture_genome,
    get_architecture_schema,
    resolve_backbone_preset,
    sample_architecture_genome,
)
from cognitive_runtime.training.model_factory.contracts import (
    ArchitectureContract,
    TrainingContract,
)
from cognitive_runtime.training.model_factory.genome import (
    GENERIC_ACTION_EFFECTS_V1,
    GenomeSchemaError,
    crossover,
    mutate,
    repair,
)
from cognitive_runtime.training.model_factory.spec import DEFAULT_MODEL, MODEL_KEYS


def _model_block(**overrides):
    model = {
        "backbone": "gru",
        "latent_width": 128,
        "hidden_dim": 256,
        "action_embed_dim": 8,
        "reconstruction_size": 64,
        "context_length": 8,
        "backbone_kwargs": {},
    }
    model.update(overrides)
    return model


def _choice_gene(*choices, default=None):
    return {
        "type": "int",
        "choices": choices,
        "default": choices[0] if default is None else default,
        "mutation": {"distribution": "categorical_resample"},
    }


# --------------------------------------------------------------- schema integrity


def test_shipped_schema_is_registered_and_content_pinned():
    assert get_architecture_schema("architecture_search_v1") is ARCHITECTURE_SEARCH_V1
    assert ARCHITECTURE_GENOME_SCHEMAS == {"architecture_search_v1": ARCHITECTURE_SEARCH_V1}
    # The pin is asserted at import time; re-deriving it here states the
    # invariant the pin exists to protect rather than restating the digest.
    rebuilt = build_architecture_schema(
        ARCHITECTURE_SEARCH_V1.version,
        {
            name: {
                "type": gene.type,
                "choices": gene.choices,
                "default": gene.default,
                "mutation": {"distribution": gene.mutation.distribution},
            }
            for name, gene in ARCHITECTURE_SEARCH_V1.genes.items()
        },
    )
    assert rebuilt.content_hash == ARCHITECTURE_SEARCH_V1.content_hash


def test_unknown_schema_version_is_a_hard_error():
    with pytest.raises(GenomeSchemaError, match="unknown architecture genome schema"):
        get_architecture_schema("architecture_search_v99")


def test_shipped_schema_declares_only_allowlisted_architecture_genes():
    assert set(ARCHITECTURE_SEARCH_V1.gene_names) <= ARCHITECTURE_GENE_ROOTS
    assert BACKBONE_PRESET_GENE in ARCHITECTURE_SEARCH_V1.gene_names


def test_every_reachable_gene_names_a_real_model_key_and_architecture_field():
    """The genes must reach fields that both exist on a spec and change identity."""
    architecture_fields = {field.name for field in dataclass_fields(ArchitectureContract)}
    reachable = {
        name for name in ARCHITECTURE_SEARCH_V1.gene_names if name != BACKBONE_PRESET_GENE
    } | {"backbone", "backbone_kwargs"}
    assert reachable <= set(MODEL_KEYS)
    assert reachable <= architecture_fields


def test_every_preset_head_count_divides_every_hidden_dim_choice():
    """No (backbone_preset, hidden_dim) pair this schema can sample is unbuildable."""
    head_counts = {
        int(preset["backbone_kwargs"]["n_heads"])
        for preset in BACKBONE_PRESETS.values()
        if "n_heads" in preset["backbone_kwargs"]
    }
    for hidden_dim in ARCHITECTURE_SEARCH_V1.genes["hidden_dim"].choices:
        for heads in head_counts:
            assert hidden_dim % heads == 0, (hidden_dim, heads)


# --------------------------------------------------------------- the guardrail


@pytest.mark.parametrize(
    "gene_name",
    [
        "optimizer.lr",          # a TrainingContract path -- the inner axis
        "seed",                  # a bare TrainingContract field
        "rollout_frames",
        "reconstruction_size",   # a real model key, deliberately excluded in v1
        "pixel_shape",           # corpus-derived, never a free choice
        "action_vocabulary",
        "horizons_ticks",
        "nonsense",
    ],
)
def test_guardrail_rejects_genes_outside_the_architecture_allowlist(gene_name):
    with pytest.raises(GenomeSchemaError):
        build_architecture_schema("guardrail-test-v1", {gene_name: _choice_gene(1, 2)})


def test_second_guard_still_refuses_a_training_field_if_the_allowlist_widens(monkeypatch):
    """The mirrored TrainingContract check is not dead code.

    Today it is unreachable: no ``TrainingContract`` field name is in the
    allowlist, so a training gene is already rejected by the allowlist
    check with the generic message. Widening the allowlist -- exactly the
    edit a future "just let it evolve the seed too" change would make --
    must still fail, and with the message that says *why*: training-procedure
    genes belong to the other schema family.
    """
    import cognitive_runtime.training.model_factory.architecture_genome as module

    training_fields = {field.name for field in dataclass_fields(TrainingContract)}
    assert "seed" in training_fields
    monkeypatch.setattr(
        module, "ARCHITECTURE_GENE_ROOTS", ARCHITECTURE_GENE_ROOTS | {"seed"},
    )
    with pytest.raises(GenomeSchemaError, match="TrainingContract"):
        module._assert_genes_are_architecture_contract_paths("widened-v1", ("seed",))


def test_training_schema_still_rejects_architecture_genes():
    """The two guardrails are a pair: neither family can declare the other's."""
    from cognitive_runtime.training.model_factory.genome import build_schema

    with pytest.raises(GenomeSchemaError):
        build_schema("reverse-guardrail-v1", {"latent_width": _choice_gene(64, 128)})


# --------------------------------------------------------------- preset merge


def test_backbone_preset_expands_into_both_model_fields():
    merged = apply_architecture_genome(
        _model_block(), {BACKBONE_PRESET_GENE: "transformer_h4_l2"},
    )
    assert merged["backbone"] == "transformer"
    assert merged["backbone_kwargs"] == {"n_heads": 4, "n_layers": 2}


def test_plain_genes_merge_as_ordinary_single_path_values():
    merged = apply_architecture_genome(
        _model_block(),
        {"latent_width": 256, "hidden_dim": 512, "action_embed_dim": 16, "context_length": 12},
    )
    assert (merged["latent_width"], merged["hidden_dim"]) == (256, 512)
    assert (merged["action_embed_dim"], merged["context_length"]) == (16, 12)
    # Untouched fields survive, including the deliberately non-evolvable one.
    assert merged["reconstruction_size"] == 64
    assert merged["backbone"] == "gru"


def test_merge_never_mutates_the_source_model_block():
    model = _model_block()
    apply_architecture_genome(model, {BACKBONE_PRESET_GENE: "transformer_h2_l1", "hidden_dim": 512})
    assert model == _model_block()


def test_preset_kwargs_are_copied_not_aliased():
    """A merged block must never share the module-level preset's kwargs mapping."""
    merged = apply_architecture_genome(_model_block(), {BACKBONE_PRESET_GENE: "transformer_h2_l1"})
    merged["backbone_kwargs"]["n_heads"] = 99
    assert BACKBONE_PRESETS["transformer_h2_l1"]["backbone_kwargs"]["n_heads"] == 2


def test_unknown_preset_is_a_hard_error():
    with pytest.raises(ArchitectureGenomeError, match="unknown backbone preset"):
        resolve_backbone_preset("transformer_h8_l8")


# ------------------------------------------------- inert genes / canonicalization


def test_context_length_is_inert_under_the_gru_and_live_under_windowed_backbones():
    """The GRU ignores context_length; the windowed backbones do not.

    Documented in ``ActionWorldModelConfig.context_length`` ("ignored by
    'gru'") and structurally true via ``TemporalBackbone.context_length_max``,
    which only the windowed backbones set.
    """
    assert inert_genes_for({BACKBONE_PRESET_GENE: "gru", "context_length": 16}) == (
        "context_length",
    )
    for preset in ("dilated_conv", "transformer_h2_l1", "transformer_h4_l2"):
        assert inert_genes_for({BACKBONE_PRESET_GENE: preset, "context_length": 16}) == ()


def test_inert_genes_are_empty_when_the_schema_does_not_evolve_the_backbone():
    assert inert_genes_for({"hidden_dim": 256}) == ()


def test_canonicalizing_collapses_gru_genomes_that_build_the_same_model():
    """Four GRU genomes differing only in context_length are one architecture.

    Left uncollapsed they would carry four distinct genomes, four distinct
    architecture_hash values (context_length is an ArchitectureContract
    field) and -- in a nested campaign -- four separate full inner budgets
    spent re-measuring one architecture.
    """
    canonical = {
        json.dumps(
            canonicalize_architecture_genome(ARCHITECTURE_SEARCH_V1, {
                BACKBONE_PRESET_GENE: "gru",
                "latent_width": 128,
                "hidden_dim": 256,
                "action_embed_dim": 8,
                "context_length": context_length,
            }),
            sort_keys=True,
        )
        for context_length in ARCHITECTURE_SEARCH_V1.genes["context_length"].choices
    }
    assert len(canonical) == 1
    assert json.loads(canonical.pop())["context_length"] == (
        ARCHITECTURE_SEARCH_V1.genes["context_length"].default
    )


def test_canonicalizing_leaves_a_windowed_backbones_context_length_alone():
    genome = {
        BACKBONE_PRESET_GENE: "transformer_h4_l2",
        "latent_width": 128,
        "hidden_dim": 256,
        "action_embed_dim": 8,
        "context_length": 16,
    }
    assert canonicalize_architecture_genome(ARCHITECTURE_SEARCH_V1, genome) == genome


def test_canonicalizing_never_drops_a_declared_gene():
    """Every operator in genome.py requires a complete genome."""
    canonical = canonicalize_architecture_genome(
        ARCHITECTURE_SEARCH_V1, default_architecture_genome(ARCHITECTURE_SEARCH_V1),
    )
    assert set(canonical) == set(ARCHITECTURE_SEARCH_V1.gene_names)


def test_canonicalized_values_stay_within_their_genes_declared_choices():
    for preset in BACKBONE_PRESETS:
        genome = dict(default_architecture_genome(ARCHITECTURE_SEARCH_V1))
        genome[BACKBONE_PRESET_GENE] = preset
        genome["context_length"] = 16
        canonical = canonicalize_architecture_genome(ARCHITECTURE_SEARCH_V1, genome)
        for name, gene in ARCHITECTURE_SEARCH_V1.genes.items():
            assert canonical[name] in gene.choices


def test_inert_gene_declaration_matches_the_real_backbones():
    """Guard the hand-maintained mirror against drift.

    ``architecture_genome`` cannot import ``brain.cortex.backbones`` (it
    imports torch, and this module must stay usable in the core-only
    install), so :data:`INERT_GENES_BY_BACKBONE` restates that module's
    ``context_length_max`` contract by hand. This test is the thing that
    keeps the restatement true.
    """
    pytest.importorskip("torch")
    from brain.cortex.backbones import build_backbone

    for preset_name, preset in BACKBONE_PRESETS.items():
        backbone = build_backbone(
            preset["backbone"], input_dim=8, hidden_dim=16, context_length=8,
            **dict(preset["backbone_kwargs"]),
        )
        ignores_context = backbone.context_length_max is None
        declared = "context_length" in INERT_GENES_BY_BACKBONE.get(preset["backbone"], ())
        assert declared == ignores_context, (
            f"{preset_name}: INERT_GENES_BY_BACKBONE says context_length inert={declared}, "
            f"but the built backbone reports context_length_max="
            f"{backbone.context_length_max!r}"
        )


# --------------------------------------------------------------- extraction


def test_extract_is_the_inverse_of_apply():
    genome = {
        BACKBONE_PRESET_GENE: "transformer_h4_l2",
        "latent_width": 192,
        "hidden_dim": 384,
        "action_embed_dim": 16,
        "context_length": 16,
    }
    merged = apply_architecture_genome(_model_block(), genome)
    assert extract_architecture_genome(ARCHITECTURE_SEARCH_V1, merged) == genome


def test_bare_transformer_default_model_block_has_a_preset_representation():
    """spec.DEFAULT_MODEL's own backbone must be readable as a genome.

    ``{backbone: "transformer", backbone_kwargs: {}}`` builds exactly what
    ``transformer_h2_l1`` builds (TransformerBackbone's constructor defaults
    are n_heads=2, n_layers=1), and it is the single most common
    hand-written model block in the repository -- refusing it would make the
    default spec unbreedable.
    """
    assert DEFAULT_MODEL["backbone"] == "transformer"
    assert dict(DEFAULT_MODEL["backbone_kwargs"]) == {}
    assert backbone_preset_for(DEFAULT_MODEL) == "transformer_h2_l1"


def test_unrepresentable_backbone_configuration_is_refused_not_guessed():
    with pytest.raises(ArchitectureGenomeError, match="does not match any declared backbone preset"):
        backbone_preset_for(_model_block(backbone="transformer", backbone_kwargs={"n_heads": 7}))


def test_extract_names_the_missing_model_field():
    incomplete = {"backbone": "gru", "backbone_kwargs": {}, "latent_width": 128}
    with pytest.raises(ArchitectureGenomeError, match="hidden_dim"):
        extract_architecture_genome(ARCHITECTURE_SEARCH_V1, incomplete)


# --------------------------------------------- reuse of genome.py's operators


def test_default_and_sampled_genomes_cover_every_declared_gene():
    defaults = default_architecture_genome(ARCHITECTURE_SEARCH_V1)
    sampled = sample_architecture_genome(ARCHITECTURE_SEARCH_V1, random.Random(1))
    assert set(defaults) == set(sampled) == set(ARCHITECTURE_SEARCH_V1.gene_names)
    for name, gene in ARCHITECTURE_SEARCH_V1.genes.items():
        assert defaults[name] in gene.choices
        assert sampled[name] in gene.choices


def test_generic_genome_operators_work_unchanged_on_an_architecture_schema():
    """crossover/mutate/repair are generic -- this schema needs no new operators."""
    rng = random.Random(11)
    parent_a = sample_architecture_genome(ARCHITECTURE_SEARCH_V1, random.Random(2))
    parent_b = sample_architecture_genome(ARCHITECTURE_SEARCH_V1, random.Random(3))

    child = crossover(
        ARCHITECTURE_SEARCH_V1, parent_a, parent_b, objective="windowed_rollout", rng=rng,
    )
    assert all(child[name] in (parent_a[name], parent_b[name]) for name in child)

    mutated = mutate(
        ARCHITECTURE_SEARCH_V1, child, objective="windowed_rollout", rng=rng, mutation_rate=1.0,
    )
    repaired = repair(ARCHITECTURE_SEARCH_V1, mutated, objective="windowed_rollout")
    assert repaired == mutated
    for name, gene in ARCHITECTURE_SEARCH_V1.genes.items():
        assert repaired[name] in gene.choices


def test_repair_rejects_a_value_outside_a_genes_declared_choices():
    from cognitive_runtime.training.model_factory.genome import GenomeRepairError

    genome = default_architecture_genome(ARCHITECTURE_SEARCH_V1)
    genome[BACKBONE_PRESET_GENE] = "mamba"
    with pytest.raises(GenomeRepairError):
        repair(ARCHITECTURE_SEARCH_V1, genome, objective="windowed_rollout")


def test_architecture_and_training_schemas_share_no_gene_names():
    """The two axes are disjoint by construction, not by convention."""
    assert not set(ARCHITECTURE_SEARCH_V1.gene_names) & set(GENERIC_ACTION_EFFECTS_V1.gene_names)
