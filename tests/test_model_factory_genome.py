"""Model Factory genome schema tests (epic #212, Phase D step 1, issue #229).

Deliberately torch-free: these run in the ``factory-contracts`` CI tier
(see .github/workflows/ci.yml), matching test_model_factory_contracts.py
and test_model_factory_spec.py.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest

from cognitive_runtime.training.model_factory.contracts import ArchitectureContract
from cognitive_runtime.training.model_factory.genome import (
    GENERIC_ACTION_EFFECTS_V1,
    Gene,
    GenomeRepairError,
    GenomeSchema,
    GenomeSchemaError,
    MutationSpec,
    build_schema,
    crossover,
    default_genome,
    get_schema,
    mutate,
    project_cost_seconds,
    repair,
    sample_genome,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bounded_spec(**overrides):
    spec = {
        "type": "float",
        "bounds": (0.0, 1.0),
        "default": 0.5,
        "mutation": {"distribution": "normal_perturb", "sigma": 0.1},
    }
    spec.update(overrides)
    return spec


def _choice_spec(**overrides):
    spec = {
        "type": "int",
        "choices": (1, 2, 3),
        "default": 2,
        "mutation": {"distribution": "categorical_resample"},
    }
    spec.update(overrides)
    return spec


# ---------------------------------------------------------------------------
# Acceptance: every gene declares type, bounds/choices, default and mutation
# distribution; a schema missing any of these fails to load.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_key", ["type", "default", "mutation"])
def test_gene_missing_required_field_fails_to_load(missing_key):
    spec = _bounded_spec()
    del spec[missing_key]
    with pytest.raises(GenomeSchemaError, match=f"missing required field.*{missing_key}"):
        build_schema("t", {"g": spec})


def test_gene_missing_bounds_and_choices_fails_to_load():
    spec = _bounded_spec()
    del spec["bounds"]
    with pytest.raises(GenomeSchemaError, match="bounds or choices"):
        build_schema("t", {"g": spec})


def test_gene_declaring_both_bounds_and_choices_fails_to_load():
    with pytest.raises(GenomeSchemaError, match="exactly one of"):
        Gene(
            name="g",
            type="float",
            default=0.5,
            mutation=MutationSpec("normal_perturb", sigma=0.1),
            bounds=(0.0, 1.0),
            choices=(0.0, 0.5, 1.0),
        )


def test_gene_default_outside_bounds_fails_to_load():
    with pytest.raises(GenomeSchemaError, match="outside its bounds"):
        build_schema("t", {"g": _bounded_spec(default=2.0)})


def test_gene_default_not_in_choices_fails_to_load():
    with pytest.raises(GenomeSchemaError, match="not among its choices"):
        build_schema("t", {"g": _choice_spec(default=99)})


def test_gene_log_scale_requires_positive_lower_bound():
    with pytest.raises(GenomeSchemaError, match="positive lower bound"):
        build_schema("t", {"g": _bounded_spec(bounds=(-1.0, 1.0), log_scale=True, default=0.5)})


def test_gene_mutation_must_match_declared_shape():
    # A choices gene must use categorical_resample, not a numeric perturb.
    with pytest.raises(GenomeSchemaError, match="categorical_resample"):
        build_schema("t", {"g": _choice_spec(mutation={"distribution": "normal_perturb", "sigma": 0.1})})
    # A bounded (non-log) gene must use normal_perturb, not log_normal_perturb.
    with pytest.raises(GenomeSchemaError, match="normal_perturb"):
        build_schema("t", {"g": _bounded_spec(mutation={"distribution": "log_normal_perturb", "sigma": 0.1})})


def test_mutation_spec_requires_positive_sigma_for_perturb_distributions():
    with pytest.raises(GenomeSchemaError, match="positive sigma"):
        MutationSpec("normal_perturb", sigma=None)
    with pytest.raises(GenomeSchemaError, match="positive sigma"):
        MutationSpec("normal_perturb", sigma=-1.0)


def test_mutation_spec_rejects_sigma_for_categorical_resample():
    with pytest.raises(GenomeSchemaError, match="does not use sigma"):
        MutationSpec("categorical_resample", sigma=0.1)


def test_unknown_gene_type_fails_to_load():
    with pytest.raises(GenomeSchemaError, match="unknown type"):
        build_schema("t", {"g": _bounded_spec(type="quaternion")})


# ---------------------------------------------------------------------------
# Acceptance: an inactive gene under the autoregressive objective is
# excluded from crossover and mutation.
# ---------------------------------------------------------------------------

def test_windowed_only_gene_inactive_under_autoregressive():
    gene = GENERIC_ACTION_EFFECTS_V1.genes["closed_loop_pixel_loss_weight"]
    assert gene.is_active("windowed_rollout")
    assert not gene.is_active("autoregressive")


def test_active_gene_names_excludes_inactive_genes_for_objective():
    active = GENERIC_ACTION_EFFECTS_V1.active_gene_names("autoregressive")
    assert "closed_loop_pixel_loss_weight" not in active
    assert "closed_loop_latent_loss_weight" not in active
    assert "optimizer.lr" in active

    active_windowed = GENERIC_ACTION_EFFECTS_V1.active_gene_names("windowed_rollout")
    assert "closed_loop_pixel_loss_weight" in active_windowed


def test_crossover_never_recombines_inactive_gene_under_autoregressive():
    rng = random.Random(0)
    parent_a = default_genome(GENERIC_ACTION_EFFECTS_V1)
    parent_b = dict(parent_a)
    parent_b["closed_loop_pixel_loss_weight"] = 0.9999
    parent_b["optimizer.lr"] = 0.005

    # Run many trials: an inactive gene's child value must always equal
    # parent_a's value (never "chosen" via the coin flip), while an active
    # gene must be able to come from either parent.
    saw_parent_b_lr = False
    for i in range(50):
        child = crossover(
            GENERIC_ACTION_EFFECTS_V1, parent_a, parent_b,
            objective="autoregressive", rng=random.Random(i),
        )
        assert child["closed_loop_pixel_loss_weight"] == parent_a["closed_loop_pixel_loss_weight"]
        if child["optimizer.lr"] == parent_b["optimizer.lr"]:
            saw_parent_b_lr = True
    assert saw_parent_b_lr


def test_mutate_never_touches_inactive_gene_under_autoregressive():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    original_inactive_value = genome["closed_loop_pixel_loss_weight"]
    rng = random.Random(1)
    for _ in range(200):
        genome = mutate(
            GENERIC_ACTION_EFFECTS_V1, genome,
            objective="autoregressive", rng=rng, mutation_rate=1.0,
        )
    assert genome["closed_loop_pixel_loss_weight"] == original_inactive_value
    # An active gene mutating at rate=1.0 for 200 rounds should have moved.
    assert genome["optimizer.lr"] != default_genome(GENERIC_ACTION_EFFECTS_V1)["optimizer.lr"]


def test_mutate_can_touch_windowed_only_gene_under_windowed_rollout():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    rng = random.Random(2)
    for _ in range(200):
        genome = mutate(
            GENERIC_ACTION_EFFECTS_V1, genome,
            objective="windowed_rollout", rng=rng, mutation_rate=1.0,
        )
    assert genome["closed_loop_pixel_loss_weight"] != default_genome(GENERIC_ACTION_EFFECTS_V1)[
        "closed_loop_pixel_loss_weight"
    ]


# ---------------------------------------------------------------------------
# Acceptance: warmup_frames + rollout_frames exceeding the minimum episode
# length is rejected by repair, naming both genes.
# ---------------------------------------------------------------------------

def test_repair_rejects_warmup_plus_rollout_exceeding_min_episode_length():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    genome["warmup_frames"] = 8
    genome["rollout_frames"] = 16
    with pytest.raises(GenomeRepairError) as excinfo:
        repair(
            GENERIC_ACTION_EFFECTS_V1, genome,
            objective="windowed_rollout", min_episode_length=12,
        )
    message = str(excinfo.value)
    assert "warmup_frames" in message
    assert "rollout_frames" in message


def test_repair_accepts_warmup_plus_rollout_within_min_episode_length():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    genome["warmup_frames"] = 3
    genome["rollout_frames"] = 8
    result = repair(
        GENERIC_ACTION_EFFECTS_V1, genome,
        objective="windowed_rollout", min_episode_length=20,
    )
    assert result["warmup_frames"] + result["rollout_frames"] <= 20


def test_repair_without_min_episode_length_is_a_noop_for_that_check():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    genome["warmup_frames"] = 8
    genome["rollout_frames"] = 16
    result = repair(GENERIC_ACTION_EFFECTS_V1, genome, objective="windowed_rollout")
    assert result["warmup_frames"] == 8
    assert result["rollout_frames"] == 16


# ---------------------------------------------------------------------------
# Acceptance: a genome whose projected cost exceeds the stage budget is
# rejected before launch.
# ---------------------------------------------------------------------------

def test_repair_rejects_genome_over_stage_budget():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    genome["warmup_frames"] = 8
    genome["rollout_frames"] = 16
    with pytest.raises(GenomeRepairError, match="projected cost"):
        repair(
            GENERIC_ACTION_EFFECTS_V1, genome,
            objective="windowed_rollout",
            stage_budget_seconds=10.0,
            cost_model=lambda g: project_cost_seconds(g, seconds_per_frame=1.0),
        )


def test_repair_accepts_genome_within_stage_budget():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    result = repair(
        GENERIC_ACTION_EFFECTS_V1, genome,
        objective="windowed_rollout",
        stage_budget_seconds=600.0,
    )
    assert result["warmup_frames"] == genome["warmup_frames"]


def test_repair_uses_custom_cost_model_when_given():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    with pytest.raises(GenomeRepairError, match="projected cost"):
        repair(
            GENERIC_ACTION_EFFECTS_V1, genome,
            objective="windowed_rollout",
            stage_budget_seconds=1_000_000.0,
            cost_model=lambda g: 2_000_000.0,
        )


# ---------------------------------------------------------------------------
# Acceptance: no architecture field, scenario-generator field, or
# reward-semantics field is reachable as a gene.
# ---------------------------------------------------------------------------

def test_shipped_schema_declares_no_architecture_contract_fields():
    architecture_fields = {f.name for f in dataclass_fields(ArchitectureContract)}
    gene_roots = {name.split(".", 1)[0] for name in GENERIC_ACTION_EFFECTS_V1.gene_names}
    assert not (gene_roots & architecture_fields)
    assert not (set(GENERIC_ACTION_EFFECTS_V1.gene_names) & architecture_fields)


@pytest.mark.parametrize("forbidden_name", ["latent_width", "backbone", "action_vocabulary"])
def test_building_a_schema_with_an_architecture_field_gene_fails(forbidden_name):
    with pytest.raises(GenomeSchemaError, match="architecture"):
        build_schema("bad", {forbidden_name: _bounded_spec()})


@pytest.mark.parametrize("forbidden_name", ["world", "reward_weights", "goal"])
def test_building_a_schema_with_a_scenario_or_reward_field_gene_fails(forbidden_name):
    with pytest.raises(GenomeSchemaError, match="scenario-generator, or reward-semantics"):
        build_schema("bad", {forbidden_name: _bounded_spec()})


def test_building_a_schema_with_a_dotted_forbidden_root_fails():
    with pytest.raises(GenomeSchemaError, match="architecture"):
        build_schema("bad", {"backbone_kwargs.dropout": _bounded_spec()})


# ---------------------------------------------------------------------------
# Acceptance: the schema is versioned and a version bump is required to
# change gene bounds.
# ---------------------------------------------------------------------------

def test_schema_version_is_generic_action_effects_v1():
    assert GENERIC_ACTION_EFFECTS_V1.version == "generic_action_effects_v1"
    assert get_schema("generic_action_effects_v1") is GENERIC_ACTION_EFFECTS_V1


def test_get_schema_rejects_unknown_version():
    with pytest.raises(GenomeSchemaError, match="unknown genome schema version"):
        get_schema("does_not_exist_v99")


def test_two_schemas_with_identical_genes_hash_identically():
    genes = {"g": _bounded_spec()}
    a = build_schema("t", genes)
    b = build_schema("t", dict(genes))
    assert a.content_hash == b.content_hash


def test_changing_a_bound_changes_the_content_hash():
    baseline = build_schema("t", {"g": _bounded_spec()})
    changed = build_schema("t", {"g": _bounded_spec(bounds=(0.0, 2.0))})
    assert baseline.content_hash != changed.content_hash


def test_changing_default_or_mutation_sigma_changes_the_content_hash():
    baseline = build_schema("t", {"g": _bounded_spec()})
    changed_default = build_schema("t", {"g": _bounded_spec(default=0.6)})
    changed_sigma = build_schema(
        "t", {"g": _bounded_spec(mutation={"distribution": "normal_perturb", "sigma": 0.2})}
    )
    assert baseline.content_hash != changed_default.content_hash
    assert baseline.content_hash != changed_sigma.content_hash


def test_shipped_schema_content_hash_matches_its_pinned_value():
    # Guards the shipped generic_action_effects_v1 schema specifically: if
    # this ever fails, a gene's bounds/choices/default/mutation changed in
    # place. The fix is a new schema version, not updating the pin here.
    assert GENERIC_ACTION_EFFECTS_V1.content_hash == (
        "d1b981a3e03826f09cc2b255bb9a178eb0f03c7609a1bb66273dfe3662e9055c"
    )


def test_hash_is_independent_of_gene_declaration_order():
    genes_forward = {"a": _bounded_spec(), "b": _choice_spec()}
    genes_reversed = {"b": _choice_spec(), "a": _bounded_spec()}
    assert build_schema("t", genes_forward).content_hash == build_schema("t", genes_reversed).content_hash


# ---------------------------------------------------------------------------
# Sampling / repair round trip and general behavior.
# ---------------------------------------------------------------------------

def test_sample_genome_produces_every_gene_in_range():
    rng = random.Random(42)
    genome = sample_genome(GENERIC_ACTION_EFFECTS_V1, rng)
    assert set(genome) == set(GENERIC_ACTION_EFFECTS_V1.gene_names)
    for name, gene in GENERIC_ACTION_EFFECTS_V1.genes.items():
        value = genome[name]
        if gene.choices is not None:
            assert value in gene.choices
        else:
            low, high = gene.bounds
            assert low <= value <= high


def test_default_genome_matches_every_gene_default():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    for name, gene in GENERIC_ACTION_EFFECTS_V1.genes.items():
        assert genome[name] == gene.default


def test_repair_rejects_genome_missing_a_gene():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    del genome["optimizer.lr"]
    with pytest.raises(GenomeRepairError, match="missing gene"):
        repair(GENERIC_ACTION_EFFECTS_V1, genome, objective="windowed_rollout")


def test_repair_rejects_genome_with_unknown_gene():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    genome["not_a_real_gene"] = 1
    with pytest.raises(GenomeRepairError, match="unknown"):
        repair(GENERIC_ACTION_EFFECTS_V1, genome, objective="windowed_rollout")


def test_repair_clips_out_of_bounds_values():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    genome["scheduled_sampling_p"] = 5.0
    result = repair(GENERIC_ACTION_EFFECTS_V1, genome, objective="windowed_rollout")
    assert result["scheduled_sampling_p"] == 0.5


def test_repair_rejects_choice_value_not_in_choices():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    genome["rollout_frames"] = 7
    with pytest.raises(GenomeRepairError, match="allowed choices"):
        repair(GENERIC_ACTION_EFFECTS_V1, genome, objective="windowed_rollout")


def test_crossover_rejects_parent_missing_a_gene():
    parent_a = default_genome(GENERIC_ACTION_EFFECTS_V1)
    parent_b = dict(parent_a)
    del parent_b["optimizer.lr"]
    with pytest.raises(GenomeRepairError, match="missing gene"):
        crossover(
            GENERIC_ACTION_EFFECTS_V1, parent_a, parent_b,
            objective="windowed_rollout", rng=random.Random(0),
        )


def test_mutate_rejects_invalid_mutation_rate():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    with pytest.raises(ValueError, match="mutation_rate"):
        mutate(
            GENERIC_ACTION_EFFECTS_V1, genome,
            objective="windowed_rollout", rng=random.Random(0), mutation_rate=1.5,
        )


def test_unknown_objective_is_rejected_everywhere():
    genome = default_genome(GENERIC_ACTION_EFFECTS_V1)
    with pytest.raises(ValueError, match="unknown training objective"):
        repair(GENERIC_ACTION_EFFECTS_V1, genome, objective="bogus")
    with pytest.raises(ValueError, match="unknown training objective"):
        mutate(GENERIC_ACTION_EFFECTS_V1, genome, objective="bogus", rng=random.Random(0))
    with pytest.raises(ValueError, match="unknown training objective"):
        crossover(
            GENERIC_ACTION_EFFECTS_V1, genome, genome, objective="bogus", rng=random.Random(0),
        )


# ---------------------------------------------------------------------------
# Acceptance: imports cleanly without torch.
# ---------------------------------------------------------------------------

_NO_TORCH_SCRIPT = """
import sys

class _BlockTorch:
    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is deliberately blocked for this test")
        return None

sys.meta_path.insert(0, _BlockTorch())
sys.path.insert(0, {path!r})
import cognitive_runtime.training.model_factory.genome  # noqa: F401
print("OK")
"""


def test_genome_module_imports_cleanly_without_torch():
    env = dict(os.environ)
    result = subprocess.run(
        [sys.executable, "-c", _NO_TORCH_SCRIPT.format(path=str(REPO_ROOT))],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
