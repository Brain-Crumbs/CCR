"""Model Factory genetic breeding tests (epic #212, Phase D step 3, §14.1-14.2,
issue #232).

Deliberately torch-free: matches test_model_factory_genome.py and
test_model_factory_search_proposals.py in the ``factory-contracts`` CI tier
(see .github/workflows/ci.yml). Parent "runs" are fabricated directly via
``allocate_run_artifacts`` plus a hand-written checkpoint compatibility
header -- never a real trained checkpoint -- so nothing here ever imports
torch.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections.abc import Mapping as ABCMapping
from dataclasses import fields as dataclass_fields, replace
from pathlib import Path
from typing import Any, Dict

import pytest

from cognitive_runtime.training.model_factory.artifacts import allocate_run_artifacts
from cognitive_runtime.training.model_factory.breeding import (
    GENOME_OPERATOR,
    LINEAGE_DETAIL_FORMAT,
    BreedingError,
    BreedingLineage,
    BreedingResult,
    ParentRecord,
    breed,
    check_compatible,
    load_parent,
    record_breeding_lineage,
    select_weight_donor,
)
from cognitive_runtime.training.model_factory.budget import budget_seconds_for_tier
from cognitive_runtime.training.model_factory.checkpoint import HEADER_FORMAT
from cognitive_runtime.training.model_factory.contracts import (
    ArchitectureContract,
    DataContract,
    TrainingContract,
)
from cognitive_runtime.training.model_factory.genome import GenomeRepairError, build_schema
from cognitive_runtime.training.model_factory.spec import DOCUMENT_FORMAT, resolve

REPO_ROOT = Path(__file__).resolve().parents[1]
ORGANISM = "Crafter"


def _schema():
    return build_schema(
        "breeding_test_v1",
        {
            "optimizer.lr": {
                "type": "float", "bounds": (1e-5, 1e-2), "log_scale": True, "default": 3e-4,
                "mutation": {"distribution": "log_normal_perturb", "sigma": 0.3},
            },
            "scheduled_sampling_p": {
                "type": "float", "bounds": (0.0, 0.5), "default": 0.25,
                "mutation": {"distribution": "normal_perturb", "sigma": 0.05},
            },
            "rollout_frames": {
                "type": "int", "choices": (4, 8), "default": 8,
                "mutation": {"distribution": "categorical_resample"},
            },
        },
    )


def _architecture_contract(tag: str = "a") -> ArchitectureContract:
    return ArchitectureContract(
        pixel_shape=(3, 64, 64), rgb_preprocessing_version="v1",
        action_vocabulary=("MOVE_UP", "NULL"), workspace_modalities={},
        workspace_layout_hash=None, latent_width=128, hidden_dim=256,
        action_embed_dim=8, reconstruction_shape=(3, 64, 64),
        visual_architecture="spatial_residual_v1", semantic_classes=0,
        horizons_ticks=(1, 2, 3, 4), direct_horizon_topology=(1, 2, 3, 4),
        backbone="transformer", context_length=8, backbone_kwargs={"tag": tag},
    )


def _data_contract(tag: str = "a") -> DataContract:
    return DataContract(
        world="crafter", backend="crafter", program_config={}, scenario_names=(),
        scenario_code_version="v1", train_session_ids=(f"train-{tag}",),
        train_session_hashes=("a",), validation_session_ids=("validation-1",),
        validation_session_hashes=("b",), test_session_ids=("test-1",),
        test_session_hashes=("c",), seed_assignments={},
        pixel_provenance="crafter", semantic_vocabulary_version="v1",
        preprocessing_version="v1", horizons_ticks=(1, 2, 3, 4), ticks_per_frame=1.0,
    )


def _spec(*, lr, scheduled_sampling_p, rollout_frames, tier, selection_metric, corpus_id):
    return resolve({
        "format": DOCUMENT_FORMAT,
        "organism": ORGANISM,
        "mode": "fresh",
        "data": {"corpus_id": corpus_id, "world": "crafter"},
        "training": {
            "optimizer": {"name": "adamw", "lr": lr, "weight_decay": 1e-5},
            "scheduled_sampling_p": scheduled_sampling_p,
            "rollout_frames": rollout_frames,
            "warmup_frames": 3,
            "max_training_seconds": budget_seconds_for_tier(tier),
        },
        "evaluation": {"selection_metric": selection_metric},
    })


def _make_parent_run(
    tmp_path: Path,
    *,
    run_id: str,
    lr: float,
    scheduled_sampling_p: float,
    rollout_frames: int,
    tier: str = "fast",
    architecture_tag: str = "a",
    corpus_tag: str = "a",
    corpus_id: str = "corpus-1",
    selection_metric: str = "rollout.t+4.model_over_copy_last_mse",
    checkpoint_sha256: str = "0" * 64,
    checkpoint_name: str = "best-validation.pt",
) -> Path:
    """Fabricate one completed parent run's on-disk artifacts: the JSON
    manifests ``load_parent`` actually reads, plus a hand-written checkpoint
    compatibility header. No torch, no real checkpoint tensors."""
    spec = _spec(
        lr=lr, scheduled_sampling_p=scheduled_sampling_p, rollout_frames=rollout_frames,
        tier=tier, selection_metric=selection_metric, corpus_id=corpus_id,
    )
    architecture = _architecture_contract(architecture_tag)
    data = _data_contract(corpus_tag)
    training = TrainingContract(**dict(spec.training))
    artifacts = allocate_run_artifacts(tmp_path, spec, architecture, data, training, run_id=run_id)
    checkpoint_path = artifacts.checkpoints_dir / checkpoint_name
    checkpoint_path.write_bytes(b"not-a-real-checkpoint")
    Path(f"{checkpoint_path}.json").write_text(
        json.dumps({"format": HEADER_FORMAT, "checkpoint_sha256": checkpoint_sha256}), encoding="utf-8",
    )
    return artifacts.directory


def _default_parents(tmp_path):
    _make_parent_run(
        tmp_path, run_id="parent-a", lr=1e-4, scheduled_sampling_p=0.1, rollout_frames=4,
        checkpoint_sha256="a" * 64,
    )
    _make_parent_run(
        tmp_path, run_id="parent-b", lr=1e-3, scheduled_sampling_p=0.4, rollout_frames=8,
        checkpoint_sha256="b" * 64,
    )
    schema = _schema()
    parent_a = load_parent(tmp_path, ORGANISM, "parent-a", schema, tier="fast")
    parent_b = load_parent(tmp_path, ORGANISM, "parent-b", schema, tier="fast")
    return parent_a, parent_b, schema


# ---------------------------------------------------------------------------
# load_parent
# ---------------------------------------------------------------------------

def test_load_parent_extracts_genome_and_identity_from_persisted_artifacts(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    assert parent_a.genome == {"optimizer.lr": 1e-4, "scheduled_sampling_p": 0.1, "rollout_frames": 4}
    assert parent_b.genome == {"optimizer.lr": 1e-3, "scheduled_sampling_p": 0.4, "rollout_frames": 8}
    assert parent_a.tier == "fast"
    assert parent_a.genome_schema_version == schema.version
    assert parent_a.checkpoint_sha256 == "a" * 64
    assert parent_a.checkpoint_name == "best-validation.pt"
    assert parent_a.architecture_hash == parent_b.architecture_hash
    assert parent_a.data_contract_hash == parent_b.data_contract_hash


def test_load_parent_rejects_declared_tier_not_matching_recorded_budget(tmp_path):
    _make_parent_run(tmp_path, run_id="run-1", lr=1e-4, scheduled_sampling_p=0.1, rollout_frames=4, tier="fast")
    with pytest.raises(BreedingError, match="does not match tier"):
        load_parent(tmp_path, ORGANISM, "run-1", _schema(), tier="scale")


def test_load_parent_raises_when_trial_spec_is_missing(tmp_path):
    (tmp_path / ORGANISM / "ghost").mkdir(parents=True)
    with pytest.raises(BreedingError, match="no trial_spec.json"):
        load_parent(tmp_path, ORGANISM, "ghost", _schema(), tier="fast")


# ---------------------------------------------------------------------------
# check_compatible: §14.1's compatibility gate
# ---------------------------------------------------------------------------

def test_check_compatible_accepts_identical_parents(tmp_path):
    parent_a, parent_b, _ = _default_parents(tmp_path)
    check_compatible(parent_a, parent_b)  # must not raise


def test_check_compatible_rejects_architecture_hash_mismatch(tmp_path):
    _make_parent_run(tmp_path, run_id="parent-a", lr=1e-4, scheduled_sampling_p=0.1, rollout_frames=4, architecture_tag="a")
    _make_parent_run(tmp_path, run_id="parent-b", lr=1e-3, scheduled_sampling_p=0.4, rollout_frames=8, architecture_tag="b")
    schema = _schema()
    parent_a = load_parent(tmp_path, ORGANISM, "parent-a", schema, tier="fast")
    parent_b = load_parent(tmp_path, ORGANISM, "parent-b", schema, tier="fast")
    with pytest.raises(BreedingError, match="architecture_hash"):
        check_compatible(parent_a, parent_b)


def test_check_compatible_rejects_corpus_mismatch(tmp_path):
    _make_parent_run(tmp_path, run_id="parent-a", lr=1e-4, scheduled_sampling_p=0.1, rollout_frames=4, corpus_tag="a")
    _make_parent_run(tmp_path, run_id="parent-b", lr=1e-3, scheduled_sampling_p=0.4, rollout_frames=8, corpus_tag="b")
    schema = _schema()
    parent_a = load_parent(tmp_path, ORGANISM, "parent-a", schema, tier="fast")
    parent_b = load_parent(tmp_path, ORGANISM, "parent-b", schema, tier="fast")
    with pytest.raises(BreedingError, match="corpus"):
        check_compatible(parent_a, parent_b)


def test_check_compatible_rejects_tier_mismatch(tmp_path):
    _make_parent_run(tmp_path, run_id="parent-a", lr=1e-4, scheduled_sampling_p=0.1, rollout_frames=4, tier="fast")
    _make_parent_run(tmp_path, run_id="parent-b", lr=1e-3, scheduled_sampling_p=0.4, rollout_frames=8, tier="scale")
    schema = _schema()
    parent_a = load_parent(tmp_path, ORGANISM, "parent-a", schema, tier="fast")
    parent_b = load_parent(tmp_path, ORGANISM, "parent-b", schema, tier="scale")
    with pytest.raises(BreedingError, match="budget tier"):
        check_compatible(parent_a, parent_b)


def test_check_compatible_rejects_evaluation_contract_mismatch(tmp_path):
    _make_parent_run(
        tmp_path, run_id="parent-a", lr=1e-4, scheduled_sampling_p=0.1, rollout_frames=4,
        selection_metric="rollout.t+4.model_over_copy_last_mse",
    )
    _make_parent_run(
        tmp_path, run_id="parent-b", lr=1e-3, scheduled_sampling_p=0.4, rollout_frames=8,
        selection_metric="rollout.t+2.model_over_copy_last_mse",
    )
    schema = _schema()
    parent_a = load_parent(tmp_path, ORGANISM, "parent-a", schema, tier="fast")
    parent_b = load_parent(tmp_path, ORGANISM, "parent-b", schema, tier="fast")
    with pytest.raises(BreedingError, match="evaluation contract"):
        check_compatible(parent_a, parent_b)


def test_check_compatible_rejects_stage_mismatch(tmp_path):
    parent_a, parent_b, _ = _default_parents(tmp_path)
    other_schema = build_schema(
        "other_stage_v1",
        {"scheduled_sampling_p": {
            "type": "float", "bounds": (0.0, 0.5), "default": 0.25,
            "mutation": {"distribution": "normal_perturb", "sigma": 0.05},
        }},
    )
    parent_b_other_stage = load_parent(tmp_path, ORGANISM, "parent-b", other_schema, tier="fast")
    with pytest.raises(BreedingError, match="stage"):
        check_compatible(parent_a, parent_b_other_stage)


def test_check_compatible_reports_every_mismatch_at_once(tmp_path):
    _make_parent_run(
        tmp_path, run_id="parent-a", lr=1e-4, scheduled_sampling_p=0.1, rollout_frames=4,
        architecture_tag="a", tier="fast",
    )
    _make_parent_run(
        tmp_path, run_id="parent-b", lr=1e-3, scheduled_sampling_p=0.4, rollout_frames=8,
        architecture_tag="b", tier="scale",
    )
    schema = _schema()
    parent_a = load_parent(tmp_path, ORGANISM, "parent-a", schema, tier="fast")
    parent_b = load_parent(tmp_path, ORGANISM, "parent-b", schema, tier="scale")
    with pytest.raises(BreedingError) as excinfo:
        check_compatible(parent_a, parent_b)
    message = str(excinfo.value)
    assert "architecture_hash" in message
    assert "budget tier" in message


# ---------------------------------------------------------------------------
# select_weight_donor: seeded, never blends
# ---------------------------------------------------------------------------

def test_select_weight_donor_only_ever_names_one_of_the_two_parents(tmp_path):
    parent_a, parent_b, _ = _default_parents(tmp_path)
    donors = {select_weight_donor(parent_a, parent_b, random.Random(seed)).run_id for seed in range(50)}
    assert donors == {parent_a.run_id, parent_b.run_id}


def test_select_weight_donor_is_deterministic_from_its_rng_state(tmp_path):
    parent_a, parent_b, _ = _default_parents(tmp_path)
    first = select_weight_donor(parent_a, parent_b, random.Random(7)).run_id
    second = select_weight_donor(parent_a, parent_b, random.Random(7)).run_id
    assert first == second


# ---------------------------------------------------------------------------
# breed(): the full deterministic pipeline
# ---------------------------------------------------------------------------

def test_breed_rejects_incompatible_parents_before_drawing_anything(tmp_path):
    _make_parent_run(tmp_path, run_id="parent-a", lr=1e-4, scheduled_sampling_p=0.1, rollout_frames=4, architecture_tag="a")
    _make_parent_run(tmp_path, run_id="parent-b", lr=1e-3, scheduled_sampling_p=0.4, rollout_frames=8, architecture_tag="b")
    schema = _schema()
    parent_a = load_parent(tmp_path, ORGANISM, "parent-a", schema, tier="fast")
    parent_b = load_parent(tmp_path, ORGANISM, "parent-b", schema, tier="fast")
    with pytest.raises(BreedingError, match="incompatible"):
        breed(parent_a, parent_b, schema, objective="windowed_rollout", generation=1, seed=0)


def test_breed_same_seed_reproduces_the_same_child_exactly(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    kwargs: Dict[str, Any] = dict(objective="windowed_rollout", generation=1, seed=42, mutation_rate=0.5)
    first = breed(parent_a, parent_b, schema, **kwargs)
    second = breed(parent_a, parent_b, schema, **kwargs)
    assert first.child_spec == second.child_spec
    assert first.lineage == second.lineage


def test_breed_different_seeds_can_produce_different_children(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    outcomes = {
        breed(parent_a, parent_b, schema, objective="windowed_rollout", generation=1, seed=seed, mutation_rate=0.5).lineage.child_genome["optimizer.lr"]
        for seed in range(10)
    }
    assert len(outcomes) > 1


def test_breed_child_is_clone_mode_with_exactly_one_weight_donor(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    result = breed(parent_a, parent_b, schema, objective="windowed_rollout", generation=2, seed=1)
    assert result.child_spec.mode == "clone"
    assert isinstance(result.child_spec.parent, ABCMapping)
    donor_run_id = result.child_spec.parent["run_id"]
    assert donor_run_id in {parent_a.run_id, parent_b.run_id}
    donor = parent_a if donor_run_id == parent_a.run_id else parent_b
    assert result.child_spec.parent["sha256"] == donor.checkpoint_sha256
    assert result.child_spec.parent["checkpoint"] == donor.checkpoint_name
    assert result.lineage.weight_donor_run_id == donor_run_id


def test_breed_records_both_configuration_parents_and_generation(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    result = breed(parent_a, parent_b, schema, objective="windowed_rollout", generation=5, seed=3)
    assert list(result.child_spec.evolution["configuration_parents"]) == [parent_a.run_id, parent_b.run_id]
    assert result.child_spec.evolution["generation"] == 5
    assert result.child_spec.evolution["genome_operator"] == GENOME_OPERATOR
    assert result.lineage.generation == 5
    assert result.lineage.parent_a_run_id == parent_a.run_id
    assert result.lineage.parent_b_run_id == parent_b.run_id
    assert result.lineage.genome_operator == GENOME_OPERATOR
    assert result.lineage.mutation_seed == 3


def test_breed_respects_explicit_weight_donor_override(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    result = breed(
        parent_a, parent_b, schema, objective="windowed_rollout", generation=1, seed=0,
        weight_donor=parent_b.run_id,
    )
    assert result.lineage.weight_donor_run_id == parent_b.run_id
    assert result.child_spec.parent["run_id"] == parent_b.run_id
    assert result.child_spec.parent["sha256"] == parent_b.checkpoint_sha256


def test_breed_rejects_weight_donor_not_naming_either_parent(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    with pytest.raises(BreedingError, match="must name one of the two breeding parents"):
        breed(
            parent_a, parent_b, schema, objective="windowed_rollout", generation=1, seed=0,
            weight_donor="not-a-parent",
        )


def test_breed_no_mutation_when_rate_is_zero(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    result = breed(parent_a, parent_b, schema, objective="windowed_rollout", generation=1, seed=9, mutation_rate=0.0)
    assert result.lineage.mutations == {}
    assert list(result.child_spec.evolution["mutations"]) == []


def test_breed_lr_mutation_stays_positive_and_is_recorded(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    result = breed(parent_a, parent_b, schema, objective="windowed_rollout", generation=1, seed=5, mutation_rate=1.0)
    assert "optimizer.lr" in result.lineage.mutations
    assert result.lineage.mutations["optimizer.lr"]["to"] > 0.0
    assert list(result.child_spec.evolution["mutations"]) == sorted(result.lineage.mutations)


def test_breed_every_gene_traces_to_a_named_parent_a_mutation_or_a_repair(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    result = breed(parent_a, parent_b, schema, objective="windowed_rollout", generation=1, seed=3, mutation_rate=0.9)
    lineage = result.lineage
    assert set(lineage.crossover_choices) == set(schema.genes)
    for name in schema.genes:
        chosen_parent_value = (
            parent_a.genome[name] if lineage.crossover_choices[name] == "parent_a" else parent_b.genome[name]
        )
        if name in lineage.mutations:
            expected_after_mutation = lineage.mutations[name]["to"]
            assert lineage.mutations[name]["from"] == chosen_parent_value
        else:
            expected_after_mutation = chosen_parent_value
        if name in lineage.repairs:
            assert lineage.repairs[name]["from"] == expected_after_mutation
            assert lineage.child_genome[name] == lineage.repairs[name]["to"]
        else:
            assert lineage.child_genome[name] == expected_after_mutation


def test_breed_rejects_an_unknown_training_objective(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    with pytest.raises(ValueError, match="unknown training objective"):
        breed(parent_a, parent_b, schema, objective="bogus", generation=1, seed=0)


def test_breed_propagates_repair_error_when_over_stage_budget(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    with pytest.raises(GenomeRepairError, match="projected cost"):
        breed(
            parent_a, parent_b, schema, objective="windowed_rollout", generation=1, seed=0,
            stage_budget_seconds=1e-9,
        )


def test_breed_clips_an_out_of_bounds_parent_gene_and_records_the_repair(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    corrupted_a = replace(parent_a, genome={**dict(parent_a.genome), "scheduled_sampling_p": 5.0})
    result = breed(corrupted_a, parent_b, schema, objective="windowed_rollout", generation=1, seed=0, mutation_rate=0.0)
    chosen = result.lineage.crossover_choices["scheduled_sampling_p"]
    if chosen == "parent_a":
        assert result.lineage.repairs["scheduled_sampling_p"] == {"from": 5.0, "to": 0.5}
        assert result.lineage.child_genome["scheduled_sampling_p"] == 0.5
    else:
        assert "scheduled_sampling_p" not in result.lineage.repairs


def test_breed_child_data_and_model_blocks_match_the_weight_donor(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    result = breed(parent_a, parent_b, schema, objective="windowed_rollout", generation=1, seed=2)
    donor = parent_a if result.lineage.weight_donor_run_id == parent_a.run_id else parent_b
    assert result.child_spec.model == donor.spec.model
    assert result.child_spec.data == donor.spec.data


def test_breed_restores_fixed_budget_fields_even_if_a_schema_declares_a_gene_for_them(tmp_path):
    parent_a, parent_b, _ = _default_parents(tmp_path)
    rogue_schema = build_schema(
        "rogue_v1",
        {"max_training_seconds": {
            "type": "float", "bounds": (1.0, 100000.0), "default": 600.0,
            "mutation": {"distribution": "normal_perturb", "sigma": 50.0},
        }},
    )
    tier_seconds = budget_seconds_for_tier("fast")
    rogue_a = replace(
        parent_a, genome={"max_training_seconds": tier_seconds}, genome_schema_version=rogue_schema.version,
    )
    rogue_b = replace(
        parent_b, genome={"max_training_seconds": tier_seconds}, genome_schema_version=rogue_schema.version,
    )
    result = breed(rogue_a, rogue_b, rogue_schema, objective="windowed_rollout", generation=1, seed=0, mutation_rate=1.0)
    assert result.child_spec.training["max_training_seconds"] == tier_seconds


# ---------------------------------------------------------------------------
# record_breeding_lineage: writes the full genetic-operator record
# ---------------------------------------------------------------------------

def test_record_breeding_lineage_enriches_an_already_allocated_child(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    result = breed(parent_a, parent_b, schema, objective="windowed_rollout", generation=4, seed=11)
    architecture = _architecture_contract("a")
    data = _data_contract("a")
    training = TrainingContract(**dict(result.child_spec.training))
    child_artifacts = allocate_run_artifacts(
        tmp_path, result.child_spec, architecture, data, training, run_id="child-1",
    )
    record_breeding_lineage(child_artifacts.directory, result.lineage)
    enriched = json.loads((child_artifacts.directory / "lineage.json").read_text())
    assert enriched["weight_donor"] == result.lineage.weight_donor_run_id
    assert enriched["configuration_parents"] == [parent_a.run_id, parent_b.run_id]
    assert enriched["breeding"] == result.lineage.to_dict()
    assert enriched["breeding"]["format"] == LINEAGE_DETAIL_FORMAT


def test_record_breeding_lineage_rejects_a_mismatched_weight_donor(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    result = breed(
        parent_a, parent_b, schema, objective="windowed_rollout", generation=1, seed=0,
        weight_donor=parent_a.run_id,
    )
    architecture = _architecture_contract("a")
    data = _data_contract("a")
    training = TrainingContract(**dict(result.child_spec.training))
    child_artifacts = allocate_run_artifacts(
        tmp_path, result.child_spec, architecture, data, training, run_id="child-2",
    )
    other_result = breed(
        parent_a, parent_b, schema, objective="windowed_rollout", generation=1, seed=0,
        weight_donor=parent_b.run_id,
    )
    with pytest.raises(BreedingError, match="does not match"):
        record_breeding_lineage(child_artifacts.directory, other_result.lineage)


def test_record_breeding_lineage_requires_an_already_allocated_run(tmp_path):
    fake_lineage = BreedingLineage(
        format=LINEAGE_DETAIL_FORMAT, generation=1, parent_a_run_id="x", parent_b_run_id="y",
        weight_donor_run_id="x", genome_operator=GENOME_OPERATOR, mutation_seed=0,
        parent_a_genome={}, parent_b_genome={}, crossover_choices={}, mutations={}, repairs={},
        child_genome={},
    )
    with pytest.raises(BreedingError, match="allocate the run first"):
        record_breeding_lineage(tmp_path / "does-not-exist", fake_lineage)


# ---------------------------------------------------------------------------
# Acceptance: no weight tensor is ever averaged or spliced across parents.
# ---------------------------------------------------------------------------

def test_no_dataclass_ever_declares_a_weight_tensor_field():
    forbidden = {"state_dict", "model_state_dict", "weights", "tensor", "model"}
    for cls in (ParentRecord, BreedingLineage, BreedingResult):
        names = {f.name for f in dataclass_fields(cls)}
        overlap = names & forbidden
        assert not overlap, f"{cls.__name__} declares a forbidden weight-shaped field: {overlap}"


def test_breed_child_parent_block_never_names_more_than_one_donor(tmp_path):
    parent_a, parent_b, schema = _default_parents(tmp_path)
    result = breed(parent_a, parent_b, schema, objective="windowed_rollout", generation=1, seed=0)
    # exactly one weight-lineage parent (a mapping naming one run),
    # independent of the two *configuration* parents recorded separately in
    # `evolution`.
    assert isinstance(result.child_spec.parent, ABCMapping)
    assert set(result.child_spec.parent) == {"run_id", "checkpoint", "sha256"}
    assert len(result.child_spec.evolution["configuration_parents"]) == 2


_NO_TORCH_SCRIPT = """
import sys

class _BlockTorch:
    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is deliberately blocked for this test")
        return None

sys.meta_path.insert(0, _BlockTorch())
sys.path.insert(0, {path!r})
import cognitive_runtime.training.model_factory.breeding  # noqa: F401
print("OK")
"""


def test_breeding_module_imports_cleanly_without_torch():
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
