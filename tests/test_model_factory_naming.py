"""Cosmetic Model Factory display names never become run identity."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cognitive_runtime.training.model_factory.naming import (
    DISPLAY_NAME_FORMAT,
    DisplayNameParent,
    assign_display_name,
)


def test_same_seed_and_parents_produce_the_same_name_in_another_process(tmp_path):
    parents = (
        DisplayNameParent("run-a", "brave-shannon-curie"),
        DisplayNameParent("run-b", "calm-turing-lovelace"),
    )
    assignment = assign_display_name(run_id="child-run", naming_seed="repeatable", parents=parents)
    script = """
from cognitive_runtime.training.model_factory.naming import DisplayNameParent, assign_display_name
print(assign_display_name(
    run_id='child-run', naming_seed='repeatable',
    parents=(DisplayNameParent('run-a', 'brave-shannon-curie'), DisplayNameParent('run-b', 'calm-turing-lovelace')),
).display_name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).parents[1], capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == assignment.display_name


def test_fresh_clone_and_genetic_names_follow_lineage_rules():
    fresh = assign_display_name(run_id="fresh", naming_seed="fresh")
    assert len(fresh.display_name.split("-")) == 2
    assert fresh.surname_source_run_ids == ()

    parent = DisplayNameParent("parent", "brave-shannon-curie")
    clone = assign_display_name(run_id="clone", naming_seed="clone", parents=(parent,))
    assert clone.display_name.split("-")[1:] == ["shannon", "curie"]
    assert clone.surname_source_run_ids == ("parent",)

    # A single-surname parent is valid at either position, including when it
    # is selected as the second cosmetic configuration parent.
    left = DisplayNameParent("left", "bold-shannon")
    right = DisplayNameParent("right", "calm-lovelace")
    child = assign_display_name(run_id="child", naming_seed="genetic", parents=(left, right))
    assert set(child.display_name.split("-")[1:]) == {"shannon", "lovelace"}
    assert set(child.surname_source_run_ids) == {"left", "right"}


def test_collision_only_suffixes_the_new_name_and_keeps_run_identity_separate():
    base = assign_display_name(run_id="first-run", naming_seed="same-seed")
    second = assign_display_name(
        run_id="crafter-20260729T145959-1-8fc9b9",
        naming_seed="same-seed",
        existing_names=(base.display_name,),
    )
    assert second.display_name == f"{base.display_name}-1-8fc9b9"
    assert base.display_name != second.display_name
    child = assign_display_name(
        run_id="child",
        naming_seed="child",
        parents=(DisplayNameParent("crafter-20260729T145959-1-8fc9b9", second.display_name),),
    )
    assert child.display_name.split("-")[1:] == base.display_name.split("-")[1:]


def test_artifacts_persist_display_name_without_using_it_for_directory_identity(tmp_path):
    from cognitive_runtime.training.model_factory.artifacts import allocate_run_artifacts
    from cognitive_runtime.training.model_factory.contracts import (
        ArchitectureContract, DataContract, TrainingContract,
    )
    from cognitive_runtime.training.model_factory.spec import resolve

    spec = resolve({
        "organism": "Crafter", "mode": "fresh",
        "data": {"corpus_id": "corpus-1", "world": "crafter"},
        "training": {"optimizer": {"name": "adamw", "lr": 0.0003}},
        "evaluation": {"selection_metric": "rollout.t+4.model_over_copy_last_mse"},
    })
    architecture = ArchitectureContract(
        pixel_shape=(3, 64, 64), rgb_preprocessing_version="v1", action_vocabulary=("NULL",),
        workspace_modalities={}, workspace_layout_hash=None, latent_width=128, hidden_dim=256,
        action_embed_dim=8, reconstruction_shape=(3, 64, 64), visual_architecture="spatial_residual_v1",
        semantic_classes=0, horizons_ticks=(1, 2, 3, 4), direct_horizon_topology=(1, 2, 3, 4),
        backbone="transformer", context_length=8,
    )
    data = DataContract(
        world="crafter", backend="crafter", program_config={}, scenario_names=(), scenario_code_version="v1",
        train_session_ids=(), train_session_hashes=(), validation_session_ids=(), validation_session_hashes=(),
        test_session_ids=(), test_session_hashes=(), seed_assignments={}, pixel_provenance="crafter",
        semantic_vocabulary_version="v1", preprocessing_version="v1", horizons_ticks=(1, 2, 3, 4),
        ticks_per_frame=1.0,
    )
    training = TrainingContract(**dict(spec.training))
    first = allocate_run_artifacts(
        tmp_path, spec, architecture, data, training, run_id="first-run", naming_seed="same-seed",
    )
    second = allocate_run_artifacts(
        tmp_path, spec, architecture, data, training, run_id="second-deadbeef", naming_seed="same-seed",
    )
    first_record = json.loads(first.display_name_path.read_text(encoding="utf-8"))
    second_record = json.loads(second.display_name_path.read_text(encoding="utf-8"))
    assert first_record["format"] == DISPLAY_NAME_FORMAT
    assert first_record["naming_seed"] == "same-seed"
    assert second_record["display_name"] == f"{first.display_name}-deadbeef"
    assert first.directory != second.directory
    assert first.run_id != second.run_id


def test_artifact_allocation_derives_clone_and_genetic_names_from_lineage(tmp_path):
    from cognitive_runtime.training.model_factory.artifacts import allocate_run_artifacts
    from cognitive_runtime.training.model_factory.contracts import (
        ArchitectureContract, DataContract, TrainingContract,
    )
    from cognitive_runtime.training.model_factory.spec import resolve

    def inputs(mode="fresh", **extra):
        raw = {
            "organism": "Crafter", "mode": mode,
            "data": {"corpus_id": "corpus-1", "world": "crafter"},
            "training": {"optimizer": {"name": "adamw", "lr": 0.0003}},
            "evaluation": {"selection_metric": "rollout.t+4.model_over_copy_last_mse"},
            **extra,
        }
        spec = resolve(raw)
        architecture = ArchitectureContract(
            pixel_shape=(3, 64, 64), rgb_preprocessing_version="v1", action_vocabulary=("NULL",),
            workspace_modalities={}, workspace_layout_hash=None, latent_width=128, hidden_dim=256,
            action_embed_dim=8, reconstruction_shape=(3, 64, 64), visual_architecture="spatial_residual_v1",
            semantic_classes=0, horizons_ticks=(1, 2, 3, 4), direct_horizon_topology=(1, 2, 3, 4),
            backbone="transformer", context_length=8,
        )
        data = DataContract(
            world="crafter", backend="crafter", program_config={}, scenario_names=(), scenario_code_version="v1",
            train_session_ids=(), train_session_hashes=(), validation_session_ids=(), validation_session_hashes=(),
            test_session_ids=(), test_session_hashes=(), seed_assignments={}, pixel_provenance="crafter",
            semantic_vocabulary_version="v1", preprocessing_version="v1", horizons_ticks=(1, 2, 3, 4),
            ticks_per_frame=1.0,
        )
        return spec, architecture, data, TrainingContract(**dict(spec.training))

    parent_spec, architecture, data, training = inputs()
    parent_a = allocate_run_artifacts(
        tmp_path, parent_spec, architecture, data, training, run_id="parent-a", naming_seed="parent-a",
    )
    parent_b = allocate_run_artifacts(
        tmp_path, parent_spec, architecture, data, training, run_id="parent-b", naming_seed="parent-b",
    )
    clone_spec, architecture, data, training = inputs(
        "clone", parent={"run_id": "parent-a", "checkpoint": "best.pt", "sha256": "a"},
    )
    clone = allocate_run_artifacts(
        tmp_path, clone_spec, architecture, data, training, run_id="clone", naming_seed="clone",
    )
    clone_record = json.loads(clone.display_name_path.read_text(encoding="utf-8"))
    assert clone_record["surname_source_run_ids"] == ["parent-a"]
    assert clone.display_name.split("-")[1:] == parent_a.display_name.split("-")[1:]

    genetic_spec, architecture, data, training = inputs(
        evolution={"configuration_parents": ["parent-a", "parent-b"], "weight_donor": "parent-b"},
    )
    genetic = allocate_run_artifacts(
        tmp_path, genetic_spec, architecture, data, training, run_id="genetic", naming_seed="genetic",
    )
    genetic_record = json.loads(genetic.display_name_path.read_text(encoding="utf-8"))
    assert set(genetic_record["surname_source_run_ids"]) == {"parent-a", "parent-b"}
    assert set(genetic.display_name.split("-")[1:]) == {
        parent_a.display_name.split("-")[-1], parent_b.display_name.split("-")[-1],
    }
