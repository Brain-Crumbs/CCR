"""Crash-safety and provenance tests for Model Factory run artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from cognitive_runtime.training.action_world_model import ActionWorldModelConfig
from cognitive_runtime.training.model_factory.artifacts import (
    CONTRACTS_FORMAT,
    DATA_MANIFEST_FORMAT,
    EXECUTION_FORMAT,
    EXPERIMENT_IDENTITY_FORMAT,
    LINEAGE_FORMAT,
    allocate_run_artifacts,
    atomic_write_json,
    execution_manifest,
)
from cognitive_runtime.training.model_factory.contracts import (
    ArchitectureContract,
    DataContract,
    TrainingContract,
)
from cognitive_runtime.training.model_factory.spec import resolve
from cognitive_runtime.training.nursery import NurseryConfig
from cognitive_runtime.training.statistical_evaluation import build_experiment_report


def _contracts():
    architecture = ArchitectureContract(
        pixel_shape=(3, 64, 64), rgb_preprocessing_version="v1",
        action_vocabulary=("MOVE_UP", "NULL"), workspace_modalities={},
        workspace_layout_hash=None, latent_width=128, hidden_dim=256,
        action_embed_dim=8, reconstruction_shape=(3, 64, 64),
        visual_architecture="spatial_residual_v1", semantic_classes=0,
        horizons_ticks=(1, 2, 3, 4), direct_horizon_topology=(1, 2, 3, 4),
        backbone="transformer", context_length=8,
    )
    data = DataContract(
        world="crafter", backend="crafter", program_config={}, scenario_names=(),
        scenario_code_version="v1", train_session_ids=("train-1",),
        train_session_hashes=("a",), validation_session_ids=("validation-1",),
        validation_session_hashes=("b",), test_session_ids=("test-1",),
        test_session_hashes=("c",), seed_assignments={},
        pixel_provenance="crafter", semantic_vocabulary_version="v1",
        preprocessing_version="v1", horizons_ticks=(1, 2, 3, 4), ticks_per_frame=1.0,
    )
    training = TrainingContract(
        objective="windowed_rollout", optimizer={"name": "adamw", "lr": 0.0003},
        batch_size=32, seed=0, rollout_frames=8, warmup_frames=4,
        scheduled_sampling_p=0.0, loss_weights={},
        checkpoint_selection_policy={"metric": "validation_loss", "mode": "min"},
        device="cpu", precision="fp32", determinism_policy={"deterministic": True, "seed": 0},
    )
    return architecture, data, training


def _spec():
    return resolve({
        "organism": "Crafter", "mode": "fresh",
        "data": {"corpus_id": "corpus-1", "world": "crafter"},
        "training": {"optimizer": {"name": "adamw", "lr": 0.0003}},
        "evaluation": {"selection_metric": "rollout.t+4.model_over_copy_last_mse"},
    })


def _artifact_inputs():
    spec = _spec()
    architecture, data, _training = _contracts()
    return spec, architecture, data, TrainingContract(**dict(spec.training))


def test_fresh_run_writes_versioned_manifest_set_and_initial_directories(tmp_path):
    spec, architecture, data, training = _artifact_inputs()
    artifacts = allocate_run_artifacts(
        tmp_path, spec, architecture, data, training, run_id="run-1",
        data_manifest={"train": [{"session": "train-1", "sha256": "a"}]},
    )

    formats = {
        "experiment.json": EXPERIMENT_IDENTITY_FORMAT,
        "trial_spec.json": "model-factory-spec-v1",
        "contracts.json": CONTRACTS_FORMAT,
        "lineage.json": LINEAGE_FORMAT,
        "execution.json": EXECUTION_FORMAT,
        "data_manifest.json": DATA_MANIFEST_FORMAT,
    }
    for name, expected_format in formats.items():
        with (artifacts.directory / name).open(encoding="utf-8") as handle:
            assert json.load(handle)["format"] == expected_format
    assert artifacts.checkpoints_dir.is_dir()
    assert artifacts.metrics_dir.is_dir()


def test_fresh_run_id_cannot_be_reused(tmp_path):
    spec, architecture, data, training = _artifact_inputs()
    allocate_run_artifacts(tmp_path, spec, architecture, data, training, run_id="run-1")
    with pytest.raises(FileExistsError, match="experiment directory already exists"):
        allocate_run_artifacts(tmp_path, spec, architecture, data, training, run_id="run-1")


def test_explicit_resume_validates_the_original_fresh_trial_without_rewriting_it(tmp_path):
    spec, architecture, data, training = _artifact_inputs()
    first = allocate_run_artifacts(tmp_path, spec, architecture, data, training, run_id="run-1")
    execution_before = first.execution_path.read_bytes()

    resumed = allocate_run_artifacts(
        tmp_path, spec, architecture, data, training, run_id="run-1", resume=True,
    )
    assert resumed.directory == first.directory
    assert resumed.execution_path.read_bytes() == execution_before


def test_execution_manifest_has_source_runtime_and_training_policy(tmp_path):
    spec, architecture, data, training = _artifact_inputs()
    artifacts = allocate_run_artifacts(tmp_path, spec, architecture, data, training, run_id="run-1")
    execution = json.loads(artifacts.execution_path.read_text(encoding="utf-8"))
    assert "git_commit" in execution and "dirty_tree" in execution
    assert "package_versions" in execution and "python_version" in execution
    assert "torch_version" in execution and "cuda_version" in execution
    assert execution["device"] == "cpu"
    assert execution["precision"] == "fp32"
    assert execution["determinism_policy"] == {"deterministic": True, "seed": 0}


def test_source_change_is_visible_in_execution_provenance(monkeypatch):
    import cognitive_runtime.training.model_factory.artifacts as artifacts_module

    monkeypatch.setattr(artifacts_module, "_git_info", lambda: {"commit": "first", "dirty": False})
    first = execution_manifest(device="cpu", precision="fp32", determinism_policy={})
    monkeypatch.setattr(artifacts_module, "_git_info", lambda: {"commit": "second", "dirty": True})
    second = execution_manifest(device="cpu", precision="fp32", determinism_policy={})

    assert (first["git_commit"], first["dirty_tree"]) == ("first", False)
    assert (second["git_commit"], second["dirty_tree"]) == ("second", True)


def test_atomic_write_never_replaces_a_valid_manifest_when_interrupted(tmp_path):
    target = tmp_path / "manifest.json"
    atomic_write_json(target, {"format": "v1", "value": "old"})

    def fail_between_temp_and_rename(_temp_path):
        raise RuntimeError("simulated kill")

    with pytest.raises(RuntimeError, match="simulated kill"):
        atomic_write_json(target, {"format": "v1", "value": "new"}, after_temp_write=fail_between_temp_and_rename)
    assert json.loads(target.read_text(encoding="utf-8")) == {"format": "v1", "value": "old"}


def test_report_contains_full_resolved_model_and_nursery_config():
    model = ActionWorldModelConfig(backbone_kwargs={"n_heads": 2}, context_length=12)
    nursery = NurseryConfig(world="crafter", expected_pixel_source="viewer")
    report = build_experiment_report(
        experiment={"experiment_id": "run-1"},
        checkpoint={"path": "/models/run-1.pt", "sha256": "abc"},
        rollout_metrics={"horizons": {1: {"beats_copy_last": True}}, "event_metrics": {}},
        action_world_model_config=model,
        nursery_config=nursery,
        data_config={"corpus_id": "corpus-1"},
    )
    assert report["resolved_action_world_model_config"] == asdict(model)
    assert report["resolved_nursery_config"] == asdict(nursery)
    assert report["resolved_data_config"] == {"corpus_id": "corpus-1"}
