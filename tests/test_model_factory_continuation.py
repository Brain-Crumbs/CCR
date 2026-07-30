"""Header-first continuation checks for Model Factory checkpoints."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from brain.cortex.predictive import PredictiveCortex, PredictiveCortexConfig  # noqa: E402
from cognitive_runtime.training.model_factory import checkpoint as checkpoint_module  # noqa: E402
from cognitive_runtime.training.model_factory.checkpoint import (  # noqa: E402
    factory_checkpoint_metadata_path,
    load_factory_checkpoint,
    save_factory_checkpoint,
    validate_continuation,
)
from cognitive_runtime.training.model_factory.artifacts import allocate_run_artifacts  # noqa: E402
from cognitive_runtime.training.model_factory.naming import DisplayNameParent  # noqa: E402
from cognitive_runtime.training.model_factory.contracts import (  # noqa: E402
    ArchitectureContract,
    DataContract,
    TrainingContract,
)
from cognitive_runtime.training.model_factory.spec import resolve  # noqa: E402


def _model():
    return PredictiveCortex(
        (4, 4, 3), ["NULL", "LEFT"], PredictiveCortexConfig(
            latent_width=4, hidden_dim=8, action_embed_dim=3, reconstruction_size=4,
            horizons_ticks=(1, 2), backbone="transformer", context_length=3,
            backbone_kwargs={"n_heads": 2, "n_layers": 1},
        )
    )


def _contracts(model):
    architecture = ArchitectureContract(
        pixel_shape=tuple(model.pixel_shape), rgb_preprocessing_version="v1",
        action_vocabulary=tuple(model.action_keys), workspace_modalities={},
        workspace_layout_hash="layout-a", latent_width=model.latent_width,
        hidden_dim=model.hidden_dim, action_embed_dim=model.config.action_embed_dim,
        reconstruction_shape=tuple(model.reconstruction_shape),
        visual_architecture=model.visual_architecture,
        semantic_classes=model.config.semantic_classes,
        horizons_ticks=tuple(model.horizons_ticks),
        direct_horizon_topology=tuple(model.horizons_ticks),
        backbone=model.config.backbone, context_length=model.config.context_length,
        backbone_kwargs=dict(model.config.backbone_kwargs),
    )
    data = DataContract(
        world="test", backend="test", program_config={}, scenario_names=("scenario-a",),
        scenario_code_version="v1", train_session_ids=("train-a",), train_session_hashes=("a",),
        validation_session_ids=(), validation_session_hashes=(), test_session_ids=(),
        test_session_hashes=(), seed_assignments={}, pixel_provenance="test",
        semantic_vocabulary_version="v1", preprocessing_version="v1",
        horizons_ticks=tuple(model.horizons_ticks), ticks_per_frame=1.0,
    )
    training = TrainingContract(
        objective="test", optimizer={"name": "AdamW", "lr": 0.01},
        batch_size=2, seed=7, rollout_frames=2, warmup_frames=1,
        scheduled_sampling_p=0.0, loss_weights={}, checkpoint_selection_policy={},
        device="cpu", precision="fp32", determinism_policy={"seed": 7},
    )
    return architecture, data, training


def _save(path):
    model = _model()
    architecture, data, training = _contracts(model)
    save_factory_checkpoint(
        str(path), model, torch.optim.AdamW(model.parameters(), lr=0.01),
        architecture_contract=architecture, data_contract_hash=data, training_contract=training,
    )
    return model, architecture, data, training


@pytest.mark.parametrize(
    ("changed", "field"),
    (
        (lambda architecture: replace(architecture, action_vocabulary=("LEFT", "NULL")), "architecture.action_vocabulary"),
        (lambda architecture: replace(architecture, workspace_layout_hash="layout-b"), "architecture.workspace_layout_hash"),
        (lambda architecture: replace(architecture, backbone_kwargs={"n_heads": 1, "n_layers": 1}), "architecture.backbone_kwargs.n_heads"),
    ),
)
def test_architecture_changes_are_rejected_with_field_diagnostics(changed, field):
    model = _model()
    architecture, data, training = _contracts(model)
    decision = validate_continuation(
        {"architecture_contract": architecture, "data_contract_hash": data, "training_contract": training},
        {"architecture_contract": changed(architecture), "data_contract_hash": data, "training_contract": training},
        "resume",
    )

    assert not decision.approved
    assert decision.mismatches[0].field == field
    assert field in decision.message


def test_resume_training_mismatch_is_rejected_before_any_tensor_load(tmp_path, monkeypatch):
    path = tmp_path / "factory.pt"
    _model, architecture, data, training = _save(path)
    attempted_tensor_load = False

    def tensor_load_spy(*args, **kwargs):
        nonlocal attempted_tensor_load
        attempted_tensor_load = True
        raise AssertionError("torch.load must not be reached")

    monkeypatch.setattr(torch, "load", tensor_load_spy)
    with pytest.raises(ValueError, match="use mode=fine_tune") as error:
        load_factory_checkpoint(
            str(path), mode="resume", architecture_contract=architecture,
            data_contract_hash=data, training_contract=replace(training, optimizer={"name": "AdamW", "lr": 0.0003}),
        )

    assert "training.optimizer.lr" in str(error.value)
    assert attempted_tensor_load is False


def test_stale_header_digest_rejects_an_overwritten_checkpoint_before_tensor_load(tmp_path, monkeypatch):
    path = tmp_path / "factory.pt"
    model, architecture, data, training = _save(path)
    header_path = factory_checkpoint_metadata_path(str(path))
    stale_header = Path(header_path).read_text(encoding="utf-8")
    save_factory_checkpoint(
        str(path), model, torch.optim.AdamW(model.parameters(), lr=0.01),
        architecture_contract=architecture, data_contract_hash=data,
        training_contract=replace(training, seed=99),
    )
    Path(header_path).write_text(stale_header, encoding="utf-8")

    def tensor_load_spy(*args, **kwargs):
        raise AssertionError("torch.load must not be reached")

    monkeypatch.setattr(torch, "load", tensor_load_spy)
    with pytest.raises(ValueError, match="bytes do not match"):
        load_factory_checkpoint(
            str(path), mode="resume", architecture_contract=architecture,
            data_contract_hash=data, training_contract=training,
        )


def test_payload_generation_mismatch_rejects_before_model_or_state_restoration(tmp_path, monkeypatch):
    path = tmp_path / "factory.pt"
    model, architecture, data, training = _save(path)
    header_path = factory_checkpoint_metadata_path(str(path))
    stale_header = json.loads(Path(header_path).read_text(encoding="utf-8"))
    save_factory_checkpoint(
        str(path), model, torch.optim.AdamW(model.parameters(), lr=0.01),
        architecture_contract=architecture, data_contract_hash=data,
        training_contract=replace(training, seed=99),
    )
    # Simulate a writer replacing the payload after the reader has checked the
    # header digest.  Updating only the digest leaves the old generation ID.
    stale_header["checkpoint_sha256"] = checkpoint_module._checkpoint_sha256(str(path))
    Path(header_path).write_text(json.dumps(stale_header), encoding="utf-8")

    monkeypatch.setattr(
        checkpoint_module, "_build_model",
        lambda definition: (_ for _ in ()).throw(AssertionError("model must not be built")),
    )
    with pytest.raises(ValueError, match="header identity"):
        load_factory_checkpoint(
            str(path), mode="resume", architecture_contract=architecture,
            data_contract_hash=data, training_contract=training,
        )


def test_changed_data_requires_fine_tune_and_records_fine_tune_lineage(tmp_path):
    path = tmp_path / "factory.pt"
    _model, architecture, data, training = _save(path)
    changed_data = replace(data, scenario_names=("scenario-b",))

    with pytest.raises(ValueError, match="use mode=fine_tune"):
        load_factory_checkpoint(
            str(path), mode="clone", architecture_contract=architecture,
            data_contract_hash=changed_data, training_contract=training,
        )

    loaded = load_factory_checkpoint(
        str(path), mode="fine_tune", architecture_contract=architecture,
        data_contract_hash=changed_data, training_contract=training,
    )
    assert loaded.continuation is not None
    assert loaded.continuation.lineage_mode == "fine_tune"
    assert loaded.resumed is False

    spec = resolve({
        "organism": "Test", "mode": "fine_tune",
        "parent": {"run_id": "parent", "sha256": "parent-sha"},
        "data": {"corpus_id": "corpus-b", "world": "test"},
        "training": {"optimizer": {"name": "adamw", "lr": 0.0003}},
        "evaluation": {"selection_metric": "validation_loss"},
    })
    artifact_training = TrainingContract(**dict(spec.training))
    artifacts = allocate_run_artifacts(
        tmp_path / "runs", spec, architecture, changed_data, artifact_training,
        run_id="fine-tune-child", naming_parents=(DisplayNameParent("parent", "calm-curie"),),
    )
    assert json.loads(artifacts.lineage_path.read_text(encoding="utf-8"))["mode"] == "fine_tune"
    with pytest.raises(ValueError, match="not within-stage A/B siblings"):
        allocate_run_artifacts(
            tmp_path / "runs", spec, architecture, changed_data, artifact_training,
            run_id="invalid-sibling", sibling_group="ab-group",
        )


def test_factory_loader_preserves_only_the_two_intentional_state_dict_exemptions(tmp_path):
    path = tmp_path / "factory.pt"
    model, architecture, data, _training = _save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    removed = [key for key in payload["model_state_dict"] if key.startswith("multi_token_heads.")]
    assert removed
    for key in removed:
        del payload["model_state_dict"][key]
    payload["model_state_dict"]["transition_backbone.position_embedding.weight"] = torch.zeros(1)
    torch.save(payload, path)
    header_path = Path(factory_checkpoint_metadata_path(str(path)))
    header = json.loads(header_path.read_text(encoding="utf-8"))
    header["checkpoint_sha256"] = checkpoint_module._checkpoint_sha256(str(path))
    header_path.write_text(json.dumps(header), encoding="utf-8")

    loaded = load_factory_checkpoint(
        str(path), mode="clone", architecture_contract=architecture, data_contract_hash=data,
    )
    assert loaded.model is not None

    payload["model_state_dict"]["unexpected.weight"] = torch.zeros(1)
    torch.save(payload, path)
    header["checkpoint_sha256"] = checkpoint_module._checkpoint_sha256(str(path))
    header_path.write_text(json.dumps(header), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected.weight"):
        load_factory_checkpoint(
            str(path), mode="clone", architecture_contract=architecture, data_contract_hash=data,
        )
