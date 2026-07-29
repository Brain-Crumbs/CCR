"""Regression coverage for the resumable Model Factory checkpoint format."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from brain.cortex.predictive import PredictiveCortex, PredictiveCortexConfig  # noqa: E402
from cognitive_runtime.training.action_world_model import save_action_world_model  # noqa: E402
from cognitive_runtime.training.model_factory.checkpoint import (  # noqa: E402
    FORMAT,
    load_factory_checkpoint,
    save_factory_checkpoint,
)
from cognitive_runtime.training.model_factory import checkpoint as checkpoint_module  # noqa: E402
from cognitive_runtime.training.model_factory.contracts import (  # noqa: E402
    ArchitectureContract,
    DataContract,
    TrainingContract,
)


def _contracts(model):
    architecture = ArchitectureContract(
        pixel_shape=tuple(model.pixel_shape), rgb_preprocessing_version="v1",
        action_vocabulary=tuple(model.action_keys), workspace_modalities={},
        workspace_layout_hash=None, latent_width=model.latent_width,
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
        world="test", backend="test", program_config={}, scenario_names=(),
        scenario_code_version="v1", train_session_ids=(), train_session_hashes=(),
        validation_session_ids=(), validation_session_hashes=(), test_session_ids=(),
        test_session_hashes=(), seed_assignments={}, pixel_provenance="test",
        semantic_vocabulary_version="v1", preprocessing_version="v1",
        horizons_ticks=tuple(model.horizons_ticks), ticks_per_frame=1.0,
    )
    training = TrainingContract(
        objective="test", optimizer={
            "name": "AdamW", "lr": 0.01, "betas": (0.9, 0.999),
            "eps": 1e-8, "weight_decay": 0.01, "grad_clip": 1.0,
        }, batch_size=2, seed=7, rollout_frames=2, warmup_frames=1,
        scheduled_sampling_p=0.0, loss_weights={},
        checkpoint_selection_policy={}, device="cpu", precision="fp32",
        determinism_policy={"seed": 7}, scheduler={"name": "StepLR", "gamma": 0.5},
    )
    return architecture, data, training


def _model():
    return PredictiveCortex(
        (4, 4, 3), ["NULL", "LEFT"], PredictiveCortexConfig(
            latent_width=4, hidden_dim=8, action_embed_dim=3, reconstruction_size=4,
            horizons_ticks=(1, 2), backbone="transformer", context_length=3,
            backbone_kwargs={"n_heads": 2, "n_layers": 1},
        )
    )


def _assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], torch.Tensor):
            assert torch.equal(left[key], right[key])
        elif isinstance(left[key], dict):
            _assert_state_equal(left[key], right[key])
        else:
            assert left[key] == right[key]


def test_factory_checkpoint_round_trips_every_resumable_state(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    loss = sum(parameter.square().mean() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    scheduler.step()
    architecture, data, training = _contracts(model)
    path = tmp_path / "factory.pt"

    random.seed(99)
    torch.manual_seed(99)
    payload = save_factory_checkpoint(
        str(path), model, optimizer, scheduler,
        {"epoch": 3, "global_step": 17, "best_validation_metric": 0.25},
        architecture_contract=architecture, data_contract_hash=data,
        training_contract=training, parent_checkpoint_sha="parent-sha",
        training_stats={"validation_loss": 0.25},
    )
    expected_python = random.random()
    expected_torch = torch.rand(3)
    random.random()
    torch.rand(3)

    target = _model()
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=1.0)
    target_scheduler = torch.optim.lr_scheduler.StepLR(target_optimizer, step_size=2)
    loaded = load_factory_checkpoint(
        str(path), model=target, optimizer=target_optimizer,
        scheduler=target_scheduler, resume=True,
        architecture_contract=architecture, data_contract_hash=data,
        training_contract=training,
    )

    assert payload["format"] == FORMAT
    assert loaded.resumed is True
    assert loaded.trainer_state == {"epoch": 3, "global_step": 17, "best_validation_metric": 0.25}
    assert loaded.training_stats == {"validation_loss": 0.25}
    assert loaded["parent_checkpoint_sha"] == "parent-sha"
    assert loaded["architecture_contract"]["hash"] == architecture.hash
    _assert_state_equal(model.state_dict(), target.state_dict())
    _assert_state_equal(target_optimizer.state_dict(), optimizer.state_dict())
    _assert_state_equal(target_scheduler.state_dict(), scheduler.state_dict())
    assert random.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)


def test_factory_clone_preserves_non_default_backbone_kwargs_and_action_embedding(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    architecture, data, training = _contracts(model)
    path = tmp_path / "factory.pt"
    save_factory_checkpoint(
        str(path), model, optimizer, architecture_contract=architecture,
        data_contract_hash=data.hash, training_contract=training,
    )

    clone = load_factory_checkpoint(str(path)).model

    assert clone.config.action_embed_dim == 3
    assert clone.config.backbone_kwargs == {"n_heads": 2, "n_layers": 1}
    assert type(clone.transition_backbone) is type(model.transition_backbone)
    _assert_state_equal(model.state_dict(), clone.state_dict())


def test_resume_rejects_any_changed_contract_before_restoring_state(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    architecture, data, training = _contracts(model)
    path = tmp_path / "factory.pt"
    save_factory_checkpoint(
        str(path), model, optimizer, architecture_contract=architecture,
        data_contract_hash=data, training_contract=training,
    )

    changed_architecture_model = _model()
    with pytest.raises(ValueError, match="data contract"):
        load_factory_checkpoint(
            str(path), model=changed_architecture_model,
            optimizer=torch.optim.AdamW(changed_architecture_model.parameters()),
            resume=True, architecture_contract=architecture,
            data_contract_hash="different-data", training_contract=training,
        )
    changed_architecture_model = _model()
    with pytest.raises(ValueError, match="architecture contract"):
        load_factory_checkpoint(
            str(path), model=changed_architecture_model,
            optimizer=torch.optim.AdamW(changed_architecture_model.parameters()),
            resume=True,
            architecture_contract=replace(architecture, action_vocabulary=("LEFT", "NULL")),
            data_contract_hash=data, training_contract=training,
        )
    changed_training_model = _model()
    with pytest.raises(ValueError, match="training contract"):
        load_factory_checkpoint(
            str(path), model=changed_training_model,
            optimizer=torch.optim.AdamW(changed_training_model.parameters()),
            resume=True, architecture_contract=architecture, data_contract_hash=data,
            training_contract=replace(training, seed=99),
        )


def test_factory_clone_restores_training_objective(tmp_path):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    architecture, data, training = _contracts(model)
    training = replace(training, objective="autoregressive")
    path = tmp_path / "factory.pt"
    save_factory_checkpoint(
        str(path), model, optimizer, architecture_contract=architecture,
        data_contract_hash=data, training_contract=training,
    )

    assert load_factory_checkpoint(str(path)).model.training_objective == "autoregressive"


def test_factory_save_keeps_existing_checkpoint_when_serialization_fails(tmp_path, monkeypatch):
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    architecture, data, training = _contracts(model)
    path = tmp_path / "factory.pt"
    save_factory_checkpoint(
        str(path), model, optimizer, architecture_contract=architecture,
        data_contract_hash=data, training_contract=training,
    )
    original = path.read_bytes()
    real_save = torch.save

    def interrupted_save(payload, temporary_path):
        real_save(payload, temporary_path)
        raise OSError("simulated full filesystem")

    monkeypatch.setattr(torch, "save", interrupted_save)
    with pytest.raises(OSError, match="full filesystem"):
        save_factory_checkpoint(
            str(path), model, optimizer, architecture_contract=architecture,
            data_contract_hash=data, training_contract=training,
        )
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".factory.pt.*.tmp"))


def test_legacy_v2_can_clone_but_cannot_resume(tmp_path):
    model = _model()
    # The legacy serializer cannot represent the non-default transformer
    # kwargs above, so use its default-compatible shape for this migration test.
    legacy = PredictiveCortex((4, 4, 3), ["NULL"], PredictiveCortexConfig(
        latent_width=4, hidden_dim=8, reconstruction_size=4,
    ))
    path = tmp_path / "legacy-v2.pt"
    save_action_world_model(str(path), legacy, {"loss": 1.0})

    cloned = load_factory_checkpoint(str(path))
    assert cloned.model is not None
    assert cloned.training_stats == {"loss": 1.0}
    with pytest.raises(ValueError, match="cannot be resumed"):
        load_factory_checkpoint(str(path), model=model, resume=True)
