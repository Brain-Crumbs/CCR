"""Phase D: action-conditioned world model (issue #26)."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.neural import CheckpointCompatibilityError, MLPWorldModel  # noqa: E402
from cognitive_runtime.neural.world_model import WorldModelOutput  # noqa: E402
from cognitive_runtime.policies import NullPolicy, ScriptedSurvivalPolicy  # noqa: E402
from cognitive_runtime.policies.neural_world_model import NeuralWorldModel  # noqa: E402
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox  # noqa: E402
from cognitive_runtime.runtime.config import RuntimeConfig  # noqa: E402
from cognitive_runtime.runtime.loop import CognitiveRuntime  # noqa: E402
from cognitive_runtime.runtime.replay import iter_cognitive_ticks  # noqa: E402
from cognitive_runtime.training.datasets import build_world_model_dataset  # noqa: E402
from cognitive_runtime.training.world_model import (  # noqa: E402
    WorldModelTrainingConfig,
    death_prediction_auc,
    load_world_model_checkpoint,
    save_world_model_checkpoint,
    train_world_model,
)


def _record_session(
    tmp_path,
    session_id: str,
    *,
    policy,
    ticks: int,
    seed: int = 0,
    max_mobs: int = 3,
    world_size: int = 16,
):
    config = {"episode_ticks": ticks, "world_size": world_size, "max_mobs": max_mobs}
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=seed,
        max_ticks_per_episode=ticks,
        record_dir=str(tmp_path),
        session_id=session_id,
        program_config=config,
    )
    summaries = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config),
        policy=policy,
        config=runtime_config,
    ).run()
    return os.path.join(str(tmp_path), session_id), summaries


def test_mlp_world_model_forward_shapes_and_checkpoint_metadata():
    torch.manual_seed(0)
    model = MLPWorldModel(
        fused_width=6, n_actions=3, hidden_dim=8, depth=2,
        layout_hash="layout-x", action_keys=["a", "b", "c"],
    )
    fused = torch.randn(4, 6)
    actions = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 0]), num_classes=3).float()

    out = model(fused, actions)

    assert isinstance(out, WorldModelOutput)
    assert out.next_latent.shape == (4, 6)
    assert out.reward.shape == (4,)
    assert out.terminal_logit.shape == (4,)
    assert out.risk.shape == (4,)
    assert out.prediction_error.shape == (4,)
    assert bool((out.prediction_error >= 0).all())

    meta = model.checkpoint_metadata()
    assert meta["fused_width"] == 6
    assert meta["n_actions"] == 3
    assert meta["action_keys"] == ["a", "b", "c"]

    with pytest.raises(ValueError):
        model(torch.randn(4, 5), actions)
    with pytest.raises(ValueError):
        model(fused, torch.randn(4, 2))


def test_build_world_model_dataset_from_recorded_session(tmp_path):
    session_dir, _ = _record_session(
        tmp_path, "wm-dataset", policy=ScriptedSurvivalPolicy(seed=1), ticks=80
    )
    dataset = build_world_model_dataset([session_dir], max_samples=32)

    assert len(dataset) == 32
    assert len(dataset.latents[0]) == len(dataset.feature_names)
    assert len(dataset.next_latents) == len(dataset)
    assert all(0 <= label < len(dataset.action_keys) for label in dataset.labels)
    assert all(d in (0.0, 1.0) for d in dataset.dones)
    assert all(r in (0.0, 1.0) for r in dataset.risks)
    assert dataset.layout_hash is not None


def test_world_model_training_decreases_all_losses_and_checkpoints(tmp_path):
    session_dir, _ = _record_session(
        tmp_path, "wm-train", policy=ScriptedSurvivalPolicy(seed=2), ticks=200
    )
    dataset = build_world_model_dataset([session_dir], max_samples=128)
    assert len(dataset) > 0

    model, stats = train_world_model(
        dataset,
        WorldModelTrainingConfig(
            epochs=15, lr=5e-3, batch_size=16, seed=4, hidden_dim=32, depth=2,
        ),
    )

    for key in (
        "next_latent_loss", "reward_loss", "death_loss", "risk_loss",
        "prediction_error_loss",
    ):
        assert stats[f"{key}_decreased"] is True, key

    path = os.path.join(str(tmp_path), "world_model.pt")
    metadata = save_world_model_checkpoint(path, model, dataset, stats)
    loaded, loaded_metadata = load_world_model_checkpoint(
        path, expected_layout_hash=dataset.layout_hash
    )

    assert (
        metadata["modules"]["world_model"]["checkpoint_metadata"]["layout_hash"]
        == dataset.layout_hash
    )
    assert loaded_metadata["training_stats"]["loss_curves"]["total_loss"]
    assert loaded.fused_width() == model.fused_width()

    with pytest.raises(CheckpointCompatibilityError, match="layout"):
        load_world_model_checkpoint(path, expected_layout_hash="different-layout")


def test_death_prediction_auc_ranks_pre_death_ticks_higher_on_held_out_episode(tmp_path):
    """Acceptance criterion #2: on held-out ticks preceding a recorded death,
    p_death should rank higher than on random (non-death) ticks.  A NullPolicy
    with no mobs dies deterministically to starvation, giving a reliable
    positive example without depending on combat RNG."""
    train_dir, train_summaries = _record_session(
        tmp_path, "wm-starve-train", policy=NullPolicy(), ticks=6650, max_mobs=0, seed=0,
    )
    assert train_summaries[0].termination_reason == "death:starvation"

    holdout_dir, holdout_summaries = _record_session(
        tmp_path, "wm-starve-holdout", policy=NullPolicy(), ticks=6650, max_mobs=0, seed=1,
    )
    assert holdout_summaries[0].termination_reason == "death:starvation"

    train_dataset = build_world_model_dataset([train_dir])
    holdout_dataset = build_world_model_dataset([holdout_dir])
    assert train_dataset.death_count() >= 1
    assert holdout_dataset.death_count() >= 1

    model, _stats = train_world_model(
        train_dataset,
        WorldModelTrainingConfig(
            epochs=20, lr=5e-3, batch_size=64, seed=7, hidden_dim=64, depth=2,
        ),
    )

    auc = death_prediction_auc(model, holdout_dataset)
    assert auc > 0.5


def test_neural_world_model_bridges_prediction_into_recorded_session(tmp_path):
    seed_dir, _ = _record_session(
        tmp_path, "wm-bridge-seed", policy=ScriptedSurvivalPolicy(seed=3), ticks=80
    )
    dataset = build_world_model_dataset([seed_dir], max_samples=32)
    model, stats = train_world_model(
        dataset,
        WorldModelTrainingConfig(epochs=2, lr=1e-3, batch_size=16, seed=1, hidden_dim=16, depth=1),
    )
    path = os.path.join(str(tmp_path), "bridge_world_model.pt")
    save_world_model_checkpoint(path, model, dataset, stats)

    world_model = NeuralWorldModel(path, action_keys=dataset.action_keys)
    config = {"episode_ticks": 40, "world_size": 16}
    session_id = "wm-bridge-run"
    runtime_config = RuntimeConfig(
        episodes=1, seed=5, max_ticks_per_episode=40,
        record_dir=str(tmp_path), session_id=session_id, program_config=config,
    )
    summaries = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config),
        policy=ScriptedSurvivalPolicy(seed=6),
        config=runtime_config,
        world_model=world_model,
    ).run()

    assert summaries[0].avg_prediction_error is not None

    session_dir = os.path.join(str(tmp_path), session_id)
    saw_p_death = False
    for decision, _sensory, _motor in iter_cognitive_ticks(session_dir, summaries[0].episode_id):
        if decision.get("p_death") is not None:
            saw_p_death = True
            assert 0.0 <= decision["risk"] <= 1.0
    assert saw_p_death
