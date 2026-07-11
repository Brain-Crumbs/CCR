from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.core.streams import TemporalBuffer, TemporalFusion  # noqa: E402
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec  # noqa: E402
from cognitive_runtime.neural import (  # noqa: E402
    CheckpointCompatibilityError,
    LatentFusionModel,
    latent_fusion_inputs_from_buffer,
)
from cognitive_runtime.policies import ScriptedSurvivalPolicy  # noqa: E402
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox  # noqa: E402
from cognitive_runtime.runtime.config import RuntimeConfig  # noqa: E402
from cognitive_runtime.runtime.loop import CognitiveRuntime  # noqa: E402
from cognitive_runtime.training.datasets import (  # noqa: E402
    build_dataset,
    build_latent_fusion_dataset,
)
from cognitive_runtime.training.fusion import (  # noqa: E402
    FusionTrainingConfig,
    load_latent_fusion_checkpoint,
    save_latent_fusion_checkpoint,
    train_latent_fusion_model,
)


def test_latent_fusion_handles_stream_going_silent_mid_episode():
    fusion = TemporalFusion(
        [
            StreamSpec("body.health", "body", range=(0.0, 20.0), neutral=20.0),
            StreamSpec("reward.scalar", "reward", range=(-1.0, 1.0), neutral=0.0),
        ]
    )
    model = LatentFusionModel.from_temporal_fusion(fusion, fused_width=6, hidden_dim=8)
    buffer = TemporalBuffer()
    buffer.extend(
        [
            StreamEvent("body.health", "body", 0.0, 1, 20.0),
            StreamEvent("reward.scalar", "reward", 0.0, 1, {"value": 0.5}),
        ]
    )
    first = latent_fusion_inputs_from_buffer(
        fusion, buffer, present_stream_ids=["body.health", "reward.scalar"]
    )
    first_out = model(first.latents, first.presence_mask, first.recency, first.staleness)

    buffer.append(StreamEvent("body.health", "body", 1.0, 2, 18.0))
    silent = latent_fusion_inputs_from_buffer(
        fusion, buffer, present_stream_ids=["body.health"]
    )
    silent_out = model(silent.latents, silent.presence_mask, silent.recency, silent.staleness)

    assert first_out.shape == (1, 6)
    assert silent_out.shape == (1, 6)
    assert silent.presence_mask.tolist()[0][silent.stream_ids.index("reward.scalar")] == 0.0


def test_omitting_attention_weights_is_byte_equivalent_to_all_ones():
    """Issue #57 acceptance: the attention-weight hook must reproduce plain
    fusion exactly until an attention controller (#59) actually plugs in."""
    fusion = TemporalFusion(
        [
            StreamSpec("body.health", "body", range=(0.0, 20.0), neutral=20.0),
            StreamSpec("reward.scalar", "reward", range=(-1.0, 1.0), neutral=0.0),
        ]
    )
    model = LatentFusionModel.from_temporal_fusion(fusion, fused_width=6, hidden_dim=8)
    model.eval()
    buffer = TemporalBuffer()
    buffer.extend(
        [
            StreamEvent("body.health", "body", 0.0, 1, 18.0),
            StreamEvent("reward.scalar", "reward", 0.0, 1, {"value": 0.5}),
        ]
    )
    inputs = latent_fusion_inputs_from_buffer(
        fusion, buffer, present_stream_ids=["body.health", "reward.scalar"]
    )

    omitted = model(inputs.latents, inputs.presence_mask, inputs.recency, inputs.staleness)
    ones = torch.ones((1, len(inputs.stream_ids)), dtype=torch.float32)
    explicit = model(
        inputs.latents, inputs.presence_mask, inputs.recency, inputs.staleness, ones
    )
    zeros = torch.zeros((1, len(inputs.stream_ids)), dtype=torch.float32)
    zeroed = model(
        inputs.latents, inputs.presence_mask, inputs.recency, inputs.staleness, zeros
    )

    assert torch.equal(omitted, explicit)
    assert not torch.equal(omitted, zeroed)


def _record_session(tmp_path, session_id: str, ticks: int = 80) -> str:
    config = {"episode_ticks": ticks, "world_size": 16}
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=0,
        max_ticks_per_episode=ticks,
        record_dir=str(tmp_path),
        session_id=session_id,
        program_config=config,
        record_frames=True,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config),
        policy=ScriptedSurvivalPolicy(seed=1),
        config=runtime_config,
    ).run()
    return os.path.join(str(tmp_path), session_id)


def test_latent_fusion_training_on_recorded_session_decreases_losses_and_checkpoints(tmp_path):
    session_dir = _record_session(tmp_path, "fusion-train", ticks=80)
    dataset = build_latent_fusion_dataset([session_dir], max_samples=64)

    assert len(dataset) == 64
    model, stats = train_latent_fusion_model(
        dataset,
        FusionTrainingConfig(
            epochs=12,
            lr=5e-3,
            batch_size=16,
            seed=3,
            fused_width=16,
            hidden_dim=32,
        ),
    )

    assert stats["action_loss_decreased"] is True
    assert stats["reward_loss_decreased"] is True
    assert stats["next_latent_loss_decreased"] is True

    path = os.path.join(str(tmp_path), "latent_fusion.pt")
    metadata = save_latent_fusion_checkpoint(path, model, dataset, stats)
    loaded, loaded_metadata = load_latent_fusion_checkpoint(
        path, expected_layout_hash=dataset.layout_hash
    )

    assert metadata["modules"]["fusion"]["checkpoint_metadata"]["layout_hash"] == dataset.layout_hash
    assert loaded_metadata["training_stats"]["loss_curves"]["action_loss"]
    assert loaded.fused_width() == model.fusion.fused_width()

    with pytest.raises(CheckpointCompatibilityError, match="layout"):
        load_latent_fusion_checkpoint(path, expected_layout_hash="different-layout")


def test_temporal_fusion_remains_default_latent_dataset_path(tmp_path):
    session_dir = _record_session(tmp_path, "temporal-default", ticks=20)
    dataset = build_dataset([session_dir], representation="latent")
    learned = build_latent_fusion_dataset([session_dir], max_samples=4)

    assert dataset.representation == "latent"
    assert dataset.layout_hash == learned.layout_hash
    assert len(dataset.features[0]) == len(dataset.feature_names)
