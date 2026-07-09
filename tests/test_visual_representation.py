from __future__ import annotations

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.cli import build_parser  # noqa: E402
from cognitive_runtime.neural import checkpoint_metadata_path  # noqa: E402
from cognitive_runtime.policies import ScriptedSurvivalPolicy  # noqa: E402
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox  # noqa: E402
from cognitive_runtime.runtime.config import RuntimeConfig  # noqa: E402
from cognitive_runtime.runtime.loop import CognitiveRuntime  # noqa: E402
from cognitive_runtime.training.datasets import (  # noqa: E402
    NeuralDataset,
    PixelSequenceDataset,
    build_pixel_sequence_dataset,
)
from cognitive_runtime.training.neural import train_neural_bc  # noqa: E402
from cognitive_runtime.training.visual_representation import (  # noqa: E402
    VisualPretrainingConfig,
    load_pretrained_pixel_encoder,
    save_pixel_encoder_pretraining_checkpoint,
    train_pixel_encoder_pretraining,
)


def _synthetic_sequence_dataset(n: int = 8) -> PixelSequenceDataset:
    frames = []
    next_frames = []
    for i in range(n):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[..., 0] = 20 + i * 20
        frame[..., 1] = 255 - i * 15
        frame[i % 8, :, 2] = 180
        next_frame = frame.copy()
        next_frame[:, (i + 1) % 8, 2] = 255
        frames.append(frame)
        next_frames.append(next_frame)
    return PixelSequenceDataset(
        pixels=frames,
        next_pixels=next_frames,
        pixel_shape=(8, 8, 3),
        layout_hash="synthetic-layout",
        sources=["synthetic"],
    )


def _train_single_loss(loss_name: str) -> list[float]:
    weights = {
        "reconstruction_loss": (1.0, 0.0, 0.0),
        "next_latent_loss": (0.0, 1.0, 0.0),
        "contrastive_loss": (0.0, 0.0, 1.0),
    }[loss_name]
    _model, stats = train_pixel_encoder_pretraining(
        _synthetic_sequence_dataset(),
        VisualPretrainingConfig(
            epochs=10,
            lr=5e-3,
            batch_size=8,
            seed=3,
            latent_width=8,
            hidden_dim=32,
            reconstruction_size=4,
            reconstruction_weight=weights[0],
            next_latent_weight=weights[1],
            contrastive_weight=weights[2],
            contrastive_temperature=0.5,
        ),
    )
    return stats["loss_curves"][loss_name]


@pytest.mark.parametrize(
    "loss_name",
    ["reconstruction_loss", "next_latent_loss", "contrastive_loss"],
)
def test_visual_representation_loss_decreases_on_synthetic_frames(loss_name):
    curve = _train_single_loss(loss_name)
    assert curve[-1] < curve[0]


def test_pixel_encoder_pretraining_checkpoint_loads_as_neural_bc_init(tmp_path):
    dataset = _synthetic_sequence_dataset()
    model, stats = train_pixel_encoder_pretraining(
        dataset,
        VisualPretrainingConfig(
            epochs=2,
            lr=1e-3,
            batch_size=4,
            seed=1,
            latent_width=8,
            hidden_dim=32,
            reconstruction_size=4,
        ),
    )
    checkpoint = os.path.join(str(tmp_path), "pixel_encoder.pt")
    save_pixel_encoder_pretraining_checkpoint(checkpoint, model, dataset, stats)

    loaded = load_pretrained_pixel_encoder(checkpoint, pixel_shape=(8, 8, 3), latent_width=8)
    neural = NeuralDataset(
        pixels=list(dataset.pixels),
        non_vision=[[] for _ in dataset.pixels],
        motor=[[] for _ in dataset.pixels],
        labels=[0 for _ in dataset.pixels],
        pixel_shape=(8, 8, 3),
        layout_hash="synthetic-non-vision",
    )
    bc_model, metrics = train_neural_bc(
        neural,
        epochs=0,
        batch_size=4,
        seed=2,
        embed_dim=8,
        encoder_init_path=checkpoint,
    )

    assert metrics["encoder_initialized"] == 1.0
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, bc_model.net.encoder.state_dict()[key])


def test_cli_pixel_encoder_pretraining_writes_checkpoint_bundle(tmp_path):
    session_id = "pixel-pretrain"
    config = {"episode_ticks": 30, "world_size": 16}
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=0,
        max_ticks_per_episode=config["episode_ticks"],
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
    session_dir = os.path.join(str(tmp_path), session_id)
    assert len(build_pixel_sequence_dataset([session_dir])) >= config["episode_ticks"] - 1

    out = os.path.join(str(tmp_path), "pretrained.pt")
    args = build_parser().parse_args(
        [
            "train",
            "--model-type",
            "pixel-encoder",
            "--sessions",
            session_dir,
            "--out",
            out,
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--latent-width",
            "8",
            "--hidden-dim",
            "32",
            "--reconstruction-size",
            "4",
        ]
    )
    args.func(args)

    assert os.path.exists(out)
    sidecar = checkpoint_metadata_path(out)
    assert os.path.exists(sidecar)
    with open(sidecar, encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["extra"]["model_type"] == "pixel-encoder"
    assert metadata["training_stats"]["loss_curves"]["reconstruction_loss"]
    assert load_pretrained_pixel_encoder(out, latent_width=8).pixel_shape
