from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec  # noqa: E402
from cognitive_runtime.models.vision import VisionBCModel, VisionPolicyNet  # noqa: E402
from cognitive_runtime.neural import (  # noqa: E402
    PIXEL_CHECKPOINT_KEY,
    PIXEL_STREAM_ID,
    NeuralAgentCheckpoint,
    PixelStreamEncoder,
)


def _frame(shape=(8, 8, 3)):
    values = np.arange(np.prod(shape), dtype=np.uint8)
    return values.reshape(shape)


def test_pixel_stream_encoder_encodes_declared_width_deterministically_in_eval():
    encoder = PixelStreamEncoder((8, 8, 3), latent_width=7)
    encoder.eval_mode()
    event = StreamEvent(PIXEL_STREAM_ID, "vision", 1.0, 1, _frame())
    spec = StreamSpec(PIXEL_STREAM_ID, "vision", shape=(8, 8, 3))

    first = encoder.encode_latent([event], spec)
    second = encoder.encode_latent([event], spec)
    token = encoder.encode([event], spec)

    assert first is not None
    assert first.shape == (7,)
    assert torch.allclose(first, second)
    assert token is not None
    assert len(token.vector) == encoder.width(spec) == 7
    assert encoder.checkpoint_metadata()["checkpoint_key"] == PIXEL_CHECKPOINT_KEY


def test_pixel_stream_encoder_weights_round_trip_through_neural_checkpoint(tmp_path):
    path = tmp_path / "agent.pt"
    encoder = PixelStreamEncoder((8, 8, 3), latent_width=5)
    encoder.eval_mode()
    event = StreamEvent(PIXEL_STREAM_ID, "vision", 1.0, 1, _frame())
    before = encoder.encode_latent([event]).detach()

    manager = NeuralAgentCheckpoint(
        str(path),
        layout_hash="pixel-layout",
        action_keys=["NOOP"],
        encoders={PIXEL_CHECKPOINT_KEY: encoder},
    )
    metadata = manager.save(reason="pixel_encoder")

    restored = PixelStreamEncoder((8, 8, 3), latent_width=5)
    restored.eval_mode()
    NeuralAgentCheckpoint(
        str(path),
        layout_hash="pixel-layout",
        action_keys=["NOOP"],
        encoders={PIXEL_CHECKPOINT_KEY: restored},
    ).load()
    after = restored.encode_latent([event]).detach()

    assert metadata["modules"]["encoders"][PIXEL_CHECKPOINT_KEY]["checkpoint_metadata"][
        "latent_width"
    ] == 5
    assert torch.allclose(before, after)


def test_vision_bc_model_loads_legacy_cnn_key_bundles(tmp_path):
    pixel_shape = (8, 8, 3)
    net = VisionPolicyNet(pixel_shape, n_non_vision=2, n_motor=3, n_actions=4)
    legacy_state = {}
    for key, value in net.state_dict().items():
        if key.startswith("encoder.cnn."):
            legacy_state[key.removeprefix("encoder.")] = value
        else:
            legacy_state[key] = value

    path = os.path.join(str(tmp_path), "legacy_vision_bc.pt")
    torch.save(
        {
            "config": net.config(),
            "state_dict": legacy_state,
            "action_keys": ["A", "B", "C", "D"],
            "meta": {},
        },
        path,
    )

    loaded = VisionBCModel.load(path)
    assert loaded.net.encoder.width() == net.embed_dim
    for key, value in net.state_dict().items():
        assert torch.equal(value, loaded.net.state_dict()[key])
