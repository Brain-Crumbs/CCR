from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec  # noqa: E402
from cognitive_runtime.models.online_q import motor_history_features_for_actions  # noqa: E402
from cognitive_runtime.neural import (  # noqa: E402
    AUDIO_CHECKPOINT_KEY,
    BODY_STATE_CHECKPOINT_KEY,
    ENTITY_CHECKPOINT_KEY,
    MOTOR_HISTORY_CHECKPOINT_KEY,
    REWARD_CHECKPOINT_KEY,
    AudioEncoder,
    BodyStateEncoder,
    EntityEncoder,
    MotorHistoryEncoder,
    NeuralAgentCheckpoint,
    RewardEncoder,
)
from cognitive_runtime.training.features import (  # noqa: E402
    ACTION_KEYS,
    motor_history_features,
)


def _assert_shape_and_determinism(encoder, events, spec):
    encoder.train_mode()
    train_first = encoder.encode_latent(events, spec)
    train_second = encoder.encode_latent(events, spec)
    encoder.eval_mode()
    eval_first = encoder.encode_latent(events, spec)
    eval_second = encoder.encode_latent(events, spec)

    assert train_first is not None
    assert train_first.shape == (encoder.width(spec),)
    assert torch.allclose(train_first, train_second)
    assert torch.allclose(eval_first, eval_second)
    assert torch.allclose(train_first, eval_first)

    token = encoder.encode(events, spec)
    assert token is not None
    assert len(token.vector) == encoder.width(spec)


def test_phase_bc_encoders_encode_declared_shapes_and_are_deterministic():
    torch.manual_seed(7)
    action_keys = ["noop", "forward", "attack"]
    cases = [
        (
            MotorHistoryEncoder(action_keys=action_keys, latent_width=6),
            [
                StreamEvent("motor.history", "motor", 1.0, 1, ["noop", "attack"]),
            ],
            StreamSpec("motor.history", "motor"),
        ),
        (
            BodyStateEncoder(latent_width=5),
            [
                StreamEvent("body.health", "body", 1.0, 1, 18.0),
                StreamEvent("body.health", "body", 2.0, 2, 16.0),
            ],
            StreamSpec("body.health", "body", range=(0.0, 20.0), neutral=20.0),
        ),
        (
            RewardEncoder(latent_width=5),
            [
                StreamEvent("reward.scalar", "reward", 1.0, 1, {"value": 0.25}),
                StreamEvent("reward.scalar", "reward", 2.0, 2, {"value": 1.0}),
            ],
            StreamSpec("reward.scalar", "reward", range=(-2.0, 2.0)),
        ),
        (
            EntityEncoder(latent_width=7),
            [
                StreamEvent(
                    "vision.entities",
                    "vision",
                    1.0,
                    1,
                    [{"kind": "zombie", "distance": 4.0, "angle": 45.0}],
                )
            ],
            StreamSpec("vision.entities", "vision", range=(0.0, 16.0)),
        ),
        (
            AudioEncoder(latent_width=4),
            [StreamEvent("audio.ambient", "audio", 1.0, 1, None)],
            StreamSpec("audio.ambient", "audio"),
        ),
    ]

    for encoder, events, spec in cases:
        _assert_shape_and_determinism(encoder, events, spec)


def test_phase_bc_encoders_round_trip_through_neural_checkpoint(tmp_path):
    path = tmp_path / "phase-bc-agent.pt"
    torch.manual_seed(11)
    action_keys = ["noop", "forward", "attack"]
    encoders = {
        MOTOR_HISTORY_CHECKPOINT_KEY: MotorHistoryEncoder(action_keys=action_keys, latent_width=6),
        BODY_STATE_CHECKPOINT_KEY: BodyStateEncoder(latent_width=5),
        REWARD_CHECKPOINT_KEY: RewardEncoder(latent_width=5),
        ENTITY_CHECKPOINT_KEY: EntityEncoder(latent_width=7),
        AUDIO_CHECKPOINT_KEY: AudioEncoder(latent_width=4),
    }
    events = {
        MOTOR_HISTORY_CHECKPOINT_KEY: (
            [StreamEvent("motor.history", "motor", 1.0, 1, ["noop", "attack"])],
            StreamSpec("motor.history", "motor"),
        ),
        BODY_STATE_CHECKPOINT_KEY: (
            [StreamEvent("body.health", "body", 1.0, 1, 16.0)],
            StreamSpec("body.health", "body", range=(0.0, 20.0), neutral=20.0),
        ),
        REWARD_CHECKPOINT_KEY: (
            [StreamEvent("reward.scalar", "reward", 1.0, 1, {"value": 1.0})],
            StreamSpec("reward.scalar", "reward", range=(-2.0, 2.0)),
        ),
        ENTITY_CHECKPOINT_KEY: (
            [
                StreamEvent(
                    "body.inventory",
                    "body",
                    1.0,
                    1,
                    {"berries": 2, "log": 1},
                )
            ],
            StreamSpec("body.inventory", "body"),
        ),
        AUDIO_CHECKPOINT_KEY: (
            [StreamEvent("audio.ambient", "audio", 1.0, 1, None)],
            StreamSpec("audio.ambient", "audio"),
        ),
    }
    for encoder in encoders.values():
        encoder.eval_mode()
    before = {
        key: encoders[key].encode_latent(*events[key]).detach().clone()
        for key in encoders
    }

    metadata = NeuralAgentCheckpoint(
        str(path),
        layout_hash="phase-bc-layout",
        action_keys=action_keys,
        encoders=encoders,
    ).save(reason="phase-bc-encoders")

    torch.manual_seed(999)
    restored = {
        MOTOR_HISTORY_CHECKPOINT_KEY: MotorHistoryEncoder(action_keys=action_keys, latent_width=6),
        BODY_STATE_CHECKPOINT_KEY: BodyStateEncoder(latent_width=5),
        REWARD_CHECKPOINT_KEY: RewardEncoder(latent_width=5),
        ENTITY_CHECKPOINT_KEY: EntityEncoder(latent_width=7),
        AUDIO_CHECKPOINT_KEY: AudioEncoder(latent_width=4),
    }
    NeuralAgentCheckpoint(
        str(path),
        layout_hash="phase-bc-layout",
        action_keys=action_keys,
        encoders=restored,
    ).load()
    for encoder in restored.values():
        encoder.eval_mode()
    after = {
        key: restored[key].encode_latent(*events[key]).detach().clone()
        for key in restored
    }

    assert metadata["modules"]["encoders"][AUDIO_CHECKPOINT_KEY]["checkpoint_metadata"][
        "fixed_stub"
    ] is True
    for key in before:
        assert torch.allclose(before[key], after[key])


def test_motor_history_encoder_parity_mode_matches_handcrafted_online_q_and_bc_features():
    recent = [ACTION_KEYS[0], ACTION_KEYS[-1]]
    encoder = MotorHistoryEncoder(action_keys=ACTION_KEYS, parity_mode=True)
    event = StreamEvent(
        "motor.history",
        "motor",
        1.0,
        1,
        {"recent_action_keys": recent},
    )
    spec = StreamSpec("motor.history", "motor")

    latent = encoder.encode_latent([event], spec)
    expected = motor_history_features(recent)

    assert latent is not None
    assert latent.tolist() == expected
    assert latent.tolist() == motor_history_features_for_actions(recent, ACTION_KEYS)
    assert encoder.width(spec) == len(ACTION_KEYS)
