import pytest

from cognitive_runtime.core.streams import (
    FixedStreamModule,
    ScalarEncoder,
    StreamEvent,
    StreamSpec,
    TemporalBuffer,
    TemporalFusion,
    fixed_stream_module,
)


def _health_spec():
    return StreamSpec(
        "body.health",
        "body",
        range=(0.0, 20.0),
        neutral=20.0,
    )


def test_fixed_stream_module_wraps_existing_encoder_without_behavior_change():
    spec = _health_spec()
    events = [
        StreamEvent("body.health", "body", 0.0, 0, 20.0),
        StreamEvent("body.health", "body", 1.0, 1, 18.0),
    ]
    encoder = ScalarEncoder()
    module = fixed_stream_module(encoder)

    fixed = encoder.encode(events, spec)
    wrapped = module.encode(events, spec)

    assert isinstance(module, FixedStreamModule)
    assert wrapped == fixed
    assert module.width(spec) == encoder.width(spec)
    assert module.neutral(spec) == encoder.neutral(spec)


def test_trainable_module_checkpoint_hooks_are_available_for_fixed_wrappers():
    module = fixed_stream_module(ScalarEncoder())
    module.train_mode()

    assert module.predict_next([1.0, 0.0]) == {}
    assert module.update({"loss": 1.0}) == {}
    assert module.state_dict() == {}
    assert module.checkpoint_payload()["format"] == "trainable-stream-module-v1"
    assert module.checkpoint_payload()["metadata"]["trainable"] is False

    module.load_state_dict({})
    with pytest.raises(ValueError, match="fixed encoder"):
        module.load_state_dict({"weights": [1.0]})


def test_existing_temporal_fusion_stays_unchanged():
    spec = _health_spec()
    fixed = TemporalFusion([spec])

    buffer = TemporalBuffer()
    buffer.append(StreamEvent("body.health", "body", 0.0, 0, 20.0))
    buffer.append(StreamEvent("body.health", "body", 1.0, 1, 18.0))

    first = fixed.fuse(None, buffer)
    second = TemporalFusion([spec]).fuse(None, buffer)

    assert first.layout_hash == second.layout_hash
    assert first.vector == second.vector
    assert first.slices == second.slices

