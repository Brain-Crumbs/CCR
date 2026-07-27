"""Learned fusion + trainable stream encoders (issue #57):
:class:`~cognitive_runtime.neural.live_fusion.LiveLearnedFusion` and
:func:`~cognitive_runtime.neural.live_fusion.build_trainable_encoder_registry`
exercised directly.

The CLI's own live-fusion wiring (``--policy actor-critic --fusion learned``)
was retired by issue #175 along with the rest of the actor/critic online-CLI
path -- the predictive cortex is the only online learner now. `live_fusion.py`
itself is untouched and still directly importable/usable (nothing else in
this repo currently drives it live), so these unit-level tests stay.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry  # noqa: E402
from cognitive_runtime.neural.live_fusion import (  # noqa: E402
    LiveLearnedFusion,
    build_trainable_encoder_registry,
)
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox  # noqa: E402
from cognitive_runtime.programs.minecraft.stream_registry import MINECRAFT_STREAM_REGISTRY  # noqa: E402


def _catalog():
    return MinecraftSurvivalBox(config={"episode_ticks": 10, "world_size": 16}).stream_catalog()


def test_trainable_encoder_registry_gives_each_stream_its_own_module():
    catalog = _catalog()
    registry, encoders = build_trainable_encoder_registry(catalog, MINECRAFT_STREAM_REGISTRY)

    assert encoders  # at least body/reward/entity streams are trainable
    assert "stream_encoder.body_health" in encoders
    assert "stream_encoder.reward_scalar" in encoders
    # Distinct streams sharing a neural_encoder class still get separate,
    # independently-weighted instances (own checkpoint key each).
    assert encoders["stream_encoder.body_health"] is not encoders["stream_encoder.body_hunger"]
    for stream_id in ("body.health", "reward.scalar"):
        assert registry.encoder_for(stream_id) is not None


def test_learned_fusion_layout_hash_differs_from_fixed():
    catalog = _catalog()
    fixed = TemporalFusion(catalog, default_encoder_registry())
    live = LiveLearnedFusion(catalog, MINECRAFT_STREAM_REGISTRY, base_layout_hash=fixed.layout_hash)

    assert live.layout_hash != fixed.layout_hash
    assert live.fused_width() > 0


def test_live_fusion_fuse_produces_expected_width_and_trains_without_crashing():
    catalog = _catalog()
    fixed = TemporalFusion(catalog, default_encoder_registry())
    live = LiveLearnedFusion(
        catalog, MINECRAFT_STREAM_REGISTRY, base_layout_hash=fixed.layout_hash, fused_width=16
    )

    from cognitive_runtime.core.streams import MotorStreamBus, SensoryStreamBus, TickSynchronizer
    from cognitive_runtime.core.memory import Memory

    program = MinecraftSurvivalBox(config={"episode_ticks": 5, "world_size": 16})
    sensory_bus = SensoryStreamBus()
    motor_bus = MotorStreamBus()
    program.attach_buses(sensory_bus, motor_bus)
    program.reset(seed=0)

    memory = Memory()
    nominal_rates = {
        spec.stream_id: spec.nominal_rate_hz for spec in catalog if spec.nominal_rate_hz
    }
    synchronizer = TickSynchronizer(nominal_rates=nominal_rates)
    for _ in range(3):
        program.step()
    window = synchronizer.collect(sensory_bus)
    memory.update(window)

    state = live.fuse(window, memory.buffer)
    assert len(state.vector) == 16
    assert state.layout_hash == live.layout_hash

    metrics = live.maybe_train_step(window, memory.buffer, reward=1.0)
    assert "live_fusion_reward_loss" in metrics

    live.eval_mode()
    assert live.maybe_train_step(window, memory.buffer, reward=1.0) == {}
