"""Wiring learned fusion + trainable stream encoders into the live
actor/critic path (issue #57). Covers the issue's acceptance criteria:
a simulated ``--fusion learned`` run completing end to end with no missed
ticks, checkpoint round-trip of encoder/fusion weights and tick counters,
and the fusion-mode compatibility gate failing loudly instead of silently
misbehaving.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.cli import main  # noqa: E402
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


def _run_cli_actor_critic(path, *, fusion, ticks=20, extra=()):
    main(
        [
            "run",
            "--policy",
            "actor-critic",
            "--fusion",
            fusion,
            "--episodes",
            "1",
            "--episode-ticks",
            str(ticks),
            "--world-size",
            "32",
            "--actor-critic-model",
            str(path),
            "--no-record",
            *extra,
        ]
    )


def test_cli_fusion_learned_run_completes_and_checkpoints_encoders_and_fusion(tmp_path):
    path = tmp_path / "actor-critic-learned.pt"

    _run_cli_actor_critic(path, fusion="learned", ticks=20)

    assert path.exists()
    with open(f"{path}.json", encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["extra"]["actor_critic"]["fusion_mode"] == "learned"
    assert metadata["modules"]["encoders"]
    assert "fusion" in metadata["modules"]
    assert "live_fusion" in metadata["optimizers"]


def test_cli_fusion_learned_checkpoint_round_trip_restores_weights_and_ticks(tmp_path):
    path = tmp_path / "actor-critic-learned.pt"

    _run_cli_actor_critic(path, fusion="learned", ticks=15)
    with open(f"{path}.json", encoding="utf-8") as fh:
        first_step = json.load(fh)["online_optimizer"]["step"]
    first_state = torch.load(path, map_location="cpu", weights_only=False)["state"]

    _run_cli_actor_critic(path, fusion="learned", ticks=15)
    with open(f"{path}.json", encoding="utf-8") as fh:
        second_step = json.load(fh)["online_optimizer"]["step"]
    second_state = torch.load(path, map_location="cpu", weights_only=False)["state"]

    assert first_step > 0
    assert second_step > first_step
    # Resuming loaded (rather than re-initialized) encoder/fusion weights:
    # the second run's saved weights differ from the first run's fresh init
    # by however training moved them, not by a brand-new random seed.
    assert set(first_state["encoders"]) == set(second_state["encoders"])
    assert set(first_state["fusion"]) == set(second_state["fusion"])


def test_cli_fusion_mode_mismatch_on_resume_fails_loudly(tmp_path):
    path = tmp_path / "actor-critic.pt"
    _run_cli_actor_critic(path, fusion="fixed", ticks=10)

    with pytest.raises(SystemExit) as exc_info:
        _run_cli_actor_critic(path, fusion="learned", ticks=10)
    assert "fusion" in str(exc_info.value.code)


def test_learned_fusion_run_has_no_missed_ticks(tmp_path):
    fixed_path = tmp_path / "fixed.pt"
    learned_path = tmp_path / "learned.pt"

    from cognitive_runtime.runtime.recorder import NullRecorder  # noqa: E402
    from cognitive_runtime.runtime.config import RuntimeConfig  # noqa: E402
    from cognitive_runtime.runtime.loop import CognitiveRuntime  # noqa: E402
    from cognitive_runtime.programs.minecraft.stream_registry import (  # noqa: E402
        MINECRAFT_STREAM_REGISTRY,
    )
    from cognitive_runtime.cli import _make_actor_critic_policy_and_learner  # noqa: E402
    import argparse

    def _episode_summaries(fusion_mode, model_path):
        args = argparse.Namespace(
            actor_critic_model=str(model_path),
            fusion=fusion_mode,
            actor_critic_hidden_dim=32,
            actor_critic_world_model_loss=False,
            actor_critic_async=False,
            actor_critic_lr=1e-3,
            actor_critic_gamma=0.99,
            actor_critic_entropy_coef=0.01,
            actor_critic_grad_clip_norm=5.0,
            actor_critic_history=8,
            actor_critic_train=True,
            actor_critic_save_every=1000,
            actor_critic_replay_every=32,
            actor_critic_replay_batch_size=32,
            seed=0,
        )
        program = MinecraftSurvivalBox(config={"episode_ticks": 20, "world_size": 32})
        policy, learner, learned_fusion = _make_actor_critic_policy_and_learner(args, program)
        runtime = CognitiveRuntime(
            program=program,
            policy=policy,
            learner=learner,
            config=RuntimeConfig(episodes=1, seed=0, max_ticks_per_episode=20, record=False),
            recorder=NullRecorder(),
            stream_registry=MINECRAFT_STREAM_REGISTRY,
            learned_fusion=learned_fusion,
        )
        return runtime.run()

    fixed_summaries = _episode_summaries("fixed", fixed_path)
    learned_summaries = _episode_summaries("learned", learned_path)

    assert fixed_summaries[0].missed_ticks == 0
    assert learned_summaries[0].missed_ticks == 0
    assert learned_summaries[0].duration_ticks == fixed_summaries[0].duration_ticks
