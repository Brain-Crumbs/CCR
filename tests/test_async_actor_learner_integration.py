"""End-to-end async actor/learner split (issue #37) against the real
``CognitiveRuntime`` loop and a real trainer subprocess -- the acceptance
criteria directly:

- realtime-shaped run with training enabled shows no missed-tick regression
- weights demonstrably change during live play and the actor picks up
  published snapshots
- ``kill -9`` the trainer mid-run: actor continues; a restarted trainer
  resumes from checkpoint
"""

from __future__ import annotations

import os
import signal
import time

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry  # noqa: E402
from cognitive_runtime.neural.checkpoint import NeuralAgentCheckpoint, read_checkpoint_metadata  # noqa: E402
from cognitive_runtime.neural.experience_queue import SharedExperienceRing  # noqa: E402
from cognitive_runtime.neural.replay_buffer import Transition  # noqa: E402
from cognitive_runtime.neural.weight_publisher import WeightSubscriber  # noqa: E402
from cognitive_runtime.policies.actor_critic import (  # noqa: E402
    ActorCriticPolicy,
    AsyncActorCriticLearner,
    world_feature_width,
)
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox  # noqa: E402
from cognitive_runtime.runtime.config import RuntimeConfig  # noqa: E402
from cognitive_runtime.runtime.loop import CognitiveRuntime  # noqa: E402
from cognitive_runtime.training.async_trainer import (  # noqa: E402
    ActorCriticArch,
    build_actor_critic_modules,
    spawn_trainer_process,
)
from cognitive_runtime.training.features import ACTION_KEYS  # noqa: E402

WORLD_CONFIG = {"episode_ticks": 200, "world_size": 16, "max_mobs": 2}


def _arch(program, hidden_dim=32):
    fusion = TemporalFusion(program.stream_catalog(), default_encoder_registry())
    action_keys = tuple(ACTION_KEYS)
    return ActorCriticArch(
        fused_width=fusion.width,
        world_feature_width=world_feature_width(action_keys),
        n_actions=len(action_keys),
        action_keys=action_keys,
        layout_hash=fusion.layout_hash,
        hidden_dim=hidden_dim,
    ), fusion


def _wait_for_checkpoint_version_above(path, floor, *, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path + ".json"):
            version = read_checkpoint_metadata(path).get("training_ticks")
            if version is not None and version > floor:
                return version
        time.sleep(0.1)
    return None


def test_live_loop_with_async_trainer_mutates_weights_and_never_misses_a_tick(tmp_path):
    program = MinecraftSurvivalBox(config=WORLD_CONFIG)
    arch, _fusion = _arch(program)
    ckpt_path = str(tmp_path / "trainer.pt")

    ring = SharedExperienceRing(capacity=5000, latent_dim=arch.fused_width)
    trainer_process, stop_event = spawn_trainer_process(
        arch, ckpt_path,
        live_ring_handle=ring.handle(),
        trainer_kwargs={
            "batch_size": 16, "min_buffer_size": 16, "publish_every_steps": 3, "seed": 0,
        },
    )
    try:
        policy_model, critic_model, _wm, _opt = build_actor_critic_modules(arch, seed=1)
        actor_bundle = NeuralAgentCheckpoint(
            ckpt_path, layout_hash=arch.layout_hash, action_keys=arch.action_keys,
            policy=policy_model, critic=critic_model,
        )
        subscriber = WeightSubscriber(path=ckpt_path, bundle=actor_bundle)
        policy = ActorCriticPolicy(
            policy_model, critic_model, list(arch.action_keys), history=8, training=True, seed=1,
        )
        learner = AsyncActorCriticLearner(
            policy, ring, weight_subscriber=subscriber, reload_every_ticks=5,
        )
        initial_params = [p.clone() for p in policy_model.parameters()]

        runtime = CognitiveRuntime(
            program=program,
            policy=policy,
            config=RuntimeConfig(
                episodes=1, seed=0, max_ticks_per_episode=WORLD_CONFIG["episode_ticks"],
                record=False, program_config=WORLD_CONFIG, realtime=False,
            ),
            learner=learner,
        )
        summaries = runtime.run()

        # Acceptance criterion: no missed-tick regression with training enabled.
        assert summaries[0].duration_ticks == WORLD_CONFIG["episode_ticks"]
        assert summaries[0].missed_ticks == 0

        # Give the (already-running-in-parallel) trainer a little more time to
        # publish and the actor a few more polls to pick it up.
        deadline = time.time() + 10
        while time.time() < deadline:
            if subscriber.maybe_reload() is not None:
                break
            time.sleep(0.1)

        changed = any(
            not torch.equal(a, b) for a, b in zip(initial_params, policy_model.parameters())
        )
        assert changed, "online updates must mutate neural weights during play"
        assert subscriber.stats()["reload_count"] > 0, "actor must pick up published snapshots"
    finally:
        stop_event.set()
        trainer_process.join(timeout=10)
        if trainer_process.is_alive():
            trainer_process.terminate()
        ring.close()
        ring.unlink()


def test_trainer_kill_9_does_not_affect_the_actor_and_restart_resumes(tmp_path):
    program = MinecraftSurvivalBox(config=WORLD_CONFIG)
    arch, _fusion = _arch(program)
    ckpt_path = str(tmp_path / "trainer.pt")

    ring = SharedExperienceRing(capacity=2000, latent_dim=arch.fused_width)
    process, stop_event = spawn_trainer_process(
        arch, ckpt_path,
        live_ring_handle=ring.handle(),
        trainer_kwargs={
            "batch_size": 8, "min_buffer_size": 8, "publish_every_steps": 2, "seed": 0,
        },
    )
    try:
        for i in range(200):
            ring.push(Transition(
                latent=[float(i % 7)] * arch.fused_width, action=i % arch.n_actions,
                reward=float(i % 3), next_latent=[float((i + 1) % 7)] * arch.fused_width,
                done=False,
            ))
        version_before_kill = _wait_for_checkpoint_version_above(ckpt_path, -1)
        assert version_before_kill is not None, "trainer never published before being killed"

        # kill -9: the actor side must be completely unaffected.
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=10)
        assert not process.is_alive()
        assert process.exitcode != 0

        for i in range(50):  # never blocks, never raises
            ring.push(Transition(
                latent=[0.0] * arch.fused_width, action=0, reward=0.0,
                next_latent=[0.0] * arch.fused_width, done=False,
            ))

        # Restart resumes from the last published checkpoint and keeps
        # advancing the version.
        process2, stop_event2 = spawn_trainer_process(
            arch, ckpt_path,
            live_ring_handle=ring.handle(),
            trainer_kwargs={
                "batch_size": 8, "min_buffer_size": 8, "publish_every_steps": 2, "seed": 0,
            },
        )
        try:
            version_after_restart = _wait_for_checkpoint_version_above(
                ckpt_path, version_before_kill,
            )
            assert version_after_restart is not None, (
                "restarted trainer did not resume/advance past the pre-kill version"
            )
        finally:
            stop_event2.set()
            process2.join(timeout=10)
            if process2.is_alive():
                process2.terminate()
    finally:
        stop_event.set()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        ring.close()
        ring.unlink()


def test_concurrent_schedule_publishes_ema_weights_and_bounds_actor_staleness(tmp_path):
    """Issue #100: concurrent mode publishes an EMA-averaged snapshot with an
    increasing version while the actor keeps acting (never pausing), and the
    actor's staleness (versions behind the latest publish) is a bounded,
    reported quantity -- it resets to caught-up rather than growing without
    limit across the run."""
    program = MinecraftSurvivalBox(config=WORLD_CONFIG)
    arch, _fusion = _arch(program)
    ckpt_path = str(tmp_path / "trainer.pt")

    ring = SharedExperienceRing(capacity=5000, latent_dim=arch.fused_width)
    trainer_process, stop_event = spawn_trainer_process(
        arch, ckpt_path,
        live_ring_handle=ring.handle(),
        trainer_kwargs={
            "batch_size": 16, "min_buffer_size": 16, "publish_every_steps": 3,
            "seed": 0, "ema_decay": 0.9,
        },
    )
    try:
        policy_model, critic_model, _wm, _opt = build_actor_critic_modules(arch, seed=1)
        actor_bundle = NeuralAgentCheckpoint(
            ckpt_path, layout_hash=arch.layout_hash, action_keys=arch.action_keys,
            policy=policy_model, critic=critic_model,
        )
        subscriber = WeightSubscriber(path=ckpt_path, bundle=actor_bundle)
        policy = ActorCriticPolicy(
            policy_model, critic_model, list(arch.action_keys), history=8, training=True, seed=1,
        )
        learner = AsyncActorCriticLearner(
            policy, ring, weight_subscriber=subscriber, reload_every_ticks=5,
        )
        initial_params = [p.clone() for p in policy_model.parameters()]

        runtime = CognitiveRuntime(
            program=program,
            policy=policy,
            config=RuntimeConfig(
                episodes=1, seed=0, max_ticks_per_episode=WORLD_CONFIG["episode_ticks"],
                record=False, program_config=WORLD_CONFIG, realtime=False,
            ),
            learner=learner,
        )
        summaries = runtime.run()

        # Same no-missed-tick acceptance criterion as the raw-weights path:
        # EMA smoothing and per-tick staleness tracking must not cost
        # realtime-shaped ticks.
        assert summaries[0].duration_ticks == WORLD_CONFIG["episode_ticks"]
        assert summaries[0].missed_ticks == 0

        assert _wait_for_checkpoint_version_above(ckpt_path, -1) is not None, (
            "trainer never published"
        )

        deadline = time.time() + 10
        while time.time() < deadline:
            if subscriber.maybe_reload() is not None:
                break
            time.sleep(0.1)

        changed = any(
            not torch.equal(a, b) for a, b in zip(initial_params, policy_model.parameters())
        )
        assert changed, "the actor must pick up the trainer's EMA-averaged weights"
        stats = subscriber.stats()
        assert stats["reload_count"] > 0, "actor must pick up published snapshots"
        # "bounds how stale" (issue #100): the gap was tracked (not None/
        # unmeasured) during the run, and is caught up again -- 0 -- right
        # after this final reload, rather than drifting upward forever.
        assert stats["max_staleness"] is not None
        assert subscriber.staleness() == 0
    finally:
        stop_event.set()
        trainer_process.join(timeout=10)
        if trainer_process.is_alive():
            trainer_process.terminate()
        ring.close()
        ring.unlink()
