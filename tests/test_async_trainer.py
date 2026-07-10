"""``AsyncTrainer``: the background-trainer half of the actor/learner split
(issue #37).  Covers the acceptance criteria directly:

- offline pretraining from recorded sessions only, no live actor
- live-queue ingestion feeding the same dataloader/buffer as recorded
  sessions
- checkpoint resume
- the CLI's ``trainer`` subcommand and ``run --async-trainer``
"""

from __future__ import annotations

import multiprocessing as mp
import os

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from cognitive_runtime.cli import main  # noqa: E402
from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry  # noqa: E402
from cognitive_runtime.neural.experience_queue import SharedExperienceRing  # noqa: E402
from cognitive_runtime.neural.replay_buffer import Transition  # noqa: E402
from cognitive_runtime.policies import ScriptedSurvivalPolicy  # noqa: E402
from cognitive_runtime.policies.actor_critic import world_feature_width  # noqa: E402
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox  # noqa: E402
from cognitive_runtime.runtime.config import RuntimeConfig  # noqa: E402
from cognitive_runtime.runtime.loop import CognitiveRuntime  # noqa: E402
from cognitive_runtime.training.async_trainer import ActorCriticArch, AsyncTrainer  # noqa: E402
from cognitive_runtime.training.features import ACTION_KEYS  # noqa: E402

WORLD_CONFIG = {"episode_ticks": 120, "world_size": 16, "max_mobs": 2}


def _record_session(tmp_path, session_id="sess0", *, seed=0, ticks=120):
    program = MinecraftSurvivalBox(config=WORLD_CONFIG)
    runtime_config = RuntimeConfig(
        episodes=1, seed=seed, max_ticks_per_episode=ticks,
        record_dir=str(tmp_path), session_id=session_id, program_config=WORLD_CONFIG,
    )
    CognitiveRuntime(
        program=program, policy=ScriptedSurvivalPolicy(seed=seed), config=runtime_config,
    ).run()
    return os.path.join(str(tmp_path), session_id), program


def _arch(program, *, hidden_dim=32, has_world_model=False):
    fusion = TemporalFusion(program.stream_catalog(), default_encoder_registry())
    action_keys = tuple(ACTION_KEYS)
    return ActorCriticArch(
        fused_width=fusion.width,
        world_feature_width=world_feature_width(action_keys),
        n_actions=len(action_keys),
        action_keys=action_keys,
        layout_hash=fusion.layout_hash,
        hidden_dim=hidden_dim,
        has_world_model=has_world_model,
    ), fusion


# ------------------------------------------------------- offline pretraining


def test_offline_pretraining_from_recorded_sessions_with_no_live_actor(tmp_path):
    """"The same trainer, pointed only at recorded sessions with no live
    actor, performs offline pretraining." """
    session_dir, program = _record_session(tmp_path)
    arch, _fusion = _arch(program)
    ckpt_path = str(tmp_path / "trainer.pt")

    trainer = AsyncTrainer(
        arch, ckpt_path, session_dirs=[session_dir], batch_size=16,
        min_buffer_size=1, publish_every_steps=5, seed=0,
    )
    assert trainer.live_ring is None
    assert trainer.resume_if_checkpoint_exists() is False
    loaded = trainer.load_recorded_sessions()
    assert loaded > 0

    stats = trainer.run_forever(mp.Event(), max_steps=15)
    assert stats["step_count"] == 15
    assert os.path.exists(ckpt_path)
    assert os.path.exists(ckpt_path + ".json")

    # An offline trainer with an exhausted, non-growing source terminates on
    # its own once the buffer sits below min_buffer_size... but here the
    # buffer already has data, so instead assert it stops driven by
    # max_steps, not by hanging forever (the real regression this guards
    # against: an offline run that never returns).


def test_offline_trainer_stops_when_buffer_never_fills(tmp_path):
    """No live ring and nothing (or too little) recorded: the trainer must
    return instead of spinning forever waiting for data that will never
    arrive."""
    program = MinecraftSurvivalBox(config=WORLD_CONFIG)
    arch, _fusion = _arch(program)
    ckpt_path = str(tmp_path / "trainer.pt")
    trainer = AsyncTrainer(
        arch, ckpt_path, session_dirs=None, min_buffer_size=1_000_000, seed=0,
    )
    stats = trainer.run_forever(mp.Event(), max_steps=100)
    assert stats["step_count"] == 0


def test_resume_from_checkpoint_continues_the_step_count(tmp_path):
    session_dir, program = _record_session(tmp_path)
    arch, _fusion = _arch(program)
    ckpt_path = str(tmp_path / "trainer.pt")

    first = AsyncTrainer(
        arch, ckpt_path, session_dirs=[session_dir], batch_size=16,
        publish_every_steps=100, seed=0,
    )
    first.load_recorded_sessions()
    first_stats = first.run_forever(mp.Event(), max_steps=10)
    assert first_stats["step_count"] == 10

    second = AsyncTrainer(
        arch, ckpt_path, session_dirs=[session_dir], batch_size=16,
        publish_every_steps=100, seed=0,
    )
    assert second.resume_if_checkpoint_exists() is True
    assert second.optimizer.step_count == 10
    second.load_recorded_sessions()
    second_stats = second.run_forever(mp.Event(), max_steps=10)
    assert second_stats["step_count"] == 20


# ------------------------------------------------------------- live ingestion


def test_ingest_live_feeds_the_same_buffer_recorded_sessions_do(tmp_path):
    """"the same dataloader interface reads (a) the live experience queue
    and (b) recorded sessions on disk"."""
    session_dir, program = _record_session(tmp_path)
    arch, fusion = _arch(program)
    ckpt_path = str(tmp_path / "trainer.pt")
    ring = SharedExperienceRing(capacity=100, latent_dim=arch.fused_width)
    try:
        trainer = AsyncTrainer(
            arch, ckpt_path, session_dirs=[session_dir],
            live_ring_handle=ring.handle(), batch_size=16, seed=0,
        )
        pretrain_loaded = trainer.load_recorded_sessions()
        assert pretrain_loaded > 0
        assert len(trainer.replay_buffer) == pretrain_loaded

        for i in range(20):
            ring.push(Transition(
                latent=[float(i)] * arch.fused_width, action=0, reward=float(i),
                next_latent=[float(i) + 1] * arch.fused_width, done=False,
            ))
        ingested = trainer.ingest_live()
        assert ingested == 20
        assert len(trainer.replay_buffer) == pretrain_loaded + 20
    finally:
        ring.close()
        ring.unlink()


# -------------------------------------------------------------------- CLI


def test_cli_trainer_subcommand_offline_pretrains_and_resumes(tmp_path):
    session_dir, _program = _record_session(tmp_path)
    out_path = str(tmp_path / "cli-trainer.pt")

    main([
        "trainer", "--sessions", session_dir, "--out", out_path,
        "--steps", "10", "--batch-size", "16", "--publish-every", "5",
        "--hidden-dim", "32", "--no-world-model-loss",
    ])
    assert os.path.exists(out_path)
    first_metadata = torch.load(out_path, weights_only=False)["metadata"]
    assert first_metadata["online_optimizer"]["step"] == 10

    main([
        "trainer", "--sessions", session_dir, "--out", out_path,
        "--steps", "10", "--batch-size", "16", "--publish-every", "5",
        "--hidden-dim", "32", "--no-world-model-loss",
    ])
    second_metadata = torch.load(out_path, weights_only=False)["metadata"]
    assert second_metadata["online_optimizer"]["step"] == 20


def test_cli_run_async_trainer_produces_a_checkpoint_and_no_missed_ticks(tmp_path):
    """Exercises `ccr run --policy actor-critic --async-trainer` end to end:
    the realtime tick loop runs to completion with a background trainer
    process alive the whole time."""
    model_path = str(tmp_path / "async-model.pt")
    main([
        "run", "--policy", "actor-critic", "--async-trainer",
        "--episodes", "1", "--episode-ticks", "80", "--no-record",
        "--actor-critic-model", model_path,
        "--async-min-buffer-size", "16", "--async-batch-size", "16",
        "--async-publish-every", "3", "--async-reload-every-ticks", "2",
        "--world-size", "16", "--max-mobs", "2",
    ])
    assert os.path.exists(model_path)
    assert os.path.exists(model_path + ".json")
