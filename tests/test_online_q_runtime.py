import json

import pytest

from cognitive_runtime.cli import main
from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry
from cognitive_runtime.models.online_q import OnlineQModel
from cognitive_runtime.policies.online_q import OnlineQLearner, OnlineQPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime


FAST_CONFIG = {"episode_ticks": 30, "world_size": 32}


def _run_cli_online(path, ticks=30, *extra):
    main(
        [
            "run",
            "--policy",
            "online",
            "--episodes",
            "1",
            "--episode-ticks",
            str(ticks),
            "--world-size",
            "32",
            "--online-model",
            str(path),
            "--online-save-every",
            "5",
            "--no-record",
            *extra,
        ]
    )


def test_cli_online_policy_runs_simulated_backend_and_creates_checkpoint(tmp_path):
    path = tmp_path / "online-q.json"

    _run_cli_online(path, ticks=20)

    model = OnlineQModel.load(str(path))
    assert path.exists()
    assert model.training_ticks > 0
    assert model.meta["learner_stats"]["training_ticks"] == model.training_ticks


def test_reloaded_online_model_continues_training_tick_count(tmp_path):
    path = tmp_path / "online-q.json"

    _run_cli_online(path, ticks=15)
    first = OnlineQModel.load(str(path)).training_ticks
    _run_cli_online(path, ticks=15)
    second = OnlineQModel.load(str(path)).training_ticks

    assert first > 0
    assert second > first


def test_online_session_metadata_records_model_details(tmp_path):
    checkpoint = tmp_path / "online-q.json"
    record_dir = tmp_path / "sessions"
    main(
        [
            "run",
            "--policy",
            "online",
            "--episodes",
            "1",
            "--episode-ticks",
            "10",
            "--world-size",
            "32",
            "--online-model",
            str(checkpoint),
            "--record-dir",
            str(record_dir),
            "--session-id",
            "online-session",
        ]
    )

    with open(record_dir / "online-session" / "session.json", encoding="utf-8") as fh:
        session = json.load(fh)
    assert session["policy"] == "online"
    assert session["online_model"]["policy"]["format"] == "online-q-v1"
    assert session["online_model"]["learner"]["checkpoint_path"] == str(checkpoint)


class InterruptingOnlineQLearner(OnlineQLearner):
    def update(self, window):
        super().update(window)
        if self.observed_ticks >= 3:
            raise KeyboardInterrupt


def test_interrupted_run_leaves_valid_online_checkpoint(tmp_path):
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    fusion = TemporalFusion(program.stream_catalog(), default_encoder_registry())
    action_space = list(program.metadata().action_space)
    model = OnlineQModel.initialize(
        [action.key() for action in action_space],
        latent_width=fusion.width,
        layout_hash=fusion.layout_hash,
        latent_feature_names=fusion.feature_names(),
        seed=0,
    )
    policy = OnlineQPolicy(model, action_space=action_space, training=True)
    path = tmp_path / "interrupted-online-q.json"
    learner = InterruptingOnlineQLearner(model, policy, training=True, checkpoint_path=str(path))
    runtime = CognitiveRuntime(
        program=program,
        policy=policy,
        learner=learner,
        config=RuntimeConfig(
            episodes=1,
            seed=0,
            max_ticks_per_episode=FAST_CONFIG["episode_ticks"],
            record=False,
            program_config=FAST_CONFIG,
        ),
    )

    with pytest.raises(KeyboardInterrupt):
        runtime.run()

    loaded = OnlineQModel.load(str(path))
    assert loaded.training_ticks > 0
    assert loaded.meta["learner_stats"]["last_checkpoint_reason"] == "keyboard_interrupt"

