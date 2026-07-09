import json
import sys
from pathlib import Path

import pytest

from cognitive_runtime.cli import main
from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry
from cognitive_runtime.models.online_q import OnlineQModel
from cognitive_runtime.policies.online_q import OnlineQLearner, OnlineQPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import NonDeterministicSessionError
from cognitive_runtime.tools.metrics_dashboard import dashboard
from cognitive_runtime.tools.replay_runner import replay_session


FAKE_BRIDGE = Path(__file__).resolve().parents[1] / "bridge" / "fake" / "sim_bridge.py"


def _use_fake_remote(monkeypatch):
    monkeypatch.setenv("CCR_MINECRAFT_BRIDGE_CMD", f"{sys.executable} {FAKE_BRIDGE}")


def test_remote_online_eval_smoke_does_not_train_and_records_nondeterministic_session(
    tmp_path, monkeypatch
):
    _use_fake_remote(monkeypatch)
    checkpoint = tmp_path / "online-q.json"

    main(
        [
            "run",
            "--backend",
            "remote",
            "--policy",
            "online",
            "--no-online-train",
            "--episodes",
            "1",
            "--episode-ticks",
            "20",
            "--online-model",
            str(checkpoint),
            "--record-dir",
            str(tmp_path),
            "--session-id",
            "remote-online-eval",
        ]
    )

    model = OnlineQModel.load(str(checkpoint))
    metadata = json.loads((tmp_path / "remote-online-eval" / "session.json").read_text())
    assert model.training_ticks == 0
    assert metadata["deterministic"] is False
    assert metadata["policy"] == "online"
    assert "online_model" in metadata
    with pytest.raises(NonDeterministicSessionError):
        replay_session(str(tmp_path / "remote-online-eval"))


def test_remote_online_training_checkpoint_reloads_across_sessions(tmp_path, monkeypatch):
    _use_fake_remote(monkeypatch)
    checkpoint = tmp_path / "online-q.json"

    base_args = [
        "run",
        "--backend",
        "remote",
        "--policy",
        "online",
        "--episodes",
        "1",
        "--episode-ticks",
        "25",
        "--online-model",
        str(checkpoint),
        "--online-save-every",
        "5",
        "--record-dir",
        str(tmp_path),
    ]
    main([*base_args, "--session-id", "remote-online-train-a"])
    first_ticks = OnlineQModel.load(str(checkpoint)).training_ticks
    main([*base_args, "--session-id", "remote-online-train-b"])
    second_ticks = OnlineQModel.load(str(checkpoint)).training_ticks

    assert first_ticks > 0
    assert second_ticks > first_ticks


def test_remote_dashboard_can_compare_online_random_and_scripted(tmp_path, monkeypatch):
    _use_fake_remote(monkeypatch)
    checkpoint = tmp_path / "online-q.json"
    common = [
        "--backend",
        "remote",
        "--episodes",
        "1",
        "--episode-ticks",
        "15",
        "--record-dir",
        str(tmp_path),
    ]

    main(
        [
            "run",
            "--policy",
            "online",
            "--no-online-train",
            "--online-model",
            str(checkpoint),
            "--session-id",
            "remote-online",
            *common,
        ]
    )
    main(["run", "--policy", "random", "--session-id", "remote-random", *common])
    main(["run", "--policy", "scripted", "--session-id", "remote-scripted", *common])

    out = dashboard(str(tmp_path))
    assert "online" in out
    assert "random" in out
    assert "scripted" in out
    assert "stream_events_per_sec" in out


class InterruptingOnlineQLearner(OnlineQLearner):
    def update(self, window):
        super().update(window)
        if self.observed_ticks >= 3:
            raise KeyboardInterrupt


def test_remote_online_checkpoint_survives_keyboard_interrupt(tmp_path, monkeypatch):
    _use_fake_remote(monkeypatch)
    config = {"episode_ticks": 40, "world_size": 32}
    program = MinecraftSurvivalBox(config=config, backend="remote")
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
    checkpoint = tmp_path / "remote-interrupted-online-q.json"
    learner = InterruptingOnlineQLearner(
        model, policy, training=True, checkpoint_path=str(checkpoint)
    )
    runtime = CognitiveRuntime(
        program=program,
        policy=policy,
        learner=learner,
        config=RuntimeConfig(
            episodes=1,
            seed=0,
            max_ticks_per_episode=config["episode_ticks"],
            record=False,
            program_config=config,
        ),
    )

    try:
        with pytest.raises(KeyboardInterrupt):
            runtime.run()
    finally:
        program.close()

    loaded = OnlineQModel.load(str(checkpoint))
    assert loaded.training_ticks > 0
    assert loaded.meta["learner_stats"]["last_checkpoint_reason"] == "keyboard_interrupt"
