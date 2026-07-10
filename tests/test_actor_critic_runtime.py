"""CLI + runtime integration for ``--policy actor-critic`` (issue #29):
checkpoint bundle creation/resume, session metadata, interrupted-run
checkpoint validity, and torch staying optional. Mirrors
tests/test_online_q_runtime.py's shape.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.cli import main  # noqa: E402
from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry  # noqa: E402
from cognitive_runtime.neural import ActorCriticOptimizer, MLPPolicyModel, MLPValueModel  # noqa: E402
from cognitive_runtime.neural.checkpoint import NeuralAgentCheckpoint  # noqa: E402
from cognitive_runtime.policies.actor_critic import (  # noqa: E402
    ActorCriticLearner,
    ActorCriticPolicy,
    world_feature_width,
)
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox  # noqa: E402
from cognitive_runtime.runtime.config import RuntimeConfig  # noqa: E402
from cognitive_runtime.runtime.loop import CognitiveRuntime  # noqa: E402

FAST_CONFIG = {"episode_ticks": 30, "world_size": 32}


def _run_cli_actor_critic(path, ticks=30, *extra):
    main(
        [
            "run",
            "--policy",
            "actor-critic",
            "--episodes",
            "1",
            "--episode-ticks",
            str(ticks),
            "--world-size",
            "32",
            "--actor-critic-model",
            str(path),
            "--actor-critic-save-every",
            "5",
            "--no-record",
            *extra,
        ]
    )


def test_cli_actor_critic_policy_runs_simulated_backend_and_creates_checkpoint(tmp_path):
    path = tmp_path / "actor-critic.pt"

    _run_cli_actor_critic(path, ticks=20)

    assert path.exists()
    assert (tmp_path / "actor-critic.pt.json").exists()
    with open(str(path) + ".json", encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["online_optimizer"]["step"] > 0


def test_reloaded_actor_critic_checkpoint_continues_step_count(tmp_path):
    path = tmp_path / "actor-critic.pt"

    _run_cli_actor_critic(path, ticks=15)
    with open(str(path) + ".json", encoding="utf-8") as fh:
        first_step = json.load(fh)["online_optimizer"]["step"]

    _run_cli_actor_critic(path, ticks=15)
    with open(str(path) + ".json", encoding="utf-8") as fh:
        second_step = json.load(fh)["online_optimizer"]["step"]

    assert first_step > 0
    assert second_step > first_step


def test_actor_critic_session_metadata_records_policy_and_optimizer_config(tmp_path):
    checkpoint = tmp_path / "actor-critic.pt"
    record_dir = tmp_path / "sessions"
    main(
        [
            "run",
            "--policy",
            "actor-critic",
            "--episodes",
            "1",
            "--episode-ticks",
            "10",
            "--world-size",
            "32",
            "--actor-critic-model",
            str(checkpoint),
            "--record-dir",
            str(record_dir),
            "--session-id",
            "actor-critic-session",
        ]
    )

    with open(record_dir / "actor-critic-session" / "session.json", encoding="utf-8") as fh:
        session = json.load(fh)
    assert session["policy"] == "actor-critic"
    assert session["online_model"]["policy"]["format"] == "actor-critic-v1"
    assert session["online_model"]["learner"]["checkpoint_path"] == str(checkpoint)
    assert "gamma" in session["online_model"]["learner"]["optimizer"]


class InterruptingActorCriticLearner(ActorCriticLearner):
    def update(self, window):
        super().update(window)
        if self.observed_ticks >= 5:
            raise KeyboardInterrupt


def test_interrupted_run_leaves_valid_actor_critic_checkpoint(tmp_path):
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    fusion = TemporalFusion(program.stream_catalog(), default_encoder_registry())
    action_space = list(program.metadata().action_space)
    action_keys = [action.key() for action in action_space]
    wf_width = world_feature_width(action_keys)

    torch.manual_seed(0)
    policy_model = MLPPolicyModel(
        fusion.width, wf_width, len(action_keys),
        hidden_dim=16, layout_hash=fusion.layout_hash, action_keys=action_keys,
    )
    critic_model = MLPValueModel(
        fusion.width, wf_width, hidden_dim=16, layout_hash=fusion.layout_hash, action_keys=action_keys,
    )
    optimizer = ActorCriticOptimizer(policy_model, critic_model, lr=0.05, seed=0)
    policy = ActorCriticPolicy(policy_model, critic_model, action_keys, action_space=action_space, training=True)
    path = tmp_path / "interrupted-actor-critic.pt"
    checkpoint = NeuralAgentCheckpoint(
        str(path), layout_hash=fusion.layout_hash, action_keys=action_keys, online_optimizer=optimizer,
    )
    learner = InterruptingActorCriticLearner(optimizer, policy, training=True, checkpoint=checkpoint)
    runtime = CognitiveRuntime(
        program=program,
        policy=policy,
        learner=learner,
        config=RuntimeConfig(
            episodes=1, seed=0, max_ticks_per_episode=FAST_CONFIG["episode_ticks"],
            record=False, program_config=FAST_CONFIG,
        ),
    )

    with pytest.raises(KeyboardInterrupt):
        runtime.run()

    assert path.exists()
    with open(str(path) + ".json", encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["training_stats"]["last_checkpoint_reason"] == "keyboard_interrupt"
    assert metadata["training_stats"]["training_ticks"] > 0


def test_eval_mode_produces_zero_weight_mutation(tmp_path):
    path = tmp_path / "actor-critic.pt"
    _run_cli_actor_critic(path, ticks=20)
    before = torch.load(path, weights_only=False)["state"]["online_optimizer"]["modules"]
    before_step = torch.load(path, weights_only=False)["metadata"]["online_optimizer"]["step"]

    _run_cli_actor_critic(path, 20, "--no-actor-critic-train")

    after_payload = torch.load(path, weights_only=False)
    after = after_payload["state"]["online_optimizer"]["modules"]
    after_step = after_payload["metadata"]["online_optimizer"]["step"]

    assert after_step == before_step
    for module_name in before:
        for key in before[module_name]:
            assert torch.equal(before[module_name][key], after[module_name][key])


def test_actor_critic_policy_without_torch_exits_with_actionable_message(tmp_path):
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "from cognitive_runtime.cli import main\n"
        "try:\n"
        "    main(['run', '--policy', 'actor-critic', '--episodes', '1', "
        "'--episode-ticks', '5', '--world-size', '16', '--no-record'])\n"
        "except SystemExit as exc:\n"
        "    msg = str(exc)\n"
        "    assert 'PyTorch' in msg and \".[neural]\" in msg, msg\n"
        "else:\n"
        "    raise AssertionError('expected SystemExit')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path)
