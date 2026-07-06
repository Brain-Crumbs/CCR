"""Phase-3 tools + CLI: stream-native viewer/dashboard and end-to-end flow."""

import json
import os

import pytest

from cognitive_runtime.cli import main
from cognitive_runtime.policies import RandomPolicy, ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import LegacyFormatError
from cognitive_runtime.tools.episode_viewer import view_episode
from cognitive_runtime.tools.metrics_dashboard import dashboard
from cognitive_runtime.tools.replay_runner import replay_session

FAST_CONFIG = {"episode_ticks": 200, "world_size": 32, "day_length": 150, "start_time": 100}


def _record(tmp_path, policy, session_id):
    config = RuntimeConfig(
        episodes=1, seed=7, max_ticks_per_episode=200,
        record_dir=str(tmp_path), session_id=session_id, program_config=FAST_CONFIG,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=FAST_CONFIG), policy=policy, config=config
    ).run()
    return os.path.join(str(tmp_path), session_id)


def test_viewer_renders_stream_native_sections(tmp_path):
    session_dir = _record(tmp_path, ScriptedSurvivalPolicy(seed=1), "view")
    out = view_episode(session_dir, "episode_00000", tail=5)
    assert "streams (count, events/sec):" in out
    assert "reward.scalar:" in out
    assert "reward components (episode totals):" in out
    assert "action distribution:" in out
    assert "last 5 decisions:" in out
    assert "world.time" in out


def test_dashboard_renders_stream_rates(tmp_path):
    _record(tmp_path, ScriptedSurvivalPolicy(seed=1), "dash")
    out = dashboard(str(tmp_path))
    assert "policy" in out and "stream_events_per_sec" in out
    assert "per-stream average events/sec:" in out
    assert "reward.scalar" in out


def test_cli_end_to_end_flow(tmp_path):
    record_dir = str(tmp_path / "sessions")
    model_path = str(tmp_path / "bc.json")

    main(["run", "--policy", "scripted", "--episodes", "2", "--seed", "5",
          "--episode-ticks", "200", "--world-size", "32",
          "--record-dir", record_dir, "--session-id", "cli"])
    session_dir = os.path.join(record_dir, "cli")
    assert os.path.exists(os.path.join(session_dir, "session.json"))

    main(["replay", "--session", session_dir])
    main(["view", "--session", session_dir, "--episode", "episode_00000"])
    main(["dashboard", "--record-dir", record_dir])
    main(["train", "--sessions", session_dir, "--out", model_path, "--epochs", "3"])
    assert os.path.exists(model_path)


def test_legacy_session_is_rejected(tmp_path):
    session_dir = _record(tmp_path, RandomPolicy(MinecraftSurvivalBox(config=FAST_CONFIG)
                                                .metadata().action_space, seed=1), "legacy")
    # Strip the format marker to look like a pre-streams-v2 session.
    meta_path = os.path.join(session_dir, "session.json")
    meta = json.load(open(meta_path, encoding="utf-8"))
    meta.pop("format", None)
    json.dump(meta, open(meta_path, "w", encoding="utf-8"))
    with pytest.raises(LegacyFormatError):
        replay_session(session_dir)
