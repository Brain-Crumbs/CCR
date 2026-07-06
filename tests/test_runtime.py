"""Runtime loop, scheduler, recorder and replay tests."""

import json
import os

from cognitive_runtime.policies import NullPolicy, RandomPolicy, ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import (
    iter_cognitive_ticks,
    list_episodes,
    load_decisions,
    load_stream_log,
    load_summary,
)
from cognitive_runtime.runtime.scheduler import FixedTickScheduler
from cognitive_runtime.tools.replay_runner import replay_session

FAST_CONFIG = {"episode_ticks": 200, "world_size": 32}


def _make_runtime(tmp_path, policy, episodes=1, seed=0, session_id="test-session"):
    config = RuntimeConfig(
        episodes=episodes,
        seed=seed,
        max_ticks_per_episode=200,
        record_dir=str(tmp_path),
        session_id=session_id,
        program_config=FAST_CONFIG,
    )
    return CognitiveRuntime(
        program=MinecraftSurvivalBox(config=FAST_CONFIG), policy=policy, config=config
    )


def test_milestone0_null_policy_runs_at_fixed_tick_rate(tmp_path):
    runtime = _make_runtime(tmp_path, NullPolicy(), episodes=2)
    summaries = runtime.run()
    assert len(summaries) == 2
    for summary in summaries:
        assert summary.duration_ticks == 200
        assert summary.null_action_ticks == 200
        assert summary.termination_reason == "episode_ticks"
    assert runtime.scheduler.stats.ticks == 200  # per-episode, reset each episode


def test_realtime_scheduler_holds_tick_rate():
    scheduler = FixedTickScheduler(tick_rate=200.0, realtime=True)
    for _ in range(20):
        scheduler.wait_for_next_tick()
    assert scheduler.stats.ticks == 20
    assert scheduler.stats.elapsed_seconds >= 20 * (1 / 200.0) * 0.5


def test_recorder_writes_streams_decisions_and_summaries(tmp_path):
    runtime = _make_runtime(tmp_path, RandomPolicy(ACTION_SPACE, seed=1))
    runtime.run()
    session_dir = os.path.join(str(tmp_path), "test-session")
    assert os.path.exists(os.path.join(session_dir, "session.json"))

    # One decision line per cognitive tick.
    decisions = load_decisions(session_dir, "episode_00000")
    assert len(decisions) == 200
    for key in ("tick_index", "window_span", "n_events_by_stream", "motor_emitted",
                "policy_name", "latency_ms", "reward_window_total"):
        assert key in decisions[0], key

    # Every stream line carries the schema fields, both directions present.
    stream_log = load_stream_log(session_dir, "episode_00000")
    assert stream_log
    dirs = {r["dir"] for r in stream_log}
    assert dirs <= {"sensory", "motor"}
    for record in stream_log[:5]:
        for key in ("dir", "stream_id", "modality", "timestamp", "seq", "hash"):
            assert key in record, key

    summary = load_summary(session_dir, "episode_00000")
    assert summary["seed"] == 0
    assert summary["duration_ticks"] == 200
    assert summary["stream_event_counts"]  # per-stream counts recorded

    with open(os.path.join(session_dir, "session.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["program"] == "MinecraftSurvivalBox"
    assert meta["format"] == "streams-v2"
    assert meta["stream_catalog"]  # catalog embedded so tools need no program


def test_recorded_lines_round_trip_to_stream_events(tmp_path):
    from cognitive_runtime.runtime.recorder import stream_event_from_log

    runtime = _make_runtime(tmp_path, RandomPolicy(ACTION_SPACE, seed=1), session_id="rt")
    runtime.run()
    session_dir = os.path.join(str(tmp_path), "rt")
    stream_log = load_stream_log(session_dir, "episode_00000")
    checked = 0
    for record in stream_log:
        if record.get("elided"):
            continue
        assert stream_event_from_log(record).hash() == record["hash"]
        checked += 1
    assert checked > 0


def test_milestone4_replay_verifies_determinism(tmp_path):
    for seed, policy in ((5, RandomPolicy(ACTION_SPACE, seed=9)), (11, ScriptedSurvivalPolicy(seed=2))):
        session_id = f"replay-{seed}"
        runtime = _make_runtime(tmp_path, policy, seed=seed, session_id=session_id)
        runtime.run()
        results = replay_session(os.path.join(str(tmp_path), session_id))
        assert len(results) == 1
        assert results[0].matched, results[0]
        assert results[0].ticks_replayed == 200


def test_replay_detects_sensory_tampering(tmp_path):
    runtime = _make_runtime(tmp_path, RandomPolicy(ACTION_SPACE, seed=3), session_id="tamper")
    runtime.run()
    session_dir = os.path.join(str(tmp_path), "tamper")
    path = os.path.join(session_dir, "episode_00000.streams.jsonl")
    lines = open(path, encoding="utf-8").read().splitlines()
    # Mutate one sensory payload while leaving its recorded hash in place.
    for i, line in enumerate(lines):
        record = json.loads(line)
        if record["dir"] == "sensory" and record.get("stream_id") == "body.hunger":
            record["payload"] = float(record["payload"]) + 5.0
            lines[i] = json.dumps(record)
            break
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    results = replay_session(session_dir)
    assert not results[0].matched
    assert results[0].first_divergence_stream == "body.hunger"


def test_episodes_use_distinct_seeds(tmp_path):
    runtime = _make_runtime(tmp_path, NullPolicy(), episodes=2, session_id="seeds")
    summaries = runtime.run()
    assert [s.seed for s in summaries] == [0, 1]
    session_dir = os.path.join(str(tmp_path), "seeds")
    assert list_episodes(session_dir) == ["episode_00000", "episode_00001"]
