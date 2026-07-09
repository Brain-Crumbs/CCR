"""Runtime loop, scheduler, recorder and replay tests."""

import json
import os

from cognitive_runtime.policies import NullPolicy, RandomPolicy, ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import (
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


# ------------------------------------------------- binary frame store (#38)


def test_record_frames_writes_frame_ref_not_inline_payload(tmp_path):
    """With record_frames=True, a frame stream's line references the binary
    store (frame_ref/shape/dtype) instead of embedding the pixel payload."""
    from cognitive_runtime.runtime.frame_store import open_frame_store
    from cognitive_runtime.runtime.recorder import stream_event_from_log

    config = RuntimeConfig(
        episodes=1, seed=0, max_ticks_per_episode=50,
        record_dir=str(tmp_path), session_id="frames",
        program_config=FAST_CONFIG, record_frames=True,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=FAST_CONFIG),
        policy=RandomPolicy(ACTION_SPACE, seed=1), config=config,
    ).run()
    session_dir = os.path.join(str(tmp_path), "frames")
    assert os.path.isdir(os.path.join(session_dir, "frames"))

    stream_log = load_stream_log(session_dir, "episode_00000")
    pixel_lines = [r for r in stream_log if r.get("stream_id") == "vision.frame.pixels"]
    assert pixel_lines
    for record in pixel_lines:
        assert "payload" not in record
        assert not record.get("elided")
        assert "frame_ref" in record and "shape" in record and "dtype" in record

    frame_store = open_frame_store(session_dir)
    assert frame_store is not None
    for record in pixel_lines[:5]:
        event = stream_event_from_log(record, frame_store=frame_store)
        assert event.hash() == record["hash"]
    frame_store.close()


def test_default_config_still_elides_frames_hash_only(tmp_path):
    """record_frames defaults to False: frame streams stay hash-only and the
    binary store is never created (unchanged from before this store existed)."""
    runtime = _make_runtime(tmp_path, RandomPolicy(ACTION_SPACE, seed=1), session_id="noframes")
    runtime.run()
    session_dir = os.path.join(str(tmp_path), "noframes")
    assert not os.path.isdir(os.path.join(session_dir, "frames"))
    stream_log = load_stream_log(session_dir, "episode_00000")
    pixel_lines = [r for r in stream_log if r.get("stream_id") == "vision.frame.pixels"]
    assert pixel_lines and all(r.get("elided") for r in pixel_lines)


def test_pin_on_streams_pins_current_segment_on_trigger(tmp_path):
    """A configured pin-trigger stream firing this tick pins the frame
    store's current segment so it survives rotation."""
    import numpy as np

    from cognitive_runtime.core.streams.events import StreamEvent
    from cognitive_runtime.runtime.recorder import DecisionRecord, Recorder

    recorder = Recorder(
        record_dir=str(tmp_path), session_id="pin-test",
        record_streams=["vision.frame.pixels"],
        pin_on_streams=["event.damage_taken"],
    )
    recorder.write_session_metadata({})
    recorder.start_episode(0)
    assert recorder.frame_store.pinned_segments == []

    frame_event = StreamEvent(
        stream_id="vision.frame.pixels", modality="vision",
        timestamp=1.0, sequence_number=0, payload=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    damage_event = StreamEvent(
        stream_id="event.damage_taken", modality="event",
        timestamp=1.0, sequence_number=0, payload={"reason": "zombie"},
    )
    recorder.write_cognitive_tick(
        sensory_events=[frame_event, damage_event], motor_events=[],
        decision=DecisionRecord(
            tick_index=0, window_span=[0.0, 1.0],
            n_events_by_stream={"vision.frame.pixels": 1, "event.damage_taken": 1},
            motor_emitted=[], policy_name="test", latency_ms=0.0, reward_window_total=0.0,
        ),
    )
    assert len(recorder.frame_store.pinned_segments) == 1
    recorder.close()


def test_legacy_session_without_frame_store_still_loads_and_replays(tmp_path):
    """A session recorded before this format (inline pixel payload, no
    frame_hash_algorithm metadata, no frames/ dir) still loads for
    training/viewing, and replay skips only the now-incomparable pixel-frame
    hash check instead of falsely reporting tampering."""
    from cognitive_runtime.runtime.recorder import stream_event_from_log

    config = RuntimeConfig(
        episodes=1, seed=0, max_ticks_per_episode=50,
        record_dir=str(tmp_path), session_id="legacy",
        program_config=FAST_CONFIG, record_frames=True,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=FAST_CONFIG),
        policy=RandomPolicy(ACTION_SPACE, seed=1), config=config,
    ).run()
    session_dir = os.path.join(str(tmp_path), "legacy")

    # Rewrite the log the way a pre-migration recorder would have: pixel
    # frames inline as nested lists, hashed via the old JSON algorithm; drop
    # the frame_hash_algorithm marker and the binary frame store entirely.
    import shutil

    from cognitive_runtime.core.hashing import canonical_json
    import hashlib
    from cognitive_runtime.runtime.frame_store import open_frame_store

    frame_store = open_frame_store(session_dir)
    path = os.path.join(session_dir, "episode_00000.streams.jsonl")
    lines = open(path, encoding="utf-8").read().splitlines()
    rewritten = []
    for line in lines:
        record = json.loads(line)
        if record.get("stream_id") == "vision.frame.pixels" and "frame_ref" in record:
            array = frame_store.read_frame(record.pop("frame_ref")).copy()
            record.pop("shape", None)
            record.pop("dtype", None)
            payload = array.tolist()
            record["payload"] = payload
            legacy_hash = hashlib.sha1(
                canonical_json(
                    {
                        "stream_id": record["stream_id"],
                        "sequence_number": record["seq"],
                        "timestamp": record["timestamp"],
                        "payload": payload,
                    }
                ).encode("utf-8")
            ).hexdigest()
            record["hash"] = legacy_hash
        rewritten.append(json.dumps(record))
    frame_store.close()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rewritten) + "\n")
    shutil.rmtree(os.path.join(session_dir, "frames"))

    with open(os.path.join(session_dir, "session.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    meta.pop("frame_hash_algorithm", None)
    with open(os.path.join(session_dir, "session.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)

    # Loading still works: stream_event_from_log needs no frame_store at all.
    stream_log = load_stream_log(session_dir, "episode_00000")
    pixel_lines = [r for r in stream_log if r.get("stream_id") == "vision.frame.pixels"]
    assert pixel_lines and all("payload" in r for r in pixel_lines)
    for record in pixel_lines[:3]:
        assert stream_event_from_log(record).hash() == record["hash"]

    # Replay: every other stream still verifies; the pixel stream's hash
    # algorithm changed, so it's skipped rather than reported as tampering.
    results = replay_session(session_dir)
    assert len(results) == 1
    assert results[0].matched, results[0]
