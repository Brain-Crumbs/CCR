"""Phase-2 tests: the cognitive loop running over stream windows.

Covers determinism of sensory+motor sequences, the cognitive/program tick
ratio and window batching, NULL cognitive-tick accounting, per-stream rate
metrics, and the environment-agnostic import boundary.
"""

import json
import os
import subprocess
import sys

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.streams import (
    MotorStreamBus,
    SensoryStreamBus,
    TickSynchronizer,
)
from cognitive_runtime.policies import NullPolicy, RandomPolicy, ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.recorder import EpisodeSummary
from cognitive_runtime.runtime.replay import iter_cognitive_ticks, load_decisions
from cognitive_runtime.tools.replay_runner import replay_session

FAST_CONFIG = {"episode_ticks": 200, "world_size": 32}
NIGHT_CONFIG = {"episode_ticks": 600, "world_size": 32, "day_length": 400, "start_time": 300}


def _runtime(tmp_path, policy, session_id, config=FAST_CONFIG, seed=0, ratio=1, episodes=1):
    runtime_config = RuntimeConfig(
        episodes=episodes,
        seed=seed,
        max_ticks_per_episode=config["episode_ticks"],
        program_ticks_per_cognitive_tick=ratio,
        record_dir=str(tmp_path),
        session_id=session_id,
        program_config=config,
    )
    return CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config), policy=policy, config=runtime_config
    )


def _sequence(session_dir, episode_id="episode_00000"):
    """Per-tick (sensory hashes, motor emission hashes) from a recorded episode."""
    seq = []
    for decision, sensory, motor in iter_cognitive_ticks(session_dir, episode_id):
        seq.append(
            (
                tuple(r["hash"] for r in sensory),
                tuple(r["hash"] for r in motor),
            )
        )
    return seq


# ------------------------------------------------------------ determinism


def test_sensory_and_motor_sequences_are_deterministic(tmp_path):
    a = _runtime(tmp_path, ScriptedSurvivalPolicy(seed=2), "det-a", seed=5)
    b = _runtime(tmp_path, ScriptedSurvivalPolicy(seed=2), "det-b", seed=5)
    a.run()
    b.run()
    seq_a = _sequence(os.path.join(str(tmp_path), "det-a"))
    seq_b = _sequence(os.path.join(str(tmp_path), "det-b"))
    assert seq_a == seq_b
    assert len(seq_a) == 200


def test_recorded_stream_loop_replays_and_verifies(tmp_path):
    runtime = _runtime(tmp_path, RandomPolicy(ACTION_SPACE, seed=9), "replay-me", seed=7)
    runtime.run()
    results = replay_session(os.path.join(str(tmp_path), "replay-me"))
    assert len(results) == 1
    assert results[0].matched, results[0]
    assert results[0].ticks_replayed == 200


def test_replay_detects_tampered_motor_emission(tmp_path):
    runtime = _runtime(tmp_path, RandomPolicy(ACTION_SPACE, seed=3), "tamper", seed=1)
    runtime.run()
    session_dir = os.path.join(str(tmp_path), "tamper")
    path = os.path.join(session_dir, "episode_00000.streams.jsonl")
    lines = open(path, encoding="utf-8").read().splitlines()
    # Flip one motor payload: the world steps differently, so the regenerated
    # sensory hashes diverge downstream.
    for i, line in enumerate(lines):
        record = json.loads(line)
        if record["dir"] == "motor":
            action = record["payload"]["action"]
            record["payload"]["action"] = "SPRINT" if action != "SPRINT" else "ATTACK"
            lines[i] = json.dumps(record)
            break
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    results = replay_session(session_dir)
    assert not results[0].matched


def test_replay_verifies_with_elided_sensory_payloads(tmp_path):
    """exclude_streams elides vision.* payloads to hash-only lines; replay must
    still verify tick-for-tick from those hashes."""
    from cognitive_runtime.runtime.config import RuntimeConfig
    from cognitive_runtime.runtime.loop import CognitiveRuntime

    config = RuntimeConfig(
        episodes=1, seed=4, max_ticks_per_episode=200,
        record_dir=str(tmp_path), session_id="elide",
        program_config=FAST_CONFIG, exclude_streams=["vision.*"],
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=FAST_CONFIG),
        policy=ScriptedSurvivalPolicy(seed=2), config=config,
    ).run()
    session_dir = os.path.join(str(tmp_path), "elide")
    lines = open(
        os.path.join(session_dir, "episode_00000.streams.jsonl"), encoding="utf-8"
    ).read().splitlines()
    elided = [json.loads(l) for l in lines if json.loads(l).get("elided")]
    assert elided and all("payload" not in r for r in elided)
    results = replay_session(session_dir)
    assert results[0].matched
    assert results[0].ticks_replayed == 200


# --------------------------------------------------- NULL-tick accounting


def test_null_cognitive_ticks_are_counted(tmp_path):
    runtime = _runtime(tmp_path, NullPolicy(), "nulls", episodes=2)
    summaries = runtime.run()
    for summary in summaries:
        assert summary.duration_ticks == 200
        assert summary.null_action_ticks == 200  # every emission is []
        assert summary.termination_reason == "episode_ticks"
    # Every recorded decision is a NULL emission (no motor events).
    decisions = load_decisions(os.path.join(str(tmp_path), "nulls"), "episode_00000")
    assert len(decisions) == 200
    assert all(d["motor_emitted"] == [] for d in decisions)


# --------------------------------------------- cognitive / program ratio


def test_cognitive_tick_ratio_decouples_decision_rate(tmp_path):
    runtime = _runtime(tmp_path, ScriptedSurvivalPolicy(seed=1), "ratio", ratio=4)
    summaries = runtime.run()
    summary = summaries[0]
    # The agent decides 4x less often, but the world still advances every
    # program tick: 200 program ticks == 50 cognitive ticks.
    assert summary.duration_ticks == 50
    assert summary.program_ticks_per_cognitive_tick == 4
    assert summary.program_stats["final_tick"] == 200


def test_ratio_batches_events_into_windows():
    """Stepping the program N times before collecting groups N ticks of the
    every-tick streams into one window (window grouping)."""
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory, motor)
    program.reset(seed=0)
    sensory.drain()  # discard the reset snapshot
    sync = TickSynchronizer(program_ticks_per_cognitive_tick=4)

    for _ in range(4):
        program.step()
    window = sync.collect(sensory, now=None)
    # world.time publishes every program tick -> 4 grouped in this window.
    assert len(window.by_stream["world.time"]) == 4
    times = [e.payload["time_of_day"] for e in window.by_stream["world.time"]]
    assert times == sorted(times) and len(set(times)) == 4


# ----------------------------------------------- per-stream rate metrics


def test_per_stream_rate_metrics_in_summary(tmp_path):
    runtime = _runtime(tmp_path, ScriptedSurvivalPolicy(seed=1), "rates")
    summary = runtime.run()[0]
    rates = summary.stream_event_rates
    assert rates  # non-empty
    # Every-tick streams have the highest rates; on-change streams are lower.
    assert rates["world.time"] > 0.0
    assert rates["reward.scalar"] > 0.0
    assert rates["world.time"] >= rates["spatial.position"]
    # The metric survives a round-trip through the summary JSON.
    with open(os.path.join(str(tmp_path), "rates", "episode_00000.summary.json")) as fh:
        raw = json.load(fh)
    assert raw["stream_event_rates"]["world.time"] == rates["world.time"]
    assert "silent_streams" in raw


# ------------------------------------------------- qualitative ordering


def test_reward_ordering_scripted_beats_random_beats_null(tmp_path):
    def total_reward(policy, session_id):
        runtime = _runtime(
            tmp_path, policy, session_id, config=NIGHT_CONFIG, seed=100, episodes=2
        )
        return sum(s.total_reward for s in runtime.run())

    scripted = total_reward(ScriptedSurvivalPolicy(seed=1), "ord-scripted")
    random = total_reward(RandomPolicy(ACTION_SPACE, seed=0), "ord-random")
    null = total_reward(NullPolicy(), "ord-null")
    assert scripted > random > null


# ------------------------------------------------ architecture boundary


def test_core_and_runtime_do_not_import_programs():
    code = (
        "import sys; import cognitive_runtime.runtime.loop; "
        "import cognitive_runtime.runtime.replay; import cognitive_runtime.core; "
        "bad = [m for m in sys.modules if m.startswith('cognitive_runtime.programs')]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
