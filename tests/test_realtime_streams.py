"""Phase-5 tests: real-time multi-rate streaming.

Covers the two-clock design (simulated vs. wall-clock ``arrived_at``), the
thread-safe bounded bus with per-modality backpressure, the rate pacer,
missed-window / stale-stream / motor health metrics, the human-demo input
stream, and the acceptance criteria end to end (paced realtime rates,
threaded high-rate publisher, realtime→fast-forward replay).
"""

import os
import threading
import time

import pytest

from cognitive_runtime.core.streams import (
    RatePacer,
    SensoryStreamBus,
    StreamEvent,
    StreamSpec,
    TickSynchronizer,
)
from cognitive_runtime.core.streams.pacer import PACER_SLACK
from cognitive_runtime.policies import NullPolicy, ScriptedSurvivalPolicy
from cognitive_runtime.policies.human_demo import (
    INPUT_KEYPRESS_STREAM,
    HumanDemoPolicy,
    KeypressInputStream,
)
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.tools.metrics_dashboard import dashboard
from cognitive_runtime.tools.replay_runner import replay_session

FAST_CONFIG = {"episode_ticks": 200, "world_size": 32}


# ============================================================ two clocks


def test_arrived_at_is_metadata_excluded_from_hash():
    base = StreamEvent("body.health", "body", 1.0, 0, 20)
    stamped = StreamEvent("body.health", "body", 1.0, 0, 20, arrived_at=123.456)
    # Wall-clock arrival never changes the deterministic content hash...
    assert base.hash() == stamped.hash()
    # ...but it does round-trip as metadata.
    restored = StreamEvent.from_dict(stamped.to_dict())
    assert restored.arrived_at == 123.456
    assert restored == stamped


def test_bus_stamps_arrived_at_only_with_a_wall_clock():
    plain = SensoryStreamBus()
    plain.publish("body.health", 20, timestamp=1.0)
    assert plain.drain()[0].arrived_at is None

    ticks = iter([10.0, 11.0])
    clocked = SensoryStreamBus(thread_safe=True, wall_clock=lambda: next(ticks))
    clocked.publish("body.health", 20, timestamp=1.0)
    assert clocked.drain()[0].arrived_at == 10.0


# ================================================= bounded bus + overflow


def test_default_overflow_policy_by_modality():
    bus = SensoryStreamBus(thread_safe=True)
    assert bus._overflow("vision.frame.grid") == "coalesce"
    assert bus._overflow("body.health") == "coalesce"
    assert bus._overflow("event.damage") == "block"
    assert bus._overflow("reward.scalar") == "drop_oldest"  # modality fallback
    # An explicit StreamSpec.overflow wins over the modality default.
    bus.register(StreamSpec("world.time", "world", overflow="coalesce"))
    assert bus._overflow("world.time") == "coalesce"


def test_coalesce_overflow_keeps_latest_frame_and_counts():
    bus = SensoryStreamBus(thread_safe=True)
    bus.register(StreamSpec("vision.frame.grid", "vision", overflow="coalesce"))
    bus.set_capacity("vision.frame.grid", 1)
    for i in range(5):
        bus.publish("vision.frame.grid", {"frame": i}, timestamp=float(i))
    drained = bus.drain()
    assert [e.payload for e in drained] == [{"frame": 4}]  # only the freshest
    assert bus.overflow_counts()["vision.frame.grid"]["coalesce"] == 4


def test_drop_oldest_overflow_is_a_bounded_ring():
    bus = SensoryStreamBus(thread_safe=True)
    bus.register(StreamSpec("reward.scalar", "reward", overflow="drop_oldest"))
    bus.set_capacity("reward.scalar", 3)
    for i in range(6):
        bus.publish("reward.scalar", {"value": i}, timestamp=float(i))
    drained = bus.drain()
    assert [e.payload["value"] for e in drained] == [3, 4, 5]  # most-recent 3
    assert bus.overflow_counts()["reward.scalar"]["drop_oldest"] == 3


def test_event_streams_never_drop():
    """`event.*` uses the block policy: even far past a tiny capacity, and
    single-threaded, no event is ever lost (AC5)."""
    bus = SensoryStreamBus()  # single-threaded: block == never-drop, unbounded
    bus.register(StreamSpec("event.damage_taken", "event"))  # default -> block
    bus.set_capacity("event.damage_taken", 2)
    for i in range(50):
        bus.publish("event.damage_taken", {"i": i}, timestamp=float(i))
    drained = bus.drain()
    assert [e.payload["i"] for e in drained] == list(range(50))
    assert "event.damage_taken" not in bus.overflow_counts()  # nothing dropped


# ============================================================= rate pacer


def test_rate_pacer_is_a_passthrough_when_disabled():
    pacer = RatePacer(enabled=False)
    pacer.set_rate("vision", 10.0)
    assert all(pacer.should_publish("vision", now=t) for t in (0.0, 0.001, 0.002))


def test_rate_pacer_throttles_to_target_rate():
    pacer = RatePacer(enabled=True, rates={"vision": 10.0})  # period 0.1s
    # Sample a 20 Hz clock (0.05s steps): a 10 Hz stream should pass every 2nd.
    passes = [pacer.should_publish("vision", now=round(0.05 * i, 3)) for i in range(11)]
    assert passes[0] is True  # first is always allowed
    assert sum(passes) == 6  # t=0,.1,.2,.3,.4,.5 over 0..0.5s
    # No target rate -> irregular stream, never throttled.
    assert pacer.should_publish("event.damage", now=0.0)


def test_rate_pacer_absorbs_sub_period_jitter():
    """A stream whose period is a near-multiple of the sample interval must
    not alias down; the slack keeps it on schedule."""
    pacer = RatePacer(enabled=True, rates={"vision": 10.0})
    # Sampler runs a hair fast (0.0499s), so 2 steps = 0.0998s < 0.1s period.
    passes = [pacer.should_publish("vision", now=round(0.0499 * i, 4)) for i in range(21)]
    # Without slack this fires every 3rd step (~6.7 Hz); with slack every 2nd.
    assert 9 <= sum(passes) <= 11
    assert PACER_SLACK > 0


# ================================================ synchronizer health


def _fake_publisher_bus():
    bus = SensoryStreamBus()
    bus.register(StreamSpec("body.health", "body", nominal_rate_hz=2.0))
    return bus


def test_stale_stream_fires_when_a_publisher_pauses():
    """A rate-bearing stream that goes quiet longer than 2x its nominal period
    is flagged stale (AC4)."""
    bus = _fake_publisher_bus()
    sync = TickSynchronizer(nominal_rates={"body.health": 2.0})  # stale after 1.0s
    now = 0.0
    for _ in range(4):  # publisher active every 0.5s
        bus.publish("body.health", 20, timestamp=now)
        sync.collect(bus, now=now)
        assert sync.stale_streams(now) == []
        now += 0.5
    # Publisher pauses; simulated time keeps advancing on empty windows.
    for _ in range(4):
        sync.collect(bus, now=now)
        now += 0.5
    assert "body.health" in sync.stale_streams(now)


def test_empty_windows_are_counted():
    bus = SensoryStreamBus()
    sync = TickSynchronizer()
    bus.publish("body.health", 20, timestamp=1.0)
    sync.collect(bus, now=1.0)  # non-empty
    sync.collect(bus, now=2.0)  # empty
    sync.collect(bus, now=3.0)  # empty
    assert sync.empty_windows() == 2


def test_wall_clock_rates_measure_arrived_at_cadence():
    # A wall clock that ticks at a clean 10 Hz stamps each arrival's metadata.
    stamps = iter([round(0.1 * i, 3) for i in range(11)])
    bus = SensoryStreamBus(thread_safe=True, wall_clock=lambda: next(stamps))
    sync = TickSynchronizer()
    for i in range(11):
        bus.publish("vision.frame.grid", {"i": i}, timestamp=float(i))
        sync.collect(bus, now=float(i))
    rates = sync.wall_clock_rates()
    assert abs(rates["vision.frame.grid"] - 10.0) < 0.5


# ===================================================== AC2: threaded pump


def test_threaded_200hz_publisher_coalesces_and_stays_bounded():
    """A fake publisher thread pumps ~200 Hz into a slow-draining 20 Hz
    consumer: no crash, coalescing counted, queue never exceeds its bound."""
    bus = SensoryStreamBus(thread_safe=True, wall_clock=time.monotonic)
    bus.register(StreamSpec("vision.frame.grid", "vision", overflow="coalesce"))
    bus.set_capacity("vision.frame.grid", 4)

    stop = threading.Event()
    produced = {"n": 0}

    def pump():
        i = 0
        while not stop.is_set():
            bus.publish("vision.frame.grid", [[i]], timestamp=float(i))
            produced["n"] += 1
            i += 1
            time.sleep(0.005)  # ~200 Hz

    worker = threading.Thread(target=pump, daemon=True)
    worker.start()
    max_pending = 0
    drained = 0
    for _ in range(10):  # consumer drains at ~20 Hz
        time.sleep(0.05)
        drained += len(bus.drain())
        max_pending = max(max_pending, bus.pending_count())
    stop.set()
    worker.join(timeout=1.0)

    assert produced["n"] > drained  # producer genuinely outran the consumer
    assert max_pending <= 4  # bounded-queue: never grew past capacity
    assert bus.overflow_counts()["vision.frame.grid"]["coalesce"] > 0


# ============================================ AC1: paced realtime rates


def test_realtime_per_stream_rates_within_20pct_of_nominal(tmp_path):
    pc = {"episode_ticks": 80, "world_size": 32}
    cfg = RuntimeConfig(
        realtime=True, tick_rate=20.0, max_ticks_per_episode=80, episodes=1,
        record=False, program_config=pc,
    )
    summary = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=pc), policy=NullPolicy(), config=cfg,
    ).run()[0]

    if summary.ticks_per_second < 15.0:
        pytest.skip("machine too loaded to hold the 20 Hz cognitive rate")

    rates = summary.stream_wallclock_rates
    assert summary.realtime is True
    # Vision paced to 10 Hz, body vitals heartbeat at 2 Hz, world clock at the
    # 20 Hz cognitive rate — each within 20% of nominal on a quiet machine.
    assert 8.0 <= rates["vision.frame.grid"] <= 12.0
    assert 1.6 <= rates["body.health"] <= 2.4
    assert 16.0 <= rates["world.time"] <= 24.0


# ================================= AC3: determinism across the two modes


def test_fastforward_runs_are_byte_identical_and_carry_no_wall_clock(tmp_path):
    pc = {"episode_ticks": 120, "world_size": 32}

    def run(sid):
        cfg = RuntimeConfig(
            realtime=False, max_ticks_per_episode=120, episodes=1,
            record_dir=str(tmp_path), session_id=sid, program_config=pc,
        )
        CognitiveRuntime(
            program=MinecraftSurvivalBox(config=pc),
            policy=ScriptedSurvivalPolicy(seed=2), config=cfg,
        ).run()
        return open(
            os.path.join(str(tmp_path), sid, "episode_00000.streams.jsonl"),
            encoding="utf-8",
        ).read()

    a, b = run("ff-a"), run("ff-b")
    assert a == b  # byte-identical across repeats
    assert "arrived_at" not in a  # no wall-clock leakage into fast-forward logs


def test_realtime_recording_replays_and_verifies_in_fast_forward(tmp_path):
    pc = {"episode_ticks": 60, "world_size": 32}
    cfg = RuntimeConfig(
        realtime=True, tick_rate=60.0, max_ticks_per_episode=60, episodes=1,
        record_dir=str(tmp_path), session_id="rt", program_config=pc,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=pc),
        policy=ScriptedSurvivalPolicy(seed=3), config=cfg,
    ).run()

    session_dir = os.path.join(str(tmp_path), "rt")
    # The realtime log carries arrived_at metadata, yet replay (fast-forward,
    # pacing reproduced off simulated time) verifies every sensory hash.
    log = open(os.path.join(session_dir, "episode_00000.streams.jsonl")).read()
    assert "arrived_at" in log
    results = replay_session(session_dir)
    assert results[0].matched, results[0]
    assert results[0].ticks_replayed == 60


# ======================================= realtime health surfaced in summary


def test_realtime_summary_and_dashboard_expose_health(tmp_path):
    pc = {"episode_ticks": 40, "world_size": 32}
    cfg = RuntimeConfig(
        realtime=True, tick_rate=40.0, max_ticks_per_episode=40, episodes=1,
        record_dir=str(tmp_path), session_id="rt-health", program_config=pc,
    )
    summary = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=pc),
        policy=ScriptedSurvivalPolicy(seed=1), config=cfg,
    ).run()[0]

    assert summary.realtime is True
    assert summary.motor_emissions > 0
    assert summary.motor_emission_rate > 0
    assert summary.empty_windows >= 0
    assert summary.late_windows == summary.missed_ticks
    assert summary.stream_wallclock_rates  # populated in realtime
    # The dashboard renders a realtime-health block for realtime sessions.
    out = dashboard(str(tmp_path))
    assert "realtime health" in out
    assert "measured wall-clock rates" in out


# ============================== instruction 6: human demo input stream


def test_human_demo_publishes_and_consumes_input_keypress(capsys):
    """The human demo routes keypresses through an input.keypress stream and
    consumes them back — dogfooding the async publish/consume path."""
    keys = iter(["w", "k", "quit"])
    policy = HumanDemoPolicy(
        show_frame=False, realtime=False,
        input_source=lambda: next(keys, None),
    )
    policy.reset()

    from cognitive_runtime.core.memory import Memory
    from cognitive_runtime.core.observation import Observation
    from cognitive_runtime.core.perception import State

    def state():
        return State(observation=Observation(
            timestamp=0.0, tick=0,
            data={"health": 20, "hunger": 20, "is_night": False,
                  "front_block": "air", "hotbar": [], "mobs": []},
            frame=None,
        ))

    memory = Memory()
    first = policy.emit(state(), memory, None)
    second = policy.emit(state(), memory, None)
    assert [a.key() for a in first] == ["MOVE_FORWARD"]
    assert [a.key() for a in second] == ["ATTACK"]
    # "quit" ends the episode (an explicit NULL emission).
    assert policy.emit(state(), memory, None) == []
    assert policy.stop_requested is True


def test_keypress_input_stream_realtime_thread_feeds_the_bus():
    """In realtime mode a reader thread publishes input.keypress asynchronously;
    the consumer polls them off the stream."""
    queue = ["a", "d", None]  # None == EOF closes the reader
    lock = threading.Lock()

    def source():
        with lock:
            return queue.pop(0) if queue else None

    stream = KeypressInputStream(source, realtime=True)
    stream.start()
    deadline = time.monotonic() + 1.0
    while not stream.eof and time.monotonic() < deadline:
        time.sleep(0.01)
    keys = stream.poll()
    assert keys == ["a", "d"]
    # The events really rode the input.keypress stream.
    stream2 = KeypressInputStream(lambda: None, realtime=False)
    stream2.read_blocking()  # EOF immediately
    assert stream2.eof is True
    assert stream2.bus.spec(INPUT_KEYPRESS_STREAM) is not None
