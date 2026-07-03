"""Phase-0 stream primitive tests: events, buses, buffer, synchronizer,
encoder registry.  Determinism of ordering and hashing is the point."""

import json

import pytest

from cognitive_runtime.core.streams import (
    MODALITIES,
    PassthroughEncoder,
    SensoryStreamBus,
    MotorStreamBus,
    StreamEncoderRegistry,
    StreamEvent,
    StreamSpec,
    TemporalBuffer,
    TickSynchronizer,
)


def _publish_interleaved(bus):
    """Interleaved multi-stream publishing with ties on timestamp."""
    bus.publish("body.health", 20, timestamp=1.0)
    bus.publish("vision.frame.grid", [[1, 2], [3, 4]], timestamp=1.0)
    bus.publish("body.health", 19, timestamp=2.0)
    bus.publish("event.damage", {"amount": 1}, timestamp=2.0)
    bus.publish("body.hunger", 17, timestamp=1.0)
    bus.publish("body.health", 18, timestamp=2.0)


# -- events -------------------------------------------------------------------


def test_stream_event_hash_is_stable_and_payload_sensitive():
    a = StreamEvent("body.health", "body", 1.0, 0, {"hp": 20, "max": 20})
    b = StreamEvent("body.health", "body", 1.0, 0, {"max": 20, "hp": 20})
    assert a.hash() == b.hash()  # key order irrelevant
    c = StreamEvent("body.health", "body", 1.0, 0, {"hp": 19, "max": 20})
    assert a.hash() != c.hash()  # payload-sensitive
    d = StreamEvent("body.health", "body", 1.0, 1, {"hp": 20, "max": 20})
    assert a.hash() != d.hash()  # sequence-sensitive
    e = StreamEvent("body.health", "body", 2.0, 0, {"hp": 20, "max": 20})
    assert a.hash() != e.hash()  # timestamp-sensitive (simulated time)
    # confidence/source are not replay-relevant
    f = StreamEvent("body.health", "body", 1.0, 0, {"hp": 20, "max": 20},
                    confidence=0.5, source="sim")
    assert a.hash() == f.hash()


def test_stream_event_json_roundtrip():
    event = StreamEvent(
        stream_id="vision.frame.grid",
        modality="vision",
        timestamp=3.5,
        sequence_number=7,
        payload={"grid": [[0, 1], [2, 3]], "night": False},
        confidence=0.9,
        source="survival_sim",
    )
    restored = StreamEvent.from_dict(json.loads(json.dumps(event.to_dict())))
    assert restored == event
    assert restored.hash() == event.hash()


def test_stream_identity_validation():
    with pytest.raises(ValueError):
        StreamEvent("Body.Health", "body", 0.0, 0, 1)  # uppercase
    with pytest.raises(ValueError):
        StreamEvent("body..health", "body", 0.0, 0, 1)  # empty segment
    with pytest.raises(ValueError):
        StreamEvent("body.health", "sound", 0.0, 0, 1)  # unknown modality
    with pytest.raises(ValueError):
        StreamEvent("body.health", "vision", 0.0, 0, 1)  # modality mismatch
    with pytest.raises(ValueError):
        StreamSpec("vision.frame", "body")
    # ids not starting with a modality segment may declare any modality
    StreamEvent("heartbeat", "body", 0.0, 0, 1)


# -- bus ----------------------------------------------------------------------


def test_per_stream_sequence_numbers_are_monotonic():
    bus = SensoryStreamBus()
    _publish_interleaved(bus)
    events = bus.drain()
    health = [e.sequence_number for e in events if e.stream_id == "body.health"]
    assert health == [0, 1, 2]
    assert [e.sequence_number for e in events if e.stream_id == "body.hunger"] == [0]
    # numbering continues across drains
    bus.publish("body.health", 17, timestamp=3.0)
    assert bus.drain()[0].sequence_number == 3


def test_drain_ordering_is_deterministic_under_interleaving():
    def run():
        bus = SensoryStreamBus()
        _publish_interleaved(bus)
        return [(e.stream_id, e.sequence_number, e.timestamp) for e in bus.drain()]

    first = run()
    assert first == run()  # same inputs => identical order across runs
    assert first == sorted(first, key=lambda t: (t[2], t[0], t[1]))
    # ties on timestamp broken by stream_id: body.health before body.hunger
    assert first[0][0] == "body.health"
    assert first[1][0] == "body.hunger"


def test_drain_clears_pending():
    bus = SensoryStreamBus()
    _publish_interleaved(bus)
    assert bus.pending_count() == 6
    assert len(bus.drain()) == 6
    assert bus.drain() == []


def test_glob_subscription_filtering():
    bus = SensoryStreamBus()
    body = bus.subscribe("body.*")
    events = bus.subscribe("event.*")
    everything = bus.subscribe("*")
    _publish_interleaved(bus)

    body_events = body.drain()
    assert {e.stream_id for e in body_events} == {"body.health", "body.hunger"}
    assert all(body.matches(e) for e in body_events)

    event_events = events.drain()
    assert [e.stream_id for e in event_events] == ["event.damage"]

    rest = everything.drain()
    assert [e.stream_id for e in rest] == ["vision.frame.grid"]
    assert bus.pending_count() == 0


def test_bus_catalog_and_modality_inference():
    bus = SensoryStreamBus()
    spec = StreamSpec("heartbeat", "body", nominal_rate_hz=1.0)
    bus.register(spec)
    bus.register(StreamSpec("body.health", "body", nominal_rate_hz=None))
    assert [s.stream_id for s in bus.catalog()] == ["body.health", "heartbeat"]
    # modality from the registered spec, not the id
    assert bus.publish("heartbeat", 1, timestamp=0.0).modality == "body"
    # unregistered stream without a modality prefix is rejected
    with pytest.raises(ValueError):
        bus.publish("mystery.value", 1, timestamp=0.0)


def test_bus_reset_clears_queue_and_sequences_but_keeps_catalog():
    bus = MotorStreamBus()
    bus.register(StreamSpec("motor.command", "motor"))
    bus.publish("motor.command", {"name": "MOVE_FORWARD"}, timestamp=1.0)
    bus.reset()
    assert bus.pending_count() == 0
    assert len(bus.catalog()) == 1
    event = bus.publish("motor.command", {"name": "NULL"}, timestamp=0.0)
    assert event.sequence_number == 0  # counters restart per episode


# -- temporal buffer ------------------------------------------------------------


def _event(stream_id, modality, timestamp, seq, payload):
    return StreamEvent(stream_id, modality, timestamp, seq, payload)


def test_temporal_buffer_latest_window_and_eviction():
    buffer = TemporalBuffer(default_capacity=8, capacity_by_modality={"vision": 2})
    for i in range(4):
        buffer.append(_event("vision.frame.grid", "vision", float(i), i, [[i]]))
        buffer.append(_event("body.health", "body", float(i), i, 20 - i))

    assert buffer.latest("body.health").payload == 17
    assert buffer.latest("missing.stream") is None
    assert [e.payload for e in buffer.window("body.health", 2)] == [18, 17]
    # vision capacity 2: oldest frames evicted
    assert [e.sequence_number for e in buffer.window("vision.frame.grid", 10)] == [2, 3]
    assert buffer.streams() == ["body.health", "vision.frame.grid"]


def test_temporal_buffer_events_since_is_deterministic():
    buffer = TemporalBuffer()
    buffer.append(_event("body.hunger", "body", 1.0, 0, 17))
    buffer.append(_event("body.health", "body", 1.0, 0, 20))
    buffer.append(_event("body.health", "body", 2.0, 1, 19))
    since = buffer.events_since(0.5)
    assert [(e.timestamp, e.stream_id) for e in since] == [
        (1.0, "body.health"),
        (1.0, "body.hunger"),
        (2.0, "body.health"),
    ]
    assert buffer.events_since(1.0) == [since[-1]]  # strictly after
    buffer.reset()
    assert buffer.streams() == []


# -- tick synchronizer -----------------------------------------------------------


def test_tick_synchronizer_groups_events_into_windows():
    bus = SensoryStreamBus()
    sync = TickSynchronizer()
    _publish_interleaved(bus)
    window = sync.collect(bus, now=2.0)
    assert window.tick_index == 0
    assert window.started_at == 0.0
    assert window.ended_at == 2.0
    assert len(window.events) == 4 + 2  # all six events
    assert set(window.by_stream) == {
        "body.health", "body.hunger", "vision.frame.grid", "event.damage",
    }
    assert [e.payload for e in window.by_stream["body.health"]] == [20, 19, 18]

    empty = sync.collect(bus, now=3.0)
    assert empty.tick_index == 1
    assert empty.started_at == 2.0
    assert empty.is_empty


def test_tick_synchronizer_tracks_arrivals_and_gaps():
    bus = SensoryStreamBus()
    sync = TickSynchronizer()
    bus.publish("body.health", 20, timestamp=1.0)
    bus.publish("event.damage", {"amount": 1}, timestamp=1.0)
    sync.collect(bus, now=1.0)
    for step in range(2, 5):
        bus.publish("body.health", 20, timestamp=float(step))
        sync.collect(bus, now=float(step))

    assert sync.arrival_counts() == {"body.health": 4, "event.damage": 1}
    assert sync.silent_streams(min_windows=3) == ["event.damage"]
    assert sync.silent_streams(min_windows=4) == []
    sync.reset()
    assert sync.arrival_counts() == {}


def test_tick_synchronizer_cognitive_tick_ratio():
    sync = TickSynchronizer(program_ticks_per_cognitive_tick=3)
    boundaries = [t for t in range(9) if sync.is_cognitive_tick_boundary(t)]
    assert boundaries == [2, 5, 8]
    assert all(
        TickSynchronizer().is_cognitive_tick_boundary(t) for t in range(5)
    )  # default ratio 1: every program tick
    with pytest.raises(ValueError):
        TickSynchronizer(program_ticks_per_cognitive_tick=0)


# -- encoder registry -------------------------------------------------------------


def test_registry_passthrough_encoding():
    bus = SensoryStreamBus()
    sync = TickSynchronizer()
    registry = StreamEncoderRegistry()
    registry.register("body.*", PassthroughEncoder())
    registry.register("vision.*", PassthroughEncoder())

    bus.publish("body.health", 20, timestamp=1.0)
    bus.publish("body.position", {"x": 1.5, "z": -2.0, "name": "spawn"}, timestamp=1.0)
    bus.publish("vision.frame.grid", [[1, 2], [3, 4]], timestamp=1.0)
    bus.publish("event.death", {"cause": "creeper"}, timestamp=1.0)  # no encoder
    window = sync.collect(bus, now=1.0)

    tokens = registry.encode_window(window)
    assert [t.stream_id for t in tokens] == [
        "body.health", "body.position", "vision.frame.grid",
    ]
    by_id = {t.stream_id: t for t in tokens}
    assert by_id["body.health"].vector == [20.0]
    assert by_id["body.position"].vector == [1.5, -2.0]  # sorted keys, non-numeric skipped
    assert by_id["vision.frame.grid"].vector == [1.0, 2.0, 3.0, 4.0]
    assert by_id["body.health"].modality == "body"
    assert by_id["body.health"].timestamp == 1.0


def test_passthrough_encoder_uses_latest_event_and_skips_non_numeric():
    encoder = PassthroughEncoder()
    events = [
        _event("body.health", "body", 1.0, 0, 20),
        _event("body.health", "body", 2.0, 1, 18),
    ]
    token = encoder.encode(events)
    assert token.vector == [18.0]
    assert token.timestamp == 2.0
    assert encoder.encode([]) is None
    assert encoder.encode([_event("language.chat", "language", 0.0, 0, "hi")]) is None


# -- architecture guards -----------------------------------------------------------


def test_streams_package_does_not_import_programs():
    import subprocess
    import sys

    code = (
        "import sys; import cognitive_runtime.core.streams; "
        "bad = [m for m in sys.modules if m.startswith('cognitive_runtime.programs')]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_modalities_match_tracking_issue_taxonomy():
    assert MODALITIES == {
        "body", "vision", "spatial", "audio", "event",
        "reward", "language", "input", "world", "motor",
    }
