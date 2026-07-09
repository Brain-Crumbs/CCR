"""Core abstraction tests: actions, observations, memory."""

from cognitive_runtime.core import (
    Action,
    Memory,
    NULL_ACTION,
    Observation,
    RewardSignal,
)
from cognitive_runtime.core.streams import SensoryStreamBus, TickSynchronizer


def test_action_key_roundtrip():
    action = Action.make("SELECT_HOTBAR_SLOT", slot=3)
    assert action.key() == "SELECT_HOTBAR_SLOT:slot=3"
    assert Action.from_key(action.key()) == action
    assert Action.from_key("NULL") == NULL_ACTION
    assert NULL_ACTION.is_null


def test_observation_hash_is_deterministic_and_content_sensitive():
    a = Observation(timestamp=1.0, tick=5, data={"health": 20, "pos": [1, 2]})
    b = Observation(timestamp=9.9, tick=5, data={"pos": [1, 2], "health": 20})
    assert a.hash() == b.hash()  # timestamp excluded, key order irrelevant
    c = Observation(timestamp=1.0, tick=5, data={"health": 19, "pos": [1, 2]})
    assert a.hash() != c.hash()


def test_reward_signal_from_components():
    signal = RewardSignal.from_components({"a": 0.5, "b": -0.2, "zero": 0.0})
    assert abs(signal.value - 0.3) < 1e-9
    assert "zero" not in signal.components


def _window(bus, sync, stream_id, payload, timestamp):
    bus.publish(stream_id, payload, timestamp)
    return sync.collect(bus, now=timestamp)


def test_memory_repetition_and_novelty():
    memory = Memory(capacity=16)
    bus, sync = SensoryStreamBus(), TickSynchronizer()
    for i in range(3):
        memory.update(_window(bus, sync, "body.health", float(i), timestamp=float(i)))
        memory.record_actions([])  # empty emission == NULL tick
    assert memory.repeated_action_streak() == 3
    memory.record_actions([Action("MOVE_FORWARD")])
    assert memory.repeated_action_streak() == 1
    assert memory.novelty_rate() == 1.0
    # Trends read numeric stream payloads straight from the buffer (0,1,2).
    assert memory.stream_trend("body.health", window=8) == 1.0
    # Re-publishing an already-seen window payload is not novel.
    memory.update(_window(bus, sync, "body.health", 0.0, timestamp=0.0))
    assert not memory.last_observation_was_novel
