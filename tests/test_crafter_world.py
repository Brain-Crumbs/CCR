"""CrafterWorld seam conformance (issue #89).

Mirrors ``tests/test_program_streams.py``'s Minecraft coverage: stream
catalog completeness, act/step parity, determinism, native cadences, NULL
ticks, rejected motor events, and the efference-copy round trip -- plus the
action/stream registry completeness checks issues #60/#21 established.
"""

import pytest

crafter = pytest.importorskip("crafter")

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.streams.bus import MotorStreamBus, SensoryStreamBus
from cognitive_runtime.core.streams.motor import publish_motor_command
from cognitive_runtime.programs.crafter.action_registry import CRAFTER_ACTION_REGISTRY
from cognitive_runtime.programs.crafter.actions import ACTION_SPACE, CRAFTER_ACTIONS
from cognitive_runtime.programs.crafter.adapter import CrafterWorld
from cognitive_runtime.programs.crafter.stream_registry import CRAFTER_STREAM_REGISTRY

FAST_CONFIG = {"episode_ticks": 200}


def _stream_program(seed, config=None):
    program = CrafterWorld(config=config or FAST_CONFIG)
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory, motor)
    program.reset(seed=seed)
    return program, sensory, motor


def _drive(seed, actions, config=None):
    program, sensory, motor = _stream_program(seed, config)
    ticks = [sensory.drain()]  # the initial post-reset snapshot
    for action in actions:
        if action is not None:
            publish_motor_command(motor, action, timestamp=0.0)
        program.step()
        ticks.append(sensory.drain())
    return program, ticks


# ------------------------------------------------------------------ mapping


def test_action_table_matches_the_live_crafter_package():
    """Hardcoded so the action space is importable without ``crafter``
    installed; this checks the table against the real package so the two
    can't silently drift."""
    env = crafter.Env()
    names_by_index = {index: name for name, index in CRAFTER_ACTIONS}
    assert env.action_names[0] == "noop"
    assert names_by_index[0] == "NULL"  # crafter's "noop" -> this Program's NULL convention
    for index, crafter_name in enumerate(env.action_names):
        if index == 0:
            continue
        assert names_by_index[index] == crafter_name.upper()


# ------------------------------------------------------------------ registries


def test_action_registry_is_complete():
    CRAFTER_ACTION_REGISTRY.assert_complete(ACTION_SPACE)


def test_stream_registry_is_complete():
    program = CrafterWorld(config=FAST_CONFIG)
    CRAFTER_STREAM_REGISTRY.assert_complete(program.stream_catalog())


# ------------------------------------------------------------------ catalog


def test_stream_catalog_covers_the_crafter_taxonomy():
    program = CrafterWorld(config=FAST_CONFIG)
    specs = {spec.stream_id: spec for spec in program.stream_catalog()}
    expected = {
        "vision.frame.grid", "vision.frame.pixels",
        "body.health", "body.food", "body.drink", "body.energy",
        "body.inventory", "body.sleeping", "body.alive",
        "spatial.position",
        "event.achievement", "event.died", "event.action_rejected",
        "reward.scalar",
    }
    assert set(specs) == expected


def test_pixel_stream_has_real_pixel_provenance():
    program, sensory, _ = _stream_program(0)
    snapshot = {e.stream_id: e.payload for e in sensory.drain()}
    pixels = snapshot["vision.frame.pixels"]
    assert pixels.shape == (64, 64, 3)
    assert pixels.dtype.name == "uint8"
    # Not a blank/constant frame: this is Crafter's own render, not a stub.
    assert pixels.min() != pixels.max()


# ------------------------------------------------------- act/step parity


def test_step_path_matches_legacy_act_path():
    seed = 7
    actions = [Action("MOVE_UP"), Action("MOVE_RIGHT"), Action("DO"), Action("NULL")] * 15

    legacy = CrafterWorld(config=FAST_CONFIG)
    legacy.reset(seed=seed)
    legacy_hashes = []
    for action in actions:
        legacy.act(action)
        legacy.reward()
        legacy_hashes.append(legacy.observe().hash())

    streamed, _, motor = _stream_program(seed)
    stream_hashes = []
    for action in actions:
        publish_motor_command(motor, action, timestamp=0.0)
        streamed.step()
        stream_hashes.append(streamed.observe().hash())

    assert stream_hashes == legacy_hashes


# --------------------------------------------------------- determinism


def test_same_seed_same_motor_events_give_identical_sensory_streams():
    actions = [Action("MOVE_UP"), Action("MOVE_LEFT"), Action("DO")] * 20

    def run():
        _, ticks = _drive(11, actions)
        return [
            [(e.stream_id, e.sequence_number, e.hash()) for e in tick]
            for tick in ticks
        ]

    first, second = run(), run()
    assert first == second  # byte-identical ids, seq numbers, hashes


def test_reset_seed_is_reproducible_across_instances():
    """reset(seed) must reproduce the world byte-for-byte -- crafter.Env's own
    reset() reseeds off an internal episode counter, so this only holds if
    CrafterWorld constructs a fresh Env per reset() (not calling env.reset()
    on a live instance twice)."""
    actions = [Action("MOVE_UP"), Action("DO")] * 10

    def run():
        program = CrafterWorld(config=FAST_CONFIG)
        program.reset(seed=42)
        hashes = []
        for action in actions:
            program.act(action)
            hashes.append(program.observe().hash())
        return hashes

    assert run() == run()


# ------------------------------------------------------- snapshot/restore


def test_snapshot_restore_round_trip_is_byte_identical():
    program, sensory, motor = _stream_program(0)
    sensory.drain()
    publish_motor_command(motor, Action("MOVE_UP"), timestamp=0.0)
    program.step()

    snapshot_id = program.snapshot()
    publish_motor_command(motor, Action("MOVE_RIGHT"), timestamp=0.0)
    program.step()
    diverged_hash = program.observe().hash()

    program.restore(snapshot_id)
    publish_motor_command(motor, Action("MOVE_RIGHT"), timestamp=0.0)
    program.step()
    replayed_hash = program.observe().hash()

    assert diverged_hash == replayed_hash


# --------------------------------------------------------------- cadence


def test_native_cadences_on_a_stationary_null_run():
    n_ticks = 60
    _, ticks = _drive(0, [None] * n_ticks)  # zero motor events every tick

    def count(stream_id):
        return sum(1 for tick in ticks for e in tick if e.stream_id == stream_id)

    initial = {e.stream_id for e in ticks[0]}
    assert {"vision.frame.grid", "vision.frame.pixels", "body.health",
            "spatial.position"} <= initial

    # Every-tick streams really are every tick.
    assert count("vision.frame.grid") == 1 + n_ticks
    assert count("vision.frame.pixels") == 1 + n_ticks
    assert count("reward.scalar") == n_ticks  # per tick, none in the snapshot


def test_zero_motor_tick_advances_the_world():
    program, ticks = _drive(0, [None] * 5)
    assert program.episode_stats()["final_tick"] == 5
    rewards = [e for tick in ticks[1:] for e in tick if e.stream_id == "reward.scalar"]
    assert len(rewards) == 5
    assert all("value" in e.payload and "components" in e.payload for e in rewards)


# ------------------------------------------------------- rejected motors


def test_malformed_motor_events_reject_but_the_world_still_steps():
    program, sensory, motor = _stream_program(0)
    sensory.drain()
    bad_payloads = [
        "not-a-dict",
        {"no_action": 1},
        {"action": 42},
        {"action": "EXPLODE"},  # unknown action name
    ]
    for i, payload in enumerate(bad_payloads):
        motor.publish("motor.command", payload, timestamp=0.0)
        program.step()
        events = sensory.drain()
        rejected = [e for e in events if e.stream_id == "event.action_rejected"]
        assert len(rejected) == 1, payload
        assert rejected[0].payload["reason"]
        assert program.episode_stats()["final_tick"] == i + 1  # world stepped

    # Two valid commands in one tick: first applies, second is rejected.
    publish_motor_command(motor, Action("MOVE_UP"), timestamp=0.0)
    publish_motor_command(motor, Action("MOVE_DOWN"), timestamp=0.0)
    program.step()
    events = sensory.drain()
    rejected = [e for e in events if e.stream_id == "event.action_rejected"]
    assert len(rejected) == 1 and "superseded" in rejected[0].payload["reason"]


# --------------------------------------------------------- efference copy


def test_efference_copy_round_trip_through_the_motor_bus():
    """A published motor.command reaches the world and is applied exactly
    once per tick -- the efference-copy contract the phase-1 acceptance
    criteria calls out explicitly."""
    program, sensory, motor = _stream_program(0)
    sensory.drain()
    before = program.observe().data["position"]

    publish_motor_command(motor, Action("MOVE_UP"), timestamp=0.0)
    program.step()
    after = program.observe().data["position"]

    assert after != before  # the drained command moved the agent


# -------------------------------------------------------------- achievements


def test_achievement_events_are_repeatable_counters():
    """Unlike Minecraft's once-only event.advancement, Crafter achievements
    are cumulative counters (e.g. wake_up increments every time)."""
    program, sensory, motor = _stream_program(0, config={"episode_ticks": 400})
    sensory.drain()
    counts = {}
    for _ in range(400):
        publish_motor_command(motor, Action("SLEEP"), timestamp=0.0)
        program.step()
        for e in sensory.drain():
            if e.stream_id == "event.achievement":
                counts[e.payload["id"]] = e.payload["count"]
        if program.is_complete():
            break
    assert any(count > 1 for count in counts.values())


# ------------------------------------------------------------ NULL convention


def test_null_action_is_a_real_action_not_crafter_specific():
    assert Action("NULL") in ACTION_SPACE
    assert Action("NULL").is_null
