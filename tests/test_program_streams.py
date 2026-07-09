"""Phase-1 tests: programs publish sensory streams, consume motor streams.

Covers the SurvivalBox migration (catalog, act/step parity, determinism,
native cadences, NULL ticks, rejected motor events) and the legacy shim.
"""

from typing import Any, Dict, Optional

import numpy as np

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.program import ActionResult, Program, ProgramMetadata
from cognitive_runtime.core.reward import RewardSignal
from cognitive_runtime.core.streams import (
    LatestValueView,
    MotorStreamBus,
    ObservationStreamShim,
    SensoryStreamBus,
    TemporalBuffer,
    publish_motor_command,
    validate_stream_identity,
)
from cognitive_runtime.policies.scripted import ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.streams import BODY_HEARTBEAT_TICKS

FAST_CONFIG = {"episode_ticks": 300, "world_size": 32}


def _stream_program(seed):
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory, motor)
    program.reset(seed=seed)
    return program, sensory, motor


def _drive(seed, actions):
    """Drive the stream path; returns per-tick sensory event lists."""
    program, sensory, motor = _stream_program(seed)
    ticks = [sensory.drain()]  # the initial post-reset snapshot
    for action in actions:
        if action is not None:
            publish_motor_command(motor, action, timestamp=0.0)
        program.step()
        ticks.append(sensory.drain())
    return program, ticks


def _scripted_trace(seed, n_ticks):
    """Action sequence the scripted policy produces over stream-derived state."""
    program, sensory, motor = _stream_program(seed)
    policy = ScriptedSurvivalPolicy(seed=0)
    memory = Memory()
    buffer = TemporalBuffer()
    buffer.extend(sensory.drain())  # initial post-reset snapshot
    actions = []
    for _ in range(n_ticks):
        state = State(observation=LatestValueView(buffer).to_observation())
        action = policy.decide(state, memory, None)
        actions.append(action)
        memory.record_action(action)
        if not action.is_null:
            publish_motor_command(motor, action, timestamp=0.0)
        program.step()
        buffer.extend(sensory.drain())
    return actions


# ------------------------------------------------------------------ catalog


def test_stream_catalog_covers_the_survival_taxonomy():
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    specs = {spec.stream_id: spec for spec in program.stream_catalog()}
    expected = {
        "vision.frame.grid", "vision.frame.pixels", "vision.entities",
        "body.health", "body.hunger", "body.oxygen",
        "body.inventory", "body.inventory_exact", "body.hotbar",
        "body.in_water", "body.alive",
        "spatial.position", "spatial.rotation", "spatial.distance_from_spawn",
        "world.time", "world.biome", "world.nearby_blocks",
        "world.nearby_blocks_exact", "world.front_block",
        "world.front_block_exact", "world.sheltered",
        "event.damage_taken", "event.item_collected", "event.item_collected_exact",
        "event.block_broken", "event.block_broken_exact",
        "event.block_placed", "event.block_placed_exact",
        "event.crafted", "event.advancement", "event.dimension_changed",
        "event.biome_entered", "event.structure_discovered",
        "event.container_interaction",
        "event.created_light_source",
        "event.mob_killed", "event.bumped", "event.food_eaten",
        "event.entered_shelter", "event.survived_night", "event.died",
        "event.action_rejected",
        "reward.scalar",
    }
    assert set(specs) == expected
    for spec in specs.values():  # ids/modalities pass Phase-0 validation
        validate_stream_identity(spec.stream_id, spec.modality)


# ------------------------------------------------------- act/step parity


def test_step_path_matches_legacy_act_path():
    seed = 7
    actions = [Action("MOVE_FORWARD"), Action("LOOK_RIGHT"), Action("ATTACK")] * 30

    legacy = MinecraftSurvivalBox(config=FAST_CONFIG)
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


def _comparable_dict(event) -> Dict[str, Any]:
    """``to_dict()`` with an ndarray payload (a pixel frame) made list-based,
    so two ticks' dicts can be compared with plain ``==`` (an elementwise
    ndarray comparison isn't a bool)."""
    out = event.to_dict()
    if isinstance(out.get("payload"), np.ndarray):
        out["payload"] = out["payload"].tolist()
    return out


def test_same_seed_same_motor_events_give_identical_sensory_streams():
    actions = _scripted_trace(seed=11, n_ticks=120)

    def run():
        _, ticks = _drive(11, actions)
        return [
            [(e.stream_id, e.sequence_number, e.hash(), _comparable_dict(e)) for e in tick]
            for tick in ticks
        ]

    first, second = run(), run()
    assert first == second  # byte-identical ids, seq numbers, payloads, hashes


# --------------------------------------------------------------- cadence


def test_native_cadences_on_a_stationary_null_run():
    n_ticks = 100
    _, ticks = _drive(0, [None] * n_ticks)  # zero motor events every tick

    def count(stream_id):
        return sum(1 for tick in ticks for e in tick if e.stream_id == stream_id)

    # Initial snapshot publishes every state stream exactly once.
    initial = {e.stream_id for e in ticks[0]}
    assert {"vision.frame.grid", "body.health", "spatial.position",
            "world.time", "world.nearby_blocks"} <= initial

    # Stationary agent: spatial.* never republishes after the snapshot.
    assert count("spatial.position") == 1
    assert count("spatial.rotation") == 1
    # Daytime, dry, no damage: health/oxygen change never — heartbeat only.
    heartbeats = n_ticks // BODY_HEARTBEAT_TICKS
    assert count("body.health") == 1 + heartbeats
    assert count("body.oxygen") == 1 + heartbeats
    # Hunger drains but rounds to visible changes on only some ticks.
    assert 1 + heartbeats <= count("body.hunger") < 1 + n_ticks
    # Every-tick streams really are every tick.
    assert count("vision.frame.grid") == 1 + n_ticks
    assert count("world.time") == 1 + n_ticks
    assert count("reward.scalar") == n_ticks  # per tick, none in the snapshot


def test_exact_streams_publish_but_do_not_replace_compact_streams():
    program, sensory, _ = _stream_program(0)
    snapshot = {e.stream_id: e.payload for e in sensory.drain()}
    assert snapshot["world.front_block_exact"] == snapshot["world.front_block"]
    assert snapshot["world.nearby_blocks_exact"] == snapshot["world.nearby_blocks"]
    assert snapshot["body.inventory_exact"] == snapshot["body.inventory"]
    assert program is not None


def test_zero_motor_tick_advances_the_world():
    program, ticks = _drive(0, [None] * 5)
    assert program.episode_stats()["final_tick"] == 5
    times = [
        e.payload["time_of_day"]
        for tick in ticks for e in tick if e.stream_id == "world.time"
    ]
    assert times == list(range(6))  # time passes on NULL ticks
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
        {"action": "CRAFT"},                    # unknown action name
        {"action": "SELECT_HOTBAR_SLOT:slot=99"},  # invalid parameter
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
    publish_motor_command(motor, Action("LOOK_RIGHT"), timestamp=0.0)
    publish_motor_command(motor, Action("LOOK_LEFT"), timestamp=0.0)
    program.step()
    events = sensory.drain()
    rejected = [e for e in events if e.stream_id == "event.action_rejected"]
    assert len(rejected) == 1 and "superseded" in rejected[0].payload["reason"]
    rotation = [e for e in events if e.stream_id == "spatial.rotation"]
    assert rotation and rotation[0].payload["yaw"] == 15.0  # LOOK_RIGHT applied


def test_bumped_and_mob_killed_events_are_published_in_sim_sessions():
    program, sensory, motor = _stream_program(0)
    sensory.drain()
    world = program._backend.world
    world.z = int(world.z) + 0.9
    bx, bz = world._front_cell()
    world.terrain[bx][bz] = "stone"
    publish_motor_command(motor, Action("MOVE_FORWARD"), timestamp=0.0)
    program.step()
    bumped = [e for e in sensory.drain() if e.stream_id == "event.bumped"]
    assert bumped

    world.terrain[bx][bz] = "dirt"
    world.mobs = [{"id": 1, "x": world.x, "z": world.z + 1.0, "hp": 1, "cooldown": 0}]
    publish_motor_command(motor, Action("ATTACK"), timestamp=0.0)
    program.step()
    killed = [e for e in sensory.drain() if e.stream_id == "event.mob_killed"]
    assert killed


# --------------------------------------------------- richer event streams (#40)


def _by_stream(events, stream_id):
    return [e for e in events if e.stream_id == stream_id]


def test_exact_block_broken_and_item_collected_events():
    program, sensory, motor = _stream_program(0)
    sensory.drain()
    world = program._backend.world
    bx, bz = world._front_cell()
    world.terrain[bx][bz] = "tree"  # BREAK_YIELD -> log

    publish_motor_command(motor, Action("ATTACK"), timestamp=0.0)
    program.step()
    events = sensory.drain()

    broken = _by_stream(events, "event.block_broken_exact")
    assert broken and broken[0].payload == {
        "block": "tree", "position": {"x": bx, "y": 64.0, "z": bz}
    }
    collected = _by_stream(events, "event.item_collected_exact")
    assert collected and collected[0].payload == {"item": "log", "count": 1}
    advancement = _by_stream(events, "event.advancement")
    assert {"id": "sim.mine_wood"} in [e.payload for e in advancement]


def test_exact_block_placed_event():
    program, sensory, motor = _stream_program(0)
    sensory.drain()
    world = program._backend.world
    bx, bz = world._front_cell()
    world.terrain[bx][bz] = "grass"
    world.inventory["dirt"] = 1
    world.hotbar[0] = "dirt"
    world.selected_slot = 0

    publish_motor_command(motor, Action("USE"), timestamp=0.0)
    program.step()
    events = sensory.drain()

    placed = _by_stream(events, "event.block_placed_exact")
    assert placed and placed[0].payload == {
        "block": "dirt", "position": {"x": bx, "y": 64.0, "z": bz}
    }


def test_container_interaction_and_crafted_events_at_a_crafting_table():
    program, sensory, motor = _stream_program(0)
    sensory.drain()
    world = program._backend.world
    bx, bz = world._front_cell()
    world.terrain[bx][bz] = "crafting_table"
    world.inventory["log"] = 1

    publish_motor_command(motor, Action("USE"), timestamp=0.0)
    program.step()
    events = sensory.drain()

    container = _by_stream(events, "event.container_interaction")
    assert container and container[0].payload == {
        "container": "crafting_table", "position": {"x": bx, "y": 64.0, "z": bz}
    }
    crafted = _by_stream(events, "event.crafted")
    assert crafted and crafted[0].payload == {
        "recipe": "log_to_planks", "inputs": {"log": 1}, "outputs": {"planks": 4}
    }
    assert world.inventory.get("log", 0) == 0
    assert world.inventory.get("planks") == 4
    collected = [e.payload for e in _by_stream(events, "event.item_collected_exact")]
    assert {"item": "planks", "count": 4} in collected
    advancement = [e.payload for e in _by_stream(events, "event.advancement")]
    assert {"id": "sim.craft_item"} in advancement


def test_container_interaction_without_ingredients_does_not_craft():
    program, sensory, motor = _stream_program(0)
    sensory.drain()
    world = program._backend.world
    bx, bz = world._front_cell()
    world.terrain[bx][bz] = "chest"

    publish_motor_command(motor, Action("USE"), timestamp=0.0)
    program.step()
    events = sensory.drain()

    assert _by_stream(events, "event.container_interaction")
    assert not _by_stream(events, "event.crafted")


def test_furnace_smelting_produces_crafted_event():
    program, sensory, motor = _stream_program(0)
    sensory.drain()
    world = program._backend.world
    bx, bz = world._front_cell()
    world.terrain[bx][bz] = "furnace"
    world.inventory["cobblestone"] = 1
    world.inventory["coal"] = 1

    publish_motor_command(motor, Action("USE"), timestamp=0.0)
    program.step()
    events = sensory.drain()

    crafted = _by_stream(events, "event.crafted")
    assert crafted and crafted[0].payload == {
        "recipe": "smelt_cobblestone",
        "inputs": {"cobblestone": 1, "coal": 1},
        "outputs": {"stone": 1},
    }


def test_dimension_changed_event_when_crossing_a_portal():
    program, sensory, motor = _stream_program(0)
    sensory.drain()
    world = program._backend.world
    bx, bz = world._front_cell()
    world.terrain[bx][bz] = "portal"
    world.z = bz - 0.1  # one WALK_SPEED (0.25) step crosses straight into it

    publish_motor_command(motor, Action("MOVE_FORWARD"), timestamp=0.0)
    program.step()
    events = sensory.drain()

    changed = _by_stream(events, "event.dimension_changed")
    assert changed and changed[0].payload == {"from": "overworld", "to": "nether"}
    assert world.dimension == "nether"
    advancement = [e.payload for e in _by_stream(events, "event.advancement")]
    assert {"id": "sim.enter_portal"} in advancement


def test_structure_discovered_and_biome_entered_events():
    program, sensory, motor = _stream_program(0)
    sensory.drain()
    world = program._backend.world
    (sx, sz), name = next(iter(world.structures.items()))
    world.x, world.z = float(sx), float(sz)
    world._biome = None  # force a biome_entered event regardless of destination biome

    publish_motor_command(motor, Action("NULL"), timestamp=0.0)
    program.step()
    events = sensory.drain()

    discovered = _by_stream(events, "event.structure_discovered")
    assert discovered and discovered[0].payload == {"structure": name}
    assert _by_stream(events, "event.biome_entered")
    advancement = [e.payload for e in _by_stream(events, "event.advancement")]
    assert {"id": "sim.explore_structure"} in advancement

    # One-shot: entering the same structure cell again does not refire.
    sensory.drain()
    publish_motor_command(motor, Action("NULL"), timestamp=0.0)
    program.step()
    assert not _by_stream(sensory.drain(), "event.structure_discovered")


# ------------------------------------------------------------ legacy shim


class CounterProgram(Program):
    """Minimal legacy pull-style Program: a counter plus a constant."""

    def __init__(self):
        self._count = 0

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    def observe(self) -> Observation:
        return Observation(
            timestamp=self._count * 0.1,
            tick=self._count,
            data={"counter": self._count, "constant": 1},
            frame=[[self._count]],
        )

    def act(self, action: Action) -> ActionResult:
        if action.name not in ("NULL", "INCREMENT"):
            return ActionResult(ok=False, info={"error": f"unknown action {action.name}"})
        if action.name == "INCREMENT":
            self._count += 1
        return ActionResult(ok=True)

    def reward(self) -> RewardSignal:
        return RewardSignal.from_components({"count": float(self._count)})

    def is_complete(self) -> bool:
        return False

    def reset(self, seed: Optional[int] = None) -> None:
        self._count = 0

    def snapshot(self) -> str:
        return str(self._count)

    def restore(self, snapshot_id: str) -> None:
        self._count = int(snapshot_id)

    def metadata(self) -> ProgramMetadata:
        return ProgramMetadata(
            name="counter", version="0", observation_keys=["counter", "constant"]
        )


def test_observation_shim_publishes_legacy_programs_as_streams():
    shim = ObservationStreamShim(CounterProgram())
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    shim.attach_buses(sensory, motor)
    shim.reset()

    snapshot = {e.stream_id: e.payload for e in sensory.drain()}
    assert snapshot["observation.counter"] == 0
    assert snapshot["observation.constant"] == 1
    assert snapshot["vision.frame.grid"] == [[0]]

    publish_motor_command(motor, Action("INCREMENT"), timestamp=0.0)
    shim.step()
    tick = {e.stream_id: e.payload for e in sensory.drain()}
    assert tick["observation.counter"] == 1
    assert "observation.constant" not in tick  # unchanged => not republished
    assert tick["vision.frame.grid"] == [[1]]
    assert tick["reward.scalar"] == {"value": 1.0, "components": {"count": 1.0}}

    shim.step()  # zero motor events: NULL tick, only the reward republishes
    tick = {e.stream_id: e.payload for e in sensory.drain()}
    assert set(tick) == {"reward.scalar"}

    motor.publish("motor.command", {"action": "EXPLODE"}, timestamp=0.0)
    shim.step()
    tick = {e.stream_id: e.payload for e in sensory.drain()}
    assert "unknown action EXPLODE" in tick["event.action_rejected"]["reason"]


def test_latest_value_view_reconstructs_an_observation():
    shim = ObservationStreamShim(CounterProgram())
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    shim.attach_buses(sensory, motor)
    shim.reset()

    buffer = TemporalBuffer()
    buffer.extend(sensory.drain())
    for _ in range(3):
        publish_motor_command(motor, Action("INCREMENT"), timestamp=0.0)
        shim.step()
        buffer.extend(sensory.drain())

    observation = LatestValueView(buffer).to_observation(tick=3)
    assert observation.data == {"counter": 3, "constant": 1}
    assert observation.frame == [[3]]
    assert observation.tick == 3
    assert abs(observation.timestamp - 0.3) < 1e-9


# --------------------------------------------------- stream-native reward


def test_stream_reward_matches_legacy_reward_on_a_scripted_run():
    """The stream reward is a mechanical port: on a real trajectory it must
    produce the same per-tick values as the legacy observation-based path
    (whole-observation novelty aside, which doesn't trigger in 120 ticks)."""
    seed = 11
    actions = _scripted_trace(seed=seed, n_ticks=120)

    legacy = MinecraftSurvivalBox(config=FAST_CONFIG)
    legacy.reset(seed=seed)
    legacy_values = []
    for action in actions:
        legacy.act(action)
        legacy_values.append(legacy.reward().value)

    program, ticks = _drive(seed, actions)
    stream_values = [
        e.payload["value"]
        for tick in ticks[1:] for e in tick if e.stream_id == "reward.scalar"
    ]
    assert stream_values == legacy_values
