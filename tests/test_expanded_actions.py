"""Issue #42: crafting, inventory, equip, targeted placement/use, interact.

Each new action is exercised through the streams path (the runtime's actual
motor contract) so both the accept case and the ``event.action_rejected``
feedback path are covered, per the issue's acceptance criteria.
"""

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.streams import MotorStreamBus, SensoryStreamBus, publish_motor_command
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.world import RECIPE_NAMES

FAST_CONFIG = {"episode_ticks": 300, "world_size": 32}


def _stream_program(seed=0):
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory, motor)
    program.reset(seed=seed)
    return program, sensory, motor


def _act(program, sensory, motor, action):
    """Publish one motor command, step, and return this tick's events."""
    publish_motor_command(motor, action, timestamp=0.0)
    program.step()
    return sensory.drain()


def _by_stream(events, stream_id):
    return [e for e in events if e.stream_id == stream_id]


def _rejections(events):
    return [e.payload["reason"] for e in _by_stream(events, "event.action_rejected")]


# ------------------------------------------------------------------- INTERACT


def test_interact_with_container_emits_container_interaction_but_no_auto_craft():
    program, sensory, motor = _stream_program()
    sensory.drain()
    world = program._backend.world
    bx, bz = world._front_cell()
    world.terrain[bx][bz] = "crafting_table"
    world.inventory["log"] = 1  # would satisfy log_to_planks if USE's auto-craft ran

    events = _act(program, sensory, motor, Action("INTERACT"))

    assert _by_stream(events, "event.container_interaction")
    assert not _by_stream(events, "event.crafted")  # INTERACT never auto-crafts
    assert world.inventory["log"] == 1  # untouched


def test_interact_with_nothing_in_front_is_rejected():
    program, sensory, motor = _stream_program()
    sensory.drain()
    events = _act(program, sensory, motor, Action("INTERACT"))
    assert "nothing to interact with" in _rejections(events)[0]


# ------------------------------------------------------------ OPEN/CLOSE inventory


def test_open_close_inventory_toggles_body_stream():
    program, sensory, motor = _stream_program()
    sensory.drain()

    events = _act(program, sensory, motor, Action("OPEN_INVENTORY"))
    opened = _by_stream(events, "body.inventory_open")
    assert opened and opened[0].payload is True
    assert not _rejections(events)

    events = _act(program, sensory, motor, Action("CLOSE_INVENTORY"))
    closed = _by_stream(events, "body.inventory_open")
    assert closed and closed[0].payload is False
    assert not _rejections(events)


def test_open_inventory_twice_is_rejected():
    program, sensory, motor = _stream_program()
    sensory.drain()
    _act(program, sensory, motor, Action("OPEN_INVENTORY"))
    events = _act(program, sensory, motor, Action("OPEN_INVENTORY"))
    assert "already open" in _rejections(events)[0]


def test_close_inventory_when_already_closed_is_rejected():
    program, sensory, motor = _stream_program()
    sensory.drain()
    events = _act(program, sensory, motor, Action("CLOSE_INVENTORY"))
    assert "already closed" in _rejections(events)[0]


# ---------------------------------------------------------------- EQUIP_ITEM


def test_equip_item_selects_the_slot():
    program, sensory, motor = _stream_program()
    sensory.drain()
    world = program._backend.world
    world.inventory["log"] = 1
    world.hotbar[3] = "log"

    events = _act(program, sensory, motor, Action.make("EQUIP_ITEM", slot=3))
    assert not _rejections(events)
    assert world.selected_slot == 3
    hotbar = _by_stream(events, "body.hotbar")
    assert hotbar and hotbar[0].payload["selected"] == 3


def test_equip_empty_slot_is_rejected():
    program, sensory, motor = _stream_program()
    sensory.drain()
    events = _act(program, sensory, motor, Action.make("EQUIP_ITEM", slot=0))
    assert "empty slot" in _rejections(events)[0]
    assert program._backend.world.selected_slot == 0  # unchanged


# --------------------------------------------------------------- PLACE_BLOCK


def test_place_block_from_a_specific_slot():
    program, sensory, motor = _stream_program()
    sensory.drain()
    world = program._backend.world
    bx, bz = world._front_cell()
    world.terrain[bx][bz] = "grass"
    world.inventory["dirt"] = 1
    world.hotbar[5] = "dirt"
    world.selected_slot = 0  # deliberately NOT the slot being placed from

    events = _act(program, sensory, motor, Action.make("PLACE_BLOCK", slot=5))
    assert not _rejections(events)
    assert world.terrain[bx][bz] == "placed_block"
    placed = _by_stream(events, "event.block_placed_exact")
    assert placed and placed[0].payload["block"] == "dirt"


def test_place_block_rejects_a_non_placeable_item():
    program, sensory, motor = _stream_program()
    sensory.drain()
    world = program._backend.world
    world.inventory["wooden_pickaxe"] = 1
    world.hotbar[0] = "wooden_pickaxe"

    events = _act(program, sensory, motor, Action.make("PLACE_BLOCK", slot=0))
    assert "not placeable" in _rejections(events)[0]


def test_place_block_rejects_an_empty_slot():
    program, sensory, motor = _stream_program()
    sensory.drain()
    events = _act(program, sensory, motor, Action.make("PLACE_BLOCK", slot=0))
    assert "no item to place" in _rejections(events)[0]


# ----------------------------------------------------------------- USE_ITEM


def test_use_item_eats_food_from_a_specific_slot():
    program, sensory, motor = _stream_program()
    sensory.drain()
    world = program._backend.world
    world.hunger = 10.0
    world.inventory["berries"] = 1
    world.hotbar[4] = "berries"
    world.selected_slot = 0  # deliberately NOT the slot being used

    events = _act(program, sensory, motor, Action.make("USE_ITEM", slot=4))
    assert not _rejections(events)
    assert _by_stream(events, "event.food_eaten")
    assert world.inventory.get("berries", 0) == 0


def test_use_item_rejects_a_non_usable_item():
    program, sensory, motor = _stream_program()
    sensory.drain()
    world = program._backend.world
    world.inventory["stone"] = 1
    world.hotbar[0] = "stone"

    events = _act(program, sensory, motor, Action.make("USE_ITEM", slot=0))
    assert "not usable" in _rejections(events)[0]


def test_use_item_rejects_an_empty_slot():
    program, sensory, motor = _stream_program()
    sensory.drain()
    events = _act(program, sensory, motor, Action.make("USE_ITEM", slot=0))
    assert "no item to use" in _rejections(events)[0]


# --------------------------------------------------------- MOVE_INVENTORY_ITEM


def test_move_inventory_item_swaps_two_slots():
    program, sensory, motor = _stream_program()
    sensory.drain()
    world = program._backend.world
    world.inventory["log"] = 1
    world.inventory["stone"] = 1
    world.hotbar[0] = "log"
    world.hotbar[1] = "stone"

    events = _act(program, sensory, motor, Action.make("MOVE_INVENTORY_ITEM", from_slot=0, to_slot=1))
    assert not _rejections(events)
    assert world.hotbar[0] == "stone"
    assert world.hotbar[1] == "log"


def test_move_inventory_item_rejects_two_empty_slots():
    program, sensory, motor = _stream_program()
    sensory.drain()
    events = _act(program, sensory, motor, Action.make("MOVE_INVENTORY_ITEM", from_slot=0, to_slot=1))
    assert "both slots" in _rejections(events)[0]


def test_move_inventory_item_same_slot_is_rejected_by_the_adapter():
    """from_slot == to_slot fails shape validation (adapter-level), unlike
    the world-level rejections above -- act() itself reports not-ok."""
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    program.reset(seed=0)
    result = program.act(Action.make("MOVE_INVENTORY_ITEM", from_slot=2, to_slot=2))
    assert not result.ok


# ------------------------------------------------------------------------ CRAFT


def test_craft_recipe_names_cover_every_recipe_in_the_world():
    from cognitive_runtime.programs.minecraft.actions import CRAFT_ACTIONS

    assert {a.param("recipe") for a in CRAFT_ACTIONS} == set(RECIPE_NAMES)


def test_craft_log_to_planks_at_a_crafting_table():
    program, sensory, motor = _stream_program()
    sensory.drain()
    world = program._backend.world
    bx, bz = world._front_cell()
    world.terrain[bx][bz] = "crafting_table"
    world.inventory["log"] = 1

    events = _act(program, sensory, motor, Action.make("CRAFT", recipe="log_to_planks"))
    assert not _rejections(events)
    crafted = _by_stream(events, "event.crafted")
    assert crafted and crafted[0].payload["recipe"] == "log_to_planks"
    assert world.inventory.get("planks") == 4
    assert world.inventory.get("log", 0) == 0


def test_craft_rejects_when_materials_are_missing():
    program, sensory, motor = _stream_program()
    sensory.drain()
    world = program._backend.world
    bx, bz = world._front_cell()
    world.terrain[bx][bz] = "crafting_table"
    # No log in inventory.

    events = _act(program, sensory, motor, Action.make("CRAFT", recipe="log_to_planks"))
    assert "insufficient materials" in _rejections(events)[0]
    assert not _by_stream(events, "event.crafted")


def test_craft_rejects_the_wrong_container():
    program, sensory, motor = _stream_program()
    sensory.drain()
    world = program._backend.world
    bx, bz = world._front_cell()
    world.terrain[bx][bz] = "furnace"  # log_to_planks needs a crafting_table
    world.inventory["log"] = 1

    events = _act(program, sensory, motor, Action.make("CRAFT", recipe="log_to_planks"))
    assert "needs a crafting_table" in _rejections(events)[0]
    assert not _by_stream(events, "event.crafted")


def test_craft_unknown_recipe_is_rejected_by_the_adapter():
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    program.reset(seed=0)
    result = program.act(Action.make("CRAFT", recipe="diamond_sword"))
    assert not result.ok
