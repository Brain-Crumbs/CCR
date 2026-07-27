"""SurvivalBox action space.

Issue #42 grows the MVP's movement/look/attack/use set with crafting,
inventory management, equip, targeted placement/use, and generic
interaction -- the minimum quest-level verb set the tier 2/3 reward goals
(#41) need.  Parameterized actions keep the ``Action.make(name, **params)``
key convention so recording/replay/BC datasets need no format change; a
whole new base action is added per distinct verb, and per-parameter variants
(hotbar slot, recipe id) are enumerated the same way ``SELECT_HOTBAR_SLOT``
already was.

``pathing/navigation primitives`` (the issue's one explicitly out-of-scope
item) are not included here.
"""

from __future__ import annotations

from typing import List

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.programs.minecraft.world import RECIPE_NAMES

HOTBAR_SLOTS = 9

BASE_ACTIONS: List[Action] = [
    NULL_ACTION,
    Action("MOVE_FORWARD"),
    Action("MOVE_BACKWARD"),
    Action("MOVE_LEFT"),
    Action("MOVE_RIGHT"),
    Action("JUMP"),
    Action("SNEAK"),
    Action("SPRINT"),
    Action("LOOK_LEFT"),
    Action("LOOK_RIGHT"),
    Action("LOOK_UP"),
    Action("LOOK_DOWN"),
    Action("ATTACK"),
    Action("USE"),
    #: Generic block/entity interaction (doors, containers, furnace,
    #: villagers): a container in the simulated world, richer live-server
    #: mechanics via the mineflayer bridge. Distinct from USE, the compact
    #: eat/place/open+auto-craft action used by the simulated world.
    Action("INTERACT"),
    Action("OPEN_INVENTORY"),
    Action("CLOSE_INVENTORY"),
]

SELECT_ACTIONS: List[Action] = [
    Action.make("SELECT_HOTBAR_SLOT", slot=i) for i in range(HOTBAR_SLOTS)
]

#: Equip the item in a hotbar slot -- semantically distinct from
#: SELECT_HOTBAR_SLOT (pure navigation, tolerant of empty slots): equipping
#: an empty slot is rejected, giving the agent explicit negative feedback.
EQUIP_ACTIONS: List[Action] = [
    Action.make("EQUIP_ITEM", slot=i) for i in range(HOTBAR_SLOTS)
]

#: Place the block held in a specific hotbar slot, independent of what is
#: currently selected.
PLACE_BLOCK_ACTIONS: List[Action] = [
    Action.make("PLACE_BLOCK", slot=i) for i in range(HOTBAR_SLOTS)
]

#: Use (consume) the item held in a specific hotbar slot, independent of
#: what is currently selected.
USE_ITEM_ACTIONS: List[Action] = [
    Action.make("USE_ITEM", slot=i) for i in range(HOTBAR_SLOTS)
]

#: Swap the contents of two hotbar slots.  Swap is symmetric, so only
#: ``from_slot < to_slot`` pairs are enumerated (a=b,b=a is the same action
#: as b=a,a=b); the world applies it as an unordered swap.
MOVE_INVENTORY_ITEM_ACTIONS: List[Action] = [
    Action.make("MOVE_INVENTORY_ITEM", from_slot=i, to_slot=j)
    for i in range(HOTBAR_SLOTS)
    for j in range(i + 1, HOTBAR_SLOTS)
]

#: Craft a specific recipe by id (see ``world.RECIPES``): rejected (not just
#: silently skipped) when the agent isn't at the matching container or lacks
#: materials, unlike the compact "USE a container tries every recipe in
#: order" behaviour.
CRAFT_ACTIONS: List[Action] = [
    Action.make("CRAFT", recipe=name) for name in RECIPE_NAMES
]

ACTION_SPACE: List[Action] = (
    BASE_ACTIONS
    + SELECT_ACTIONS
    + EQUIP_ACTIONS
    + PLACE_BLOCK_ACTIONS
    + USE_ITEM_ACTIONS
    + MOVE_INVENTORY_ITEM_ACTIONS
    + CRAFT_ACTIONS
)
