"""SurvivalBox action space (MVP: kept deliberately small).

Later actions (CRAFT, DROP_ITEM, OPEN_INVENTORY, PLACE_BLOCK as a distinct
verb, EQUIP_ITEM, TYPE_COMMAND) are out of scope for the MVP.
"""

from __future__ import annotations

from typing import List

from cognitive_runtime.core.action import NULL_ACTION, Action

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
]

SELECT_ACTIONS: List[Action] = [
    Action.make("SELECT_HOTBAR_SLOT", slot=i) for i in range(HOTBAR_SLOTS)
]

ACTION_SPACE: List[Action] = BASE_ACTIONS + SELECT_ACTIONS
