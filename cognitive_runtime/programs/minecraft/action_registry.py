"""Minecraft SurvivalBox action classification (issue #60): every action
name in `programs.minecraft.actions.ACTION_SPACE` classified world_changing
vs information_gathering. `tests/test_action_registry.py` enforces this is
complete (the issue #32 completeness-test pattern, applied to actions).
"""

from __future__ import annotations

from cognitive_runtime.core.action_registry import ActionDeclaration, ActionRegistry

MINECRAFT_ACTION_REGISTRY = ActionRegistry(
    [
        ActionDeclaration(
            "NULL", world_changing=False, information_gathering=True,
            note="Waiting: how the agent lets more information arrive without acting.",
        ),
        ActionDeclaration(
            "MOVE_FORWARD", world_changing=True, information_gathering=True,
            note="Repositioning changes agent/world state and exposes a new view (issue #60's own example).",
        ),
        ActionDeclaration(
            "MOVE_BACKWARD", world_changing=True, information_gathering=True,
            note="Repositioning changes agent/world state and exposes a new view.",
        ),
        ActionDeclaration(
            "MOVE_LEFT", world_changing=True, information_gathering=True,
            note="Repositioning changes agent/world state and exposes a new view.",
        ),
        ActionDeclaration(
            "MOVE_RIGHT", world_changing=True, information_gathering=True,
            note="Repositioning changes agent/world state and exposes a new view.",
        ),
        ActionDeclaration(
            "JUMP", world_changing=True, information_gathering=True,
            note="Vertical repositioning; briefly changes the view too.",
        ),
        ActionDeclaration(
            "SNEAK", world_changing=True, information_gathering=False,
            note="Stance toggle: changes collision/mob-detection behavior, not the view.",
        ),
        ActionDeclaration(
            "SPRINT", world_changing=True, information_gathering=False,
            note="Speed toggle: changes movement/hunger drain, not the view.",
        ),
        ActionDeclaration(
            "LOOK_LEFT", world_changing=False, information_gathering=True,
            note="Pure camera turn -- the orienting reflex's own action (issue #60).",
        ),
        ActionDeclaration(
            "LOOK_RIGHT", world_changing=False, information_gathering=True,
            note="Pure camera turn -- the orienting reflex's own action (issue #60).",
        ),
        ActionDeclaration(
            "LOOK_UP", world_changing=False, information_gathering=True,
            note="Pure camera turn.",
        ),
        ActionDeclaration(
            "LOOK_DOWN", world_changing=False, information_gathering=True,
            note="Pure camera turn.",
        ),
        ActionDeclaration(
            "ATTACK", world_changing=True, information_gathering=False,
            note="Damages a block/entity -- the issue's own 'mining changes the world' example.",
        ),
        ActionDeclaration(
            "USE", world_changing=True, information_gathering=False,
            note="Eats/places/opens+auto-crafts: always changes world or inventory state.",
        ),
        ActionDeclaration(
            "INTERACT", world_changing=True, information_gathering=True,
            note="Generic interaction (doors, containers, villagers): changes state and can reveal it.",
        ),
        ActionDeclaration(
            "OPEN_INVENTORY", world_changing=False, information_gathering=True,
            note="Reveals the agent's own inventory UI; no world effect.",
        ),
        ActionDeclaration(
            "CLOSE_INVENTORY", world_changing=False, information_gathering=True,
            note="Hides the inventory UI; no world effect.",
        ),
        ActionDeclaration(
            "SELECT_HOTBAR_SLOT", world_changing=True, information_gathering=False,
            note="Changes which item is active; agent-state change, not perceptual.",
        ),
        ActionDeclaration(
            "EQUIP_ITEM", world_changing=True, information_gathering=False,
            note="Changes equipped item; agent-state change, rejected (not silently skipped) if empty.",
        ),
        ActionDeclaration(
            "PLACE_BLOCK", world_changing=True, information_gathering=False,
            note="Adds a block to the world.",
        ),
        ActionDeclaration(
            "USE_ITEM", world_changing=True, information_gathering=False,
            note="Consumes a held item.",
        ),
        ActionDeclaration(
            "MOVE_INVENTORY_ITEM", world_changing=True, information_gathering=False,
            note="Swaps two hotbar slots; agent-state change.",
        ),
        ActionDeclaration(
            "CRAFT", world_changing=True, information_gathering=False,
            note="Consumes materials, produces an item.",
        ),
    ]
)
