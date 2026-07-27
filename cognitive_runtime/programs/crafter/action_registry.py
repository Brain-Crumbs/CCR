"""Crafter action classification (issue #89): every action name in
``programs.crafter.actions.ACTION_SPACE`` classified world_changing vs
information_gathering, mirroring ``programs.minecraft.action_registry``
(issue #60's completeness pattern applied to Crafter's action space).
"""

from __future__ import annotations

from cognitive_runtime.core.action_registry import ActionDeclaration, ActionRegistry

CRAFTER_ACTION_REGISTRY = ActionRegistry(
    [
        ActionDeclaration(
            "NULL", world_changing=False, information_gathering=True,
            note="Crafter's noop: waiting. Time still passes (vitals drain), but "
                 "nothing the agent chose changes.",
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
            "MOVE_UP", world_changing=True, information_gathering=True,
            note="Repositioning changes agent/world state and exposes a new view.",
        ),
        ActionDeclaration(
            "MOVE_DOWN", world_changing=True, information_gathering=True,
            note="Repositioning changes agent/world state and exposes a new view.",
        ),
        ActionDeclaration(
            "DO", world_changing=True, information_gathering=False,
            note="Crafter's overloaded context action (chop/mine/attack/drink/collect "
                 "depending on what's faced): always changes world or inventory state, "
                 "the same 'ATTACK+USE' shape as Minecraft's own overloaded verbs.",
        ),
        ActionDeclaration(
            "SLEEP", world_changing=True, information_gathering=False,
            note="Stance toggle: regenerates energy and changes night-time vulnerability, "
                 "not the view.",
        ),
        ActionDeclaration(
            "PLACE_STONE", world_changing=True, information_gathering=False,
            note="Adds a block to the world.",
        ),
        ActionDeclaration(
            "PLACE_TABLE", world_changing=True, information_gathering=False,
            note="Adds a block to the world.",
        ),
        ActionDeclaration(
            "PLACE_FURNACE", world_changing=True, information_gathering=False,
            note="Adds a block to the world.",
        ),
        ActionDeclaration(
            "PLACE_PLANT", world_changing=True, information_gathering=False,
            note="Adds a sapling to the world.",
        ),
        ActionDeclaration(
            "MAKE_WOOD_PICKAXE", world_changing=True, information_gathering=False,
            note="Consumes materials, produces a tool.",
        ),
        ActionDeclaration(
            "MAKE_STONE_PICKAXE", world_changing=True, information_gathering=False,
            note="Consumes materials, produces a tool.",
        ),
        ActionDeclaration(
            "MAKE_IRON_PICKAXE", world_changing=True, information_gathering=False,
            note="Consumes materials, produces a tool.",
        ),
        ActionDeclaration(
            "MAKE_WOOD_SWORD", world_changing=True, information_gathering=False,
            note="Consumes materials, produces a tool.",
        ),
        ActionDeclaration(
            "MAKE_STONE_SWORD", world_changing=True, information_gathering=False,
            note="Consumes materials, produces a tool.",
        ),
        ActionDeclaration(
            "MAKE_IRON_SWORD", world_changing=True, information_gathering=False,
            note="Consumes materials, produces a tool.",
        ),
    ]
)
