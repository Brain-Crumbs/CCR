"""Crafter nursery-world action space.

Crafter exposes 17 discrete actions (a no-op plus move/interact/build/craft
verbs).  ``noop`` folds into the universal ``NULL_ACTION`` rather than a
Crafter-specific name -- waiting is how the agent lets more information
arrive without acting, the same convention every Program shares (see
``core/action_registry.py``).

The (name, crafter action index) pairs are hardcoded rather than read off
``crafter.constants.actions`` at import time, so this module -- and
anything that only needs the static action space (the action registry, CLI
wiring) -- stays importable without the optional ``crafter`` package
installed.  ``tests/test_crafter_world.py`` checks this table against the
live package's own ``action_names`` so the two can't silently drift.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from cognitive_runtime.core.action import NULL_ACTION, Action

#: (Action name, crafter's own integer action id), in crafter's
#: ``action_names`` order.
CRAFTER_ACTIONS: Tuple[Tuple[str, int], ...] = (
    ("NULL", 0),                 # crafter's "noop"
    ("MOVE_LEFT", 1),
    ("MOVE_RIGHT", 2),
    ("MOVE_UP", 3),
    ("MOVE_DOWN", 4),
    ("DO", 5),                   # context action: chop/mine/attack/drink/collect
    ("SLEEP", 6),
    ("PLACE_STONE", 7),
    ("PLACE_TABLE", 8),
    ("PLACE_FURNACE", 9),
    ("PLACE_PLANT", 10),
    ("MAKE_WOOD_PICKAXE", 11),
    ("MAKE_STONE_PICKAXE", 12),
    ("MAKE_IRON_PICKAXE", 13),
    ("MAKE_WOOD_SWORD", 14),
    ("MAKE_STONE_SWORD", 15),
    ("MAKE_IRON_SWORD", 16),
)

ACTION_SPACE: List[Action] = [
    NULL_ACTION if name == "NULL" else Action(name) for name, _ in CRAFTER_ACTIONS
]

#: Action name -> crafter's integer action id, for encoding a motor command
#: as the index ``env.step()`` expects.
ACTION_NAME_TO_INDEX: Dict[str, int] = dict(CRAFTER_ACTIONS)
