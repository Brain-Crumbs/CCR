"""CrafterWorld observation assembly.

The single source of state both the legacy pull-style ``observe()`` and the
stream publisher build from, so the two paths stay byte-identical (mirrors
``programs.minecraft.observations``).  Reaches into ``crafter.Env``'s
private ``_player``/``_sem_view`` -- the package exposes vitals/inventory
only through ``step()``'s ``info`` dict, which doesn't exist right after
``reset()``; the private attributes are the one state source available in
both places.
"""

from __future__ import annotations

from typing import Any, Dict

from cognitive_runtime.core.observation import Observation
from cognitive_runtime.programs.crafter.streams import crop_semantic_grid

#: Inventory keys that aren't a vital -- issue #89's body.inventory
#: summarizes resources/tools; the four vitals get their own dedicated
#: streams (mirrors body.health/hunger/oxygen vs. body.inventory in
#: programs.minecraft.streams).
_INVENTORY_KEYS = (
    "sapling", "wood", "stone", "coal", "iron", "diamond",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
)

OBSERVATION_KEYS = (
    "health", "food", "drink", "energy", "inventory", "sleeping", "alive",
    "position", "facing", "achievements",
)


def build_state(env: Any, grid_radius: int) -> Dict[str, Any]:
    """Read-only snapshot of the current Crafter env state -- never advances
    the world (mirrors ``SurvivalBackend.observe``'s pull semantics)."""
    player = env._player
    position = (int(player.pos[0]), int(player.pos[1]))
    facing = (int(player.facing[0]), int(player.facing[1]))
    return {
        "health": float(player.health),
        "food": float(player.inventory["food"]),
        "drink": float(player.inventory["drink"]),
        "energy": float(player.inventory["energy"]),
        "inventory": {k: int(player.inventory[k]) for k in _INVENTORY_KEYS},
        "sleeping": bool(player.sleeping),
        "alive": player.health > 0,
        "position": {"x": position[0], "y": position[1]},
        # Crafter's facing is a discrete grid direction, flipped on every
        # directional move *attempt* -- even a blocked one
        # (``crafter.objects.Player._move`` sets ``self.facing`` before
        # checking collision) -- so a boxed-in agent can still turn without
        # displacing (issue #90's discrete-facing ``turn`` scenario).
        "facing": {"x": facing[0], "y": facing[1]},
        "achievements": dict(player.achievements),
        "grid": crop_semantic_grid(env._sem_view(), position, grid_radius),
    }


def build_observation(
    state: Dict[str, Any], pixels: Any, timestamp: float, tick: int
) -> Observation:
    data = {k: v for k, v in state.items() if k != "grid"}
    return Observation(timestamp=timestamp, tick=tick, data=data, frame=state["grid"], pixels=pixels)
