"""Build Observations from the SurvivalBox world state.

MVP observation: timestamp, screen frame (coarse top-down grid standing in
for pixels), health, hunger, oxygen, position, yaw/pitch, inventory summary,
selected hotbar slot, and nearby block metadata.
"""

from __future__ import annotations

import math

from cognitive_runtime.core.observation import Observation
from cognitive_runtime.programs.minecraft.world import SimulatedWorld

OBSERVATION_KEYS = [
    "health", "hunger", "oxygen", "position", "yaw", "pitch", "time_of_day",
    "day_length", "is_night", "biome", "in_water", "sheltered", "selected_slot",
    "hotbar", "inventory", "inventory_exact", "nearby_blocks",
    "nearby_blocks_exact", "front_block", "front_block_exact", "mobs",
    "distance_from_spawn", "dead",
]


def build_observation(world: SimulatedWorld, timestamp: float) -> Observation:
    ix, iz = int(world.x), int(world.z)
    bx, bz = world._front_cell()
    bx = min(max(bx, 0), world.size - 1)
    bz = min(max(bz, 0), world.size - 1)
    data = {
        "health": round(world.health, 2),
        "hunger": round(world.hunger, 2),
        "oxygen": round(world.oxygen, 2),
        "position": {"x": round(world.x, 3), "y": 64.0, "z": round(world.z, 3)},
        "yaw": round(world.yaw, 1),
        "pitch": round(world.pitch, 1),
        "time_of_day": world.time_of_day,
        "day_length": world.cfg.day_length,
        "is_night": world.is_night,
        "biome": world.biome_map[ix][iz],
        "in_water": world.in_water,
        "sheltered": world.is_sheltered(),
        "selected_slot": world.selected_slot,
        "hotbar": list(world.hotbar),
        "inventory": dict(sorted(world.inventory.items())),
        "inventory_exact": dict(sorted(world.inventory.items())),
        "nearby_blocks": world.nearby_blocks(radius=2),
        "nearby_blocks_exact": world.nearby_blocks(radius=2),
        "front_block": world.terrain[bx][bz],
        "front_block_exact": world.terrain[bx][bz],
        "mobs": world.mob_summary(),
        "distance_from_spawn": round(math.dist((world.x, world.z), world.spawn), 2),
        "dead": world.dead,
    }
    return Observation(
        timestamp=round(timestamp, 3),
        tick=world.tick,
        data=data,
        frame=world.render_frame(radius=5),
        pixels=world.render_pixels(),
    )
