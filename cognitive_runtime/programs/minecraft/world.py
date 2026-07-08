"""Deterministic simulated survival world.

This is the MVP backend for MinecraftSurvivalBox.  It is not Minecraft; it
is a seeded, fully deterministic survival sandbox with the same shape of
experience -- terrain, biomes, a day/night cycle, hunger/health/oxygen,
hostile mobs at night, block breaking and placing, and an inventory --
so the whole runtime stack (loop, rewards, recording, replay, training)
can be exercised end to end without a Minecraft server.  A real-Minecraft
backend can replace it behind the same adapter (see adapter.py).

Determinism contract: given the same seed and the same action sequence,
the world produces byte-identical observations.  Replay depends on this.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from cognitive_runtime.core.action import Action
from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig

# Ground (passable) and feature (solid) cell types.
PASSABLE = {"grass", "dirt", "sand", "water"}
SOLID = {"tree", "stone", "coal_ore", "berry_bush", "placed_block", "barrier"}

BREAK_YIELD = {
    "tree": "log",
    "stone": "cobblestone",
    "coal_ore": "coal",
    "berry_bush": "berries",
    "placed_block": "dirt",
}
FOOD_ITEMS = {"berries": 3.0}  # hunger restored per unit
PLACEABLE_ITEMS = {"log", "cobblestone", "dirt", "sand"}

BLOCK_IDS = {
    "grass": 1, "dirt": 2, "sand": 3, "water": 4, "tree": 5,
    "stone": 6, "coal_ore": 7, "berry_bush": 8, "placed_block": 9, "barrier": 10,
}
MOB_FRAME_ID = 90
AGENT_FRAME_ID = 99

#: Deterministic RGB palette for the pixel render -- a stand-in for "what the
#: player sees".  Pure function of world state, so ``render_pixels`` stays
#: replay-safe (a real backend would swap in a rendered/captured frame here).
BLOCK_COLORS = {
    "grass": (86, 168, 74),
    "dirt": (134, 96, 67),
    "sand": (219, 209, 145),
    "water": (54, 108, 209),
    "tree": (37, 92, 38),
    "stone": (128, 128, 128),
    "coal_ore": (54, 54, 60),
    "berry_bush": (150, 40, 70),
    "placed_block": (170, 140, 100),
    "barrier": (20, 20, 20),
}
AGENT_COLOR = (240, 220, 40)
MOB_COLOR = (200, 40, 40)

#: Pixel-render geometry: a (2*radius+1) cell local view, each cell upscaled to
#: ``scale``x``scale`` pixels -> an 11*3 = 33 px square RGB image at the default.
PIXEL_RADIUS = 5
PIXEL_SCALE = 3

#: Frame-code -> RGB, so a semantic grid frame (the same one every backend
#: emits, including the mineflayer bridge) colorizes to pixels the same way.
FRAME_CODE_COLORS: Dict[int, Tuple[int, int, int]] = {
    **{BLOCK_IDS[name]: rgb for name, rgb in BLOCK_COLORS.items()},
    MOB_FRAME_ID: MOB_COLOR,
    AGENT_FRAME_ID: AGENT_COLOR,
}
_UNKNOWN_CELL_COLOR = (255, 0, 255)  # magenta: an out-of-vocab frame code


def pixels_from_frame(
    frame: List[List[int]], scale: int = PIXEL_SCALE
) -> List[List[List[int]]]:
    """Colorize a semantic grid frame (frame codes) into an H*scale x W*scale x 3
    RGB image.  Backend-agnostic: any world that emits the shared grid frame
    gets identical pixel vision, and the mapping is a pure function of the grid
    (so it stays deterministic / replay-safe)."""
    image: List[List[List[int]]] = []
    for grid_row in frame:
        pixel_row = [
            list(FRAME_CODE_COLORS.get(code, _UNKNOWN_CELL_COLOR))
            for code in grid_row
            for _ in range(scale)
        ]
        for _ in range(scale):
            image.append([list(px) for px in pixel_row])
    return image

WALK_SPEED = 0.25
SPRINT_SPEED = 0.45
SNEAK_SPEED = 0.10
LOOK_STEP_DEG = 15.0
PITCH_STEP_DEG = 10.0

HUNGER_PER_TICK = 0.004
SPRINT_HUNGER_COST = 0.01
JUMP_HUNGER_COST = 0.05
REGEN_INTERVAL = 80      # ticks between regen points at high hunger
STARVE_INTERVAL = 80     # ticks between starvation damage at hunger 0
DROWN_INTERVAL = 20      # ticks between drowning damage at oxygen 0

ZOMBIE_HP = 3
ZOMBIE_SPEED = 0.18
ZOMBIE_REACH = 1.2
ZOMBIE_ATTACK_COOLDOWN = 20
ZOMBIE_BASE_DAMAGE = 2.0
ZOMBIE_SPAWN_RATE = 0.008  # per tick, scaled by difficulty
ATTACK_REACH = 2.0
ATTACK_CONE_DEG = 60.0


class SimulatedWorld:
    def __init__(self, config: SurvivalBoxConfig, seed: int = 0):
        self.cfg = config
        self.reset(seed)

    # ------------------------------------------------------------------ setup

    def reset(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.size = self.cfg.world_size
        self.terrain: List[List[str]] = []
        self.biome_map: List[List[str]] = []
        self._generate_terrain()

        self.tick = 0
        spawn = self._find_spawn()
        self.x = spawn[0] + 0.5
        self.z = spawn[1] + 0.5
        self.spawn = (self.x, self.z)
        self.yaw = 0.0
        self.pitch = 0.0
        self.health = 20.0
        self.hunger = 20.0
        self.oxygen = 20.0
        self.inventory: Dict[str, int] = {}
        self.hotbar: List[Optional[str]] = [None] * 9
        self.selected_slot = 0
        self.mobs: List[Dict[str, Any]] = []
        self._mob_serial = 0
        self.dead = False
        self.death_reason: Optional[str] = None
        self._regen_counter = 0
        self._starve_counter = 0
        self._drown_counter = 0
        self._was_night = False
        self._sheltered = False
        # Episode statistics.
        self.stats: Dict[str, Any] = {
            "damage_taken": 0.0,
            "food_consumed": 0,
            "blocks_broken": 0,
            "blocks_placed": 0,
            "mobs_killed": 0,
            "max_distance_from_spawn": 0.0,
            "unique_items_collected": 0,
            "survived_night": False,
        }
        self._snapshots: Dict[str, Any] = {}

    def _generate_terrain(self) -> None:
        size = self.size
        half = size // 2
        self.terrain = [["grass"] * size for _ in range(size)]
        self.biome_map = [["plains"] * size for _ in range(size)]
        for x in range(size):
            for z in range(size):
                if x >= half and z < half:
                    biome = "forest"
                elif x < half and z >= half:
                    biome = "desert"
                elif x >= half and z >= half:
                    biome = "lake"
                else:
                    biome = "plains"
                self.biome_map[x][z] = biome
                if biome == "desert":
                    self.terrain[x][z] = "sand"

        # Lake: water ellipse in the lake quadrant.
        cx, cz = half + half // 2, half + half // 2
        radius = half // 3
        for x in range(size):
            for z in range(size):
                if (x - cx) ** 2 + (z - cz) ** 2 <= radius ** 2:
                    self.terrain[x][z] = "water"

        # Scatter features deterministically.
        for x in range(1, size - 1):
            for z in range(1, size - 1):
                if self.terrain[x][z] == "water":
                    continue
                roll = self.rng.random()
                biome = self.biome_map[x][z]
                if biome == "forest":
                    if roll < 0.10:
                        self.terrain[x][z] = "tree"
                    elif roll < 0.13:
                        self.terrain[x][z] = "berry_bush"
                    elif roll < 0.145:
                        self.terrain[x][z] = "stone"
                elif biome == "plains":
                    if roll < 0.02:
                        self.terrain[x][z] = "tree"
                    elif roll < 0.04:
                        self.terrain[x][z] = "berry_bush"
                    elif roll < 0.06:
                        self.terrain[x][z] = "stone"
                    elif roll < 0.07:
                        self.terrain[x][z] = "coal_ore"
                elif biome == "desert":
                    if roll < 0.04:
                        self.terrain[x][z] = "stone"
                    elif roll < 0.06:
                        self.terrain[x][z] = "coal_ore"

        # Impassable boundary wall.
        for i in range(size):
            self.terrain[i][0] = "barrier"
            self.terrain[i][size - 1] = "barrier"
            self.terrain[0][i] = "barrier"
            self.terrain[size - 1][i] = "barrier"

    def _find_spawn(self) -> Tuple[int, int]:
        # Spawn in the plains quadrant, near its centre.
        target = self.size // 4
        for radius in range(self.size):
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    x, z = target + dx, target + dz
                    if 0 < x < self.size - 1 and 0 < z < self.size - 1:
                        if self.terrain[x][z] in PASSABLE and self.terrain[x][z] != "water":
                            if self.cfg.start_near_resources:
                                self._ensure_resources_near(x, z)
                            return x, z
        raise RuntimeError("no passable spawn cell found")

    def _ensure_resources_near(self, x: int, z: int) -> None:
        def place(offset: Tuple[int, int], block: str) -> None:
            px, pz = x + offset[0], z + offset[1]
            if 0 < px < self.size - 1 and 0 < pz < self.size - 1:
                if self.terrain[px][pz] in PASSABLE and self.terrain[px][pz] != "water":
                    self.terrain[px][pz] = block

        place((3, 0), "berry_bush")
        place((0, 3), "tree")
        place((-3, 0), "stone")

    # ------------------------------------------------------------- geometry

    def cell(self, x: float, z: float) -> str:
        ix = min(max(int(x), 0), self.size - 1)
        iz = min(max(int(z), 0), self.size - 1)
        return self.terrain[ix][iz]

    def _facing_vector(self) -> Tuple[float, float]:
        rad = math.radians(self.yaw)
        return (-math.sin(rad), math.cos(rad))

    def _front_cell(self) -> Tuple[int, int]:
        dx, dz = self._facing_vector()
        return (int(self.x + dx), int(self.z + dz))

    @property
    def time_of_day(self) -> int:
        return (self.cfg.start_time + self.tick) % self.cfg.day_length

    @property
    def is_night(self) -> bool:
        return self.time_of_day >= self.cfg.day_length // 2

    @property
    def in_water(self) -> bool:
        return self.cell(self.x, self.z) == "water"

    def is_sheltered(self) -> bool:
        ix, iz = int(self.x), int(self.z)
        solid_neighbors = 0
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if self.terrain[min(max(ix + dx, 0), self.size - 1)][
                min(max(iz + dz, 0), self.size - 1)
            ] in SOLID:
                solid_neighbors += 1
        return solid_neighbors >= 3

    # ------------------------------------------------------------------ step

    def step(self, action: Action) -> List[str]:
        """Advance the world one tick under `action`; returns semantic events."""
        if self.dead:
            return []
        events: List[str] = []
        self.tick += 1

        self._apply_action(action, events)
        self._update_vitals(action, events)
        self._update_mobs(events)
        self._update_time(events)
        self._update_shelter(events)

        dist = math.dist((self.x, self.z), self.spawn)
        if dist > self.stats["max_distance_from_spawn"]:
            self.stats["max_distance_from_spawn"] = round(dist, 2)

        if self.health <= 0 and not self.dead:
            self.dead = True
            self.health = 0.0
            events.append("died")
        return events

    def _apply_action(self, action: Action, events: List[str]) -> None:
        name = action.name
        if name == "NULL":
            return
        if name in ("MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "SPRINT", "SNEAK", "JUMP"):
            self._move(name, events)
        elif name == "LOOK_LEFT":
            self.yaw = (self.yaw - LOOK_STEP_DEG) % 360.0
        elif name == "LOOK_RIGHT":
            self.yaw = (self.yaw + LOOK_STEP_DEG) % 360.0
        elif name == "LOOK_UP":
            self.pitch = max(-90.0, self.pitch - PITCH_STEP_DEG)
        elif name == "LOOK_DOWN":
            self.pitch = min(90.0, self.pitch + PITCH_STEP_DEG)
        elif name == "ATTACK":
            self._attack(events)
        elif name == "USE":
            self._use(events)
        elif name == "SELECT_HOTBAR_SLOT":
            slot = int(action.param("slot", 0))
            if 0 <= slot < len(self.hotbar):
                self.selected_slot = slot

    def _move(self, name: str, events: List[str]) -> None:
        fx, fz = self._facing_vector()
        speed = WALK_SPEED
        dx, dz = fx, fz
        if name == "MOVE_BACKWARD":
            dx, dz = -fx, -fz
        elif name == "MOVE_LEFT":
            dx, dz = fz, -fx
        elif name == "MOVE_RIGHT":
            dx, dz = -fz, fx
        elif name == "SPRINT":
            speed = SPRINT_SPEED
            self.hunger = max(0.0, self.hunger - SPRINT_HUNGER_COST)
        elif name == "SNEAK":
            speed = SNEAK_SPEED
        elif name == "JUMP":
            speed = 0.6
            self.hunger = max(0.0, self.hunger - JUMP_HUNGER_COST)

        nx, nz = self.x + dx * speed, self.z + dz * speed
        nx = min(max(nx, 1.01), self.size - 1.01)
        nz = min(max(nz, 1.01), self.size - 1.01)
        if self.cell(nx, nz) in SOLID:
            events.append("bumped")
            return
        self.x, self.z = nx, nz

    def _attack(self, events: List[str]) -> None:
        # Prefer a mob within reach and inside the attack cone.
        fx, fz = self._facing_vector()
        for mob in self.mobs:
            vx, vz = mob["x"] - self.x, mob["z"] - self.z
            dist = math.hypot(vx, vz)
            if dist > ATTACK_REACH or dist == 0.0:
                continue
            dot = (vx * fx + vz * fz) / dist
            if dot >= math.cos(math.radians(ATTACK_CONE_DEG)):
                mob["hp"] -= 1
                events.append("hit_mob")
                if mob["hp"] <= 0:
                    self.mobs.remove(mob)
                    self.stats["mobs_killed"] += 1
                    events.append("killed_mob")
                return
        # Otherwise break the block in front, if solid.
        bx, bz = self._front_cell()
        if 0 < bx < self.size - 1 and 0 < bz < self.size - 1:
            block = self.terrain[bx][bz]
            if block in SOLID and block != "barrier":
                self.terrain[bx][bz] = "dirt"
                self.stats["blocks_broken"] += 1
                events.append(f"broke_block:{block}")
                item = BREAK_YIELD.get(block)
                if item:
                    self._add_item(item, events)

    def _use(self, events: List[str]) -> None:
        item = self.hotbar[self.selected_slot]
        if item is None or self.inventory.get(item, 0) <= 0:
            return
        if item in FOOD_ITEMS:
            self.hunger = min(20.0, self.hunger + FOOD_ITEMS[item])
            self._remove_item(item)
            self.stats["food_consumed"] += 1
            events.append("ate_food")
        elif item in PLACEABLE_ITEMS:
            bx, bz = self._front_cell()
            if 0 < bx < self.size - 1 and 0 < bz < self.size - 1:
                if self.terrain[bx][bz] in PASSABLE and (bx, bz) != (int(self.x), int(self.z)):
                    self.terrain[bx][bz] = "placed_block"
                    self._remove_item(item)
                    self.stats["blocks_placed"] += 1
                    events.append("placed_block")

    def _add_item(self, item: str, events: List[str]) -> None:
        is_new = item not in self.inventory
        self.inventory[item] = self.inventory.get(item, 0) + 1
        if is_new:
            self.stats["unique_items_collected"] += 1
            events.append(f"new_item:{item}")
            if item in FOOD_ITEMS:
                events.append("acquired_food")
            if item not in self.hotbar:
                for i, slot in enumerate(self.hotbar):
                    if slot is None:
                        self.hotbar[i] = item
                        break

    def _remove_item(self, item: str) -> None:
        count = self.inventory.get(item, 0) - 1
        if count <= 0:
            self.inventory.pop(item, None)
            self.hotbar = [None if s == item else s for s in self.hotbar]
        else:
            self.inventory[item] = count

    # ---------------------------------------------------------------- vitals

    def _damage(self, amount: float, reason: str, events: List[str]) -> None:
        self.health = max(0.0, self.health - amount)
        self.stats["damage_taken"] = round(self.stats["damage_taken"] + amount, 2)
        self.death_reason = reason
        events.append(f"damage:{reason}")

    def _update_vitals(self, action: Action, events: List[str]) -> None:
        self.hunger = max(0.0, self.hunger - HUNGER_PER_TICK)

        if self.in_water:
            self.oxygen = max(0.0, self.oxygen - 0.5)
            if self.oxygen <= 0.0:
                self._drown_counter += 1
                if self._drown_counter >= DROWN_INTERVAL:
                    self._drown_counter = 0
                    self._damage(1.0, "drowning", events)
        else:
            self.oxygen = min(20.0, self.oxygen + 1.0)
            self._drown_counter = 0

        if self.hunger <= 0.0:
            self._starve_counter += 1
            if self._starve_counter >= STARVE_INTERVAL:
                self._starve_counter = 0
                self._damage(1.0, "starvation", events)
        else:
            self._starve_counter = 0

        if self.hunger >= 18.0 and self.health < 20.0 and self.health > 0.0:
            self._regen_counter += 1
            if self._regen_counter >= REGEN_INTERVAL:
                self._regen_counter = 0
                self.health = min(20.0, self.health + 1.0)
        else:
            self._regen_counter = 0

    # ------------------------------------------------------------------ mobs

    def _update_mobs(self, events: List[str]) -> None:
        if self.is_night:
            spawn_chance = ZOMBIE_SPAWN_RATE * self.cfg.difficulty
            if len(self.mobs) < self.cfg.max_mobs and self.rng.random() < spawn_chance:
                self._spawn_zombie()
        else:
            if self.mobs:
                self.mobs = []  # zombies burn at dawn

        for mob in self.mobs:
            vx, vz = self.x - mob["x"], self.z - mob["z"]
            dist = math.hypot(vx, vz)
            if mob["cooldown"] > 0:
                mob["cooldown"] -= 1
            if dist > ZOMBIE_REACH and dist > 0.0:
                step = min(ZOMBIE_SPEED, dist)
                nx = mob["x"] + vx / dist * step
                nz = mob["z"] + vz / dist * step
                # Axis-separated collision so walls actually block mobs.
                if self.cell(nx, mob["z"]) not in SOLID:
                    mob["x"] = nx
                if self.cell(mob["x"], nz) not in SOLID:
                    mob["z"] = nz
            elif dist <= ZOMBIE_REACH and mob["cooldown"] == 0:
                self._damage(ZOMBIE_BASE_DAMAGE * self.cfg.difficulty, "zombie", events)
                mob["cooldown"] = ZOMBIE_ATTACK_COOLDOWN

    def _spawn_zombie(self) -> None:
        for _ in range(8):  # a few placement attempts
            angle = self.rng.uniform(0.0, 2.0 * math.pi)
            dist = self.rng.uniform(8.0, 14.0)
            mx = self.x + math.cos(angle) * dist
            mz = self.z + math.sin(angle) * dist
            if 1.0 < mx < self.size - 1 and 1.0 < mz < self.size - 1:
                if self.cell(mx, mz) not in SOLID:
                    self._mob_serial += 1
                    self.mobs.append(
                        {"id": self._mob_serial, "x": mx, "z": mz, "hp": ZOMBIE_HP, "cooldown": 0}
                    )
                    return

    # ------------------------------------------------------------------ time

    def _update_time(self, events: List[str]) -> None:
        night = self.is_night
        if self._was_night and not night and not self.dead:
            if not self.stats["survived_night"]:
                self.stats["survived_night"] = True
                events.append("survived_night")
        self._was_night = night

    def _update_shelter(self, events: List[str]) -> None:
        sheltered = self.is_sheltered()
        if sheltered and not self._sheltered:
            events.append("entered_shelter")
        self._sheltered = sheltered

    # ----------------------------------------------------------- observation

    def nearby_blocks(self, radius: int = 2) -> List[List[str]]:
        ix, iz = int(self.x), int(self.z)
        patch = []
        for dx in range(-radius, radius + 1):
            row = []
            for dz in range(-radius, radius + 1):
                x = min(max(ix + dx, 0), self.size - 1)
                z = min(max(iz + dz, 0), self.size - 1)
                row.append(self.terrain[x][z])
            patch.append(row)
        return patch

    def render_frame(self, radius: int = 5) -> List[List[int]]:
        """Coarse top-down 'screen' -- the MVP stand-in for pixels."""
        ix, iz = int(self.x), int(self.z)
        frame = []
        mob_cells = {(int(m["x"]), int(m["z"])) for m in self.mobs}
        for dx in range(-radius, radius + 1):
            row = []
            for dz in range(-radius, radius + 1):
                x = min(max(ix + dx, 0), self.size - 1)
                z = min(max(iz + dz, 0), self.size - 1)
                if dx == 0 and dz == 0:
                    row.append(AGENT_FRAME_ID)
                elif (x, z) in mob_cells:
                    row.append(MOB_FRAME_ID)
                else:
                    row.append(BLOCK_IDS[self.terrain[x][z]])
            frame.append(row)
        return frame

    def render_pixels(
        self, radius: int = PIXEL_RADIUS, scale: int = PIXEL_SCALE
    ) -> List[List[List[int]]]:
        """Deterministic RGB pixel render of the local view (H x W x 3, 0..255).

        The pixel counterpart to :meth:`render_frame`: the same semantic grid,
        colorized and upscaled so a small CNN gets real pixels while the result
        stays a pure function of world state -- byte-identical under replay.  A
        real backend renders/captures a frame here instead; the stream contract
        is the same.
        """
        return pixels_from_frame(self.render_frame(radius), scale)

    def mob_summary(self, limit: int = 4) -> List[Dict[str, float]]:
        fx, fz = self._facing_vector()
        facing_deg = math.degrees(math.atan2(-fx, fz))
        out = []
        for mob in self.mobs:
            vx, vz = mob["x"] - self.x, mob["z"] - self.z
            dist = math.hypot(vx, vz)
            bearing = math.degrees(math.atan2(-vx, vz))
            rel = (bearing - facing_deg + 180.0) % 360.0 - 180.0
            out.append({"distance": round(dist, 2), "angle": round(rel, 1)})
        out.sort(key=lambda m: m["distance"])
        return out[:limit]

    # ------------------------------------------------------------- snapshots

    _SNAPSHOT_FIELDS = (
        "seed", "tick", "x", "z", "spawn", "yaw", "pitch", "health", "hunger",
        "oxygen", "inventory", "hotbar", "selected_slot", "mobs", "_mob_serial",
        "dead", "death_reason", "_regen_counter", "_starve_counter",
        "_drown_counter", "_was_night", "_sheltered", "stats", "terrain",
    )

    def snapshot(self) -> str:
        snapshot_id = f"snap-{self.tick}-{len(self._snapshots)}"
        state = {name: copy.deepcopy(getattr(self, name)) for name in self._SNAPSHOT_FIELDS}
        state["_rng_state"] = self.rng.getstate()
        self._snapshots[snapshot_id] = state
        return snapshot_id

    def restore(self, snapshot_id: str) -> None:
        state = self._snapshots[snapshot_id]
        for name in self._SNAPSHOT_FIELDS:
            setattr(self, name, copy.deepcopy(state[name]))
        self.rng.setstate(state["_rng_state"])
