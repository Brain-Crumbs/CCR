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
import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cognitive_runtime.core.action import Action
from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig

# Exploration-coverage chunk size (issue #44), in world units.
EXPLORATION_CHUNK_SIZE = 8.0

# Ground (passable) and feature (solid) cell types.
PASSABLE = {"grass", "dirt", "sand", "water"}
SOLID = {
    "tree", "stone", "coal_ore", "berry_bush", "placed_block", "barrier",
    "crafting_table", "furnace", "chest",
}

#: Blocks that respond to USE with a container/crafting interaction instead
#: of the food/placeable-item logic (issue #40: container / crafting table /
#: furnace interactions).
CONTAINER_BLOCKS = {"crafting_table", "furnace", "chest"}

BREAK_YIELD = {
    "tree": "log",
    "stone": "cobblestone",
    "coal_ore": "coal",
    "berry_bush": "berries",
    "placed_block": "dirt",
}
FOOD_ITEMS = {"berries": 3.0}  # hunger restored per unit
PLACEABLE_ITEMS = {"log", "cobblestone", "dirt", "sand", "torch"}

#: Item names the "tool use" / "first tool" reward goals key on (issue #30):
#: real vanilla tool/weapon suffixes plus a few whole-name tool items. Shared
#: with `rewards.py` so world mechanics and reward rules agree on vocabulary.
_TOOL_SUFFIXES = ("_pickaxe", "_axe", "_shovel", "_hoe", "_sword")
TOOL_ITEMS = {
    "shears", "fishing_rod", "flint_and_steel", "bucket",
    "bow", "crossbow", "trident", "shield",
}


def is_tool_or_weapon(item: str) -> bool:
    return item in TOOL_ITEMS or item.endswith(_TOOL_SUFFIXES)


#: The sim's minimal crafting/smelting table: each container type tries its
#: recipes in order, applying the first whose inputs are satisfied --
#: (recipe id, inputs, outputs) per entry.  Real vanilla recipes are far
#: richer; this is the smallest slice that exercises the `event.crafted`
#: stream (issue #40) and, via `planks_to_pickaxe` / `smelt_torch`, the
#: "tool use" and "light placement" reward goals (issue #30) end to end
#: without a live server.
RECIPES: Dict[str, List[Tuple[str, Dict[str, int], Dict[str, int]]]] = {
    "crafting_table": [
        ("log_to_planks", {"log": 1}, {"planks": 4}),
        ("planks_to_pickaxe", {"planks": 3}, {"wooden_pickaxe": 1}),
    ],
    "furnace": [
        ("smelt_cobblestone", {"cobblestone": 1, "coal": 1}, {"stone": 1}),
        ("smelt_torch", {"coal": 1}, {"torch": 4}),
    ],
}

#: recipe id -> container it requires, derived from RECIPES so a CRAFT(recipe)
#: action (issue #42) can validate placement without duplicating the table.
RECIPE_CONTAINER: Dict[str, str] = {
    name: container for container, recipes in RECIPES.items() for name, _, _ in recipes
}
#: Stable (insertion) order of every recipe id, for enumerating CRAFT(recipe)
#: as a discrete action per recipe.
RECIPE_NAMES: Tuple[str, ...] = tuple(RECIPE_CONTAINER)

BLOCK_IDS = {
    "grass": 1, "dirt": 2, "sand": 3, "water": 4, "tree": 5,
    "stone": 6, "coal_ore": 7, "berry_bush": 8, "placed_block": 9, "barrier": 10,
    "crafting_table": 11, "furnace": 12, "chest": 13, "portal": 14,
}
MOB_FRAME_ID = 90
AGENT_FRAME_ID = 99

#: Milestone -> synthetic advancement id, each earned once per episode.
#: "sim.*" ids are the toy sandbox's stand-in for vanilla advancements (a
#: real backend forwards actual vanilla ids instead; see event.advancement in
#: streams.py).  Each predicate reads the tick's raw event-string list.
_ADVANCEMENT_TRIGGERS: Tuple[Tuple[str, Any], ...] = (
    ("sim.mine_wood", lambda events: "new_item:log" in events),
    ("sim.mine_stone", lambda events: "new_item:cobblestone" in events),
    ("sim.eat_food", lambda events: "ate_food" in events),
    ("sim.kill_mob", lambda events: "killed_mob" in events),
    ("sim.build_shelter", lambda events: "entered_shelter" in events),
    ("sim.survive_night", lambda events: "survived_night" in events),
    ("sim.craft_item", lambda events: any(e.startswith("crafted:") for e in events)),
    ("sim.explore_structure",
     lambda events: any(e.startswith("structure_discovered:") for e in events)),
    ("sim.enter_portal",
     lambda events: any(e.startswith("dimension_changed:") for e in events)),
)

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
    "crafting_table": (120, 80, 40),
    "furnace": (90, 90, 90),
    "chest": (150, 110, 40),
    "portal": (130, 60, 200),
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


#: Frame-code -> RGB lookup table, built once and indexed vectorized (a code
#: past the known vocabulary clips to the last row, then gets overwritten by
#: the unknown-cell color below).
_MAX_FRAME_CODE = max(FRAME_CODE_COLORS) if FRAME_CODE_COLORS else 0
_FRAME_COLOR_LUT = np.full((_MAX_FRAME_CODE + 1, 3), _UNKNOWN_CELL_COLOR, dtype=np.uint8)
for _code, _rgb in FRAME_CODE_COLORS.items():
    _FRAME_COLOR_LUT[_code] = _rgb


def pixels_from_frame(
    frame: List[List[int]],
    scale: int = PIXEL_SCALE,
    yaw_degrees: Optional[float] = None,
) -> np.ndarray:
    """Colorize a semantic grid frame (frame codes) into an H*scale x W*scale x 3
    RGB image (uint8 ndarray).  Backend-agnostic: any world that emits the
    shared grid frame gets identical pixel vision, and the mapping is a pure
    function of the grid (so it stays deterministic / replay-safe).

    When ``yaw_degrees`` is provided, the grid is first resampled into an
    ego-centric orientation: the center cell remains the agent, but the local
    patch rotates with the agent's camera heading. This is still a compact
    semantic fallback, not a true perspective render, but turn-in-place now
    produces changing pixels instead of a static minimap.
    """
    grid = np.asarray(frame, dtype=np.int64)
    if yaw_degrees is not None and grid.ndim == 2 and grid.shape[0] == grid.shape[1]:
        grid = _orient_frame_grid(grid, float(yaw_degrees))
    unknown = grid > _MAX_FRAME_CODE
    codes = np.where(unknown, 0, grid)
    colored = _FRAME_COLOR_LUT[codes]
    if unknown.any():
        colored[unknown] = _UNKNOWN_CELL_COLOR
    return np.repeat(np.repeat(colored, scale, axis=0), scale, axis=1)


def _orient_frame_grid(grid: np.ndarray, yaw_degrees: float) -> np.ndarray:
    size = grid.shape[0]
    radius = size // 2
    yaw = math.radians(yaw_degrees)
    forward_x, forward_z = -math.sin(yaw), math.cos(yaw)
    right_x, right_z = math.cos(yaw), math.sin(yaw)
    out = np.empty_like(grid)
    for row in range(size):
        lateral = row - radius
        for col in range(size):
            forward = col - radius
            src_dx = int(round(right_x * lateral + forward_x * forward))
            src_dz = int(round(right_z * lateral + forward_z * forward))
            src_row = min(max(src_dx + radius, 0), size - 1)
            src_col = min(max(src_dz + radius, 0), size - 1)
            out[row, col] = grid[src_row, src_col]
    return out

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
        self._place_features()
        self.structures: Dict[Tuple[int, int], str] = self._place_structures()

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
        self.inventory_open = False
        self.mobs: List[Dict[str, Any]] = []
        self._mob_serial = 0
        self.dead = False
        self.death_reason: Optional[str] = None
        self._regen_counter = 0
        self._starve_counter = 0
        self._drown_counter = 0
        self._was_night = False
        self._sheltered = False
        self.dimension = "overworld"
        self._biome = self.biome_map[spawn[0]][spawn[1]]
        self._discovered_structures: set = set()
        self._advancements_earned: set = set()
        # Exploration coverage (issue #44): unique 8x8 position chunks visited
        # this episode -- a spatial-novelty proxy independent of reward-profile
        # configuration, unlike the profile's own capped-novelty components.
        self._visited_chunks: set = set()
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
            "exploration_coverage": 0,
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

    def _safe_place(self, x: int, z: int, block: str) -> bool:
        """Overwrite one non-boundary, non-water cell; no-op if out of range
        or already water/boundary (keeps tiny world sizes safe)."""
        if 0 < x < self.size - 1 and 0 < z < self.size - 1 and self.terrain[x][z] not in (
            "barrier", "water",
        ):
            self.terrain[x][z] = block
            return True
        return False

    def _place_features(self) -> None:
        """Deterministic (non-random) placement of the container/portal
        blocks, so every seed offers the same reachable set for exercising
        `event.container_interaction` / `event.crafted` / `event.dimension_changed`
        without depending on the procedural terrain roll (issue #40)."""
        half = self.size // 2
        self._safe_place(half + half // 2, max(1, half // 4), "crafting_table")
        self._safe_place(max(1, half // 4), half + half // 2, "furnace")
        self._safe_place(self.size - 2, half + 2, "chest")
        self._safe_place(2, 2, "portal")

    def _place_structures(self) -> Dict[Tuple[int, int], str]:
        """Fixed marker cells for the three discoverable structures.

        The sim has no real village/stronghold/fortress generation; these are
        location labels one biome quadrant apiece so `event.structure_discovered`
        is exercisable deterministically without a live server.  A real
        backend reports actual generated structures instead."""
        half = self.size // 2
        return {
            (min(self.size - 2, half + half // 4), max(1, half // 2)): "village",
            (max(1, half - 2), 2): "stronghold",
            (max(1, half - 2), self.size - 2): "fortress",
        }

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
        self._update_biome(events)
        self._update_structures(events)

        dist = math.dist((self.x, self.z), self.spawn)
        if dist > self.stats["max_distance_from_spawn"]:
            self.stats["max_distance_from_spawn"] = round(dist, 2)

        chunk = (
            math.floor(self.x / EXPLORATION_CHUNK_SIZE),
            math.floor(self.z / EXPLORATION_CHUNK_SIZE),
        )
        if chunk not in self._visited_chunks:
            self._visited_chunks.add(chunk)
            self.stats["exploration_coverage"] = len(self._visited_chunks)

        if self.health <= 0 and not self.dead:
            self.dead = True
            self.health = 0.0
            events.append("died")
        self._check_advancements(events)
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
        elif name == "INTERACT":
            self._interact(events)
        elif name == "OPEN_INVENTORY":
            self._open_inventory(events)
        elif name == "CLOSE_INVENTORY":
            self._close_inventory(events)
        elif name == "EQUIP_ITEM":
            self._equip_item(int(action.param("slot", -1)), events)
        elif name == "PLACE_BLOCK":
            self._place_block(int(action.param("slot", -1)), events)
        elif name == "USE_ITEM":
            self._use_item(int(action.param("slot", -1)), events)
        elif name == "MOVE_INVENTORY_ITEM":
            self._move_inventory_item(
                int(action.param("from_slot", -1)), int(action.param("to_slot", -1)), events
            )
        elif name == "CRAFT":
            self._craft(str(action.param("recipe", "")), events)

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
        entering_portal = self.cell(nx, nz) == "portal"
        self.x, self.z = nx, nz
        if entering_portal:
            to_dim = "nether" if self.dimension == "overworld" else "overworld"
            events.append(f"dimension_changed:{self.dimension}:{to_dim}")
            self.dimension = to_dim

    def _attack(self, events: List[str]) -> None:
        # "Tool use" reward goal (issue #30): a swing while a tool/weapon is
        # equipped is the signal, independent of whether it lands -- rewards
        # state the goal (use your tools), never the strategy.
        held = self.hotbar[self.selected_slot]
        if held is not None and is_tool_or_weapon(held):
            events.append(f"used_tool:{held}")
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
                events.append(
                    "block_broken_exact:" + json.dumps(
                        {"block": block, "position": {"x": bx, "y": 64.0, "z": bz}}
                    )
                )
                item = BREAK_YIELD.get(block)
                if item:
                    self._add_item(item, events)

    def _use(self, events: List[str]) -> None:
        bx, bz = self._front_cell()
        if 0 < bx < self.size - 1 and 0 < bz < self.size - 1:
            front = self.terrain[bx][bz]
            if front in CONTAINER_BLOCKS:
                events.append(
                    "container_interact:" + json.dumps(
                        {"container": front, "position": {"x": bx, "y": 64.0, "z": bz}}
                    )
                )
                self._try_craft(front, events)
                return

        item = self.hotbar[self.selected_slot]
        if item is None or self.inventory.get(item, 0) <= 0:
            return
        if item in FOOD_ITEMS:
            self.hunger = min(20.0, self.hunger + FOOD_ITEMS[item])
            self._remove_item(item)
            self.stats["food_consumed"] += 1
            events.append("ate_food")
        elif item in PLACEABLE_ITEMS:
            if 0 < bx < self.size - 1 and 0 < bz < self.size - 1:
                if self.terrain[bx][bz] in PASSABLE and (bx, bz) != (int(self.x), int(self.z)):
                    self.terrain[bx][bz] = "placed_block"
                    self._remove_item(item)
                    self.stats["blocks_placed"] += 1
                    events.append("placed_block")
                    events.append(
                        "block_placed_exact:" + json.dumps(
                            {"block": item, "position": {"x": bx, "y": 64.0, "z": bz}}
                        )
                    )
                    if item == "torch":
                        # "Light placement" reward goal (issue #30): completes
                        # the light_source reward, dormant since it had no
                        # sim-side trigger.
                        events.append("created_light_source")

    def _reject(self, reason: str, events: List[str]) -> None:
        """Record an ``event.action_rejected`` cause (issue #42): the agent
        gets feedback instead of silence for an invalid parameterized action
        (craft without materials, equip an empty slot, place/use out of
        range, ...)."""
        events.append(f"action_rejected:{reason}")

    def _interact(self, events: List[str]) -> None:
        """Generic interaction with whatever is directly in front: a
        container in the simulated world (chest/furnace/crafting table --
        same event as USE's container branch, but never auto-crafts; CRAFT
        is the explicit trigger for that).  Doors/villagers have no
        sim-side model yet; a live server supplies them via the mineflayer
        bridge."""
        bx, bz = self._front_cell()
        if 0 < bx < self.size - 1 and 0 < bz < self.size - 1:
            front = self.terrain[bx][bz]
            if front in CONTAINER_BLOCKS:
                events.append(
                    "container_interact:" + json.dumps(
                        {"container": front, "position": {"x": bx, "y": 64.0, "z": bz}}
                    )
                )
                return
        self._reject("nothing to interact with", events)

    def _open_inventory(self, events: List[str]) -> None:
        if self.inventory_open:
            self._reject("inventory already open", events)
            return
        self.inventory_open = True

    def _close_inventory(self, events: List[str]) -> None:
        if not self.inventory_open:
            self._reject("inventory already closed", events)
            return
        self.inventory_open = False

    def _equip_item(self, slot: int, events: List[str]) -> None:
        if not 0 <= slot < len(self.hotbar):
            self._reject(f"invalid slot {slot}", events)
            return
        if self.hotbar[slot] is None:
            self._reject(f"cannot equip empty slot {slot}", events)
            return
        self.selected_slot = slot

    def _place_block(self, slot: int, events: List[str]) -> None:
        if not 0 <= slot < len(self.hotbar):
            self._reject(f"invalid slot {slot}", events)
            return
        item = self.hotbar[slot]
        if item is None or self.inventory.get(item, 0) <= 0:
            self._reject(f"no item to place in slot {slot}", events)
            return
        if item not in PLACEABLE_ITEMS:
            self._reject(f"{item} is not placeable", events)
            return
        bx, bz = self._front_cell()
        if not (0 < bx < self.size - 1 and 0 < bz < self.size - 1):
            self._reject("cannot place block out of bounds", events)
            return
        if self.terrain[bx][bz] not in PASSABLE or (bx, bz) == (int(self.x), int(self.z)):
            self._reject("target cell is not placeable", events)
            return
        self.terrain[bx][bz] = "placed_block"
        self._remove_item(item)
        self.stats["blocks_placed"] += 1
        events.append("placed_block")
        events.append(
            "block_placed_exact:" + json.dumps(
                {"block": item, "position": {"x": bx, "y": 64.0, "z": bz}}
            )
        )
        if item == "torch":
            events.append("created_light_source")

    def _use_item(self, slot: int, events: List[str]) -> None:
        if not 0 <= slot < len(self.hotbar):
            self._reject(f"invalid slot {slot}", events)
            return
        item = self.hotbar[slot]
        if item is None or self.inventory.get(item, 0) <= 0:
            self._reject(f"no item to use in slot {slot}", events)
            return
        if item not in FOOD_ITEMS:
            self._reject(f"{item} is not usable", events)
            return
        self.hunger = min(20.0, self.hunger + FOOD_ITEMS[item])
        self._remove_item(item)
        self.stats["food_consumed"] += 1
        events.append("ate_food")

    def _move_inventory_item(self, from_slot: int, to_slot: int, events: List[str]) -> None:
        if not (0 <= from_slot < len(self.hotbar) and 0 <= to_slot < len(self.hotbar)):
            self._reject(f"invalid slots {from_slot},{to_slot}", events)
            return
        if from_slot == to_slot:
            self._reject("cannot move a slot onto itself", events)
            return
        if self.hotbar[from_slot] is None and self.hotbar[to_slot] is None:
            self._reject(f"both slots {from_slot},{to_slot} are empty", events)
            return
        self.hotbar[from_slot], self.hotbar[to_slot] = self.hotbar[to_slot], self.hotbar[from_slot]

    def _craft(self, recipe: str, events: List[str]) -> None:
        """Craft a specific recipe by id (issue #42), parameterized unlike
        the implicit "USE a container" path -- rejected, not silently
        skipped, when the container or materials aren't right."""
        container = RECIPE_CONTAINER.get(recipe)
        if container is None:
            self._reject(f"unknown recipe {recipe!r}", events)
            return
        bx, bz = self._front_cell()
        front = self.terrain[bx][bz] if 0 < bx < self.size - 1 and 0 < bz < self.size - 1 else None
        if front != container:
            self._reject(f"recipe {recipe!r} needs a {container}", events)
            return
        _, inputs, outputs = next(
            (n, i, o) for n, i, o in RECIPES[container] if n == recipe
        )
        if not all(self.inventory.get(item, 0) >= count for item, count in inputs.items()):
            self._reject(f"insufficient materials for {recipe!r}", events)
            return
        self._apply_recipe(recipe, inputs, outputs, events)

    def _apply_recipe(
        self, name: str, inputs: Dict[str, int], outputs: Dict[str, int], events: List[str]
    ) -> None:
        for item, count in inputs.items():
            self._remove_item(item, count)
        for item, count in outputs.items():
            self._add_item(item, events, count)
        events.append(
            "crafted:" + json.dumps({"recipe": name, "inputs": inputs, "outputs": outputs})
        )

    def _try_craft(self, container: str, events: List[str]) -> None:
        """Minimal deterministic crafting/smelting: fixed recipes per
        container type, tried in order (see `RECIPES`).  This is USE's
        implicit auto-craft (issue #40), kept for backward compatibility
        with recorded sessions; CRAFT(recipe) (issue #42) is the explicit,
        parameterized, rejection-on-failure alternative."""
        for name, inputs, outputs in RECIPES.get(container, []):
            if not all(self.inventory.get(item, 0) >= count for item, count in inputs.items()):
                continue
            self._apply_recipe(name, inputs, outputs, events)
            return

    def _add_item(self, item: str, events: List[str], count: int = 1) -> None:
        is_new = item not in self.inventory
        self.inventory[item] = self.inventory.get(item, 0) + count
        events.append(
            "item_collected_exact:" + json.dumps({"item": item, "count": count})
        )
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

    def _remove_item(self, item: str, amount: int = 1) -> None:
        remaining = self.inventory.get(item, 0) - amount
        if remaining <= 0:
            self.inventory.pop(item, None)
            self.hotbar = [None if s == item else s for s in self.hotbar]
        else:
            self.inventory[item] = remaining

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

    def _update_biome(self, events: List[str]) -> None:
        biome = self.biome_map[int(self.x)][int(self.z)]
        if biome != self._biome:
            self._biome = biome
            events.append(f"biome_entered:{biome}")

    def _update_structures(self, events: List[str]) -> None:
        name = self.structures.get((int(self.x), int(self.z)))
        if name is not None and name not in self._discovered_structures:
            self._discovered_structures.add(name)
            events.append(f"structure_discovered:{name}")

    def _check_advancements(self, events: List[str]) -> None:
        for advancement_id, predicate in _ADVANCEMENT_TRIGGERS:
            if advancement_id in self._advancements_earned:
                continue
            if predicate(events):
                self._advancements_earned.add(advancement_id)
                events.append(f"advancement:{advancement_id}")

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
    ) -> np.ndarray:
        """Deterministic RGB pixel render of the local view (H x W x 3, 0..255).

        The pixel counterpart to :meth:`render_frame`: the same semantic grid,
        rotated by yaw, colorized and upscaled so a small CNN gets changing
        ego-centric pixels while the result stays a pure function of world
        state -- byte-identical under replay. A real backend renders/captures
        a frame here instead; the stream contract is the same.
        """
        return pixels_from_frame(self.render_frame(radius), scale, yaw_degrees=self.yaw)

    def _has_line_of_sight(self, x0: float, z0: float, x1: float, z1: float) -> bool:
        """True when no SOLID cell lies strictly between the two points.

        A coarse raycast on the terrain grid, sampled roughly every half
        cell -- the sim's stand-in for a real backend's occlusion check, so
        a mob standing behind a wall stops appearing in ``vision.entities``
        instead of always being "visible" purely by distance (issue #27:
        object permanence needs entities that can actually go out of view).
        """
        distance = math.hypot(x1 - x0, z1 - z0)
        if distance <= 0:
            return True
        steps = max(1, int(distance * 2))
        for i in range(1, steps):
            t = i / steps
            if self.cell(x0 + (x1 - x0) * t, z0 + (z1 - z0) * t) in SOLID:
                return False
        return True

    def mob_summary(self, limit: int = 4) -> List[Dict[str, float]]:
        """Visible mobs, nearest first: id + distance/angle, occluded ones
        dropped by ``_has_line_of_sight``.  The stable ``id`` (assigned once
        at spawn, see ``_spawn_zombie``) is what lets a consumer track one
        mob's identity across an occlusion gap instead of only ever seeing
        an anonymous nearest-mob blob.
        """
        fx, fz = self._facing_vector()
        facing_deg = math.degrees(math.atan2(-fx, fz))
        out = []
        for mob in self.mobs:
            if not self._has_line_of_sight(self.x, self.z, mob["x"], mob["z"]):
                continue
            vx, vz = mob["x"] - self.x, mob["z"] - self.z
            dist = math.hypot(vx, vz)
            bearing = math.degrees(math.atan2(-vx, vz))
            rel = (bearing - facing_deg + 180.0) % 360.0 - 180.0
            out.append({"id": mob["id"], "distance": round(dist, 2), "angle": round(rel, 1)})
        out.sort(key=lambda m: m["distance"])
        return out[:limit]

    # ------------------------------------------------------------- snapshots

    _SNAPSHOT_FIELDS = (
        "seed", "tick", "x", "z", "spawn", "yaw", "pitch", "health", "hunger",
        "oxygen", "inventory", "hotbar", "selected_slot", "inventory_open",
        "mobs", "_mob_serial",
        "dead", "death_reason", "_regen_counter", "_starve_counter",
        "_drown_counter", "_was_night", "_sheltered", "stats", "terrain",
        "dimension", "_biome", "_discovered_structures", "_advancements_earned",
        "_visited_chunks",
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
