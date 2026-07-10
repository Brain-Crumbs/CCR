"""Curriculum presets for the Minecraft nursery (issue #30).

A named, documented bundle of ``SurvivalBoxConfig`` overrides,
``SurvivalRewardConfig`` weight overrides, and a fixed default seed, staged
flat safe world -> resource world -> night survival -> caves -> combat ->
crafting.  Selected from the CLI with ``--curriculum <name>``
(`cognitive_runtime.cli`); see ``docs/curriculum.md`` for the full table and
example commands.

Each preset's ``seed`` gives deterministic regression runs: the same
curriculum always generates the same episode content unless the caller
passes an explicit ``--seed``.  Presets only tune existing world/reward
knobs -- no policy or heuristic behavior is encoded here (rewards state
goals, never actions to take).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class CurriculumPreset:
    name: str
    description: str
    world_config: Dict[str, Any] = field(default_factory=dict)
    reward_config: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0


CURRICULA: Dict[str, CurriculumPreset] = {
    "flat-safe": CurriculumPreset(
        name="flat-safe",
        description=(
            "Step 1: no mobs, a day/night cycle long enough that night never "
            "arrives within the episode -- learn movement, vitals and basic "
            "exploration without any threat."
        ),
        world_config={
            "difficulty": 0.0,
            "max_mobs": 0,
            "day_length": 24000,
            "episode_ticks": 3000,
            "world_size": 48,
        },
        reward_config={
            "tick_alive": 0.02,
            "health_maintained": 0.1,
        },
        seed=100,
    ),
    "resource-world": CurriculumPreset(
        name="resource-world",
        description=(
            "Step 2: still no mobs, full-size world -- learn to gather block "
            "types and inventory items across biomes."
        ),
        world_config={
            "difficulty": 0.0,
            "max_mobs": 0,
            "day_length": 24000,
            "episode_ticks": 6000,
            "world_size": 64,
        },
        reward_config={
            "new_block_type": 0.15,
            "new_block_cap": 3.0,
            "new_item": 0.75,
            "new_item_cap": 6.0,
        },
        seed=200,
    ),
    "night-survival": CurriculumPreset(
        name="night-survival",
        description=(
            "Step 3: a short day/night cycle brings the first night quickly "
            "-- learn to find shelter and light before dark."
        ),
        world_config={
            "difficulty": 1.0,
            "max_mobs": 3,
            "day_length": 1200,
            "start_time": 0,
            "episode_ticks": 3600,
            "world_size": 64,
        },
        reward_config={
            "shelter": 1.5,
            "light_source": 1.5,
            "survived_night": 2.0,
        },
        seed=300,
    ),
    "caves": CurriculumPreset(
        name="caves",
        description=(
            "Step 4: mining focus. No mobs -- the simulated world has no "
            "literal underground yet (issue #30's out-of-scope note defers "
            "cave generation); reward instead emphasises stone/coal_ore "
            "extraction and tool use as the mining-progress signal."
        ),
        world_config={
            "difficulty": 0.0,
            "max_mobs": 0,
            "day_length": 24000,
            "episode_ticks": 6000,
            "world_size": 64,
        },
        reward_config={
            "new_block_type": 0.2,
            "new_block_cap": 3.0,
            "tool_used_item": 0.5,
            "tool_used_cap": 2.0,
        },
        seed=400,
    ),
    "combat": CurriculumPreset(
        name="combat",
        description=(
            "Step 5: spawn at dusk with more mobs allowed -- learn to fight "
            "or flee."
        ),
        world_config={
            "difficulty": 2.0,
            "max_mobs": 6,
            "day_length": 6000,
            "start_time": 3000,
            "episode_ticks": 4000,
            "world_size": 64,
        },
        reward_config={
            "damage_taken": -0.75,
            "tool_used_item": 0.75,
            "tool_used_cap": 3.0,
        },
        seed=500,
    ),
    "crafting": CurriculumPreset(
        name="crafting",
        description=(
            "Step 6: no mobs, a longer episode -- learn the gather -> craft "
            "-> smelt chain (crafting table, furnace)."
        ),
        world_config={
            "difficulty": 0.0,
            "max_mobs": 0,
            "day_length": 24000,
            "episode_ticks": 8000,
            "world_size": 64,
        },
        reward_config={
            "craft_progress": 1.0,
            "craft_progress_cap": 4.0,
            "first_tool": 1.5,
        },
        seed=600,
    ),
}

#: Curriculum step order, matching the issue's staging (docs + any tooling
#: that wants to iterate presets in sequence rather than alphabetically).
CURRICULUM_ORDER = (
    "flat-safe", "resource-world", "night-survival", "caves", "combat", "crafting",
)


def get_curriculum(name: str) -> CurriculumPreset:
    try:
        return CURRICULA[name]
    except KeyError:
        raise KeyError(
            f"unknown curriculum {name!r}; available: {', '.join(CURRICULUM_ORDER)}"
        ) from None
