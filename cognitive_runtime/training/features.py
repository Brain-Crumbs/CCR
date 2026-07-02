"""Feature extraction for the SurvivalBox learned policy.

Turns a structured observation plus recent action history into a fixed
feature vector.  This is program-specific (it knows SurvivalBox observation
keys); it lives in training/ because it is shared by the offline trainer
and the online LearnedPolicy, and must stay identical between them.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE

ACTION_KEYS: List[str] = [a.key() for a in ACTION_SPACE]

_HARVESTABLE = {"tree", "berry_bush", "stone", "coal_ore"}
_SOLID = {"tree", "stone", "coal_ore", "berry_bush", "placed_block", "barrier"}
_FOOD = {"berries"}
_PLACEABLE = {"log", "cobblestone", "dirt", "sand"}

FEATURE_NAMES: List[str] = (
    [
        "health", "hunger", "oxygen", "is_night", "in_water", "sheltered",
        "time_sin", "time_cos", "yaw_sin", "yaw_cos", "pitch",
        "distance_from_spawn",
        "mob_present", "mob_distance", "mob_angle",
        "front_water", "front_harvestable", "front_solid",
        "patch_water", "patch_solid", "patch_harvestable", "patch_food",
        "inv_food", "inv_placeable", "selected_is_food",
        "repeat_streak",
    ]
    + [f"last_action:{key}" for key in ACTION_KEYS]
)


def featurize(obs_data: Dict[str, Any], recent_action_keys: Sequence[str]) -> List[float]:
    """recent_action_keys: most recent last."""
    health = float(obs_data.get("health", 0.0))
    hunger = float(obs_data.get("hunger", 0.0))
    oxygen = float(obs_data.get("oxygen", 0.0))
    time_frac = 2.0 * math.pi * float(obs_data.get("time_of_day", 0)) / max(
        float(obs_data.get("day_length", 1)), 1.0
    )
    yaw = math.radians(float(obs_data.get("yaw", 0.0)))

    mobs = obs_data.get("mobs") or []
    if mobs:
        mob_present = 1.0
        mob_distance = min(float(mobs[0]["distance"]), 16.0) / 16.0
        mob_angle = float(mobs[0]["angle"]) / 180.0
    else:
        mob_present, mob_distance, mob_angle = 0.0, 1.0, 0.0

    front = obs_data.get("front_block", "grass")
    patch = [b for row in obs_data.get("nearby_blocks", []) for b in row]
    n_patch = max(len(patch), 1)

    inventory = obs_data.get("inventory") or {}
    hotbar = obs_data.get("hotbar") or []
    selected = obs_data.get("selected_slot", 0)
    selected_item = hotbar[selected] if 0 <= selected < len(hotbar) else None

    streak = 0
    for key in reversed(recent_action_keys):
        if key == (recent_action_keys[-1] if recent_action_keys else None):
            streak += 1
        else:
            break

    features = [
        health / 20.0,
        hunger / 20.0,
        oxygen / 20.0,
        1.0 if obs_data.get("is_night") else 0.0,
        1.0 if obs_data.get("in_water") else 0.0,
        1.0 if obs_data.get("sheltered") else 0.0,
        math.sin(time_frac),
        math.cos(time_frac),
        math.sin(yaw),
        math.cos(yaw),
        float(obs_data.get("pitch", 0.0)) / 90.0,
        min(float(obs_data.get("distance_from_spawn", 0.0)), 32.0) / 32.0,
        mob_present,
        mob_distance,
        mob_angle,
        1.0 if front == "water" else 0.0,
        1.0 if front in _HARVESTABLE else 0.0,
        1.0 if front in _SOLID else 0.0,
        sum(1 for b in patch if b == "water") / n_patch,
        sum(1 for b in patch if b in _SOLID) / n_patch,
        sum(1 for b in patch if b in _HARVESTABLE) / n_patch,
        sum(1 for b in patch if b in _FOOD or b == "berry_bush") / n_patch,
        min(sum(c for i, c in inventory.items() if i in _FOOD), 5.0) / 5.0,
        min(sum(c for i, c in inventory.items() if i in _PLACEABLE), 10.0) / 10.0,
        1.0 if selected_item in _FOOD else 0.0,
        min(streak, 20.0) / 20.0,
    ]
    last_key = recent_action_keys[-1] if recent_action_keys else None
    features.extend(1.0 if key == last_key else 0.0 for key in ACTION_KEYS)
    return features
