"""SurvivalBox reward function.

Implements the survival reward design:

- base survival (+ per tick alive, large penalty on death)
- body state (health maintained, damage, hunger loss, critical vitals)
- exploration (new block types, new biomes, meaningful distance) -- capped
- item diversity (new item types, first tool/food, first block placed) -- capped
- safety/shelter (enclosure, light source, surviving the first night)
- anti-stagnation (repeated actions, contextless idling, spinning, no novelty)

Novelty rewards are capped so the agent cannot optimise for endless
wandering or junk collection.  The reward module keeps its own per-episode
state and is reset alongside the Program.

Note: "first tool" and "light source" rules are implemented but dormant in
the simulated backend (no crafting yet); they activate as soon as a backend
can emit `new_item:<tool>` / `created_light_source` events.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.reward import RewardSignal
from cognitive_runtime.core.streams.events import StreamEvent

_TOOL_SUFFIXES = ("_pickaxe", "_axe", "_shovel", "_hoe", "_sword")
TOOL_ITEMS = {
    "shears", "fishing_rod", "flint_and_steel", "bucket",
    "bow", "crossbow", "trident", "shield",
}
FOOD_ITEM_NAMES = {"berries", "apple", "bread", "cooked_meat"}


def _is_tool_or_weapon(item: str) -> bool:
    return item in TOOL_ITEMS or item.endswith(_TOOL_SUFFIXES)


@dataclass
class SurvivalRewardConfig:
    # Base survival.
    tick_alive: float = 0.01
    death: float = -10.0
    # Body state.
    health_maintained: float = 0.05      # per window without damage, at healthy hp
    health_window_ticks: int = 100
    damage_taken: float = -0.5           # per damage event
    hunger_decrease: float = -0.25       # per whole hunger point lost
    critical_health: float = -1.0        # on entering health < threshold
    critical_hunger: float = -1.0
    critical_threshold: float = 4.0
    # Exploration (capped).
    new_block_type: float = 0.1
    new_block_cap: float = 2.0
    new_biome: float = 0.2
    new_biome_cap: float = 1.0
    distance_step: float = 0.1           # per `distance_unit` of new max distance
    distance_unit: float = 10.0
    distance_cap: float = 2.0
    # Item diversity (capped).
    new_item: float = 0.5
    new_item_cap: float = 5.0
    first_tool: float = 1.0
    first_food: float = 1.0
    first_block_placed: float = 1.0
    # Safety / shelter (each once per episode).
    shelter: float = 1.0
    light_source: float = 1.0
    survived_night: float = 1.0
    # Anti-stagnation.
    repeated_action: float = -0.01
    repeated_action_threshold: int = 20
    idle: float = -0.05
    idle_threshold: int = 40
    spinning: float = -0.1
    spinning_window: int = 24
    no_novelty: float = -0.1
    no_novelty_ticks: int = 200


class SurvivalReward:
    def __init__(self, config: SurvivalRewardConfig | None = None):
        self.cfg = config or SurvivalRewardConfig()
        self.reset()

    def reset(self) -> None:
        self._prev_health: float | None = None
        self._prev_hunger: float | None = None
        self._health_critical = False
        self._hunger_critical = False
        self._ticks_without_damage = 0
        self._seen_blocks: Set[str] = set()
        self._seen_biomes: Set[str] = set()
        self._seen_items: Set[str] = set()
        self._block_reward_total = 0.0
        self._biome_reward_total = 0.0
        self._distance_reward_total = 0.0
        self._item_reward_total = 0.0
        self._max_distance_rewarded = 0.0
        self._first_tool = False
        self._first_food = False
        self._first_block_placed = False
        self._shelter_rewarded = False
        self._light_rewarded = False
        self._night_rewarded = False
        self._recent_actions: Deque[str] = deque(maxlen=max(64, self.cfg.spinning_window))
        self._action_streak = 0
        self._last_action_key: str | None = None
        self._null_streak = 0
        self._seen_obs_hashes: Set[str] = set()
        self._ticks_since_novel = 0
        # Stream-path state: latest value per stream + spawn position.
        self._latest_streams: Dict[str, Any] = {}
        self._spawn: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------ eval

    def evaluate(
        self,
        obs_data: Dict[str, Any],
        events: List[str],
        action: Action,
        observation_hash: str,
    ) -> RewardSignal:
        """Legacy pull-style entry point: inputs from a whole Observation."""
        return self._evaluate(
            health=float(obs_data.get("health", 0.0)),
            hunger=float(obs_data.get("hunger", 0.0)),
            nearby_blocks=obs_data.get("nearby_blocks", []),
            biome=obs_data.get("biome"),
            distance=float(obs_data.get("distance_from_spawn", 0.0)),
            mobs_visible=bool(obs_data.get("mobs")),
            events=events,
            action=action,
            novelty_hash=observation_hash,
        )

    def prime_stream_state(self, stream_events: List[StreamEvent]) -> None:
        """Absorb state stream values without evaluating a tick.

        Called with the initial post-reset snapshot so the latest-value
        cache (and the spawn position) is populated before the first tick.
        """
        for event in stream_events:
            if event.modality in ("body", "vision", "spatial", "world"):
                self._latest_streams[event.stream_id] = event.payload
                if self._spawn is None and event.stream_id == "spatial.position":
                    self._spawn = (event.payload["x"], event.payload["z"])

    def evaluate_stream_window(
        self, stream_events: List[StreamEvent], action: Action
    ) -> RewardSignal:
        """Stream-native entry point: same rules, inputs from the tick's
        sensory stream events.

        On-change publishing means the latest published value of a state
        stream *is* the current value, so a per-stream latest cache stands
        in for the whole observation.  Novelty hashes the tick's sensory
        events instead of the whole-observation hash.
        """
        semantic_events: List[str] = []
        for event in stream_events:
            self.prime_stream_state([event])
            translated = _SEMANTIC_EVENTS.get(event.stream_id)
            if translated is not None:
                semantic_events.append(translated(event.payload))

        latest = self._latest_streams
        position = latest.get("spatial.position")
        distance = 0.0
        if position is not None and self._spawn is not None:
            distance = round(
                math.dist((position["x"], position["z"]), self._spawn), 2
            )
        window_digest = hashlib.sha1(
            "".join(e.hash() for e in stream_events).encode("utf-8")
        ).hexdigest()

        return self._evaluate(
            health=float(latest.get("body.health", 0.0)),
            hunger=float(latest.get("body.hunger", 0.0)),
            nearby_blocks=latest.get("world.nearby_blocks", []),
            biome=latest.get("world.biome"),
            distance=distance,
            mobs_visible=bool(latest.get("vision.entities")),
            events=semantic_events,
            action=action,
            novelty_hash=window_digest,
        )

    def _evaluate(
        self,
        health: float,
        hunger: float,
        nearby_blocks: List[List[str]],
        biome: Optional[str],
        distance: float,
        mobs_visible: bool,
        events: List[str],
        action: Action,
        novelty_hash: str,
    ) -> RewardSignal:
        cfg = self.cfg
        components: Dict[str, float] = {}
        died = "died" in events

        # ------------------------------------------------- base survival
        if died:
            components["death"] = cfg.death
        else:
            components["tick_alive"] = cfg.tick_alive

        # ---------------------------------------------------- body state
        damage_events = sum(1 for e in events if e.startswith("damage:"))
        if damage_events:
            components["damage_taken"] = cfg.damage_taken * damage_events
            self._ticks_without_damage = 0
        else:
            self._ticks_without_damage += 1
            if (
                self._ticks_without_damage % cfg.health_window_ticks == 0
                and health >= 16.0
            ):
                components["health_maintained"] = cfg.health_maintained

        if self._prev_hunger is not None:
            hunger_points_lost = int(self._prev_hunger) - int(hunger)
            if hunger_points_lost > 0:
                components["hunger_decrease"] = cfg.hunger_decrease * hunger_points_lost

        if health < cfg.critical_threshold and not self._health_critical and not died:
            components["critical_health"] = cfg.critical_health
        self._health_critical = health < cfg.critical_threshold
        if hunger < cfg.critical_threshold and not self._hunger_critical:
            components["critical_hunger"] = cfg.critical_hunger
        self._hunger_critical = hunger < cfg.critical_threshold
        self._prev_health = health
        self._prev_hunger = hunger

        # --------------------------------------------------- exploration
        block_bonus = 0.0
        for row in nearby_blocks:
            for block in row:
                if block not in self._seen_blocks:
                    self._seen_blocks.add(block)
                    block_bonus += cfg.new_block_type
        block_bonus = min(block_bonus, cfg.new_block_cap - self._block_reward_total)
        if block_bonus > 0:
            components["new_block_type"] = block_bonus
            self._block_reward_total += block_bonus

        if biome and biome not in self._seen_biomes:
            self._seen_biomes.add(biome)
            bonus = min(cfg.new_biome, cfg.new_biome_cap - self._biome_reward_total)
            if bonus > 0:
                components["new_biome"] = bonus
                self._biome_reward_total += bonus

        while (
            distance >= self._max_distance_rewarded + cfg.distance_unit
            and self._distance_reward_total < cfg.distance_cap
        ):
            self._max_distance_rewarded += cfg.distance_unit
            bonus = min(cfg.distance_step, cfg.distance_cap - self._distance_reward_total)
            components["distance"] = components.get("distance", 0.0) + bonus
            self._distance_reward_total += bonus

        # ------------------------------------------------ item diversity
        for event in events:
            if event.startswith("new_item:"):
                item = event.split(":", 1)[1]
                if item not in self._seen_items:
                    self._seen_items.add(item)
                    bonus = min(cfg.new_item, cfg.new_item_cap - self._item_reward_total)
                    if bonus > 0:
                        components["new_item"] = components.get("new_item", 0.0) + bonus
                        self._item_reward_total += bonus
                    if _is_tool_or_weapon(item) and not self._first_tool:
                        self._first_tool = True
                        components["first_tool"] = cfg.first_tool
                    if item in FOOD_ITEM_NAMES and not self._first_food:
                        self._first_food = True
                        components["first_food"] = cfg.first_food
        if "placed_block" in events and not self._first_block_placed:
            self._first_block_placed = True
            components["first_block_placed"] = cfg.first_block_placed

        # ----------------------------------------------- safety / shelter
        if "entered_shelter" in events and not self._shelter_rewarded:
            self._shelter_rewarded = True
            components["shelter"] = cfg.shelter
        if "created_light_source" in events and not self._light_rewarded:
            self._light_rewarded = True
            components["light_source"] = cfg.light_source
        if "survived_night" in events and not self._night_rewarded:
            self._night_rewarded = True
            components["survived_night"] = cfg.survived_night

        # ----------------------------------------------- anti-stagnation
        key = action.key()
        if key == self._last_action_key:
            self._action_streak += 1
        else:
            self._action_streak = 1
            self._last_action_key = key
        self._recent_actions.append(key)
        if self._action_streak > cfg.repeated_action_threshold:
            components["repeated_action"] = cfg.repeated_action

        if action.is_null:
            self._null_streak += 1
        else:
            self._null_streak = 0
        threatened = mobs_visible or health < 10.0
        if self._null_streak > cfg.idle_threshold and not threatened:
            components["idle"] = cfg.idle

        window = list(self._recent_actions)[-cfg.spinning_window :]
        if len(window) == cfg.spinning_window and all(
            k in ("LOOK_LEFT", "LOOK_RIGHT") for k in window
        ):
            components["spinning"] = cfg.spinning
            self._recent_actions.clear()

        if novelty_hash in self._seen_obs_hashes:
            self._ticks_since_novel += 1
        else:
            self._seen_obs_hashes.add(novelty_hash)
            self._ticks_since_novel = 0
        if self._ticks_since_novel >= cfg.no_novelty_ticks:
            components["no_novelty"] = cfg.no_novelty
            self._ticks_since_novel = 0

        return RewardSignal.from_components(components, events=tuple(events))


# Stream event → legacy semantic event string, so the reward core keeps
# reading the exact event vocabulary it always has.
_SEMANTIC_EVENTS = {
    "event.damage_taken": lambda p: f"damage:{p['reason']}",
    "event.item_collected": lambda p: f"new_item:{p['item']}",
    "event.block_broken": lambda p: f"broke_block:{p['block']}",
    "event.block_placed": lambda p: "placed_block",
    "event.created_light_source": lambda p: "created_light_source",
    "event.mob_killed": lambda p: "killed_mob",
    "event.bumped": lambda p: "bumped",
    "event.food_eaten": lambda p: "ate_food",
    "event.entered_shelter": lambda p: "entered_shelter",
    "event.survived_night": lambda p: "survived_night",
    "event.died": lambda p: "died",
}
