"""Scripted survival policy (Minecraft-specific policy experiment).

A simple hardcoded policy that validates the reward function and metrics
and proves the Program is usable:

    if in danger: face the threat and fight, or flee when weak
    if hungry and food exists: eat
    if hungry and food visible: harvest it
    if stuck: gather what blocks the way, or turn
    else: wander forward

Program-specific knowledge is allowed here (policies are experiments);
it must never leak into the runtime core.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import Policy
from cognitive_runtime.core.world_model import Prediction

_MOVE = Action("MOVE_FORWARD")
_BACK = Action("MOVE_BACKWARD")
_SPRINT = Action("SPRINT")
_LEFT = Action("LOOK_LEFT")
_RIGHT = Action("LOOK_RIGHT")
_ATTACK = Action("ATTACK")
_USE = Action("USE")

_FOOD = "berries"
_HARVESTABLE = {"berry_bush", "tree", "stone", "coal_ore"}
_IMPASSABLE = {"barrier", "placed_block"}


class ScriptedSurvivalPolicy(Policy):
    name = "scripted"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.rng = random.Random(self.seed)
        self._last_pos: Optional[tuple] = None
        self._stuck_ticks = 0
        self._turn_bias = _RIGHT

    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        obs: Dict[str, Any] = state.observation.data
        health = obs.get("health", 20.0)
        hunger = obs.get("hunger", 20.0)
        hotbar = obs.get("hotbar", [])
        mobs = obs.get("mobs", [])
        front = obs.get("front_block", "grass")
        pos = obs.get("position", {})
        here = (round(pos.get("x", 0.0), 2), round(pos.get("z", 0.0), 2))

        self._track_stuck(here, memory)

        # --- danger handling -------------------------------------------------
        if mobs:
            nearest = mobs[0]
            dist, angle = nearest["distance"], nearest["angle"]
            if health <= 8.0:
                # Too weak to fight: turn away and sprint.
                if abs(angle) < 120.0:
                    return _RIGHT if angle <= 0 else _LEFT
                return _SPRINT
            if dist <= 6.0:
                if angle > 20.0:
                    return _RIGHT
                if angle < -20.0:
                    return _LEFT
                if dist <= 2.0:
                    return _ATTACK
                return _MOVE  # close the gap while facing the mob

        # --- water escape -----------------------------------------------------
        if obs.get("in_water"):
            return _BACK

        # --- eating -----------------------------------------------------------
        if hunger <= 12.0 and _FOOD in hotbar:
            slot = hotbar.index(_FOOD)
            if obs.get("selected_slot") != slot:
                return Action.make("SELECT_HOTBAR_SLOT", slot=slot)
            return _USE

        # --- low health: hold still and let regen work ------------------------
        if health <= 6.0 and hunger > 12.0 and not mobs:
            return NULL_ACTION

        # --- harvesting -------------------------------------------------------
        if front in _HARVESTABLE:
            return _ATTACK

        # --- walls: turn instead of walking into them --------------------------
        if front in _IMPASSABLE:
            return self._turn_bias

        # --- stuck recovery ---------------------------------------------------
        if self._stuck_ticks >= 4:
            self._stuck_ticks = 0
            self._turn_bias = self.rng.choice((_LEFT, _RIGHT))
            return self._turn_bias

        # --- avoid walking into water ------------------------------------------
        if front == "water":
            return self._turn_bias

        # --- wander -------------------------------------------------------------
        if self.rng.random() < 0.04:
            return self.rng.choice((_LEFT, _RIGHT))
        return _MOVE

    def _track_stuck(self, here: tuple, memory: Memory) -> None:
        last_action = memory.last_actions(1)
        moved = self._last_pos is None or (
            abs(here[0] - self._last_pos[0]) + abs(here[1] - self._last_pos[1]) > 0.05
        )
        if last_action and last_action[0].name in (
            "MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT", "SPRINT", "JUMP",
        ) and not moved:
            self._stuck_ticks += 1
        elif moved:
            self._stuck_ticks = 0
        self._last_pos = here
