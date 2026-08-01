"""CrafterWorld configuration.

Much smaller than ``programs.minecraft.config.SurvivalBoxConfig``: Crafter
has no day/night-cycle knob, no difficulty/mob-count dial and no backend
choice (the ``crafter`` package *is* the backend).  ``episode_ticks`` reuses
the same field name the CLI's ``--episode-ticks`` flag already fills in
(``cli.py:_program_config``), so passing a Minecraft-shaped config dict here
is a safe no-op for everything Crafter doesn't understand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from cognitive_runtime.core.goal import GoalDistributionConfig
from cognitive_runtime.training.navigation_reward import GoalRewardConfig


@dataclass
class CrafterConfig:
    episode_ticks: int = 10000        # crafter's own default episode length
    area: Tuple[int, int] = (64, 64)  # world grid size
    view: Tuple[int, int] = (9, 9)    # crafter's local render window, in cells
    size: Tuple[int, int] = (64, 64)  # rendered RGB pixel-frame resolution
    grid_radius: int = 4              # vision.frame.grid egocentric crop half-width
    # Realtime (--realtime) per-stream wall-clock pacing targets, matching
    # SurvivalBoxConfig's convention; ignored in fast-forward mode.
    realtime_vision_hz: float = 10.0
    realtime_body_heartbeat_hz: float = 2.0
    # Caregiver goal setting (epic #212 §12.4, issue #238). Off by default --
    # the generic-training suite (§12.2) doesn't use a goal; a navigation
    # scenario (§12.4 stage 1+, gated on this issue) turns it on. The four
    # knobs below feed one `GoalDistributionConfig` (`goal_distribution_
    # config()`) rather than being a nested dataclass field directly, so a
    # flat `program_config` dict (CLI/JSON) can still set them by name like
    # every other Crafter config key.
    goal_enabled: bool = False
    goal_min_initial_distance: float = 6.0
    goal_max_initial_distance: Optional[float] = None
    goal_commitment_horizon_ticks: int = 20
    goal_arrival_radius: float = 1.5
    # Optional deterministic caregiver goal used by solvability-checked
    # navigation corpus generators (issue #240).  It is an offset from the
    # episode's ground-truth spawn, not an absolute world coordinate, so the
    # same declared layout can be translated safely inside different world
    # sizes.  ``None`` preserves the ordinary seeded distribution above.
    goal_fixed_offset: Optional[Tuple[float, float]] = None
    # A* oracle labels + potential-based geodesic reward shaping (epic #212
    # §12.4/12.5, issue #239). Both ride on `goal_enabled` -- there is no
    # separate on/off knob -- since neither means anything without an active
    # goal to plan/shape toward. The five knobs below feed one
    # `GoalRewardConfig` (`goal_reward_config()`), same flattening convention
    # as the `goal_distribution_config()` knobs above.
    goal_reward_potential: str = "linear"
    goal_reward_strength: float = 1.0
    goal_reward_falloff: Optional[float] = None
    goal_reward_discount: float = 1.0
    goal_reward_success_bonus: float = 1.0
    # A blocked directional move's penalty (epic §12.5: "keep ... collision
    # ... separate in the recording and report"); 0.0 (off) by default so
    # enabling navigation doesn't silently change episode returns for a
    # caller that hasn't opted into a collision cost.
    goal_collision_penalty: float = 0.0

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "CrafterConfig":
        cfg = CrafterConfig()
        for key, value in raw.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def goal_distribution_config(self) -> GoalDistributionConfig:
        """The caregiver goal-proposal parameters as one
        `GoalDistributionConfig` (epic #212 §12.6: exposed for the
        `DataContract` via `GoalDistributionConfig.to_contract_dict()`)."""
        return GoalDistributionConfig(
            min_initial_distance=self.goal_min_initial_distance,
            max_initial_distance=self.goal_max_initial_distance,
            commitment_horizon_ticks=self.goal_commitment_horizon_ticks,
            arrival_radius=self.goal_arrival_radius,
        )

    def goal_reward_config(self) -> GoalRewardConfig:
        """The potential-based shaping parameters as one `GoalRewardConfig`
        (epic #212 §12.6: exposed for the `DataContract` via
        `GoalRewardConfig.to_contract_dict()`)."""
        return GoalRewardConfig(
            potential=self.goal_reward_potential,
            strength=self.goal_reward_strength,
            falloff=self.goal_reward_falloff,
            discount=self.goal_reward_discount,
            success_bonus=self.goal_reward_success_bonus,
        )


def area_from_world_size(world_size: int) -> Tuple[int, int]:
    """Translate a single ``world_size`` knob into Crafter's own ``area``
    config key (issue #202): ``CrafterConfig.from_dict`` silently drops an
    unrecognized ``world_size`` key (it isn't a field on this dataclass), so
    a caller building a Crafter program config from a Minecraft-shaped
    single-``world_size`` setting (``training.nursery.NurseryConfig``) must
    go through this rather than passing ``world_size`` straight through --
    the bug that left a notebook's requested 128-unit world silently
    recording against Crafter's default 64x64 ``area``."""
    return (int(world_size), int(world_size))
