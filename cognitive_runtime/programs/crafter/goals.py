"""Caregiver goal setting for CrafterWorld (epic #212 §12.4, issue #238).

The caregiver -- never the organism, yet -- proposes exactly one navigation
goal per episode, at reset: a point sampled deterministically from the
episode's own seed, at least ``GoalDistributionConfig.min_initial_distance``
from the spawn position. ``CrafterWorld`` (``adapter.py``) owns calling
:meth:`CaregiverGoalSetter.tick` every program tick with its own
ground-truth position and publishing the resulting :class:`~cognitive_runtime.
core.goal.GoalState` as the ``internal.goal`` stream; this module only
proposes the goal and evaluates arrival, wrapping
:class:`~cognitive_runtime.core.goal.GoalTracker`.
"""

from __future__ import annotations

import random
from typing import Tuple

from cognitive_runtime.core.goal import (
    CAREGIVER_SOURCE,
    GoalDistributionConfig,
    GoalState,
    GoalTracker,
)

__all__ = ["CaregiverGoalSetter", "sample_caregiver_goal"]


def sample_caregiver_goal(
    *,
    spawn_position: Tuple[float, float],
    world_size: Tuple[int, int],
    config: GoalDistributionConfig,
    rng: random.Random,
    max_attempts: int = 64,
) -> Tuple[float, float]:
    """Sample a point inside ``world_size``'s bounds at least
    ``config.min_initial_distance`` (and at most ``config.
    max_initial_distance``, if set) from ``spawn_position``.

    Plain rejection sampling over the world's bounding box: cheap and exact
    enough at Crafter's grid scale (tens of cells per side); it is not an
    unbiased annulus sampler (a small bias toward the world's corners is
    acceptable here, unlike the eventual ``navigate_open_goal`` corpus
    generator's own solvability-checked layout distribution, epic §12.4
    stage 1 -- out of this issue's scope).
    """
    width, height = world_size
    for _ in range(max_attempts):
        x = rng.uniform(0, width - 1)
        y = rng.uniform(0, height - 1)
        dx = x - spawn_position[0]
        dy = y - spawn_position[1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance < config.min_initial_distance:
            continue
        if config.max_initial_distance is not None and distance > config.max_initial_distance:
            continue
        return (x, y)
    raise ValueError(
        f"could not sample a goal >= {config.min_initial_distance} "
        f"(and <= {config.max_initial_distance}) from spawn {spawn_position} within a "
        f"{world_size} world after {max_attempts} attempts; widen the world or lower "
        "min_initial_distance"
    )


class CaregiverGoalSetter:
    """Owns one episode's :class:`GoalTracker`, proposing a caregiver goal at
    reset. Source is always ``"caregiver"`` (issue #238; organism-proposed
    goals are future work this issue's anti-gaming rules are built ahead
    of)."""

    def __init__(self, config: GoalDistributionConfig):
        self.config = config
        self.tracker = GoalTracker(config)

    def reset(
        self,
        *,
        spawn_position: Tuple[float, float],
        world_size: Tuple[int, int],
        seed: int,
        fixed_offset: Tuple[float, float] | None = None,
    ) -> None:
        """Propose a fresh episode goal, deterministic in ``seed`` and the
        optional ``fixed_offset``. Without an offset, a fresh
        ``random.Random`` every reset matches ``CrafterWorld``'s own "one
        seed, one fully determined episode" contract -- see
        ``adapter.py:reset``'s docstring on why a new ``crafter.Env`` is built
        per reset rather than reusing one.

        A fresh ``GoalTracker`` every episode, not the same one reused: the
        commitment horizon (epic #212 §12.4) guards goal swaps *within* one
        episode's tick clock, which restarts at 0 every reset -- reusing one
        tracker across episodes would make a brand-new episode's first
        proposal collide with the *previous* episode's still-"active"
        commitment window and be wrongly rejected.
        """
        self.tracker = GoalTracker(self.config)
        rng = random.Random(seed)
        if fixed_offset is None:
            goal_position = sample_caregiver_goal(
                spawn_position=spawn_position, world_size=world_size, config=self.config, rng=rng,
            )
        else:
            goal_position = (
                spawn_position[0] + float(fixed_offset[0]),
                spawn_position[1] + float(fixed_offset[1]),
            )
            width, height = world_size
            if not (0 <= goal_position[0] < width and 0 <= goal_position[1] < height):
                raise ValueError(
                    f"fixed caregiver goal {goal_position} from offset {fixed_offset} is outside "
                    f"world bounds {world_size}"
                )
        rejection = self.tracker.propose(
            goal_position=goal_position,
            current_position=spawn_position,
            tick=0,
            source=CAREGIVER_SOURCE,
        )
        assert rejection is None, (
            f"caregiver goal violated its declared distance/commitment policy: {rejection}"
        )

    def tick(self, *, current_position: Tuple[float, float]) -> GoalState:
        """This program tick's ``internal.goal`` state: evaluates arrival
        against ``current_position`` (the harness's own ground truth) first,
        then reports it."""
        self.tracker.evaluate_arrival(current_position)
        return self.tracker.state(current_position)
