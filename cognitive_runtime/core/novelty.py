"""Combine the world model's prediction error and the entity-persistence
probe's surprise into one novelty score per tick (issue #27).

Both inputs are optional -- either signal may be unavailable (a heuristic
``TrendWorldModel``/``NullEntityPersistence``, or nothing occluded this
tick) -- so the combination is the mean of whichever are present, `None`
when neither is.  Kept torch-free and dependency-free so the runtime loop
can always import it, whether or not a neural bridge is wired in.
"""

from __future__ import annotations

from typing import Optional


def combine_novelty(
    world_model_error: Optional[float], entity_surprise: Optional[float]
) -> Optional[float]:
    values = [v for v in (world_model_error, entity_surprise) if v is not None]
    if not values:
        return None
    return sum(values) / len(values)
