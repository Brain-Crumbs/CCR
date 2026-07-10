"""Per-entity feature vector shared by the entity tracker and the neural
persistence model (issue #27).

``vision.entities`` payloads are a list of ``{id, distance, angle}`` dicts
(one per currently-visible mob, see ``programs.minecraft.world.mob_summary``).
This module turns *one* such dict into a small, fixed-width feature vector --
the "identity/position" latent a persistence model learns to hold onto while
its entity is occluded -- independent of how many other entities are visible
that tick.  Kept torch-free so both :mod:`cognitive_runtime.core.entity_tracker`
(pure Python, always available) and the neural model can share one
definition of what an entity "looks like".
"""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Tuple

#: The stream carrying raw ``{id, distance, angle}`` entity records.
VISION_ENTITIES_STREAM = "vision.entities"

#: [distance_norm, sin(bearing), cos(bearing)].
ENTITY_FEATURE_WIDTH = 3

#: "Forget immediately" stand-in: far away, no bearing -- the same neutral
#: shape ``core.streams.encoders.entity.EntityEncoder`` uses for "nothing
#: near", so the persistence baseline and the fixed encoder's blank state
#: agree on what "no information" looks like.
NEUTRAL_ENTITY_FEATURE: List[float] = [1.0, 0.0, 0.0]

DEFAULT_DISTANCE_RANGE: Tuple[float, float] = (0.0, 16.0)


def entity_feature_vector(
    entity: Mapping[str, Any], distance_range: Tuple[float, float] = DEFAULT_DISTANCE_RANGE
) -> List[float]:
    """``{distance, angle}`` -> ``[distance_norm, sin(bearing), cos(bearing)]``."""
    lo, hi = distance_range
    span = hi - lo
    distance = float(entity.get("distance", 0.0))
    distance_norm = 0.0 if span == 0 else min(1.0, max(0.0, (distance - lo) / span))
    bearing = math.radians(float(entity.get("angle", 0.0)))
    return [distance_norm, math.sin(bearing), math.cos(bearing)]


def entity_id(entity: Mapping[str, Any]) -> Any:
    """The stable identity key for one ``vision.entities`` record."""
    return entity.get("id")
