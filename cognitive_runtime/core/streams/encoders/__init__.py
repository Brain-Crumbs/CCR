"""Per-modality stream encoders (Phase 4).

Each turns one stream's recent window into a fixed-width latent vector, driven
only by generic :class:`StreamSpec` metadata (ranges, legends, neutrals) — no
environment-specific constants live here.
"""

from cognitive_runtime.core.streams.encoders.category import CategoryEncoder
from cognitive_runtime.core.streams.encoders.entity import EntityEncoder
from cognitive_runtime.core.streams.encoders.event import EventEncoder
from cognitive_runtime.core.streams.encoders.grid_vision import GridVisionEncoder
from cognitive_runtime.core.streams.encoders.scalar import ScalarEncoder
from cognitive_runtime.core.streams.encoders.spatial import SpatialEncoder

__all__ = [
    "ScalarEncoder",
    "SpatialEncoder",
    "GridVisionEncoder",
    "EventEncoder",
    "EntityEncoder",
    "CategoryEncoder",
]
