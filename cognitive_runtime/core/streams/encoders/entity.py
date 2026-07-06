"""Entity stream encoder (``vision.entities``).

Encodes a list of visible entities (``[{distance, angle}, ...]``) into a fixed
4-vector: ``[nearest_distance_norm, bearing_sin, bearing_cos, count_norm]``.
Distance is normalized by ``StreamSpec.range``; bearing (degrees, relative to
facing) becomes sin/cos; count is normalized by a small cap.  An empty list
encodes as "nothing near": far distance, zero bearing, zero count.
"""

from __future__ import annotations

import math
from typing import List, Optional

from cognitive_runtime.core.streams.encoder_registry import LatentToken, StreamEncoder
from cognitive_runtime.core.streams.encoders.common import normalize, spec_range
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec

WIDTH = 4
COUNT_CAP = 4.0


class EntityEncoder(StreamEncoder):
    def width(self, spec: Optional[StreamSpec] = None) -> int:
        return WIDTH

    def neutral(self, spec: Optional[StreamSpec] = None) -> List[float]:
        return [1.0, 0.0, 0.0, 0.0]  # far, no bearing, no entities

    def encode(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[LatentToken]:
        if not events:
            return None
        entities = events[-1].payload
        if not isinstance(entities, list) or not entities:
            vector = self.neutral(spec)
        else:
            nearest = min(
                entities,
                key=lambda e: e.get("distance", float("inf")) if isinstance(e, dict) else float("inf"),
            )
            distance = float(nearest.get("distance", 0.0)) if isinstance(nearest, dict) else 0.0
            bearing = math.radians(float(nearest.get("angle", 0.0))) if isinstance(nearest, dict) else 0.0
            vector = [
                normalize(distance, spec_range(spec)),
                math.sin(bearing),
                math.cos(bearing),
                min(len(entities), COUNT_CAP) / COUNT_CAP,
            ]
        return LatentToken(
            stream_id=events[-1].stream_id,
            modality=events[-1].modality,
            timestamp=events[-1].timestamp,
            vector=vector,
        )
