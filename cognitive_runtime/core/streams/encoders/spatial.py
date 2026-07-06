"""Spatial stream encoder (``spatial.*``).

Handles both position (``{x, y, z}``) and rotation (``{yaw, pitch}``) payloads
with a single fixed 8-vector so the whole ``spatial`` modality has one width::

    [x_norm, y_norm, z_norm, dx, dz, yaw_sin, yaw_cos, pitch_norm]

Coordinate components are bounded-normalized by ``StreamSpec.range``;
displacement (dx, dz) is the range-scaled step since the previous event; angles
become sin/cos (yaw) and a ``/90`` normalization (pitch).  Components absent
from a given payload are filled with the neutral value.
"""

from __future__ import annotations

import math
from typing import List, Optional

from cognitive_runtime.core.streams.encoder_registry import LatentToken, StreamEncoder
from cognitive_runtime.core.streams.encoders.common import normalize, spec_range
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec

WIDTH = 8


def _coord(payload: object, key: str) -> Optional[float]:
    if isinstance(payload, dict) and isinstance(payload.get(key), (int, float)):
        return float(payload[key])
    return None


class SpatialEncoder(StreamEncoder):
    def width(self, spec: Optional[StreamSpec] = None) -> int:
        return WIDTH

    def encode(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[LatentToken]:
        if not events:
            return None
        rng = spec_range(spec)
        span = (rng[1] - rng[0]) if rng is not None else 1.0
        payload = events[-1].payload
        prev = events[-2].payload if len(events) >= 2 else payload

        x, y, z = (_coord(payload, k) for k in ("x", "y", "z"))
        px, pz = _coord(prev, "x"), _coord(prev, "z")
        dx = (x - px) / span if (x is not None and px is not None and span) else 0.0
        dz = (z - pz) / span if (z is not None and pz is not None and span) else 0.0

        yaw = _coord(payload, "yaw")
        pitch = _coord(payload, "pitch")
        vector = [
            normalize(x, rng) if x is not None else 0.0,
            normalize(y, rng) if y is not None else 0.0,
            normalize(z, rng) if z is not None else 0.0,
            dx,
            dz,
            math.sin(math.radians(yaw)) if yaw is not None else 0.0,
            math.cos(math.radians(yaw)) if yaw is not None else 0.0,
            (pitch / 90.0) if pitch is not None else 0.0,
        ]
        return LatentToken(
            stream_id=events[-1].stream_id,
            modality=events[-1].modality,
            timestamp=events[-1].timestamp,
            vector=vector,
        )
