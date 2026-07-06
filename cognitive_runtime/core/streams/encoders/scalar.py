"""Scalar stream encoder (``body.*`` / ``reward.*``).

Encodes a numeric stream's recent window into a fixed 4-vector:
``[normalized_latest, trend, normalized_mean, normalized_max]``.  Trend is the
step change of the normalized value from the previous event; mean/max pool the
window.  Normalization ranges come from ``StreamSpec.range``.
"""

from __future__ import annotations

from typing import List, Optional

from cognitive_runtime.core.streams.encoder_registry import LatentToken, StreamEncoder
from cognitive_runtime.core.streams.encoders.common import (
    normalize,
    scalar_leaf,
    spec_range,
)
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec

WIDTH = 4


class ScalarEncoder(StreamEncoder):
    def width(self, spec: Optional[StreamSpec] = None) -> int:
        return WIDTH

    def neutral(self, spec: Optional[StreamSpec] = None) -> List[float]:
        n = normalize(spec.neutral if spec is not None else 0.0, spec_range(spec))
        return [n, 0.0, n, n]

    def encode(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[LatentToken]:
        rng = spec_range(spec)
        values = [
            normalize(v, rng)
            for v in (scalar_leaf(e.payload) for e in events)
            if v is not None
        ]
        if not values:
            return None
        latest = values[-1]
        prev = values[-2] if len(values) >= 2 else latest
        vector = [latest, latest - prev, sum(values) / len(values), max(values)]
        return LatentToken(
            stream_id=events[-1].stream_id,
            modality=events[-1].modality,
            timestamp=events[-1].timestamp,
            vector=vector,
        )
