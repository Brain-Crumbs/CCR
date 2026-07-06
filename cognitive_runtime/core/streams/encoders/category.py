"""Categorical stream encoder (generic string payloads, e.g. ``world.*``).

One-hots a string payload against the closed vocabulary declared in
``StreamSpec.categories``, plus a trailing "unknown / other" slot.  The
vocabulary is published by the Program, so the encoder never spells out a
single world category itself.
"""

from __future__ import annotations

from typing import List, Optional

from cognitive_runtime.core.streams.encoder_registry import LatentToken, StreamEncoder
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec


def _categories(spec: Optional[StreamSpec]) -> List[str]:
    if spec is None or not spec.categories:
        raise ValueError("CategoryEncoder requires StreamSpec.categories")
    return list(spec.categories)


class CategoryEncoder(StreamEncoder):
    def width(self, spec: Optional[StreamSpec] = None) -> int:
        return len(_categories(spec)) + 1  # +1 for the "other" bucket

    def encode(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[LatentToken]:
        if not events or spec is None or not spec.categories:
            return None
        categories = _categories(spec)
        index = {name: i for i, name in enumerate(categories)}
        vector = [0.0] * (len(categories) + 1)
        payload = events[-1].payload
        vector[index.get(payload, len(categories))] = 1.0
        return LatentToken(
            stream_id=events[-1].stream_id,
            modality=events[-1].modality,
            timestamp=events[-1].timestamp,
            vector=vector,
        )
