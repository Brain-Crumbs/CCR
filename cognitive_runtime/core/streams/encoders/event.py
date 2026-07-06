"""Event stream encoder (``event.*``).

Each ``event.*`` stream contributes one slot; a window that carries the event
encodes to ``[1.0]``.  The multi-hot over the episode's event vocabulary and
the recency weighting emerge in :class:`~cognitive_runtime.core.streams.fusion`
(one slot per event stream in the layout, decayed by time since the last
occurrence), so the encoder itself stays a trivial, generic per-stream mark.
"""

from __future__ import annotations

from typing import List, Optional

from cognitive_runtime.core.streams.encoder_registry import LatentToken, StreamEncoder
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec


class EventEncoder(StreamEncoder):
    def width(self, spec: Optional[StreamSpec] = None) -> int:
        return 1

    def encode(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[LatentToken]:
        if not events:
            return None
        return LatentToken(
            stream_id=events[-1].stream_id,
            modality=events[-1].modality,
            timestamp=events[-1].timestamp,
            vector=[1.0],
        )
