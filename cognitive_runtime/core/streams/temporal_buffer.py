"""Bounded per-stream event history.

The stream-native counterpart of ``core/memory.py``: instead of a window of
whole states, keep a bounded deque of recent events per stream.  Capacity is
configurable per modality — vision buffers can be short, event buffers long.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

from cognitive_runtime.core.streams.events import StreamEvent


class TemporalBuffer:
    def __init__(
        self,
        default_capacity: int = 256,
        capacity_by_modality: Optional[Dict[str, int]] = None,
    ):
        self.default_capacity = default_capacity
        self.capacity_by_modality = dict(capacity_by_modality or {})
        self._buffers: Dict[str, Deque[StreamEvent]] = {}

    def capacity_for(self, modality: str) -> int:
        return self.capacity_by_modality.get(modality, self.default_capacity)

    def append(self, event: StreamEvent) -> None:
        buffer = self._buffers.get(event.stream_id)
        if buffer is None:
            buffer = deque(maxlen=self.capacity_for(event.modality))
            self._buffers[event.stream_id] = buffer
        buffer.append(event)

    def extend(self, events: List[StreamEvent]) -> None:
        for event in events:
            self.append(event)

    def latest(self, stream_id: str) -> Optional[StreamEvent]:
        buffer = self._buffers.get(stream_id)
        return buffer[-1] if buffer else None

    def window(self, stream_id: str, n: int) -> List[StreamEvent]:
        """The most recent `n` events of a stream, oldest first."""
        buffer = self._buffers.get(stream_id)
        if not buffer:
            return []
        return list(buffer)[-n:]

    def events_since(self, timestamp: float) -> List[StreamEvent]:
        """All buffered events strictly after `timestamp`, across streams,
        in deterministic ``(timestamp, stream_id, sequence_number)`` order."""
        out = [
            event
            for buffer in self._buffers.values()
            for event in buffer
            if event.timestamp > timestamp
        ]
        out.sort(key=lambda e: (e.timestamp, e.stream_id, e.sequence_number))
        return out

    def streams(self) -> List[str]:
        return sorted(self._buffers)

    def reset(self) -> None:
        self._buffers.clear()
