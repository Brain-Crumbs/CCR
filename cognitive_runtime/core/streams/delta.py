"""On-change publishing helper.

Streams are multi-rate by design: most state streams should publish only
when their value changes (plus an optional heartbeat).  `DeltaPublisher`
keeps the on-change detection in one place by remembering the last
published payload per stream.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from cognitive_runtime.core.streams.bus import StreamBus
from cognitive_runtime.core.streams.events import StreamEvent


class DeltaPublisher:
    def __init__(self, bus: StreamBus):
        self.bus = bus
        self._last_payload: Dict[str, Any] = {}

    def publish(
        self,
        stream_id: str,
        payload: Any,
        timestamp: float,
        force: bool = False,
        confidence: float = 1.0,
        source: str = "",
    ) -> Optional[StreamEvent]:
        """Publish unless the payload equals the last published one.

        A stream's first publication always goes out (so subscribers never
        start blind).  `force=True` bypasses the comparison — used for
        every-tick streams and heartbeats.
        """
        if (
            not force
            and stream_id in self._last_payload
            and self._last_payload[stream_id] == payload
        ):
            return None
        self._last_payload[stream_id] = payload
        return self.bus.publish(
            stream_id, payload, timestamp, confidence=confidence, source=source
        )

    def reset(self) -> None:
        self._last_payload.clear()
