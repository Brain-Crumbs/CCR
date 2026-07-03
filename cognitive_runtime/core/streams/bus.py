"""Deterministic in-process stream buses.

Programs publish sensory events onto a :class:`SensoryStreamBus`; the policy
publishes motor events onto a :class:`MotorStreamBus`.  Both are the same
thin pub/sub mechanism flowing in opposite directions.

Determinism contract: ``drain()`` always returns pending events sorted by
``(timestamp, stream_id, sequence_number)``.  Same publishes in ⇒ identical
order out, regardless of interleaving.  Replay depends on this.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from typing import Any, Dict, List, Optional

from cognitive_runtime.core.streams.events import (
    MODALITIES,
    StreamEvent,
    StreamSpec,
)


def stream_matches(pattern: str, stream_id: str) -> bool:
    """Glob-style stream filter: "body.*", "event.*", "*"."""
    return fnmatchcase(stream_id, pattern)


class StreamSubscription:
    """A consumer's filtered view of a bus.

    MVP: filters are applied at drain time and a single consumer (the
    runtime) is sufficient, but the handle keeps the API shape
    multi-consumer-ready.
    """

    def __init__(self, bus: "StreamBus", pattern: str):
        self._bus = bus
        self.pattern = pattern

    def matches(self, event: StreamEvent) -> bool:
        return stream_matches(self.pattern, event.stream_id)

    def drain(self) -> List[StreamEvent]:
        """Return and remove pending events matching this subscription."""
        return self._bus._drain_matching(self.pattern)


class StreamBus:
    """Thin, deterministic, in-process pub/sub bus."""

    def __init__(self) -> None:
        self._pending: List[StreamEvent] = []
        self._sequence_numbers: Dict[str, int] = {}
        self._specs: Dict[str, StreamSpec] = {}

    # -- publishing ---------------------------------------------------------

    def publish(
        self,
        stream_id: str,
        payload: Any,
        timestamp: float,
        confidence: float = 1.0,
        source: str = "",
    ) -> StreamEvent:
        """Append an event, assigning the next per-stream sequence number.

        The modality comes from the registered :class:`StreamSpec` when the
        stream is registered, otherwise from the id's first segment (which
        must then be a modality name).
        """
        event = StreamEvent(
            stream_id=stream_id,
            modality=self._modality_for(stream_id),
            timestamp=timestamp,
            sequence_number=self._sequence_numbers.get(stream_id, 0),
            payload=payload,
            confidence=confidence,
            source=source,
        )
        self._sequence_numbers[stream_id] = event.sequence_number + 1
        self._pending.append(event)
        return event

    def _modality_for(self, stream_id: str) -> str:
        spec = self._specs.get(stream_id)
        if spec is not None:
            return spec.modality
        head = stream_id.split(".", 1)[0]
        if head in MODALITIES:
            return head
        raise ValueError(
            f"cannot infer modality for unregistered stream {stream_id!r}; "
            "register a StreamSpec or start the id with a modality segment"
        )

    # -- consuming ----------------------------------------------------------

    def drain(self) -> List[StreamEvent]:
        """Return and clear all pending events in deterministic order."""
        return self._drain_matching("*")

    def _drain_matching(self, pattern: str) -> List[StreamEvent]:
        matched = [e for e in self._pending if stream_matches(pattern, e.stream_id)]
        self._pending = [
            e for e in self._pending if not stream_matches(pattern, e.stream_id)
        ]
        matched.sort(key=lambda e: (e.timestamp, e.stream_id, e.sequence_number))
        return matched

    def subscribe(self, pattern: str) -> StreamSubscription:
        return StreamSubscription(self, pattern)

    def pending_count(self) -> int:
        return len(self._pending)

    # -- catalog ------------------------------------------------------------

    def register(self, spec: StreamSpec) -> None:
        self._specs[spec.stream_id] = spec

    def catalog(self) -> List[StreamSpec]:
        return [self._specs[k] for k in sorted(self._specs)]

    def spec(self, stream_id: str) -> Optional[StreamSpec]:
        return self._specs.get(stream_id)

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Clear queues and sequence counters (called on episode reset).

        The catalog survives: what a Program *can* publish does not change
        between episodes.
        """
        self._pending.clear()
        self._sequence_numbers.clear()


class SensoryStreamBus(StreamBus):
    """Program → runtime: Programs publish, the runtime drains."""


class MotorStreamBus(StreamBus):
    """Runtime → program: the policy publishes ``motor.*``, the Program drains."""
