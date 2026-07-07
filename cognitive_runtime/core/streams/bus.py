"""Deterministic in-process stream buses.

Programs publish sensory events onto a :class:`SensoryStreamBus`; the policy
publishes motor events onto a :class:`MotorStreamBus`.  Both are the same
thin pub/sub mechanism flowing in opposite directions.

Determinism contract: ``drain()`` always returns pending events sorted by
``(timestamp, stream_id, sequence_number)``.  Same publishes in ⇒ identical
order out, regardless of interleaving.  Replay depends on this.

Two operating modes share one implementation:

- **Simulated / fast-forward (the default).** Single-threaded and
  *lock-free-deterministic*: no locks, no wall clock, and the per-stream
  queues are drained every cognitive tick so their bounds are never reached.
  Byte-identical across repeats.
- **Realtime (opt in via ``thread_safe=True``).** A real backend (a
  mineflayer bridge, a screen-capture thread, a terminal reader) can publish
  from its own thread while the cognitive loop drains on the main thread.
  Publishes are serialized under a lock; bounded per-stream queues apply a
  declared overflow policy so a fast publisher can never grow memory without
  bound, and every drop/coalesce is counted, never silent.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import nullcontext
from fnmatch import fnmatchcase
from typing import Any, Callable, Deque, Dict, List, Optional

from cognitive_runtime.core.streams.events import (
    MODALITIES,
    StreamEvent,
    StreamSpec,
)

#: Per-stream queue bound in realtime mode.  Generous enough that the
#: single-threaded fast-forward path (drained every tick) never reaches it,
#: so overflow handling stays inert there and ordering is unchanged.
DEFAULT_QUEUE_CAPACITY = 1024

#: Overflow policy chosen when a StreamSpec does not declare one, by modality.
#: Vision frames and body vitals want the freshest value (coalesce); discrete
#: events must never be dropped (block); everything else rings.
_MODALITY_OVERFLOW_DEFAULT: Dict[str, str] = {
    "vision": "coalesce",
    "body": "coalesce",
    "event": "block",
}
_DEFAULT_OVERFLOW = "drop_oldest"

#: How long a ``block``-policy publisher waits for the consumer before giving
#: up and appending anyway (never dropping) — a safety valve against deadlock.
_BLOCK_WAIT_SECONDS = 0.25


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
    """Thin, deterministic, in-process pub/sub bus.

    ``thread_safe`` turns on a lock + a condition variable so publishers on
    other threads are safe against the draining consumer; it is off by
    default so the simulated path pays nothing.  ``wall_clock`` (a
    ``() -> float`` callable, realtime only) stamps ``StreamEvent.arrived_at``
    metadata without ever touching the deterministic simulated timestamp.
    """

    def __init__(
        self,
        thread_safe: bool = False,
        default_capacity: int = DEFAULT_QUEUE_CAPACITY,
        wall_clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._queues: Dict[str, Deque[StreamEvent]] = {}
        self._sequence_numbers: Dict[str, int] = {}
        self._specs: Dict[str, StreamSpec] = {}
        self.default_capacity = default_capacity
        self._capacity_overrides: Dict[str, int] = {}
        #: stream_id -> {policy -> count of events dropped/coalesced}
        self._overflow_counts: Dict[str, Dict[str, int]] = {}
        self._thread_safe = thread_safe
        self._wall_clock = wall_clock
        self._cond = threading.Condition() if thread_safe else None

    # -- concurrency helpers -----------------------------------------------

    def _guard(self):
        """Context manager guarding shared state; a no-op when single-threaded."""
        return self._cond if self._cond is not None else nullcontext()

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
        must then be a modality name).  Thread-safe when ``thread_safe`` is
        set; the bounded queue applies the stream's overflow policy.
        """
        with self._guard():
            event = StreamEvent(
                stream_id=stream_id,
                modality=self._modality_for(stream_id),
                timestamp=timestamp,
                sequence_number=self._sequence_numbers.get(stream_id, 0),
                payload=payload,
                confidence=confidence,
                source=source,
                arrived_at=self._wall_clock() if self._wall_clock is not None else None,
            )
            self._sequence_numbers[stream_id] = event.sequence_number + 1
            self._enqueue(event)
            return event

    def _enqueue(self, event: StreamEvent) -> None:
        """Append under the stream's overflow policy.  Caller holds the guard."""
        stream_id = event.stream_id
        queue = self._queues.setdefault(stream_id, deque())
        policy = self._overflow(stream_id)
        capacity = self._capacity(stream_id)

        if policy == "coalesce":
            # Bounded, and on overflow collapse to the single freshest event:
            # an un-drained frame is stale the moment a newer one exists.
            queue.append(event)
            if len(queue) > capacity:
                self._count_overflow(stream_id, "coalesce", len(queue) - 1)
                latest = queue[-1]
                queue.clear()
                queue.append(latest)
            return

        if policy == "block" and self._cond is not None:
            # Never drop: wait for the consumer to make room, then append.  A
            # bounded wait is a safety valve so a stalled consumer cannot
            # deadlock the publisher forever (it appends past the bound as a
            # last resort — correctness over the bound, because events must
            # never be lost).
            deadline = time.monotonic() + _BLOCK_WAIT_SECONDS
            while len(queue) >= capacity:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)
            queue.append(event)
            return

        if policy == "block":
            # Lock-free path (single-threaded fast-forward): nothing to wait
            # on, and events must never drop.  The consumer drains every tick,
            # so the queue cannot actually grow without bound here.
            queue.append(event)
            return

        # drop_oldest: bounded ring — keep the most-recent ``capacity`` events.
        queue.append(event)
        while len(queue) > capacity:
            queue.popleft()
            self._count_overflow(stream_id, "drop_oldest", 1)

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

    # -- overflow policy ----------------------------------------------------

    def _overflow(self, stream_id: str) -> str:
        spec = self._specs.get(stream_id)
        if spec is not None and spec.overflow:
            return spec.overflow
        return _MODALITY_OVERFLOW_DEFAULT.get(
            self._modality_for(stream_id), _DEFAULT_OVERFLOW
        )

    def _capacity(self, stream_id: str) -> int:
        return self._capacity_overrides.get(stream_id, self.default_capacity)

    def set_capacity(self, stream_id: str, capacity: int) -> None:
        """Override the bounded-queue capacity for one stream (realtime tuning)."""
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity_overrides[stream_id] = capacity

    def _count_overflow(self, stream_id: str, policy: str, n: int) -> None:
        counts = self._overflow_counts.setdefault(stream_id, {})
        counts[policy] = counts.get(policy, 0) + n

    def overflow_counts(self) -> Dict[str, Dict[str, int]]:
        """Per-stream overflow tallies since the last reset ({stream: {policy: n}})."""
        with self._guard():
            return {sid: dict(c) for sid, c in self._overflow_counts.items()}

    def total_overflows(self) -> int:
        return sum(
            sum(c.values()) for c in self.overflow_counts().values()
        )

    # -- consuming ----------------------------------------------------------

    def drain(self) -> List[StreamEvent]:
        """Return and clear all pending events in deterministic order."""
        return self._drain_matching("*")

    def _drain_matching(self, pattern: str) -> List[StreamEvent]:
        with self._guard():
            matched: List[StreamEvent] = []
            for stream_id, queue in self._queues.items():
                if stream_matches(pattern, stream_id) and queue:
                    matched.extend(queue)
                    queue.clear()
            matched.sort(key=lambda e: (e.timestamp, e.stream_id, e.sequence_number))
            if self._cond is not None:
                self._cond.notify_all()  # wake any blocked publishers
            return matched

    def subscribe(self, pattern: str) -> StreamSubscription:
        return StreamSubscription(self, pattern)

    def pending_count(self) -> int:
        with self._guard():
            return sum(len(q) for q in self._queues.values())

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
        with self._guard():
            self._queues.clear()
            self._sequence_numbers.clear()
            self._overflow_counts.clear()
            if self._cond is not None:
                self._cond.notify_all()


class SensoryStreamBus(StreamBus):
    """Program → runtime: Programs publish, the runtime drains."""


class MotorStreamBus(StreamBus):
    """Runtime → program: the policy publishes ``motor.*``, the Program drains."""
