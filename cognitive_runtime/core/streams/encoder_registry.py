"""Stream encoder registry (Phase-0 skeleton).

Encoders turn a window of stream events into latent tokens the cognitive
core can fuse.  Phase 0 ships only the interface and a numeric passthrough
encoder so the registry is testable; real modality encoders are Phase 4.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from cognitive_runtime.core.streams.bus import stream_matches
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec
from cognitive_runtime.core.streams.synchronizer import TickWindow


@dataclass(frozen=True)
class LatentToken:
    """One encoded stream sample: what the cognitive core actually consumes."""

    stream_id: str
    modality: str
    timestamp: float
    vector: List[float]


class StreamEncoder(abc.ABC):
    """Encodes the recent events of one stream into a fixed-width latent token.

    Fixed width is what lets a fusion layout reserve a stable slice per stream
    and fill it with :meth:`neutral` when the stream is silent.  Modality
    encoders (Phase 4) implement :meth:`width` and :meth:`neutral`; the
    Phase-0 passthrough encoder leaves them unimplemented (variable width) and
    is only used outside a fusion layout.
    """

    @abc.abstractmethod
    def encode(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[LatentToken]:
        """Return a latent token for the window, or None if nothing usable."""

    def width(self, spec: Optional[StreamSpec] = None) -> int:
        """Fixed length of this encoder's vector for `spec`."""
        raise NotImplementedError(f"{type(self).__name__} has no fixed width")

    def neutral(self, spec: Optional[StreamSpec] = None) -> List[float]:
        """The vector used when the stream is silent (default: zeros)."""
        return [0.0] * self.width(spec)


def _numeric_leaves(payload: Any) -> List[float]:
    """Flatten numeric leaves of a JSON-like payload, deterministically
    (dict keys sorted).  Non-numeric leaves are skipped."""
    if isinstance(payload, bool):
        return [1.0 if payload else 0.0]
    if isinstance(payload, (int, float)):
        return [float(payload)]
    if isinstance(payload, (list, tuple)):
        out: List[float] = []
        for item in payload:
            out.extend(_numeric_leaves(item))
        return out
    if isinstance(payload, dict):
        out = []
        for key in sorted(payload):
            out.extend(_numeric_leaves(payload[key]))
        return out
    return []


class PassthroughEncoder(StreamEncoder):
    """Flattens the latest event's numeric payload into the vector as-is."""

    def encode(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[LatentToken]:
        if not events:
            return None
        latest = events[-1]
        vector = _numeric_leaves(latest.payload)
        if not vector:
            return None
        return LatentToken(
            stream_id=latest.stream_id,
            modality=latest.modality,
            timestamp=latest.timestamp,
            vector=vector,
        )


class StreamEncoderRegistry:
    """Maps stream-id glob patterns to encoders; first registered match wins."""

    def __init__(self) -> None:
        self._encoders: List[Tuple[str, StreamEncoder]] = []

    def register(self, pattern: str, encoder: StreamEncoder) -> None:
        self._encoders.append((pattern, encoder))

    def encoder_for(self, stream_id: str) -> Optional[StreamEncoder]:
        for pattern, encoder in self._encoders:
            if stream_matches(pattern, stream_id):
                return encoder
        return None

    def encode_window(self, window: TickWindow) -> List[LatentToken]:
        """Encode every stream in the window that has a registered encoder,
        in deterministic (sorted stream_id) order."""
        tokens: List[LatentToken] = []
        for stream_id in sorted(window.by_stream):
            encoder = self.encoder_for(stream_id)
            if encoder is None:
                continue
            token = encoder.encode(window.by_stream[stream_id])
            if token is not None:
                tokens.append(token)
        return tokens
