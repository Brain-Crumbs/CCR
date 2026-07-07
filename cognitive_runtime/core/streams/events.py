"""Stream event primitives.

Every sensory or motor input to the runtime is a time-indexed
:class:`StreamEvent` on a named stream ("body.health", "vision.frame.grid",
"motor.command").  Programs advertise the streams they publish with
:class:`StreamSpec`.  Nothing here knows about any specific Program.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

#: The generic sensory/motor taxonomy.  Minecraft health, Linux battery and
#: robot joint stress are all "body" streams; frames, desktop pixels and
#: cameras are all "vision" streams.  The brain consumes modalities, not
#: environment-specific fields.
MODALITIES = frozenset(
    {
        "body",
        "vision",
        "spatial",
        "audio",
        "event",
        "reward",
        "language",
        "input",
        "world",
        "motor",
    }
)

_STREAM_ID_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)*$")

#: Backpressure behaviors a bounded bus queue applies when a stream overflows.
#:  - ``drop_oldest``  ring buffer: keep the most-recent ``capacity`` events.
#:  - ``coalesce``     keep only the latest event (frames — the fresh one wins).
#:  - ``block``        never drop; the publisher waits for the consumer to drain.
OVERFLOW_POLICIES = frozenset({"drop_oldest", "coalesce", "block"})


def validate_stream_identity(stream_id: str, modality: str) -> None:
    """Validate a stream id / modality pair.

    Stream ids are lowercase dotted paths.  When the first path segment is
    itself a modality name (the recommended convention, e.g. "body.health"),
    it must match the declared modality.
    """
    if not _STREAM_ID_RE.match(stream_id):
        raise ValueError(
            f"invalid stream_id {stream_id!r}: must be a lowercase dotted path"
        )
    if modality not in MODALITIES:
        raise ValueError(
            f"unknown modality {modality!r} for stream {stream_id!r}; "
            f"expected one of {sorted(MODALITIES)}"
        )
    head = stream_id.split(".", 1)[0]
    if head in MODALITIES and head != modality:
        raise ValueError(
            f"stream {stream_id!r} starts with modality segment {head!r} "
            f"but declares modality {modality!r}"
        )


@dataclass(frozen=True)
class StreamEvent:
    """One time-indexed sample on a named stream.

    Two clocks live here, and only one of them is deterministic:

    - ``timestamp`` is SIMULATED time — the deterministic replay clock.
      Replay verification hashes include it, so wall-clock leakage would
      break replay.
    - ``arrived_at`` is the WALL-CLOCK instant the event reached the bus, in
      realtime mode (``None`` in fast-forward).  It is **metadata only**:
      excluded from :meth:`hash`, so replay and hashing never depend on it.
      It exists so runtime-health metrics can measure real-time cadence
      without contaminating determinism.
    """

    stream_id: str
    modality: str
    timestamp: float
    sequence_number: int  # per-stream monotonic, assigned by the publishing bus
    payload: Any  # JSON-serializable
    confidence: float = 1.0
    source: str = ""  # publishing program/backend name
    #: Wall-clock arrival instant (monotonic seconds), realtime mode only.
    #: Metadata only — never part of hash(), replay, or fast-forward logs.
    arrived_at: Optional[float] = None

    def __post_init__(self) -> None:
        validate_stream_identity(self.stream_id, self.modality)

    def hash(self) -> str:
        """Deterministic content hash; the replay-verification unit.

        Mirrors ``Observation.hash()``: canonical JSON with sorted keys over
        the replay-relevant fields.
        """
        payload = json.dumps(
            {
                "stream_id": self.stream_id,
                "sequence_number": self.sequence_number,
                "timestamp": self.timestamp,
                "payload": self.payload,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "modality": self.modality,
            "timestamp": self.timestamp,
            "sequence_number": self.sequence_number,
            "payload": self.payload,
            "confidence": self.confidence,
            "source": self.source,
            "arrived_at": self.arrived_at,
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "StreamEvent":
        return StreamEvent(
            stream_id=raw["stream_id"],
            modality=raw["modality"],
            timestamp=raw.get("timestamp", 0.0),
            sequence_number=raw.get("sequence_number", 0),
            payload=raw.get("payload"),
            confidence=raw.get("confidence", 1.0),
            source=raw.get("source", ""),
            arrived_at=raw.get("arrived_at"),
        )


@dataclass(frozen=True)
class StreamSpec:
    """A Program's advertisement of one stream it publishes.

    The optional encoder-facing metadata keeps the modality encoders
    (Phase 4) environment-agnostic: normalization ranges, grid class legends
    and neutral fill values arrive here from the Program instead of being
    hardcoded per world.
    """

    stream_id: str
    modality: str
    description: str = ""
    nominal_rate_hz: Optional[float] = None  # None = irregular / event-driven
    payload_schema: str = ""  # informal hint, e.g. "float 0..20" or "2-D int grid"
    #: (lo, hi) normalization range for scalar/coordinate payloads.
    range: Optional[Tuple[float, float]] = None
    #: grid cell-class id -> generic class name, for vision grid encoders.
    legend: Optional[Dict[int, str]] = None
    #: closed vocabulary for a categorical (string) payload, for one-hot encoders.
    categories: Optional[Tuple[str, ...]] = None
    #: value a fusion layout fills in when this stream is silent/missing.
    neutral: float = 0.0
    #: bounded-queue overflow behavior in realtime mode (see OVERFLOW_POLICIES).
    #: ``None`` = fall back to the bus's per-modality default.
    overflow: Optional[str] = None

    def __post_init__(self) -> None:
        validate_stream_identity(self.stream_id, self.modality)
        if self.overflow is not None and self.overflow not in OVERFLOW_POLICIES:
            raise ValueError(
                f"invalid overflow {self.overflow!r} for stream {self.stream_id!r}; "
                f"expected one of {sorted(OVERFLOW_POLICIES)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "modality": self.modality,
            "description": self.description,
            "nominal_rate_hz": self.nominal_rate_hz,
            "payload_schema": self.payload_schema,
            "range": list(self.range) if self.range is not None else None,
            "legend": (
                {str(k): v for k, v in self.legend.items()}
                if self.legend is not None
                else None
            ),
            "categories": list(self.categories) if self.categories is not None else None,
            "neutral": self.neutral,
            "overflow": self.overflow,
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "StreamSpec":
        rng = raw.get("range")
        legend = raw.get("legend")
        categories = raw.get("categories")
        return StreamSpec(
            stream_id=raw["stream_id"],
            modality=raw["modality"],
            description=raw.get("description", ""),
            nominal_rate_hz=raw.get("nominal_rate_hz"),
            payload_schema=raw.get("payload_schema", ""),
            range=tuple(rng) if rng is not None else None,
            legend=(
                {int(k): v for k, v in legend.items()} if legend is not None else None
            ),
            categories=tuple(categories) if categories is not None else None,
            neutral=raw.get("neutral", 0.0),
            overflow=raw.get("overflow"),
        )
