"""Temporal fusion: latent tokens -> a single fixed-width latent state.

`TemporalFusion` turns the per-stream encoder outputs plus recent history into
one flat vector with a **deterministic, versioned layout**: streams ordered by
id, a fixed slice per stream, silent streams filled with their spec's neutral
value.  MVP implementation is windowed concat-and-pool — the latest encoded
token per stream (mean/max pooling for scalar streams lives inside
`ScalarEncoder`), and a recency-weighted activation for event streams.

The layout is hashed (`LatentState.layout_hash`) so a trained model that saved
one layout fails loudly against an incompatible one instead of silently
mis-predicting.

Environment-agnostic: the fusion knows only modalities and generic
`StreamSpec` metadata, never world fields.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from cognitive_runtime.core.streams.encoder_registry import (
    LatentToken,
    StreamEncoder,
    StreamEncoderRegistry,
)
from cognitive_runtime.core.streams.encoders import (
    CategoryEncoder,
    EntityEncoder,
    EventEncoder,
    GridVisionEncoder,
    ScalarEncoder,
    SpatialEncoder,
)
from cognitive_runtime.core.streams.events import StreamSpec
from cognitive_runtime.core.streams.temporal_buffer import TemporalBuffer

FUSION_VERSION = "fusion-v1"


def default_encoder_registry() -> StreamEncoderRegistry:
    """Generic modality -> encoder mapping (first match wins)."""
    registry = StreamEncoderRegistry()
    registry.register("body.*", ScalarEncoder())
    registry.register("reward.*", ScalarEncoder())
    # Scalar spatial streams (a distance is one number, not a pose) must match
    # before the generic pose encoder; first registered match wins.
    registry.register("spatial.distance_from_spawn", ScalarEncoder())
    registry.register("spatial.*", SpatialEncoder())
    registry.register("vision.frame.grid", GridVisionEncoder())
    registry.register("vision.entities", EntityEncoder())
    registry.register("event.*", EventEncoder())
    # Generic categorical/scalar world state (vocab/range from the spec).
    registry.register("world.front_block", CategoryEncoder())
    registry.register("world.sheltered", ScalarEncoder())
    return registry


@dataclass(frozen=True)
class LatentState:
    """One fused cognitive-tick state: a flat vector + named per-stream slices."""

    vector: List[float]
    slices: Dict[str, Tuple[int, int]]
    layout_hash: str

    @property
    def width(self) -> int:
        return len(self.vector)

    def slice(self, stream_id: str) -> List[float]:
        lo, hi = self.slices[stream_id]
        return self.vector[lo:hi]


@dataclass(frozen=True)
class _LayoutEntry:
    stream_id: str
    spec: StreamSpec
    encoder: StreamEncoder
    modality: str
    width: int


class TemporalFusion:
    def __init__(
        self,
        catalog: List[StreamSpec],
        registry: Optional[StreamEncoderRegistry] = None,
        window: int = 8,
        half_life_seconds: float = 1.0,
    ):
        self.registry = registry or default_encoder_registry()
        self.window = window
        self.half_life_seconds = half_life_seconds
        self.layout: List[_LayoutEntry] = []
        for spec in sorted(catalog, key=lambda s: s.stream_id):
            encoder = self.registry.encoder_for(spec.stream_id)
            if encoder is None:
                continue
            self.layout.append(
                _LayoutEntry(
                    stream_id=spec.stream_id,
                    spec=spec,
                    encoder=encoder,
                    modality=spec.modality,
                    width=encoder.width(spec),
                )
            )
        self.width = sum(e.width for e in self.layout)
        self.layout_hash = self._compute_layout_hash()

    def _compute_layout_hash(self) -> str:
        items = [
            [
                e.stream_id,
                type(e.encoder).__name__,
                e.width,
                list(e.spec.range) if e.spec.range is not None else None,
                e.spec.neutral,
                sorted(set(e.spec.legend.values())) if e.spec.legend else None,
                list(e.spec.categories) if e.spec.categories else None,
            ]
            for e in self.layout
        ]
        blob = json.dumps([FUSION_VERSION, items], separators=(",", ":"), default=str)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def feature_names(self) -> List[str]:
        names: List[str] = []
        for e in self.layout:
            if e.width == 1:
                names.append(e.stream_id)
            else:
                names.extend(f"{e.stream_id}[{i}]" for i in range(e.width))
        return names

    def _reference_time(self, buffer: TemporalBuffer) -> float:
        ref = 0.0
        for e in self.layout:
            latest = buffer.latest(e.stream_id)
            if latest is not None:
                ref = max(ref, latest.timestamp)
        return ref

    def _event_recency(self, last_ts: float, reference_time: float) -> float:
        dt = max(reference_time - last_ts, 0.0)
        if self.half_life_seconds <= 0:
            return 1.0
        return 0.5 ** (dt / self.half_life_seconds)

    def fuse(
        self,
        tokens: Optional[List[LatentToken]],
        temporal_buffer: TemporalBuffer,
    ) -> LatentState:
        """Assemble the fixed-width latent vector from recent stream history.

        `tokens` (the current window's encodings) are advisory; the vector is
        rebuilt from `temporal_buffer` so a silent stream carries its last
        value and online/offline paths that feed identical buffers produce
        identical vectors.
        """
        reference_time = self._reference_time(temporal_buffer)
        vector: List[float] = []
        slices: Dict[str, Tuple[int, int]] = {}
        for entry in self.layout:
            events = temporal_buffer.window(entry.stream_id, self.window)
            if entry.modality == "event":
                recency = self._event_recency(events[-1].timestamp, reference_time) if events else 0.0
                vec = [recency]
            elif events:
                token = entry.encoder.encode(events, entry.spec)
                vec = token.vector if token is not None else entry.encoder.neutral(entry.spec)
            else:
                vec = entry.encoder.neutral(entry.spec)
            if len(vec) != entry.width:
                raise ValueError(
                    f"{entry.stream_id}: encoder produced width {len(vec)}, "
                    f"layout expects {entry.width}"
                )
            slices[entry.stream_id] = (len(vector), len(vector) + entry.width)
            vector.extend(vec)
        return LatentState(vector=vector, slices=slices, layout_hash=self.layout_hash)
