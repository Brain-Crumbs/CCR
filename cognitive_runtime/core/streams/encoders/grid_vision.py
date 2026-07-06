"""Grid vision encoder (``vision.frame.grid``).

Turns a small int-grid frame into a fixed vector:

- a per-cell-class histogram over the frame's legend classes,
- 4-quadrant pooled occupancy of the generic ``"solid"`` and ``"entity"``
  classes,
- a 3-bin ``"solid"`` profile across the center row.

Cell-id semantics come entirely from ``StreamSpec.legend`` (``{id: class}``),
so the encoder never names a single world block — ``"solid"`` and ``"entity"``
are generic class tags the Program assigns.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from cognitive_runtime.core.streams.encoder_registry import LatentToken, StreamEncoder
from cognitive_runtime.core.streams.events import StreamEvent, StreamSpec

SOLID_CLASS = "solid"
#: Generic classes pooled per quadrant (directional occupancy).  These are
#: generic scene tags a Program assigns via its legend, never world blocks.
POOLED_CLASSES = ("solid", "entity", "resource")
QUADRANT_FEATURES = 4 * len(POOLED_CLASSES)
CENTER_BINS = 3


def _classes(spec: Optional[StreamSpec]) -> List[str]:
    if spec is None or not spec.legend:
        raise ValueError("GridVisionEncoder requires StreamSpec.legend")
    return sorted(set(spec.legend.values()))


class GridVisionEncoder(StreamEncoder):
    def width(self, spec: Optional[StreamSpec] = None) -> int:
        return len(_classes(spec)) + QUADRANT_FEATURES + CENTER_BINS

    def encode(
        self, events: List[StreamEvent], spec: Optional[StreamSpec] = None
    ) -> Optional[LatentToken]:
        if not events or spec is None or not spec.legend:
            return None
        legend: Dict[int, str] = spec.legend
        classes = _classes(spec)
        class_index = {name: i for i, name in enumerate(classes)}
        grid = events[-1].payload
        rows = [row for row in grid if isinstance(row, list)] if isinstance(grid, list) else []
        cells = [c for row in rows for c in row]
        if not cells:
            return LatentToken(events[-1].stream_id, events[-1].modality,
                               events[-1].timestamp, [0.0] * self.width(spec))

        hist = [0.0] * len(classes)
        for cell in cells:
            name = legend.get(cell)
            if name in class_index:
                hist[class_index[name]] += 1.0
        hist = [h / len(cells) for h in hist]

        quad = self._quadrant_occupancy(rows, legend)
        center = self._center_profile(rows, legend)
        return LatentToken(
            stream_id=events[-1].stream_id,
            modality=events[-1].modality,
            timestamp=events[-1].timestamp,
            vector=hist + quad + center,
        )

    @staticmethod
    def _quadrant_occupancy(rows: List[list], legend: Dict[int, str]) -> List[float]:
        n_rows = len(rows)
        n_cols = max((len(r) for r in rows), default=0)
        midr, midc = n_rows / 2.0, n_cols / 2.0
        # Per-quadrant occupancy of each pooled class, order TL, TR, BL, BR.
        counts = {name: [0.0, 0.0, 0.0, 0.0] for name in POOLED_CLASSES}
        total = [0, 0, 0, 0]
        for ri, row in enumerate(rows):
            for ci, cell in enumerate(row):
                q = (0 if ri < midr else 2) + (0 if ci < midc else 1)
                total[q] += 1
                name = legend.get(cell)
                if name in counts:
                    counts[name][q] += 1.0
        out: List[float] = []
        for q in range(4):
            denom = total[q] or 1
            for name in POOLED_CLASSES:
                out.append(counts[name][q] / denom)
        return out

    @staticmethod
    def _center_profile(rows: List[list], legend: Dict[int, str]) -> List[float]:
        if not rows:
            return [0.0] * CENTER_BINS
        row = rows[len(rows) // 2]
        if not row:
            return [0.0] * CENTER_BINS
        bins = [0.0] * CENTER_BINS
        counts = [0] * CENTER_BINS
        for ci, cell in enumerate(row):
            b = min(ci * CENTER_BINS // len(row), CENTER_BINS - 1)
            counts[b] += 1
            if legend.get(cell) == SOLID_CLASS:
                bins[b] += 1.0
        return [bins[b] / (counts[b] or 1) for b in range(CENTER_BINS)]
