"""Generic observation representation.

An observation is a timestamped bundle of structured data plus an optional
"frame" (a 2-D grid standing in for screen pixels).  The runtime never
interprets the contents; only Programs, program-specific reward modules and
policies do.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from cognitive_runtime.core.hashing import canonical_json


@dataclass
class Observation:
    timestamp: float
    tick: int
    data: Dict[str, Any] = field(default_factory=dict)
    frame: Optional[List[List[int]]] = None
    #: Optional true RGB pixel frame (H x W x 3, 0..255) -- the neural-vision
    #: counterpart to the coarse ``frame`` grid.  ``None`` when a Program does
    #: not render pixels.
    pixels: Optional[np.ndarray] = None

    def hash(self) -> str:
        """Deterministic content hash used for replay verification and novelty.

        ``pixels`` hashes its raw bytes when it's an ndarray, so a real frame
        never gets serialized element-by-element to JSON just to be hashed.
        """
        if isinstance(self.pixels, np.ndarray):
            h = hashlib.sha1()
            h.update(
                canonical_json(
                    {"tick": self.tick, "data": self.data, "frame": self.frame}
                ).encode("utf-8")
            )
            contiguous = np.ascontiguousarray(self.pixels)
            h.update(f"|pixels|{contiguous.dtype}|{contiguous.shape}|".encode("utf-8"))
            h.update(contiguous.tobytes())
            return h.hexdigest()
        payload = canonical_json(
            {"tick": self.tick, "data": self.data, "frame": self.frame,
             "pixels": self.pixels}
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def to_dict(self, include_frame: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "tick": self.tick,
            "data": self.data,
        }
        if include_frame:
            out["frame"] = self.frame
            out["pixels"] = self.pixels
        return out

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "Observation":
        pixels = raw.get("pixels")
        if pixels is not None and not isinstance(pixels, np.ndarray):
            pixels = np.asarray(pixels, dtype=np.uint8)
        return Observation(
            timestamp=raw.get("timestamp", 0.0),
            tick=raw.get("tick", 0),
            data=raw.get("data", {}),
            frame=raw.get("frame"),
            pixels=pixels,
        )
