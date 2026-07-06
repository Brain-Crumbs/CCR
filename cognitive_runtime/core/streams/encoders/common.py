"""Shared numeric helpers for the modality encoders.

Everything here is environment-agnostic: normalization ranges and neutral
values come from :class:`StreamSpec` metadata, never from world constants.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from cognitive_runtime.core.streams.events import StreamSpec


def clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def normalize(value: float, rng: Optional[Tuple[float, float]]) -> float:
    """Map `value` into [0, 1] using `(lo, hi)`; identity when no range given."""
    if rng is None:
        return float(value)
    lo, hi = rng
    span = hi - lo
    if span == 0:
        return 0.0
    return clamp01((float(value) - lo) / span)


def spec_range(spec: Optional[StreamSpec]) -> Optional[Tuple[float, float]]:
    return spec.range if spec is not None else None


def scalar_leaf(payload: Any) -> Optional[float]:
    """Extract a single scalar from a payload.

    Numbers pass through; a dict yields its ``"value"`` when numeric, else its
    first numeric leaf in sorted-key order — enough for both bare scalars
    (``body.health``) and wrapped ones (``reward.scalar`` = ``{value, ...}``).
    """
    if isinstance(payload, bool):
        return 1.0 if payload else 0.0
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, dict):
        value = payload.get("value")
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)
        for key in sorted(payload):
            leaf = scalar_leaf(payload[key])
            if leaf is not None:
                return leaf
    return None
