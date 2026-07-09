"""Deterministic content hashing shared across events, observations and memory.

Array-shaped payloads (pixel frames) must never pass through ``json.dumps``:
serializing a 128x128x3 frame to JSON means building ~49k Python ints and a
matching string just to hash it, on every tick.  :func:`hash_array` hashes
raw bytes plus dtype/shape instead.  Everything else keeps the exact
canonical-JSON digest this codebase has always used, so no non-frame stream's
hash value changes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_array(array: np.ndarray) -> str:
    """Content hash of an ndarray's raw bytes (dtype/shape included, no JSON)."""
    contiguous = np.ascontiguousarray(array)
    h = hashlib.sha1()
    h.update(f"ndarray|{contiguous.dtype}|{contiguous.shape}|".encode("utf-8"))
    h.update(contiguous.tobytes())
    return h.hexdigest()


def hash_payload(payload: Any) -> str:
    """SHA-1 hex digest of a single payload value (array-aware)."""
    if isinstance(payload, np.ndarray):
        return hash_array(payload)
    return hashlib.sha1(canonical_json(payload).encode("utf-8")).hexdigest()
