"""Action-space hashing (issue #42's versioning story).

Shared by session metadata (``runtime/loop.py``) and the neural checkpoint
bundle (``neural/checkpoint.py``).  Kept torch-free, unlike the rest of
``cognitive_runtime.neural``, so the pure-Python runtime loop can record and
compare action-space hashes without pulling in the optional ``neural``
extra.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from cognitive_runtime.core.hashing import canonical_json

ACTION_SPACE_HASH_VERSION = "action-space-v1"


def action_space_hash(action_keys: Sequence[str]) -> str:
    """Stable hash for the ordered action space a policy head was trained on."""

    blob = canonical_json([ACTION_SPACE_HASH_VERSION, list(action_keys)])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()
