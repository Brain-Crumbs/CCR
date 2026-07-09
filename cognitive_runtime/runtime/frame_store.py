"""Binary frame store: rolling-window segments with pinning (streams-v2).

Frame pixel payloads are the bulk of a recording; instead of embedding them
inline in ``streams.jsonl`` (or round-tripping through JSON just to compute a
content hash), they live in a content-addressed binary sidecar under
``<session_dir>/frames/``::

    frames/
        segment_00000.bin           raw concatenated frame bytes
        segment_00000.index.jsonl   one line per frame: {hash, offset, length, shape, dtype}
        pinned_segments.json        {"pinned": ["segment_00000", ...]}

Segments rotate on a size or time threshold -- the rolling window.  Rotation
then reclaims disk by dropping the oldest *unpinned* segment(s) until the
store is back under its configured budget.  Pinned segments (deaths, damage,
hand-flagged checkpoints) are never dropped by rotation; only an explicit
unpin does that.  A write that hits a full disk evicts the oldest unpinned
segment and retries once; if that still doesn't fit (only pinned/open data
remains), the write is dropped and the caller degrades to hash-only for that
one event -- the store never raises out of ``write_frame``.

Reading a closed segment mmaps it once and views frames out of that buffer
with ``np.frombuffer`` (zero-copy); the still-open segment is read with a
plain seek+read since its length changes under the writer.
"""

from __future__ import annotations

import json
import mmap
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from cognitive_runtime.core.hashing import hash_array

#: Bumped only if the on-disk segment/index layout itself changes; the event
#: log's hash algorithm (see runtime.recorder) is versioned separately.
SEGMENT_PREFIX = "segment_"
FRAME_HASH_ALGORITHM = "content-bytes-v1"

DEFAULT_SEGMENT_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_SEGMENT_MAX_SECONDS = 60.0
DEFAULT_DISK_BUDGET_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class _FrameLocation:
    segment_id: str
    offset: int
    length: int
    shape: Tuple[int, ...]
    dtype: str


class FrameStore:
    """Content-addressed binary frame sidecar for one session."""

    def __init__(
        self,
        root: str,
        segment_max_bytes: int = DEFAULT_SEGMENT_MAX_BYTES,
        segment_max_seconds: float = DEFAULT_SEGMENT_MAX_SECONDS,
        disk_budget_bytes: Optional[int] = DEFAULT_DISK_BUDGET_BYTES,
    ) -> None:
        self.root = root
        self.segment_max_bytes = segment_max_bytes
        self.segment_max_seconds = segment_max_seconds
        self.disk_budget_bytes = disk_budget_bytes

        self._index: Dict[str, _FrameLocation] = {}
        self._segment_bytes: Dict[str, int] = {}
        self._segment_order: List[str] = []
        self._pinned: set = set()
        self._next_serial = 0

        self._current_id: Optional[str] = None
        self._current_fh = None
        self._current_index_fh = None
        self._current_started_at: float = 0.0
        self._mmaps: Dict[str, mmap.mmap] = {}

        if os.path.isdir(root):
            self._load_existing()

    # -- discovery -----------------------------------------------------------

    def _load_existing(self) -> None:
        pinned_path = os.path.join(self.root, "pinned_segments.json")
        if os.path.exists(pinned_path):
            with open(pinned_path, encoding="utf-8") as fh:
                self._pinned = set(json.load(fh).get("pinned", []))
        for name in sorted(os.listdir(self.root)):
            if not (name.startswith(SEGMENT_PREFIX) and name.endswith(".bin")):
                continue
            segment_id = name[: -len(".bin")]
            self._segment_order.append(segment_id)
            serial = int(segment_id[len(SEGMENT_PREFIX):])
            self._next_serial = max(self._next_serial, serial + 1)
            self._segment_bytes[segment_id] = os.path.getsize(os.path.join(self.root, name))
            index_path = os.path.join(self.root, f"{segment_id}.index.jsonl")
            if not os.path.exists(index_path):
                continue
            with open(index_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._index[rec["hash"]] = _FrameLocation(
                        segment_id, rec["offset"], rec["length"],
                        tuple(rec["shape"]), rec["dtype"],
                    )

    # -- writing ---------------------------------------------------------------

    def write_frame(self, array: np.ndarray) -> Optional[str]:
        """Persist ``array`` (deduped by content), returning its hash.

        Never raises: a write that can't be made to fit even after evicting
        the oldest unpinned rolling segments returns ``None`` so the caller
        can degrade that one event to hash-only instead of crashing the loop.
        """
        contiguous = np.ascontiguousarray(array)
        frame_hash = hash_array(contiguous)
        if frame_hash in self._index:
            return frame_hash
        payload = contiguous.tobytes()

        for attempt in range(2):
            segment_id = self._ensure_current()
            try:
                offset = self._current_fh.tell()
                self._current_fh.write(payload)
                self._current_fh.flush()
            except OSError:
                self._close_current(delete_if_empty=True)
                if attempt == 0 and self._evict_oldest_unpinned():
                    continue
                return None
            location = _FrameLocation(
                segment_id, offset, len(payload), contiguous.shape, str(contiguous.dtype)
            )
            self._index[frame_hash] = location
            self._segment_bytes[segment_id] = self._segment_bytes.get(segment_id, 0) + len(payload)
            record = {
                "hash": frame_hash, "offset": offset, "length": len(payload),
                "shape": list(contiguous.shape), "dtype": str(contiguous.dtype),
            }
            self._current_index_fh.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._current_index_fh.flush()
            self._maybe_rotate()
            return frame_hash
        return None

    def _ensure_current(self) -> str:
        if self._current_id is None:
            self._open_new_segment()
        assert self._current_id is not None
        return self._current_id

    def _open_new_segment(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        segment_id = f"{SEGMENT_PREFIX}{self._next_serial:05d}"
        self._next_serial += 1
        self._current_fh = open(os.path.join(self.root, f"{segment_id}.bin"), "ab")
        self._current_index_fh = open(
            os.path.join(self.root, f"{segment_id}.index.jsonl"), "a", encoding="utf-8"
        )
        self._current_id = segment_id
        self._current_started_at = time.monotonic()
        self._segment_order.append(segment_id)
        self._segment_bytes.setdefault(segment_id, 0)

    # -- rotation / retention ---------------------------------------------------

    def _maybe_rotate(self) -> None:
        if self._current_id is None:
            return
        elapsed = time.monotonic() - self._current_started_at
        if (
            self._segment_bytes[self._current_id] >= self.segment_max_bytes
            or elapsed >= self.segment_max_seconds
        ):
            self.rotate()

    def rotate(self) -> None:
        """Close the current segment (a new one opens lazily on the next
        write) and enforce the disk budget over unpinned rolling segments."""
        self._close_current()
        self._enforce_budget()

    def _close_current(self, delete_if_empty: bool = False) -> None:
        segment_id = self._current_id
        if segment_id is None:
            return
        if self._current_fh is not None:
            self._current_fh.close()
        if self._current_index_fh is not None:
            self._current_index_fh.close()
        if delete_if_empty and self._segment_bytes.get(segment_id, 0) == 0:
            self._remove_segment_files(segment_id)
            if segment_id in self._segment_order:
                self._segment_order.remove(segment_id)
            self._segment_bytes.pop(segment_id, None)
        self._current_id = None
        self._current_fh = None
        self._current_index_fh = None

    def _enforce_budget(self) -> None:
        if self.disk_budget_bytes is None:
            return
        while self.total_bytes > self.disk_budget_bytes:
            if not self._evict_oldest_unpinned():
                break  # only pinned/open data remains; can't reclaim further

    def _evict_oldest_unpinned(self) -> bool:
        victim = self._oldest_unpinned_closed_segment()
        if victim is None:
            return False
        self._evict_segment(victim)
        return True

    def _oldest_unpinned_closed_segment(self) -> Optional[str]:
        for segment_id in self._segment_order:
            if segment_id == self._current_id or segment_id in self._pinned:
                continue
            return segment_id
        return None

    def _evict_segment(self, segment_id: str) -> None:
        mm = self._mmaps.pop(segment_id, None)
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                pass  # an outstanding zero-copy view; released once GC'd
        self._remove_segment_files(segment_id)
        if segment_id in self._segment_order:
            self._segment_order.remove(segment_id)
        self._segment_bytes.pop(segment_id, None)
        for frame_hash in [h for h, loc in self._index.items() if loc.segment_id == segment_id]:
            del self._index[frame_hash]

    def _remove_segment_files(self, segment_id: str) -> None:
        for suffix in (".bin", ".index.jsonl"):
            path = os.path.join(self.root, f"{segment_id}{suffix}")
            if os.path.exists(path):
                os.remove(path)

    # -- pinning -----------------------------------------------------------------

    def pin_current(self) -> Optional[str]:
        """Pin whichever segment is presently open (or was last written) so
        rotation never drops it.  Returns the pinned segment id, if any."""
        segment_id = self._current_id or (self._segment_order[-1] if self._segment_order else None)
        if segment_id is None:
            return None
        self.pin_segment(segment_id)
        return segment_id

    def pin_segment(self, segment_id: str) -> None:
        if segment_id in self._pinned:
            return
        self._pinned.add(segment_id)
        self._save_pinned()

    def unpin_segment(self, segment_id: str) -> None:
        if segment_id not in self._pinned:
            return
        self._pinned.discard(segment_id)
        self._save_pinned()

    def _save_pinned(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, "pinned_segments.json"), "w", encoding="utf-8") as fh:
            json.dump({"pinned": sorted(self._pinned)}, fh)

    # -- reading -------------------------------------------------------------------

    def read_frame(self, frame_hash: str) -> np.ndarray:
        """Look up ``frame_hash`` and return its array.  A closed segment is
        mmapped and viewed zero-copy; the open segment is read directly."""
        location = self._index.get(frame_hash)
        if location is None:
            raise KeyError(f"frame {frame_hash!r} not found in frame store {self.root!r}")
        if location.segment_id == self._current_id:
            self._current_fh.flush()
            with open(os.path.join(self.root, f"{location.segment_id}.bin"), "rb") as fh:
                fh.seek(location.offset)
                raw = fh.read(location.length)
            return np.frombuffer(raw, dtype=location.dtype).reshape(location.shape)
        mm = self._mmap_for(location.segment_id)
        itemsize = np.dtype(location.dtype).itemsize
        count = location.length // itemsize
        array = np.frombuffer(mm, dtype=location.dtype, count=count, offset=location.offset)
        return array.reshape(location.shape)

    def _mmap_for(self, segment_id: str) -> mmap.mmap:
        mm = self._mmaps.get(segment_id)
        if mm is not None:
            return mm
        path = os.path.join(self.root, f"{segment_id}.bin")
        fh = open(path, "rb")
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        fh.close()
        self._mmaps[segment_id] = mm
        return mm

    def has_frame(self, frame_hash: str) -> bool:
        return frame_hash in self._index

    # -- introspection ---------------------------------------------------------------

    @property
    def total_bytes(self) -> int:
        return sum(self._segment_bytes.values())

    @property
    def pinned_segments(self) -> List[str]:
        return sorted(self._pinned)

    @property
    def rolling_segments(self) -> List[str]:
        return [s for s in self._segment_order if s not in self._pinned]

    @property
    def segments(self) -> List[str]:
        return list(self._segment_order)

    def close(self) -> None:
        self._close_current()
        for mm in self._mmaps.values():
            try:
                mm.close()
            except BufferError:
                # A caller still holds a zero-copy view (np.frombuffer) into
                # this segment; the mapping is released once that view is
                # garbage collected instead.
                pass
        self._mmaps.clear()


def open_frame_store(session_dir: str, **kwargs) -> Optional[FrameStore]:
    """Open the frame store for a recorded session, or ``None`` if it never
    used binary frame storage (frames elided, or a legacy inline-payload
    session that predates this store entirely)."""
    root = os.path.join(session_dir, "frames")
    if not os.path.isdir(root):
        return None
    return FrameStore(root, **kwargs)
