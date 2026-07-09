"""Binary frame store: content addressing, rotation, pinning, disk budget."""

import os

import numpy as np
import pytest

from cognitive_runtime.runtime.frame_store import FrameStore, open_frame_store


def _frame(fill: int, shape=(4, 4, 3)):
    return np.full(shape, fill, dtype=np.uint8)


def test_write_read_round_trips_and_dedups(tmp_path):
    store = FrameStore(str(tmp_path / "frames"))
    h1 = store.write_frame(_frame(1))
    h2 = store.write_frame(_frame(1))  # identical content
    h3 = store.write_frame(_frame(2))
    assert h1 == h2, "identical frame content must dedup to the same hash"
    assert h1 != h3
    assert np.array_equal(store.read_frame(h1), _frame(1))
    assert np.array_equal(store.read_frame(h3), _frame(2))
    store.close()


def test_read_is_zero_copy_view_into_closed_segment(tmp_path):
    store = FrameStore(str(tmp_path / "frames"), segment_max_bytes=1)  # rotate every write
    h1 = store.write_frame(_frame(7))
    assert store._current_id is None, "the segment must already be rotated closed"
    array = store.read_frame(h1)
    assert array.base is not None, "a closed-segment read must be a view, not a copy"
    store.close()


def test_rotation_by_size_creates_new_segment(tmp_path):
    store = FrameStore(str(tmp_path / "frames"), segment_max_bytes=32)  # one 4x4x3 frame = 48 bytes
    store.write_frame(_frame(1))
    store.write_frame(_frame(2))
    assert len(store.segments) >= 2
    store.close()


def test_disk_budget_drops_oldest_unpinned_segment(tmp_path):
    frame_bytes = 4 * 4 * 3  # 48 bytes/frame
    store = FrameStore(
        str(tmp_path / "frames"),
        segment_max_bytes=frame_bytes,  # one frame per segment
        disk_budget_bytes=frame_bytes * 2,  # room for ~2 segments
    )
    hashes = [store.write_frame(_frame(i)) for i in range(5)]
    assert store.total_bytes <= frame_bytes * 2 + frame_bytes  # budget honored (+ the open segment)
    assert len(store.segments) < 5, "old unpinned segments must be evicted on rotation"
    # The oldest frame's segment should be gone; its content is no longer readable.
    with pytest.raises(KeyError):
        store.read_frame(hashes[0])
    # The most recent frame must survive.
    assert np.array_equal(store.read_frame(hashes[-1]), _frame(4))
    store.close()


def test_pinned_segment_survives_budget_eviction(tmp_path):
    frame_bytes = 4 * 4 * 3
    store = FrameStore(
        str(tmp_path / "frames"),
        segment_max_bytes=frame_bytes,
        disk_budget_bytes=frame_bytes,  # room for ~1 segment
    )
    h0 = store.write_frame(_frame(0))
    store.pin_current()  # pin the segment h0 landed in
    for i in range(1, 6):
        store.write_frame(_frame(i))
    assert np.array_equal(store.read_frame(h0), _frame(0)), "pinned segment must survive rotation"
    store.close()


def test_pinned_segments_persist_across_reopen(tmp_path):
    root = str(tmp_path / "frames")
    store = FrameStore(root)
    h0 = store.write_frame(_frame(9))
    pinned_id = store.pin_current()
    store.close()

    reopened = FrameStore(root)
    assert pinned_id in reopened.pinned_segments
    assert np.array_equal(reopened.read_frame(h0), _frame(9))
    reopened.close()


class _FullDiskFile:
    """A file-like stand-in whose ``write`` always raises ENOSPC-style OSError."""

    def tell(self) -> int:
        return 0

    def write(self, _data) -> None:
        raise OSError("No space left on device")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_write_frame_degrades_gracefully_on_persistent_disk_failure(tmp_path, monkeypatch):
    store = FrameStore(str(tmp_path / "frames"))
    store._ensure_current()
    store._current_fh = _FullDiskFile()
    monkeypatch.setattr(store, "_evict_oldest_unpinned", lambda: False)

    result = store.write_frame(_frame(3))
    assert result is None, "a write that can't be persisted must return None, not raise"
    store.close()


def test_open_frame_store_returns_none_when_absent(tmp_path):
    assert open_frame_store(str(tmp_path)) is None
    os.makedirs(str(tmp_path / "frames"))
    assert open_frame_store(str(tmp_path)) is not None
