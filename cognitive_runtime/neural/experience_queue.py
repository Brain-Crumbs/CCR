"""Bounded shared-memory live-experience queue for the actor/learner split
(issue #37, "Async actor/learner split: background trainer over live
experience + recorded sessions").

The realtime loop (the *actor*) must never block on training: it calls
:meth:`SharedExperienceRing.push` once per cognitive tick and that call is
O(1) and non-blocking, full stop.  A separate *trainer* process calls
:meth:`SharedExperienceRing.drain` on its own schedule to pull batches of
:class:`~cognitive_runtime.neural.replay_buffer.Transition` out of the ring.

Backpressure policy is drop-oldest: once the ring is full, ``push`` silently
overwrites the oldest not-yet-drained row instead of blocking or raising --
"explicit backpressure policy (drop-oldest, never block the actor)" per the
issue.  A trainer that falls behind (or has crashed) simply misses older
transitions; it never stalls the actor.

Rows hold fused-latent vectors (small float arrays already computed by the
fusion pipeline, per issue #28) directly in a shared-memory ``float32``
block -- "frames by shared tensor/mmap, not pickled copies" -- rather than
going through a ``multiprocessing.Queue``, which would pickle/copy every
transition through a pipe.  ``source`` (a transition's provenance string,
used by :func:`~cognitive_runtime.neural.replay_buffer.load_session_into_buffer`
for recorded sessions) is not carried through the ring; drained transitions
are labeled ``"live"``.

Only the small numpy-backed row block is shared memory; the write cursor,
drop counter and add counter live in ``multiprocessing.Value``s guarded by a
single ``multiprocessing.Lock`` so a push and a drain can never observe a
half-written row.  Both are held only long enough to copy one row (push) or
memcpy a slice (drain), so contention never approaches tick-budget
timescales.
"""

from __future__ import annotations

import math
import multiprocessing
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional

import numpy as np

from cognitive_runtime.neural.replay_buffer import Transition

#: Always "spawn" (never the platform-default "fork") so a ring's
#: `Lock`/`Value` are safe to hand to a child process regardless of what
#: threads (torch/OpenMP workers, a test runner's own threads, ...) happen
#: to be alive in the parent at the moment a child is spawned -- fork only
#: clones the calling thread, so any lock held by another parent thread at
#: fork time stays locked forever in the child.  Every process that talks
#: to a `SharedExperienceRing` (see `training.async_trainer`) must launch
#: its children the same way for this to line up.
MP_CONTEXT = multiprocessing.get_context("spawn")

#: Row layout within the shared float32 block, beyond the two
#: ``latent_dim``-wide latent vectors: action, reward, done, damage,
#: novelty, prediction_error.  ``novelty``/``prediction_error`` use NaN as
#: the "unavailable" sentinel (mirrors ``Transition.novelty``/
#: ``prediction_error`` being ``Optional[float]``).
_SCALAR_FIELDS = ("action", "reward", "done", "damage", "novelty", "prediction_error")
_N_SCALARS = len(_SCALAR_FIELDS)


def _row_width(latent_dim: int) -> int:
    return 2 * latent_dim + _N_SCALARS


@dataclass(frozen=True)
class ExperienceRingStats:
    capacity: int
    latent_dim: int
    total_pushed: int
    total_dropped: int
    size: int


class SharedExperienceRing:
    """Bounded ring of transitions in shared memory, safe to hand to a child
    process as a plain constructor argument (``multiprocessing.Process(...,
    args=(ring,))``): the underlying shared-memory block is looked up by
    name rather than pickled, and ``Lock``/``Value`` use the standard
    multiprocessing reduction that ``Process`` already relies on.

    Construct once in the parent (owning) process; call :meth:`close` in
    every process done with it and :meth:`unlink` exactly once (normally
    from the owner) when the ring is no longer needed by anyone.
    """

    def __init__(self, capacity: int, latent_dim: int, *, name: Optional[str] = None):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity!r}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim!r}")
        self.capacity = capacity
        self.latent_dim = latent_dim
        self._row_width = _row_width(latent_dim)
        self._owns_shm = name is None
        nbytes = capacity * self._row_width * np.dtype(np.float32).itemsize
        if self._owns_shm:
            self._shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
        else:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
        self.name = self._shm.name
        self._array = np.ndarray(
            (capacity, self._row_width), dtype=np.float32, buffer=self._shm.buf
        )
        if self._owns_shm:
            self._array[:] = 0.0
            self._write_index = MP_CONTEXT.Value("q", 0)
            self._total_pushed = MP_CONTEXT.Value("q", 0)
            self._total_dropped = MP_CONTEXT.Value("q", 0)
            self._drain_index = MP_CONTEXT.Value("q", 0)
            self._lock = MP_CONTEXT.Lock()
        # ``attach``-constructed rings fill these in via __setstate__/attach();
        # the branch above only runs for the owning constructor.

    # ------------------------------------------------------------- attach

    @classmethod
    def attach(
        cls,
        name: str,
        capacity: int,
        latent_dim: int,
        write_index: Any,
        total_pushed: Any,
        total_dropped: Any,
        drain_index: Any,
        lock: Any,
    ) -> "SharedExperienceRing":
        """Reattach to an existing ring's shared memory from another process
        using the synchronization primitives handed to that process at
        ``Process(args=...)`` time (see :meth:`handle`)."""
        ring = cls(capacity, latent_dim, name=name)
        ring._write_index = write_index
        ring._total_pushed = total_pushed
        ring._total_dropped = total_dropped
        ring._drain_index = drain_index
        ring._lock = lock
        return ring

    def handle(self) -> Dict[str, Any]:
        """Everything a child process needs to :meth:`attach` to this same
        ring -- pass this dict (or its unpacked fields) as a ``Process``
        argument."""
        return {
            "name": self.name,
            "capacity": self.capacity,
            "latent_dim": self.latent_dim,
            "write_index": self._write_index,
            "total_pushed": self._total_pushed,
            "total_dropped": self._total_dropped,
            "drain_index": self._drain_index,
            "lock": self._lock,
        }

    # ------------------------------------------------------------- actor side

    def push(self, transition: Transition) -> None:
        """Write one transition; never blocks beyond a brief row-copy lock.

        Drop-oldest: once ``capacity`` rows have been written, the next push
        overwrites the oldest slot the drain cursor hasn't consumed yet, and
        ``total_dropped`` increments -- the actor is never made to wait for
        (or fail because of) a slow or dead trainer.
        """
        if len(transition.latent) != self.latent_dim or len(transition.next_latent) != self.latent_dim:
            raise ValueError(
                f"transition latent width {len(transition.latent)}/"
                f"{len(transition.next_latent)} does not match ring latent_dim "
                f"{self.latent_dim}"
            )
        row = np.empty(self._row_width, dtype=np.float32)
        row[: self.latent_dim] = transition.latent
        row[self.latent_dim : 2 * self.latent_dim] = transition.next_latent
        row[2 * self.latent_dim + 0] = float(transition.action)
        row[2 * self.latent_dim + 1] = transition.reward
        row[2 * self.latent_dim + 2] = 1.0 if transition.done else 0.0
        row[2 * self.latent_dim + 3] = 1.0 if transition.damage else 0.0
        row[2 * self.latent_dim + 4] = (
            transition.novelty if transition.novelty is not None else math.nan
        )
        row[2 * self.latent_dim + 5] = (
            transition.prediction_error
            if transition.prediction_error is not None
            else math.nan
        )
        with self._lock:
            index = self._write_index.value % self.capacity
            self._array[index, :] = row
            was_full = self._write_index.value >= self.capacity
            self._write_index.value += 1
            self._total_pushed.value += 1
            if was_full:
                # The slot just overwritten was the oldest undrained row (or
                # already-drained, either way it's gone now): advance the
                # drain cursor past it if the trainer hadn't reached it yet,
                # and count the drop only when it hadn't.
                if self._drain_index.value <= self._write_index.value - self.capacity - 1:
                    self._drain_index.value = self._write_index.value - self.capacity
                    self._total_dropped.value += 1

    # ------------------------------------------------------------ trainer side

    def drain(self, max_items: Optional[int] = None) -> List[Transition]:
        """Pop every row written since the last drain (oldest-write order),
        capped at ``max_items``.  Safe to call from a single trainer
        process/thread; concurrent drainers would race on ``drain_index``."""
        with self._lock:
            write_index = self._write_index.value
            drain_index = self._drain_index.value
            available = min(write_index - drain_index, self.capacity)
            if available <= 0:
                return []
            take = available if max_items is None else min(available, max_items)
            rows = np.empty((take, self._row_width), dtype=np.float32)
            for offset in range(take):
                rows[offset] = self._array[(drain_index + offset) % self.capacity]
            self._drain_index.value = drain_index + take
        return [self._row_to_transition(rows[i]) for i in range(take)]

    def _row_to_transition(self, row: np.ndarray) -> Transition:
        d = self.latent_dim
        novelty = float(row[2 * d + 4])
        prediction_error = float(row[2 * d + 5])
        return Transition(
            latent=row[:d].tolist(),
            action=int(round(float(row[2 * d + 0]))),
            reward=float(row[2 * d + 1]),
            next_latent=row[d : 2 * d].tolist(),
            done=bool(row[2 * d + 2]),
            damage=bool(row[2 * d + 3]),
            novelty=None if math.isnan(novelty) else novelty,
            prediction_error=None if math.isnan(prediction_error) else prediction_error,
            source="live",
        )

    # ------------------------------------------------------------------ misc

    def stats(self) -> ExperienceRingStats:
        with self._lock:
            write_index = self._write_index.value
            drain_index = self._drain_index.value
            return ExperienceRingStats(
                capacity=self.capacity,
                latent_dim=self.latent_dim,
                total_pushed=self._total_pushed.value,
                total_dropped=self._total_dropped.value,
                size=min(write_index - drain_index, self.capacity),
            )

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()
