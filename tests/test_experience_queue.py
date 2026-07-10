"""Shared-memory live-experience ring buffer (issue #37): bounded capacity,
drop-oldest backpressure, and cross-process push/drain."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402  (Transition lives under cognitive_runtime.neural)

from cognitive_runtime.neural.experience_queue import (  # noqa: E402
    MP_CONTEXT,
    SharedExperienceRing,
)
from cognitive_runtime.neural.replay_buffer import Transition  # noqa: E402


def _transition(i: int, latent_dim: int = 2) -> Transition:
    return Transition(
        latent=[float(i)] * latent_dim,
        action=i % 3,
        reward=float(i),
        next_latent=[float(i) + 1.0] * latent_dim,
        done=(i % 7 == 0),
    )


def test_drain_returns_pushed_transitions_in_order():
    ring = SharedExperienceRing(capacity=8, latent_dim=2)
    try:
        for i in range(5):
            ring.push(_transition(i))
        drained = ring.drain()
        assert [t.reward for t in drained] == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert [t.action for t in drained] == [0, 1, 2, 0, 1]
        assert drained[0].source == "live"
        # A second drain with nothing new pushed returns nothing.
        assert ring.drain() == []
    finally:
        ring.close()
        ring.unlink()


def test_never_exceeds_capacity_and_drops_oldest_when_full():
    ring = SharedExperienceRing(capacity=4, latent_dim=2)
    try:
        for i in range(10):  # capacity 4, no draining in between -> heavy overwrite
            ring.push(_transition(i))
        stats = ring.stats()
        assert stats.total_pushed == 10
        assert stats.total_dropped == 6
        assert stats.size == 4

        drained = ring.drain()
        # Only the newest `capacity` transitions survive, oldest-first.
        assert [t.reward for t in drained] == [6.0, 7.0, 8.0, 9.0]
    finally:
        ring.close()
        ring.unlink()


def test_partial_drain_leaves_the_rest_for_next_call():
    ring = SharedExperienceRing(capacity=10, latent_dim=2)
    try:
        for i in range(6):
            ring.push(_transition(i))
        first = ring.drain(max_items=4)
        assert [t.reward for t in first] == [0.0, 1.0, 2.0, 3.0]
        second = ring.drain()
        assert [t.reward for t in second] == [4.0, 5.0]
    finally:
        ring.close()
        ring.unlink()


def test_optional_fields_round_trip_through_nan_sentinel():
    ring = SharedExperienceRing(capacity=4, latent_dim=2)
    try:
        ring.push(Transition(
            latent=[0.0, 0.0], action=1, reward=1.0, next_latent=[1.0, 1.0],
            done=True, damage=True, novelty=0.5, prediction_error=0.25,
        ))
        ring.push(Transition(
            latent=[0.0, 0.0], action=1, reward=1.0, next_latent=[1.0, 1.0],
            done=False, novelty=None, prediction_error=None,
        ))
        with_signals, without_signals = ring.drain()
        assert with_signals.novelty == pytest.approx(0.5)
        assert with_signals.prediction_error == pytest.approx(0.25)
        assert with_signals.done is True
        assert with_signals.damage is True
        assert without_signals.novelty is None
        assert without_signals.prediction_error is None
        assert without_signals.done is False
    finally:
        ring.close()
        ring.unlink()


def test_push_rejects_mismatched_latent_width():
    ring = SharedExperienceRing(capacity=4, latent_dim=3)
    try:
        with pytest.raises(ValueError):
            ring.push(_transition(0, latent_dim=2))
    finally:
        ring.close()
        ring.unlink()


def _push_in_subprocess(handle, n, latent_dim):
    ring = SharedExperienceRing.attach(**handle)
    for i in range(n):
        ring.push(_transition(i, latent_dim=latent_dim))
    ring.close()


def test_push_from_a_real_subprocess_is_visible_to_the_parent():
    """The actor and trainer are different OS processes (issue #37: "separate
    process, not thread"); this exercises the actual `handle()`/`attach()`
    cross-process protocol, not just in-process calls."""
    ring = SharedExperienceRing(capacity=1000, latent_dim=2)
    try:
        process = MP_CONTEXT.Process(target=_push_in_subprocess, args=(ring.handle(), 300, 2))
        process.start()
        process.join(timeout=30)
        assert process.exitcode == 0

        drained = ring.drain()
        assert len(drained) == 300
        assert [t.reward for t in drained] == [float(i) for i in range(300)]
        assert ring.stats().total_dropped == 0
    finally:
        ring.close()
        ring.unlink()


def test_push_never_blocks_when_full_even_with_no_drainer():
    """Backpressure policy is drop-oldest, not block (issue #37: "explicit
    backpressure policy (drop-oldest, never block the actor)")."""
    import time

    ring = SharedExperienceRing(capacity=16, latent_dim=2)
    try:
        start = time.monotonic()
        for i in range(5000):  # far beyond capacity, never drained
            ring.push(_transition(i))
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"push should stay O(1); took {elapsed:.2f}s for 5000 pushes"
        assert ring.stats().size == 16
    finally:
        ring.close()
        ring.unlink()
