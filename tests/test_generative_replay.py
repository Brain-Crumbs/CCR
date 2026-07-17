"""Generative replay + bootstrap guardrail (issue #99): the dream fraction
is gated on measured quality, capped, and dreaming never touches the real
reservoir or draws a dream-only batch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from cognitive_runtime.neural.replay_buffer import (  # noqa: E402
    ReplayBuffer,
    ReplayBufferConfig,
    Transition,
)
from sleep.replay import (  # noqa: E402
    DreamFractionGate,
    dreamed_batch_size,
    sample_generative_replay_batch,
)


def _transition(i: int, width: int = 4) -> Transition:
    return Transition(
        latent=[float(i)] * width, action=i % 2, reward=float(i),
        next_latent=[float(i) + 1] * width, done=False,
    )


def _real_buffer(n: int, width: int = 4) -> ReplayBuffer:
    buffer = ReplayBuffer(ReplayBufferConfig(capacity=max(n, 1)))
    for i in range(n):
        buffer.add(_transition(i, width))
    return buffer


# --------------------------------------------------------------- the gate itself


def test_dream_fraction_is_zero_below_the_quality_bar():
    gate = DreamFractionGate(margin=0.9, floor_ratio=0.4, cap=0.5)
    assert gate.fraction(1.0) == 0.0
    assert gate.fraction(0.9) == 0.0  # at the margin, not yet past it


def test_dream_fraction_ramps_between_margin_and_floor():
    gate = DreamFractionGate(margin=0.9, floor_ratio=0.4, cap=0.5)
    midpoint_ratio = (0.9 + 0.4) / 2
    assert gate.fraction(midpoint_ratio) == pytest.approx(0.25)
    # Monotonic: a better (lower) ratio never dreams less.
    better_ratio = 0.5
    worse_ratio = 0.8
    assert gate.fraction(better_ratio) > gate.fraction(worse_ratio)


def test_dream_fraction_caps_at_and_below_the_floor():
    gate = DreamFractionGate(margin=0.9, floor_ratio=0.4, cap=0.5)
    assert gate.fraction(0.4) == 0.5
    assert gate.fraction(0.0) == 0.5  # a near-perfect model still caps


def test_gate_rejects_a_floor_at_or_above_its_margin():
    with pytest.raises(ValueError, match="floor_ratio"):
        DreamFractionGate(margin=0.5, floor_ratio=0.5)


def test_gate_rejects_a_cap_outside_zero_one():
    with pytest.raises(ValueError, match="cap"):
        DreamFractionGate(cap=0.0)
    with pytest.raises(ValueError, match="cap"):
        DreamFractionGate(cap=1.5)


# ----------------------------------------------------------- batch-level guardrail


def test_dreamed_batch_size_never_fills_an_entire_batch():
    # Even a dream_fraction of 1.0 (e.g. a badly misconfigured gate) must
    # leave at least one real transition -- "never train on dreams alone"
    # enforced at the batch level, not only via DreamFractionGate.cap.
    assert dreamed_batch_size(8, 1.0) == 7
    assert dreamed_batch_size(1, 1.0) == 0
    assert dreamed_batch_size(8, 0.0) == 0
    assert dreamed_batch_size(8, 0.5) == 4


def test_sample_generative_replay_batch_never_draws_dream_only_and_retains_reservoir():
    real_buffer = _real_buffer(20)
    dream_calls = []

    def dream_source(n):
        dream_calls.append(n)
        return [_transition(1000 + i) for i in range(n)]

    gate = DreamFractionGate(margin=0.9, floor_ratio=0.4, cap=0.5)
    batch, n_dream = sample_generative_replay_batch(
        real_buffer, dream_source, batch_size=10, n_actions=2,
        quality_ratio=0.0,  # a maximally "good" model -> the gate's cap
        gate=gate,
    )
    assert n_dream == dreamed_batch_size(10, gate.cap)
    assert n_dream < 10  # never dream-only
    assert dream_calls == [n_dream]
    assert batch["fused_latent"].shape[0] == 10

    # Reservoir untouched: sampling doesn't drain it, and dreamed
    # transitions are never written into it.
    assert len(real_buffer) == 20
    assert all(t.latent[0] < 1000 for t in real_buffer.transitions())


def test_sample_generative_replay_batch_draws_no_dreams_below_the_quality_bar():
    real_buffer = _real_buffer(20)

    def dream_source(n):
        raise AssertionError("must not be called when the dream fraction is 0")

    gate = DreamFractionGate(margin=0.9, floor_ratio=0.4, cap=0.5)
    batch, n_dream = sample_generative_replay_batch(
        real_buffer, dream_source, batch_size=10, n_actions=2,
        quality_ratio=1.0, gate=gate,  # worse than copy-last -> gate closed
    )
    assert n_dream == 0
    assert batch["fused_latent"].shape[0] == 10


def test_sample_generative_replay_batch_ramps_dream_count_with_quality():
    real_buffer = _real_buffer(20)
    gate = DreamFractionGate(margin=0.9, floor_ratio=0.4, cap=0.5)

    def dream_source(n):
        return [_transition(1000 + i) for i in range(n)]

    _batch_bad, n_dream_bad = sample_generative_replay_batch(
        real_buffer, dream_source, batch_size=10, n_actions=2,
        quality_ratio=0.9, gate=gate,
    )
    _batch_mid, n_dream_mid = sample_generative_replay_batch(
        real_buffer, dream_source, batch_size=10, n_actions=2,
        quality_ratio=0.65, gate=gate,
    )
    _batch_good, n_dream_good = sample_generative_replay_batch(
        real_buffer, dream_source, batch_size=10, n_actions=2,
        quality_ratio=0.4, gate=gate,
    )
    assert n_dream_bad == 0
    assert 0 < n_dream_mid < n_dream_good
    assert len(real_buffer) == 20


def test_sample_generative_replay_batch_rejects_a_dream_source_returning_the_wrong_count():
    real_buffer = _real_buffer(20)

    def bad_dream_source(n):
        return [_transition(1000 + i) for i in range(n + 1)]

    with pytest.raises(ValueError, match="requested"):
        sample_generative_replay_batch(
            real_buffer, bad_dream_source, batch_size=10, n_actions=2,
            quality_ratio=0.0, gate=DreamFractionGate(),
        )
