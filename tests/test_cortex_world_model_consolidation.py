"""Online-consolidation wiring on :class:`CortexWorldModel` (issue #175):
the CLI's own online learner used to be the actor/critic RL stack; this
repoints ``--async-trainer`` at the predictive cortex instead. Covers:

- ``predict()`` feeds a live ``sleep.cortex_consolidation.CortexConsolidator``
  (or any duck-typed double) exactly the real transition that just completed
  -- last tick's latent, the action taken since, and this tick's actually-
  observed latent -- via ``record_transition``, on every tick after the
  first (there is no "last tick" to pair on the very first one);
- a consolidation-then-publish pass fires exactly every
  ``consolidate_every_ticks`` ticks and not otherwise;
- with no consolidator (the default), behavior is unchanged from before this
  wiring existed -- byte-for-byte identical predictions to an instance built
  without any of the new constructor arguments.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brain.cortex.predictive import PredictiveCortex, PredictiveCortexConfig  # noqa: E402
from cognitive_runtime.core.action import Action  # noqa: E402
from cognitive_runtime.core.memory import Memory  # noqa: E402
from cognitive_runtime.core.perception import State  # noqa: E402
from cognitive_runtime.core.streams.events import StreamEvent  # noqa: E402
from cognitive_runtime.neural.pixel_stream_encoder import PIXEL_STREAM_ID  # noqa: E402
from cognitive_runtime.policies.cortex_world_model import CortexWorldModel  # noqa: E402

_ACTION_KEYS = ["noop", "move_forward", "turn_left", "turn_right"]


class FakeConsolidator:
    """Records calls instead of doing real gradient work -- enough to check
    the wiring's call pattern without needing a real
    ``sleep.cortex_consolidation.CortexConsolidator``."""

    def __init__(self):
        self.recorded = []  # list of (z0, actions, next_latents)
        self.consolidate_calls = []  # list of `steps` args
        self.publish_calls = 0

    def record_transition(self, z0, actions, next_latents):
        self.recorded.append((list(z0), list(actions), next_latents.clone()))

    def consolidate(self, steps):
        self.consolidate_calls.append(steps)
        return len(self.consolidate_calls)

    def publish_to(self, world_model):
        self.publish_calls += 1
        return self.publish_calls


def _small_cortex(pixel_shape=(8, 8, 3), horizons=(1, 4)) -> PredictiveCortex:
    torch.manual_seed(0)
    cfg = PredictiveCortexConfig(
        latent_width=8, hidden_dim=16, reconstruction_size=8, horizons_ticks=horizons
    )
    return PredictiveCortex(pixel_shape, _ACTION_KEYS, cfg)


def _push_frame(memory: Memory, frame: np.ndarray, seq: int) -> None:
    memory.buffer.extend(
        [
            StreamEvent(
                stream_id=PIXEL_STREAM_ID,
                modality="vision",
                timestamp=float(seq),
                sequence_number=seq,
                payload=frame,
            )
        ]
    )


def _frame(rng: np.random.Generator, shape=(8, 8, 3)) -> np.ndarray:
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _run_ticks(wm: CortexWorldModel, n: int, seed: int = 1):
    """Feeds `n` predict() ticks of distinct random frames/actions, returning
    the list of frames used (for cross-checking encoded latents)."""
    memory = Memory()
    rng = np.random.default_rng(seed)
    state = State(observation=None)
    frames = []
    action_cycle = ["noop", "move_forward", "turn_left"]
    for seq in range(n):
        if seq > 0:
            memory.record_action(Action.from_key(action_cycle[(seq - 1) % len(action_cycle)]))
        frame = _frame(rng)
        frames.append(frame)
        _push_frame(memory, frame, seq)
        wm.predict(state, memory)
    return frames


def test_record_transition_pairs_previous_latent_action_and_this_ticks_latent():
    cortex = _small_cortex()
    fake = FakeConsolidator()
    wm = CortexWorldModel(cortex, action_keys=_ACTION_KEYS, consolidator=fake)

    frames = _run_ticks(wm, 4)

    # No prior latent on the first tick -- nothing to pair, so no call yet.
    assert len(fake.recorded) == 3

    action_cycle = ["noop", "move_forward", "turn_left"]
    cortex.eval()
    with torch.no_grad():
        expected_latents = [
            cortex.encoder.encode_frame(frame).unsqueeze(0) for frame in frames
        ]

    for i, (z0, actions, next_latents) in enumerate(fake.recorded):
        # Tick i+1's record pairs tick i's latent (z0) with tick i+1's latent
        # (next_latents) and the action emitted between them.
        assert torch.allclose(torch.tensor([z0]), expected_latents[i], atol=1e-5)
        assert actions == [action_cycle[i % len(action_cycle)]]
        assert torch.allclose(next_latents, expected_latents[i + 1], atol=1e-5)


def test_no_consolidator_means_no_record_transition_calls():
    cortex = _small_cortex()
    wm = CortexWorldModel(cortex, action_keys=_ACTION_KEYS)  # consolidator=None default
    assert wm.consolidator is None
    _run_ticks(wm, 4)  # must not raise; nothing to assert on a fake here


def test_consolidation_fires_exactly_every_n_ticks():
    cortex = _small_cortex()
    fake = FakeConsolidator()
    wm = CortexWorldModel(
        cortex, action_keys=_ACTION_KEYS, consolidator=fake,
        consolidate_every_ticks=3, consolidation_steps=2,
    )

    _run_ticks(wm, 7)

    # Ticks 1..7; fires on ticks 3 and 6 (7 % 3 != 0, so not a third time).
    assert fake.consolidate_calls == [2, 2]
    assert fake.publish_calls == 2


def test_consolidation_disabled_by_default_even_with_a_consolidator_set():
    """`consolidate_every_ticks=0` (the default) disables consolidation even
    when a consolidator is attached -- e.g. a caller that wires
    `world_model.consolidator = consolidator` post-construction without also
    setting a cadence."""
    cortex = _small_cortex()
    fake = FakeConsolidator()
    wm = CortexWorldModel(cortex, action_keys=_ACTION_KEYS, consolidator=fake)
    assert wm.consolidate_every_ticks == 0

    _run_ticks(wm, 10)

    assert fake.consolidate_calls == []
    assert fake.publish_calls == 0


def test_consolidator_none_default_is_byte_for_byte_unchanged():
    """Constructing with none of the new issue #175 arguments predicts
    identically to explicitly passing their defaults -- no regression for
    existing callers."""
    torch.manual_seed(0)
    cortex_a = _small_cortex()
    torch.manual_seed(0)
    cortex_b = _small_cortex()

    wm_a = CortexWorldModel(cortex_a, action_keys=_ACTION_KEYS)
    wm_b = CortexWorldModel(
        cortex_b, action_keys=_ACTION_KEYS,
        consolidator=None, consolidate_every_ticks=0, consolidation_steps=1,
    )

    memory_a, memory_b = Memory(), Memory()
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    state = State(observation=None)
    for seq in range(4):
        frame_a, frame_b = _frame(rng_a), _frame(rng_b)
        assert np.array_equal(frame_a, frame_b)
        _push_frame(memory_a, frame_a, seq)
        _push_frame(memory_b, frame_b, seq)
        pred_a = wm_a.predict(state, memory_a)
        pred_b = wm_b.predict(state, memory_b)
        assert pred_a.next_latent == pred_b.next_latent
        assert pred_a.risk == pred_b.risk
        assert pred_a.predicted_reward == pred_b.predicted_reward
