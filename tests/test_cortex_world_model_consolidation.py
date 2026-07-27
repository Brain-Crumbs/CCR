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
        # Mirror the real `CortexConsolidator.publish_to`: it resets the
        # world model's rolling state so fresh weights take effect cleanly
        # (issue #175 review: this is exactly what used to silently wipe
        # `_latent` and drop every post-consolidation transition).
        world_model.reset()
        self.publish_calls += 1
        return self.publish_calls

    def stats(self):
        return {"consolidations": len(self.consolidate_calls), "publishes": self.publish_calls}


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


def test_record_transition_keeps_firing_after_every_consolidation_cadence():
    """Issue #175 review (P2): `publish_to` resets the world model's rolling
    state, which used to wipe `_latent` too -- dropping the transition
    immediately following *every* consolidation, permanently at
    `consolidate_every_ticks=1` (a no-op forever: the reservoir never
    receives anything). `_latent` must survive the reset so the very next
    tick can still record its transition."""
    cortex = _small_cortex()
    fake = FakeConsolidator()
    wm = CortexWorldModel(
        cortex, action_keys=_ACTION_KEYS, consolidator=fake,
        consolidate_every_ticks=1, consolidation_steps=1,
    )

    _run_ticks(wm, 5)

    # Ticks 1..5 all trigger consolidation (cadence 1); tick 1 has no prior
    # latent to record (episode start), but ticks 2-5 each still record a
    # transition despite the consolidation between every pair of ticks.
    assert fake.consolidate_calls == [1, 1, 1, 1, 1]
    assert len(fake.recorded) == 4


def test_consolidation_publish_preserves_this_ticks_live_prediction():
    """A publish resets rolling episode state inside ``predict()``, but the
    decoded forecast already produced for this tick must remain available to
    the recorder after ``predict()`` returns."""
    fake = FakeConsolidator()
    wm = CortexWorldModel(
        _small_cortex(), action_keys=_ACTION_KEYS, consolidator=fake,
        consolidate_every_ticks=1, consolidation_steps=1,
    )

    _run_ticks(wm, 1)

    assert fake.publish_calls == 1
    live = wm.live_prediction_record()
    assert live is not None
    assert set(live["frames"]) == {"1", "4"}


def test_consolidation_publish_persists_checkpoint_to_disk(tmp_path):
    """Issue #175 review (P1): with `--async-trainer`, `publish_to`'s
    in-memory weight update was the only hand-off -- nothing wrote the
    consolidated cortex back to disk, so a completed run (or a crash/
    KeyboardInterrupt) silently discarded all online learning. A
    `checkpoint_path` must be persisted on every publish."""
    from cognitive_runtime.training.action_world_model import load_action_world_model

    cortex = _small_cortex()
    fake = FakeConsolidator()
    checkpoint_path = str(tmp_path / "cortex.pt")
    wm = CortexWorldModel(
        cortex, action_keys=_ACTION_KEYS, consolidator=fake,
        consolidate_every_ticks=2, consolidation_steps=1,
        checkpoint_path=checkpoint_path,
    )

    _run_ticks(wm, 3)

    assert fake.publish_calls == 1
    reloaded, stats = load_action_world_model(checkpoint_path)
    assert reloaded.action_keys == _ACTION_KEYS
    assert stats == fake.stats()


def test_checkpoint_path_defaults_from_a_string_model_argument(tmp_path):
    """Loading `CortexWorldModel` from a checkpoint path (the
    `--world-model cortex:PATH` CLI case) defaults the consolidation save
    target to that same path, with no extra CLI wiring needed."""
    from cognitive_runtime.training.action_world_model import save_action_world_model

    checkpoint_path = str(tmp_path / "cortex.pt")
    save_action_world_model(checkpoint_path, _small_cortex(), {})

    wm = CortexWorldModel(checkpoint_path, action_keys=_ACTION_KEYS)
    assert wm.checkpoint_path == checkpoint_path

    wm_override = CortexWorldModel(
        checkpoint_path, action_keys=_ACTION_KEYS, checkpoint_path=str(tmp_path / "other.pt"),
    )
    assert wm_override.checkpoint_path == str(tmp_path / "other.pt")
