"""Generative replay (docs/v2/phases/phase-5-sleep-consolidation.md task 4,
issue #99): dream fraction is 0 below the quality bar, ramps/caps above it;
no dream-only batch is ever drawn; the reservoir is retained (sampling never
depletes or mutates it)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from brain.cortex import PredictiveCortex, PredictiveCortexConfig
from brain.hippocampus import Hippocampus, SeedTags
from sleep.replay_mix import (
    GenerativeReplayMixer,
    ReplaySample,
    Reservoir,
    copy_last_quality_margin,
    dream_fraction,
)

ACTIONS = ["wait", "left"]


def _cortex(seed: int = 3) -> PredictiveCortex:
    torch.manual_seed(seed)
    return PredictiveCortex(
        (4, 4, 3), ACTIONS, PredictiveCortexConfig(latent_width=5, hidden_dim=8, reconstruction_size=4),
    )


def _real_sample(model: PredictiveCortex, value: float = 0.0) -> ReplaySample:
    return ReplaySample(
        z0=[value] * model.latent_width,
        actions=["wait"],
        targets=torch.zeros(1, model.latent_width),
        source="real",
    )


def _hippocampus_with_seeds(model: PredictiveCortex, n: int) -> Hippocampus:
    hippocampus = Hippocampus()
    for i in range(n):
        hippocampus.encode(
            z=torch.randn(model.latent_width).tolist(),
            actions=["wait", "left"],
            tags=SeedTags(reward=1.0),
            tick_index=i,
        )
    return hippocampus


# --------------------------------------------------------------- dream_fraction


def test_dream_fraction_is_zero_at_or_below_ramp_start():
    assert dream_fraction(-1.0) == 0.0
    assert dream_fraction(0.0) == 0.0


def test_dream_fraction_ramps_linearly_between_start_and_end():
    fraction = dream_fraction(0.5, ramp_start=0.0, ramp_end=1.0, cap=0.5)
    assert fraction == pytest.approx(0.25)


def test_dream_fraction_caps_and_never_exceeds_cap_beyond_ramp_end():
    assert dream_fraction(1.0, ramp_end=1.0, cap=0.5) == pytest.approx(0.5)
    assert dream_fraction(50.0, ramp_end=1.0, cap=0.5) == pytest.approx(0.5)


def test_dream_fraction_rejects_invalid_cap_and_ramp():
    with pytest.raises(ValueError, match="cap"):
        dream_fraction(0.5, cap=0.0)
    with pytest.raises(ValueError, match="cap"):
        dream_fraction(0.5, cap=1.5)
    with pytest.raises(ValueError, match="ramp_end"):
        dream_fraction(0.5, ramp_start=1.0, ramp_end=1.0)


def test_copy_last_quality_margin_positive_when_model_beats_copy_last():
    assert copy_last_quality_margin(model_mse=0.5, copy_last_mse=1.0) == pytest.approx(0.5)
    assert copy_last_quality_margin(model_mse=1.0, copy_last_mse=1.0) == pytest.approx(0.0)
    assert copy_last_quality_margin(model_mse=2.0, copy_last_mse=1.0) == pytest.approx(-1.0)


def test_copy_last_quality_margin_degenerate_baseline_reads_as_no_headroom():
    assert copy_last_quality_margin(model_mse=0.1, copy_last_mse=0.0) == 0.0


# --------------------------------------------------------------- Reservoir


def test_reservoir_rejects_dream_samples():
    model = _cortex()
    reservoir = Reservoir(capacity=4)
    dream_sample = ReplaySample(
        z0=[0.0] * model.latent_width, actions=["wait"],
        targets=torch.zeros(1, model.latent_width), source="dream",
    )
    with pytest.raises(ValueError, match="real"):
        reservoir.add(dream_sample)


def test_reservoir_sampling_is_retained_not_depleted():
    model = _cortex()
    reservoir = Reservoir(capacity=4)
    for i in range(3):
        reservoir.add(_real_sample(model, value=float(i)))
    assert len(reservoir) == 3

    rng = __import__("random").Random(0)
    reservoir.sample(10, rng)
    assert len(reservoir) == 3
    # A second, independent draw still succeeds -- the store was never
    # drained by the first.
    reservoir.sample(10, rng)
    assert len(reservoir) == 3


def test_reservoir_evicts_oldest_beyond_capacity():
    model = _cortex()
    reservoir = Reservoir(capacity=2)
    reservoir.add(_real_sample(model, value=0.0))
    reservoir.add(_real_sample(model, value=1.0))
    reservoir.add(_real_sample(model, value=2.0))
    assert len(reservoir) == 2


def test_reservoir_sample_from_empty_raises():
    reservoir = Reservoir(capacity=2)
    with pytest.raises(ValueError, match="empty"):
        reservoir.sample(1, __import__("random").Random(0))


# --------------------------------------------------------------- GenerativeReplayMixer


def _mixer(model: PredictiveCortex, *, n_real: int = 8, n_seeds: int = 8, cap: float = 0.5) -> GenerativeReplayMixer:
    reservoir = Reservoir(capacity=64)
    for i in range(n_real):
        reservoir.add(_real_sample(model, value=float(i)))
    hippocampus = _hippocampus_with_seeds(model, n_seeds)
    return GenerativeReplayMixer(
        reservoir=reservoir, hippocampus=hippocampus, dream_cortex=model, cap=cap, seed=1,
    )


def test_weak_cortex_quality_below_bar_draws_zero_dreams():
    model = _cortex()
    mixer = _mixer(model)
    batch = mixer.mix_batch(8, quality_margin=0.0)
    assert batch.n_dream == 0
    assert batch.n_real == 8
    assert batch.fraction_requested == 0.0


def test_quality_above_bar_ramps_dream_share_toward_cap():
    model = _cortex()
    mixer = _mixer(model, cap=0.5)
    weak = mixer.mix_batch(10, quality_margin=0.1)
    strong = mixer.mix_batch(10, quality_margin=0.9)
    assert weak.n_dream < strong.n_dream
    assert strong.fraction_requested <= 0.5 + 1e-9


def test_dream_only_batch_is_never_drawn_even_at_full_quality():
    model = _cortex()
    mixer = _mixer(model, cap=1.0)
    batch = mixer.mix_batch(4, quality_margin=100.0)
    assert batch.n_real >= 1
    assert batch.n_dream <= batch.batch_size - 1


def test_batch_shapes_match_stacked_real_and_dream_samples():
    model = _cortex()
    mixer = _mixer(model, cap=0.5)
    batch = mixer.mix_batch(6, quality_margin=1.0)
    assert batch.z0.shape == (6, model.latent_width)
    assert batch.actions.shape[0] == 6
    assert batch.targets.shape[0] == 6
    assert batch.targets.shape[-1] == model.latent_width


def test_mix_batch_rejects_non_positive_batch_size():
    model = _cortex()
    mixer = _mixer(model)
    with pytest.raises(ValueError, match="batch_size"):
        mixer.mix_batch(0, quality_margin=1.0)


def test_mix_batch_with_empty_reservoir_raises():
    model = _cortex()
    mixer = GenerativeReplayMixer(
        reservoir=Reservoir(capacity=4), hippocampus=_hippocampus_with_seeds(model, 4),
        dream_cortex=model, seed=1,
    )
    with pytest.raises(ValueError, match="reservoir is empty"):
        mixer.mix_batch(4, quality_margin=1.0)


def test_mix_batch_falls_back_to_all_real_when_hippocampus_is_empty():
    model = _cortex()
    reservoir = Reservoir(capacity=8)
    for i in range(8):
        reservoir.add(_real_sample(model, value=float(i)))
    mixer = GenerativeReplayMixer(
        reservoir=reservoir, hippocampus=Hippocampus(), dream_cortex=model, cap=0.5, seed=1,
    )
    batch = mixer.mix_batch(4, quality_margin=1.0)
    assert batch.n_dream == 0
    assert batch.n_real == 4


def test_mix_batch_raises_when_hippocampus_has_fewer_seeds_than_requested():
    model = _cortex()
    reservoir = Reservoir(capacity=8)
    for i in range(8):
        reservoir.add(_real_sample(model, value=float(i)))
    mixer = GenerativeReplayMixer(
        reservoir=reservoir, hippocampus=_hippocampus_with_seeds(model, 1),
        dream_cortex=model, cap=1.0, seed=1,
    )
    with pytest.raises(ValueError, match="hippocampus holds only"):
        mixer.mix_batch(4, quality_margin=100.0)


def test_zero_action_seeds_are_excluded_from_dream_candidates_not_sampled_and_crashed_on():
    """`Hippocampus.encode` permits `actions=[]` (a tick with no emitted
    motor command); such a seed can never be dreamed forward one step and
    must not be a candidate `mix_batch` can randomly draw and then choke on.
    """
    model = _cortex()
    reservoir = Reservoir(capacity=8)
    for i in range(8):
        reservoir.add(_real_sample(model, value=float(i)))
    hippocampus = Hippocampus()
    # Mostly zero-action seeds, with just enough eligible (>=1 action) seeds
    # to satisfy a small dream request -- a random draw over *all* seeds
    # would very likely pick a zero-action one and crash without filtering.
    for i in range(20):
        hippocampus.encode(z=[0.0] * model.latent_width, actions=[], tags=SeedTags(reward=1.0), tick_index=i)
    for i in range(2):
        hippocampus.encode(
            z=torch.randn(model.latent_width).tolist(), actions=["wait"],
            tags=SeedTags(reward=1.0), tick_index=100 + i,
        )
    mixer = GenerativeReplayMixer(
        reservoir=reservoir, hippocampus=hippocampus, dream_cortex=model, cap=0.5, seed=1,
    )
    batch = mixer.mix_batch(4, quality_margin=100.0)
    assert batch.n_dream == 2  # exactly the two eligible (non-zero-action) seeds
    assert batch.n_real == 2


def test_mix_batch_rejects_reservoir_samples_whose_action_length_mismatches_dream_length():
    model = _cortex()
    reservoir = Reservoir(capacity=4)
    two_step_targets = torch.zeros(2, model.latent_width)
    reservoir.add(
        ReplaySample(z0=[0.0] * model.latent_width, actions=["wait", "left"], targets=two_step_targets, source="real")
    )
    mixer = GenerativeReplayMixer(
        reservoir=reservoir, hippocampus=_hippocampus_with_seeds(model, 4), dream_cortex=model, cap=0.5, seed=1,
    )
    with pytest.raises(ValueError, match="dream_length"):
        mixer.mix_batch(2, quality_margin=0.0)  # dream_length defaults to 1
