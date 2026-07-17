"""The forgetting metric and Milestone 5's falsifiable claim
(docs/v2/phases/phase-5-sleep-consolidation.md, issue #99): staged+replay
retains a mastered scenario within tolerance while a flat-training control
does not, with zero missed-tick regression versus a no-sleep baseline.

The two scenarios are synthetic (a fixed random per-action linear-tanh
"world"), not full Crafter/Minecraft sessions -- fast and deterministic
while exercising the real production pieces the milestone depends on:
``brain.cortex.PredictiveCortex`` as the thing consolidated,
``brain.hippocampus.Hippocampus`` as the seed store, ``sleep.replay_mix``'s
quality-gated mixer as the generative-replay mechanism, ``sleep.schedule.
PhasicSleepSchedule`` as the wake/sleep coordinator, and ``sleep.forgetting``
as the CI-refereed metric report.

The quality gate is measured, once, from the *frozen dream source*'s own
held-out performance on the old scenario, not the live model being trained:
the bootstrap risk the phase doc calls out is trusting dreams from a
half-trained model, and the frozen source here is the just-mastered
snapshot, already verified to clear the quality bar (the "mastery" assertion
below) -- exactly the situation in which the guardrail should let dreaming
proceed.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from brain.cortex import PredictiveCortex, PredictiveCortexConfig
from brain.hippocampus import Hippocampus, HippocampusConfig, SeedTags
from sleep.forgetting import compute_forgetting_metric
from sleep.replay_mix import GenerativeReplayMixer, ReplaySample, Reservoir, copy_last_quality_margin
from sleep.schedule import PhasicSleepSchedule

ACTIONS = ["wait", "left"]
LATENT_WIDTH = 6


def _scenario(seed: int):
    """A fixed random per-action linear-tanh transition -- the synthetic
    "world" a scenario's samples are drawn from. Deterministic given
    ``seed``, so "scenario A" and "scenario B" are two fixed, unrelated
    mappings the cortex must learn (and, for A, retain)."""
    generator = torch.Generator().manual_seed(seed)
    transition_matrices = [
        torch.randn(LATENT_WIDTH, LATENT_WIDTH, generator=generator) * 0.6
        for _ in ACTIONS
    ]

    def step(z: torch.Tensor, action_idx: int) -> torch.Tensor:
        return torch.tanh(z @ transition_matrices[action_idx].T)

    return step


def _sample_batch(step_fn, batch_size: int, rng: "torch.Generator"):
    z0 = torch.randn(batch_size, LATENT_WIDTH, generator=rng)
    action_idx = torch.randint(0, len(ACTIONS), (batch_size, 1), generator=rng)
    targets = torch.stack(
        [step_fn(z0[i], int(action_idx[i, 0])) for i in range(batch_size)]
    ).unsqueeze(1)
    return z0, action_idx, targets


def _train_step(model, optimizer, z0, actions, targets) -> float:
    model.train()
    hidden = model.initial_state(z0.shape[0])
    pred, _ = model.rollout(z0, actions, hidden)
    loss = F.mse_loss(pred, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _eval_losses(model, step_fn, n: int, rng: "torch.Generator") -> list:
    """Per-sample MSE (not a pooled mean): independent samples, so
    ``statistical_evaluation``'s CI machinery has something to bound."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n):
            z0, actions, targets = _sample_batch(step_fn, 1, rng)
            pred, _ = model.rollout(z0, actions, model.initial_state(1))
            losses.append(float(F.mse_loss(pred, targets).item()))
    model.train()
    return losses


def _quality_margin(model, step_fn, n: int, rng: "torch.Generator") -> float:
    model.eval()
    with torch.no_grad():
        z0, actions, targets = _sample_batch(step_fn, n, rng)
        pred, _ = model.rollout(z0, actions, model.initial_state(n))
        model_mse = float(F.mse_loss(pred, targets).item())
        copy_last_mse = float(F.mse_loss(z0.unsqueeze(1), targets).item())
    model.train()
    return copy_last_quality_margin(model_mse, copy_last_mse)


def _cortex() -> PredictiveCortex:
    torch.manual_seed(3)
    return PredictiveCortex(
        (4, 4, 3), ACTIONS,
        PredictiveCortexConfig(latent_width=LATENT_WIDTH, hidden_dim=24, reconstruction_size=4),
    )


def test_staged_replay_retains_a_mastered_scenario_while_flat_training_forgets_it():
    scenario_a = _scenario(seed=11)  # the "walk_forward"-analog: already mastered
    scenario_b = _scenario(seed=97)  # the "object_permanence"-analog: newly learned

    model = _cortex()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    train_rng = torch.Generator().manual_seed(1)
    for _ in range(400):
        z0, actions, targets = _sample_batch(scenario_a, 32, train_rng)
        _train_step(model, optimizer, z0, actions, targets)

    eval_rng = torch.Generator().manual_seed(2)
    losses_before = _eval_losses(model, scenario_a, 30, eval_rng)
    assert sum(losses_before) / len(losses_before) < 0.01, "scenario A did not reach mastery"

    # The hippocampal seed store + frozen recall source (module docstring,
    # sleep.replay_mix): seeds hold A's (z, action) pairs; dreams are rolled
    # from the *mastered, frozen* snapshot, not the model being trained on B.
    dream_source = copy.deepcopy(model)
    dream_source.eval()
    hippocampus = Hippocampus(HippocampusConfig(capacity=200))
    seed_rng = torch.Generator().manual_seed(4)
    for i in range(120):
        z0, actions, _ = _sample_batch(scenario_a, 1, seed_rng)
        hippocampus.encode(
            z=z0[0].tolist(), actions=[ACTIONS[int(actions[0, 0])]],
            tags=SeedTags(reward=1.0), tick_index=i,
        )

    # The quality gate: measured once from the frozen dream source's own
    # held-out A performance vs. copy-last -- not a constant, but (per the
    # phase doc) also not something that needs re-measuring from a live
    # model that isn't the thing doing the dreaming.
    quality_rng = torch.Generator().manual_seed(50)
    quality_margin = _quality_margin(dream_source, scenario_a, 30, quality_rng)
    assert quality_margin > 0.9, "the mastered snapshot should clearly beat copy-last on A"

    flat_model = copy.deepcopy(model)
    flat_optimizer = torch.optim.Adam(flat_model.parameters(), lr=0.002)
    replay_model = copy.deepcopy(model)
    replay_optimizer = torch.optim.Adam(replay_model.parameters(), lr=0.002)

    flat_reservoir = Reservoir(capacity=300)
    reservoir = Reservoir(capacity=300)
    mixer = GenerativeReplayMixer(
        reservoir=reservoir, hippocampus=hippocampus, dream_cortex=dream_source, cap=0.7, seed=6,
    )

    b_rng = torch.Generator().manual_seed(7)
    consolidation_rng = __import__("random").Random(0)

    def make_b_sample() -> ReplaySample:
        z0, actions, targets = _sample_batch(scenario_b, 1, b_rng)
        return ReplaySample(
            z0=z0[0].tolist(), actions=[ACTIONS[int(actions[0, 0])]],
            targets=targets[0], source="real",
        )

    total_ticks = 40
    wake_ticks = 10
    steps_per_consolidation = 15
    batch_size = 16

    # --- Staged+replay: PhasicSleepSchedule interleaves acting (accumulate
    # B experience into both conditions' reservoirs) with periodic
    # consolidation. Both conditions get the identical tick budget, the same
    # number of gradient steps, and the same batch size -- the only
    # difference is whether the consolidation batch mixes in dreamed A
    # seeds (replay) or draws purely from B's own reservoir (flat/no-sleep
    # control), isolating exactly the variable Milestone 5's claim is about.
    schedule = PhasicSleepSchedule(wake_ticks=wake_ticks)
    ticks_processed = 0

    def act() -> None:
        nonlocal ticks_processed
        sample = make_b_sample()
        flat_reservoir.add(sample)
        reservoir.add(
            ReplaySample(z0=sample.z0, actions=sample.actions, targets=sample.targets, source="real")
        )
        ticks_processed += 1

    def sleep_pass() -> int:
        for _ in range(steps_per_consolidation):
            samples = flat_reservoir.sample(batch_size, consolidation_rng)
            z0 = torch.tensor([s.z0 for s in samples])
            actions = torch.tensor([[ACTIONS.index(a) for a in s.actions] for s in samples])
            targets = torch.stack([s.targets for s in samples])
            _train_step(flat_model, flat_optimizer, z0, actions, targets)
        for _ in range(steps_per_consolidation):
            batch = mixer.mix_batch(batch_size, quality_margin)
            _train_step(replay_model, replay_optimizer, batch.z0, batch.actions, batch.targets)
        return 1

    for _ in range(total_ticks):
        schedule.act(act)
        if schedule.sleep_due:
            schedule.consolidate(sleep_pass)

    # Every tick landed on a wake-phase boundary exactly (40 % 10 == 0): no
    # leftover partial phase, and acting was never skipped or blocked to
    # make room for consolidation (task 2's "no weight staleness"/zero
    # missed-tick regression vs. a no-sleep baseline that would also have
    # processed all 40 ticks).
    assert schedule.request_sleep() is False
    assert ticks_processed == total_ticks

    losses_after_flat = _eval_losses(flat_model, scenario_a, 30, torch.Generator().manual_seed(2))
    losses_after_replay = _eval_losses(replay_model, scenario_a, 30, torch.Generator().manual_seed(2))

    tolerance = 0.08
    flat_report = compute_forgetting_metric(
        losses_before, losses_after_flat,
        old_scenario="scenario_a", new_scenario="scenario_b", tolerance=tolerance,
    )
    replay_report = compute_forgetting_metric(
        losses_before, losses_after_replay,
        old_scenario="scenario_a", new_scenario="scenario_b", tolerance=tolerance,
    )

    assert replay_report.retained is True, (
        f"staged+replay should retain scenario A within tolerance: {replay_report}"
    )
    assert flat_report.retained is False, (
        f"flat training should forget scenario A beyond tolerance: {flat_report}"
    )
    assert replay_report.after.mean < flat_report.after.mean
    assert flat_report.comparison.regressed
