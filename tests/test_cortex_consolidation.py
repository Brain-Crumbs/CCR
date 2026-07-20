"""Live cortex consolidation: micro-sleep + generative replay (issue #167).

Exercises ``sleep.cortex_consolidation.CortexConsolidator`` as the sleep-phase
learner target -- the cortex, not the legacy actor/critic stack -- driven from
``sleep.schedule.PhasicSleepSchedule`` (the "live loop"). The three acceptance
criteria of the issue map to the three headline tests here:

- a continuous run measurably improves held-out cortex prediction (loss drops
  across micro-sleeps);
- the dream fraction obeys the quality gate (0% until the frozen snapshot beats
  copy-last, capped ~=0.5);
- the Milestone 5 forgetting metric runs from this live-loop path, not just the
  ``tests/test_forgetting_metric.py`` harness.
"""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from brain.cortex import PredictiveCortex, PredictiveCortexConfig
from brain.hippocampus import Hippocampus, HippocampusConfig, SeedTags
from sleep.cortex_consolidation import CortexConsolidator
from sleep.forgetting import compute_forgetting_metric
from sleep.replay_mix import ReplaySample
from sleep.schedule import PhasicSleepSchedule

ACTIONS = ["wait", "left"]
LATENT_WIDTH = 6


def _scenario(seed: int):
    """A fixed random per-action linear-tanh transition -- the same synthetic
    "world" ``tests/test_forgetting_metric.py`` draws its scenarios from."""
    generator = torch.Generator().manual_seed(seed)
    matrices = [
        torch.randn(LATENT_WIDTH, LATENT_WIDTH, generator=generator) * 0.6
        for _ in ACTIONS
    ]

    def step(z: torch.Tensor, action_idx: int) -> torch.Tensor:
        return torch.tanh(z @ matrices[action_idx].T)

    return step


def _cortex(seed: int = 3) -> PredictiveCortex:
    torch.manual_seed(seed)
    return PredictiveCortex(
        (4, 4, 3), ACTIONS,
        PredictiveCortexConfig(latent_width=LATENT_WIDTH, hidden_dim=24, reconstruction_size=4),
    )


def _sample(step_fn, rng: "torch.Generator") -> ReplaySample:
    z0 = torch.randn(LATENT_WIDTH, generator=rng)
    action_idx = int(torch.randint(0, len(ACTIONS), (1,), generator=rng))
    target = step_fn(z0, action_idx).unsqueeze(0)  # [1, L]
    return ReplaySample(z0=z0.tolist(), actions=[ACTIONS[action_idx]], targets=target, source="real")


def _eval_losses(model, step_fn, n: int, rng: "torch.Generator") -> list:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n):
            s = _sample(step_fn, rng)
            z0 = torch.tensor([s.z0])
            actions = torch.tensor([[ACTIONS.index(a) for a in s.actions]])
            pred, _ = model.rollout(z0, actions, model.initial_state(1))
            losses.append(float(F.mse_loss(pred, s.targets.unsqueeze(0))))
    model.train()
    return losses


# --------------------------------------------------------------- loss drops


def test_micro_sleeps_improve_held_out_cortex_prediction():
    """A continuous wake/sleep run drives held-out cortex loss down across
    micro-sleeps (issue #167 acceptance line 1)."""
    scenario = _scenario(seed=11)
    cortex = _cortex()
    consolidator = CortexConsolidator(
        cortex, Hippocampus(HippocampusConfig(capacity=200)),
        lr=0.02, batch_size=16, held_out_every=6, seed=1,
    )

    wake_rng = torch.Generator().manual_seed(20)
    schedule = PhasicSleepSchedule(wake_ticks=8)

    losses_over_time = []

    def act() -> None:
        s = _sample(scenario, wake_rng)
        consolidator.record_transition(s.z0, s.actions, s.targets)

    # Prime the reservoir/held-out with a first wake phase so the very first
    # micro-sleep has real experience to train on.
    for _ in range(8):
        schedule.act(act)
    schedule.consolidate(lambda: consolidator.consolidate(20))
    losses_over_time.append(consolidator.held_out_loss())

    for _ in range(6):
        for _ in range(8):
            schedule.act(act)
        schedule.consolidate(lambda: consolidator.consolidate(20))
        losses_over_time.append(consolidator.held_out_loss())

    assert all(loss is not None for loss in losses_over_time)
    # Held-out prediction measurably improved end-to-end, and the version
    # advanced once per micro-sleep.
    assert losses_over_time[-1] < losses_over_time[0] * 0.6, losses_over_time
    assert consolidator.version == 7


def test_micro_sleep_before_any_wake_experience_is_a_noop():
    consolidator = CortexConsolidator(_cortex(), Hippocampus(), batch_size=8)
    assert consolidator.consolidate(10) == 0  # version unchanged, no crash
    assert consolidator.last_metrics.steps == 0
    assert consolidator.held_out_loss() is None


# --------------------------------------------------------------- quality gate


def _mastered_cortex(scenario, *, steps: int = 400) -> PredictiveCortex:
    cortex = _cortex()
    optimizer = torch.optim.Adam(cortex.parameters(), lr=0.03)
    rng = torch.Generator().manual_seed(1)
    for _ in range(steps):
        batch = [_sample(scenario, rng) for _ in range(32)]
        z0 = torch.tensor([s.z0 for s in batch])
        actions = torch.tensor([[ACTIONS.index(a) for a in s.actions] for s in batch])
        targets = torch.stack([s.targets for s in batch])
        cortex.train()
        pred, _ = cortex.rollout(z0, actions, cortex.initial_state(z0.shape[0]))
        loss = F.mse_loss(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return cortex


def _hippocampus_of(scenario, n: int) -> Hippocampus:
    hippocampus = Hippocampus(HippocampusConfig(capacity=300))
    rng = torch.Generator().manual_seed(4)
    for i in range(n):
        s = _sample(scenario, rng)
        hippocampus.encode(z=s.z0, actions=s.actions, tags=SeedTags(reward=1.0), tick_index=i)
    return hippocampus


def _held_out(scenario, n: int, seed: int) -> list:
    rng = torch.Generator().manual_seed(seed)
    return [_sample(scenario, rng) for _ in range(n)]


def test_dream_fraction_is_zero_below_the_bar_and_ramps_after_a_good_snapshot():
    scenario = _scenario(seed=11)
    hippocampus = _hippocampus_of(scenario, 120)
    held_out = _held_out(scenario, 40, seed=50)

    # A snapshot that has not cleared the quality bar (margin at/below
    # ``ramp_start``) draws no dreams -- the guardrail keeps consolidation on
    # real experience until the frozen source has earned trust. Asserted on
    # both a negative margin and one sitting exactly at the bar.
    weak = CortexConsolidator(
        _cortex(), hippocampus, lr=0.002, batch_size=16, ramp_start=0.7, cap=0.5, seed=6,
    )
    for s in _held_out(scenario, 40, seed=99):
        weak.ingest_sample(s)
    for below_bar in (-0.3, 0.0, 0.5):  # all at/below ramp_start=0.7
        weak.set_quality_margin(below_bar)
        weak.consolidate(6)
        assert weak.last_metrics.mean_dream_fraction == 0.0, below_bar

    # A mastered snapshot, measured on its own held-out, clears the bar -> the
    # dream share ramps in and never exceeds the cap.
    mastered = _mastered_cortex(scenario)
    strong = CortexConsolidator(
        copy.deepcopy(mastered), hippocampus, lr=0.002, batch_size=16,
        ramp_start=0.7, cap=0.5, seed=6,
    )
    for s in _held_out(scenario, 40, seed=99):
        strong.ingest_sample(s)
    margin = strong.refresh_dream_source(held_out=held_out)
    assert margin > 0.7, margin  # measured, not overridden: the snapshot earned it
    strong.consolidate(6)
    assert 0.0 < strong.last_metrics.mean_dream_fraction <= 0.5 + 1e-9


# --------------------------------- Milestone 5 forgetting metric (live loop)


def test_forgetting_metric_runs_from_the_live_consolidation_loop():
    """Milestone 5's falsifiable claim, driven through the live
    ``CortexConsolidator`` + ``PhasicSleepSchedule`` path rather than the
    hand-rolled test harness: staged+replay retains a mastered scenario A while
    flat training on B forgets it (issue #167 acceptance line 3)."""
    scenario_a = _scenario(seed=11)   # mastered, rehearsed via dreams
    scenario_b = _scenario(seed=97)   # newly learned during the run

    mastered = _mastered_cortex(scenario_a)
    losses_before = _eval_losses(mastered, scenario_a, 30, torch.Generator().manual_seed(2))
    assert sum(losses_before) / len(losses_before) < 0.02, "scenario A did not reach mastery"

    hippocampus = _hippocampus_of(scenario_a, 120)
    a_held_out = _held_out(scenario_a, 30, seed=50)

    replay = CortexConsolidator(
        copy.deepcopy(mastered), hippocampus, lr=0.002, batch_size=16, cap=0.7, seed=6,
    )
    # Freeze the mastered snapshot as the dream source and gate on its own A
    # held-out quality (the bootstrap guardrail: dreams come from the frozen,
    # already-mastered snapshot, not the model being trained on B).
    margin = replay.refresh_dream_source(held_out=a_held_out)
    assert margin > 0.9, margin

    flat = CortexConsolidator(
        copy.deepcopy(mastered), Hippocampus(), lr=0.002, batch_size=16, seed=6,
    )

    b_rng = torch.Generator().manual_seed(7)
    schedule = PhasicSleepSchedule(wake_ticks=10)

    def act() -> None:
        s = _sample(scenario_b, b_rng)
        replay.ingest_sample(s)
        flat.ingest_sample(ReplaySample(z0=s.z0, actions=s.actions, targets=s.targets, source="real"))

    for _ in range(40):
        schedule.act(act)
        if schedule.sleep_due:
            # The replay condition consolidates through the schedule (paused
            # acting, no staleness); the flat control gets an identical-budget
            # pass off to the side, isolating dreams as the only difference.
            schedule.consolidate(lambda: replay.consolidate(15))
            flat.consolidate(15)

    assert schedule.request_sleep() is False  # 40 % 10 == 0: clean boundaries

    losses_after_replay = _eval_losses(replay.cortex, scenario_a, 30, torch.Generator().manual_seed(2))
    losses_after_flat = _eval_losses(flat.cortex, scenario_a, 30, torch.Generator().manual_seed(2))

    tolerance = 0.08
    replay_report = compute_forgetting_metric(
        losses_before, losses_after_replay,
        old_scenario="scenario_a", new_scenario="scenario_b", tolerance=tolerance,
    )
    flat_report = compute_forgetting_metric(
        losses_before, losses_after_flat,
        old_scenario="scenario_a", new_scenario="scenario_b", tolerance=tolerance,
    )

    assert replay_report.retained is True, f"replay should retain A: {replay_report}"
    assert flat_report.retained is False, f"flat should forget A: {flat_report}"
    assert replay_report.after.mean < flat_report.after.mean


# --------------------------------------------------------------- publish-back


class _FakeWorldModel:
    """Stands in for the A1 ``CortexWorldModel`` adapter: a ``model`` cortex
    plus a rolling world state that must reset when weights are republished."""

    def __init__(self, cortex: PredictiveCortex):
        self.model = cortex
        self._hidden = "stale-state"

    def reset(self) -> None:
        self._hidden = None


def test_publish_to_hands_raw_weights_back_and_resets_world_state():
    scenario = _scenario(seed=11)
    cortex = _cortex()
    consolidator = CortexConsolidator(cortex, Hippocampus(), lr=0.05, batch_size=8, seed=1)
    rng = torch.Generator().manual_seed(20)
    for _ in range(20):
        s = _sample(scenario, rng)
        consolidator.ingest_sample(s)
    consolidator.consolidate(30)

    live = _FakeWorldModel(_cortex(seed=99))  # different init from the trained one
    version = consolidator.publish_to(live)

    assert version == consolidator.version
    assert live._hidden is None  # rolling state reset for the fresh weights
    for published, trained in zip(
        live.model.state_dict().values(), consolidator.cortex.state_dict().values()
    ):
        assert torch.equal(published, trained)


def test_ema_publish_tracks_a_slow_moving_target_for_the_concurrent_schedule():
    scenario = _scenario(seed=11)
    cortex = _cortex()
    consolidator = CortexConsolidator(
        cortex, Hippocampus(), lr=0.05, batch_size=8, seed=1, ema_decay=0.9,
    )
    rng = torch.Generator().manual_seed(20)
    for _ in range(20):
        consolidator.ingest_sample(_sample(scenario, rng))
    consolidator.consolidate(30)

    live = _FakeWorldModel(_cortex(seed=99))
    consolidator.publish_to(live)  # defaults to EMA when built with ema_decay

    # The EMA snapshot lags the raw trained weights: at least one float tensor
    # differs from the live cortex's current parameters.
    differs = any(
        not torch.equal(pub, raw)
        for pub, raw in zip(
            live.model.state_dict().values(), consolidator.cortex.state_dict().values()
        )
    )
    assert differs
