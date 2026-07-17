"""Forgetting metric (issue #99, Milestone 5's falsifiable claim): does an
old scenario's accuracy survive learning a new one?

Staged training with generative replay must retain a previously-mastered
scenario (still beats copy-last on it) while flat training on the same new
data does not, and neither condition costs the tick loop a single missed
tick versus a no-sleep baseline.
"""

from __future__ import annotations

import copy
import random

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from cognitive_runtime.neural.replay_buffer import (  # noqa: E402
    ReplayBuffer,
    ReplayBufferConfig,
    Transition,
)
from cognitive_runtime.neural.world_model import MLPWorldModel  # noqa: E402
from cognitive_runtime.training.forgetting import (  # noqa: E402
    compare_forgetting_conditions,
    compute_forgetting_metric,
)
from cognitive_runtime.training.statistical_evaluation import cortex_horizon_statistics  # noqa: E402
from sleep.replay import (  # noqa: E402
    DreamFractionGate,
    evaluate_next_latent_quality,
    train_with_generative_replay,
)

LATENT_WIDTH = 4
N_ACTIONS = 3

#: Two scenarios sharing one action space and mostly-agreeing action->latent
#: dynamics, but colliding on one action: training only on
#: `object_permanence` overwrites the weight that action's transition needs
#: for `walk_forward` -- partial interference between two real tasks
#: sharing a model, not a scripted stand-in for forgetting.
_WALK_FORWARD_OFFSETS = [1.0, -1.0, 0.5]
_OBJECT_PERMANENCE_OFFSETS = [1.0, -1.0, 3.0]
_NOISE_STD = 0.05


def _scenario_transitions(offsets, n, seed, *, width=LATENT_WIDTH):
    rng = random.Random(seed)
    transitions = []
    for _ in range(n):
        latent = [rng.uniform(-1.0, 1.0) for _ in range(width)]
        action = rng.randrange(len(offsets))
        next_latent = [v + offsets[action] + rng.gauss(0.0, _NOISE_STD) for v in latent]
        transitions.append(
            Transition(latent=latent, action=action, reward=0.0, next_latent=next_latent, done=False)
        )
    return transitions


def _quality_stats(model, transitions, n_actions, *, n_boot=10, sample=20, seed=0):
    """Bootstrap `evaluate_next_latent_quality`'s scalar `model_mse` into a
    mean +/- CI over resampled held-out subsets, routed through the same
    `statistical_evaluation` machinery every other metric in this codebase
    reports through."""
    rng = random.Random(seed)
    values = [
        evaluate_next_latent_quality(model, rng.sample(transitions, min(sample, len(transitions))), n_actions)[
            "model_mse"
        ]
        for _ in range(n_boot)
    ]
    return cortex_horizon_statistics({0: values})[0]


def _train_steps(model, optimizer, transitions, *, steps, batch_size, n_actions, seed):
    rng = random.Random(seed)
    for _ in range(steps):
        batch = rng.sample(transitions, min(batch_size, len(transitions)))
        latents = torch.tensor([t.latent for t in batch], dtype=torch.float32)
        next_latents = torch.tensor([t.next_latent for t in batch], dtype=torch.float32)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        onehot = torch.nn.functional.one_hot(actions, num_classes=n_actions).float()
        model.train()
        optimizer.zero_grad()
        predicted = model(latents, onehot).next_latent
        loss = torch.nn.functional.mse_loss(predicted, next_latents)
        loss.backward()
        optimizer.step()


def _make_dream_source(frozen_model, old_pool, n_actions, seed):
    """A generative-replay dream source: hallucinates `next_latent` from
    `frozen_model`'s own belief about `old_pool`'s (latent, action) pairs --
    exactly `sleep.dream`'s "no ground truth, only the model's own closed
    loop" contract, at the latent-transition granularity instead of pixels."""
    rng = random.Random(seed)

    def dream_source(n):
        sampled = rng.sample(old_pool, min(n, len(old_pool)))
        latents = torch.tensor([t.latent for t in sampled], dtype=torch.float32)
        actions = torch.tensor([t.action for t in sampled], dtype=torch.long)
        onehot = torch.nn.functional.one_hot(actions, num_classes=n_actions).float()
        frozen_model.eval()
        with torch.no_grad():
            dreamed_next = frozen_model(latents, onehot).next_latent
        return [
            Transition(
                latent=sampled[i].latent, action=sampled[i].action,
                reward=0.0, next_latent=dreamed_next[i].tolist(), done=False,
            )
            for i in range(len(sampled))
        ]

    return dream_source


def test_staged_replay_retains_old_scenario_while_flat_training_regresses():
    torch.manual_seed(0)
    old_train = _scenario_transitions(_WALK_FORWARD_OFFSETS, 300, seed=1)
    old_holdout = _scenario_transitions(_WALK_FORWARD_OFFSETS, 100, seed=2)
    new_train = _scenario_transitions(_OBJECT_PERMANENCE_OFFSETS, 300, seed=3)

    # Master walk_forward first -- both conditions continue from here.
    base_model = MLPWorldModel(fused_width=LATENT_WIDTH, n_actions=N_ACTIONS, hidden_dim=32, depth=2)
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-2)
    _train_steps(base_model, base_optimizer, old_train, steps=300, batch_size=32, n_actions=N_ACTIONS, seed=10)
    before = _quality_stats(base_model, old_holdout, N_ACTIONS, seed=20)
    copy_last_mse = evaluate_next_latent_quality(base_model, old_holdout, N_ACTIONS)["copy_last_mse"]

    # Flat: continue training on object_permanence only, no replay.
    flat_model = copy.deepcopy(base_model)
    flat_optimizer = torch.optim.Adam(flat_model.parameters(), lr=1e-2)
    _train_steps(flat_model, flat_optimizer, new_train, steps=60, batch_size=32, n_actions=N_ACTIONS, seed=11)
    after_flat = _quality_stats(flat_model, old_holdout, N_ACTIONS, seed=20)

    # Staged+replay: the same object_permanence data, mixed with dreamed
    # walk_forward seeds gated on measured quality (sleep.replay).
    staged_model = copy.deepcopy(base_model)
    staged_optimizer = torch.optim.Adam(staged_model.parameters(), lr=1e-2)
    new_buffer = ReplayBuffer(ReplayBufferConfig(capacity=len(new_train)))
    for t in new_train:
        new_buffer.add(t)
    dream_source = _make_dream_source(base_model, old_train, N_ACTIONS, seed=12)
    # The dream generator is the frozen, already-mastered `base_model`, so
    # its quality (not the continuing `staged_model`'s, which is expected to
    # drift while learning the new scenario) is what the guardrail gates on.
    dream_quality_ratio = evaluate_next_latent_quality(base_model, old_holdout, N_ACTIONS)[
        "model_over_copy_last_mse"
    ]
    train_with_generative_replay(
        staged_model, staged_optimizer, new_buffer, dream_source, dream_quality_ratio,
        steps=60, batch_size=32, n_actions=N_ACTIONS, gate=DreamFractionGate(),
    )
    after_staged = _quality_stats(staged_model, old_holdout, N_ACTIONS, seed=20)

    flat_report = compute_forgetting_metric(
        "walk_forward", "object_permanence", "flat", before, after_flat, copy_last_mse,
    )
    staged_report = compute_forgetting_metric(
        "walk_forward", "object_permanence", "staged+replay", before, after_staged, copy_last_mse,
    )

    assert not flat_report.retained, "flat training on a colliding scenario must forget"
    assert staged_report.retained, "generative replay must retain the mastered scenario"
    assert compare_forgetting_conditions(staged_report, flat_report)


def test_compare_forgetting_conditions_rejects_mismatched_scenario_pairs():
    stats = cortex_horizon_statistics({0: [1.0, 1.0]})[0]
    a = compute_forgetting_metric("walk_forward", "object_permanence", "flat", stats, stats, 1.0)
    b = compute_forgetting_metric("turn", "object_permanence", "flat", stats, stats, 1.0)
    with pytest.raises(ValueError, match="same scenario pair"):
        compare_forgetting_conditions(a, b)


def test_retained_is_gated_on_beating_copy_last_not_matching_the_baseline_exactly():
    perfect = cortex_horizon_statistics({0: [0.01, 0.01]})[0]
    slightly_worse = cortex_horizon_statistics({0: [0.2, 0.2]})[0]
    much_worse = cortex_horizon_statistics({0: [5.0, 5.0]})[0]
    copy_last_mse = 1.0

    still_useful = compute_forgetting_metric(
        "walk_forward", "object_permanence", "staged+replay", perfect, slightly_worse, copy_last_mse,
    )
    forgotten = compute_forgetting_metric(
        "walk_forward", "object_permanence", "flat", perfect, much_worse, copy_last_mse,
    )
    assert still_useful.retained
    assert not forgotten.retained


# ------------------------------------------------------- zero missed-tick regression


def _run_wake_phase(seed):
    from cognitive_runtime.policies import ScriptedSurvivalPolicy
    from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
    from cognitive_runtime.runtime.config import RuntimeConfig
    from cognitive_runtime.runtime.loop import CognitiveRuntime

    world_config = {"episode_ticks": 40, "world_size": 16, "max_mobs": 2}
    program = MinecraftSurvivalBox(config=world_config)
    runtime = CognitiveRuntime(
        program=program, policy=ScriptedSurvivalPolicy(seed=seed),
        config=RuntimeConfig(
            episodes=1, seed=seed, max_ticks_per_episode=world_config["episode_ticks"],
            record=False, program_config=world_config,
        ),
    )
    return runtime.run()


def test_zero_missed_tick_regression_around_a_generative_replay_consolidation_pass():
    """Consolidation (mixing dreamed + real replay) runs with acting fully
    paused (`sleep.PhasicSleepSchedule`'s contract: sleep never overlaps
    wake), so wake-phase ticks immediately before and after it see exactly
    as many missed ticks as a no-sleep baseline run -- zero, either way."""
    baseline_summaries = _run_wake_phase(0) + _run_wake_phase(1)
    assert all(s.missed_ticks == 0 for s in baseline_summaries)

    before_sleep = _run_wake_phase(2)
    assert all(s.missed_ticks == 0 for s in before_sleep)

    # The sleep pass itself: a real generative-replay consolidation step,
    # entirely outside the tick loop.
    old_train = _scenario_transitions(_WALK_FORWARD_OFFSETS, 64, seed=30)
    new_train = _scenario_transitions(_OBJECT_PERMANENCE_OFFSETS, 64, seed=31)
    old_holdout = _scenario_transitions(_WALK_FORWARD_OFFSETS, 32, seed=32)
    model = MLPWorldModel(fused_width=LATENT_WIDTH, n_actions=N_ACTIONS, hidden_dim=16, depth=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    buffer = ReplayBuffer(ReplayBufferConfig(capacity=len(new_train)))
    for t in new_train:
        buffer.add(t)
    dream_source = _make_dream_source(model, old_train, N_ACTIONS, seed=33)
    dream_quality_ratio = evaluate_next_latent_quality(model, old_holdout, N_ACTIONS)[
        "model_over_copy_last_mse"
    ]
    train_with_generative_replay(
        model, optimizer, buffer, dream_source, dream_quality_ratio,
        steps=10, batch_size=16, n_actions=N_ACTIONS,
    )

    after_sleep = _run_wake_phase(3)
    assert all(s.missed_ticks == 0 for s in after_sleep)
