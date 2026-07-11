"""Prioritized replay buffer and session loader (issue #28)."""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.neural import (  # noqa: E402
    MixedTrainingSchedule,
    NeuralAgentCheckpoint,
    PriorityWeights,
    ReplayBuffer,
    ReplayBufferConfig,
    Transition,
    load_session_into_buffer,
    transition_priority,
)
from cognitive_runtime.core.world_model import Prediction, WorldModel  # noqa: E402
from cognitive_runtime.policies import ScriptedSurvivalPolicy  # noqa: E402
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox  # noqa: E402
from cognitive_runtime.runtime.config import RuntimeConfig  # noqa: E402
from cognitive_runtime.runtime.loop import CognitiveRuntime  # noqa: E402
from cognitive_runtime.training.features import ACTION_KEYS  # noqa: E402


class _FakeWorldModel(WorldModel):
    """Deterministic non-None prediction_error/predicted_reward every tick
    (issue #58), so a recorded session carries `internal.novelty` and
    `internal.reward_prediction_error` for the session-loader tests below."""

    def predict(self, state, memory) -> Prediction:
        return Prediction(risk=0.1, predicted_reward=0.0, prediction_error=0.3)


def _record_modulated_session(tmp_path, session_id: str, *, ticks: int, seed: int = 0):
    config = {"episode_ticks": ticks, "world_size": 16, "max_mobs": 3}
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=seed,
        max_ticks_per_episode=ticks,
        record_dir=str(tmp_path),
        session_id=session_id,
        program_config=config,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config),
        policy=ScriptedSurvivalPolicy(seed=seed),
        config=runtime_config,
        world_model=_FakeWorldModel(),
    ).run()
    return os.path.join(str(tmp_path), session_id)


def _transition(source: str, reward: float = 0.0, done: bool = False) -> Transition:
    return Transition(
        latent=[reward, 0.0],
        action=0,
        reward=reward,
        next_latent=[reward, 1.0],
        done=done,
        source=source,
    )


def _record_session(tmp_path, session_id: str, *, ticks: int, seed: int = 0):
    config = {"episode_ticks": ticks, "world_size": 16, "max_mobs": 3}
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=seed,
        max_ticks_per_episode=ticks,
        record_dir=str(tmp_path),
        session_id=session_id,
        program_config=config,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config),
        policy=ScriptedSurvivalPolicy(seed=seed),
        config=runtime_config,
    ).run()
    return os.path.join(str(tmp_path), session_id)


# --------------------------------------------------------------------- bounds


def test_bounded_eviction_keeps_only_the_most_recent_transitions():
    buffer = ReplayBuffer(ReplayBufferConfig(capacity=5, seed=0))
    for i in range(8):
        buffer.add(_transition(f"t{i}"))

    assert len(buffer) == 5
    assert buffer.total_added == 8
    assert buffer.total_evicted == 3
    sources = {t.source for t in buffer.transitions()}
    assert sources == {"t3", "t4", "t5", "t6", "t7"}


def test_capacity_and_alpha_must_be_valid():
    with pytest.raises(ValueError):
        ReplayBufferConfig(capacity=0)
    with pytest.raises(ValueError):
        ReplayBufferConfig(alpha=-1)


def test_sample_from_empty_buffer_raises():
    buffer = ReplayBuffer()
    with pytest.raises(ValueError):
        buffer.sample(1)


# ------------------------------------------------------------ priority weights


def test_transition_priority_degrades_gracefully_without_novelty_or_error():
    weights = PriorityWeights(reward=1.0, death=1.0, damage=0.0, novelty=1.0, prediction_error=1.0)
    with_signals = Transition(
        latent=[], action=0, reward=1.0, next_latent=[], done=False,
        novelty=0.5, prediction_error=0.5,
    )
    without_signals = Transition(
        latent=[], action=0, reward=1.0, next_latent=[], done=False,
        novelty=None, prediction_error=None,
    )
    # Both transitions have identical reward/death/damage; because
    # novelty/prediction_error are renormalized away when absent, the two
    # priorities land on the same scale (equal here, since all signals agree).
    p_with = transition_priority(with_signals, weights)
    p_without = transition_priority(without_signals, weights)
    assert p_with == pytest.approx(p_without, abs=1e-6)


def test_transition_priority_is_never_zero():
    weights = PriorityWeights(reward=0.0, death=0.0, damage=0.0, novelty=0.0, prediction_error=0.0)
    dead_quiet = Transition(latent=[], action=0, reward=0.0, next_latent=[], done=False)
    assert transition_priority(dead_quiet, weights) > 0.0


# ---------------------------------------------------------- sampling distribution


def test_priority_sampling_distribution_matches_configured_weights():
    weights = PriorityWeights(reward=1.0, death=0.0, damage=0.0, novelty=0.0, prediction_error=0.0)
    buffer = ReplayBuffer(ReplayBufferConfig(capacity=10, alpha=1.0, seed=7, weights=weights))
    buffer.add(_transition("high", reward=10.0))
    for i in range(9):
        buffer.add(_transition(f"low{i}", reward=0.01))

    samples = buffer.sample(4000)
    high_fraction = sum(1 for t in samples if t.source == "high") / len(samples)

    priorities = buffer.priorities()
    expected_fraction = priorities[0] / sum(priorities)
    assert high_fraction == pytest.approx(expected_fraction, abs=0.03)
    assert high_fraction > 0.5  # heavily favored by the reward-only weighting


def test_sampling_is_deterministic_under_a_seeded_rng():
    def build():
        buffer = ReplayBuffer(ReplayBufferConfig(capacity=10, seed=123))
        for i in range(10):
            buffer.add(_transition(f"t{i}", reward=float(i)))
        return buffer

    a, b = build(), build()
    samples_a = [t.source for t in a.sample(50)]
    samples_b = [t.source for t in b.sample(50)]
    assert samples_a == samples_b


def test_different_seeds_usually_diverge():
    def build(seed):
        buffer = ReplayBuffer(ReplayBufferConfig(capacity=10, seed=seed))
        for i in range(10):
            buffer.add(_transition(f"t{i}", reward=float(i)))
        return buffer

    samples_a = [t.source for t in build(1).sample(50)]
    samples_b = [t.source for t in build(2).sample(50)]
    assert samples_a != samples_b


# ------------------------------------------------------------------- checkpoint


def test_replay_buffer_state_dict_round_trip_restores_counters_and_config():
    weights = PriorityWeights(reward=2.0, death=3.0, damage=0.1, novelty=0.2, prediction_error=0.3)
    buffer = ReplayBuffer(ReplayBufferConfig(capacity=4, alpha=0.8, seed=42, weights=weights))
    for i in range(6):
        buffer.add(_transition(f"t{i}", reward=float(i)))
    buffer.sample(3)
    buffer.sample(2)

    state = buffer.state_dict()

    restored = ReplayBuffer()
    restored.load_state_dict(state)

    assert restored.config.capacity == 4
    assert restored.config.alpha == pytest.approx(0.8)
    assert restored.config.seed == 42
    assert restored.config.weights == weights
    assert restored.total_added == buffer.total_added == 6
    assert restored.total_evicted == buffer.total_evicted == 2
    assert restored.total_sampled == buffer.total_sampled == 5
    # Contents are explicitly not part of the checkpoint (issue #28: "metadata;
    # contents optional").
    assert len(restored) == 0


def test_replay_buffer_metadata_round_trips_through_neural_agent_checkpoint(tmp_path):
    buffer = ReplayBuffer(ReplayBufferConfig(capacity=8, seed=1))
    for i in range(3):
        buffer.add(_transition(f"t{i}", reward=float(i)))
    buffer.sample(2)

    path = os.path.join(str(tmp_path), "ckpt.pt")
    checkpoint = NeuralAgentCheckpoint(
        path,
        layout_hash="layout-x",
        action_keys=["NULL", "JUMP"],
        replay_metadata=buffer.state_dict(),
    )
    checkpoint.save(reason="test")

    loaded = NeuralAgentCheckpoint(path, layout_hash="layout-x", action_keys=["NULL", "JUMP"])
    loaded.load(path)

    restored_buffer = ReplayBuffer()
    restored_buffer.load_state_dict(loaded.replay_metadata)
    assert restored_buffer.config.capacity == 8
    assert restored_buffer.config.seed == 1
    assert restored_buffer.total_added == 3
    assert restored_buffer.total_sampled == 2


# ------------------------------------------------------------- mixed schedule


def test_mixed_training_schedule_fires_replay_every_n_ticks_once_buffer_is_ready():
    schedule = MixedTrainingSchedule(replay_every_n_ticks=4, min_buffer_size=2)

    decisions = [schedule.on_tick(buffer_size=1) for _ in range(4)]
    assert all(d["on_policy"] for d in decisions)
    assert not any(d["replay"] for d in decisions)  # buffer never reached min size

    schedule.reset()
    decisions = [schedule.on_tick(buffer_size=10) for _ in range(8)]
    replay_ticks = [i for i, d in enumerate(decisions, start=1) if d["replay"]]
    assert replay_ticks == [4, 8]


# --------------------------------------------------------------- session loader


def test_load_session_into_buffer_and_iterate_minibatches(tmp_path):
    session_dir = _record_session(tmp_path, "replay-loader", ticks=120, seed=3)

    buffer = ReplayBuffer(ReplayBufferConfig(capacity=1000, seed=0))
    added = load_session_into_buffer(buffer, session_dir)

    assert added > 0
    assert len(buffer) == added
    assert buffer.total_added == added

    n_actions = len(ACTION_KEYS)
    batches = list(buffer.iter_minibatches(batch_size=8, n_actions=n_actions, n_batches=3))
    assert len(batches) == 3
    for batch in batches:
        assert batch["fused_latent"].shape[0] == 8
        assert batch["action_onehot"].shape == (8, n_actions)
        assert batch["reward"].shape == (8,)
        assert batch["next_fused_latent"].shape == batch["fused_latent"].shape
        assert batch["done"].shape == (8,)


def test_load_session_into_buffer_respects_max_transitions(tmp_path):
    session_dir = _record_session(tmp_path, "replay-loader-cap", ticks=120, seed=4)

    buffer = ReplayBuffer(ReplayBufferConfig(capacity=1000, seed=0))
    added = load_session_into_buffer(buffer, session_dir, max_transitions=10)

    assert added == 10
    assert len(buffer) == 10


def test_load_session_into_buffer_missing_directory_raises(tmp_path):
    buffer = ReplayBuffer()
    with pytest.raises(FileNotFoundError):
        load_session_into_buffer(buffer, os.path.join(str(tmp_path), "does-not-exist"))


# --------------------------------------------------- internal.* modulation streams


def test_load_session_into_buffer_reads_novelty_and_rpe_from_internal_streams(tmp_path):
    """Issue #58: the loader reads `internal.novelty`/
    `internal.reward_prediction_error` per tick instead of always leaving
    them `None` for a loaded session."""
    session_dir = _record_modulated_session(tmp_path, "modulated-session", ticks=60, seed=5)

    buffer = ReplayBuffer(ReplayBufferConfig(capacity=200, seed=0))
    added = load_session_into_buffer(buffer, session_dir)

    assert added > 0
    transitions = buffer.transitions()
    assert any(t.novelty is not None for t in transitions)
    assert any(t.reward_prediction_error is not None for t in transitions)


def test_replay_prioritization_is_driven_by_recorded_internal_streams(tmp_path):
    """Issue #58 acceptance: prioritization reads the recorded internal.*
    streams instead of recomputing anything reward-side. Weighting
    `reward_prediction_error` alone must differentiate transitions by their
    recorded RPE magnitude."""
    session_dir = _record_modulated_session(tmp_path, "modulated-priority", ticks=60, seed=6)

    zero_weights = PriorityWeights(
        reward=0.0, death=0.0, damage=0.0, novelty=0.0, prediction_error=0.0,
        reward_prediction_error=0.0,
    )
    buffer_flat = ReplayBuffer(ReplayBufferConfig(capacity=200, seed=0, weights=zero_weights))
    load_session_into_buffer(buffer_flat, session_dir)
    # Every configured weight at 0: every transition degrades to the same
    # eps floor -- proportional sampling has nothing to key off.
    assert len({round(p, 9) for p in buffer_flat.priorities()}) == 1

    rpe_weights = PriorityWeights(
        reward=0.0, death=0.0, damage=0.0, novelty=0.0, prediction_error=0.0,
        reward_prediction_error=1.0,
    )
    buffer_rpe = ReplayBuffer(ReplayBufferConfig(capacity=200, seed=0, weights=rpe_weights))
    load_session_into_buffer(buffer_rpe, session_dir)
    rpes = [t.reward_prediction_error for t in buffer_rpe.transitions()]
    assert any(r is not None for r in rpes)
    # Weighting the recorded reward-prediction-error alone now differentiates
    # transitions by it.
    assert len({round(p, 9) for p in buffer_rpe.priorities()}) > 1
    priority_by_rpe = {
        round(t.reward_prediction_error, 6): p
        for t, p in zip(buffer_rpe.transitions(), buffer_rpe.priorities())
        if t.reward_prediction_error is not None
    }
    biggest_surprise = max(priority_by_rpe, key=lambda rpe: abs(rpe))
    assert priority_by_rpe[biggest_surprise] == max(priority_by_rpe.values())
