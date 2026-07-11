"""Internal modulation streams (issue #58): EMA math, RPE sign, and a
simulated run recording all five `internal.*` streams every tick."""

from __future__ import annotations

import os

import pytest

from cognitive_runtime.core.modulation import (
    INTERNAL_MODULATION_STREAM_IDS,
    LEARNING_PROGRESS_STREAM,
    NOVELTY_STREAM,
    PREDICTED_RISK_AVERSION_STREAM,
    PREDICTION_ERROR_STREAM,
    REWARD_PREDICTION_ERROR_STREAM,
    RISK_GATE_STREAM,
    RISK_STREAM,
    SAFE_NOVELTY_STREAM,
    LearningProgressTracker,
    ModulationTracker,
    compute_reward_prediction_error,
    safe_gate,
)
from cognitive_runtime.core.world_model import Prediction, WorldModel
from cognitive_runtime.policies import ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import iter_cognitive_ticks
from cognitive_runtime.tools.episode_viewer import view_episode


# ------------------------------------------------------------ learning progress


def test_learning_progress_is_positive_when_error_is_improving():
    tracker = LearningProgressTracker(fast_alpha=0.5, slow_alpha=0.05)
    errors = [1.0 - 0.03 * i for i in range(40)]  # steadily decreasing
    signal = None
    for error in errors:
        signal = tracker.update(error)
    assert signal > 0.0


def test_learning_progress_is_near_zero_when_error_has_plateaued():
    tracker = LearningProgressTracker(fast_alpha=0.5, slow_alpha=0.05)
    signal = None
    for _ in range(60):
        signal = tracker.update(0.4)  # constant: nothing left to learn
    assert signal == pytest.approx(0.0, abs=1e-6)


def test_learning_progress_is_near_zero_when_error_is_noisy_but_static():
    tracker = LearningProgressTracker(fast_alpha=0.5, slow_alpha=0.05)
    # Deterministic pseudo-noise oscillating around a fixed mean -- no trend.
    signals = []
    for i in range(200):
        noisy = 0.5 + (0.1 if i % 2 == 0 else -0.1)
        signals.append(tracker.update(noisy))
    tail = signals[-40:]
    assert sum(tail) / len(tail) == pytest.approx(0.0, abs=0.02)


def test_learning_progress_first_sample_is_zero_not_none():
    tracker = LearningProgressTracker()
    assert tracker.update(0.7) == 0.0


def test_learning_progress_none_when_no_prediction_error():
    tracker = LearningProgressTracker()
    tracker.update(0.5)
    assert tracker.update(None) is None


def test_learning_progress_tracker_rejects_slow_alpha_not_less_than_fast():
    with pytest.raises(ValueError, match="slow_alpha"):
        LearningProgressTracker(fast_alpha=0.1, slow_alpha=0.2)


def test_learning_progress_state_dict_round_trips():
    tracker = LearningProgressTracker()
    for error in (1.0, 0.8, 0.6):
        tracker.update(error)
    state = tracker.state_dict()

    restored = LearningProgressTracker()
    restored.load_state_dict(state)
    assert restored.update(0.6) == pytest.approx(tracker.update(0.6))


# ------------------------------------------------------- reward prediction error


def test_reward_prediction_error_sign_against_scripted_reward_sequence():
    # Predicted reward stays flat; actual reward swings above and below it.
    cases = [
        (1.0, 0.4, "positive"),   # got more than expected -> positive surprise
        (-1.0, 0.4, "negative"),  # got less than expected -> negative surprise
        (0.4, 0.4, "zero"),       # exactly as predicted -> no surprise
    ]
    for actual, predicted, expectation in cases:
        rpe = compute_reward_prediction_error(actual, predicted)
        if expectation == "positive":
            assert rpe > 0.0
        elif expectation == "negative":
            assert rpe < 0.0
        else:
            assert rpe == pytest.approx(0.0)
        assert rpe == pytest.approx(actual - predicted)


def test_reward_prediction_error_is_none_without_a_reward_head():
    assert compute_reward_prediction_error(1.0, None) is None


# --------------------------------------------------------------- ModulationTracker


def test_modulation_tracker_publishes_risk_and_gated_terms_every_tick_even_without_other_signals():
    tracker = ModulationTracker()
    signals = tracker.update(Prediction(risk=0.25), entity_surprise=None, actual_reward=0.0)
    payloads = signals.as_payloads()
    # risk, risk_gate and predicted_risk_aversion are always available
    # (Prediction.risk defaults to 0.0, never None); safe_novelty is not,
    # since there is no novelty signal this tick to gate.
    assert payloads.keys() == {RISK_STREAM, RISK_GATE_STREAM, PREDICTED_RISK_AVERSION_STREAM}
    assert payloads[RISK_STREAM] == {"value": 0.25}
    assert payloads[PREDICTED_RISK_AVERSION_STREAM] == {"value": -0.25}
    assert 0.0 < payloads[RISK_GATE_STREAM]["value"] < 1.0


def test_modulation_tracker_publishes_all_eight_when_every_signal_is_available():
    tracker = ModulationTracker()
    prediction = Prediction(risk=0.1, predicted_reward=0.2, prediction_error=0.5)
    signals = tracker.update(prediction, entity_surprise=0.3, actual_reward=0.6)
    payloads = signals.as_payloads()
    assert set(payloads) == set(INTERNAL_MODULATION_STREAM_IDS)
    assert payloads[PREDICTION_ERROR_STREAM] == {"value": 0.5}
    assert payloads[REWARD_PREDICTION_ERROR_STREAM] == {"value": pytest.approx(0.4)}
    assert payloads[NOVELTY_STREAM] == {"value": pytest.approx(0.4)}  # mean(0.5, 0.3)
    assert payloads[LEARNING_PROGRESS_STREAM] == {"value": 0.0}  # first sample
    assert payloads[PREDICTED_RISK_AVERSION_STREAM] == {"value": -0.1}
    # risk=0.1 is well below the default 0.5 threshold -> gate close to 1,
    # so safe_novelty is close to (uncapped) novelty.
    assert payloads[RISK_GATE_STREAM]["value"] > 0.9
    assert payloads[SAFE_NOVELTY_STREAM]["value"] == pytest.approx(
        payloads[NOVELTY_STREAM]["value"] * payloads[RISK_GATE_STREAM]["value"], abs=1e-5
    )


# ------------------------------------------------------------------ safe_gate


def test_safe_gate_is_one_half_exactly_at_the_threshold():
    assert safe_gate(0.5, risk_threshold=0.5, temperature=0.15) == pytest.approx(0.5)


def test_safe_gate_approaches_one_well_below_threshold():
    assert safe_gate(0.0, risk_threshold=0.5, temperature=0.05) > 0.999


def test_safe_gate_approaches_zero_well_above_threshold():
    assert safe_gate(1.0, risk_threshold=0.5, temperature=0.05) < 0.001


def test_safe_gate_is_monotonically_decreasing_in_risk():
    risks = [round(0.05 * i, 2) for i in range(21)]  # 0.0 .. 1.0
    gates = [safe_gate(r, risk_threshold=0.5, temperature=0.15) for r in risks]
    assert all(a >= b for a, b in zip(gates, gates[1:]))


def test_safe_gate_rejects_non_positive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        safe_gate(0.5, risk_threshold=0.5, temperature=0.0)


def test_modulation_tracker_risk_threshold_and_temperature_are_configurable():
    permissive = ModulationTracker(risk_threshold=0.9, temperature=0.15)
    strict = ModulationTracker(risk_threshold=0.1, temperature=0.15)
    prediction = Prediction(risk=0.5, prediction_error=0.4)
    permissive_gate = permissive.update(prediction, None, 0.0).risk_gate
    strict_gate = strict.update(prediction, None, 0.0).risk_gate
    # Same risk reading, different thresholds: a lenient threshold (0.9)
    # still treats risk=0.5 as safe; a strict one (0.1) treats it as
    # dangerous.
    assert permissive_gate > 0.9
    assert strict_gate < 0.1


def test_modulation_tracker_state_dict_round_trips():
    tracker = ModulationTracker()
    tracker.update(Prediction(risk=0.0, prediction_error=0.5), None, 0.0)
    tracker.update(Prediction(risk=0.0, prediction_error=0.4), None, 0.0)
    state = tracker.state_dict()

    restored = ModulationTracker()
    restored.load_state_dict(state)
    a = tracker.update(Prediction(risk=0.0, prediction_error=0.3), None, 0.0)
    b = restored.update(Prediction(risk=0.0, prediction_error=0.3), None, 0.0)
    assert a == pytest.approx(b)


# ---------------------------------------------------------------- simulated run


class _FakeWorldModel(WorldModel):
    """Deterministic non-None prediction/reward/error every tick, so a short
    run exercises all eight `internal.*` streams without a trained model."""

    def __init__(self):
        self.tick = 0

    def predict(self, state, memory) -> Prediction:
        self.tick += 1
        return Prediction(
            risk=0.1,
            predicted_reward=0.05,
            prediction_error=max(0.01, 0.5 - 0.01 * self.tick),
            next_latent=[0.0],
        )

    def reset(self) -> None:
        self.tick = 0


def test_simulated_run_records_all_eight_internal_streams_every_tick(tmp_path):
    config = {"episode_ticks": 15, "world_size": 16, "max_mobs": 1}
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=0,
        max_ticks_per_episode=15,
        record_dir=str(tmp_path),
        session_id="modulation-session",
        program_config=config,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config),
        policy=ScriptedSurvivalPolicy(seed=0),
        config=runtime_config,
        world_model=_FakeWorldModel(),
    ).run()

    # Runtime-computed streams (like the existing `model.novelty`) are
    # published after a tick's window is already collected, so they first
    # appear in the *next* tick's window -- tick 0 never carries them.
    session_dir = os.path.join(str(tmp_path), "modulation-session")
    ticks_seen = 0
    ticks_with_all_eight = 0
    for decision, sensory, _motor in iter_cognitive_ticks(session_dir, "episode_00000"):
        ids = {r["stream_id"] for r in sensory if not r.get("elided")}
        ticks_seen += 1
        if set(INTERNAL_MODULATION_STREAM_IDS) <= ids:
            ticks_with_all_eight += 1
    assert ticks_seen == 15
    assert ticks_with_all_eight == 14

    rendered = view_episode(session_dir, "episode_00000")
    for stream_id in INTERNAL_MODULATION_STREAM_IDS:
        assert stream_id in rendered
