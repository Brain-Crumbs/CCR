"""The Arbiter (issue #95): the hand-authored 2x2 (surprise, pain) truth
table, hysteresis against tick-to-tick flapping, the surprise calibrator,
and the Milestone 3 three-region scripted-scenario proof."""

from __future__ import annotations

import os

import pytest

from brain.amygdala import Amygdala
from brain.arbiter import (
    ARBITER_MODE_STREAM,
    FIGHT_OR_FLIGHT,
    INFO_GATHERING,
    REWARD_SEEKING,
    Arbiter,
    ArbiterConfig,
    SurpriseCalibrator,
    SurpriseCalibratorConfig,
)
from cognitive_runtime.core.world_model import Prediction, WorldModel
from cognitive_runtime.policies import ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import iter_cognitive_ticks

# --------------------------------------------------------------- config


def test_config_rejects_non_positive_hysteresis_ticks():
    with pytest.raises(ValueError, match="hysteresis_ticks"):
        ArbiterConfig(hysteresis_ticks=0)


def test_config_rejects_unknown_initial_mode():
    with pytest.raises(ValueError, match="initial_mode"):
        ArbiterConfig(initial_mode="curious")


# --------------------------------------------------------------- 2x2 truth table
#
# Hysteresis is disabled (k=1) for the truth-table tests themselves so each
# quadrant's *first* tick already reflects the raw lookup -- hysteresis gets
# its own dedicated tests below.


@pytest.mark.parametrize(
    "surprise, pain, expected",
    [
        (0.1, 0.1, REWARD_SEEKING),    # low surprise, safe -> bored
        (0.9, 0.1, INFO_GATHERING),    # high surprise, safe -> curious
        (0.1, 0.9, FIGHT_OR_FLIGHT),   # low surprise, threatened -> pain dominates
        (0.9, 0.9, FIGHT_OR_FLIGHT),   # high surprise, threatened -> afraid
    ],
)
def test_2x2_truth_table(surprise, pain, expected):
    arbiter = Arbiter(ArbiterConfig(hysteresis_ticks=1))
    assert arbiter.decide(surprise=surprise, pain=pain) == expected


def test_thresholds_are_inclusive_at_the_boundary():
    arbiter = Arbiter(ArbiterConfig(hysteresis_ticks=1, surprise_threshold=0.5, pain_threshold=0.5))
    assert arbiter.decide(surprise=0.5, pain=0.0) == INFO_GATHERING
    arbiter2 = Arbiter(ArbiterConfig(hysteresis_ticks=1, surprise_threshold=0.5, pain_threshold=0.5))
    assert arbiter2.decide(surprise=0.0, pain=0.5) == FIGHT_OR_FLIGHT


def test_starts_in_the_configured_initial_mode_before_any_decision():
    arbiter = Arbiter(ArbiterConfig(initial_mode=INFO_GATHERING))
    assert arbiter.mode == INFO_GATHERING


# --------------------------------------------------------------- hysteresis


def test_single_tick_blip_does_not_flip_the_mode():
    arbiter = Arbiter(ArbiterConfig(hysteresis_ticks=3))
    assert arbiter.mode == REWARD_SEEKING
    # One high-surprise tick, then back to bored -- the blip must not stick.
    assert arbiter.decide(surprise=0.9, pain=0.0) == REWARD_SEEKING
    assert arbiter.decide(surprise=0.1, pain=0.0) == REWARD_SEEKING
    assert arbiter.mode == REWARD_SEEKING


def test_sustained_crossing_flips_after_exactly_k_ticks():
    arbiter = Arbiter(ArbiterConfig(hysteresis_ticks=3))
    assert arbiter.decide(surprise=0.9, pain=0.0) == REWARD_SEEKING  # streak 1
    assert arbiter.decide(surprise=0.9, pain=0.0) == REWARD_SEEKING  # streak 2
    assert arbiter.decide(surprise=0.9, pain=0.0) == INFO_GATHERING  # streak 3 -> flips
    assert arbiter.mode == INFO_GATHERING


def test_streak_resets_when_the_challenger_changes():
    """A run of ticks toward one candidate mode, interrupted by a *different*
    candidate, must not accumulate across the switch."""
    arbiter = Arbiter(ArbiterConfig(hysteresis_ticks=3))
    arbiter.decide(surprise=0.9, pain=0.0)  # candidate=info_gathering, streak 1
    arbiter.decide(surprise=0.9, pain=0.0)  # streak 2
    arbiter.decide(surprise=0.9, pain=0.9)  # candidate=fight_or_flight, streak resets to 1
    assert arbiter.mode == REWARD_SEEKING  # never reached streak 3 for either candidate
    arbiter.decide(surprise=0.9, pain=0.9)  # streak 2
    arbiter.decide(surprise=0.9, pain=0.9)  # streak 3 -> flips
    assert arbiter.mode == FIGHT_OR_FLIGHT


def test_returning_to_the_active_mode_clears_a_pending_streak():
    arbiter = Arbiter(ArbiterConfig(hysteresis_ticks=3))
    arbiter.decide(surprise=0.9, pain=0.0)  # streak 1 toward info_gathering
    arbiter.decide(surprise=0.9, pain=0.0)  # streak 2
    arbiter.decide(surprise=0.1, pain=0.0)  # back to reward_seeking -- clears the streak
    arbiter.decide(surprise=0.9, pain=0.0)  # streak 1 again, not 3
    arbiter.decide(surprise=0.9, pain=0.0)  # streak 2
    assert arbiter.mode == REWARD_SEEKING


def test_hysteresis_also_applies_leaving_fight_or_flight():
    arbiter = Arbiter(ArbiterConfig(hysteresis_ticks=2, initial_mode=FIGHT_OR_FLIGHT))
    assert arbiter.decide(surprise=0.1, pain=0.1) == FIGHT_OR_FLIGHT  # streak 1
    assert arbiter.decide(surprise=0.1, pain=0.1) == REWARD_SEEKING  # streak 2 -> flips


def test_reset_returns_to_initial_mode_and_clears_pending_streak():
    arbiter = Arbiter(ArbiterConfig(hysteresis_ticks=3))
    arbiter.decide(surprise=0.9, pain=0.0)
    arbiter.decide(surprise=0.9, pain=0.0)
    arbiter.reset()
    assert arbiter.mode == REWARD_SEEKING
    # The pending streak was cleared too -- two more highs shouldn't finish it.
    arbiter.decide(surprise=0.9, pain=0.0)
    arbiter.decide(surprise=0.9, pain=0.0)
    assert arbiter.mode == REWARD_SEEKING


# --------------------------------------------------------------- payload


def test_as_payload_reports_mode_surprise_pain_and_calibration():
    arbiter = Arbiter(ArbiterConfig(hysteresis_ticks=1))
    arbiter.decide(surprise=0.42, pain=0.07)
    payload = arbiter.as_payload(calibration_error=0.031)
    assert payload == {
        "mode": REWARD_SEEKING,
        "surprise": pytest.approx(0.42),
        "pain": pytest.approx(0.07),
        "calibration_error": pytest.approx(0.031),
    }


def test_as_payload_calibration_error_is_none_before_a_fit():
    arbiter = Arbiter()
    arbiter.decide(surprise=0.1, pain=0.1)
    assert arbiter.as_payload()["calibration_error"] is None


def test_arbiter_mode_stream_id_is_the_named_internal_stream():
    assert ARBITER_MODE_STREAM == "internal.arbiter_mode"


# --------------------------------------------------------------- SurpriseCalibrator


def test_calibrator_config_rejects_invalid_window():
    with pytest.raises(ValueError, match="window"):
        SurpriseCalibratorConfig(window=1)


def test_calibrator_config_rejects_invalid_outcome_quantile():
    with pytest.raises(ValueError, match="outcome_quantile"):
        SurpriseCalibratorConfig(outcome_quantile=1.0)


def test_calibrator_starts_with_a_pass_through_temperature_and_no_error():
    calibrator = SurpriseCalibrator()
    assert calibrator.temperature == pytest.approx(1.0)
    assert calibrator.calibration_error is None


def test_calibrator_update_returns_a_bounded_probability():
    calibrator = SurpriseCalibrator()
    for raw in (0.0, 0.5, 3.0, 1000.0):
        calibrated = calibrator.update(raw)
        assert 0.0 <= calibrated < 1.0


def test_calibrator_fits_and_reports_calibration_error_once_enough_observations_arrive():
    calibrator = SurpriseCalibrator(
        SurpriseCalibratorConfig(window=50, min_observations=20, refit_every=20)
    )
    import random

    rng = random.Random(7)
    for _ in range(40):
        calibrator.update(rng.random())
    assert calibrator.calibration_error is not None
    assert calibrator.calibration_error >= 0.0


def test_calibrator_reset_clears_the_fitted_temperature():
    calibrator = SurpriseCalibrator(
        SurpriseCalibratorConfig(window=50, min_observations=10, refit_every=10)
    )
    import random

    rng = random.Random(8)
    for _ in range(30):
        calibrator.update(rng.random())
    assert calibrator.calibration_error is not None
    calibrator.reset()
    assert calibrator.temperature == pytest.approx(1.0)
    assert calibrator.calibration_error is None


def test_calibrator_stays_strictly_below_one_even_at_a_low_fitted_temperature():
    """A sharpening (low) fitted temperature pushes `logit(bounded) /
    temperature` far enough that `sigmoid` alone would underflow to exactly
    `1.0` for a large-enough raw reading -- `update()` must still honor its
    documented `[0, 1)` range so `ArbiterConfig.surprise_threshold`
    comparisons never degenerate."""
    calibrator = SurpriseCalibrator()
    calibrator._temperature = 0.15  # a real value `fit_temperature` can select
    calibrated = calibrator.update(1000.0)
    assert calibrated < 1.0


def test_calibrator_higher_raw_reading_never_yields_a_lower_calibrated_value():
    """Temperature scaling must stay monotone: it reshapes confidence, it
    never reorders it."""
    calibrator = SurpriseCalibrator()
    low = calibrator.update(0.2)
    high = calibrator.update(0.8)
    assert high > low


# --------------------------------------------------------------- Milestone 3:
# the three-region scripted-scenario test.


class _ScriptedThreeRegionWorldModel(WorldModel):
    """Scripts (risk, prediction_error) through three ten-tick regions so a
    real `CognitiveRuntime` run exercises all three arbiter modes end to
    end, deterministically: bored (low error, low risk), a harmless
    surprise (high error, low risk), then a harmful one (high error, high
    risk) -- the phase doc's "scripted scene with a harmless surprise and a
    harmful one" (Milestone 3 exit gate)."""

    BORED, HARMLESS_SURPRISE, HARMFUL_SURPRISE = range(3)

    def __init__(self):
        self.tick = 0

    def region(self) -> int:
        if self.tick < 10:
            return self.BORED
        if self.tick < 20:
            return self.HARMLESS_SURPRISE
        return self.HARMFUL_SURPRISE

    def predict(self, state, memory) -> Prediction:
        region = self.region()
        self.tick += 1
        if region == self.BORED:
            return Prediction(risk=0.02, prediction_error=0.02, predicted_reward=0.0)
        if region == self.HARMLESS_SURPRISE:
            # A raw prediction_error this large safely clears the surprise
            # calibrator's saturating `x / (1 + x)` transform (issue #95's
            # `SurpriseCalibrator.update`) above `ArbiterConfig.
            # surprise_threshold` (0.5) -- `5.0 -> 5/6 ~= 0.83`.
            return Prediction(risk=0.02, prediction_error=5.0, predicted_reward=0.0)
        return Prediction(risk=0.95, prediction_error=5.0, predicted_reward=0.0)

    def reset(self) -> None:
        self.tick = 0


class _UncertaintyOnlyWorldModel(WorldModel):
    """``prediction_error`` stays at 0.0 (would calibrate as "boring") the
    whole run; ``predicted_uncertainty`` stays at 5.0 (must calibrate as
    "surprising") -- proves the arbiter's raw-surprise input is sourced from
    a dedicated ``predicted_uncertainty`` head (issue #169) whenever a
    ``WorldModel`` bridge exposes one, not from ``prediction_error``."""

    def predict(self, state, memory) -> Prediction:
        return Prediction(
            risk=0.02, prediction_error=0.0, predicted_reward=0.0, predicted_uncertainty=5.0
        )

    def reset(self) -> None:
        pass


def test_arbiter_surprise_input_prefers_predicted_uncertainty_over_prediction_error(tmp_path):
    config = {"episode_ticks": 30, "world_size": 16, "max_mobs": 1}
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=0,
        max_ticks_per_episode=30,
        record_dir=str(tmp_path),
        session_id="arbiter-uncertainty-source",
        program_config=config,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config),
        policy=ScriptedSurvivalPolicy(seed=0),
        config=runtime_config,
        world_model=_UncertaintyOnlyWorldModel(),
    ).run()

    session_dir = os.path.join(str(tmp_path), "arbiter-uncertainty-source")
    surprises = []
    for decision, sensory, _motor in iter_cognitive_ticks(session_dir, "episode_00000"):
        for record in sensory:
            if record.get("stream_id") == ARBITER_MODE_STREAM and not record.get("elided"):
                surprises.append(record["payload"]["surprise"])

    assert surprises, "expected the arbiter mode stream to be recorded"
    # Pass-through temperature (1.0) until the calibrator's window is full
    # enough to refit, and a constant reading never triggers a refit (see
    # SurpriseCalibrator._refit's degenerate-window guard) -- so a raw 5.0
    # calibrates to 5/6 ~= 0.83 every tick. Sourced from prediction_error
    # (always 0.0) it would instead sit at ~0.0, well under the arbiter's
    # 0.5 surprise_threshold.
    assert all(s > 0.5 for s in surprises)


def test_milestone_3_three_region_scripted_scenario(tmp_path):
    """Milestone 3 exit gate: a harmless surprise drives info-gathering, a
    harmful one drives fight-or-flight, and boredom drives reward-seeking
    -- each visible in the recorded arbiter-mode timeline."""
    config = {"episode_ticks": 30, "world_size": 16, "max_mobs": 1}
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=0,
        max_ticks_per_episode=30,
        record_dir=str(tmp_path),
        session_id="arbiter-milestone-3",
        program_config=config,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config),
        policy=ScriptedSurvivalPolicy(seed=0),
        config=runtime_config,
        world_model=_ScriptedThreeRegionWorldModel(),
    ).run()

    session_dir = os.path.join(str(tmp_path), "arbiter-milestone-3")
    modes_by_tick = {}
    for decision, sensory, _motor in iter_cognitive_ticks(session_dir, "episode_00000"):
        for record in sensory:
            if record.get("stream_id") == ARBITER_MODE_STREAM and not record.get("elided"):
                modes_by_tick[decision["tick_index"]] = record["payload"]["mode"]

    assert modes_by_tick, "expected the arbiter mode stream to be recorded"
    ticks = sorted(modes_by_tick)
    # One-tick publication lag (matches every other internal.* stream: it is
    # computed after this tick's prediction, so it first appears in the next
    # tick's window) -- the last few ticks of each 10-tick region are what
    # prove the mode settled there, once hysteresis has had time to catch up.
    bored_settled = modes_by_tick[max(t for t in ticks if t < 10)]
    harmless_settled = modes_by_tick[max(t for t in ticks if 10 <= t < 20)]
    harmful_settled = modes_by_tick[max(t for t in ticks if 20 <= t < 30)]

    assert bored_settled == REWARD_SEEKING
    assert harmless_settled == INFO_GATHERING
    assert harmful_settled == FIGHT_OR_FLIGHT
