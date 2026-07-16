"""The three human-named neuromodulators over existing math (issue #94,
task 1): dopamine mirrors reward-prediction error with no new math,
acetylcholine's precision term, and the `cognitive_runtime.core.modulation`
re-export shim."""

from __future__ import annotations

import os

import pytest

import cognitive_runtime.core.modulation as modulation_shim
from brain.amygdala import ADRENALINE_STREAM, Amygdala
from brain.neuromod import (
    ACETYLCHOLINE_STREAM,
    DOPAMINE_STREAM,
    NAMED_NEUROMODULATOR_STREAM_IDS,
    ModulationSignals,
    compute_acetylcholine,
    named_neuromodulator_payloads,
)
from cognitive_runtime.core.world_model import Prediction, WorldModel
from cognitive_runtime.policies import ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import iter_cognitive_ticks
from cognitive_runtime.tools.episode_viewer import view_episode


# ------------------------------------------------------------------ shim identity


def test_modulation_shim_reexports_the_same_objects_as_brain_neuromod():
    """`cognitive_runtime.core.modulation` (issue #58) is now a thin
    re-export shim over `brain.neuromod.modulation` (issue #94): every
    name it used to define must resolve to the exact same object, not a
    reimplementation, so existing pickled state / isinstance checks /
    imports throughout the codebase keep working unchanged."""
    import brain.neuromod.modulation as promoted

    for name in modulation_shim.__all__:
        assert getattr(modulation_shim, name) is getattr(promoted, name), name


# ------------------------------------------------------------------- dopamine


def test_dopamine_is_reward_prediction_error_with_no_new_math():
    signals = ModulationSignals(
        prediction_error=0.2, reward_prediction_error=0.37, learning_progress=0.0,
        novelty=0.1, risk=0.05, risk_gate=0.9, safe_novelty=0.09, predicted_risk_aversion=-0.05,
    )
    payloads = named_neuromodulator_payloads(signals, acetylcholine=0.0, adrenaline=0.0)
    assert payloads[DOPAMINE_STREAM] == {"value": pytest.approx(0.37)}
    assert payloads[DOPAMINE_STREAM]["value"] == pytest.approx(signals.reward_prediction_error)


def test_dopamine_omitted_when_no_reward_head_this_tick():
    signals = ModulationSignals(
        prediction_error=None, reward_prediction_error=None, learning_progress=None,
        novelty=None, risk=0.0, risk_gate=1.0, safe_novelty=None, predicted_risk_aversion=0.0,
    )
    payloads = named_neuromodulator_payloads(signals, acetylcholine=0.0, adrenaline=0.0)
    assert DOPAMINE_STREAM not in payloads
    assert set(payloads) == {ACETYLCHOLINE_STREAM, ADRENALINE_STREAM}


# --------------------------------------------------------------- acetylcholine


def test_acetylcholine_is_quiescent_without_an_uncertainty_reading():
    assert compute_acetylcholine(uncertainty=None, learning_progress=0.5) == 0.0


def test_acetylcholine_rises_with_uncertainty():
    low = compute_acetylcholine(uncertainty=0.1, learning_progress=0.0)
    high = compute_acetylcholine(uncertainty=0.9, learning_progress=0.0)
    assert 0.0 <= low < high < 1.0


def test_acetylcholine_rises_further_when_learning_has_stalled():
    improving = compute_acetylcholine(uncertainty=0.5, learning_progress=0.3)
    stalled = compute_acetylcholine(uncertainty=0.5, learning_progress=-0.3)
    assert improving < stalled


def test_acetylcholine_is_bounded_below_one_even_for_a_large_sigma_spike():
    assert compute_acetylcholine(uncertainty=1000.0, learning_progress=-1000.0) < 1.0


# ------------------------------------------------------------- amygdala wiring


def test_adrenaline_payload_passes_through_the_supplied_amygdala_level():
    amygdala = Amygdala()
    for _ in range(5):
        adrenaline = amygdala.appraise(risk=0.9)
    signals = ModulationSignals(
        prediction_error=None, reward_prediction_error=None, learning_progress=None,
        novelty=None, risk=0.9, risk_gate=0.0, safe_novelty=None, predicted_risk_aversion=-0.9,
    )
    payloads = named_neuromodulator_payloads(signals, acetylcholine=0.0, adrenaline=adrenaline)
    assert payloads[ADRENALINE_STREAM]["value"] == pytest.approx(adrenaline, abs=1e-6)


# --------------------------------------------------------------- simulated run


class _FakeWorldModel(WorldModel):
    """Deterministic non-None prediction/reward/error every tick (mirrors
    `test_modulation.py`'s fixture), so a short run exercises all three
    named `internal.*` streams without a trained model."""

    def __init__(self):
        self.tick = 0

    def predict(self, state, memory) -> Prediction:
        self.tick += 1
        return Prediction(
            risk=0.6,
            predicted_reward=0.05,
            prediction_error=max(0.01, 0.5 - 0.01 * self.tick),
            next_latent=[0.0],
        )

    def reset(self) -> None:
        self.tick = 0


def test_simulated_run_records_the_three_named_neuromodulator_streams(tmp_path):
    config = {"episode_ticks": 15, "world_size": 16, "max_mobs": 1}
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=0,
        max_ticks_per_episode=15,
        record_dir=str(tmp_path),
        session_id="neuromod-session",
        program_config=config,
    )
    CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config),
        policy=ScriptedSurvivalPolicy(seed=0),
        config=runtime_config,
        world_model=_FakeWorldModel(),
    ).run()

    session_dir = os.path.join(str(tmp_path), "neuromod-session")
    ticks_seen = 0
    ticks_with_ach_and_adrenaline = 0
    for decision, sensory, _motor in iter_cognitive_ticks(session_dir, "episode_00000"):
        ids = {r["stream_id"] for r in sensory if not r.get("elided")}
        ticks_seen += 1
        # acetylcholine/adrenaline publish every tick (dopamine only when a
        # reward head is available, same availability rule as
        # reward_prediction_error).
        if {ACETYLCHOLINE_STREAM, ADRENALINE_STREAM} <= ids:
            ticks_with_ach_and_adrenaline += 1
    assert ticks_seen == 15
    # One-tick lag (module docstring): runtime-computed streams first show
    # up in the window *after* the tick that computed them.
    assert ticks_with_ach_and_adrenaline == 14

    rendered = view_episode(session_dir, "episode_00000")
    for stream_id in NAMED_NEUROMODULATOR_STREAM_IDS:
        assert stream_id in rendered
