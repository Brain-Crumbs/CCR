"""Predictive-cortex live world-model bridge (issue #166).

Covers the acceptance criteria: the cortex hidden state persists across ticks
within an episode and resets between episodes, prediction error is derived from
the cortex's own forecast, and a recorded session run through the bridge
publishes cortex-sourced novelty/prediction-error telemetry.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brain.cortex.predictive import PredictiveCortex, PredictiveCortexConfig  # noqa: E402
from cognitive_runtime.core.action import Action  # noqa: E402
from cognitive_runtime.core.memory import Memory  # noqa: E402
from cognitive_runtime.core.perception import State  # noqa: E402
from cognitive_runtime.core.streams.events import StreamEvent  # noqa: E402
from cognitive_runtime.neural.pixel_stream_encoder import PIXEL_STREAM_ID  # noqa: E402
from cognitive_runtime.policies import ScriptedSurvivalPolicy  # noqa: E402
from cognitive_runtime.policies.cortex_world_model import CortexWorldModel  # noqa: E402
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox  # noqa: E402
from cognitive_runtime.programs.minecraft.streams import PIXEL_SHAPE  # noqa: E402
from cognitive_runtime.runtime.config import RuntimeConfig  # noqa: E402
from cognitive_runtime.runtime.loop import CognitiveRuntime  # noqa: E402
from cognitive_runtime.runtime.replay import iter_cognitive_ticks  # noqa: E402

_ACTION_KEYS = ["noop", "move_forward", "turn_left", "turn_right"]


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


def test_first_tick_has_no_prediction_error_then_forecast_drives_it():
    """No prior forecast on the first tick (like the heuristic model); every
    subsequent tick scores the cortex's own one-step forecast."""
    wm = CortexWorldModel(_small_cortex(), action_keys=_ACTION_KEYS)
    memory = Memory()
    rng = np.random.default_rng(1)
    state = State(observation=None)

    _push_frame(memory, _frame(rng), 0)
    first = wm.predict(state, memory)
    assert first.prediction_error is None
    assert 0.0 <= first.risk <= 1.0
    assert first.p_death is not None and 0.0 <= first.p_death <= 1.0
    assert first.next_latent is not None and len(first.next_latent) == 8
    # issue #169: the cortex's own uncertainty head, not just risk/p_death,
    # rides along in Prediction -- the arbiter's dedicated sigma source.
    assert first.predicted_uncertainty is not None and first.predicted_uncertainty >= 0.0

    for seq in range(1, 4):
        memory.record_action(Action.from_key("move_forward"))
        _push_frame(memory, _frame(rng), seq)
        pred = wm.predict(state, memory)
        assert pred.prediction_error is not None
        assert pred.prediction_error >= 0.0
        assert pred.predicted_uncertainty is not None and pred.predicted_uncertainty >= 0.0


def test_hidden_state_persists_across_ticks_and_resets_between_episodes():
    """The backbone hidden state advances tick to tick within an episode and is
    cleared by reset() on the episode boundary."""
    wm = CortexWorldModel(_small_cortex(), action_keys=_ACTION_KEYS)
    memory = Memory()
    rng = np.random.default_rng(2)
    state = State(observation=None)

    assert wm._hidden is None and wm._predicted_latent is None

    _push_frame(memory, _frame(rng), 0)
    wm.predict(state, memory)
    hidden_after_first = wm._hidden
    assert hidden_after_first is not None
    assert wm._predicted_latent is not None

    # A second tick with the *same* frame still advances the recurrent state,
    # so a memoryless model could not produce this difference.
    same_frame = _frame(rng)
    _push_frame(memory, same_frame, 1)
    memory.record_action(Action.from_key("noop"))
    wm.predict(state, memory)
    forecast_tick_2 = wm._predicted_latent.clone()

    _push_frame(memory, same_frame, 2)
    memory.record_action(Action.from_key("noop"))
    wm.predict(state, memory)
    forecast_tick_3 = wm._predicted_latent.clone()

    assert not torch.allclose(forecast_tick_2, forecast_tick_3), (
        "identical inputs produced identical forecasts -- hidden state is not "
        "actually carried across ticks"
    )

    # Episode boundary: the loop calls reset().
    wm.reset()
    assert wm._hidden is None and wm._predicted_latent is None

    _push_frame(memory, _frame(rng), 3)
    first_of_next_episode = wm.predict(state, memory)
    assert first_of_next_episode.prediction_error is None


def test_mismatched_pixel_shape_is_rejected():
    wm = CortexWorldModel(_small_cortex(pixel_shape=(8, 8, 3)), action_keys=_ACTION_KEYS)
    memory = Memory()
    _push_frame(memory, np.zeros((10, 10, 3), dtype=np.uint8), 0)
    with pytest.raises(ValueError, match="pixel-frame shape"):
        wm.predict(State(observation=None), memory)


def test_cortex_bridges_prediction_into_recorded_session(tmp_path):
    """A recorded Crafter-style session run with the cortex bridge publishes
    prediction-error/novelty derived from the cortex's own forecast, and the
    per-tick decision telemetry carries the cortex's risk/p_death."""
    program_config = {"episode_ticks": 40, "world_size": 16}
    action_keys = [
        a.key() for a in MinecraftSurvivalBox(config=program_config).metadata().action_space
    ]
    cortex = PredictiveCortex(
        PIXEL_SHAPE,
        action_keys,
        PredictiveCortexConfig(
            latent_width=8, hidden_dim=16, reconstruction_size=8, horizons_ticks=(1, 4)
        ),
    )
    world_model = CortexWorldModel(cortex, action_keys=action_keys)

    session_id = "cortex-bridge-run"
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=5,
        max_ticks_per_episode=40,
        record_dir=str(tmp_path),
        session_id=session_id,
        program_config=program_config,
    )
    summaries = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=program_config),
        policy=ScriptedSurvivalPolicy(seed=6),
        config=runtime_config,
        world_model=world_model,
    ).run()

    assert summaries[0].avg_prediction_error is not None

    session_dir = os.path.join(str(tmp_path), session_id)
    saw_p_death = False
    for decision, _sensory, _motor in iter_cognitive_ticks(session_dir, summaries[0].episode_id):
        if decision.get("p_death") is not None:
            saw_p_death = True
            assert 0.0 <= decision["risk"] <= 1.0
    assert saw_p_death
