"""Internal modulation streams (issue #58): the dopamine-analog signals,
modeled as first-class ``internal.*`` streams rather than hidden variables or
decision-record-only fields.

Five per-tick interoceptive streams:

- ``internal.prediction_error`` — world-model next-latent error (already
  computed by ``core.world_model.Prediction``; this promotes it to a stream).
- ``internal.reward_prediction_error`` — actual reward minus the world
  model's predicted reward. The dopamine analog: it modulates replay
  priority and memory tagging, not the whole attention system.
- ``internal.learning_progress`` — is prediction error improving? Positive
  means "getting better at predicting this", not raw error.
- ``internal.novelty`` — the combined novelty scalar (issue #27), promoted
  to a stream.
- ``internal.risk`` — the world model's risk/p_death head output: predicted
  pain/injury/death made visible.

Every payload is the uniform ``{"value": <float>}`` shape the reward
engine's ``intrinsic_stream`` component kind already expects
(``programs/minecraft/reward_profile.py``), so any of these streams can be
wired straight into a reward profile's ``intrinsic`` slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from cognitive_runtime.core.novelty import combine_novelty
from cognitive_runtime.core.streams.events import StreamSpec
from cognitive_runtime.core.world_model import Prediction

PREDICTION_ERROR_STREAM = "internal.prediction_error"
REWARD_PREDICTION_ERROR_STREAM = "internal.reward_prediction_error"
LEARNING_PROGRESS_STREAM = "internal.learning_progress"
NOVELTY_STREAM = "internal.novelty"
RISK_STREAM = "internal.risk"

#: Every internal.* stream id this module publishes, in a stable order.
INTERNAL_MODULATION_STREAM_IDS: Tuple[str, ...] = (
    PREDICTION_ERROR_STREAM,
    REWARD_PREDICTION_ERROR_STREAM,
    LEARNING_PROGRESS_STREAM,
    NOVELTY_STREAM,
    RISK_STREAM,
)


def _spec(stream_id: str, description: str) -> StreamSpec:
    # "internal" is not a registered StreamEvent modality (core.streams.events
    # .MODALITIES); "event" is the established stand-in for a runtime
    # introspection scalar with no real sensory head, matching the precedent
    # `runtime.loop.NOVELTY_STREAM` already set for `model.novelty`.
    return StreamSpec(stream_id, "event", description, payload_schema="{value}")


#: `StreamSpec`s for every internal.* stream, in `INTERNAL_MODULATION_STREAM_IDS`
#: order — registered directly on the sensory bus (they are runtime/model
#: -computed signals, not part of any Program's catalog).
INTERNAL_MODULATION_STREAM_SPECS: Tuple[StreamSpec, ...] = (
    _spec(
        PREDICTION_ERROR_STREAM,
        "World-model next-latent prediction error, promoted from "
        "Prediction.prediction_error (issue #26/#58).",
    ),
    _spec(
        REWARD_PREDICTION_ERROR_STREAM,
        "Actual reward minus the world model's predicted reward; the "
        "dopamine analog (issue #58).",
    ),
    _spec(
        LEARNING_PROGRESS_STREAM,
        "Slow-EMA minus fast-EMA of prediction error; positive means "
        "prediction error is improving here (issue #58).",
    ),
    _spec(
        NOVELTY_STREAM,
        "Combined novelty score (issue #27), promoted to a first-class "
        "internal stream (issue #58).",
    ),
    _spec(
        RISK_STREAM,
        "World-model risk/p_death head output: predicted pain/injury/death, "
        "made visible (issue #58).",
    ),
)


def compute_reward_prediction_error(
    actual_reward: float, predicted_reward: Optional[float]
) -> Optional[float]:
    """Actual minus predicted reward; ``None`` when the world model has no
    reward head (the heuristic ``TrendWorldModel`` never sets
    ``predicted_reward``)."""
    if predicted_reward is None:
        return None
    return actual_reward - predicted_reward


@dataclass
class LearningProgressTracker:
    """Two-timescale EMA of prediction error (issue #58).

    ``learning_progress = slow_ema - fast_ema``. A steadily falling error
    pulls the fast EMA down before the slower one catches up, so the
    difference is positive while the model is "getting better at predicting
    this"; a plateaued or noisy-but-static error keeps both EMAs together,
    so the difference settles near zero.
    """

    fast_alpha: float = 0.3
    slow_alpha: float = 0.02
    _fast_ema: Optional[float] = field(default=None, init=False, repr=False)
    _slow_ema: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.fast_alpha <= 1.0:
            raise ValueError(f"fast_alpha must be in (0, 1], got {self.fast_alpha!r}")
        if not 0.0 < self.slow_alpha <= 1.0:
            raise ValueError(f"slow_alpha must be in (0, 1], got {self.slow_alpha!r}")
        if self.slow_alpha >= self.fast_alpha:
            raise ValueError(
                f"slow_alpha ({self.slow_alpha!r}) must be < fast_alpha "
                f"({self.fast_alpha!r}) -- the slow EMA has to lag the fast one"
            )

    def update(self, prediction_error: Optional[float]) -> Optional[float]:
        """Feed one tick's prediction error; returns learning progress, or
        ``None`` when no error is available this tick (nothing to track)."""
        if prediction_error is None:
            return None
        if self._fast_ema is None or self._slow_ema is None:
            self._fast_ema = prediction_error
            self._slow_ema = prediction_error
            return 0.0
        self._fast_ema += self.fast_alpha * (prediction_error - self._fast_ema)
        self._slow_ema += self.slow_alpha * (prediction_error - self._slow_ema)
        return self._slow_ema - self._fast_ema

    def reset(self) -> None:
        self._fast_ema = None
        self._slow_ema = None

    def state_dict(self) -> Dict[str, Any]:
        return {"fast_ema": self._fast_ema, "slow_ema": self._slow_ema}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._fast_ema = state.get("fast_ema")
        self._slow_ema = state.get("slow_ema")


@dataclass(frozen=True)
class ModulationSignals:
    """One tick's internal.* modulation values. ``None`` where the
    underlying signal is unavailable this tick (e.g. a heuristic world model
    with no reward head); ``risk`` is always available (``Prediction.risk``
    defaults to 0.0, never ``None``)."""

    prediction_error: Optional[float]
    reward_prediction_error: Optional[float]
    learning_progress: Optional[float]
    novelty: Optional[float]
    risk: float

    def as_payloads(self) -> Dict[str, Dict[str, float]]:
        """stream_id -> ``{"value": ...}`` for every signal that fired this
        tick, in the uniform payload shape the reward engine's
        ``intrinsic_stream`` components already expect."""
        payloads: Dict[str, Dict[str, float]] = {
            RISK_STREAM: {"value": round(self.risk, 6)}
        }
        if self.prediction_error is not None:
            payloads[PREDICTION_ERROR_STREAM] = {"value": round(self.prediction_error, 6)}
        if self.reward_prediction_error is not None:
            payloads[REWARD_PREDICTION_ERROR_STREAM] = {
                "value": round(self.reward_prediction_error, 6)
            }
        if self.learning_progress is not None:
            payloads[LEARNING_PROGRESS_STREAM] = {"value": round(self.learning_progress, 6)}
        if self.novelty is not None:
            payloads[NOVELTY_STREAM] = {"value": round(self.novelty, 6)}
        return payloads


class ModulationTracker:
    """Computes one tick's :class:`ModulationSignals` from the world model's
    `Prediction`, the entity-persistence surprise, and the tick's realized
    reward; owns the learning-progress EMA state across ticks (and episodes
    -- a run's world model keeps predicting across episode resets, so its
    predictive skill is tracked continuously, not reset per episode).

    ``state_dict``/``load_state_dict`` are ready for a checkpoint's
    ``training_stats`` (issue #20), so a resumed run doesn't reset the EMA
    baselines -- wiring that into a concrete learner's checkpoint bundle is
    left to whichever learner owns one.
    """

    def __init__(self, learning_progress: Optional[LearningProgressTracker] = None):
        self.learning_progress = learning_progress or LearningProgressTracker()

    def update(
        self,
        prediction: Prediction,
        entity_surprise: Optional[float],
        actual_reward: float,
    ) -> ModulationSignals:
        novelty = combine_novelty(prediction.prediction_error, entity_surprise)
        return ModulationSignals(
            prediction_error=prediction.prediction_error,
            reward_prediction_error=compute_reward_prediction_error(
                actual_reward, prediction.predicted_reward
            ),
            learning_progress=self.learning_progress.update(prediction.prediction_error),
            novelty=novelty,
            risk=prediction.risk,
        )

    def reset(self) -> None:
        self.learning_progress.reset()

    def state_dict(self) -> Dict[str, Any]:
        return {"learning_progress": self.learning_progress.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.learning_progress.load_state_dict(state.get("learning_progress", {}))
