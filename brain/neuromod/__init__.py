"""Neuromodulators: the three behaviour-changing chemicals, named over
existing math (docs/v2/phases/phase-3-neuromodulators-arbiter.md, issue
#94, task 1).

Promoted from ``cognitive_runtime.core.modulation`` (kept as a re-export
shim so existing imports keep resolving) into ``brain.neuromod.modulation``,
unchanged. This package adds the human-named surface over it -- three
``internal.*`` streams, each backed by math that already exists or is
computed elsewhere in ``brain``:

- **dopamine** (:data:`DOPAMINE_STREAM`) -- already
  ``internal.reward_prediction_error`` (:attr:`ModulationSignals.
  reward_prediction_error`). No new math: this is a rename, not a
  recomputation.
- **acetylcholine** (:data:`ACETYLCHOLINE_STREAM`) -- a precision/expected-
  uncertainty term (:func:`compute_acetylcholine`), derived from the
  cortex's predicted-error estimate ("sigma") plus learning progress
  (:attr:`ModulationSignals.learning_progress`). Fed into
  ``cognitive_runtime.core.attention`` as the Thalamus's precision term
  (task 3 of the same issue).
- **adrenaline** (:data:`ADRENALINE_STREAM`) -- the amygdala's appraised
  threat release (``brain.amygdala.Amygdala``, task 2 of the same issue);
  re-exported here only as a stream id, computed there.

No chemistry cosplay: serotonin/patience and an explicit norepinephrine-
arousal signal are deferred until a concrete behaviour needs them (phase
doc, "Risks / notes").
"""

from __future__ import annotations

from typing import Dict, Optional

from cognitive_runtime.core.streams.events import StreamSpec

from brain.neuromod.modulation import (
    DEFAULT_RISK_TEMPERATURE,
    DEFAULT_RISK_THRESHOLD,
    INTERNAL_MODULATION_STREAM_IDS,
    INTERNAL_MODULATION_STREAM_SPECS,
    LEARNING_PROGRESS_STREAM,
    NOVELTY_STREAM,
    PREDICTED_RISK_AVERSION_STREAM,
    PREDICTION_ERROR_STREAM,
    REWARD_PREDICTION_ERROR_STREAM,
    RISK_GATE_STREAM,
    RISK_STREAM,
    SAFE_NOVELTY_STREAM,
    LearningProgressTracker,
    ModulationSignals,
    ModulationTracker,
    compute_reward_prediction_error,
    safe_gate,
)

#: Dopamine analog: a rename of `REWARD_PREDICTION_ERROR_STREAM`, not a new
#: signal -- published alongside it (both ids carry the same value).
DOPAMINE_STREAM = "internal.dopamine"
#: Expected-uncertainty precision term (:func:`compute_acetylcholine`).
ACETYLCHOLINE_STREAM = "internal.acetylcholine"
#: The amygdala's threat/adrenaline release (`brain.amygdala.Amygdala`).
ADRENALINE_STREAM = "internal.adrenaline"

#: Every named-neuromodulator stream id this package publishes, in a stable
#: order (mirrors `INTERNAL_MODULATION_STREAM_IDS`'s convention).
NAMED_NEUROMODULATOR_STREAM_IDS = (DOPAMINE_STREAM, ACETYLCHOLINE_STREAM, ADRENALINE_STREAM)

NAMED_NEUROMODULATOR_STREAM_SPECS = (
    StreamSpec(
        DOPAMINE_STREAM, "event",
        "Dopamine analog: reward-prediction error under its human name "
        "(issue #94) -- same value as internal.reward_prediction_error.",
        payload_schema="{value}",
    ),
    StreamSpec(
        ACETYLCHOLINE_STREAM, "event",
        "Acetylcholine analog: expected-uncertainty/precision term derived "
        "from the cortex's predicted-error estimate and learning progress "
        "(issue #94) -- feeds core.attention as a precision term.",
        payload_schema="{value}",
    ),
    StreamSpec(
        ADRENALINE_STREAM, "event",
        "Adrenaline analog: the amygdala's fast threat appraisal of the "
        "cortex risk head (issue #94, brain.amygdala.Amygdala).",
        payload_schema="{value}",
    ),
)


def compute_acetylcholine(
    uncertainty: Optional[float], learning_progress: Optional[float]
) -> float:
    """The acetylcholine precision term: expected uncertainty (Yu & Dayan)
    -- high when the world model is uncertain about what happens next
    *and* not currently getting better at it, i.e. persistent,
    expected unpredictability that calls for weighting fresh sensory
    evidence over stale priors/habit.

    ``uncertainty`` is the cortex's predicted-error estimate ("sigma";
    ``brain.cortex.predictive.PredictiveCortex``'s ``uncertainty_head``,
    or ``Prediction.prediction_error`` as a stand-in where no dedicated
    sigma head is wired into the ``WorldModel`` interface yet). ``None``
    when no such estimate is available this tick -- acetylcholine is then
    quiescent (``0.0``), not undefined: nothing raises the demand for
    precision when there is nothing to be uncertain about.

    ``learning_progress`` (:attr:`ModulationSignals.learning_progress`)
    only ever *adds* to the demand for precision when it is negative
    (error getting worse -- "stalled"); positive learning progress (error
    improving) does not suppress an already-uncertain reading below what
    ``uncertainty`` alone would give.

    Bounded to ``[0, 1)`` via ``x / (1 + x)`` -- monotonically increasing,
    saturating rather than blowing up on a large sigma spike.
    """
    if uncertainty is None:
        return 0.0
    stalled = 0.0 if learning_progress is None else max(0.0, -learning_progress)
    raw = max(0.0, uncertainty) + stalled
    return raw / (1.0 + raw)


def named_neuromodulator_payloads(
    modulation: ModulationSignals, acetylcholine: float, adrenaline: float
) -> Dict[str, Dict[str, float]]:
    """This tick's three named-neuromodulator payloads, in the uniform
    ``{"value": ...}`` shape every ``internal.*`` stream uses.

    ``dopamine`` mirrors ``modulation.reward_prediction_error`` and is
    omitted exactly when that is (no reward head this tick);
    ``acetylcholine``/``adrenaline`` are supplied by the caller
    (:func:`compute_acetylcholine` / ``brain.amygdala.Amygdala.appraise``)
    and always published.
    """
    payloads: Dict[str, Dict[str, float]] = {
        ACETYLCHOLINE_STREAM: {"value": round(acetylcholine, 6)},
        ADRENALINE_STREAM: {"value": round(adrenaline, 6)},
    }
    if modulation.reward_prediction_error is not None:
        payloads[DOPAMINE_STREAM] = {"value": round(modulation.reward_prediction_error, 6)}
    return payloads


__all__ = [
    "DEFAULT_RISK_TEMPERATURE",
    "DEFAULT_RISK_THRESHOLD",
    "INTERNAL_MODULATION_STREAM_IDS",
    "INTERNAL_MODULATION_STREAM_SPECS",
    "LEARNING_PROGRESS_STREAM",
    "NOVELTY_STREAM",
    "PREDICTED_RISK_AVERSION_STREAM",
    "PREDICTION_ERROR_STREAM",
    "REWARD_PREDICTION_ERROR_STREAM",
    "RISK_GATE_STREAM",
    "RISK_STREAM",
    "SAFE_NOVELTY_STREAM",
    "LearningProgressTracker",
    "ModulationSignals",
    "ModulationTracker",
    "compute_reward_prediction_error",
    "safe_gate",
    "DOPAMINE_STREAM",
    "ACETYLCHOLINE_STREAM",
    "ADRENALINE_STREAM",
    "NAMED_NEUROMODULATOR_STREAM_IDS",
    "NAMED_NEUROMODULATOR_STREAM_SPECS",
    "compute_acetylcholine",
    "named_neuromodulator_payloads",
]
