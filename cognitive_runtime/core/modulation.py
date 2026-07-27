"""Re-export shim (issue #94): the ``internal.*`` modulation math -- EMA
learning progress, reward-prediction error, the risk gate, and
:class:`ModulationTracker` -- now lives in ``brain.neuromod.modulation``,
promoted out of this module as part of naming the three behaviour-changing
neuromodulators over it (``brain.neuromod``: dopamine, acetylcholine,
adrenaline; see ``docs/v2/phases/phase-3-neuromodulators-arbiter.md``).
Every name this module used to define is re-exported unchanged, so existing
imports of ``cognitive_runtime.core.modulation`` keep resolving.
"""

from __future__ import annotations

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
]
