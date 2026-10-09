"""Cortex: the predictive world model (docs/v2/phases/phase-2-predictive-cortex.md)."""

__all__ = [
    "CortexHorizonPrediction",
    "CortexRolloutOutput",
    "PredictiveCortex",
    "PredictiveCortexConfig",
    "build_predictive_cortex",
]


def __getattr__(name):
    # Contracts must be importable without the optional neural extra.
    if name in __all__:
        from brain.cortex import predictive
        return getattr(predictive, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
