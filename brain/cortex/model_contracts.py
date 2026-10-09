"""Framework-free inference and temporal model interfaces.

Inference carries only prepared finite feature vectors and opaque sample IDs.
Targets and terminal metadata belong to task_contracts, never to this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class InferenceInput:
    feature_schema: str
    sample_ids: tuple[str, ...]
    features: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        if not isinstance(self.feature_schema, str) or not self.feature_schema:
            raise ValueError("feature_schema is required")
        ids = tuple(self.sample_ids)
        rows = tuple(tuple(row) for row in self.features)
        if len(ids) != len(rows) or len(set(ids)) != len(ids):
            raise ValueError("features require unique, aligned sample IDs")
        if any(type(value) is not str or not value for value in ids):
            raise TypeError("sample IDs must be nonempty strings")
        widths = {len(row) for row in rows}
        if len(widths) > 1 or 0 in widths:
            raise ValueError("features must be rectangular nonempty vectors")
        if any(type(x) not in (int, float) or not math.isfinite(x) for row in rows for x in row):
            raise TypeError("inference features must be finite numbers, not records/metadata")
        object.__setattr__(self, "sample_ids", ids)
        object.__setattr__(self, "features", rows)


def require_inference_input(value: InferenceInput) -> InferenceInput:
    # Reject training/evaluation batches and subclasses with extra fields.
    if type(value) is not InferenceInput:
        raise TypeError("predict requires exactly InferenceInput; labels/terminal metadata are forbidden")
    return value


class PredictiveModel(Protocol):
    def predict(self, inputs: InferenceInput) -> tuple[tuple[float, ...], ...]: ...


class TemporalModel(Protocol):
    """Structural surface already implemented by TemporalBackbone; no torch import."""
    def initial_state(self, batch: int) -> Any: ...
    def step(self, x: Any, state: Any) -> tuple[Any, Any]: ...
    def readout(self, state: Any) -> Any: ...
    def forward_sequence(self, inputs: Any) -> Any: ...
