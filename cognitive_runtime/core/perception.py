"""Perception: encode raw observations into runtime state.

The default implementation is a generic structured encoder: it flattens all
numeric leaves of the observation into a feature dict and summarises the
frame, without interpreting any environment-specific meaning.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict

from cognitive_runtime.core.observation import Observation


@dataclass
class State:
    observation: Observation
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def tick(self) -> int:
        return self.observation.tick


class Perception(abc.ABC):
    @abc.abstractmethod
    def encode(self, observation: Observation) -> State:
        ...


class StructuredPerception(Perception):
    """Flattens numeric observation fields into dotted feature names."""

    def encode(self, observation: Observation) -> State:
        features: Dict[str, float] = {}
        self._flatten("", observation.data, features)
        if observation.frame:
            cells = [c for row in observation.frame for c in row]
            if cells:
                features["frame.mean"] = sum(cells) / len(cells)
                features["frame.min"] = float(min(cells))
                features["frame.max"] = float(max(cells))
        return State(observation=observation, features=features)

    def _flatten(self, prefix: str, node: Any, out: Dict[str, float]) -> None:
        if isinstance(node, bool):
            out[prefix] = 1.0 if node else 0.0
        elif isinstance(node, (int, float)):
            out[prefix] = float(node)
        elif isinstance(node, dict):
            for key, value in node.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                self._flatten(name, value, out)
        elif isinstance(node, (list, tuple)):
            numeric = [v for v in node if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if numeric and len(numeric) == len(node):
                for i, value in enumerate(node):
                    out[f"{prefix}.{i}"] = float(value)
            else:
                out[f"{prefix}.len"] = float(len(node))
