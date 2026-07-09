"""Runtime state handed to policies each cognitive tick.

`State` wraps the stream-derived observation the loop assembles from
`Memory.latest_values()`.  The old generic `StructuredPerception` numeric
flattener was removed from the product path (see
docs/neural-stream-agent.md): stream encoders + fusion are the
representation layer, and future neural encoders replace hand-written
feature extraction entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from cognitive_runtime.core.observation import Observation


@dataclass
class State:
    observation: Observation
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def tick(self) -> int:
        return self.observation.tick
