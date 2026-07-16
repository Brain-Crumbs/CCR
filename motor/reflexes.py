"""Configured reflex and caregiver precedence for the motor system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from cognitive_runtime.core.action import Action


@dataclass(frozen=True)
class Stimulus:
    """A stimulus advertised by a World, without motor-system semantics."""

    kind: str
    intensity: float
    source: str = "world"
    data: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ReflexConfig:
    """Organism-owned (genotype) mapping from stimulus to response."""

    name: str
    stimulus: str
    action: Action
    threshold: float = 0.0
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.stimulus:
            raise ValueError("reflex name and stimulus must be non-empty")


@dataclass(frozen=True)
class ReflexDecision:
    name: str
    action: Action
    reason: str
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "action": self.action.key(), "reason": self.reason,
                "priority": self.priority}


@dataclass(frozen=True)
class CaregiverOverride:
    action: Action
    reason: str = "caregiver"


@dataclass(frozen=True)
class MotorDecision:
    """Complete efference record for one tick."""

    voluntary: Action
    reflex: Optional[ReflexDecision]
    caregiver_override: Optional[CaregiverOverride]
    actuated: Action

    @property
    def diverged(self) -> bool:
        return self.voluntary != self.actuated

    def to_dict(self) -> dict[str, Any]:
        return {
            "voluntary": self.voluntary.key(),
            "reflex": None if self.reflex is None else self.reflex.to_dict(),
            "caregiver_override": None if self.caregiver_override is None else {
                "action": self.caregiver_override.action.key(),
                "reason": self.caregiver_override.reason,
            },
            "actuated": self.actuated.key(),
        }


class ReflexStack:
    """Apply ``caregiver > highest-priority reflex > voluntary``."""

    def __init__(self, configs: Sequence[ReflexConfig]) -> None:
        self.configs = tuple(configs)
        self.ticks = 0
        self.reflex_ticks = 0

    def evaluate(self, stimuli: Iterable[Stimulus]) -> Optional[ReflexDecision]:
        candidates: list[tuple[int, int, ReflexConfig, Stimulus]] = []
        for order, config in enumerate(self.configs):
            for stimulus in stimuli:
                if stimulus.kind == config.stimulus and stimulus.intensity >= config.threshold:
                    candidates.append((config.priority, -order, config, stimulus))
        if not candidates:
            return None
        _, _, config, stimulus = max(candidates, key=lambda item: (item[0], item[1]))
        return ReflexDecision(config.name, config.action,
                              f"{stimulus.source}:{stimulus.kind}>={config.threshold}",
                              config.priority)

    def decide(
        self,
        voluntary: Action,
        stimuli: Iterable[Stimulus] = (),
        caregiver: Optional[CaregiverOverride] = None,
    ) -> MotorDecision:
        reflex = self.evaluate(stimuli)
        actuated = caregiver.action if caregiver is not None else (
            reflex.action if reflex is not None else voluntary
        )
        self.ticks += 1
        self.reflex_ticks += reflex is not None
        return MotorDecision(voluntary, reflex, caregiver, actuated)

    @property
    def activation_rate(self) -> float:
        return self.reflex_ticks / self.ticks if self.ticks else 0.0

    def metrics(self) -> dict[str, float | int]:
        return {"motor.reflex_activation_rate": self.activation_rate,
                "motor.reflex_activations": self.reflex_ticks,
                "motor.ticks": self.ticks}
