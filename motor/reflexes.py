"""Configured reflex and caregiver precedence for the motor system."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.action_registry import ActionRegistry, DEFAULT_ACTION_REGISTRY
from cognitive_runtime.core.attention import AttentionState


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
        stimuli = tuple(stimuli)
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


class CaregiverChannel:
    """The development-stage hook (babbling / guided movement) injects motor
    commands here; the runtime drains one pending override per tick and hands
    it to :meth:`ReflexStack.decide`, the top of the precedence stack."""

    def __init__(self) -> None:
        self._pending: Optional[CaregiverOverride] = None

    def inject(self, action: Action, reason: str = "caregiver") -> None:
        self._pending = CaregiverOverride(action, reason)

    def clear(self) -> None:
        self._pending = None

    def drain(self) -> Optional[CaregiverOverride]:
        override, self._pending = self._pending, None
        return override


#: Bearings within this many degrees of dead ahead count as "already facing
#: it" (`OrientingReflexConfig.bearing_deadzone_deg`'s old default) -- reused
#: here as the `orient-left`/`orient-right` reflexes' threshold, since
#: `stimulus_from_attention`'s intensity *is* `abs(bearing_deg)`.
DEFAULT_BEARING_DEADZONE_DEG = 15.0


def stimulus_from_attention(attention_state: AttentionState) -> Optional[Stimulus]:
    """The bottom-up attention capture (`core.attention`, issue #59) as a
    single-tick ``salience-left``/``salience-right`` stimulus pulse -- the
    generic input `OrientingReflex` (issue #60) used to orient toward,
    migrated to the stimulus/reflex-config seam. `None` when nothing
    localizable captured focus this tick.

    `bottom_up_capture` is a one-tick pulse (see `AttentionState`), so a
    caller wanting `OrientingReflex`'s bounded multi-tick hold instead of a
    one-tick flick should drive this through :class:`AttentionStimulusSource`
    rather than calling this function directly every tick."""
    if not attention_state.bottom_up_capture or attention_state.focus_stream is None:
        return None
    reason = attention_state.reasons.get(attention_state.focus_stream)
    if reason is None or reason.signal.direction is None:
        return None
    bearing = reason.signal.direction.bearing_deg
    if bearing is None:
        return None
    kind = "salience-right" if bearing > 0 else "salience-left"
    return Stimulus(kind, abs(bearing), source=attention_state.focus_stream,
                    data={"bearing_deg": bearing})


class AttentionStimulusSource:
    """Holds a captured ``salience`` stimulus for `hold_ticks` ticks
    (`OrientingReflexConfig.hold_ticks`'s old default of 3), so a single
    bottom-up capture pulse commands a bounded orientation rather than
    firing the reflex for one tick and vanishing -- `OrientingReflex`'s
    `_hold_remaining` behavior, migrated to the stateless `ReflexStack` by
    keeping the hold state in the stimulus source instead."""

    def __init__(self, hold_ticks: int = 3) -> None:
        if hold_ticks <= 0:
            raise ValueError(f"hold_ticks must be positive, got {hold_ticks!r}")
        self.hold_ticks = hold_ticks
        self._held: Optional[Stimulus] = None
        self._remaining = 0

    def reset(self) -> None:
        self._held = None
        self._remaining = 0

    def poll(self, attention_state: AttentionState) -> Optional[Stimulus]:
        captured = stimulus_from_attention(attention_state)
        if captured is not None:
            self._held = captured
            self._remaining = self.hold_ticks - 1
            return captured
        if self._remaining > 0:
            self._remaining -= 1
            return self._held
        self._held = None
        return None


def eligible_orienting_stimuli(
    stimuli: Iterable[Stimulus],
    voluntary: Action,
    action_registry: ActionRegistry = DEFAULT_ACTION_REGISTRY,
) -> List[Stimulus]:
    """Drop ``salience-*`` stimuli when `voluntary` is a *purely*
    world-changing action (world-changing and not information-gathering) --
    `OrientingReflex._is_survival_critical`'s veto, migrated so the orienting
    reflex never suppresses a survival-critical voluntary response (fleeing,
    eating) with a mere look. Locomotion classified world-changing *and*
    information-gathering ("walking does both") does not veto. Every other
    stimulus kind passes through untouched -- this veto is orienting-specific,
    not a general eligibility gate."""
    if (action_registry.is_world_changing(voluntary)
            and not action_registry.is_information_gathering(voluntary)):
        return [s for s in stimuli if not s.kind.startswith("salience")]
    return list(stimuli)


def stimulus_from_threat(level: float, source: str = "amygdala") -> Stimulus:
    """The Amygdala's adrenaline level (`brain.amygdala`, issue #94) as a
    World-declared ``threat`` stimulus -- the threat/withdrawal response's
    trigger. Adrenaline is already the appraised, EMA-smoothed reading, so
    this is a direct wrap, not a re-derivation."""
    return Stimulus("threat", level, source=source)


def stimulus_from_hazard(active: bool, source: str) -> Optional[Stimulus]:
    """A boolean hazard flag (e.g. Minecraft's ``body.in_water``, the
    scripted survival policy's water-escape trigger) as a ``hazard``
    stimulus. `None` when the hazard isn't active -- reflexes only ever see
    stimuli that are actually present this tick."""
    return Stimulus("hazard", 1.0, source=source) if active else None


def default_reflex_genome(
    *,
    withdraw_action: Action,
    orient_left_action: Action,
    orient_right_action: Action,
    hazard_action: Optional[Action] = None,
    threat_threshold: float = 0.5,
    bearing_deadzone_deg: float = DEFAULT_BEARING_DEADZONE_DEG,
    hazard_threshold: float = 0.5,
) -> list[ReflexConfig]:
    """The organism's default reflex genome (Phase 6, issue #102): the
    `withdraw` reflex fires on a `threat` stimulus (migrated from
    `brain.amygdala`'s adrenaline release), `orient-left`/`orient-right` fire
    on a `salience-left`/`salience-right` stimulus (migrated from
    `OrientingReflex`), and an
    optional `hazard-escape` reflex covers boundary-condition survival
    behaviours migrated from `policies.scripted.ScriptedSurvivalPolicy`
    (e.g. water escape). Action names are Program-supplied -- the genome
    reasons about stimulus *kind*, never what an action *is*. `withdraw`
    outranks `hazard-escape`, which outranks orienting: survival-critical
    responses must never be suppressed by a mere look toward salience.

    A bearing exactly at `bearing_deadzone_deg` does not fire, matching
    `OrientingReflexConfig`'s inclusive deadzone (`abs(bearing) <= deadzone`
    counts as "already facing it"); `ReflexStack.evaluate`'s threshold check
    is otherwise inclusive (`>=`), so the orient reflexes' threshold is nudged
    up by one representable float to keep the boundary itself inert."""
    orient_threshold = math.nextafter(bearing_deadzone_deg, math.inf)
    genome = [
        ReflexConfig("withdraw", "threat", withdraw_action,
                     threshold=threat_threshold, priority=10),
        ReflexConfig("orient-left", "salience-left", orient_left_action,
                     threshold=orient_threshold, priority=1),
        ReflexConfig("orient-right", "salience-right", orient_right_action,
                     threshold=orient_threshold, priority=1),
    ]
    if hazard_action is not None:
        genome.insert(1, ReflexConfig("hazard-escape", "hazard", hazard_action,
                                       threshold=hazard_threshold, priority=8))
    return genome
