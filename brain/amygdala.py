"""The Amygdala: fast threat appraisal -> adrenaline release
(docs/v2/phases/phase-3-neuromodulators-arbiter.md, issue #94, task 2).

Appraises the cortex's risk head -- ``internal.risk`` and its sign-flipped
derivative ``internal.predicted_risk_aversion``
(``brain.neuromod.modulation.ModulationSignals``) -- into a fast
threat/adrenaline release. Reuses :func:`brain.neuromod.safe_gate` rather
than reimplementing the risk-to-gate curve: adrenaline is
``1 - safe_gate(risk)``, i.e. the same sigmoid cutover around
``risk_threshold``, inverted to read "how dangerous", not "how safe".

Adrenaline is deliberately *not* a bare per-tick reading of that inverted
gate: real adrenaline release is fast but its clearance is slower than its
onset, so a single-tick risk spike still reads as elevated arousal a few
ticks later rather than snapping back to quiet immediately -- the "spike"
shape the phase doc's acceptance criterion asks for. :class:`Amygdala`
tracks this with a two-rate EMA (:attr:`AmygdalaConfig.rise_alpha` >
:attr:`AmygdalaConfig.fall_alpha`), the same asymmetric-EMA idiom
``brain.neuromod.modulation.LearningProgressTracker`` already uses for a
different pair of timescales.

This module only produces the *signal* -- "pre-empt deliberation and gate
reflexes" (the phase doc's task 2 acceptance line) is the arbiter's job
(issue #95, the sibling issue) and the reflex-precedence stack (Phase 6);
here, adrenaline is published as an ``internal.*`` stream like any other
neuromodulator, for the arbiter to consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from brain.neuromod import ADRENALINE_STREAM
from brain.neuromod.modulation import (
    DEFAULT_RISK_TEMPERATURE,
    DEFAULT_RISK_THRESHOLD,
    safe_gate,
)

__all__ = ["ADRENALINE_STREAM", "AmygdalaConfig", "Amygdala"]


@dataclass(frozen=True)
class AmygdalaConfig:
    """Shape of the threat gate plus the release/clearance asymmetry."""

    #: Same shape knobs `safe_gate` uses for the risk gate (issue #61) --
    #: the risk level at which the threat reading crosses 0.5, and the
    #: sigmoid's softness around it.
    threat_threshold: float = DEFAULT_RISK_THRESHOLD
    threat_temperature: float = DEFAULT_RISK_TEMPERATURE
    #: EMA rate while this tick's threat reading exceeds the tracked
    #: level (release: fast).
    rise_alpha: float = 0.6
    #: EMA rate while this tick's threat reading is at or below the
    #: tracked level (clearance: slower -- adrenaline lingers).
    fall_alpha: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 < self.rise_alpha <= 1.0:
            raise ValueError(f"rise_alpha must be in (0, 1], got {self.rise_alpha!r}")
        if not 0.0 < self.fall_alpha <= 1.0:
            raise ValueError(f"fall_alpha must be in (0, 1], got {self.fall_alpha!r}")


class Amygdala:
    """Tracks one running adrenaline level, updated by :meth:`appraise`
    every cognitive tick from the cortex's risk head."""

    def __init__(self, config: Optional[AmygdalaConfig] = None):
        self.config = config or AmygdalaConfig()
        self._level: float = 0.0

    @property
    def level(self) -> float:
        """The current adrenaline level (``[0, 1)``); ``0.0`` at rest."""
        return self._level

    def appraise(self, risk: float, predicted_risk_aversion: Optional[float] = None) -> float:
        """One tick's threat appraisal; returns (and updates) the tracked
        adrenaline level.

        ``predicted_risk_aversion`` is ``-risk`` under
        ``ModulationTracker``'s current math, so passing it changes
        nothing today -- it is accepted (and, if given, combined via
        ``max`` so a more-aversive reading never gets diluted by a less
        alarming raw risk reading) so a future risk-aversion term that
        diverges from a strict sign-flip of risk (e.g. reward-profile
        shaping) still drives the amygdala correctly without a signature
        change here.
        """
        aversion_risk = -predicted_risk_aversion if predicted_risk_aversion is not None else risk
        effective_risk = max(risk, aversion_risk)
        threat = 1.0 - safe_gate(
            effective_risk, self.config.threat_threshold, self.config.threat_temperature
        )
        alpha = self.config.rise_alpha if threat > self._level else self.config.fall_alpha
        self._level += alpha * (threat - self._level)
        return self._level

    def reset(self) -> None:
        self._level = 0.0

    def as_payload(self) -> dict:
        """The current level in the uniform ``{"value": ...}`` shape every
        ``internal.*`` stream uses, ready to publish under
        :data:`ADRENALINE_STREAM`."""
        return {"value": round(self._level, 6)}
