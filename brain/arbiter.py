"""The Arbiter: a hand-authored three-mode switch over (surprise, predicted
pain) (docs/v2/phases/phase-3-neuromodulators-arbiter.md, issue #95).

Every cognitive tick the arbiter looks at two scalars -- **surprise**
(calibrated prediction error) and **pain** (the amygdala's appraised
threat, `brain.amygdala.Amygdala.level`) -- and picks one of three modes
via a **hand-authored 2x2 lookup**:

- low pain, low surprise -> ``REWARD_SEEKING`` (bored: nothing unexpected,
  nothing dangerous, seek reward).
- low pain, high surprise -> ``INFO_GATHERING`` (curious: something
  unexpected but safe -- orient + sample to reduce the error, drives
  ``internal.safe_novelty``).
- high pain, either surprise -> ``FIGHT_OR_FLIGHT`` (afraid: a known
  ongoing threat is still a threat even once it stops being surprising, so
  pain takes precedence over surprise -- reflex overrides deliberation,
  adrenaline).

**This is authored, not emergent.** The lookup is a plain table plus a
hysteresis counter -- nothing here is learned or "arises" from training
(decision log #1, `docs/v2/direction-and-critique-response.md`). Its
correctness is the table + calibrated inputs + hysteresis, nothing more.

Full motor-override precedence (which concrete reflex/controller wins) is
finalised in Phase 6; here the arbiter only selects the mode and publishes
it as a recorded stream for whatever gates on it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from brain.calibration import TemperatureFit, fit_temperature, logit, sigmoid
from cognitive_runtime.core.streams.events import StreamSpec

__all__ = [
    "REWARD_SEEKING",
    "INFO_GATHERING",
    "FIGHT_OR_FLIGHT",
    "ARBITER_MODES",
    "ARBITER_MODE_STREAM",
    "ARBITER_MODE_STREAM_SPEC",
    "ArbiterConfig",
    "Arbiter",
    "SurpriseCalibratorConfig",
    "SurpriseCalibrator",
]

REWARD_SEEKING = "reward_seeking"
INFO_GATHERING = "info_gathering"
FIGHT_OR_FLIGHT = "fight_or_flight"

#: Every mode the arbiter can select, in a stable order.
ARBITER_MODES: tuple = (REWARD_SEEKING, INFO_GATHERING, FIGHT_OR_FLIGHT)

#: The arbiter's chosen mode, published every tick like any other
#: runtime-computed signal (issue #95). "event" is the same modality
#: stand-in `internal.*` (issue #58) and `internal.attention.weights`
#: (issue #59) already use for a non-sensory, model-introspection stream.
ARBITER_MODE_STREAM = "internal.arbiter_mode"
ARBITER_MODE_STREAM_SPEC = StreamSpec(
    ARBITER_MODE_STREAM, "event",
    "The arbiter's chosen mode this tick (reward_seeking / info_gathering / "
    "fight_or_flight), the (surprise, pain) reading that drove it, and the "
    "calibrated-surprise reliability metric (issue #95).",
    payload_schema="{mode, surprise, pain, calibration_error}",
)


def _lookup_mode(surprise_high: bool, pain_high: bool) -> str:
    """The hand-authored 2x2 table (module docstring): pain dominates --
    a threat that is no longer surprising is still a threat -- so only the
    low-pain row is split on surprise."""
    if pain_high:
        return FIGHT_OR_FLIGHT
    return INFO_GATHERING if surprise_high else REWARD_SEEKING


@dataclass(frozen=True)
class ArbiterConfig:
    #: Calibrated surprise `>= surprise_threshold` counts as "high" this tick.
    surprise_threshold: float = 0.5
    #: Pain (amygdala adrenaline / risk) `>= pain_threshold` counts as "high".
    pain_threshold: float = 0.5
    #: A mode change must be the raw table's answer for `k` consecutive
    #: ticks before it takes -- hysteresis against tick-to-tick flapping.
    hysteresis_ticks: int = 3
    #: The mode an `Arbiter` starts (and `reset()`s) in.
    initial_mode: str = REWARD_SEEKING

    def __post_init__(self) -> None:
        if self.hysteresis_ticks <= 0:
            raise ValueError(f"hysteresis_ticks must be positive, got {self.hysteresis_ticks!r}")
        if self.initial_mode not in ARBITER_MODES:
            raise ValueError(
                f"unknown initial_mode {self.initial_mode!r}; expected one of {ARBITER_MODES}"
            )


class Arbiter:
    """Stateful across ticks: `decide()` runs the raw 2x2 lookup every tick
    but only lets it change the *active* mode once the same raw answer has
    held for `ArbiterConfig.hysteresis_ticks` consecutive ticks."""

    def __init__(self, config: Optional[ArbiterConfig] = None):
        self.config = config or ArbiterConfig()
        self._active_mode: str = self.config.initial_mode
        self._candidate_mode: Optional[str] = None
        self._candidate_streak: int = 0
        self._last_surprise: float = 0.0
        self._last_pain: float = 0.0

    @property
    def mode(self) -> str:
        """The currently active (hysteresis-protected) mode."""
        return self._active_mode

    def reset(self) -> None:
        self._active_mode = self.config.initial_mode
        self._candidate_mode = None
        self._candidate_streak = 0
        self._last_surprise = 0.0
        self._last_pain = 0.0

    def decide(self, surprise: float, pain: float) -> str:
        """One tick's (surprise, pain) reading; returns (and updates) the
        active mode."""
        self._last_surprise = surprise
        self._last_pain = pain
        raw_mode = _lookup_mode(
            surprise_high=surprise >= self.config.surprise_threshold,
            pain_high=pain >= self.config.pain_threshold,
        )
        if raw_mode == self._active_mode:
            # Already here (or back to it before the streak matured) --
            # nothing pending.
            self._candidate_mode = None
            self._candidate_streak = 0
            return self._active_mode
        if raw_mode == self._candidate_mode:
            self._candidate_streak += 1
        else:
            self._candidate_mode = raw_mode
            self._candidate_streak = 1
        if self._candidate_streak >= self.config.hysteresis_ticks:
            self._active_mode = raw_mode
            self._candidate_mode = None
            self._candidate_streak = 0
        return self._active_mode

    def as_payload(self, calibration_error: Optional[float] = None) -> Dict[str, object]:
        """This tick's mode payload, ready to publish under
        :data:`ARBITER_MODE_STREAM`."""
        return {
            "mode": self._active_mode,
            "surprise": round(self._last_surprise, 6),
            "pain": round(self._last_pain, 6),
            "calibration_error": (
                round(calibration_error, 6) if calibration_error is not None else None
            ),
        }


def _quantile(values: List[float], q: float) -> float:
    """Linear-interpolated quantile over a copy of `values` -- no numpy
    dependency needed for a handful of rolling-window floats."""
    if not values:
        raise ValueError("quantile of an empty sequence is undefined")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] + frac * (ordered[upper] - ordered[lower])


@dataclass(frozen=True)
class SurpriseCalibratorConfig:
    #: Rolling holdout size: how many recent raw readings the calibrator
    #: fits its temperature against.
    window: int = 200
    n_bins: int = 10
    #: Refit the temperature every this many ticks (a fresh grid search
    #: every single tick is wasted work -- the rolling distribution moves
    #: slowly relative to one tick).
    refit_every: int = 20
    #: Don't attempt a fit until the window holds at least this many
    #: readings -- too few points make both the fit and its reported ECE
    #: meaningless.
    min_observations: int = 20
    #: A raw reading in the top `outcome_quantile` fraction of the rolling
    #: window counts as this tick's ground-truth "surprising" label -- see
    #: `SurpriseCalibrator`'s docstring for why the label has to be
    #: self-referential here.
    outcome_quantile: float = 0.75

    def __post_init__(self) -> None:
        if self.window <= 1:
            raise ValueError(f"window must be > 1, got {self.window!r}")
        if self.min_observations < 2:
            raise ValueError(f"min_observations must be >= 2, got {self.min_observations!r}")
        if self.refit_every <= 0:
            raise ValueError(f"refit_every must be positive, got {self.refit_every!r}")
        if not 0.0 < self.outcome_quantile < 1.0:
            raise ValueError(
                f"outcome_quantile must be in (0, 1), got {self.outcome_quantile!r}"
            )


class SurpriseCalibrator:
    """Calibrates a raw per-tick surprise reading (cortex sigma / the
    prediction-error stand-in `brain.neuromod.compute_acetylcholine` also
    uses -- no dedicated sigma head is wired into the `WorldModel`
    interface yet) into a `[0, 1)` "this tick is surprising" probability,
    temperature-scaled against a rolling holdout of its own recent
    readings, and reports Expected Calibration Error (ECE,
    `brain.calibration.expected_calibration_error`) as the first-class
    calibration metric the phase doc asks for.

    Ground truth is necessarily self-referential: there is no separate
    forward-uncertainty-vs-realized-error pair available at the runtime-loop
    level (unlike `cognitive_runtime.training.world_model.
    uncertainty_calibration`'s batch setting, which has both a model's
    predicted uncertainty and the realized squared error from held-out
    data). Instead each raw reading is labeled against the *rolling
    distribution it belongs to*: readings in the top
    `SurpriseCalibratorConfig.outcome_quantile` fraction of the window are
    "surprising". This still catches genuine miscalibration -- a raw signal
    that is a monotonic but badly-shaped transform of "true" surprise
    (saturating, or scaled to the wrong range) reports a confidence that
    disagrees with its own empirical rate, and `fit_temperature` corrects
    the shape.
    """

    def __init__(self, config: Optional[SurpriseCalibratorConfig] = None):
        self.config = config or SurpriseCalibratorConfig()
        self._raw: Deque[float] = deque(maxlen=self.config.window)
        self._temperature: float = 1.0
        self._calibration_error: Optional[float] = None
        self._ticks_since_fit: int = 0

    def reset(self) -> None:
        self._raw.clear()
        self._temperature = 1.0
        self._calibration_error = None
        self._ticks_since_fit = 0

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def calibration_error(self) -> Optional[float]:
        """The rolling-holdout ECE at the current temperature; `None`
        before `SurpriseCalibratorConfig.min_observations` readings have
        accumulated."""
        return self._calibration_error

    def update(self, raw: float) -> float:
        """One tick's raw surprise reading (any non-negative scalar);
        returns this tick's calibrated surprise probability in `[0, 1)`,
        periodically refitting the temperature against the rolling
        window."""
        bounded = max(0.0, raw) / (1.0 + max(0.0, raw))  # saturate, don't clip
        self._raw.append(bounded)
        self._ticks_since_fit += 1
        if (
            len(self._raw) >= self.config.min_observations
            and self._ticks_since_fit >= self.config.refit_every
        ):
            self._refit()
            self._ticks_since_fit = 0
        calibrated = sigmoid(logit(bounded) / self._temperature)
        # A low fitted temperature (sharpening an underconfident head) can
        # push `logit(bounded) / temperature` far enough that `exp(-x)`
        # underflows float64 precision and `sigmoid` rounds to exactly
        # `1.0` -- clamp so this always honors its documented `[0, 1)` range
        # (a `>= 1.0` reading here would make `ArbiterConfig.
        # surprise_threshold` comparisons degenerate).
        return min(calibrated, 1.0 - 1e-9)

    def _refit(self) -> None:
        values = list(self._raw)
        threshold = _quantile(values, self.config.outcome_quantile)
        outcomes = [v >= threshold for v in values]
        if len(set(outcomes)) < 2:
            # Every reading fell on the same side of its own quantile (a
            # degenerate/constant window) -- nothing to fit against; keep
            # whatever the last real fit produced.
            return
        fit: TemperatureFit = fit_temperature(values, outcomes, n_bins=self.config.n_bins)
        self._temperature = fit.temperature
        self._calibration_error = fit.ece_after
