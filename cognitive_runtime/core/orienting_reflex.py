"""Scripted orienting reflex (issue #60): the motor half of biological
attention. When bottom-up salience capture (`core.attention`, issue #59)
lands on a stream carrying a localizable stimulus, this brainstem-level
reflex turns toward it for a bounded number of ticks -- deterministic and
scripted, the way a real nervous system orients long before any learned
policy exists.

Generic, Program-agnostic core: the reflex only ever sees an
`AttentionState`, the world model's `internal.risk` reading, and the
policy's own emitted actions (classified by a Program-supplied
`ActionRegistry`, issue #60's action-registry half). Which concrete action
name means "turn left"/"turn right" is Program-specific and is supplied
through `OrientingReflexConfig`, exactly like `AttentionMetadata` keeps
Minecraft stream ids out of `core.attention`.

Precedence, highest to lowest:

1. `mode != "on"` -- the reflex never fires (`"off"`; `"learned-only"`
   leaves orienting to the policy/neural attention instead).
2. `risk >= risk_veto_threshold` -- a high predicted-risk reading
   (`internal.risk`, issue #58) vetoes the reflex outright, in favor of
   whatever response the policy comes up with.
3. The policy already emitted a world-changing action this tick -- the
   reflex must never suppress a survival-critical response (fleeing, eating
   at critical hunger are both world-changing); it only ever substitutes
   for an empty or purely-perceptual tick.
4. No bottom-up capture landed this tick, or the captured stream carries no
   direction hint, or the bearing is within the dead zone -- nothing to
   orient toward.

Otherwise the reflex turns toward the stimulus (left/right, by the sign of
`bearing_deg`), held for `hold_ticks` so a single spike commands a short,
bounded look rather than a one-tick flick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.action_registry import ActionRegistry, DEFAULT_ACTION_REGISTRY
from cognitive_runtime.core.attention import AttentionState

REFLEX_MODES = frozenset({"on", "off", "learned-only"})

#: `DecisionRecord.reflex["reason"]` when the reflex fires this tick.
REASON_ORIENTING_REFLEX = "orienting_reflex"


@dataclass(frozen=True)
class OrientingReflexConfig:
    mode: str = "on"
    #: Ticks a single reflex activation holds its look/turn action for,
    #: before the next bottom-up capture (or its absence) is reconsidered.
    hold_ticks: int = 3
    #: `internal.risk` (issue #58) at/above this vetoes the reflex outright.
    risk_veto_threshold: float = 0.7
    #: Bearings within this many degrees of dead ahead count as "already
    #: facing it" -- no turn action needed.
    bearing_deadzone_deg: float = 15.0
    #: Program-supplied action names for "turn toward a stimulus on my
    #: left/right" (Minecraft: LOOK_LEFT/LOOK_RIGHT).
    left_action: str = "LOOK_LEFT"
    right_action: str = "LOOK_RIGHT"

    def __post_init__(self) -> None:
        if self.mode not in REFLEX_MODES:
            raise ValueError(
                f"unknown reflex mode {self.mode!r}; expected one of {sorted(REFLEX_MODES)}"
            )
        if self.hold_ticks <= 0:
            raise ValueError(f"hold_ticks must be positive, got {self.hold_ticks!r}")
        if not 0.0 <= self.risk_veto_threshold <= 1.0:
            raise ValueError(
                f"risk_veto_threshold must be in [0, 1], got {self.risk_veto_threshold!r}"
            )
        if self.bearing_deadzone_deg < 0:
            raise ValueError(
                f"bearing_deadzone_deg must be >= 0, got {self.bearing_deadzone_deg!r}"
            )
        if not self.left_action or not self.right_action:
            raise ValueError("left_action/right_action must be non-empty action names")


@dataclass(frozen=True)
class OrientingDecision:
    """One tick's reflex activation, ready to drop into `DecisionRecord`."""

    action: Action
    stimulus_stream: str
    direction: Dict[str, Any]
    ticks_remaining: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": REASON_ORIENTING_REFLEX,
            "stimulus_stream": self.stimulus_stream,
            "direction": self.direction,
            "ticks_remaining": self.ticks_remaining,
        }


class OrientingReflex:
    """Stateful across ticks: a firing reflex holds its look direction for
    `hold_ticks` before yielding, so a single capture commands a short,
    bounded orientation instead of a single-tick flick. A hold in progress
    is still subject to the risk-veto and world-changing-policy-action
    precedence checks every tick, so it can be cut short but never
    preempted early by a *new* stimulus."""

    def __init__(
        self,
        config: Optional[OrientingReflexConfig] = None,
        action_registry: Optional[ActionRegistry] = None,
    ) -> None:
        self.config = config or OrientingReflexConfig()
        self.action_registry = action_registry or DEFAULT_ACTION_REGISTRY
        self._hold_action: Optional[Action] = None
        self._hold_remaining: int = 0
        self._hold_stimulus: Optional[str] = None
        self._hold_direction: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self._hold_action = None
        self._hold_remaining = 0
        self._hold_stimulus = None
        self._hold_direction = None

    def decide(
        self,
        attention_state: AttentionState,
        risk: float,
        policy_actions: List[Action],
    ) -> Optional[OrientingDecision]:
        """`None` when the reflex doesn't fire this tick; otherwise the
        look/turn action the runtime should emit *instead of*
        `policy_actions` this tick."""
        if self.config.mode != "on":
            self.reset()
            return None
        if risk >= self.config.risk_veto_threshold:
            self.reset()
            return None
        if any(self.action_registry.is_world_changing(a) for a in policy_actions):
            self.reset()
            return None

        if self._hold_remaining > 0:
            self._hold_remaining -= 1
            return OrientingDecision(
                action=self._hold_action,  # type: ignore[arg-type]
                stimulus_stream=self._hold_stimulus,  # type: ignore[arg-type]
                direction=self._hold_direction,  # type: ignore[arg-type]
                ticks_remaining=self._hold_remaining,
            )

        if not attention_state.bottom_up_capture or attention_state.focus_stream is None:
            return None
        reason = attention_state.reasons.get(attention_state.focus_stream)
        if reason is None or reason.signal.direction is None:
            return None
        direction = reason.signal.direction
        action = self._orient_action(direction.bearing_deg)
        if action is None:
            return None

        self._hold_action = action
        self._hold_remaining = self.config.hold_ticks - 1
        self._hold_stimulus = attention_state.focus_stream
        self._hold_direction = direction.to_dict()
        return OrientingDecision(
            action=action,
            stimulus_stream=self._hold_stimulus,
            direction=self._hold_direction,
            ticks_remaining=self._hold_remaining,
        )

    def _orient_action(self, bearing_deg: Optional[float]) -> Optional[Action]:
        if bearing_deg is None:
            return None
        if abs(bearing_deg) <= self.config.bearing_deadzone_deg:
            return None
        name = self.config.right_action if bearing_deg > 0 else self.config.left_action
        return Action(name)
