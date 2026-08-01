"""Potential-based geodesic navigation reward shaping (epic #212 §12.4/§12.5,
issue #239, gated on MF-F1/#238).

```text
Phi(s) = -strength * geodesic_distance(s, goal)
reward.goal_progress = gamma * Phi(s_next) - Phi(s)
reward.goal_reached  = success_bonus, once inside goal_radius
```

Potential-based shaping (Ng, Harada & Russell 1999) rather than paying the
agent every tick for remaining near a target: with ``gamma == 1`` (the
default), the telescoping identity ``sum_t [Phi(s_t+1) - Phi(s_t)] =
Phi(s_T) - Phi(s_0)`` makes any round trip that returns to its starting
state accumulate exactly zero net ``goal_progress`` -- a detour around an
obstacle is rewarded for shortening the *geodesic* route, not punished for
temporarily increasing Cartesian distance. A ``gamma < 1`` stays
policy-invariant in the usual RL sense (it doesn't change which policy is
optimal) but no longer telescopes to a literal zero over a closed loop,
since discounting the "next" term alone breaks the cancellation.

``geodesic_distance`` -- never Cartesian -- is what makes this correct once
walls exist (epic §12.5): the caller is expected to supply the oracle's
shortest-path distance (``programs.crafter.oracle.OracleLabel.
geodesic_distance``), not a straight-line distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

_POTENTIAL_KINDS: tuple = ("linear", "saturating")


@dataclass(frozen=True)
class GoalRewardConfig:
    """The shaping parameters (epic #212 §12.6: "reward potential, strength,
    falloff, discount, success radius and bonus are all ``DataContract``
    fields")."""

    #: "A linear potential is the initial default. Saturating falloff
    #: potentials are an optional later ``DataContract`` parameter ... and
    #: should not replace the simple baseline without a controlled
    #: comparison" (epic §12.5) -- so ``"saturating"`` is opt-in, never the
    #: default.
    potential: str = "linear"
    strength: float = 1.0
    #: Only meaningful (and only set) for ``potential == "saturating"``:
    #: the distance scale at which the potential's gradient starts flattening.
    falloff: Optional[float] = None
    #: ``gamma`` in the shaping formula above. ``1.0`` is what makes a round
    #: trip's shaped reward telescope to exactly zero (see module docstring).
    discount: float = 1.0
    success_bonus: float = 1.0

    def __post_init__(self) -> None:
        if self.potential not in _POTENTIAL_KINDS:
            raise ValueError(
                f"unknown potential kind {self.potential!r}; expected one of {_POTENTIAL_KINDS}"
            )
        if self.strength <= 0:
            raise ValueError(f"strength must be > 0, got {self.strength!r}")
        if self.potential == "saturating":
            if self.falloff is None or self.falloff <= 0:
                raise ValueError("a saturating potential requires falloff > 0")
        elif self.falloff is not None:
            raise ValueError(
                "falloff only applies to the saturating potential; leave it None for linear"
            )
        if not (0.0 < self.discount <= 1.0):
            raise ValueError(f"discount must be in (0, 1], got {self.discount!r}")

    def potential_of(self, distance: Optional[float]) -> float:
        """``Phi(s)`` for a state at ``distance`` from the goal.

        ``None``/non-finite distance (the oracle's "unreachable") maps to
        ``-inf``: the most negative potential this config can express, so an
        unreachable state is never preferred over a reachable-but-distant
        one, and the resulting non-finite value is deliberately excluded
        from the progress delta by :meth:`NavigationRewardTracker.compute`
        rather than silently producing ``nan``.
        """
        if distance is None or not math.isfinite(distance):
            return -math.inf
        if self.potential == "linear":
            return -self.strength * distance
        # saturating: bounded in (-strength*falloff, 0], flattening the
        # gradient far from the goal.
        return -self.strength * self.falloff * (1.0 - math.exp(-distance / self.falloff))

    def to_contract_dict(self, success_radius: float) -> Dict[str, object]:
        return {
            "potential": self.potential,
            "strength": self.strength,
            "falloff": self.falloff,
            "discount": self.discount,
            "success_radius": success_radius,
            "success_bonus": self.success_bonus,
        }


class NavigationRewardTracker:
    """One episode's potential-based progress reward plus one-time arrival
    bonus.

    ``goal_progress`` and ``goal_reached`` are returned as separate
    components -- never summed into a single opaque scalar -- so that,
    together with the caller's own ``task``/``collision`` components (epic
    §12.5: "keep goal progress, completion, collision and task reward
    separate in the recording and report"), a ``RewardSignal.components``
    dict fully itemizes where a tick's reward came from.
    """

    def __init__(self, config: GoalRewardConfig):
        self.config = config
        self._previous_potential: Optional[float] = None
        self._bonus_paid: bool = False

    def reset(self) -> None:
        self._previous_potential = None
        self._bonus_paid = False

    def prime(self, *, geodesic_distance: Optional[float]) -> None:
        """Seed ``Phi(s_0)`` when a fresh goal becomes active (episode
        start), so the first :meth:`compute` call already has a previous
        potential to diff against -- without this, the very first
        transition's progress would be silently dropped rather than scored.
        """
        self._previous_potential = self.config.potential_of(geodesic_distance)

    def compute(
        self, *, geodesic_distance: Optional[float], goal_active: bool, just_arrived: bool,
    ) -> Dict[str, float]:
        """This tick's ``{"goal_progress": ..., "goal_reached": ...}``
        components (either key omitted when it doesn't apply this tick,
        matching ``RewardSignal.from_components``'s own zero-omission
        convention).

        ``just_arrived`` must be the harness's own arrival transition (e.g.
        ``GoalTracker.evaluate_arrival`` flipping false -> true this tick,
        issue #238's anti-gaming guarantee) -- never a value the policy
        could influence directly -- so the bonus can't be gamed by whatever
        drives ``geodesic_distance`` to zero.
        """
        components: Dict[str, float] = {}
        if goal_active:
            phi_next = self.config.potential_of(geodesic_distance)
            if (
                self._previous_potential is not None
                and math.isfinite(self._previous_potential)
                and math.isfinite(phi_next)
            ):
                progress = self.config.discount * phi_next - self._previous_potential
                components["goal_progress"] = round(progress, 6)
            self._previous_potential = phi_next
        else:
            self._previous_potential = None
        if just_arrived and not self._bonus_paid:
            components["goal_reached"] = self.config.success_bonus
            self._bonus_paid = True
        return components


__all__ = ["GoalRewardConfig", "NavigationRewardTracker"]
