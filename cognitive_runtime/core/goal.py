"""Goal representation and the anti-gaming commitment rules (epic #212 §12.4,
issue #238).

The navigation policy planned for the epic's goal-conditioned schedule
(``navigate_open_goal`` and friends, §12.4) receives a goal via the sensory
workspace like any other stream:

    internal.goal.relative_position = [dx, dy]
    internal.goal.distance
    internal.goal.active
    internal.goal.source = caregiver | organism

One composite stream, ``internal.goal`` -- matching the payload shape every
other multi-field ``internal.*`` stream already uses (e.g.
``internal.arbiter_mode``'s ``{mode, surprise, pain, calibration_error}``,
``internal.attention.weights``'s ``{mode, weights, ...}``), not four separate
stream ids.

The source is *always* ``"caregiver"`` today -- ``Program``s in this repo
have no organism-goal-proposal path yet. The organism may eventually propose
its own goals, but it must never receive a completion reward for choosing
its current position. Three mechanisms, required together, defend against
that before they are ever load-bearing:

- a **minimum initial distance** (:attr:`GoalDistributionConfig.
  min_initial_distance`): :meth:`GoalTracker.propose` rejects a candidate
  goal too close to reject picking a goal that is already satisfied;
- a **goal-commitment horizon** (:attr:`GoalDistributionConfig.
  commitment_horizon_ticks`): :meth:`GoalTracker.propose` rejects swapping
  the active goal before the horizon elapses, so a goal cannot be
  re-proposed mid-approach to launder a near-miss into a fresh "arrival";
- **externally evaluated arrival** (:meth:`GoalTracker.evaluate_arrival`):
  only the harness -- the ``Program``'s own ground-truth position, e.g.
  ``CrafterWorld`` reading ``player.pos`` -- can flip ``active`` to
  ``False``. Nothing the policy writes (a motor command, a stream payload)
  feeds this decision.

Building these in now, while the source is still hardcoded to caregiver, is
much cheaper than retrofitting them after organism-proposed goals exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from cognitive_runtime.core.streams.events import StreamSpec

#: One composite internal stream for the whole goal representation -- see
#: the module docstring for why this is one stream, not four.
GOAL_STREAM = "internal.goal"

#: The only sources a goal's payload may declare. ``"organism"`` is reserved
#: for future organism-proposed goals; nothing in this repo passes it yet
#: (issue #238's acceptance criterion: caregiver in every recording this
#: issue produces).
CAREGIVER_SOURCE = "caregiver"
ORGANISM_SOURCE = "organism"
GOAL_SOURCES: Tuple[str, ...] = (CAREGIVER_SOURCE, ORGANISM_SOURCE)

#: "event" is the same modality stand-in every other runtime-computed
#: ``internal.*``/``model.*`` stream uses (``internal.arbiter_mode``,
#: ``internal.attention.weights``, ``core/modulation.py``'s eight streams) --
#: there is no registered "goal" sensory modality, and this is a hand-computed
#: introspection signal, not raw sensing.
GOAL_STREAM_SPEC = StreamSpec(
    GOAL_STREAM, "event",
    "The active navigation goal this tick: position relative to the agent, "
    "Euclidean distance, whether a goal is currently active, and its source "
    "(always 'caregiver' until organism-proposed goals exist; issue #238).",
    payload_schema="{relative_position, distance, active, source}",
)


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class GoalState:
    """One tick's ``internal.goal`` payload, in the agent's own frame.

    Frozen and payload-only: nothing downstream (recorder, policy, replay)
    can mutate ``active`` after the fact -- the harness computes a fresh one
    every tick from :meth:`GoalTracker.state`.
    """

    relative_position: Tuple[float, float]
    distance: float
    active: bool
    source: str

    def __post_init__(self) -> None:
        if self.source not in GOAL_SOURCES:
            raise ValueError(f"unknown goal source {self.source!r}; expected one of {GOAL_SOURCES}")
        if self.distance < 0:
            raise ValueError(f"distance must be >= 0, got {self.distance!r}")

    def as_payload(self) -> dict:
        """The ``internal.goal`` stream payload -- matches the existing
        ``internal.*`` convention of rounding floats for a stable, compact
        recording (``ModulationSignals.as_payloads``)."""
        dx, dy = self.relative_position
        return {
            "relative_position": [round(dx, 6), round(dy, 6)],
            "distance": round(self.distance, 6),
            "active": self.active,
            "source": self.source,
        }


#: The neutral payload published when no goal has ever been proposed --
#: source still declares "caregiver" (issue #238: never "organism" until an
#: organism-proposal path exists), matching AC that every recording this
#: issue produces reports a caregiver source even while inactive.
INACTIVE_GOAL_STATE = GoalState((0.0, 0.0), 0.0, False, CAREGIVER_SOURCE)


@dataclass(frozen=True)
class GoalDistributionConfig:
    """The goal-proposal parameters (epic #212 §12.6): "the goal-distribution
    parameters are exposed for the ``DataContract``."

    Plain, hashable data -- :meth:`to_contract_dict` feeds
    ``DataContract.goal_distribution_policy`` directly, the same way
    MF-E4 (#237) exposed its scenario generator parameters.
    """

    #: A proposed goal closer than this to the current position is rejected
    #: at proposal time (the anti-gaming floor).
    min_initial_distance: float
    #: A goal-commitment horizon, in ticks: the active goal cannot be
    #: replaced until this many ticks have elapsed since it was set.
    commitment_horizon_ticks: int
    #: Distance at or below which the harness considers the goal reached.
    arrival_radius: float
    #: Optional upper bound on the initial distance a proposed goal may sit
    #: at; ``None`` means unbounded (still respecting the world's extent).
    max_initial_distance: Optional[float] = None
    #: Name of the sampling shape, e.g. "uniform_in_bounds"; free-form,
    #: recorded for provenance/contract identity rather than interpreted here.
    distribution: str = "uniform_in_bounds"
    #: Bumped whenever the sampling procedure's meaning changes.
    generator_version: str = "goal_distribution_v1"

    def __post_init__(self) -> None:
        if self.min_initial_distance <= 0:
            raise ValueError(
                f"min_initial_distance must be > 0, got {self.min_initial_distance!r}"
            )
        if self.commitment_horizon_ticks <= 0:
            raise ValueError(
                f"commitment_horizon_ticks must be > 0, got {self.commitment_horizon_ticks!r}"
            )
        if self.arrival_radius < 0:
            raise ValueError(f"arrival_radius must be >= 0, got {self.arrival_radius!r}")
        if self.arrival_radius >= self.min_initial_distance:
            raise ValueError(
                f"arrival_radius ({self.arrival_radius!r}) must be < min_initial_distance "
                f"({self.min_initial_distance!r}), or a proposed goal could already count as "
                "arrived"
            )
        if self.max_initial_distance is not None and (
            self.max_initial_distance < self.min_initial_distance
        ):
            raise ValueError(
                f"max_initial_distance ({self.max_initial_distance!r}) must be >= "
                f"min_initial_distance ({self.min_initial_distance!r})"
            )

    def to_contract_dict(self) -> dict:
        """Plain-dict shape for ``DataContract.goal_distribution_policy``
        (epic #212 §12.6)."""
        return {
            "min_initial_distance": self.min_initial_distance,
            "max_initial_distance": self.max_initial_distance,
            "commitment_horizon_ticks": self.commitment_horizon_ticks,
            "arrival_radius": self.arrival_radius,
            "distribution": self.distribution,
            "generator_version": self.generator_version,
        }


class GoalTracker:
    """Owns one Program's active goal plus the anti-gaming commitment rules.

    Program-agnostic: any ``Program`` (Crafter today; Minecraft or another
    world later) can drive one of these with its own ground-truth position.
    Nothing here reads or writes the motor/sensory buses -- callers own
    wiring :meth:`state` into a published stream.
    """

    def __init__(self, config: GoalDistributionConfig):
        self.config = config
        self._goal_position: Optional[Tuple[float, float]] = None
        self._source: str = CAREGIVER_SOURCE
        self._committed_at_tick: Optional[int] = None
        self._arrived: bool = False

    @property
    def has_goal(self) -> bool:
        return self._goal_position is not None

    @property
    def active(self) -> bool:
        """A goal is active once proposed and until the harness decides it
        has been reached -- never flipped by anything but
        :meth:`evaluate_arrival`."""
        return self._goal_position is not None and not self._arrived

    def propose(
        self,
        *,
        goal_position: Tuple[float, float],
        current_position: Tuple[float, float],
        tick: int,
        source: str = CAREGIVER_SOURCE,
    ) -> Optional[str]:
        """Attempt to set a new goal.

        Returns ``None`` on success, or a human-readable rejection reason on
        failure -- mirrors the ``event.action_rejected`` convention
        (``programs/crafter/adapter.py:step``) rather than raising, so a
        caller can record *why* a proposal was refused instead of choosing
        between a crash and swallowing an exception.

        Rejects (without changing any existing goal):

        - an unknown ``source``;
        - a candidate closer than ``config.min_initial_distance`` to
          ``current_position`` (the anti-gaming floor);
        - a candidate while an existing goal is still active and its
          commitment horizon has not yet elapsed.
        """
        if source not in GOAL_SOURCES:
            return f"unknown goal source {source!r}; expected one of {GOAL_SOURCES}"
        distance = _euclidean(goal_position, current_position)
        if distance < self.config.min_initial_distance:
            return (
                f"proposed goal {goal_position} is {distance:.3f} from current position "
                f"{current_position}, below the minimum initial distance "
                f"{self.config.min_initial_distance}"
            )
        if self.active and self._committed_at_tick is not None:
            unlocks_at = self._committed_at_tick + self.config.commitment_horizon_ticks
            if tick < unlocks_at:
                return (
                    f"active goal committed at tick {self._committed_at_tick} cannot change "
                    f"before tick {unlocks_at} (commitment horizon "
                    f"{self.config.commitment_horizon_ticks} ticks); proposal at tick {tick}"
                )
        self._goal_position = goal_position
        self._source = source
        self._committed_at_tick = tick
        self._arrived = False
        return None

    def evaluate_arrival(self, current_position: Tuple[float, float]) -> bool:
        """Harness-only arrival decision: distance from ``current_position``
        (the caller's own ground truth, never a policy-writable value) to
        the active goal versus ``config.arrival_radius``.

        Idempotent and monotonic: once arrived, stays arrived until the next
        accepted :meth:`propose` call. Returns the (possibly unchanged)
        arrived state.
        """
        if self._goal_position is None or self._arrived:
            return self._arrived
        if _euclidean(self._goal_position, current_position) <= self.config.arrival_radius:
            self._arrived = True
        return self._arrived

    def state(self, current_position: Tuple[float, float]) -> GoalState:
        """This tick's ``internal.goal`` payload, given the harness's own
        current position. Does not itself evaluate arrival -- callers that
        want arrival reflected here must call :meth:`evaluate_arrival` first
        (typically once per tick, right after advancing the world)."""
        if self._goal_position is None:
            return GoalState((0.0, 0.0), 0.0, False, self._source)
        dx = self._goal_position[0] - current_position[0]
        dy = self._goal_position[1] - current_position[1]
        return GoalState(
            (dx, dy), _euclidean(self._goal_position, current_position), self.active, self._source
        )
