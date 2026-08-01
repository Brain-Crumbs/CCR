"""A* oracle: training-only geodesic navigation labels (epic #212 §12.4/§12.5,
issue #239, gated on MF-F1/#238).

The simulator has a complete map; the deployed policy does not -- it only
ever sees local observations plus the relative goal (``internal.goal``,
issue #238). This module is the "complete map" side: an A* planner over
Crafter's *raw* per-cell materials (not the compact 6-class vision
vocabulary ``programs.crafter.streams`` publishes -- see
:func:`build_walkable_grid`'s docstring for why the two must not be
conflated), producing shortest-path/geodesic distance, the next optimal
action, a legal-action mask, a path-progress delta, and a replan-required
flag.

**These are training-only labels.** Nothing in this module touches a
``StreamSpec``, a ``SensoryStreamBus``, or ``CrafterWorld.stream_catalog()``
-- the structural barrier the epic asks for is simply that this module has
no import of, or reference to, the streams machinery at all. ``CrafterWorld.
oracle_labels()`` (``adapter.py``) is the only thing that calls into this
module, and it hands the result to the runtime's decision record, never to
the sensory bus (see ``runtime.loop``'s ``oracle_labels`` wiring). A test
that enumerates ``stream_catalog()``'s stream ids is what makes this a
mechanically enforced rule rather than a naming convention.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from cognitive_runtime.programs.crafter.streams import SEMANTIC_LEGEND_NAMES

Position = Tuple[int, int]

#: Bump whenever the search algorithm or its tie-breaking/heuristic changes
#: meaning. Surfaced in the Model Factory ``DataContract`` (epic #212 §12.6:
#: "oracle-planner version and its map-cost assumptions").
ORACLE_PLANNER_VERSION = "crafter-astar-v1"

#: Crafter's own movement mapping (``crafter.objects.Player._move``):
#: left/right move along x, up/down along y, with up/down using screen-space
#: sign (up decreases y). Hardcoded here rather than imported from
#: ``programs.crafter.actions`` so this module stays crafter-package-free,
#: matching ``programs.crafter.streams``/``goals.py``'s own convention.
MOVE_DIRECTIONS: Dict[str, Position] = {
    "MOVE_LEFT": (-1, 0),
    "MOVE_RIGHT": (1, 0),
    "MOVE_UP": (0, -1),
    "MOVE_DOWN": (0, 1),
}
_DIRECTION_TO_ACTION: Dict[Position, str] = {v: k for k, v in MOVE_DIRECTIONS.items()}

_NAME_TO_RAW_ID: Dict[str, int] = {name: raw_id for raw_id, name in SEMANTIC_LEGEND_NAMES.items()}

#: Crafter's own ``walkable`` materials (``crafter/data.yaml``): the only
#: materials ``Object.is_free`` accepts for movement, before the player's own
#: ``+lava`` extension (``crafter.objects.Player.walkable``).
_DEFAULT_WALKABLE_NAMES: Tuple[str, ...] = ("grass", "path", "sand")


@dataclass(frozen=True)
class OracleMapCostAssumptions:
    """The oracle planner's declared movement-legality/cost model (epic
    #212 §12.6: "oracle-planner version and its map-cost assumptions are
    exposed for the ``DataContract``").

    A cell is walkable iff its material is in ``walkable_materials`` (plus
    ``lava`` when ``avoid_lava`` is off) *and* no tracked object (a mob, an
    arrow, a sapling -- ids 13+ in ``SEMANTIC_LEGEND_NAMES``) currently
    occupies it, mirroring ``crafter.objects.Object.is_free``'s own
    ``obj is None and material in materials`` check. Object occupancy is
    read fresh from the simulator's full state every tick (``OracleLabelWriter.
    label``), so a cell blocked by a moving mob this tick is not a permanent
    wall -- the oracle simply replans against the current snapshot.
    """

    connectivity: str = "4-directional"
    step_cost: float = 1.0
    walkable_materials: Tuple[str, ...] = _DEFAULT_WALKABLE_NAMES
    #: Lava is *walkable* to Crafter's own physics (``Player.walkable``) --
    #: stepping onto it sets health to 0 outright (``Player._move``). The
    #: oracle excludes it from its own walkable set by default so a
    #: shortest-path/expert-action label never routes the demonstrator
    #: through a lethal cell just because it happens to be geodesically
    #: shorter.
    avoid_lava: bool = True

    def __post_init__(self) -> None:
        if self.step_cost <= 0:
            raise ValueError(f"step_cost must be > 0, got {self.step_cost!r}")
        unknown = set(self.walkable_materials) - set(_NAME_TO_RAW_ID)
        if unknown:
            raise ValueError(f"unknown walkable material name(s): {sorted(unknown)}")

    def walkable_raw_ids(self) -> frozenset:
        ids = {_NAME_TO_RAW_ID[name] for name in self.walkable_materials}
        if not self.avoid_lava:
            ids.add(_NAME_TO_RAW_ID["lava"])
        return frozenset(ids)

    def to_contract_dict(self) -> Dict[str, object]:
        return {
            "planner_version": ORACLE_PLANNER_VERSION,
            "connectivity": self.connectivity,
            "step_cost": self.step_cost,
            "walkable_materials": list(self.walkable_materials),
            "avoid_lava": self.avoid_lava,
        }


DEFAULT_MAP_COST_ASSUMPTIONS = OracleMapCostAssumptions()


def build_walkable_grid(
    semantic: np.ndarray, assumptions: OracleMapCostAssumptions = DEFAULT_MAP_COST_ASSUMPTIONS,
) -> np.ndarray:
    """A boolean walkable grid from the simulator's *full raw* semantic map
    (e.g. ``crafter.Env._sem_view()``) -- deliberately not the compact
    6-class ``vision.frame.grid`` vocabulary (``programs.crafter.streams``):
    that encoder collapses lava, stone, trees, ore and crafted structures
    into one "solid" class for the policy's local egocentric crop, which
    would make lava (physically walkable, merely lethal) and stone
    (genuinely impassable) indistinguishable to the oracle. The oracle plans
    against the simulator's own ground truth instead, per its "complete map"
    mandate (epic #212 §12.4).
    """
    walkable_ids = assumptions.walkable_raw_ids()
    return np.isin(semantic, list(walkable_ids))


def _in_bounds(pos: Position, world_size: Tuple[int, int]) -> bool:
    width, height = world_size
    return 0 <= pos[0] < width and 0 <= pos[1] < height


def astar_path(
    walkable: np.ndarray, start: Position, goal: Position, world_size: Tuple[int, int],
) -> Optional[List[Position]]:
    """Shortest 4-connected, unit-cost path from ``start`` to ``goal``
    (inclusive of both endpoints), or ``None`` if no path exists.

    ``start`` is always a valid path origin regardless of its own
    walkability (the agent's actual current cell, e.g. it stepped onto lava
    this tick) -- only cells *entered* along the path must be walkable,
    matching how ``goal`` itself must be walkable too (this planner does not
    special-case an unwalkable goal cell into "adjacent counts"; a goal
    sampled onto a wall is honestly reported as unreachable rather than
    silently redefined).
    """
    if not _in_bounds(start, world_size) or not _in_bounds(goal, world_size):
        return None
    if start == goal:
        return [start]

    def heuristic(pos: Position) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    counter = 0  # heap tie-breaker: positions aren't orderable against each other
    open_heap: List[Tuple[int, int, int, Position]] = [(heuristic(start), 0, counter, start)]
    came_from: Dict[Position, Position] = {}
    best_g: Dict[Position, int] = {start: 0}
    closed: set = set()

    while open_heap:
        _, g, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        closed.add(current)
        for dx, dy in MOVE_DIRECTIONS.values():
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in closed or not _in_bounds(neighbor, world_size):
                continue
            if not bool(walkable[neighbor[0], neighbor[1]]):
                continue
            tentative_g = g + 1
            if tentative_g < best_g.get(neighbor, math.inf):
                best_g[neighbor] = tentative_g
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(open_heap, (tentative_g + heuristic(neighbor), tentative_g, counter, neighbor))
    return None


def compute_legal_action_mask(
    walkable: np.ndarray, position: Position, world_size: Tuple[int, int],
) -> Dict[str, bool]:
    """Which of the four movement actions land on an in-bounds, walkable
    cell from ``position`` right now -- the simulator's own ground-truth
    legality, independent of any particular shortest path."""
    mask: Dict[str, bool] = {}
    for name, (dx, dy) in MOVE_DIRECTIONS.items():
        neighbor = (position[0] + dx, position[1] + dy)
        mask[name] = _in_bounds(neighbor, world_size) and bool(walkable[neighbor[0], neighbor[1]])
    return mask


@dataclass(frozen=True)
class OracleLabel:
    """One tick's training-only oracle labels (epic #212 §12.4)."""

    #: Shortest-path distance in grid steps; ``None`` when the goal is
    #: currently unreachable from the agent's position.
    geodesic_distance: Optional[float]
    #: The action that begins the shortest path; ``None`` at the goal itself
    #: or when unreachable.
    next_action: Optional[str]
    legal_action_mask: Dict[str, bool]
    #: ``previous_tick_distance - this_tick_distance``; ``0.0`` on the first
    #: tick a goal is active (no previous distance yet) and whenever either
    #: side of the comparison is unreachable.
    path_progress_delta: float
    #: True when the goal became unreachable, or this tick's geodesic
    #: distance is *worse* than last tick's despite the world otherwise
    #: advancing -- a previously assumed route was invalidated (a new
    #: obstacle, a mob blocking the planned cell) and the policy's plan
    #: needs to be recomputed, not merely continued.
    replan_required: bool
    planner_version: str
    map_cost_assumptions: Dict[str, object]

    def as_payload(self) -> Dict[str, object]:
        return {
            "geodesic_distance": self.geodesic_distance,
            "next_action": self.next_action,
            "legal_action_mask": dict(self.legal_action_mask),
            "path_progress_delta": round(self.path_progress_delta, 6),
            "replan_required": self.replan_required,
            "planner_version": self.planner_version,
            "map_cost_assumptions": dict(self.map_cost_assumptions),
        }


class OracleLabelWriter:
    """Owns one episode's A*-derived training-only navigation labels.

    Stateful only in the sense of remembering the previous tick's geodesic
    distance (for ``path_progress_delta``/``replan_required``); everything
    else is recomputed fresh from the simulator's full state every call, so
    a wall the player places or a tree they chop is reflected immediately.
    """

    def __init__(self, assumptions: OracleMapCostAssumptions = DEFAULT_MAP_COST_ASSUMPTIONS):
        self.assumptions = assumptions
        self._previous_distance: Optional[float] = None

    def reset(self) -> None:
        self._previous_distance = None

    def label(
        self, *, semantic: np.ndarray, position: Position, goal_position: Position,
        world_size: Tuple[int, int],
    ) -> OracleLabel:
        walkable = build_walkable_grid(semantic, self.assumptions)
        path = astar_path(walkable, position, goal_position, world_size)
        distance: Optional[float] = float(len(path) - 1) if path is not None else None

        next_action: Optional[str] = None
        if path is not None and len(path) > 1:
            dx = path[1][0] - path[0][0]
            dy = path[1][1] - path[0][1]
            next_action = _DIRECTION_TO_ACTION.get((dx, dy))

        legal_mask = compute_legal_action_mask(walkable, position, world_size)

        if self._previous_distance is not None and distance is not None:
            progress_delta = self._previous_distance - distance
        else:
            progress_delta = 0.0

        replan_required = distance is None or (
            self._previous_distance is not None and distance > self._previous_distance
        )

        label = OracleLabel(
            geodesic_distance=distance,
            next_action=next_action,
            legal_action_mask=legal_mask,
            path_progress_delta=progress_delta,
            replan_required=replan_required,
            planner_version=ORACLE_PLANNER_VERSION,
            map_cost_assumptions=self.assumptions.to_contract_dict(),
        )
        self._previous_distance = distance
        return label


__all__ = [
    "Position",
    "ORACLE_PLANNER_VERSION",
    "MOVE_DIRECTIONS",
    "OracleMapCostAssumptions",
    "DEFAULT_MAP_COST_ASSUMPTIONS",
    "build_walkable_grid",
    "astar_path",
    "compute_legal_action_mask",
    "OracleLabel",
    "OracleLabelWriter",
]
