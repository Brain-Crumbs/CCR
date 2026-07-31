"""Per-transition action-effect labels (issue #234, MF-E1 -- epic #212 Sec 12.3).

Derives, per transition, from **already-recorded** Crafter fields: position
delta and movement magnitude, facing delta, a blocked/contact indicator,
local semantic-grid delta, and a mutually-exclusive, exhaustive
action-effect class (``moved`` / ``turned_only`` / ``blocked`` /
``interacted`` / ``no_op``).

Derivation only (epic Sec 12.3): no re-recording, no new model heads --
those are a subsequent, separately-declared architecture branch. This
module aligns with ``TransitionLabel``/``compute_scenario_transition_labels``
(``training.nursery``, issue #202) rather than inventing a second, divergent
taxonomy, and reuses the one-tick actuation-latency alignment and tick
accessors ``_tick_position``/``_tick_facing``/``_tick_action_name`` that
``action_world_model.build_action_sequence_dataset`` already defines.

Crafter-specific by design, not by accident: the epic's own text scopes this
first increment to Crafter ("Crafter already records position, discrete
facing, local semantic grids, and actions..."), and only
``MOVEMENT_ACTION_NAMES``/``INTERACTION_ACTION_NAMES`` -- which need to know
Crafter's own verb vocabulary (``MOVE_*`` vs. ``DO``/``SLEEP``/``PLACE_*``/
``MAKE_*``) -- actually require that dependency. The stream ids this module
reads (``vision.frame.grid``, ``spatial.position``, ``spatial.facing``) are a
generic ``core.streams`` naming convention Minecraft's own catalog shares, so
they're declared locally rather than imported from ``programs.crafter``, the
same way ``action_world_model.py`` declares its own. Generalizing the
action-name partition to other Programs is a later increment's job.

The classification rule itself (``classify_action_effect``) and its five-class
taxonomy live in ``action_effect_taxonomy.py``, a dependency-free module this
one re-exports from: issue #237 reuses that exact rule for event-stratified
evaluation (``event_evaluation.py``) without pulling Crafter or this module's
torch-adjacent imports into that lighter-weight evaluation path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from cognitive_runtime.programs.crafter.actions import ACTION_SPACE
from cognitive_runtime.runtime.replay import iter_cognitive_ticks
from cognitive_runtime.training.action_effect_taxonomy import (  # noqa: F401
    ACTION_EFFECT_CLASSES, ACTION_EFFECT_LABEL_VERSION, ActionEffectClass, classify_action_effect,
)
from cognitive_runtime.training.action_world_model import (
    PIXEL_STREAM, _tick_action_name, _tick_facing, _tick_position,
)

#: Same generic stream id both ``programs.minecraft.streams`` and
#: ``programs.crafter.streams`` publish under (a shared "vision.frame.grid"
#: convention, not Crafter-specific content) -- hardcoded here rather than
#: imported from either Program module, matching how
#: ``action_world_model.py`` declares ``PIXEL_STREAM``/``FACING_STREAM``/
#: ``POSITION_STREAM`` as its own local constants instead of reaching into a
#: particular Program's stream catalog.
VISION_STREAM = "vision.frame.grid"

#: Crafter's four directional actions -- the only actions that move the
#: agent, and the only ones ``spatial.facing`` tracks: every directional
#: attempt updates facing even when the target cell refuses the move
#: (``programs.crafter.observations.build_state``'s facing comment, issue
#: #90).
MOVEMENT_ACTION_NAMES = frozenset(
    action.name for action in ACTION_SPACE if action.name.startswith("MOVE_")
)

#: Every other non-``NULL`` action: chop/mine/attack/drink/collect, sleep,
#: place, craft. Crafter's action space has no action that is neither a
#: move, an interaction, nor ``NULL``
#: (``programs.crafter.actions.CRAFTER_ACTIONS``).
INTERACTION_ACTION_NAMES = frozenset(
    action.name for action in ACTION_SPACE
    if action.name not in MOVEMENT_ACTION_NAMES and action.name != "NULL"
)


@dataclass
class ActionEffectLabel:
    """One transition's derived action-effect label."""

    #: The action that actually drove this transition (one-tick actuation
    #: latency already resolved -- see ``compute_action_effect_labels``).
    action: str
    #: ``(dx, dy)`` grid displacement, or ``None`` when position wasn't
    #: recorded at both endpoints of the transition.
    position_delta: Optional[Tuple[float, float]]
    #: Euclidean norm of ``position_delta``; ``0.0`` when unavailable.
    movement_magnitude: float
    facing_changed: bool
    #: A directional move was attempted and the world is positively known to
    #: have refused it (both endpoint positions recorded and equal) -- the
    #: agent acted and the world refused, distinct from an ordinary idle
    #: frame. Never set when position is simply unrecorded.
    blocked: bool
    semantic_grid_changed: bool
    action_effect_class: ActionEffectClass


def _euclidean(delta: Optional[Tuple[float, float]]) -> float:
    if delta is None:
        return 0.0
    return math.hypot(delta[0], delta[1])


def compute_action_effect_labels(session_dir: str, episode_id: str) -> List[ActionEffectLabel]:
    """Per-transition action-effect labels for one recorded Crafter episode.

    Reads ``spatial.position``, ``spatial.facing`` and ``vision.frame.grid``
    the same tick-grouped way ``compute_scenario_transition_labels`` does,
    aligned to the action that actually drove each transition with the same
    one-tick actuation-latency convention
    ``action_world_model.build_action_sequence_dataset`` uses (issue #202):
    the action reported for transition ``i -> i+1`` is the one
    ``program.step()`` applied between those two frames, not the action
    recorded alongside either frame's own tick.

    Deterministic: reading the same session twice yields identical labels,
    since it only replays already-recorded fields.
    """
    positions: List[Optional[Tuple[float, float]]] = []
    facings: List[Optional[Tuple[float, float]]] = []
    grids: List[Optional[Tuple[Tuple[int, ...], ...]]] = []
    driving_actions: List[str] = []

    last_committed_action: str = "NULL"
    last_facing: Optional[Tuple[float, float]] = None
    last_position: Optional[Tuple[float, float]] = None
    last_grid: Optional[Tuple[Tuple[int, ...], ...]] = None

    for _decision, sensory, motor in iter_cognitive_ticks(session_dir, episode_id):
        # A NULL-emitting policy publishes zero motor.command events this
        # tick (``Policy.emit``'s "an empty list is NULL"), and
        # ``CrafterWorld.step()`` applies NULL when it drains none -- so an
        # absent tick action means NULL was actually applied, not "keep
        # whatever the last real action was" (a move-then-idle sequence
        # would otherwise mislabel the idle transition as another move).
        this_tick_action_name = _tick_action_name(motor) or "NULL"
        facing = _tick_facing(sensory) or last_facing
        if facing is not None:
            last_facing = facing
        position = _tick_position(sensory) or last_position
        if position is not None:
            last_position = position
        for record in sensory:
            if record.get("stream_id") == VISION_STREAM:
                grid = record.get("payload")
                if isinstance(grid, list):
                    last_grid = tuple(tuple(row) for row in grid)

        if any(record.get("stream_id") == PIXEL_STREAM for record in sensory):
            if positions:
                driving_actions.append(last_committed_action)
            positions.append(position)
            facings.append(facing)
            grids.append(last_grid)

        # Fold this tick's own emission in only after this tick's frame (if
        # any) was recorded against the *previous* tick's command --
        # program.step() applies a motor command at the start of the
        # following tick, so it drives the *next* frame, not this one.
        last_committed_action = this_tick_action_name

    labels: List[ActionEffectLabel] = []
    for i, action in enumerate(driving_actions):
        pos_a, pos_b = positions[i], positions[i + 1]
        position_known = pos_a is not None and pos_b is not None
        position_delta = (
            (pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]) if position_known else None
        )
        position_changed = position_known and position_delta != (0.0, 0.0)

        facing_a, facing_b = facings[i], facings[i + 1]
        facing_changed = facing_a is not None and facing_b is not None and facing_a != facing_b

        grid_a, grid_b = grids[i], grids[i + 1]
        semantic_grid_changed = grid_a is not None and grid_b is not None and grid_a != grid_b

        # Only assert "the world refused this move" when both endpoint
        # positions are actually known and equal -- a movement action whose
        # position is simply unrecorded (older/filtered recordings) must not
        # be fabricated into a contact label; it falls through to
        # turned_only/no_op below instead.
        blocked = position_known and not position_changed and action in MOVEMENT_ACTION_NAMES
        interacted = not position_changed and action in INTERACTION_ACTION_NAMES

        action_effect_class = classify_action_effect(
            position_changed=position_changed, blocked=blocked,
            interacted=interacted, facing_changed=facing_changed,
        )

        labels.append(
            ActionEffectLabel(
                action=action,
                position_delta=position_delta,
                movement_magnitude=_euclidean(position_delta),
                facing_changed=facing_changed,
                blocked=blocked,
                semantic_grid_changed=semantic_grid_changed,
                action_effect_class=action_effect_class,
            )
        )
    return labels
