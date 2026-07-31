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
action-name partition to other Programs is #237's job
(``generic_action_effects_v1``), not this one's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from cognitive_runtime.programs.crafter.actions import ACTION_SPACE
from cognitive_runtime.runtime.replay import iter_cognitive_ticks
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

#: Bump whenever the classification rule below changes meaning. This label
#: schema version is surfaced here for inclusion in the Model Factory
#: ``DataContract`` (epic #212 Sec 12.6, mirroring how
#: ``streams.SEMANTIC_VOCABULARY_VERSION`` is wired into it) -- a changed
#: derivation rule must change the corpus's data-contract hash.
ACTION_EFFECT_LABEL_VERSION = "action-effect-v1"

ActionEffectClass = Literal["moved", "turned_only", "blocked", "interacted", "no_op"]

#: The five classes, exhaustive and mutually exclusive (see
#: ``classify_action_effect``'s precedence rule for how ambiguous transitions
#: resolve to exactly one).
ACTION_EFFECT_CLASSES: Tuple[str, ...] = (
    "moved", "turned_only", "blocked", "interacted", "no_op",
)

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


def classify_action_effect(
    *,
    position_changed: bool,
    blocked: bool,
    interacted: bool,
    facing_changed: bool,
) -> ActionEffectClass:
    """The action-effect class for one transition, given its derived signals.

    Classes are mutually exclusive and total; this precedence order resolves
    every ambiguous case a transition's signals can produce:

    1. ``moved`` -- position changed, regardless of what else did. A
       transition that both moved and would otherwise qualify as
       ``interacted`` (a world-changing, non-movement action fired the same
       tick displacement happened) is reported as ``moved``: displacement is
       the more specific fact.
    2. ``blocked`` -- a directional move was attempted and *positively known*
       to have been refused (``blocked`` is only ever set when both
       endpoint positions are recorded and equal -- see
       ``compute_action_effect_labels``; a movement action whose position is
       simply unrecorded is never inferred as blocked). Takes priority over
       a same-transition facing change: Crafter turns *by* attempting a
       blocked move (``programs.crafter.observations`` documents this), so a
       blocked directional attempt is reported as ``blocked``, not
       ``turned_only`` -- the agent acted and the world refused, which is
       the distinction this label exists to capture, not folded into
       ``no_op``.
    3. ``interacted`` -- a non-movement, non-``NULL`` action (chop/mine/
       attack/drink/collect/sleep/place/craft) with no position change.
    4. ``turned_only`` -- facing changed with no position change and no
       movement/interaction action drove it. Doesn't occur in current Crafter
       recordings (every facing change there comes from a directional move,
       already caught by ``blocked`` when position is known, or falls
       through to here when it isn't), but keeps the taxonomy total for a
       world whose turn is a dedicated action rather than a blocked move.
    5. ``no_op`` -- none of the above: the ``NULL`` action, or an unrecognized
       one, with no observed effect.
    """
    if position_changed:
        return "moved"
    if blocked:
        return "blocked"
    if interacted:
        return "interacted"
    if facing_changed:
        return "turned_only"
    return "no_op"


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
