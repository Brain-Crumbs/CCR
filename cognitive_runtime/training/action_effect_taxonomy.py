"""Pure MF-E1 action-effect taxonomy (epic #212 Sec 12.3), shared by
per-transition label derivation (``action_effects.py``, issue #234) and
event-stratified evaluation (``event_evaluation.py``, issue #237).

Deliberately dependency-free (stdlib only): no Crafter, no torch. This keeps
it importable from ``event_evaluation.py``, which must stay usable in a
minimal install, without creating an import cycle through
``action_effects.py -> action_world_model.py -> event_evaluation.py``.
"""

from __future__ import annotations

from typing import Literal, Tuple

ActionEffectClass = Literal["moved", "turned_only", "blocked", "interacted", "no_op"]

#: Bump whenever the classification rule below changes meaning. Surfaced in
#: the Model Factory ``DataContract`` (epic #212 Sec 12.6, mirroring how
#: ``streams.SEMANTIC_VOCABULARY_VERSION`` is wired into it) -- a changed
#: derivation rule must change the corpus's data-contract hash.
ACTION_EFFECT_LABEL_VERSION = "action-effect-v1"

#: The five classes, exhaustive and mutually exclusive (see
#: ``classify_action_effect``'s precedence rule for how ambiguous transitions
#: resolve to exactly one).
ACTION_EFFECT_CLASSES: Tuple[str, ...] = (
    "moved", "turned_only", "blocked", "interacted", "no_op",
)


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


__all__ = [
    "ActionEffectClass",
    "ACTION_EFFECT_LABEL_VERSION",
    "ACTION_EFFECT_CLASSES",
    "classify_action_effect",
]
