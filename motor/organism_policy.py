"""Motor-freedom-driven policy (issue #105): the organism's motor output
under one of the ladder's three ``motor_freedom`` values.

Neither ``motor/reflexes.py`` (caregiver/reflex precedence, issue #102/#103)
nor ``motor/voluntary.py`` (the Phase 6 MPC/voluntary path) previously
implemented ``cognitive_runtime.core.policy.Policy`` -- there was no seam
driving either one from a stage's declared ``motor_freedom``
(``development.definitions.CurriculumStageSpec``). This module is that seam:

- ``frozen``: no motor output at all -- Gestation is genuinely inert, not
  "voluntary chooses NULL every tick".
- ``overridden``: the "voluntary" input each tick comes from a
  caregiver/scripted ``Policy`` (e.g. ``ConstantActionPolicy``,
  ``ScriptedSequencePolicy``), not the organism's own decision; a
  ``CaregiverChannel`` injection, when present, still outranks it (and any
  reflex) per ``ReflexStack``'s precedence contract.
- ``learned``: the "voluntary" input comes from a ``VoluntaryController``
  (Phase 6); reflexes can still veto on top, but no ``CaregiverChannel``
  drives it -- only ``overridden`` stages exercise one (Phase 7 table:
  "the caregiver override is active exactly in the stages that declare it").

Wiring this into a live ``CognitiveRuntime``/``CrafterWorld`` episode end to
end (which would need the real predictive cortex as a ``learned`` stage's
``VoluntaryController.predictor``) is out of scope for issue #105 -- see the
phase doc's Milestone 7, which only requires Gestation->Crawling to be
CI-runnable. This module is unit-tested standalone instead.
"""

from __future__ import annotations

from typing import Optional, Sequence

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import Policy, SingleActionPolicy
from cognitive_runtime.core.world_model import Prediction
from development.definitions import MOTOR_FREEDOMS, CurriculumStageSpec
from motor.reflexes import CaregiverChannel, ReflexStack, Stimulus
from motor.voluntary import VoluntaryController


def _single_action(policy: Policy, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
    """Adapt any ``Policy`` (single- or multi-action) to one ``Action`` for
    the reflex stack's ``voluntary`` input -- an empty emission is NULL,
    same convention ``SingleActionPolicy.emit`` already uses in reverse."""
    emitted = policy.emit(state, memory, prediction)
    return emitted[0] if emitted else NULL_ACTION


class MotorFreedomPolicy(SingleActionPolicy):
    """Drives one stage's declared ``motor_freedom`` every tick. See the
    module docstring for the three freedoms' exact behavior."""

    name = "motor-freedom"

    def __init__(
        self,
        motor_freedom: str,
        action_space: Sequence[Action],
        *,
        scripted: Optional[Policy] = None,
        voluntary: Optional[VoluntaryController] = None,
        reflexes: Optional[ReflexStack] = None,
        caregiver: Optional[CaregiverChannel] = None,
        stimuli: Sequence[Stimulus] = (),
        goal: object = None,
    ) -> None:
        if motor_freedom not in MOTOR_FREEDOMS:
            raise ValueError(
                f"unknown motor_freedom {motor_freedom!r}; expected one of {MOTOR_FREEDOMS}"
            )
        if motor_freedom == "overridden" and scripted is None:
            raise ValueError(
                "motor_freedom='overridden' requires a scripted/caregiver policy "
                "(the tick's voluntary input has nowhere else to come from)"
            )
        if motor_freedom == "learned" and voluntary is None:
            raise ValueError(
                "motor_freedom='learned' requires a voluntary controller "
                "(the tick's voluntary input has nowhere else to come from)"
            )
        self.motor_freedom = motor_freedom
        self.action_space = list(action_space)
        self.scripted = scripted
        self.voluntary = voluntary
        self.reflexes = reflexes
        #: Only ever drained on the ``overridden`` path (see ``decide``) --
        #: a caller may still pass one for ``frozen``/``learned``, but it is
        #: never consulted, matching "the caregiver override is active
        #: exactly in the stages that declare it".
        self.caregiver = caregiver
        self.stimuli = tuple(stimuli)
        self.goal = goal

    def reset(self) -> None:
        if self.scripted is not None:
            self.scripted.reset()

    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        if self.motor_freedom == "frozen":
            return NULL_ACTION

        if self.motor_freedom == "overridden":
            assert self.scripted is not None
            voluntary_action = _single_action(self.scripted, state, memory, prediction)
            if self.reflexes is None:
                return voluntary_action
            caregiver_override = self.caregiver.drain() if self.caregiver is not None else None
            return self.reflexes.decide(voluntary_action, self.stimuli, caregiver_override).actuated

        # "learned": Objects/Foraging hand control to the voluntary path.
        assert self.voluntary is not None
        voluntary_action = self.voluntary.choose(state, self.action_space, self.goal)
        if self.reflexes is None:
            return voluntary_action
        # No caregiver channel drives a learned-motor stage, by design --
        # `self.caregiver` (if any) is deliberately never drained here.
        return self.reflexes.decide(voluntary_action, self.stimuli, None).actuated


def build_stage_policy(
    stage: CurriculumStageSpec,
    action_space: Sequence[Action],
    *,
    scripted: Optional[Policy] = None,
    voluntary: Optional[VoluntaryController] = None,
    reflexes: Optional[ReflexStack] = None,
    caregiver: Optional[CaregiverChannel] = None,
    stimuli: Sequence[Stimulus] = (),
    goal: object = None,
) -> Policy:
    """Build the ``Policy`` a stage's declared ``motor_freedom`` drives
    (Phase 7 table: Gestation freezes, Babbling/Crawling caregiver-override
    a scripted policy, Objects/Foraging hand control to the voluntary path).
    Raises if ``stage.motor_freedom`` is unset, or if the collaborator its
    freedom requires (``scripted`` for ``overridden``, ``voluntary`` for
    ``learned``) is missing -- a stage that declares a freedom but is run
    without the thing that freedom needs is a wiring bug, not something to
    default around."""
    if stage.motor_freedom is None:
        raise ValueError(f"stage {stage.name!r} has no declared motor_freedom")
    return MotorFreedomPolicy(
        stage.motor_freedom, action_space,
        scripted=scripted, voluntary=voluntary, reflexes=reflexes, caregiver=caregiver,
        stimuli=stimuli, goal=goal,
    )
