"""Scripted orienting reflex (issue #60): precedence rules (risk veto,
never suppressing a world-changing policy action), the localized-capture
firing condition, bounded hold duration, and a simulated end-to-end scenario
(entity appears -> agent turns toward it, decision record attributes it)."""

from __future__ import annotations

import pytest

from typing import Any, Dict, List, Optional

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.action_registry import ActionDeclaration, ActionRegistry
from cognitive_runtime.core.attention import (
    AttentionReason,
    AttentionSignal,
    AttentionState,
    StimulusDirection,
)
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.orienting_reflex import (
    REFLEX_MODES,
    OrientingReflex,
    OrientingReflexConfig,
)
from cognitive_runtime.core.program import ActionResult, Program, ProgramMetadata
from cognitive_runtime.core.reward import RewardSignal
from cognitive_runtime.core.streams.encoders import ScalarEncoder
from cognitive_runtime.core.streams.registry import (
    DEFAULT_STREAM_REGISTRY,
    AttentionMetadata,
    StreamDeclaration,
    StreamRegistry,
)
from cognitive_runtime.core.streams.shim import ObservationStreamShim
from cognitive_runtime.policies.null_policy import NullPolicy
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.replay import iter_cognitive_ticks

_REGISTRY = ActionRegistry(
    [
        ActionDeclaration("LOOK_LEFT", world_changing=False, information_gathering=True),
        ActionDeclaration("LOOK_RIGHT", world_changing=False, information_gathering=True),
        ActionDeclaration("MOVE_FORWARD", world_changing=True, information_gathering=True),
        ActionDeclaration("NULL", world_changing=False, information_gathering=True),
    ]
)


def _state(
    focus_stream="stimulus",
    bearing_deg=30.0,
    bottom_up_capture=True,
    reasons=None,
):
    if reasons is None:
        signal = AttentionSignal(
            stream_id=focus_stream, novelty=1.0, prediction_error=0.0, uncertainty=None,
            reward_relevance=0.0, risk=0.0, recency=1.0, boredom=0.0, compute_cost=0.0,
            direction=StimulusDirection(bearing_deg=bearing_deg) if bearing_deg is not None else None,
        )
        reasons = {focus_stream: AttentionReason(signal=signal, score=1.0, components={})}
    return AttentionState(
        tick_index=0, mode="budgeted", weights={focus_stream: 1.0},
        selected_streams=(focus_stream,), focus_stream=focus_stream,
        budget_used=1.0, budget_total=4.0, reasons=reasons,
        bottom_up_capture=bottom_up_capture,
    )


# ------------------------------------------------------------- config validation


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown reflex mode"):
        OrientingReflexConfig(mode="bogus")


def test_config_rejects_non_positive_hold_ticks():
    with pytest.raises(ValueError, match="hold_ticks"):
        OrientingReflexConfig(hold_ticks=0)


def test_config_rejects_out_of_range_risk_threshold():
    with pytest.raises(ValueError, match="risk_veto_threshold"):
        OrientingReflexConfig(risk_veto_threshold=1.5)


def test_reflex_modes_contains_the_documented_three():
    assert REFLEX_MODES == {"on", "off", "learned-only"}


# ------------------------------------------------------------- firing condition


def test_fires_on_bottom_up_capture_with_direction():
    reflex = OrientingReflex(action_registry=_REGISTRY)
    decision = reflex.decide(_state(bearing_deg=30.0), risk=0.0, policy_actions=[])
    assert decision is not None
    assert decision.action == Action("LOOK_RIGHT")
    assert decision.stimulus_stream == "stimulus"
    assert decision.direction == {"bearing_deg": 30.0, "region": None}


def test_negative_bearing_turns_left():
    reflex = OrientingReflex(action_registry=_REGISTRY)
    decision = reflex.decide(_state(bearing_deg=-45.0), risk=0.0, policy_actions=[])
    assert decision.action == Action("LOOK_LEFT")


def test_no_fire_without_bottom_up_capture():
    reflex = OrientingReflex(action_registry=_REGISTRY)
    state = _state(bearing_deg=30.0, bottom_up_capture=False)
    assert reflex.decide(state, risk=0.0, policy_actions=[]) is None


def test_no_fire_when_focus_stream_carries_no_direction():
    reflex = OrientingReflex(action_registry=_REGISTRY)
    signal = AttentionSignal(
        stream_id="body.health", novelty=1.0, prediction_error=0.0, uncertainty=None,
        reward_relevance=0.0, risk=0.0, recency=1.0, boredom=0.0, compute_cost=0.0,
        direction=None,
    )
    reasons = {"body.health": AttentionReason(signal=signal, score=1.0, components={})}
    state = _state(focus_stream="body.health", reasons=reasons)
    assert reflex.decide(state, risk=0.0, policy_actions=[]) is None


def test_no_fire_when_no_focus_stream():
    reflex = OrientingReflex(action_registry=_REGISTRY)
    state = AttentionState(
        tick_index=0, mode="budgeted", weights={}, selected_streams=(),
        focus_stream=None, budget_used=0.0, budget_total=4.0, reasons={},
        bottom_up_capture=True,
    )
    assert reflex.decide(state, risk=0.0, policy_actions=[]) is None


def test_bearing_within_deadzone_does_not_fire():
    reflex = OrientingReflex(
        config=OrientingReflexConfig(bearing_deadzone_deg=15.0), action_registry=_REGISTRY,
    )
    assert reflex.decide(_state(bearing_deg=10.0), risk=0.0, policy_actions=[]) is None


# ------------------------------------------------------------- precedence rules


def test_off_mode_never_fires():
    reflex = OrientingReflex(config=OrientingReflexConfig(mode="off"), action_registry=_REGISTRY)
    assert reflex.decide(_state(), risk=0.0, policy_actions=[]) is None


def test_learned_only_mode_never_fires():
    reflex = OrientingReflex(
        config=OrientingReflexConfig(mode="learned-only"), action_registry=_REGISTRY,
    )
    assert reflex.decide(_state(), risk=0.0, policy_actions=[]) is None


def test_high_risk_vetoes_the_reflex():
    reflex = OrientingReflex(
        config=OrientingReflexConfig(risk_veto_threshold=0.7), action_registry=_REGISTRY,
    )
    assert reflex.decide(_state(), risk=0.9, policy_actions=[]) is None
    # Just under threshold: fires normally.
    assert reflex.decide(_state(), risk=0.5, policy_actions=[]) is not None


def test_world_changing_policy_action_blocks_the_reflex():
    """The issue's own example: fleeing (movement, world-changing) must
    never be suppressed by the reflex."""
    reflex = OrientingReflex(action_registry=_REGISTRY)
    flee = [Action("MOVE_FORWARD")]
    assert reflex.decide(_state(), risk=0.0, policy_actions=flee) is None


def test_information_gathering_or_null_policy_action_does_not_block():
    reflex = OrientingReflex(action_registry=_REGISTRY)
    assert reflex.decide(_state(), risk=0.0, policy_actions=[]) is not None
    assert reflex.decide(_state(), risk=0.0, policy_actions=[NULL_ACTION]) is not None
    assert reflex.decide(_state(), risk=0.0, policy_actions=[Action("LOOK_LEFT")]) is not None


def test_undeclared_action_is_conservatively_treated_as_world_changing():
    reflex = OrientingReflex(action_registry=_REGISTRY)
    assert reflex.decide(_state(), risk=0.0, policy_actions=[Action("SOME_UNKNOWN_VERB")]) is None


# ------------------------------------------------------------- bounded hold


def test_hold_lasts_exactly_hold_ticks_then_reconsiders():
    reflex = OrientingReflex(
        config=OrientingReflexConfig(hold_ticks=3), action_registry=_REGISTRY,
    )
    first = reflex.decide(_state(bearing_deg=30.0), risk=0.0, policy_actions=[])
    assert first is not None and first.ticks_remaining == 2

    # Even though bottom_up_capture is False on these ticks (no new spike),
    # the hold keeps returning the same action.
    held_state = _state(bearing_deg=30.0, bottom_up_capture=False)
    second = reflex.decide(held_state, risk=0.0, policy_actions=[])
    assert second is not None and second.action == first.action and second.ticks_remaining == 1
    third = reflex.decide(held_state, risk=0.0, policy_actions=[])
    assert third is not None and third.ticks_remaining == 0

    # Hold exhausted: with no new bottom-up capture, the reflex goes quiet.
    fourth = reflex.decide(held_state, risk=0.0, policy_actions=[])
    assert fourth is None


def test_hold_is_cut_short_by_a_world_changing_policy_action():
    reflex = OrientingReflex(
        config=OrientingReflexConfig(hold_ticks=5), action_registry=_REGISTRY,
    )
    reflex.decide(_state(bearing_deg=30.0), risk=0.0, policy_actions=[])
    held_state = _state(bearing_deg=30.0, bottom_up_capture=False)
    interrupted = reflex.decide(held_state, risk=0.0, policy_actions=[Action("MOVE_FORWARD")])
    assert interrupted is None
    # The hold was reset, not merely skipped one tick.
    resumed = reflex.decide(_state(bearing_deg=30.0), risk=0.0, policy_actions=[])
    assert resumed is not None and resumed.ticks_remaining == reflex.config.hold_ticks - 1


def test_reset_clears_an_in_progress_hold():
    reflex = OrientingReflex(
        config=OrientingReflexConfig(hold_ticks=5), action_registry=_REGISTRY,
    )
    reflex.decide(_state(bearing_deg=30.0), risk=0.0, policy_actions=[])
    reflex.reset()
    held_state = _state(bearing_deg=30.0, bottom_up_capture=False)
    assert reflex.decide(held_state, risk=0.0, policy_actions=[]) is None


# --------------------------------------------------- simulated scenario (acceptance)


class _PeripheralStimulusProgram(Program):
    """Minimal legacy pull-style Program (`ObservationStreamShim`-wrapped):
    publishes a flat, directionless `stimulus` observation until
    `spike_tick`, then a strong, localized one that stays -- a deterministic
    stand-in for "an entity/block change appears in the periphery" that
    isn't subject to Minecraft's stochastic mob-spawn timing or its other
    streams' salience noise, so the reflex's end-to-end behavior in the real
    `CognitiveRuntime` loop can be asserted exactly."""

    def __init__(self, spike_tick: int = 5, bearing_deg: float = 40.0):
        self._tick = 0
        self.spike_tick = spike_tick
        self.bearing_deg = bearing_deg

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    def observe(self) -> Observation:
        if self._tick < self.spike_tick:
            stimulus = {"value": 0.0, "direction": None}
        else:
            stimulus = {"value": 1.0, "direction": {"bearing_deg": self.bearing_deg}}
        return Observation(
            timestamp=self._tick * 0.05, tick=self._tick,
            data={"stimulus": stimulus}, frame=[[0]],
        )

    def act(self, action: Action) -> ActionResult:
        self._tick += 1
        return ActionResult(ok=True)

    def reward(self) -> RewardSignal:
        return RewardSignal.from_components({})

    def is_complete(self) -> bool:
        return False

    def reset(self, seed: Optional[int] = None) -> None:
        self._tick = 0

    def snapshot(self) -> str:
        return str(self._tick)

    def restore(self, snapshot_id: str) -> None:
        self._tick = int(snapshot_id)

    def metadata(self) -> ProgramMetadata:
        return ProgramMetadata(name="peripheral-stimulus-test", version="0",
                                observation_keys=["stimulus"])


#: `observation.stimulus`'s `{"value", "direction"}` payload needs an
#: `agent_input` + `localization_hint=True` declaration to be visible to the
#: attention controller and the reflex -- `DEFAULT_STREAM_REGISTRY` has no
#: pattern for the shim's generic `observation.*` prefix, so the test adds
#: one exactly like a Program's own registry would (issue #32/#59/#60).
#: `vision.frame.grid` also needs overriding to a legend-free stub: the
#: shim always publishes that stream, but `DEFAULT_STREAM_REGISTRY`'s
#: `GridVisionEncoder` requires a `StreamSpec.legend` the shim's generic
#: spec doesn't carry. Both overrides go first (`StreamRegistry` priority is
#: first-match-wins), with `DEFAULT_STREAM_REGISTRY` extended in afterward.
_STIMULUS_STREAM_REGISTRY = StreamRegistry(
    [
        StreamDeclaration(
            "observation.stimulus",
            ScalarEncoder,
            classification="agent_input",
            attention=AttentionMetadata(modality="world", localization_hint=True),
        ),
        StreamDeclaration(
            # A real (non-`None`) `encoder_factory` is required here: an
            # `encoder_factory=None` declaration is skipped entirely by
            # `to_encoder_registry` (it never shadows a later pattern), so
            # only a harmless placeholder encoder actually keeps
            # `DEFAULT_STREAM_REGISTRY`'s legend-requiring `GridVisionEncoder`
            # from being reached for this legend-free spec. `ScalarEncoder`
            # finds no numeric leaf in the shim's list-of-lists frame payload
            # and contributes nothing to fusion -- irrelevant to this test.
            "vision.frame.grid",
            ScalarEncoder,
            classification="aux_debug",
        ),
    ]
).extend(DEFAULT_STREAM_REGISTRY.declarations)


def _run_stimulus_scenario(tmp_path, session_id: str, reflex_mode: str, spike_tick: int = 5):
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=0,
        max_ticks_per_episode=20,
        record_dir=str(tmp_path),
        session_id=session_id,
        attention_mode="budgeted",
        reflex_mode=reflex_mode,
    )
    program = ObservationStreamShim(_PeripheralStimulusProgram(spike_tick=spike_tick, bearing_deg=40.0))
    summary = CognitiveRuntime(
        program=program,
        policy=NullPolicy(),
        config=runtime_config,
        stream_registry=_STIMULUS_STREAM_REGISTRY,
        encoders=_STIMULUS_STREAM_REGISTRY.to_encoder_registry(),
        action_registry=_REGISTRY,
    ).run()[0]
    return summary, str(tmp_path / session_id)


def test_simulated_scenario_agent_turns_toward_a_newly_appearing_stimulus(tmp_path):
    """Acceptance criterion: a salient, localizable stimulus appears in the
    periphery -> the agent turns toward it within N ticks; the decision
    record attributes the action to the reflex and names the triggering
    stream."""
    summary, session_dir = _run_stimulus_scenario(tmp_path, "reflex-session", reflex_mode="on")
    assert summary.reflex_activations > 0

    activations = [
        (decision["tick_index"], decision["reflex"], motor)
        for decision, _sensory, motor in iter_cognitive_ticks(session_dir, "episode_00000")
        if decision.get("reflex")
    ]
    assert activations, "expected at least one orienting reflex activation"

    tick, reflex, motor = activations[0]
    assert tick <= 5 + 3  # within N ticks of the spike (spike_tick + a couple ticks' slack)
    assert reflex["reason"] == "orienting_reflex"
    assert reflex["stimulus_stream"] == "observation.stimulus"
    assert reflex["direction"] == {"bearing_deg": 40.0, "region": None}
    # A positive bearing turns right; the reflex's chosen action is what
    # actually reached the motor bus.
    motor_actions = {m["payload"]["action"] for m in motor if "payload" in m}
    assert motor_actions == {"LOOK_RIGHT"}


def test_ablation_reflex_off_never_activates(tmp_path):
    summary, session_dir = _run_stimulus_scenario(tmp_path, "reflex-off-session", reflex_mode="off")
    assert summary.reflex_mode == "off"
    assert summary.reflex_activations == 0
    for decision, _sensory, _motor in iter_cognitive_ticks(session_dir, "episode_00000"):
        assert decision.get("reflex") is None


def test_ablation_reflex_learned_only_never_activates(tmp_path):
    summary, session_dir = _run_stimulus_scenario(
        tmp_path, "reflex-learned-only-session", reflex_mode="learned-only",
    )
    assert summary.reflex_activations == 0
    for decision, _sensory, _motor in iter_cognitive_ticks(session_dir, "episode_00000"):
        assert decision.get("reflex") is None
