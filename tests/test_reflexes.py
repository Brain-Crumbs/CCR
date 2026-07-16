from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.attention import AttentionReason, AttentionSignal, AttentionState, StimulusDirection
from motor.reflexes import (
    CaregiverChannel,
    CaregiverOverride,
    ReflexConfig,
    ReflexStack,
    Stimulus,
    default_reflex_genome,
    stimulus_from_attention,
    stimulus_from_hazard,
    stimulus_from_threat,
)


def stack():
    return ReflexStack([
        ReflexConfig("orient", "salience", Action("LOOK"), threshold=.5, priority=1),
        ReflexConfig("withdraw", "threat", Action("BACK"), threshold=.7, priority=10),
    ])


def test_world_stimulus_fires_organism_configured_reflex():
    decision = stack().decide(Action("FORWARD"), [Stimulus("threat", .9, "crafter")])
    assert decision.actuated == Action("BACK")
    assert decision.reflex.name == "withdraw"
    assert "crafter:threat" in decision.reflex.reason


def test_precedence_is_caregiver_then_priority_reflex_then_voluntary():
    reflexes = stack()
    stimuli = [Stimulus("salience", 1), Stimulus("threat", 1)]
    assert reflexes.decide(Action("FORWARD"), stimuli).actuated == Action("BACK")
    assert reflexes.decide(Action("FORWARD"), stimuli,
                           CaregiverOverride(Action("GUIDED"))).actuated == Action("GUIDED")
    assert reflexes.decide(Action("FORWARD")).actuated == Action("FORWARD")


def test_one_shot_stimuli_iterable_considers_every_reflex():
    stimuli = (stimulus for stimulus in [
        Stimulus("salience", 1),
        Stimulus("threat", 1),
    ])

    decision = stack().decide(Action("FORWARD"), stimuli)

    assert decision.reflex.name == "withdraw"
    assert decision.actuated == Action("BACK")


def test_null_remains_an_explicit_recorded_voluntary_choice():
    decision = stack().decide(NULL_ACTION)
    assert decision.voluntary == NULL_ACTION
    assert decision.to_dict()["voluntary"] == "NULL"
    assert decision.actuated == NULL_ACTION


def test_reflex_activation_rate_is_exposed_as_session_metric():
    reflexes = stack()
    reflexes.decide(Action("GO"), [Stimulus("threat", 1)])
    reflexes.decide(Action("GO"))
    assert reflexes.metrics()["motor.reflex_activation_rate"] == .5


# ------------------------------------------------------- migrated stimuli (issue #102)


def _attention_state(bearing_deg=30.0, bottom_up_capture=True, focus_stream="stimulus"):
    signal = AttentionSignal(
        stream_id=focus_stream, novelty=1.0, prediction_error=0.0, uncertainty=None,
        reward_relevance=0.0, risk=0.0, recency=1.0, boredom=0.0, compute_cost=0.0,
        direction=StimulusDirection(bearing_deg=bearing_deg) if bearing_deg is not None else None,
    )
    return AttentionState(
        tick_index=0, mode="budgeted", weights={focus_stream: 1.0},
        selected_streams=(focus_stream,), focus_stream=focus_stream,
        budget_used=1.0, budget_total=4.0,
        reasons={focus_stream: AttentionReason(signal=signal, score=1.0, components={})},
        bottom_up_capture=bottom_up_capture,
    )


def test_stimulus_from_attention_migrates_orienting_reflex_capture():
    right = stimulus_from_attention(_attention_state(bearing_deg=30.0))
    assert right.kind == "salience-right"
    assert right.intensity == 30.0

    left = stimulus_from_attention(_attention_state(bearing_deg=-10.0))
    assert left.kind == "salience-left"
    assert left.intensity == 10.0


def test_stimulus_from_attention_is_none_without_a_localized_capture():
    assert stimulus_from_attention(_attention_state(bottom_up_capture=False)) is None
    assert stimulus_from_attention(_attention_state(bearing_deg=None)) is None


def test_stimulus_from_threat_wraps_the_amygdala_adrenaline_level():
    stimulus = stimulus_from_threat(0.8)
    assert stimulus.kind == "threat"
    assert stimulus.intensity == 0.8
    assert stimulus.source == "amygdala"


def test_stimulus_from_hazard_is_none_when_inactive():
    assert stimulus_from_hazard(False, source="minecraft.body.in_water") is None
    hazard = stimulus_from_hazard(True, source="minecraft.body.in_water")
    assert hazard.kind == "hazard"
    assert hazard.source == "minecraft.body.in_water"


def test_default_reflex_genome_migrates_orienting_and_withdrawal_with_correct_precedence():
    genome = default_reflex_genome(
        withdraw_action=Action("SPRINT"),
        orient_left_action=Action("LOOK_LEFT"),
        orient_right_action=Action("LOOK_RIGHT"),
        hazard_action=Action("MOVE_BACKWARD"),
    )
    reflexes = ReflexStack(genome)

    # Salience alone: orient toward it.
    attention = _attention_state(bearing_deg=30.0)
    decision = reflexes.decide(Action("MOVE_FORWARD"), [stimulus_from_attention(attention)])
    assert decision.actuated == Action("LOOK_RIGHT")

    # A locomotion+threat scenario: the amygdala's threat reading fires
    # `withdraw`, overriding both voluntary output and a simultaneous
    # orienting capture (survival-critical response is never suppressed).
    decision = reflexes.decide(
        Action("MOVE_FORWARD"),
        [stimulus_from_attention(attention), stimulus_from_threat(0.9)],
    )
    assert decision.actuated == Action("SPRINT")
    assert decision.reflex.name == "withdraw"

    # A migrated hazard (water escape) outranks orienting but not withdrawal.
    decision = reflexes.decide(
        Action("MOVE_FORWARD"),
        [stimulus_from_attention(attention),
         stimulus_from_hazard(True, source="minecraft.body.in_water")],
    )
    assert decision.actuated == Action("MOVE_BACKWARD")
    assert decision.reflex.name == "hazard-escape"

    # Below the bearing deadzone: no orienting reflex fires.
    decision = reflexes.decide(Action("MOVE_FORWARD"), [stimulus_from_attention(_attention_state(bearing_deg=5.0))])
    assert decision.actuated == Action("MOVE_FORWARD")


def test_caregiver_channel_injects_at_the_top_of_precedence():
    channel = CaregiverChannel()
    reflexes = ReflexStack(default_reflex_genome(
        withdraw_action=Action("SPRINT"),
        orient_left_action=Action("LOOK_LEFT"),
        orient_right_action=Action("LOOK_RIGHT"),
    ))

    # No injection pending: the reflex stack behaves as if there were no
    # caregiver at all.
    decision = reflexes.decide(Action("MOVE_FORWARD"), [stimulus_from_threat(0.9)],
                               channel.drain())
    assert decision.actuated == Action("SPRINT")

    # A babbling-stage hook injects a guided command -- it supersedes both
    # the pending threat reflex and voluntary output.
    channel.inject(Action("MOVE_LEFT"), reason="babbling")
    decision = reflexes.decide(Action("MOVE_FORWARD"), [stimulus_from_threat(0.9)],
                               channel.drain())
    assert decision.actuated == Action("MOVE_LEFT")
    assert decision.caregiver_override.reason == "babbling"

    # `drain` is one-shot: the next tick has nothing pending again.
    decision = reflexes.decide(Action("MOVE_FORWARD"), [], channel.drain())
    assert decision.caregiver_override is None
    assert decision.actuated == Action("MOVE_FORWARD")
