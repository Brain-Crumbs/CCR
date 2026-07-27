from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.action_registry import ActionDeclaration, ActionRegistry
from cognitive_runtime.core.attention import AttentionReason, AttentionSignal, AttentionState, StimulusDirection
from motor.reflexes import (
    AttentionStimulusSource,
    CaregiverChannel,
    CaregiverOverride,
    ReflexConfig,
    ReflexStack,
    Stimulus,
    default_reflex_genome,
    eligible_orienting_stimuli,
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


def test_reflex_stack_reset_clears_per_episode_metrics():
    reflexes = stack()
    reflexes.decide(Action("GO"), [Stimulus("threat", 1)])
    reflexes.reset()
    assert reflexes.metrics() == {
        "motor.reflex_activation_rate": 0.0,
        "motor.reflex_activations": 0,
        "motor.ticks": 0,
    }


# --------------------------------------------------------- issue #103: developmental trend


def test_reflex_activation_rate_falls_across_development_on_locomotion_and_threat_scenario():
    import random

    from motor.reflexes import reflex_activation_series

    def session(threat_probability, ticks=200, seed=0):
        reflexes = ReflexStack([
            ReflexConfig("withdraw", "threat", Action("BACK"), threshold=.5, priority=10),
        ])
        rng = random.Random(seed)
        for _ in range(ticks):
            stimuli = [Stimulus("threat", 1.0)] if rng.random() < threat_probability else []
            reflexes.decide(Action("FORWARD"), stimuli)
        return reflexes

    # The cortex "matures" across sessions: it learns to steer clear of the
    # threat zone voluntarily, so fewer ticks ever present a
    # threshold-crossing threat stimulus -- the falling activation-rate
    # curve issue #103 asks the clinic to chart (reflex *integration*,
    # not the withdraw reflex itself changing).
    maturity_threat_probabilities = [0.8, 0.6, 0.4, 0.2, 0.05]
    sessions = [session(p, seed=i) for i, p in enumerate(maturity_threat_probabilities)]

    rates = reflex_activation_series(sessions)

    assert rates == sorted(rates, reverse=True)
    assert rates[0] > rates[-1]
    assert rates[-1] < 0.15


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
    assert stimulus_from_attention(_attention_state(bearing_deg=0.0)) is None


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


# --------------------------------------------------- OrientingReflex parity fixes


def test_bearing_exactly_at_the_deadzone_boundary_does_not_fire():
    genome = default_reflex_genome(
        withdraw_action=Action("SPRINT"),
        orient_left_action=Action("LOOK_LEFT"),
        orient_right_action=Action("LOOK_RIGHT"),
        bearing_deadzone_deg=15.0,
    )
    reflexes = ReflexStack(genome)

    at_boundary = stimulus_from_attention(_attention_state(bearing_deg=15.0))
    decision = reflexes.decide(Action("MOVE_FORWARD"), [at_boundary])
    assert decision.actuated == Action("MOVE_FORWARD")
    assert decision.reflex is None

    just_past = stimulus_from_attention(_attention_state(bearing_deg=15.0001))
    decision = reflexes.decide(Action("MOVE_FORWARD"), [just_past])
    assert decision.actuated == Action("LOOK_RIGHT")


def test_attention_stimulus_source_holds_the_capture_across_ticks():
    source = AttentionStimulusSource(hold_ticks=3)
    capture = _attention_state(bearing_deg=30.0)
    quiet = _attention_state(bottom_up_capture=False)

    first = source.poll(capture)
    assert first.kind == "salience-right"

    # Two more ticks with no fresh capture still hold the same stimulus...
    second = source.poll(quiet)
    third = source.poll(quiet)
    assert second == first
    assert third == first

    # ...and the hold expires on the fourth tick.
    assert source.poll(quiet) is None


def test_attention_stimulus_source_reset_clears_an_in_progress_hold():
    source = AttentionStimulusSource(hold_ticks=3)
    source.poll(_attention_state(bearing_deg=30.0))
    source.reset()
    assert source.poll(_attention_state(bottom_up_capture=False)) is None


_ACTION_REGISTRY = ActionRegistry([
    ActionDeclaration("LOOK_LEFT", world_changing=False, information_gathering=True),
    ActionDeclaration("LOOK_RIGHT", world_changing=False, information_gathering=True),
    ActionDeclaration("MOVE_FORWARD", world_changing=True, information_gathering=True),
    ActionDeclaration("ATTACK", world_changing=True, information_gathering=False),
])


def test_eligible_orienting_stimuli_drops_salience_for_a_survival_critical_voluntary_action():
    stimuli = [stimulus_from_attention(_attention_state(bearing_deg=30.0)), stimulus_from_threat(0.1)]

    # ATTACK is purely world-changing (not information-gathering): orienting
    # must not substitute a look for it.
    filtered = eligible_orienting_stimuli(stimuli, Action("ATTACK"), _ACTION_REGISTRY)
    assert all(not s.kind.startswith("salience") for s in filtered)

    # MOVE_FORWARD is world-changing *and* information-gathering ("walking
    # does both") -- it does not veto orienting.
    filtered = eligible_orienting_stimuli(stimuli, Action("MOVE_FORWARD"), _ACTION_REGISTRY)
    assert any(s.kind.startswith("salience") for s in filtered)


def test_eligible_orienting_stimuli_end_to_end_preserves_the_survival_critical_veto():
    genome = default_reflex_genome(
        withdraw_action=Action("SPRINT"),
        orient_left_action=Action("LOOK_LEFT"),
        orient_right_action=Action("LOOK_RIGHT"),
    )
    reflexes = ReflexStack(genome)
    stimuli = eligible_orienting_stimuli(
        [stimulus_from_attention(_attention_state(bearing_deg=30.0))],
        Action("ATTACK"),
        _ACTION_REGISTRY,
    )
    decision = reflexes.decide(Action("ATTACK"), stimuli)
    assert decision.actuated == Action("ATTACK")
    assert decision.reflex is None
