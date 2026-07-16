from cognitive_runtime.core.action import NULL_ACTION, Action
from motor.reflexes import CaregiverOverride, ReflexConfig, ReflexStack, Stimulus


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
