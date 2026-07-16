from cognitive_runtime.core.action import Action
from motor.reflexes import CaregiverOverride, ReflexConfig, ReflexStack, Stimulus


def test_every_tick_records_the_whole_motor_stack_and_divergence():
    stack = ReflexStack([ReflexConfig("withdraw", "threat", Action("BACK"))])
    decision = stack.decide(
        Action("FORWARD"), [Stimulus("threat", 1)], CaregiverOverride(Action("STOP"))
    )
    record = decision.to_dict()
    assert set(record) == {"voluntary", "reflex", "caregiver_override", "actuated"}
    assert record["voluntary"] == "FORWARD"
    assert record["reflex"]["action"] == "BACK"
    assert record["caregiver_override"]["action"] == "STOP"
    assert record["actuated"] == "STOP"
    assert decision.diverged
