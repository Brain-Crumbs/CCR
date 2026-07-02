"""Baseline policy behavior tests."""

from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.perception import StructuredPerception
from cognitive_runtime.policies import NullPolicy, RandomPolicy, ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE

_PERCEPTION = StructuredPerception()


def _state(data):
    return _PERCEPTION.encode(Observation(timestamp=0.0, tick=1, data=data))


BASE = {
    "health": 20.0, "hunger": 20.0, "position": {"x": 10.0, "z": 10.0},
    "front_block": "grass", "hotbar": [None] * 9, "selected_slot": 0,
    "mobs": [], "in_water": False,
}


def _obs(**overrides):
    data = dict(BASE)
    data.update(overrides)
    return data


def test_null_policy_always_null():
    policy = NullPolicy()
    action = policy.decide(_state(_obs()), Memory(), None)
    assert action.is_null


def test_random_policy_samples_action_space_deterministically():
    a = RandomPolicy(ACTION_SPACE, seed=4)
    b = RandomPolicy(ACTION_SPACE, seed=4)
    memory = Memory()
    seq_a = [a.decide(_state(_obs()), memory, None) for _ in range(50)]
    seq_b = [b.decide(_state(_obs()), memory, None) for _ in range(50)]
    assert seq_a == seq_b
    assert all(action in ACTION_SPACE for action in seq_a)
    assert len({action.key() for action in seq_a}) > 5


def test_scripted_policy_eats_when_hungry():
    policy = ScriptedSurvivalPolicy(seed=0)
    hotbar = ["berries"] + [None] * 8
    action = policy.decide(
        _state(_obs(hunger=8.0, hotbar=hotbar, selected_slot=3)), Memory(), None
    )
    assert action.key() == "SELECT_HOTBAR_SLOT:slot=0"
    action = policy.decide(
        _state(_obs(hunger=8.0, hotbar=hotbar, selected_slot=0)), Memory(), None
    )
    assert action.name == "USE"


def test_scripted_policy_fights_when_threatened():
    policy = ScriptedSurvivalPolicy(seed=0)
    obs = _obs(mobs=[{"distance": 1.5, "angle": 5.0}])
    action = policy.decide(_state(obs), Memory(), None)
    assert action.name == "ATTACK"
    obs = _obs(mobs=[{"distance": 4.0, "angle": 90.0}])
    action = policy.decide(_state(obs), Memory(), None)
    assert action.name == "LOOK_RIGHT"


def test_scripted_policy_flees_when_weak():
    policy = ScriptedSurvivalPolicy(seed=0)
    obs = _obs(health=4.0, mobs=[{"distance": 3.0, "angle": 170.0}])
    action = policy.decide(_state(obs), Memory(), None)
    assert action.name == "SPRINT"


def test_scripted_policy_harvests_food_in_front():
    policy = ScriptedSurvivalPolicy(seed=0)
    action = policy.decide(_state(_obs(front_block="berry_bush")), Memory(), None)
    assert action.name == "ATTACK"
