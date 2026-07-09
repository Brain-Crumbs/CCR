"""Baseline policy behavior tests."""

from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.perception import State
from cognitive_runtime.policies import NullPolicy, RandomPolicy, ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE


def _state(data):
    # Policies read state.observation.data (the stream-derived view the loop
    # builds); State.features stays empty, exactly as in the runtime loop.
    return State(observation=Observation(timestamp=0.0, tick=1, data=data))


# Stream-keyed observation data, as the loop's stream-derived view produces.
BASE = {
    "body.health": 20.0, "body.hunger": 20.0,
    "spatial.position": {"x": 10.0, "z": 10.0},
    "world.front_block": "grass",
    "body.hotbar": {"slots": [None] * 9, "selected": 0},
    "vision.entities": [], "body.in_water": False,
}


def _obs(**overrides):
    """Build stream-keyed data; flat aliases map onto their stream ids."""
    data = dict(BASE)
    aliases = {
        "health": "body.health", "hunger": "body.hunger",
        "front_block": "world.front_block", "mobs": "vision.entities",
        "in_water": "body.in_water", "position": "spatial.position",
    }
    hotbar = dict(data["body.hotbar"])
    for key, value in overrides.items():
        if key == "hotbar":
            hotbar["slots"] = value
        elif key == "selected_slot":
            hotbar["selected"] = value
        else:
            data[aliases.get(key, key)] = value
    data["body.hotbar"] = hotbar
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
