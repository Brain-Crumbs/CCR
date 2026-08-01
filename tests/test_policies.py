"""Baseline policy behavior tests."""

import pytest

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.perception import State
from cognitive_runtime.policies import (
    ActionBurstPolicy,
    NullPolicy,
    RandomPolicy,
    ScriptedSurvivalPolicy,
    TerminalRecoveryActionBurstPolicy,
)
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


_BABBLING_ACTIONS = [
    Action("MOVE_UP"), Action("MOVE_DOWN"), Action("MOVE_LEFT"), Action("MOVE_RIGHT"), Action("NULL"),
]


def test_action_burst_policy_is_deterministic_given_seed():
    a = ActionBurstPolicy(_BABBLING_ACTIONS, seed=7)
    b = ActionBurstPolicy(_BABBLING_ACTIONS, seed=7)
    memory = Memory()
    seq_a = [a.decide(_state(_obs()), memory, None) for _ in range(200)]
    seq_b = [b.decide(_state(_obs()), memory, None) for _ in range(200)]
    assert seq_a == seq_b


def test_action_burst_policy_only_samples_its_action_subset():
    policy = ActionBurstPolicy(_BABBLING_ACTIONS, seed=1)
    memory = Memory()
    seq = [policy.decide(_state(_obs()), memory, None) for _ in range(200)]
    assert set(seq) <= set(_BABBLING_ACTIONS)
    assert len(set(seq)) > 1


def test_action_burst_policy_holds_each_action_within_its_burst_bounds():
    """Sampled burst lengths (``rng.randint(min, max)``) must land in
    [1, 4] -- checked by spying on the burst-length draw itself, not by
    inferring runs from the emitted action sequence: two consecutive bursts
    can coincidentally sample the same action back to back (5 actions, so a
    repeat has ~20% odds per resample), which would otherwise read as one
    longer run."""
    policy = ActionBurstPolicy(_BABBLING_ACTIONS, min_burst_ticks=1, max_burst_ticks=4, seed=2)
    drawn_lengths = []
    original_randint = policy.rng.randint

    def spy_randint(a, b):
        length = original_randint(a, b)
        drawn_lengths.append(length)
        return length

    policy.rng.randint = spy_randint
    memory = Memory()
    for _ in range(500):
        policy.decide(_state(_obs()), memory, None)

    assert drawn_lengths
    assert min(drawn_lengths) >= 1
    assert max(drawn_lengths) <= 4
    # Every burst length in [1, 4] should be observed across a long sample.
    assert set(drawn_lengths) == {1, 2, 3, 4}


def test_action_burst_policy_reset_restarts_the_same_deterministic_sequence():
    policy = ActionBurstPolicy(_BABBLING_ACTIONS, seed=9)
    memory = Memory()
    first = [policy.decide(_state(_obs()), memory, None) for _ in range(50)]
    policy.reset()
    second = [policy.decide(_state(_obs()), memory, None) for _ in range(50)]
    assert first == second


def test_action_burst_policy_rejects_empty_action_space():
    with pytest.raises(ValueError):
        ActionBurstPolicy([], seed=0)


def test_action_burst_policy_rejects_invalid_burst_bounds():
    with pytest.raises(ValueError):
        ActionBurstPolicy(_BABBLING_ACTIONS, min_burst_ticks=0, seed=0)
    with pytest.raises(ValueError):
        ActionBurstPolicy(_BABBLING_ACTIONS, min_burst_ticks=4, max_burst_ticks=1, seed=0)


def test_terminal_recovery_actions_are_all_applied_despite_motor_delay():
    terminal_actions = tuple(Action(name) for name in (
        "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN",
    ))
    policy = TerminalRecoveryActionBurstPolicy(
        _BABBLING_ACTIONS,
        episode_ticks=7,
        terminal_actions=terminal_actions,
        seed=3,
    )
    memory = Memory()
    emissions = [policy.emit(_state(_obs()), memory, None) for _ in range(7)]

    # The runtime applies the preceding emission before each policy decision.
    applied = [[]] + emissions[:-1]
    assert applied[-4:] == [[action] for action in terminal_actions]
    assert emissions[-1] == []


def test_terminal_recovery_requires_a_motor_delay_tick():
    with pytest.raises(ValueError, match="motor delay"):
        TerminalRecoveryActionBurstPolicy(
            _BABBLING_ACTIONS,
            episode_ticks=4,
            terminal_actions=_BABBLING_ACTIONS[:4],
        )


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
