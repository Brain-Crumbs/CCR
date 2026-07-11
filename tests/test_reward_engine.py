"""ProfileRewardEngine tests (issue #41): components driven entirely by a
loaded profile, milestone once-only behavior across interrupt/resume,
intrinsic slots, and normalization."""

import pytest

from cognitive_runtime.core.action import Action, NULL_ACTION
from cognitive_runtime.core.streams.events import StreamEvent
from cognitive_runtime.programs.minecraft.reward_engine import ProfileRewardEngine
from cognitive_runtime.programs.minecraft.reward_profile import (
    default_profile,
    reward_profile_from_dict,
)

BODY_EVENTS = [
    StreamEvent("body.health", "body", 0.0, 0, 20.0),
    StreamEvent("body.hunger", "body", 0.0, 0, 20.0),
    StreamEvent("world.nearby_blocks", "world", 0.0, 0, [["grass"]]),
    StreamEvent("world.biome", "world", 0.0, 0, "plains"),
    StreamEvent("spatial.position", "spatial", 0.0, 0, {"x": 0.0, "y": 64.0, "z": 0.0}),
]


def _profile(tiers, intrinsic=None, normalization=None):
    data = {"name": "t", "tiers": tiers}
    if intrinsic:
        data["intrinsic"] = intrinsic
    if normalization:
        data["normalization"] = normalization
    return reward_profile_from_dict(data)


def _primed_engine(profile):
    engine = ProfileRewardEngine(profile)
    engine.prime_stream_state(BODY_EVENTS)
    return engine


def _tick(engine, events, action=NULL_ACTION):
    return engine.evaluate_stream_window(events, action)


def test_tick_alive_and_death():
    profile = default_profile()
    engine = _primed_engine(profile)
    alive = _tick(engine, [])
    assert alive.components.get("tick_alive") == 0.01
    died = _tick(engine, [
        StreamEvent("event.damage_taken", "event", 1.0, 0, {"reason": "zombie"}),
        StreamEvent("event.died", "event", 1.0, 1, {}),
    ])
    assert died.components["death"] == -10.0
    assert died.components["damage_taken"] == -0.5
    assert "tick_alive" not in died.components


def test_hunger_delta_decrease():
    profile = default_profile()
    engine = _primed_engine(profile)
    _tick(engine, [StreamEvent("body.hunger", "body", 1.0, 0, 10.0)])
    signal = _tick(engine, [StreamEvent("body.hunger", "body", 2.0, 1, 8.0)])
    assert abs(signal.components["hunger_decrease"] - (-0.5)) < 1e-9


def test_capped_novelty_is_capped():
    profile = _profile({
        "capability": {
            "new_block_type": {
                "kind": "capped_novelty", "value": 0.1, "cap": 0.2,
                "params": {"source": "nearby_blocks"},
            },
        },
    })
    engine = _primed_engine(profile)  # grass already seen via priming
    first = _tick(engine, [StreamEvent("world.nearby_blocks", "world", 1.0, 0, [["water", "sand"]])])
    assert abs(first.components["new_block_type"] - 0.2) < 1e-9
    second = _tick(engine, [StreamEvent("world.nearby_blocks", "world", 2.0, 1, [["stone", "tree"]])])
    assert "new_block_type" not in second.components


def test_once_predicate_first_tool_and_once_event_light_source():
    profile = default_profile()
    engine = _primed_engine(profile)
    signal = engine.evaluate_stream_window([
        StreamEvent("event.item_collected", "event", 1.0, 0, {"item": "stone_pickaxe"}),
        StreamEvent("event.created_light_source", "event", 1.0, 0, {}),
    ], NULL_ACTION)
    assert signal.components["first_tool"] == 1.0
    assert signal.components["light_source"] == 1.0
    # Second life (reset): light_source is scope="life" -> fires again.
    engine.reset()
    engine.prime_stream_state(BODY_EVENTS)
    signal = engine.evaluate_stream_window(
        [StreamEvent("event.created_light_source", "event", 1.0, 0, {})], NULL_ACTION
    )
    assert signal.components["light_source"] == 1.0


def test_decaying_repeat_diminishes_and_caps():
    profile = _profile({
        "capability": {
            "repeated_common_item": {
                "kind": "decaying_repeat", "value": 0.2, "decay": 0.5,
                "decay_floor": 0.01, "cap": 0.35,
                "params": {"source": "event:new_item"},
            },
        },
    })
    engine = _primed_engine(profile)
    ev = lambda item: [StreamEvent("event.item_collected", "event", 1.0, 0, {"item": item})]
    first = _tick(engine, ev("dirt"))
    assert abs(first.components["repeated_common_item"] - 0.2) < 1e-9
    second = _tick(engine, ev("dirt"))
    assert abs(second.components["repeated_common_item"] - 0.1) < 1e-9
    third = _tick(engine, ev("dirt"))
    # capped: only 0.05 of budget remains (0.35 - 0.2 - 0.1)
    assert abs(third.components["repeated_common_item"] - 0.05) < 1e-9
    fourth = _tick(engine, ev("dirt"))
    assert "repeated_common_item" not in fourth.components


def test_milestone_brain_scope_persists_across_reset():
    profile = _profile({
        "quest": {
            "entered_village": {
                "kind": "once_event", "value": 10.0, "scope": "brain",
                "params": {"event": "entered_village"},
            },
        },
    })
    engine = _primed_engine(profile)
    signal = _tick(engine, [StreamEvent("event.entered_shelter", "event", 1.0, 0, {})])
    # not the village event -- no reward yet
    assert "entered_village" not in signal.components

    engine.reset()  # new life; brain-scope state must survive
    engine.prime_stream_state(BODY_EVENTS)

    # Simulate the event via a synthetic semantic event by monkeypatching a
    # translator is unnecessary -- once_event reads ctx.events, and there is
    # no real "entered_village" stream translator yet (issue #41's quest
    # tier is schema-only), so fire it as a raw event through evaluate().
    signal = engine.evaluate(
        {"health": 20.0, "hunger": 20.0}, ["entered_village"], NULL_ACTION, "h1",
    )
    assert signal.components["entered_village"] == 10.0

    state = engine.state_dict()
    assert state["brain_state"]["entered_village"]["_fired"] is True

    # A brand new engine restoring that state must not re-grant it.
    engine2 = ProfileRewardEngine(profile)
    engine2.load_state_dict(state)
    signal2 = engine2.evaluate(
        {"health": 20.0, "hunger": 20.0}, ["entered_village"], NULL_ACTION, "h2",
    )
    assert "entered_village" not in signal2.components


def test_load_state_dict_rejects_mismatched_profile():
    profile_a = _profile({"survival": {"tick_alive": {"kind": "tick", "value": 0.01}}})
    profile_b = _profile({"survival": {"tick_alive": {"kind": "tick", "value": 0.02}}})
    engine_a = ProfileRewardEngine(profile_a)
    state = engine_a.state_dict()
    engine_b = ProfileRewardEngine(profile_b)
    with pytest.raises(ValueError, match="different reward profile"):
        engine_b.load_state_dict(state)


def test_intrinsic_slot_reads_weighted_capped_stream_value():
    profile = _profile(
        tiers={},
        intrinsic={
            "learning_progress": {
                "stream": "internal.learning_progress", "weight": 2.0, "cap": 1.0,
            },
        },
    )
    engine = _primed_engine(profile)
    signal = _tick(engine, [
        StreamEvent("internal.learning_progress", "event", 1.0, 0, {"value": 0.3}),
    ])
    assert abs(signal.components["learning_progress"] - 0.6) < 1e-9
    # Capped: a second large reading only pays out up to the remaining budget.
    signal = _tick(engine, [
        StreamEvent("internal.learning_progress", "event", 2.0, 1, {"value": 0.9}),
    ])
    assert abs(signal.components["learning_progress"] - 0.4) < 1e-9


def test_intrinsic_slot_disabled_never_fires():
    profile = _profile(
        tiers={},
        intrinsic={
            "safe_novelty": {"stream": "internal.safe_novelty", "weight": 1.0, "disabled": True},
        },
    )
    engine = _primed_engine(profile)
    signal = _tick(engine, [
        StreamEvent("internal.safe_novelty", "event", 1.0, 0, {"value": 5.0}),
    ])
    assert "safe_novelty" not in signal.components


def test_malformed_profile_never_reaches_evaluation():
    """Reward profile validation happens at load time (reward_profile.py),
    well before any ProfileRewardEngine is constructed -- this asserts the
    contract the CLI depends on: a bad profile can never turn into an
    in-episode crash because it is rejected before evaluate() is callable."""
    from cognitive_runtime.programs.minecraft.reward_profile import (
        RewardProfileError,
        reward_profile_from_dict,
    )

    with pytest.raises(RewardProfileError):
        reward_profile_from_dict({"name": "bad", "tiers": {"survival": {
            "x": {"kind": "capped_novelty", "value": 1.0, "params": {"source": "nearby_blocks"}},
        }}})


def test_normalization_clips_large_raw_reward():
    profile = _profile(
        tiers={
            "quest": {
                "huge": {"kind": "once_event", "value": 1_000_000.0, "params": {"event": "won"}},
            },
        },
        normalization={"method": "none", "clip": 5.0},
    )
    engine = _primed_engine(profile)
    signal = _tick(engine, [StreamEvent("event.entered_shelter", "event", 1.0, 0, {})])
    assert signal.value == 0.0
    signal = engine.evaluate(
        {"health": 20.0, "hunger": 20.0}, ["won"], NULL_ACTION, "h1",
    )
    assert signal.value == 1_000_000.0  # raw, for logging/dashboards
    assert signal.training_value == 5.0  # clipped, for the optimizer


def test_anti_stagnation_penalties():
    profile = _profile({
        "shaping": {
            "repeated_action": {
                "kind": "streak_penalty", "value": -0.01, "params": {"threshold": 3},
            },
            "idle": {
                "kind": "idle_penalty", "value": -0.05, "params": {"threshold": 3},
            },
            "spinning": {
                "kind": "spinning_penalty", "value": -0.1,
                "params": {"window": 4, "actions": ["LOOK_LEFT", "LOOK_RIGHT"]},
            },
            "no_novelty": {
                "kind": "no_novelty_penalty", "value": -0.1, "params": {"ticks": 5},
            },
        },
    })
    move = Action("MOVE_FORWARD")
    engine = _primed_engine(profile)
    for _ in range(3):
        signal = _tick(engine, [], action=move)
        assert "repeated_action" not in signal.components
    signal = _tick(engine, [], action=move)
    assert signal.components["repeated_action"] == -0.01

    engine = _primed_engine(profile)
    for _ in range(3):
        _tick(engine, [], action=NULL_ACTION)
    signal = _tick(engine, [], action=NULL_ACTION)
    assert signal.components["idle"] == -0.05

    engine = _primed_engine(profile)
    for _ in range(3):
        _tick(engine, [], action=Action("LOOK_LEFT"))
    signal = _tick(engine, [], action=Action("LOOK_LEFT"))
    assert signal.components["spinning"] == -0.1
