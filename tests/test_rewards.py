"""Survival reward function tests."""

from cognitive_runtime.core.action import Action, NULL_ACTION
from cognitive_runtime.core.streams.events import StreamEvent
from cognitive_runtime.programs.minecraft.rewards import SurvivalReward, SurvivalRewardConfig

BASE_OBS = {
    "health": 20.0, "hunger": 20.0, "oxygen": 20.0,
    "time_of_day": 100, "day_length": 6000,
    "biome": "plains", "distance_from_spawn": 0.0,
    "nearby_blocks": [["grass", "grass"], ["grass", "grass"]],
    "mobs": [],
}


def _obs(**overrides):
    obs = dict(BASE_OBS)
    obs.update(overrides)
    return obs


def _eval(reward, obs, events=(), action=NULL_ACTION, obs_hash=None):
    _eval.counter = getattr(_eval, "counter", 0) + 1
    return reward.evaluate(obs, list(events), action, obs_hash or f"h{_eval.counter}")


def test_tick_alive_and_death():
    reward = SurvivalReward()
    alive = _eval(reward, _obs())
    assert alive.components.get("tick_alive") == 0.01
    died = _eval(reward, _obs(health=0.0), events=["damage:zombie", "died"])
    assert died.components["death"] == -10.0
    assert died.components["damage_taken"] == -0.5


def test_hunger_loss_and_critical_transitions():
    reward = SurvivalReward()
    _eval(reward, _obs(hunger=10.0))
    signal = _eval(reward, _obs(hunger=8.0))
    assert abs(signal.components["hunger_decrease"] - (-0.5)) < 1e-9  # 2 points
    signal = _eval(reward, _obs(hunger=3.0, health=3.0))
    assert signal.components["critical_hunger"] == -1.0
    assert signal.components["critical_health"] == -1.0
    # No re-trigger while still critical.
    signal = _eval(reward, _obs(hunger=2.0, health=2.0))
    assert "critical_hunger" not in signal.components
    assert "critical_health" not in signal.components


def test_exploration_rewards_are_capped():
    cfg = SurvivalRewardConfig(new_block_cap=0.2)
    reward = SurvivalReward(cfg)
    first = _eval(reward, _obs())  # sees "grass": +0.1
    assert abs(first.components["new_block_type"] - 0.1) < 1e-9
    blocks = [["water", "sand"], ["stone", "tree"]]
    second = _eval(reward, _obs(nearby_blocks=blocks))  # 4 new but capped at 0.2 total
    assert abs(second.components["new_block_type"] - 0.1) < 1e-9
    third = _eval(reward, _obs(nearby_blocks=[["coal_ore", "dirt"], ["dirt", "dirt"]]))
    assert "new_block_type" not in third.components


def test_item_and_firsts_rewards():
    reward = SurvivalReward()
    signal = _eval(reward, _obs(), events=["new_item:berries", "acquired_food"])
    assert signal.components["new_item"] == 0.5
    assert signal.components["first_food"] == 1.0
    signal = _eval(reward, _obs(), events=["placed_block"])
    assert signal.components["first_block_placed"] == 1.0
    signal = _eval(reward, _obs(), events=["placed_block"])
    assert "first_block_placed" not in signal.components


def test_stream_rewards_activate_first_tool_and_light_source():
    reward = SurvivalReward()
    reward.prime_stream_state([
        StreamEvent("body.health", "body", 0.0, 0, 20.0),
        StreamEvent("body.hunger", "body", 0.0, 0, 20.0),
        StreamEvent("world.nearby_blocks", "world", 0.0, 0, [["grass"]]),
        StreamEvent("world.biome", "world", 0.0, 0, "plains"),
        StreamEvent("spatial.position", "spatial", 0.0, 0, {"x": 0.0, "y": 64.0, "z": 0.0}),
    ])
    signal = reward.evaluate_stream_window([
        StreamEvent("event.item_collected", "event", 1.0, 0, {"item": "stone_pickaxe"}),
        StreamEvent("event.created_light_source", "event", 1.0, 0, {}),
    ], NULL_ACTION)
    assert signal.components["first_tool"] == 1.0
    assert signal.components["light_source"] == 1.0


def test_new_chunk_exploration_reward_is_capped():
    """Issue #30: new-chunk/new-cell visitation, distinct from new_block_type
    (which rewards new terrain, not new area)."""
    cfg = SurvivalRewardConfig(new_chunk=0.1, new_chunk_cap=0.15, chunk_size=8.0)
    reward = SurvivalReward(cfg)
    first = _eval(reward, _obs(position={"x": 0.0, "y": 64.0, "z": 0.0}))
    assert abs(first.components["new_chunk"] - 0.1) < 1e-9
    # Same chunk again: no further bonus.
    second = _eval(reward, _obs(position={"x": 1.0, "y": 64.0, "z": 1.0}))
    assert "new_chunk" not in second.components
    # A new chunk, but capped.
    third = _eval(reward, _obs(position={"x": 20.0, "y": 64.0, "z": 0.0}))
    assert abs(third.components["new_chunk"] - 0.05) < 1e-9
    fourth = _eval(reward, _obs(position={"x": 40.0, "y": 64.0, "z": 0.0}))
    assert "new_chunk" not in fourth.components


def test_tool_used_reward_activates_once_per_tool_type():
    """Issue #30: tool use, distinct from first_tool (which rewards merely
    acquiring one)."""
    cfg = SurvivalRewardConfig(tool_used_item=0.3, tool_used_cap=0.4)
    reward = SurvivalReward(cfg)
    reward.prime_stream_state([
        StreamEvent("body.health", "body", 0.0, 0, 20.0),
        StreamEvent("body.hunger", "body", 0.0, 0, 20.0),
        StreamEvent("spatial.position", "spatial", 0.0, 0, {"x": 0.0, "y": 64.0, "z": 0.0}),
    ])
    signal = reward.evaluate_stream_window(
        [StreamEvent("event.tool_used", "event", 1.0, 0, {"item": "wooden_pickaxe"})],
        NULL_ACTION,
    )
    assert abs(signal.components["tool_used"] - 0.3) < 1e-9
    # Same tool again: no further bonus.
    signal = reward.evaluate_stream_window(
        [StreamEvent("event.tool_used", "event", 2.0, 1, {"item": "wooden_pickaxe"})],
        NULL_ACTION,
    )
    assert "tool_used" not in signal.components
    # A different tool type, but capped.
    signal = reward.evaluate_stream_window(
        [StreamEvent("event.tool_used", "event", 3.0, 2, {"item": "stone_sword"})],
        NULL_ACTION,
    )
    assert abs(signal.components["tool_used"] - 0.1) < 1e-9


def test_craft_progress_reward_activates_once_per_recipe():
    """Issue #30: crafting progress, keyed on distinct recipe ids from
    event.crafted (issue #40's structured crafting event)."""
    cfg = SurvivalRewardConfig(craft_progress=0.5, craft_progress_cap=0.75)
    reward = SurvivalReward(cfg)
    reward.prime_stream_state([
        StreamEvent("body.health", "body", 0.0, 0, 20.0),
        StreamEvent("body.hunger", "body", 0.0, 0, 20.0),
        StreamEvent("spatial.position", "spatial", 0.0, 0, {"x": 0.0, "y": 64.0, "z": 0.0}),
    ])
    signal = reward.evaluate_stream_window(
        [StreamEvent("event.crafted", "event", 1.0, 0,
                     {"recipe": "log_to_planks", "inputs": {"log": 1}, "outputs": {"planks": 4}})],
        NULL_ACTION,
    )
    assert abs(signal.components["craft_progress"] - 0.5) < 1e-9
    # Same recipe again: no further bonus.
    signal = reward.evaluate_stream_window(
        [StreamEvent("event.crafted", "event", 2.0, 1,
                     {"recipe": "log_to_planks", "inputs": {"log": 1}, "outputs": {"planks": 4}})],
        NULL_ACTION,
    )
    assert "craft_progress" not in signal.components
    # A different recipe, but capped.
    signal = reward.evaluate_stream_window(
        [StreamEvent("event.crafted", "event", 3.0, 2,
                     {"recipe": "smelt_torch", "inputs": {"coal": 1}, "outputs": {"torch": 4}})],
        NULL_ACTION,
    )
    assert abs(signal.components["craft_progress"] - 0.25) < 1e-9


def test_shelter_and_night_once_per_episode():
    reward = SurvivalReward()
    signal = _eval(reward, _obs(), events=["entered_shelter", "survived_night"])
    assert signal.components["shelter"] == 1.0
    assert signal.components["survived_night"] == 1.0
    signal = _eval(reward, _obs(), events=["entered_shelter", "survived_night"])
    assert "shelter" not in signal.components
    assert "survived_night" not in signal.components


def test_anti_stagnation_penalties():
    cfg = SurvivalRewardConfig(repeated_action_threshold=3, idle_threshold=3,
                               spinning_window=4, no_novelty_ticks=5)
    reward = SurvivalReward(cfg)
    move = Action("MOVE_FORWARD")
    for i in range(3):
        signal = _eval(reward, _obs(), action=move)
        assert "repeated_action" not in signal.components
    signal = _eval(reward, _obs(), action=move)
    assert signal.components["repeated_action"] == -0.01

    reward = SurvivalReward(cfg)
    for i in range(3):
        _eval(reward, _obs(), action=NULL_ACTION)
    signal = _eval(reward, _obs(), action=NULL_ACTION)
    assert signal.components["idle"] == -0.05
    # Idling is contextual: no penalty while threatened.
    reward = SurvivalReward(cfg)
    for i in range(6):
        signal = _eval(reward, _obs(mobs=[{"distance": 2.0, "angle": 0.0}]), action=NULL_ACTION)
    assert "idle" not in signal.components

    reward = SurvivalReward(cfg)
    for i in range(3):
        signal = _eval(reward, _obs(), action=Action("LOOK_LEFT"))
    signal = _eval(reward, _obs(), action=Action("LOOK_LEFT"))
    assert signal.components["spinning"] == -0.1

    reward = SurvivalReward(cfg)
    for i in range(6):
        signal = _eval(reward, _obs(), action=move, obs_hash="same")
    assert signal.components["no_novelty"] == -0.1
