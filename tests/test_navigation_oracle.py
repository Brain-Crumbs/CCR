"""A* oracle training-only labels and potential-based geodesic reward (epic
#212 §12.4/§12.5, issue #239, gated on MF-F1/#238).

Covers this issue's acceptance criteria:

- oracle labels are present in recordings and absent from the deployed
  sensory workspace -- asserted by a test that enumerates the workspace
  stream keys;
- geodesic distance is used whenever walls are present; a detour that
  increases Cartesian distance while decreasing geodesic distance is
  rewarded positively;
- potential-based shaping is policy-invariant: a round trip returning to the
  start accumulates zero net progress reward;
- the arrival bonus is paid exactly once per episode;
- goal progress, completion, collision and task reward are recorded as
  separate channels;
- the oracle-planner version and its map-cost assumptions are exposed for
  the ``DataContract``;
- reward potential, strength, falloff, discount, success radius and bonus
  are all ``DataContract`` fields.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from cognitive_runtime.programs.crafter.oracle import (
    DEFAULT_MAP_COST_ASSUMPTIONS,
    ORACLE_PLANNER_VERSION,
    OracleLabelWriter,
    OracleMapCostAssumptions,
    astar_path,
    build_walkable_grid,
    compute_legal_action_mask,
)
from cognitive_runtime.programs.crafter.streams import SEMANTIC_LEGEND_NAMES
from cognitive_runtime.training.model_factory.contracts import DataContract
from cognitive_runtime.training.navigation_reward import (
    GoalRewardConfig,
    NavigationRewardTracker,
)

_RAW = {name: raw_id for raw_id, name in SEMANTIC_LEGEND_NAMES.items()}


def _grid(rows) -> np.ndarray:
    """Build a raw-semantic-id grid from a list of material-name rows,
    indexed the same way ``build_walkable_grid``/``astar_path`` are:
    ``grid[x, y]``, so each *row* here is one fixed-y strip across x."""
    ids = [[_RAW[name] for name in row] for row in rows]
    return np.array(ids, dtype=np.uint8).T  # transpose: rows were y-major


# ------------------------------------------------------------- oracle: A*


def test_build_walkable_grid_excludes_walls_and_lava_by_default():
    grid = _grid([
        ["grass", "stone", "water"],
        ["path", "lava", "sand"],
    ])
    walkable = build_walkable_grid(grid)
    assert walkable[0, 0] and walkable[0, 1]  # grass, path
    assert walkable[2, 1]  # sand
    assert not walkable[1, 0]  # stone
    assert not walkable[2, 0]  # water
    assert not walkable[1, 1]  # lava excluded by default (avoid_lava=True)


def test_build_walkable_grid_can_include_lava_when_avoid_lava_is_off():
    grid = _grid([["lava"]])
    assumptions = OracleMapCostAssumptions(avoid_lava=False)
    assert build_walkable_grid(grid, assumptions)[0, 0]


def test_oracle_map_cost_assumptions_rejects_unknown_material():
    with pytest.raises(ValueError):
        OracleMapCostAssumptions(walkable_materials=("grass", "not_a_material"))


def test_oracle_map_cost_assumptions_rejects_non_positive_step_cost():
    with pytest.raises(ValueError):
        OracleMapCostAssumptions(step_cost=0.0)


def test_astar_path_start_equals_goal_is_a_single_cell_path():
    grid = _grid([["grass", "grass"]])
    walkable = build_walkable_grid(grid)
    assert astar_path(walkable, (0, 0), (0, 0), (2, 1)) == [(0, 0)]


def test_astar_path_out_of_bounds_endpoints_are_unreachable():
    grid = _grid([["grass", "grass"]])
    walkable = build_walkable_grid(grid)
    assert astar_path(walkable, (0, 0), (5, 5), (2, 1)) is None


def test_astar_path_start_cell_need_not_itself_be_walkable():
    """The agent's own current cell is a valid path origin even when its
    raw semantic id isn't in the walkable set -- e.g. it always shows the
    ``player`` object id, never the underlying material."""
    grid = _grid([["player", "grass", "grass"]])
    walkable = build_walkable_grid(grid)
    assert not walkable[0, 0]
    path = astar_path(walkable, (0, 0), (2, 0), (3, 1))
    assert path == [(0, 0), (1, 0), (2, 0)]


def test_astar_path_is_unreachable_through_a_sealed_wall():
    grid = _grid([
        ["grass", "stone", "grass"],
        ["stone", "stone", "stone"],
        ["grass", "stone", "grass"],
    ])
    walkable = build_walkable_grid(grid)
    assert astar_path(walkable, (0, 0), (2, 2), (3, 3)) is None


def test_astar_path_routes_around_a_wall_with_one_gap():
    # A vertical wall at x=1 for y in {0,1}, open at y=2.
    grid = _grid([
        ["grass", "stone", "grass"],
        ["grass", "stone", "grass"],
        ["grass", "grass", "grass"],
    ])
    walkable = build_walkable_grid(grid)
    path = astar_path(walkable, (0, 0), (2, 0), (3, 3))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (2, 0)
    # Must detour through the gap at y=2.
    assert (1, 2) in path
    assert len(path) - 1 == 6  # down, down, right, right, up, up


def test_compute_legal_action_mask_reflects_walkability_and_bounds():
    grid = _grid([
        ["stone", "grass", "grass"],
        ["grass", "grass", "grass"],
        ["grass", "grass", "grass"],
    ])
    walkable = build_walkable_grid(grid)
    mask = compute_legal_action_mask(walkable, (0, 1), (3, 3))
    assert mask["MOVE_LEFT"] is False  # x=-1 out of bounds
    assert mask["MOVE_UP"] is False    # (0,0) is stone
    assert mask["MOVE_RIGHT"] is True
    assert mask["MOVE_DOWN"] is True


def test_oracle_label_writer_reports_next_action_and_planner_identity():
    grid = _grid([["grass", "grass", "grass"]])
    writer = OracleLabelWriter()
    label = writer.label(semantic=grid, position=(0, 0), goal_position=(2, 0), world_size=(3, 1))
    assert label.geodesic_distance == 2.0
    assert label.next_action == "MOVE_RIGHT"
    assert label.planner_version == ORACLE_PLANNER_VERSION
    assert label.map_cost_assumptions == DEFAULT_MAP_COST_ASSUMPTIONS.to_contract_dict()
    payload = label.as_payload()
    assert set(payload) == {
        "geodesic_distance", "next_action", "legal_action_mask", "path_progress_delta",
        "replan_required", "planner_version", "map_cost_assumptions",
    }


def test_oracle_label_writer_geodesic_distance_used_over_cartesian_once_walls_matter():
    """The acceptance criterion, stated directly against the planner: a
    detour that *increases* Cartesian distance from the goal must
    *decrease* geodesic distance, once the agent is past the wall's near
    corner."""
    # Vertical wall at x=1, gap only at y=3. Agent starts at (0,0), goal (2,0).
    rows = [["grass", "stone" if y != 3 else "grass", "grass"] for y in range(4)]
    grid = _grid(rows)
    walkable = build_walkable_grid(grid)
    world_size = (3, 4)

    def cartesian(pos, goal=(2, 0)):
        return math.hypot(pos[0] - goal[0], pos[1] - goal[1])

    writer = OracleLabelWriter()
    path = astar_path(walkable, (0, 0), (2, 0), world_size)
    assert path is not None

    distances = [
        writer.label(semantic=grid, position=p, goal_position=(2, 0), world_size=world_size)
        .geodesic_distance
        for p in path
    ]
    cartesians = [cartesian(p) for p in path]
    # Somewhere along the detour toward the gap, Cartesian distance grows
    # tick-over-tick while geodesic distance (by construction of A*) never
    # does.
    assert any(
        cartesians[i + 1] > cartesians[i] for i in range(len(cartesians) - 1)
    ), "test setup didn't actually force a Cartesian-increasing detour"
    assert all(
        distances[i + 1] <= distances[i] for i in range(len(distances) - 1)
    )


def test_oracle_label_writer_path_progress_delta_tracks_distance_change():
    grid = _grid([["grass", "grass", "grass", "grass"]])
    writer = OracleLabelWriter()
    first = writer.label(semantic=grid, position=(0, 0), goal_position=(3, 0), world_size=(4, 1))
    assert first.path_progress_delta == 0.0  # no previous distance yet
    second = writer.label(semantic=grid, position=(1, 0), goal_position=(3, 0), world_size=(4, 1))
    assert second.path_progress_delta == pytest.approx(1.0)  # 3.0 -> 2.0


def test_oracle_label_writer_flags_replan_required_when_a_wall_appears():
    open_grid = _grid([["grass", "grass", "grass"]])
    writer = OracleLabelWriter()
    first = writer.label(
        semantic=open_grid, position=(0, 0), goal_position=(2, 0), world_size=(3, 1)
    )
    assert first.replan_required is False
    assert first.geodesic_distance == 2.0

    # A wall now seals the only route (a 1-row world has no detour at all).
    blocked_grid = _grid([["grass", "stone", "grass"]])
    second = writer.label(
        semantic=blocked_grid, position=(0, 0), goal_position=(2, 0), world_size=(3, 1)
    )
    assert second.geodesic_distance is None
    assert second.replan_required is True


def test_oracle_label_writer_reset_clears_previous_distance_memory():
    grid = _grid([["grass", "grass", "grass"]])
    writer = OracleLabelWriter()
    writer.label(semantic=grid, position=(0, 0), goal_position=(2, 0), world_size=(3, 1))
    writer.reset()
    label = writer.label(semantic=grid, position=(0, 0), goal_position=(2, 0), world_size=(3, 1))
    assert label.path_progress_delta == 0.0
    assert label.replan_required is False


# ------------------------------------------------------------ DataContract


def test_oracle_map_cost_assumptions_to_contract_dict_shape():
    payload = DEFAULT_MAP_COST_ASSUMPTIONS.to_contract_dict()
    assert payload == {
        "planner_version": "crafter-astar-v1",
        "connectivity": "4-directional",
        "step_cost": 1.0,
        "walkable_materials": ["grass", "path", "sand"],
        "avoid_lava": True,
    }


def _minimal_data_contract(**overrides) -> DataContract:
    fields = dict(
        world="crafter", backend="crafter", program_config={},
        scenario_names=("navigate_open_goal",), scenario_code_version="v1",
        train_session_ids=("s1",), train_session_hashes=("h1",),
        validation_session_ids=(), validation_session_hashes=(),
        test_session_ids=(), test_session_hashes=(),
        seed_assignments={"train": 0}, pixel_provenance="crafter-renderer-v1",
        semantic_vocabulary_version="v1", preprocessing_version="v1",
        horizons_ticks=(1,), ticks_per_frame=1.0,
    )
    fields.update(overrides)
    return DataContract(**fields)


def test_data_contract_exposes_oracle_planner_policy():
    contract = _minimal_data_contract(
        oracle_planner_policy=DEFAULT_MAP_COST_ASSUMPTIONS.to_contract_dict()
    )
    # `DataContract.__post_init__` deep-freezes nested containers (lists ->
    # tuples) so a cached `.hash` can never go stale; compare field-by-field
    # rather than assuming the stored value round-trips as the same list type.
    stored = dict(contract.oracle_planner_policy)
    expected = DEFAULT_MAP_COST_ASSUMPTIONS.to_contract_dict()
    assert stored["planner_version"] == expected["planner_version"]
    assert stored["connectivity"] == expected["connectivity"]
    assert stored["step_cost"] == expected["step_cost"]
    assert list(stored["walkable_materials"]) == expected["walkable_materials"]
    assert stored["avoid_lava"] == expected["avoid_lava"]


def test_data_contract_without_oracle_labels_defaults_to_empty_policy():
    contract = _minimal_data_contract()
    assert dict(contract.oracle_planner_policy) == {}


def test_oracle_planner_policy_participates_in_the_data_contract_hash():
    baseline = _minimal_data_contract()
    with_oracle = _minimal_data_contract(
        oracle_planner_policy=DEFAULT_MAP_COST_ASSUMPTIONS.to_contract_dict()
    )
    other_oracle = _minimal_data_contract(
        oracle_planner_policy=OracleMapCostAssumptions(avoid_lava=False).to_contract_dict()
    )
    assert baseline.hash != with_oracle.hash
    assert with_oracle.hash != other_oracle.hash


def test_goal_reward_config_to_contract_dict_exposes_all_declared_fields():
    """Epic §12.6: "reward potential, strength, falloff, discount, success
    radius and bonus are all DataContract fields"."""
    config = GoalRewardConfig(potential="linear", strength=2.0, discount=0.9, success_bonus=5.0)
    payload = config.to_contract_dict(success_radius=1.5)
    assert payload == {
        "potential": "linear", "strength": 2.0, "falloff": None,
        "discount": 0.9, "success_radius": 1.5, "success_bonus": 5.0,
    }


def test_goal_reward_config_saturating_potential_carries_falloff_in_the_contract():
    config = GoalRewardConfig(potential="saturating", falloff=4.0)
    payload = config.to_contract_dict(success_radius=1.0)
    assert payload["potential"] == "saturating"
    assert payload["falloff"] == 4.0


def test_data_contract_exposes_goal_reward_policy():
    config = GoalRewardConfig()
    contract = _minimal_data_contract(
        goal_reward_policy=config.to_contract_dict(success_radius=1.5)
    )
    assert dict(contract.goal_reward_policy) == config.to_contract_dict(success_radius=1.5)


def test_data_contract_without_goal_reward_defaults_to_empty_policy():
    contract = _minimal_data_contract()
    assert dict(contract.goal_reward_policy) == {}


def test_goal_reward_policy_participates_in_the_data_contract_hash():
    baseline = _minimal_data_contract()
    with_reward = _minimal_data_contract(
        goal_reward_policy=GoalRewardConfig(strength=1.0).to_contract_dict(success_radius=1.5)
    )
    other_reward = _minimal_data_contract(
        goal_reward_policy=GoalRewardConfig(strength=2.0).to_contract_dict(success_radius=1.5)
    )
    assert baseline.hash != with_reward.hash
    assert with_reward.hash != other_reward.hash


# --------------------------------------------------------- GoalRewardConfig


def test_goal_reward_config_rejects_unknown_potential_kind():
    with pytest.raises(ValueError):
        GoalRewardConfig(potential="quadratic")


def test_goal_reward_config_rejects_non_positive_strength():
    with pytest.raises(ValueError):
        GoalRewardConfig(strength=0.0)


def test_goal_reward_config_saturating_requires_positive_falloff():
    with pytest.raises(ValueError):
        GoalRewardConfig(potential="saturating", falloff=None)
    with pytest.raises(ValueError):
        GoalRewardConfig(potential="saturating", falloff=0.0)


def test_goal_reward_config_linear_rejects_a_falloff_value():
    with pytest.raises(ValueError):
        GoalRewardConfig(potential="linear", falloff=2.0)


def test_goal_reward_config_rejects_discount_out_of_range():
    with pytest.raises(ValueError):
        GoalRewardConfig(discount=0.0)
    with pytest.raises(ValueError):
        GoalRewardConfig(discount=1.5)


def test_linear_potential_is_negative_strength_times_distance():
    config = GoalRewardConfig(strength=2.0)
    assert config.potential_of(5.0) == pytest.approx(-10.0)
    assert config.potential_of(0.0) == pytest.approx(0.0)


def test_linear_potential_of_unreachable_state_is_negative_infinity():
    config = GoalRewardConfig()
    assert config.potential_of(None) == -math.inf


def test_saturating_potential_is_bounded_and_flattens_far_from_goal():
    config = GoalRewardConfig(potential="saturating", strength=1.0, falloff=10.0)
    near = config.potential_of(1.0)
    far = config.potential_of(1000.0)
    bound = -1.0 * 10.0
    assert bound < near < 0.0
    assert far == pytest.approx(bound, rel=1e-3)
    # Monotonic: farther from the goal is never a higher (better) potential.
    assert config.potential_of(5.0) < config.potential_of(1.0)


# ----------------------------------------------------- NavigationRewardTracker


def test_navigation_reward_tracker_first_tick_after_prime_matches_the_formula():
    config = GoalRewardConfig(strength=1.0, discount=1.0)
    tracker = NavigationRewardTracker(config)
    tracker.prime(geodesic_distance=10.0)
    components = tracker.compute(geodesic_distance=9.0, goal_active=True, just_arrived=False)
    # gamma * Phi(9) - Phi(10) = 1*(-9) - (-10) = 1.0
    assert components == {"goal_progress": 1.0}


def test_navigation_reward_tracker_round_trip_returning_to_start_nets_zero_progress():
    """The acceptance criterion, stated directly: with gamma=1 (the
    default), a closed loop of geodesic distances sums to zero net
    ``goal_progress`` (Ng et al. 1999's telescoping identity)."""
    config = GoalRewardConfig(strength=1.5, discount=1.0)
    tracker = NavigationRewardTracker(config)
    distances = [10.0, 8.0, 6.0, 9.0, 7.0, 10.0]  # out, in, back out, home
    tracker.prime(geodesic_distance=distances[0])
    total = 0.0
    for d in distances[1:]:
        components = tracker.compute(geodesic_distance=d, goal_active=True, just_arrived=False)
        total += components.get("goal_progress", 0.0)
    assert total == pytest.approx(0.0, abs=1e-9)


def test_navigation_reward_tracker_discount_below_one_breaks_exact_telescoping():
    """Documents the boundary of the invariant above: gamma < 1 stays
    policy-invariant in the usual RL sense but no longer sums to a literal
    zero over a closed loop -- the module docstring's explicit caveat."""
    config = GoalRewardConfig(strength=1.0, discount=0.9)
    tracker = NavigationRewardTracker(config)
    distances = [10.0, 5.0, 10.0]
    tracker.prime(geodesic_distance=distances[0])
    total = 0.0
    for d in distances[1:]:
        components = tracker.compute(geodesic_distance=d, goal_active=True, just_arrived=False)
        total += components.get("goal_progress", 0.0)
    assert total != pytest.approx(0.0, abs=1e-9)


def test_navigation_reward_tracker_pays_arrival_bonus_exactly_once():
    config = GoalRewardConfig(success_bonus=3.0)
    tracker = NavigationRewardTracker(config)
    tracker.prime(geodesic_distance=1.0)
    first = tracker.compute(geodesic_distance=0.0, goal_active=True, just_arrived=True)
    assert first["goal_reached"] == 3.0
    # `just_arrived` staying (incorrectly) True on a later call must not
    # re-pay -- the tracker's own bookkeeping is the actual guard, not the
    # caller's honesty about the harness transition.
    second = tracker.compute(geodesic_distance=0.0, goal_active=False, just_arrived=True)
    assert "goal_reached" not in second


def test_navigation_reward_tracker_no_progress_component_when_goal_inactive():
    config = GoalRewardConfig()
    tracker = NavigationRewardTracker(config)
    tracker.prime(geodesic_distance=5.0)
    components = tracker.compute(geodesic_distance=None, goal_active=False, just_arrived=False)
    assert "goal_progress" not in components
    assert "goal_reached" not in components


def test_navigation_reward_tracker_reset_clears_bonus_and_potential_memory():
    config = GoalRewardConfig(success_bonus=1.0)
    tracker = NavigationRewardTracker(config)
    tracker.prime(geodesic_distance=1.0)
    tracker.compute(geodesic_distance=0.0, goal_active=True, just_arrived=True)
    tracker.reset()
    tracker.prime(geodesic_distance=1.0)
    # A fresh episode's arrival can pay the bonus again.
    components = tracker.compute(geodesic_distance=0.0, goal_active=True, just_arrived=True)
    assert components["goal_reached"] == 1.0


def test_navigation_reward_tracker_unreachable_to_unreachable_yields_no_progress():
    """Neither side of the diff is finite -- must not silently emit `nan`."""
    config = GoalRewardConfig()
    tracker = NavigationRewardTracker(config)
    tracker.prime(geodesic_distance=None)
    components = tracker.compute(geodesic_distance=None, goal_active=True, just_arrived=False)
    assert "goal_progress" not in components


# ------------------------------------------------------- CrafterWorld: wiring


def _crafter_world(**config_overrides):
    pytest.importorskip("crafter")
    from cognitive_runtime.core.streams.bus import MotorStreamBus, SensoryStreamBus
    from cognitive_runtime.programs.crafter.adapter import CrafterWorld

    config = dict(episode_ticks=200, area=(32, 32), size=(32, 32))
    config.update(config_overrides)
    program = CrafterWorld(config=config)
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory, motor)
    program.reset(seed=0)
    return program, sensory, motor


def _remove_wildlife(env) -> None:
    """Strip incidental world-gen creatures so a scripted layout is fully
    deterministic (mirrors ``training.nursery._crafter_remove_wildlife``'s
    approach without pulling in that module's torch dependency)."""
    for obj in list(env._world.objects):
        if obj is not env._player:
            env._world.remove(obj)


def test_stream_catalog_never_declares_an_oracle_stream():
    """Acceptance criterion, stated directly: enumerate the deployed
    sensory workspace's stream keys and confirm no oracle label ever
    appears there -- the structural barrier, not a naming convention."""
    program, _, _ = _crafter_world(
        goal_enabled=True, goal_min_initial_distance=2.0, goal_arrival_radius=1.0,
    )
    ids = {spec.stream_id for spec in program.stream_catalog()}
    assert ids  # sanity: the catalog isn't accidentally empty
    assert not any("oracle" in stream_id for stream_id in ids)
    # And the exact fixed set the non-navigation catalog also publishes,
    # confirming oracle labels don't sneak in under an unrelated-looking id.
    assert ids == {
        "vision.frame.grid", "vision.frame.pixels", "body.health", "body.food",
        "body.drink", "body.energy", "body.inventory", "body.sleeping", "body.alive",
        "spatial.position", "spatial.facing", "event.achievement", "event.died",
        "event.action_rejected", "reward.scalar", "internal.goal",
    }


def test_oracle_labels_is_none_when_goal_conditioning_is_disabled():
    program, _, _ = _crafter_world()
    assert program.oracle_labels() is None


def test_oracle_labels_has_the_expected_shape_when_goal_enabled():
    program, _, _ = _crafter_world(
        goal_enabled=True, goal_min_initial_distance=2.0, goal_arrival_radius=1.0,
    )
    label = program.oracle_labels()
    assert label is not None
    assert set(label) == {
        "geodesic_distance", "next_action", "legal_action_mask", "path_progress_delta",
        "replan_required", "planner_version", "map_cost_assumptions",
    }
    assert label["planner_version"] == ORACLE_PLANNER_VERSION


def _find_reachable_seed(config_overrides, max_seed=30):
    pytest.importorskip("crafter")
    for seed in range(max_seed):
        program, _, _ = _crafter_world(**config_overrides)
        program.reset(seed=seed)
        label = program.oracle_labels()
        if label is not None and label["geodesic_distance"] is not None:
            return program, seed
    pytest.skip("no reachable-goal seed found in range; flaky world-gen, not a code bug")


def test_following_the_oracles_own_suggestion_monotonically_reduces_geodesic_distance():
    from cognitive_runtime.core.action import Action
    from cognitive_runtime.core.streams.motor import publish_motor_command

    config = dict(
        goal_enabled=True, goal_min_initial_distance=3.0, goal_arrival_radius=1.0,
        goal_commitment_horizon_ticks=5,
    )
    program, seed = _find_reachable_seed(config)
    from cognitive_runtime.core.streams.bus import MotorStreamBus, SensoryStreamBus
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory, motor)
    program.reset(seed=seed)
    sensory.drain()

    previous = program.oracle_labels()["geodesic_distance"]
    for tick in range(int(previous) + 2):
        label = program.oracle_labels()
        action_name = label["next_action"]
        if action_name is None:
            break
        publish_motor_command(motor, Action(action_name), timestamp=float(tick))
        program.step()
        current = program.oracle_labels()["geodesic_distance"]
        assert current is not None
        assert current <= previous
        previous = current
    assert previous == 0.0


def test_goal_reached_reward_component_paid_exactly_once_for_the_whole_episode():
    from cognitive_runtime.core.action import Action
    from cognitive_runtime.core.streams.bus import MotorStreamBus, SensoryStreamBus
    from cognitive_runtime.core.streams.motor import publish_motor_command

    config = dict(
        goal_enabled=True, goal_min_initial_distance=3.0, goal_arrival_radius=1.0,
        goal_commitment_horizon_ticks=5,
    )
    program, seed = _find_reachable_seed(config)
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory, motor)
    program.reset(seed=seed)
    sensory.drain()

    arrivals = 0
    for tick in range(60):
        label = program.oracle_labels()
        action_name = label["next_action"] or "NULL"
        publish_motor_command(motor, Action(action_name), timestamp=float(tick))
        program.step()
        if "goal_reached" in program.reward().components:
            arrivals += 1
    assert arrivals == 1


def test_reward_components_keep_task_goal_progress_and_goal_reached_separate():
    from cognitive_runtime.core.action import Action
    from cognitive_runtime.core.streams.bus import MotorStreamBus, SensoryStreamBus
    from cognitive_runtime.core.streams.motor import publish_motor_command

    config = dict(
        goal_enabled=True, goal_min_initial_distance=3.0, goal_arrival_radius=1.0,
        goal_commitment_horizon_ticks=5,
    )
    program, seed = _find_reachable_seed(config)
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory, motor)
    program.reset(seed=seed)
    sensory.drain()

    seen_progress = False
    seen_reached = False
    for tick in range(60):
        label = program.oracle_labels()
        action_name = label["next_action"] or "NULL"
        publish_motor_command(motor, Action(action_name), timestamp=float(tick))
        program.step()
        components = program.reward().components
        # No component ever folds goal-navigation reward into the legacy
        # "crafter" key -- each stays independently named.
        assert "crafter" not in components
        if "goal_progress" in components:
            seen_progress = True
            assert isinstance(components["goal_progress"], float)
        if "goal_reached" in components:
            seen_reached = True
    assert seen_progress and seen_reached


def test_collision_penalty_is_reported_as_its_own_component():
    pytest.importorskip("crafter")
    from cognitive_runtime.core.action import Action
    from cognitive_runtime.core.streams.bus import MotorStreamBus, SensoryStreamBus
    from cognitive_runtime.core.streams.motor import publish_motor_command
    from cognitive_runtime.programs.crafter.adapter import CrafterWorld

    def hook(env):
        _remove_wildlife(env)
        px, py = int(env._player.pos[0]), int(env._player.pos[1])
        env._world[(px + 1, py)] = "stone"  # wall directly to the right

    program = CrafterWorld(config=dict(
        episode_ticks=200, area=(32, 32), size=(32, 32),
        goal_enabled=True, goal_min_initial_distance=2.0, goal_arrival_radius=1.0,
        goal_commitment_horizon_ticks=5, goal_collision_penalty=0.3,
    ))
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory, motor)
    program.set_post_reset_hook(hook)
    program.reset(seed=0)
    sensory.drain()

    before = program._position_tuple()
    publish_motor_command(motor, Action("MOVE_RIGHT"), timestamp=0.0)
    program.step()
    after = program._position_tuple()
    assert after == before  # genuinely blocked
    assert program.reward().components["collision"] == pytest.approx(-0.3)


def test_collision_penalty_is_zero_by_default_and_omitted():
    pytest.importorskip("crafter")
    from cognitive_runtime.core.action import Action
    from cognitive_runtime.core.streams.bus import MotorStreamBus, SensoryStreamBus
    from cognitive_runtime.core.streams.motor import publish_motor_command
    from cognitive_runtime.programs.crafter.adapter import CrafterWorld

    def hook(env):
        _remove_wildlife(env)
        px, py = int(env._player.pos[0]), int(env._player.pos[1])
        env._world[(px + 1, py)] = "stone"

    program = CrafterWorld(config=dict(
        episode_ticks=200, area=(32, 32), size=(32, 32),
        goal_enabled=True, goal_min_initial_distance=2.0, goal_arrival_radius=1.0,
        goal_commitment_horizon_ticks=5,  # goal_collision_penalty left at its 0.0 default
    ))
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory, motor)
    program.set_post_reset_hook(hook)
    program.reset(seed=0)
    sensory.drain()

    publish_motor_command(motor, Action("MOVE_RIGHT"), timestamp=0.0)
    program.step()
    assert "collision" not in program.reward().components


def test_detour_around_a_wall_increases_cartesian_distance_but_is_rewarded_positively():
    """The end-to-end acceptance criterion against the real `CrafterWorld`
    reward pipeline, not just the planner in isolation."""
    pytest.importorskip("crafter")
    from cognitive_runtime.core.action import Action
    from cognitive_runtime.core.streams.bus import MotorStreamBus, SensoryStreamBus
    from cognitive_runtime.core.streams.motor import publish_motor_command
    from cognitive_runtime.programs.crafter.adapter import CrafterWorld

    spawn = {}

    def hook(env):
        _remove_wildlife(env)
        px, py = int(env._player.pos[0]), int(env._player.pos[1])
        width, height = env._world.area
        lo_x, hi_x = max(0, px - 10), min(width, px + 11)
        lo_y, hi_y = max(0, py - 10), min(height, py + 11)
        for x in range(lo_x, hi_x):
            for y in range(lo_y, hi_y):
                env._world[(x, y)] = "grass"
        wall_x = px + 3
        gap_y = py + 8
        for y in range(lo_y, hi_y):
            if y != gap_y:
                env._world[(wall_x, y)] = "stone"
        spawn["pos"] = (px, py)

    program = CrafterWorld(config=dict(
        episode_ticks=200, area=(32, 32), size=(32, 32),
        goal_enabled=True, goal_min_initial_distance=2.0, goal_arrival_radius=1.0,
        goal_commitment_horizon_ticks=200,
    ))
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory, motor)
    program.set_post_reset_hook(hook)
    program.reset(seed=0)

    px, py = spawn["pos"]
    goal = (float(px + 6), float(py))
    program._goal_setter.tracker._goal_position = goal
    program._oracle_writer.reset()
    program._pending_oracle_label = program._compute_oracle_label()
    program._reward_tracker.reset()
    program._reward_tracker.prime(
        geodesic_distance=program._pending_oracle_label.geodesic_distance
    )
    program._pending_goal_state = program._goal_setter.tracker.state(program._position_tuple())
    sensory.drain()

    def cartesian(pos):
        return math.hypot(pos[0] - goal[0], pos[1] - goal[1])

    previous_cartesian = cartesian(program._position_tuple())
    saw_detour_with_positive_reward = False
    for tick in range(30):
        label = program.oracle_labels()
        action_name = label["next_action"]
        if action_name is None:
            break
        publish_motor_command(motor, Action(action_name), timestamp=float(tick))
        program.step()
        current_cartesian = cartesian(program._position_tuple())
        progress = program.reward().components.get("goal_progress", 0.0)
        if current_cartesian > previous_cartesian and progress > 0.0:
            saw_detour_with_positive_reward = True
        previous_cartesian = current_cartesian

    assert saw_detour_with_positive_reward


# --------------------------------------------------- CognitiveRuntime: recording


def _make_crafter_runtime(tmp_path, policy, episodes=1, seed=0):
    pytest.importorskip("crafter")
    from cognitive_runtime.programs.crafter.adapter import CrafterWorld
    from cognitive_runtime.runtime.config import RuntimeConfig
    from cognitive_runtime.runtime.loop import CognitiveRuntime

    program_config = dict(
        episode_ticks=40, area=(32, 32), size=(32, 32),
        goal_enabled=True, goal_min_initial_distance=2.0, goal_arrival_radius=1.0,
        goal_commitment_horizon_ticks=10,
    )
    config = RuntimeConfig(
        episodes=episodes, seed=seed, max_ticks_per_episode=40,
        record_dir=str(tmp_path), session_id="nav-oracle-session",
        program_config=program_config,
    )
    return CognitiveRuntime(program=CrafterWorld(config=program_config), policy=policy, config=config)


def test_oracle_labels_appear_in_decision_records_but_never_in_the_stream_log(tmp_path):
    """The full acceptance criterion end to end: training-only oracle labels
    reach the recorded *decisions*, never the recorded *streams* -- i.e.
    never the deployed sensory workspace a policy actually reads from."""
    pytest.importorskip("crafter")
    from cognitive_runtime.policies import RandomPolicy
    from cognitive_runtime.programs.crafter.actions import ACTION_SPACE
    from cognitive_runtime.runtime.replay import load_decisions, load_stream_log

    policy = RandomPolicy(ACTION_SPACE, seed=1)
    runtime = _make_crafter_runtime(tmp_path, policy)
    runtime.run()

    session_dir = os.path.join(str(tmp_path), "nav-oracle-session")
    decisions = load_decisions(session_dir, "episode_00000")
    assert decisions
    assert any(d.get("oracle_labels") is not None for d in decisions)
    for d in decisions:
        labels = d.get("oracle_labels")
        if labels is not None:
            assert "geodesic_distance" in labels
            assert "next_action" in labels

    stream_log = load_stream_log(session_dir, "episode_00000")
    assert stream_log
    stream_ids = {record["stream_id"] for record in stream_log}
    assert not any("oracle" in stream_id for stream_id in stream_ids)
    for record in stream_log:
        payload = record.get("payload")
        if isinstance(payload, dict):
            assert "geodesic_distance" not in payload
            assert "legal_action_mask" not in payload
