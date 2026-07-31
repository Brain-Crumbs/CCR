"""Goal representation and caregiver goal source (epic #212 §12.4, issue
#238).

Covers this issue's acceptance criteria:

- the four goal fields form one well-formed ``internal.*`` stream, matching
  the existing internal-stream payload shape;
- ``internal.goal.source`` is ``"caregiver"`` in every recording this issue
  produces;
- a goal below the minimum initial distance is rejected at proposal time;
- a goal cannot be changed within the commitment horizon;
- arrival is decided by the harness, not by a value the policy can write;
- the goal-distribution parameters are exposed for the ``DataContract``.
"""

from __future__ import annotations

import dataclasses

import pytest

from cognitive_runtime.core.goal import (
    CAREGIVER_SOURCE,
    GOAL_SOURCES,
    GOAL_STREAM,
    GOAL_STREAM_SPEC,
    INACTIVE_GOAL_STATE,
    ORGANISM_SOURCE,
    GoalDistributionConfig,
    GoalState,
    GoalTracker,
)
from cognitive_runtime.core.streams.events import StreamEvent, validate_stream_identity
from cognitive_runtime.core.streams.registry import DEFAULT_STREAM_REGISTRY
from cognitive_runtime.programs.crafter.goals import CaregiverGoalSetter, sample_caregiver_goal
from cognitive_runtime.training.model_factory.contracts import DataContract


def _config(**overrides) -> GoalDistributionConfig:
    fields = dict(
        min_initial_distance=6.0,
        commitment_horizon_ticks=20,
        arrival_radius=1.5,
    )
    fields.update(overrides)
    return GoalDistributionConfig(**fields)


# --------------------------------------------------------------- stream shape


def test_goal_stream_is_a_valid_internal_event_stream():
    validate_stream_identity(GOAL_STREAM, "event")
    assert GOAL_STREAM == "internal.goal"
    assert GOAL_STREAM_SPEC.stream_id == GOAL_STREAM
    assert GOAL_STREAM_SPEC.modality == "event"


def test_goal_stream_matches_the_generic_internal_wildcard_declaration():
    """``internal.goal`` gets the same agent-input/interoceptive treatment
    every other ``internal.*`` stream (dopamine, risk, ...) already has --
    no bespoke declaration needed."""
    decl = DEFAULT_STREAM_REGISTRY.declaration_for(GOAL_STREAM)
    assert decl is not None
    assert decl.pattern == "internal.*"
    assert decl.classification == "agent_input"
    assert decl.attention is not None
    assert decl.attention.modality == "internal"


def test_goal_state_payload_matches_existing_internal_stream_shape():
    state = GoalState(relative_position=(3.0, -4.0), distance=5.0, active=True, source="caregiver")
    payload = state.as_payload()
    assert set(payload) == {"relative_position", "distance", "active", "source"}
    assert payload["relative_position"] == [3.0, -4.0]
    assert payload["distance"] == 5.0
    assert payload["active"] is True
    assert payload["source"] == "caregiver"
    # The payload must itself be a legal StreamEvent payload on this stream id
    # (mirrors how internal.arbiter_mode/internal.attention.weights publish a
    # multi-field dict under one stream id).
    event = StreamEvent(GOAL_STREAM, "event", timestamp=0.0, sequence_number=0, payload=payload)
    assert event.payload == payload


def test_goal_state_is_frozen_so_downstream_code_cannot_rewrite_it():
    state = GoalState((0.0, 0.0), 0.0, False, CAREGIVER_SOURCE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.active = True  # type: ignore[misc]


def test_goal_state_rejects_unknown_source():
    with pytest.raises(ValueError):
        GoalState((0.0, 0.0), 0.0, False, "the_organism_totally_not_gaming_this")


def test_inactive_goal_state_still_declares_the_caregiver_source():
    assert INACTIVE_GOAL_STATE.source == CAREGIVER_SOURCE
    assert INACTIVE_GOAL_STATE.active is False


def test_goal_sources_include_organism_for_future_use_but_default_is_caregiver():
    assert GOAL_SOURCES == (CAREGIVER_SOURCE, ORGANISM_SOURCE)


# ----------------------------------------------------------- distribution config


def test_goal_distribution_config_rejects_non_positive_min_distance():
    with pytest.raises(ValueError):
        _config(min_initial_distance=0.0)


def test_goal_distribution_config_rejects_non_positive_commitment_horizon():
    with pytest.raises(ValueError):
        _config(commitment_horizon_ticks=0)


def test_goal_distribution_config_rejects_negative_arrival_radius():
    with pytest.raises(ValueError):
        _config(arrival_radius=-1.0)


def test_goal_distribution_config_rejects_arrival_radius_at_or_past_min_distance():
    """An arrival radius >= the minimum initial distance would let a freshly
    proposed goal already count as arrived -- exactly the self-gaming this
    issue exists to prevent."""
    with pytest.raises(ValueError):
        _config(min_initial_distance=2.0, arrival_radius=2.0)


def test_goal_distribution_config_rejects_max_below_min():
    with pytest.raises(ValueError):
        _config(min_initial_distance=5.0, max_initial_distance=4.0)


def test_goal_distribution_config_to_contract_dict_is_plain_data():
    config = _config(max_initial_distance=12.0)
    payload = config.to_contract_dict()
    assert payload == {
        "min_initial_distance": 6.0,
        "max_initial_distance": 12.0,
        "commitment_horizon_ticks": 20,
        "arrival_radius": 1.5,
        "distribution": "uniform_in_bounds",
        "generator_version": "goal_distribution_v1",
    }


# --------------------------------------------------------------- GoalTracker


def test_goal_below_minimum_initial_distance_is_rejected_at_proposal_time():
    tracker = GoalTracker(_config(min_initial_distance=6.0))
    reason = tracker.propose(
        goal_position=(3.0, 0.0), current_position=(0.0, 0.0), tick=0, source=CAREGIVER_SOURCE,
    )
    assert reason is not None
    assert "minimum initial distance" in reason
    assert tracker.has_goal is False
    assert tracker.active is False


def test_goal_at_or_above_minimum_initial_distance_is_accepted():
    tracker = GoalTracker(_config(min_initial_distance=6.0))
    reason = tracker.propose(
        goal_position=(6.0, 0.0), current_position=(0.0, 0.0), tick=0, source=CAREGIVER_SOURCE,
    )
    assert reason is None
    assert tracker.active is True


def test_goal_cannot_change_within_the_commitment_horizon():
    tracker = GoalTracker(_config(min_initial_distance=6.0, commitment_horizon_ticks=10))
    assert tracker.propose(
        goal_position=(6.0, 0.0), current_position=(0.0, 0.0), tick=0,
    ) is None

    reason = tracker.propose(
        goal_position=(0.0, 6.0), current_position=(0.0, 0.0), tick=9,
    )
    assert reason is not None
    assert "commitment horizon" in reason
    # The original goal survives the rejected proposal unchanged.
    state = tracker.state((0.0, 0.0))
    assert state.relative_position == (6.0, 0.0)


def test_goal_can_change_once_the_commitment_horizon_elapses():
    tracker = GoalTracker(_config(min_initial_distance=6.0, commitment_horizon_ticks=10))
    tracker.propose(goal_position=(6.0, 0.0), current_position=(0.0, 0.0), tick=0)

    reason = tracker.propose(
        goal_position=(0.0, 6.0), current_position=(0.0, 0.0), tick=10,
    )
    assert reason is None
    state = tracker.state((0.0, 0.0))
    assert state.relative_position == (0.0, 6.0)


def test_goal_can_change_immediately_after_the_previous_goal_was_reached():
    """The commitment horizon guards an *active* goal; once the harness has
    already decided arrival, a new one may be proposed without waiting out
    the rest of the horizon."""
    tracker = GoalTracker(_config(min_initial_distance=6.0, commitment_horizon_ticks=100))
    tracker.propose(goal_position=(6.0, 0.0), current_position=(0.0, 0.0), tick=0)
    assert tracker.evaluate_arrival((6.0, 0.0)) is True

    reason = tracker.propose(goal_position=(0.0, 6.0), current_position=(6.0, 0.0), tick=1)
    assert reason is None


def test_goal_proposal_rejects_unknown_source():
    tracker = GoalTracker(_config())
    reason = tracker.propose(
        goal_position=(6.0, 0.0), current_position=(0.0, 0.0), tick=0, source="policy_self_report",
    )
    assert reason is not None
    assert tracker.has_goal is False


# ------------------------------------------------------- harness-only arrival


def test_arrival_is_not_reached_while_outside_the_arrival_radius():
    tracker = GoalTracker(_config(min_initial_distance=6.0, arrival_radius=1.0))
    tracker.propose(goal_position=(6.0, 0.0), current_position=(0.0, 0.0), tick=0)
    assert tracker.evaluate_arrival((0.0, 0.0)) is False
    assert tracker.active is True
    assert tracker.state((0.0, 0.0)).active is True


def test_arrival_is_reached_once_inside_the_arrival_radius():
    tracker = GoalTracker(_config(min_initial_distance=6.0, arrival_radius=1.0))
    tracker.propose(goal_position=(6.0, 0.0), current_position=(0.0, 0.0), tick=0)
    assert tracker.evaluate_arrival((5.5, 0.0)) is True
    assert tracker.active is False
    assert tracker.state((5.5, 0.0)).active is False


def test_arrival_decision_cannot_be_set_directly_only_evaluated():
    """No public API sets ``active``/arrival except ``evaluate_arrival``,
    which always takes the caller's *own* position argument -- there is no
    ``mark_arrived()``/``force_arrival()`` a policy path could call instead,
    and ``active`` is a read-only property."""
    tracker = GoalTracker(_config())
    assert not hasattr(tracker, "mark_arrived")
    assert not hasattr(tracker, "force_arrival")
    assert not hasattr(tracker, "set_active")
    with pytest.raises(AttributeError):
        tracker.active = True  # type: ignore[misc]


def test_evaluate_arrival_requires_the_caller_to_supply_ground_truth_position():
    tracker = GoalTracker(_config())
    with pytest.raises(TypeError):
        tracker.evaluate_arrival()  # type: ignore[call-arg]


def test_evaluate_arrival_before_any_goal_is_proposed_stays_false():
    tracker = GoalTracker(_config())
    assert tracker.evaluate_arrival((0.0, 0.0)) is False
    assert tracker.state((0.0, 0.0)) == INACTIVE_GOAL_STATE


# ------------------------------------------------------------- relative_position


def test_state_relative_position_and_distance_track_the_current_position():
    tracker = GoalTracker(_config(min_initial_distance=6.0, arrival_radius=1.0))
    tracker.propose(goal_position=(10.0, 0.0), current_position=(0.0, 0.0), tick=0)
    state = tracker.state((4.0, 0.0))
    assert state.relative_position == (6.0, 0.0)
    assert state.distance == pytest.approx(6.0)
    assert state.source == CAREGIVER_SOURCE


# ---------------------------------------------------------- caregiver goal setter


def test_sample_caregiver_goal_always_respects_the_minimum_distance():
    config = _config(min_initial_distance=5.0, arrival_radius=1.0)
    for seed in range(50):
        rng = __import__("random").Random(seed)
        goal = sample_caregiver_goal(
            spawn_position=(8.0, 8.0), world_size=(16, 16), config=config, rng=rng,
        )
        dx = goal[0] - 8.0
        dy = goal[1] - 8.0
        assert (dx * dx + dy * dy) ** 0.5 >= config.min_initial_distance - 1e-9


def test_sample_caregiver_goal_raises_when_unsatisfiable_in_a_tiny_world():
    config = _config(min_initial_distance=100.0, arrival_radius=1.0)
    rng = __import__("random").Random(0)
    with pytest.raises(ValueError):
        sample_caregiver_goal(
            spawn_position=(2.0, 2.0), world_size=(4, 4), config=config, rng=rng, max_attempts=16,
        )


def test_caregiver_goal_setter_always_sources_caregiver():
    setter = CaregiverGoalSetter(_config(min_initial_distance=5.0, arrival_radius=1.0))
    setter.reset(spawn_position=(8.0, 8.0), world_size=(32, 32), seed=7)
    state = setter.tick(current_position=(8.0, 8.0))
    assert state.source == CAREGIVER_SOURCE
    assert state.active is True
    assert state.distance >= 5.0 - 1e-9


def test_caregiver_goal_setter_is_deterministic_given_the_same_seed():
    config = _config(min_initial_distance=5.0, arrival_radius=1.0)
    setter_a = CaregiverGoalSetter(config)
    setter_a.reset(spawn_position=(8.0, 8.0), world_size=(32, 32), seed=42)
    setter_b = CaregiverGoalSetter(config)
    setter_b.reset(spawn_position=(8.0, 8.0), world_size=(32, 32), seed=42)
    assert setter_a.tracker.state((8.0, 8.0)) == setter_b.tracker.state((8.0, 8.0))


def test_caregiver_goal_setter_arrival_deactivates_the_goal():
    setter = CaregiverGoalSetter(_config(min_initial_distance=5.0, arrival_radius=1.0))
    setter.reset(spawn_position=(8.0, 8.0), world_size=(32, 32), seed=7)
    goal_position = setter.tracker._goal_position
    assert goal_position is not None
    state = setter.tick(current_position=goal_position)
    assert state.active is False
    assert state.source == CAREGIVER_SOURCE


def test_caregiver_goal_setter_reset_starts_a_fresh_episode_goal_every_time():
    """A new episode's ``reset()`` must not collide with the previous
    episode's still-"active" commitment window -- both proposals happen at
    tick 0 of their own episode."""
    setter = CaregiverGoalSetter(
        _config(min_initial_distance=5.0, commitment_horizon_ticks=1000, arrival_radius=1.0)
    )
    setter.reset(spawn_position=(8.0, 8.0), world_size=(32, 32), seed=1)
    first_goal = setter.tracker._goal_position
    assert first_goal is not None

    setter.reset(spawn_position=(8.0, 8.0), world_size=(32, 32), seed=2)
    second_state = setter.tracker.state((8.0, 8.0))
    assert second_state.active is True
    assert second_state.source == CAREGIVER_SOURCE


def test_caregiver_goal_setter_cannot_be_reproposed_mid_commitment_by_a_bare_propose_call():
    """Even though ``CaregiverGoalSetter`` itself never calls ``propose``
    again mid-episode, the underlying tracker it wraps still enforces the
    commitment horizon against any other caller -- the same anti-gaming
    guarantee ``CrafterWorld`` inherits for free."""
    setter = CaregiverGoalSetter(_config(min_initial_distance=5.0, commitment_horizon_ticks=50))
    setter.reset(spawn_position=(8.0, 8.0), world_size=(32, 32), seed=7)
    reason = setter.tracker.propose(
        goal_position=(0.0, 0.0), current_position=(8.0, 8.0), tick=1, source=CAREGIVER_SOURCE,
    )
    assert reason is not None


# --------------------------------------------------------------- DataContract


def _minimal_data_contract(**overrides) -> DataContract:
    fields = dict(
        world="crafter",
        backend="crafter",
        program_config={},
        scenario_names=("navigate_open_goal",),
        scenario_code_version="v1",
        train_session_ids=("s1",),
        train_session_hashes=("h1",),
        validation_session_ids=(),
        validation_session_hashes=(),
        test_session_ids=(),
        test_session_hashes=(),
        seed_assignments={"train": 0},
        pixel_provenance="crafter-renderer-v1",
        semantic_vocabulary_version="v1",
        preprocessing_version="v1",
        horizons_ticks=(1,),
        ticks_per_frame=1.0,
    )
    fields.update(overrides)
    return DataContract(**fields)


def test_goal_distribution_parameters_are_exposed_on_the_data_contract():
    config = _config(min_initial_distance=6.0, max_initial_distance=20.0)
    contract = _minimal_data_contract(goal_distribution_policy=config.to_contract_dict())
    assert dict(contract.goal_distribution_policy) == config.to_contract_dict()


def test_goal_distribution_policy_participates_in_the_data_contract_hash():
    baseline = _minimal_data_contract()
    with_goal = _minimal_data_contract(
        goal_distribution_policy=_config(min_initial_distance=6.0).to_contract_dict()
    )
    other_goal = _minimal_data_contract(
        goal_distribution_policy=_config(min_initial_distance=9.0).to_contract_dict()
    )
    assert baseline.hash != with_goal.hash
    assert with_goal.hash != other_goal.hash


def test_data_contract_without_a_goal_defaults_to_an_empty_policy():
    contract = _minimal_data_contract()
    assert dict(contract.goal_distribution_policy) == {}


# ------------------------------------------------------- CrafterWorld end-to-end


def _crafter_world(**config_overrides):
    pytest.importorskip("crafter")
    from cognitive_runtime.core.streams.bus import MotorStreamBus, SensoryStreamBus
    from cognitive_runtime.programs.crafter.adapter import CrafterWorld

    config = dict(episode_ticks=64, area=(24, 24), size=(24, 24))
    config.update(config_overrides)
    program = CrafterWorld(config=config)
    sensory, motor = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory, motor)
    program.reset(seed=3)
    return program, sensory, motor


def _goal_events(sensory):
    return [e for e in sensory.drain() if e.stream_id == GOAL_STREAM]


def test_crafter_world_publishes_a_well_formed_active_caregiver_goal_when_enabled():
    program, sensory, motor = _crafter_world(
        goal_enabled=True, goal_min_initial_distance=4.0, goal_arrival_radius=1.0,
        goal_commitment_horizon_ticks=8,
    )
    events = _goal_events(sensory)
    assert len(events) == 1  # one full snapshot right after reset
    payload = events[0].payload
    assert set(payload) == {"relative_position", "distance", "active", "source"}
    assert payload["source"] == CAREGIVER_SOURCE
    assert payload["active"] is True
    assert payload["distance"] >= 4.0 - 1e-6

    for _ in range(5):
        program.step()
        for event in _goal_events(sensory):
            # Every recorded internal.goal event across this episode reports
            # the caregiver source (issue #238 acceptance criterion).
            assert event.payload["source"] == CAREGIVER_SOURCE


def test_crafter_world_goal_disabled_by_default_still_reports_caregiver_source():
    program, sensory, motor = _crafter_world()
    events = _goal_events(sensory)
    assert len(events) == 1
    payload = events[0].payload
    assert payload == {
        "relative_position": [0.0, 0.0], "distance": 0.0, "active": False, "source": "caregiver",
    }


def test_crafter_world_goal_reset_is_deterministic_for_a_fixed_seed():
    program_a, sensory_a, _ = _crafter_world(
        goal_enabled=True, goal_min_initial_distance=4.0, goal_arrival_radius=1.0,
    )
    program_b, sensory_b, _ = _crafter_world(
        goal_enabled=True, goal_min_initial_distance=4.0, goal_arrival_radius=1.0,
    )
    assert _goal_events(sensory_a)[0].payload == _goal_events(sensory_b)[0].payload


def test_crafter_world_second_episode_reset_gets_its_own_active_goal():
    """Regression: ``CognitiveRuntime.run()`` calls ``program.reset(seed)``
    at the start of every episode on the same ``Program`` instance -- a
    second episode's goal proposal must not be rejected by the first
    episode's still-"active" commitment horizon (both proposals land at
    that episode's own tick 0)."""
    program, sensory, motor = _crafter_world(
        goal_enabled=True, goal_min_initial_distance=4.0, goal_arrival_radius=1.0,
        goal_commitment_horizon_ticks=1000,
    )
    sensory.drain()
    program.reset(seed=9)
    events = _goal_events(sensory)
    assert len(events) == 1
    assert events[0].payload["active"] is True
    assert events[0].payload["source"] == CAREGIVER_SOURCE


def test_crafter_world_harness_arrival_deactivates_the_published_goal():
    """Arrival flips ``active`` only once the harness's own ground-truth
    position (``CrafterWorld``'s cached env state) reaches the goal --
    nothing published to the motor bus can do this directly."""
    program, sensory, motor = _crafter_world(
        goal_enabled=True, goal_min_initial_distance=4.0, goal_arrival_radius=1.0,
        goal_commitment_horizon_ticks=8,
    )
    goal_position = program._goal_setter.tracker._goal_position
    assert program._goal_setter.tracker.evaluate_arrival(goal_position) is True

    program.step()
    events = _goal_events(sensory)
    assert events, "internal.goal must still publish once the goal is inactive"
    assert events[-1].payload["active"] is False
    assert events[-1].payload["source"] == CAREGIVER_SOURCE
