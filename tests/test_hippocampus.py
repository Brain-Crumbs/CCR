"""The Hippocampus (issue #96): capacity bound, priority ordering that
matches `transition_priority` directly, priority-based eviction, and a real
recorded run populating the store without a tick stall."""

from __future__ import annotations

import os
import time

import pytest

from brain.hippocampus import (
    HippocampalRetrievalConfig,
    Hippocampus,
    HippocampusConfig,
    SeedTags,
)
from cognitive_runtime.core.priority import PriorityWeights, Transition, transition_priority
from cognitive_runtime.policies import ScriptedSurvivalPolicy
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime

# --------------------------------------------------------------- config


def test_config_rejects_non_positive_capacity():
    with pytest.raises(ValueError, match="capacity"):
        HippocampusConfig(capacity=0)


def test_config_rejects_out_of_range_threat_threshold():
    with pytest.raises(ValueError, match="threat_threshold"):
        HippocampusConfig(threat_threshold=1.5)


def test_retrieval_config_rejects_invalid_gates():
    with pytest.raises(ValueError, match="top_k"):
        HippocampalRetrievalConfig(top_k=0)
    with pytest.raises(ValueError, match="min_similarity"):
        HippocampalRetrievalConfig(min_similarity=1.1)
    with pytest.raises(ValueError, match="min_surprise"):
        HippocampalRetrievalConfig(min_surprise=-0.1)


# --------------------------------------------------------------- encoding + priority


def test_encode_returns_the_stored_seed_and_counts_it():
    hippocampus = Hippocampus()
    seed = hippocampus.encode(
        z=[0.1, 0.2],
        actions=["move_forward"],
        tags=SeedTags(reward=1.0),
        tick_index=3,
        cortex_version=2,
        context_z=[0.3],
    )
    assert seed is not None
    assert seed.z == [0.1, 0.2]
    assert seed.actions == ["move_forward"]
    assert seed.tick_index == 3
    assert seed.cortex_version == 2
    assert seed.context_z == [0.3]
    assert len(hippocampus) == 1
    assert hippocampus.total_encoded == 1


def test_priority_matches_transition_priority_for_the_documented_field_mapping():
    """`SeedTags` -> `Transition` mapping (module docstring): reward/novelty
    carry over directly, `surprise` -> `prediction_error`, `dopamine` ->
    `reward_prediction_error`, and a sub-threshold `threat` leaves `damage`
    false."""
    weights = PriorityWeights(reward=2.0, novelty=0.3, prediction_error=0.7)
    config = HippocampusConfig(weights=weights, threat_threshold=0.5)
    hippocampus = Hippocampus(config)
    tags = SeedTags(reward=0.8, novelty=0.4, surprise=0.6, dopamine=0.2, threat=0.1)

    seed = hippocampus.encode(z=[0.0], actions=[], tags=tags)

    expected = transition_priority(
        Transition(
            latent=[], action=0, reward=0.8, next_latent=[], done=False, damage=False,
            novelty=0.4, prediction_error=0.6, reward_prediction_error=0.2,
        ),
        weights,
    )
    assert seed.priority == pytest.approx(expected)


def test_threat_at_or_above_threshold_folds_into_the_damage_flag():
    weights = PriorityWeights()
    config = HippocampusConfig(weights=weights, threat_threshold=0.5)

    calm = Hippocampus(config).encode(z=[0.0], actions=[], tags=SeedTags(threat=0.1))
    threatened = Hippocampus(config).encode(z=[0.0], actions=[], tags=SeedTags(threat=0.9))

    expected_calm = transition_priority(
        Transition(latent=[], action=0, reward=0.0, next_latent=[], done=False, damage=False),
        weights,
    )
    expected_threatened = transition_priority(
        Transition(latent=[], action=0, reward=0.0, next_latent=[], done=False, damage=True),
        weights,
    )
    assert calm.priority == pytest.approx(expected_calm)
    assert threatened.priority == pytest.approx(expected_threatened)
    assert threatened.priority > calm.priority


def test_explicit_damage_flag_dominates_regardless_of_threat():
    config = HippocampusConfig(threat_threshold=0.9)
    hippocampus = Hippocampus(config)
    seed = hippocampus.encode(z=[0.0], actions=[], tags=SeedTags(damage=True, threat=0.0))
    expected = transition_priority(
        Transition(latent=[], action=0, reward=0.0, next_latent=[], done=False, damage=True),
        config.weights,
    )
    assert seed.priority == pytest.approx(expected)


# --------------------------------------------------------------- capacity + eviction


def test_capacity_bound_is_enforced():
    hippocampus = Hippocampus(HippocampusConfig(capacity=10))
    for i in range(100):
        hippocampus.encode(z=[float(i)], actions=[], tags=SeedTags(reward=float(i % 5)))
    assert len(hippocampus) <= 10
    assert len(hippocampus) == 10
    assert hippocampus.total_encoded == 100


def test_high_priority_seeds_are_retained_over_bland_ones_when_full():
    hippocampus = Hippocampus(HippocampusConfig(capacity=5))
    for _ in range(5):
        hippocampus.encode(z=[0.0], actions=[], tags=SeedTags(reward=0.0))
    assert len(hippocampus) == 5
    assert hippocampus.total_evicted == 0

    salient = hippocampus.encode(z=[9.0], actions=["eat"], tags=SeedTags(reward=1.0, done=True))
    assert salient is not None
    assert hippocampus.total_evicted == 1
    assert len(hippocampus) == 5
    assert any(s.tags.done for s in hippocampus.seeds())


def test_a_bland_seed_is_skipped_once_the_store_is_full_of_better_seeds():
    hippocampus = Hippocampus(HippocampusConfig(capacity=3))
    for _ in range(3):
        hippocampus.encode(z=[0.0], actions=[], tags=SeedTags(reward=1.0, done=True))
    assert len(hippocampus) == 3

    bland = hippocampus.encode(z=[0.0], actions=[], tags=SeedTags(reward=0.0))
    assert bland is None
    assert hippocampus.total_skipped == 1
    assert len(hippocampus) == 3
    assert all(s.tags.reward == pytest.approx(1.0) for s in hippocampus.seeds())


def test_seeds_snapshot_is_sorted_highest_priority_first():
    hippocampus = Hippocampus(HippocampusConfig(capacity=20))
    for reward in (0.1, 0.9, 0.5, 0.3, 0.7):
        hippocampus.encode(z=[0.0], actions=[], tags=SeedTags(reward=reward))
    priorities = [seed.priority for seed in hippocampus.seeds()]
    assert priorities == sorted(priorities, reverse=True)


# --------------------------------------------------------------- online retrieval


def test_cosine_retrieval_returns_nearest_seed_only_after_surprise_gate():
    hippocampus = Hippocampus()
    nearest = hippocampus.encode(
        z=[1.0, 0.0],
        actions=["turn_left"],
        tags=SeedTags(reward=1.0),
        tick_index=1,
        cortex_version=4,
    )
    hippocampus.encode(
        z=[0.0, 1.0], actions=["turn_right"], tags=SeedTags(), cortex_version=4
    )
    config = HippocampalRetrievalConfig(
        top_k=1, min_similarity=0.8, min_surprise=0.5
    )

    assert hippocampus.retrieve(
        [0.99, 0.01], surprise=0.49, current_cortex_version=4, config=config
    ) == ()
    recalls = hippocampus.retrieve(
        [0.99, 0.01], surprise=0.5, current_cortex_version=4, config=config
    )
    assert len(recalls) == 1
    assert recalls[0].seed is nearest
    assert recalls[0].similarity > 0.99


def test_stale_and_unknown_cortex_provenance_are_excluded():
    hippocampus = Hippocampus()
    hippocampus.encode(
        z=[1.0, 0.0], actions=[], tags=SeedTags(), cortex_version=2
    )
    hippocampus.encode(z=[1.0, 0.0], actions=[], tags=SeedTags())
    current = hippocampus.encode(
        z=[1.0, 0.0], actions=[], tags=SeedTags(), cortex_version=3
    )

    recalls = hippocampus.retrieve(
        [1.0, 0.0], surprise=1.0, current_cortex_version=3
    )
    assert [recall.seed for recall in recalls] == [current]


def test_retrieval_prefers_consolidated_then_recent_seed_on_equal_similarity():
    hippocampus = Hippocampus()
    hippocampus.encode(
        z=[1.0], actions=[], tags=SeedTags(), tick_index=20, cortex_version=1
    )
    consolidated = hippocampus.encode(
        z=[1.0],
        actions=[],
        tags=SeedTags(),
        tick_index=10,
        cortex_version=1,
        consolidated=True,
    )
    recalls = hippocampus.retrieve(
        [1.0],
        surprise=1.0,
        current_cortex_version=1,
        config=HippocampalRetrievalConfig(top_k=1),
    )
    assert recalls[0].seed is consolidated


def test_reset_clears_the_store_and_counters():
    hippocampus = Hippocampus(HippocampusConfig(capacity=4))
    for _ in range(10):
        hippocampus.encode(z=[0.0], actions=[], tags=SeedTags(reward=1.0))
    assert len(hippocampus) > 0

    hippocampus.reset()
    assert len(hippocampus) == 0
    assert hippocampus.total_encoded == 0
    assert hippocampus.total_evicted == 0
    assert hippocampus.total_skipped == 0
    assert hippocampus.seeds() == ()


def test_state_dict_reports_counters():
    hippocampus = Hippocampus(HippocampusConfig(capacity=4))
    for i in range(6):
        # Strictly increasing reward so each new arrival, once full, always
        # outranks the current minimum -- otherwise an equal-priority seed
        # is skipped, not evicted (a tie never displaces anything).
        hippocampus.encode(z=[0.0], actions=[], tags=SeedTags(reward=float(i)))
    state = hippocampus.state_dict()
    assert state["capacity"] == 4
    assert state["size"] == 4
    assert state["total_encoded"] == 6
    assert state["total_evicted"] == 2


# --------------------------------------------------------------- wired into the loop


def test_a_recorded_run_populates_the_hippocampus_without_a_tick_stall(tmp_path):
    """Exit criteria (issue #96): encoding N ticks yields <= capacity seeds;
    wiring `Hippocampus.encode` into the loop's per-tick record path costs
    only a bounded handful of ops per tick, so a real episode still runs at
    its usual speed."""
    config = {"episode_ticks": 40, "world_size": 16, "max_mobs": 1}
    capacity = 15
    runtime_config = RuntimeConfig(
        episodes=1,
        seed=0,
        max_ticks_per_episode=40,
        record_dir=str(tmp_path),
        session_id="hippocampus-smoke",
        program_config=config,
    )
    runtime = CognitiveRuntime(
        program=MinecraftSurvivalBox(config=config),
        policy=ScriptedSurvivalPolicy(seed=0),
        config=runtime_config,
    )
    runtime.hippocampus = Hippocampus(HippocampusConfig(capacity=capacity))

    started = time.perf_counter()
    summaries = runtime.run()
    elapsed = time.perf_counter() - started

    assert len(runtime.hippocampus) > 0
    assert len(runtime.hippocampus) <= capacity
    assert runtime.hippocampus.total_encoded >= summaries[0].duration_ticks
    assert summaries[0].hippocampus_seeds == len(runtime.hippocampus)
    # Not a tight perf bound (shared/loaded CI hosts vary) -- just a sanity
    # check that per-tick encoding didn't turn a 40-tick episode into a
    # multi-second stall.
    assert elapsed < 30.0
