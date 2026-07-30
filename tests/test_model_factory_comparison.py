"""Regression coverage for paired Model Factory validation statistics (#223)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from cognitive_runtime.training.model_factory.comparison import (
    COMPARISON_FORMAT,
    compare_paired_episodes,
    paired_bootstrap_interval,
    paired_permutation_interval,
)


def test_intervals_are_exactly_reproducible_for_a_fixed_seed():
    deltas = [-0.8, -0.5, -0.2, 0.1, -0.4, -0.6]

    assert paired_bootstrap_interval(deltas, resamples=1_000, seed=41) == paired_bootstrap_interval(
        deltas, resamples=1_000, seed=41,
    )
    assert paired_permutation_interval(deltas, resamples=1_000, seed=41) == paired_permutation_interval(
        deltas, resamples=1_000, seed=41,
    )


def test_constant_shift_interval_covers_true_shift_and_excludes_zero():
    baseline = {f"episode-{index}": 10.0 + index for index in range(12)}
    candidate = {episode_id: value - 2.0 for episode_id, value in baseline.items()}

    report = compare_paired_episodes(candidate, baseline, resamples=2_000, seed=9)

    assert report.mean_delta == -2.0
    for interval in (report.bootstrap_interval, report.permutation_interval):
        assert interval[0] <= -2.0 <= interval[1]
        assert interval[1] < 0.0
    assert report.win_rate == 1.0


def test_zero_difference_intervals_cover_zero_across_repeated_seeds():
    baseline = {f"episode-{index}": float(index) for index in range(20)}
    candidate = dict(baseline)

    for seed in range(10):
        report = compare_paired_episodes(candidate, baseline, resamples=1_000, seed=seed)
        for interval in (report.bootstrap_interval, report.permutation_interval):
            assert interval[0] <= 0.0 <= interval[1]


def test_mismatched_episode_ids_raise_and_name_each_difference():
    with pytest.raises(
        ValueError,
        match=r"candidate_only=\['candidate-only'\].*baseline_only=\['baseline-only'\]",
    ):
        compare_paired_episodes(
            {"shared": 1.0, "candidate-only": 2.0},
            {"shared": 1.0, "baseline-only": 2.0},
        )


def test_existing_list_valued_evaluator_output_requires_and_uses_episode_ids():
    report = compare_paired_episodes(
        [0.3, 0.5, 0.7], [0.4, 0.6, 0.8],
        candidate_episode_ids=["one", "two", "three"],
        baseline_episode_ids=["one", "two", "three"],
        minimum_episode_count=3,
    )
    assert report.episode_ids == ("one", "three", "two")

    with pytest.raises(ValueError, match="need explicit candidate_episode_ids"):
        compare_paired_episodes([0.3, 0.5, 0.7], [0.4, 0.6, 0.8])


def test_too_few_pairs_is_explicitly_not_evaluable_with_reason():
    report = compare_paired_episodes(
        {"one": 0.5, "two": 0.7}, {"one": 0.6, "two": 0.8},
        minimum_episode_count=3,
    )

    assert report.not_evaluable
    assert report.reason == "requires at least 3 paired episodes; received 2"
    assert report.bootstrap_interval is None
    assert report.permutation_interval is None


def test_episode_ids_that_collide_as_json_object_keys_are_rejected():
    with pytest.raises(ValueError, match=r"collide once serialized as JSON object keys"):
        compare_paired_episodes(
            {1: 1.0, "1": 2.0, "two": 3.0},
            {1: 1.5, "1": 2.5, "two": 3.5},
            minimum_episode_count=3,
        )


def test_ratio_is_never_reported_without_the_raw_candidate_and_baseline_errors():
    report = compare_paired_episodes(
        {"episode-a": 2.0, "episode-b": 1.0, "episode-c": 3.0},
        {"episode-a": 4.0, "episode-b": 0.0, "episode-c": 6.0},
        minimum_episode_count=3,
    ).to_dict()

    assert report["format"] == COMPARISON_FORMAT
    per_episode = report["per_episode"]
    assert per_episode["candidate_raw_errors"] == {"episode-a": 2.0, "episode-b": 1.0, "episode-c": 3.0}
    assert per_episode["baseline_raw_errors"] == {"episode-a": 4.0, "episode-b": 0.0, "episode-c": 6.0}
    assert per_episode["candidate_over_baseline_ratio"] == [0.5, None, 0.5]


def test_module_imports_without_torch():
    """Run in a fresh interpreter so cached parent packages cannot mask torch."""
    script = """
import builtins
original_import = builtins.__import__
def deny_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('comparison must not import torch')
    return original_import(name, *args, **kwargs)
builtins.__import__ = deny_torch
import cognitive_runtime.training.model_factory.comparison
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
