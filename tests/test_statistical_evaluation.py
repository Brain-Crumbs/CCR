"""Statistical evaluation harness (issue #44).

Two tiers: pure unit tests on synthetic `EpisodeSummary` data (no simulation
cost), plus one sim-integration test per acceptance criterion --
"produces the statistical report from recorded sessions" and "a deliberately
degraded checkpoint is flagged as a regression by the harness in sim".
"""

from __future__ import annotations

import copy

import pytest

from cognitive_runtime.runtime.recorder import EpisodeSummary
from cognitive_runtime.training.statistical_evaluation import (
    MetricStats,
    _mean_ci,
    compare_statistics,
    compute_statistics,
    evaluate_recorded_sessions,
    flagged_regressions,
    format_comparison_report,
    format_statistics_report,
    run_statistical_evaluation,
)


def _summary(
    ticks: int,
    reward: float,
    *,
    termination_reason: str = "episode_ticks",
    death_reason=None,
    exploration_coverage=None,
    reward_by_tier=None,
    avg_prediction_error=None,
    avg_novelty=None,
) -> EpisodeSummary:
    program_stats = {}
    if exploration_coverage is not None:
        program_stats["exploration_coverage"] = exploration_coverage
    if reward_by_tier is not None:
        program_stats["reward_by_tier"] = reward_by_tier
    if death_reason is not None:
        program_stats["death_reason"] = death_reason
    return EpisodeSummary(
        session_id="s", episode_id="e", seed=0, policy_name="p",
        duration_ticks=ticks, total_reward=reward, success=False,
        termination_reason=termination_reason, program_stats=program_stats,
        avg_prediction_error=avg_prediction_error, avg_novelty=avg_novelty,
    )


# ------------------------------------------------------------------- _mean_ci


def test_mean_ci_empty_and_single_sample():
    empty = _mean_ci([])
    assert empty == MetricStats(n=0, mean=0.0, std=0.0, ci_low=0.0, ci_high=0.0, confidence=0.95)

    single = _mean_ci([5.0])
    assert single.n == 1
    assert single.mean == 5.0
    assert single.ci_low == single.ci_high == 5.0  # not enough data to bound uncertainty


def test_mean_ci_narrows_around_the_true_mean_with_known_variance():
    # Alternating +/-1 around 10: mean 10, stdev 1.xx; the 95% CI must
    # contain the true mean and be symmetric around it.
    values = [9.0, 11.0, 9.0, 11.0, 9.0, 11.0, 9.0, 11.0]
    stats = _mean_ci(values, confidence=0.95)
    assert stats.n == 8
    assert stats.mean == 10.0
    assert stats.ci_low < 10.0 < stats.ci_high
    assert stats.ci_high - stats.mean == pytest.approx(stats.mean - stats.ci_low, abs=1e-9)

    # A tighter (99%) confidence level must widen the interval.
    wider = _mean_ci(values, confidence=0.99)
    assert wider.ci_high - wider.ci_low > stats.ci_high - stats.ci_low


# --------------------------------------------------------------- compute_statistics


def test_compute_statistics_empty_summaries():
    stats = compute_statistics("p", [])
    assert stats.episodes == 0
    assert stats.survival_ticks.n == 0
    assert stats.death_rate == 0.0
    assert stats.death_causes == {}
    assert stats.exploration_coverage is None
    assert stats.reward_by_tier == {}


def test_compute_statistics_aggregates_survival_reward_deaths_coverage_and_tiers():
    summaries = [
        _summary(100, 1.0, termination_reason="death:starvation", death_reason="starvation",
                 exploration_coverage=3, reward_by_tier={"survival": 1.0, "capability": 0.0}),
        _summary(200, 2.0, termination_reason="death:zombie", death_reason="zombie",
                 exploration_coverage=5, reward_by_tier={"survival": 1.5, "capability": 0.5}),
        _summary(400, 4.0, termination_reason="episode_ticks",
                 exploration_coverage=7, reward_by_tier={"survival": 2.0}),
    ]
    stats = compute_statistics("scripted", summaries)

    assert stats.policy == "scripted"
    assert stats.episodes == 3
    assert stats.survival_ticks.mean == pytest.approx((100 + 200 + 400) / 3)
    assert stats.total_reward.mean == pytest.approx((1.0 + 2.0 + 4.0) / 3)
    assert stats.death_rate == pytest.approx(2 / 3, abs=1e-3)
    assert stats.death_causes == {
        "starvation": pytest.approx(1 / 3, abs=1e-3), "zombie": pytest.approx(1 / 3, abs=1e-3),
    }
    assert stats.exploration_coverage.mean == pytest.approx((3 + 5 + 7) / 3)
    # A tier missing from one episode's program_stats defaults to 0.0, not
    # dropped -- otherwise a tier that only fires sometimes would look
    # artificially high.
    assert stats.reward_by_tier["capability"].mean == pytest.approx((0.0 + 0.5 + 0.0) / 3, abs=1e-5)
    assert stats.reward_by_tier["survival"].mean == pytest.approx((1.0 + 1.5 + 2.0) / 3, abs=1e-5)


def test_compute_statistics_prediction_error_and_novelty_are_none_when_absent():
    summaries = [_summary(100, 1.0), _summary(200, 2.0)]
    stats = compute_statistics("p", summaries)
    assert stats.prediction_error is None
    assert stats.novelty is None

    summaries_with_pe = [
        _summary(100, 1.0, avg_prediction_error=0.1, avg_novelty=0.2),
        _summary(200, 2.0, avg_prediction_error=0.3, avg_novelty=0.4),
    ]
    stats2 = compute_statistics("p", summaries_with_pe)
    assert stats2.prediction_error.mean == pytest.approx(0.2)
    assert stats2.novelty.mean == pytest.approx(0.3)


# --------------------------------------------------------------- compare_statistics


def test_compare_statistics_flags_regression_on_non_overlapping_worse_interval():
    baseline = compute_statistics("baseline", [
        _summary(1000, 10.0), _summary(1010, 10.5), _summary(990, 9.5),
        _summary(1005, 10.2), _summary(995, 9.8), _summary(1000, 10.0),
    ])
    candidate = compute_statistics("candidate", [
        _summary(100, -5.0), _summary(110, -4.5), _summary(90, -5.5),
        _summary(105, -4.8), _summary(95, -5.2), _summary(100, -5.0),
    ])

    comparisons = compare_statistics(baseline, candidate)
    regressions = flagged_regressions(comparisons)
    regressed_metrics = {c.metric for c in regressions}
    assert "survival_ticks" in regressed_metrics
    assert "total_reward" in regressed_metrics
    for c in comparisons:
        if c.metric in regressed_metrics:
            assert c.regressed
            assert c.direction == "regressed"


def test_compare_statistics_flags_improvement_the_other_direction():
    baseline = compute_statistics("baseline", [_summary(100, 1.0) for _ in range(6)])
    candidate = compute_statistics("candidate", [_summary(500, 8.0) for _ in range(6)])
    # Identical-valued episodes have zero variance; give them a hair of spread
    # so the CI is non-degenerate but still tight and non-overlapping.
    baseline = compute_statistics(
        "baseline", [_summary(100 + i, 1.0 + 0.01 * i) for i in range(6)]
    )
    candidate = compute_statistics(
        "candidate", [_summary(500 + i, 8.0 + 0.01 * i) for i in range(6)]
    )

    comparisons = compare_statistics(baseline, candidate)
    by_metric = {c.metric: c for c in comparisons}
    assert by_metric["survival_ticks"].direction == "improved"
    assert by_metric["total_reward"].direction == "improved"
    assert not flagged_regressions(comparisons)


def test_compare_statistics_overlapping_intervals_are_not_significant():
    summaries = [_summary(100 + i, 1.0 + 0.1 * i) for i in range(6)]
    baseline = compute_statistics("baseline", summaries)
    candidate = compute_statistics("candidate", summaries)  # identical distribution
    comparisons = compare_statistics(baseline, candidate)
    assert all(c.direction == "no_significant_difference" for c in comparisons)
    assert not flagged_regressions(comparisons)


def test_compare_statistics_single_episode_is_never_significant():
    baseline = compute_statistics("baseline", [_summary(1000, 10.0)])
    candidate = compute_statistics("candidate", [_summary(1, -10.0)])
    comparisons = compare_statistics(baseline, candidate)
    assert all(c.direction == "no_significant_difference" for c in comparisons)


def test_compare_statistics_restricts_to_requested_metrics():
    baseline = compute_statistics("baseline", [_summary(100 + i, 1.0) for i in range(4)])
    candidate = compute_statistics("candidate", [_summary(1 + i, 1.0) for i in range(4)])
    comparisons = compare_statistics(baseline, candidate, metrics=["survival_ticks"])
    assert [c.metric for c in comparisons] == ["survival_ticks"]


def test_reports_are_json_safe_and_human_readable():
    stats = compute_statistics("p", [
        _summary(100, 1.0, exploration_coverage=2, reward_by_tier={"survival": 1.0}),
        _summary(120, 1.2, exploration_coverage=4, reward_by_tier={"survival": 1.2}),
    ])
    import json
    json.dumps(stats.to_dict())  # must serialize for checkpoint/dashboard use

    report = format_statistics_report([stats])
    assert "p" in report
    assert "reward by tier" in report

    comparisons = compare_statistics(stats, stats)
    json.dumps([c.to_dict() for c in comparisons])
    text = format_comparison_report(comparisons)
    assert "no_significant_difference" in text
    assert format_comparison_report([]) == "(no comparable metrics)"


# ------------------------------------------------------------- sim integration


def _small_config():
    return {"episode_ticks": 300, "world_size": 32, "day_length": 800, "start_time": 300}


def test_run_statistical_evaluation_and_recorded_sessions_report_agree(tmp_path):
    """The live-run harness and the recorded-sessions loader must agree on
    the same policy/seeds -- "produces the statistical report from recorded
    sessions" (acceptance criterion)."""
    from cognitive_runtime.policies import RandomPolicy
    from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
    from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox

    cfg = _small_config()
    record_dir = tmp_path / "sessions"

    live_stats = run_statistical_evaluation(
        program_factory=lambda: MinecraftSurvivalBox(config=cfg),
        policy_factory=lambda: RandomPolicy(ACTION_SPACE, seed=0),
        episodes=5, seed=1, max_ticks=cfg["episode_ticks"],
        record_dir=str(record_dir), session_id="random-baseline",
    )
    assert live_stats.episodes == 5
    assert live_stats.policy == "random"

    by_group = evaluate_recorded_sessions(str(record_dir))
    assert ("-", "random") in by_group
    recorded_stats = by_group[("-", "random")]

    assert recorded_stats.episodes == live_stats.episodes
    assert recorded_stats.survival_ticks == live_stats.survival_ticks
    assert recorded_stats.total_reward == live_stats.total_reward


def test_evaluate_recorded_sessions_empty_directory(tmp_path):
    assert evaluate_recorded_sessions(str(tmp_path / "does-not-exist")) == {}


def test_scripted_beats_random_statistically_in_sim():
    """A concrete, deterministic (fixed seeds, simulated backend) instance of
    the harness catching a real behavioral gap: the scripted survival policy
    significantly outsurvives/outscores random on identical matched seeds."""
    from cognitive_runtime.policies import RandomPolicy, ScriptedSurvivalPolicy
    from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE
    from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox

    cfg = _small_config()
    random_stats = run_statistical_evaluation(
        program_factory=lambda: MinecraftSurvivalBox(config=cfg),
        policy_factory=lambda: RandomPolicy(ACTION_SPACE, seed=0),
        episodes=10, seed=0, max_ticks=cfg["episode_ticks"],
    )
    scripted_stats = run_statistical_evaluation(
        program_factory=lambda: MinecraftSurvivalBox(config=cfg),
        policy_factory=lambda: ScriptedSurvivalPolicy(seed=1),
        episodes=10, seed=0, max_ticks=cfg["episode_ticks"],
    )

    comparisons = compare_statistics(random_stats, scripted_stats)
    regressions = flagged_regressions(comparisons)
    assert not regressions  # scripted must not look worse than random on any core metric

    by_metric = {c.metric: c for c in comparisons}
    assert by_metric["total_reward"].direction == "improved"
    assert by_metric["exploration_coverage"].direction == "improved"


# ------------------------------------------------------- degraded checkpoint


def test_noised_actor_critic_checkpoint_is_flagged_as_regression_in_sim():
    """Acceptance criterion: "a deliberately degraded checkpoint (e.g. noised
    weights) is flagged as a regression by the harness in sim". Trains a
    small actor/critic (mirroring
    training.actor_critic_acceptance.run_simulated_actor_critic_acceptance's
    budget), then evaluates a heavily weight-noised copy against the trained
    baseline on identical eval seeds."""
    pytest.importorskip("torch")
    import torch

    from cognitive_runtime.policies.actor_critic import ActorCriticPolicy
    from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
    from cognitive_runtime.training.actor_critic_acceptance import (
        DEFAULT_ACCEPTANCE_CONFIG,
        _new_stack,
        _run_actor_critic,
    )

    cfg = dict(DEFAULT_ACCEPTANCE_CONFIG)
    action_keys, policy_model, critic_model, optimizer = _new_stack(
        cfg, 1, lr=1e-2, entropy_coef=0.05
    )
    _run_actor_critic(policy_model, critic_model, optimizer, action_keys, cfg,
                       episodes=20, seed=100, train=True)

    noised_policy_model = copy.deepcopy(policy_model)
    with torch.no_grad():
        for param in noised_policy_model.parameters():
            param.add_(torch.randn_like(param) * 8.0)

    eval_episodes, eval_seed = 6, 500

    def _policy_factory(model):
        def make():
            return ActorCriticPolicy(
                model, critic_model, action_keys,
                action_space=MinecraftSurvivalBox(config=cfg).metadata().action_space,
                training=False, seed=eval_seed,
            )
        return make

    baseline_stats = run_statistical_evaluation(
        program_factory=lambda: MinecraftSurvivalBox(config=cfg),
        policy_factory=_policy_factory(policy_model),
        episodes=eval_episodes, seed=eval_seed, max_ticks=cfg["episode_ticks"],
    )
    noised_stats = run_statistical_evaluation(
        program_factory=lambda: MinecraftSurvivalBox(config=cfg),
        policy_factory=_policy_factory(noised_policy_model),
        episodes=eval_episodes, seed=eval_seed, max_ticks=cfg["episode_ticks"],
    )

    comparisons = compare_statistics(baseline_stats, noised_stats)
    regressions = flagged_regressions(comparisons)
    assert regressions, (
        f"expected the noised checkpoint to be flagged as a regression; "
        f"comparisons were: {[str(c) for c in comparisons]}"
    )
