"""Evaluation gate 1 (issue #31): the actor/critic beats random on identical
seeds.

Torch-gated/skippable like tests/test_neural.py.  The slow test asserts only
gate 1 (the hard requirement) on the default gate config, mirroring
tests/test_actor_critic_acceptance.py's budget, plus the two data paths the
issue's acceptance criteria call out -- recorded eval sessions for the
dashboard and the gate report in the checkpoint bundle's training stats.
Gates 2-3 are reported, not asserted: they are the documented manual command
(the CLI ``evaluation-gates`` subcommand, docs/history/online-learning.md), so a shift in
their outcome does not fail CI.
"""

import json

import pytest

pytest.importorskip("torch")

from cognitive_runtime.neural import read_checkpoint_metadata  # noqa: E402
from cognitive_runtime.training.online_q_acceptance import EvaluationSummary  # noqa: E402
from cognitive_runtime.training.evaluation_gates import (  # noqa: E402
    DEFAULT_GATE_CONFIG,
    GATE_METRIC,
    EvaluationGateResult,
    run_evaluation_gates,
)


def test_phase_e_gate1_beats_random_with_recordings_and_checkpoint(tmp_path):
    record_dir = tmp_path / "sessions"
    checkpoint = tmp_path / "actor-critic.pt"

    result = run_evaluation_gates(
        config=DEFAULT_GATE_CONFIG,
        record_dir=str(record_dir),
        checkpoint_path=str(checkpoint),
    )

    # Gate 1 (hard requirement): actor/critic > random on identical seeds. The
    # signal is survival -- the trained policy reaches the end of the lethal
    # night episodes that kill random before the episode budget is spent.
    assert result.gate1_beats_random
    assert (
        result.summaries["actor-critic"].total_ticks
        > result.summaries["random"].total_ticks
    )
    assert result.actor_critic_training_steps > 0
    assert result.online_q_training_ticks > 0

    # Gates 2-3 are reported, not asserted (documented manual command).
    assert isinstance(result.gate2_beats_linear_q, bool)
    assert result.gate3_reproducible is None  # check_reproducible defaulted off

    # All four policies were evaluated on the same seeds.
    assert set(result.summaries) == {"actor-critic", "online", "scripted", "random"}

    # Recorded eval sessions exist for dashboard/viewer inspection.
    assert (record_dir / "eval-gate-actor-critic").is_dir()
    assert (record_dir / "eval-gate-online-q").is_dir()
    assert (record_dir / "eval-gate-scripted").is_dir()
    assert (record_dir / "eval-gate-random").is_dir()

    # Gate results written into the checkpoint bundle's training stats (#20).
    meta = read_checkpoint_metadata(str(checkpoint))
    gate_report = meta["training_stats"]["evaluation_gates"]
    assert gate_report["gates"]["actor_critic_gt_random"] is True
    assert gate_report["metric"] == GATE_METRIC


def _summary(name: str, reward: float, ticks: int) -> EvaluationSummary:
    return EvaluationSummary(
        policy=name,
        total_reward=reward,
        total_ticks=ticks,
        average_reward=round(reward / 2, 6),
        average_ticks=ticks / 2,
        termination_reasons=["timeout", "timeout"],
    )


def test_gate_summary_is_json_safe_and_structured():
    """The gate report embedded in checkpoint training stats must be JSON-safe
    and carry every gate/eval field -- exercised without paying for training."""
    result = EvaluationGateResult(
        summaries={
            "actor-critic": _summary("actor-critic", -4.63, 2400),
            "online": _summary("online", 1.0, 2400),
            "scripted": _summary("scripted", 18.0, 2400),
            "random": _summary("random", 12.76, 2337),
        },
        gate1_beats_random=True,
        gate2_beats_linear_q=False,
        gate3_reproducible=True,
        metric=GATE_METRIC,
        actor_critic_training_steps=123,
        online_q_training_ticks=456,
        config=dict(DEFAULT_GATE_CONFIG),
        seeds={"model": 1, "train": 100, "eval": 500},
        curriculum=None,
    )

    # accepted requires both ordering gates; gate 2 is False here.
    assert result.accepted is False

    summary = result.gate_summary()
    assert summary["issue"] == 31
    assert summary["metric"] == GATE_METRIC
    assert summary["gates"] == {
        "actor_critic_gt_random": True,
        "actor_critic_gt_linear_q": False,
        "reproducible_improvement": True,
    }
    assert summary["training"] == {"actor_critic_steps": 123, "online_q_ticks": 456}
    assert set(summary["eval"]) == {"actor-critic", "online", "scripted", "random"}
    assert summary["eval"]["random"]["total_ticks"] == 2337

    json.dumps(summary)  # must serialize for the checkpoint sidecar
