"""Smoke acceptance for the actor/critic online learner (issue #29 acceptance
criterion: "actor/critic >= random baseline on identical seeds"). The full
actor-critic-vs-online-Q gate is issue #31; this only checks against random,
mirroring tests/test_online_q_acceptance.py's shape.

One test, two runs: proves both "beats random" and "deterministic per seed"
without paying for a third ~1200-tick*20-episode simulated training run.
"""

import pytest

pytest.importorskip("torch")

from cognitive_runtime.training.actor_critic_acceptance import (  # noqa: E402
    run_simulated_actor_critic_acceptance,
)


def test_simulated_actor_critic_beats_random_reproducibly():
    first = run_simulated_actor_critic_acceptance()
    second = run_simulated_actor_critic_acceptance()

    assert first.accepted
    assert first.acceptance_metric in ("reward", "ticks")
    if first.acceptance_metric == "reward":
        assert first.actor_critic_eval.total_reward > first.random_eval.total_reward
    else:
        assert first.actor_critic_eval.total_ticks > first.random_eval.total_ticks
    assert first.training_steps > 0

    assert first.actor_critic_eval == second.actor_critic_eval
    assert first.training_steps == second.training_steps
