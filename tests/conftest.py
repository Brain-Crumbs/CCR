"""Default test-suite policy for the V2/Crafter development loop.

The deferred tests remain collected and can be deliberately run with
``--run-deferred``.  They cover retired learners, migration-only world models,
or Minecraft-graduation integrations; none should block Cortex or Model
Factory iteration by default.
"""

from __future__ import annotations

from pathlib import Path

import pytest


DEFERRED_MODULES = frozenset(
    {
        # Retired online-Q learner and optional actor/critic experiment.
        "test_online_q.py",
        "test_online_q_acceptance.py",
        "test_online_q_live_rollout.py",
        "test_online_q_policy.py",
        "test_online_q_runtime.py",
        "test_actor_critic_acceptance.py",
        "test_actor_critic_models.py",
        "test_actor_critic_policy.py",
        "test_actor_critic_runtime.py",
        # Pre-Cortex MLP world-model implementations and their end-to-end runs.
        "test_multi_horizon_world_model.py",
        "test_world_model_phase_d.py",
        # Minecraft graduation, remote bridge, and legacy runtime integrations.
        "test_action_registry.py",
        "test_cli_cortex_consolidation.py",
        "test_policies.py",
        "test_program.py",
        "test_program_streams.py",
        "test_realtime_streams.py",
        "test_remote_backend.py",
        "test_reward_engine.py",
        "test_reward_profile.py",
        "test_rewards.py",
        "test_runtime.py",
        "test_runtime_streams.py",
        "test_tools.py",
        "test_training.py",
    }
)


DEFERRED_NODEIDS = frozenset(
    {
        # Compatibility-only readers for retired checkpoint formats.
        "tests/test_organism_name.py::test_legacy_nameless_checkpoint_still_loads",
        "tests/test_pixel_stream_encoder.py::test_vision_bc_model_loads_legacy_cnn_key_bundles",
        "tests/test_spatial_visual_model.py::test_legacy_global_pool_checkpoint_has_actionable_error",
        # Optional actor/critic integrations outside the dedicated modules.
        "tests/test_live_run_protocol.py::test_actor_critic_policy_publishes_value_estimate_stream",
        "tests/test_statistical_evaluation.py::test_noised_actor_critic_checkpoint_is_flagged_as_regression_in_sim",
        "tests/test_voluntary_motor.py::test_policy_controller_wires_the_actor_critic_policy_head",
    }
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-deferred",
        action="store_true",
        default=False,
        help="run deferred legacy, experimental, and Minecraft-graduation tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-deferred"):
        return

    skip = pytest.mark.skip(
        reason=(
            "deferred from the default V2/Crafter suite; "
            "pass --run-deferred to run this legacy, experimental, or "
            "Minecraft-graduation coverage"
        )
    )
    for item in items:
        module = Path(str(item.path)).name
        if module in DEFERRED_MODULES or item.nodeid in DEFERRED_NODEIDS:
            item.add_marker(skip)
