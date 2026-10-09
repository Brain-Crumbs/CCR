"""Bounded opt-in preservation of production-size legacy Crafter replay."""
import pytest

pytestmark = pytest.mark.market_extended


def test_default_size_crafter_record_replay(tmp_path):
    from cognitive_runtime.policies import RandomPolicy
    from cognitive_runtime.programs.crafter.actions import ACTION_SPACE
    from cognitive_runtime.programs.crafter.adapter import CrafterWorld
    from cognitive_runtime.runtime.config import RuntimeConfig
    from cognitive_runtime.runtime.loop import CognitiveRuntime
    from cognitive_runtime.tools.replay_runner import replay_session
    # No area or renderer-size override: preserve the production defaults.
    program_config = {"episode_ticks": 3}
    program = CrafterWorld(config=program_config)
    program.reset(seed=297)
    assert tuple(program._env._world.area) == (64, 64)
    assert program._env.render().shape == (64, 64, 3)
    config = RuntimeConfig(episodes=1, seed=297, max_ticks_per_episode=3,
                           record_dir=str(tmp_path), session_id="default-size",
                           program_config=program_config)
    CognitiveRuntime(program=program, policy=RandomPolicy(ACTION_SPACE, seed=297), config=config).run()
    results = replay_session(str(tmp_path / "default-size"))
    assert len(results) == 1
    assert results[0].matched, results[0]
    assert results[0].ticks_replayed == 3
