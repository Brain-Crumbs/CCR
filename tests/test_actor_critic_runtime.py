"""Runtime integration for :class:`ActorCriticPolicy`/:class:`ActorCriticLearner`
wired directly into :class:`CognitiveRuntime` (issue #29). The CLI's own
``--policy actor-critic``/``--async-trainer`` online-learner path was retired
by issue #175 (the predictive cortex is the only online learner now;
actor/critic survives only as a selectable, inference-only ``motor.voluntary``
A/B controller -- see ``motor/policy.py`` and its tests), so this file no
longer drives the CLI; it keeps the one thing worth covering here: a run
interrupted mid-episode still leaves a valid, resumable checkpoint.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.core.streams import TemporalFusion, default_encoder_registry  # noqa: E402
from cognitive_runtime.neural import ActorCriticOptimizer, MLPPolicyModel, MLPValueModel  # noqa: E402
from cognitive_runtime.neural.checkpoint import NeuralAgentCheckpoint  # noqa: E402
from cognitive_runtime.policies.actor_critic import (  # noqa: E402
    ActorCriticLearner,
    ActorCriticPolicy,
    world_feature_width,
)
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox  # noqa: E402
from cognitive_runtime.runtime.config import RuntimeConfig  # noqa: E402
from cognitive_runtime.runtime.loop import CognitiveRuntime  # noqa: E402

FAST_CONFIG = {"episode_ticks": 30, "world_size": 32}


class InterruptingActorCriticLearner(ActorCriticLearner):
    def update(self, window):
        super().update(window)
        if self.observed_ticks >= 5:
            raise KeyboardInterrupt


def test_interrupted_run_leaves_valid_actor_critic_checkpoint(tmp_path):
    program = MinecraftSurvivalBox(config=FAST_CONFIG)
    fusion = TemporalFusion(program.stream_catalog(), default_encoder_registry())
    action_space = list(program.metadata().action_space)
    action_keys = [action.key() for action in action_space]
    wf_width = world_feature_width(action_keys)

    torch.manual_seed(0)
    policy_model = MLPPolicyModel(
        fusion.width, wf_width, len(action_keys),
        hidden_dim=16, layout_hash=fusion.layout_hash, action_keys=action_keys,
    )
    critic_model = MLPValueModel(
        fusion.width, wf_width, hidden_dim=16, layout_hash=fusion.layout_hash, action_keys=action_keys,
    )
    optimizer = ActorCriticOptimizer(policy_model, critic_model, lr=0.05, seed=0)
    policy = ActorCriticPolicy(policy_model, critic_model, action_keys, action_space=action_space, training=True)
    path = tmp_path / "interrupted-actor-critic.pt"
    checkpoint = NeuralAgentCheckpoint(
        str(path), layout_hash=fusion.layout_hash, action_keys=action_keys, online_optimizer=optimizer,
    )
    learner = InterruptingActorCriticLearner(optimizer, policy, training=True, checkpoint=checkpoint)
    runtime = CognitiveRuntime(
        program=program,
        policy=policy,
        learner=learner,
        config=RuntimeConfig(
            episodes=1, seed=0, max_ticks_per_episode=FAST_CONFIG["episode_ticks"],
            record=False, program_config=FAST_CONFIG,
        ),
    )

    with pytest.raises(KeyboardInterrupt):
        runtime.run()

    assert path.exists()
    with open(str(path) + ".json", encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["training_stats"]["last_checkpoint_reason"] == "keyboard_interrupt"
    assert metadata["training_stats"]["training_ticks"] > 0
