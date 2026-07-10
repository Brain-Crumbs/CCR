"""Phase-E concrete actor/critic modules (issue #29): shapes, mutation, and
checkpoint round-tripping for :class:`MLPPolicyModel`, :class:`MLPValueModel`,
and :class:`ActorCriticOptimizer`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.neural import (  # noqa: E402
    ActorCriticOptimizer,
    MLPPolicyModel,
    MLPValueModel,
    MLPWorldModel,
)

FUSED_WIDTH = 5
WORLD_FEATURE_WIDTH = 4
N_ACTIONS = 3


def _policy(**kwargs):
    return MLPPolicyModel(FUSED_WIDTH, WORLD_FEATURE_WIDTH, N_ACTIONS, hidden_dim=8, **kwargs)


def _critic(**kwargs):
    return MLPValueModel(FUSED_WIDTH, WORLD_FEATURE_WIDTH, hidden_dim=8, **kwargs)


def _batch(batch_size=4, n_actions=N_ACTIONS):
    actions = torch.randint(0, n_actions, (batch_size,))
    return {
        "fused_latent": torch.randn(batch_size, FUSED_WIDTH),
        "world_features": torch.randn(batch_size, WORLD_FEATURE_WIDTH),
        "action_onehot": torch.nn.functional.one_hot(actions, n_actions).float(),
        "reward": torch.randn(batch_size),
        "next_fused_latent": torch.randn(batch_size, FUSED_WIDTH),
        "next_world_features": torch.randn(batch_size, WORLD_FEATURE_WIDTH),
        "done": torch.zeros(batch_size),
    }


def test_policy_model_action_space_size_and_logit_shape():
    policy = _policy()
    assert policy.action_space_size() == N_ACTIONS
    logits = policy(torch.randn(4, FUSED_WIDTH), torch.randn(4, WORLD_FEATURE_WIDTH))
    assert logits.shape == (4, N_ACTIONS)


def test_policy_model_rejects_wrong_shaped_inputs():
    policy = _policy()
    with pytest.raises(ValueError, match="fused_latent"):
        policy(torch.randn(4, FUSED_WIDTH + 1), torch.randn(4, WORLD_FEATURE_WIDTH))
    with pytest.raises(ValueError, match="world_features"):
        policy(torch.randn(4, FUSED_WIDTH), torch.randn(4, WORLD_FEATURE_WIDTH + 1))


def test_value_model_scalar_output_shape():
    critic = _critic()
    value = critic(torch.randn(6, FUSED_WIDTH), torch.randn(6, WORLD_FEATURE_WIDTH))
    assert value.shape == (6,)


def test_policy_and_value_checkpoint_metadata_round_trip_architecture():
    policy = _policy(layout_hash="layout-a", action_keys=["NULL", "JUMP", "ATTACK"])
    meta = policy.checkpoint_metadata()
    rebuilt = MLPPolicyModel(
        meta["fused_width"], meta["world_feature_width"], meta["n_actions"],
        hidden_dim=meta["hidden_dim"], depth=meta["depth"], dropout=meta["dropout"],
        layout_hash=meta["layout_hash"], action_keys=meta["action_keys"],
    )
    rebuilt.load_state_dict(policy.state_dict())
    x = (torch.randn(2, FUSED_WIDTH), torch.randn(2, WORLD_FEATURE_WIDTH))
    assert torch.allclose(policy(*x), rebuilt(*x))


def test_actor_critic_optimizer_step_mutates_policy_and_critic_and_returns_metrics():
    policy, critic = _policy(), _critic()
    optimizer = ActorCriticOptimizer(policy, critic, lr=0.05, seed=0)
    before_policy = [p.clone() for p in policy.parameters()]
    before_critic = [p.clone() for p in critic.parameters()]

    metrics = optimizer.step(_batch())

    assert set(metrics) >= {"policy_loss", "value_loss", "entropy", "grad_norm"}
    assert any(
        not torch.equal(a, b) for a, b in zip(before_policy, policy.parameters())
    )
    assert any(
        not torch.equal(a, b) for a, b in zip(before_critic, critic.parameters())
    )
    assert optimizer.step_count == 1


def test_actor_critic_optimizer_syncs_target_critic_toward_critic():
    policy, critic = _policy(), _critic()
    optimizer = ActorCriticOptimizer(policy, critic, lr=0.1, target_tau=1.0, seed=0)
    optimizer.step(_batch())
    for target_param, param in zip(optimizer.target_critic.parameters(), critic.parameters()):
        assert torch.allclose(target_param, param)


def test_actor_critic_optimizer_jointly_trains_world_model_when_provided():
    policy, critic = _policy(), _critic()
    world_model = MLPWorldModel(FUSED_WIDTH, N_ACTIONS, hidden_dim=8)
    optimizer = ActorCriticOptimizer(policy, critic, world_model=world_model, lr=0.05, seed=0)
    before = [p.clone() for p in world_model.parameters()]

    metrics = optimizer.step(_batch())

    assert metrics["world_model_loss"] > 0.0
    assert any(not torch.equal(a, b) for a, b in zip(before, world_model.parameters()))


def test_actor_critic_optimizer_state_dict_round_trip_restores_behavior():
    policy, critic = _policy(), _critic()
    optimizer = ActorCriticOptimizer(policy, critic, lr=0.05, seed=0)
    optimizer.step(_batch(batch_size=8))
    state = optimizer.state_dict()

    restored_policy, restored_critic = _policy(), _critic()
    restored_optimizer = ActorCriticOptimizer(restored_policy, restored_critic, lr=0.05, seed=1)
    restored_optimizer.load_state_dict(state)

    assert restored_optimizer.step_count == optimizer.step_count
    x = (torch.randn(3, FUSED_WIDTH), torch.randn(3, WORLD_FEATURE_WIDTH))
    assert torch.allclose(policy(*x), restored_policy(*x))
    assert torch.allclose(critic(*x), restored_critic(*x))
    for target_param, restored_target_param in zip(
        optimizer.target_critic.parameters(), restored_optimizer.target_critic.parameters()
    ):
        assert torch.allclose(target_param, restored_target_param)


def test_actor_critic_optimizer_normalizes_reward_and_updates_running_stats():
    policy, critic = _policy(), _critic()
    optimizer = ActorCriticOptimizer(policy, critic, lr=0.01, seed=0)
    batch = _batch(batch_size=16)
    batch["reward"] = torch.full((16,), 100.0)
    optimizer.step(batch)
    assert optimizer.reward_normalizer.mean > 0.0


def test_policy_model_and_value_model_are_not_directly_instantiable():
    from cognitive_runtime.neural import PolicyModel, ValueModel

    with pytest.raises(TypeError):
        PolicyModel()
    with pytest.raises(TypeError):
        ValueModel()
