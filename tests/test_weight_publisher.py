"""Versioned weight publication between trainer and actor (issue #37):
"trainer publishes versioned policy snapshots; the actor swaps them in
atomically between ticks"."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

from cognitive_runtime.neural.checkpoint import NeuralAgentCheckpoint  # noqa: E402
from cognitive_runtime.neural.optimizer import ActorCriticOptimizer  # noqa: E402
from cognitive_runtime.neural.policy import MLPPolicyModel  # noqa: E402
from cognitive_runtime.neural.value import MLPValueModel  # noqa: E402
from cognitive_runtime.neural.weight_publisher import WeightPublisher, WeightSubscriber  # noqa: E402

FUSED_WIDTH, WORLD_FEATURE_WIDTH, N_ACTIONS = 6, 4, 3
LAYOUT_HASH = "test-layout"
ACTION_KEYS = ["a", "b", "c"]


def _trainer_side(path, seed=0):
    torch.manual_seed(seed)
    policy = MLPPolicyModel(FUSED_WIDTH, WORLD_FEATURE_WIDTH, N_ACTIONS)
    critic = MLPValueModel(FUSED_WIDTH, WORLD_FEATURE_WIDTH)
    optimizer = ActorCriticOptimizer(policy, critic, lr=1e-2)
    bundle = NeuralAgentCheckpoint(
        path, layout_hash=LAYOUT_HASH, action_keys=ACTION_KEYS,
        policy=policy, critic=critic, optimizers={"adam": optimizer.optimizer},
    )
    return policy, critic, optimizer, bundle


def _actor_side(path, seed=999):
    torch.manual_seed(seed)
    policy = MLPPolicyModel(FUSED_WIDTH, WORLD_FEATURE_WIDTH, N_ACTIONS)
    critic = MLPValueModel(FUSED_WIDTH, WORLD_FEATURE_WIDTH)
    bundle = NeuralAgentCheckpoint(
        path, layout_hash=LAYOUT_HASH, action_keys=ACTION_KEYS, policy=policy, critic=critic,
    )
    return policy, critic, WeightSubscriber(path=str(path), bundle=bundle)


def _random_batch(batch_size=4):
    return {
        "fused_latent": torch.randn(batch_size, FUSED_WIDTH),
        "world_features": torch.randn(batch_size, WORLD_FEATURE_WIDTH),
        "action_onehot": torch.nn.functional.one_hot(
            torch.randint(0, N_ACTIONS, (batch_size,)), N_ACTIONS
        ).float(),
        "reward": torch.randn(batch_size),
        "next_fused_latent": torch.randn(batch_size, FUSED_WIDTH),
        "next_world_features": torch.randn(batch_size, WORLD_FEATURE_WIDTH),
        "done": torch.zeros(batch_size),
    }


def test_subscriber_sees_nothing_before_first_publish(tmp_path):
    path = str(tmp_path / "trainer.pt")
    _actor_policy, _actor_critic, subscriber = _actor_side(path)
    assert subscriber.poll_version() is None
    assert subscriber.maybe_reload() is None


def test_subscriber_hot_swaps_weights_in_place_on_publish(tmp_path):
    path = str(tmp_path / "trainer.pt")
    _policy, _critic, optimizer, trainer_bundle = _trainer_side(path)
    publisher = WeightPublisher(trainer_bundle)
    actor_policy, actor_critic, subscriber = _actor_side(path)

    before_policy = [p.clone() for p in actor_policy.parameters()]
    before_critic = [p.clone() for p in actor_critic.parameters()]

    trainer_bundle.training_ticks = optimizer.step_count
    v1 = publisher.publish(reason="t1")
    reloaded = subscriber.maybe_reload()

    assert reloaded == v1
    # Same module objects (in-place swap), different weights.
    assert any(
        not torch.equal(a, b) for a, b in zip(before_policy, actor_policy.parameters())
    )
    assert any(
        not torch.equal(a, b) for a, b in zip(before_critic, actor_critic.parameters())
    )


def test_republishing_without_a_new_step_does_not_trigger_a_reload(tmp_path):
    path = str(tmp_path / "trainer.pt")
    _policy, _critic, optimizer, trainer_bundle = _trainer_side(path)
    publisher = WeightPublisher(trainer_bundle)
    _actor_policy, _actor_critic, subscriber = _actor_side(path)

    trainer_bundle.training_ticks = optimizer.step_count
    publisher.publish(reason="t1")
    assert subscriber.maybe_reload() is not None

    # No optimizer step happened; training_ticks is unchanged, so this is a
    # no-op republish -- the actor must not reload (nothing actually changed).
    publisher.publish(reason="t2-noop")
    assert subscriber.maybe_reload() is None


def test_new_gradient_step_bumps_the_version_and_reloads_again(tmp_path):
    path = str(tmp_path / "trainer.pt")
    policy, critic, optimizer, trainer_bundle = _trainer_side(path)
    publisher = WeightPublisher(trainer_bundle)
    _actor_policy, _actor_critic, subscriber = _actor_side(path)

    trainer_bundle.training_ticks = optimizer.step_count
    v1 = publisher.publish(reason="t1")
    subscriber.maybe_reload()
    assert subscriber.maybe_reload() is None  # nothing new yet -> a skipped poll

    optimizer.step(_random_batch())
    trainer_bundle.training_ticks = optimizer.step_count
    v2 = publisher.publish(reason="t2")

    assert v2 > v1
    assert subscriber.maybe_reload() == v2
    assert subscriber.stats()["reload_count"] == 2
    assert subscriber.stats()["skipped_count"] >= 1


def test_subscriber_rejects_incompatible_action_space(tmp_path):
    path = str(tmp_path / "trainer.pt")
    _policy, _critic, optimizer, trainer_bundle = _trainer_side(path)
    WeightPublisher(trainer_bundle).publish(reason="t1")

    torch.manual_seed(0)
    mismatched_policy = MLPPolicyModel(FUSED_WIDTH, WORLD_FEATURE_WIDTH, N_ACTIONS + 1)
    mismatched_critic = MLPValueModel(FUSED_WIDTH, WORLD_FEATURE_WIDTH)
    mismatched_bundle = NeuralAgentCheckpoint(
        path, layout_hash=LAYOUT_HASH, action_keys=ACTION_KEYS + ["extra"],
        policy=mismatched_policy, critic=mismatched_critic,
    )
    subscriber = WeightSubscriber(path=path, bundle=mismatched_bundle)
    # The bad-compatibility load is caught and treated as a skipped poll,
    # not raised into the caller -- a transient bad snapshot must not crash
    # the actor's tick loop.
    assert subscriber.maybe_reload() is None
    assert subscriber.stats()["skipped_count"] == 1
