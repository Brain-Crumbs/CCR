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
from sleep.weight_publisher import EMAWeightPublisher  # noqa: E402

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


# ---------------------------------------------------------- EMA (issue #100)


def test_ema_publisher_rejects_an_out_of_range_decay(tmp_path):
    path = str(tmp_path / "trainer.pt")
    _policy, _critic, _optimizer, trainer_bundle = _trainer_side(path)
    for bad_decay in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="decay"):
            EMAWeightPublisher(trainer_bundle, decay=bad_decay)


def test_ema_publisher_first_publish_seeds_the_shadow_with_raw_weights(tmp_path):
    """No prior EMA state exists yet, so the very first snapshot must equal
    the raw weights exactly -- otherwise the actor's first-ever reload would
    be diluted toward an untrained init instead of what was actually
    trained."""
    path = str(tmp_path / "trainer.pt")
    policy, critic, optimizer, trainer_bundle = _trainer_side(path)
    publisher = EMAWeightPublisher(trainer_bundle, decay=0.9)
    actor_policy, actor_critic, subscriber = _actor_side(path)

    optimizer.step(_random_batch())
    trainer_bundle.training_ticks = optimizer.step_count
    publisher.publish(reason="t1")
    subscriber.maybe_reload()

    for raw, published in zip(policy.parameters(), actor_policy.parameters()):
        assert torch.equal(raw, published)
    for raw, published in zip(critic.parameters(), actor_critic.parameters()):
        assert torch.equal(raw, published)


def test_ema_publisher_smooths_subsequent_publishes(tmp_path):
    """A second publish is a genuine Polyak blend of the first snapshot and
    the newly-trained raw weights -- not a straight copy of either -- which
    is exactly the "slow-moving target" that kills tick-to-tick
    oscillation."""
    path = str(tmp_path / "trainer.pt")
    policy, critic, optimizer, trainer_bundle = _trainer_side(path)
    decay = 0.7
    publisher = EMAWeightPublisher(trainer_bundle, decay=decay)
    _actor_policy, _actor_critic, subscriber = _actor_side(path)

    optimizer.step(_random_batch())
    trainer_bundle.training_ticks = optimizer.step_count
    publisher.publish(reason="t1")
    first_shadow = {
        name: {k: v.clone() for k, v in state.items()}
        for name, state in publisher._shadow.items()
    }

    optimizer.step(_random_batch())
    trainer_bundle.training_ticks = optimizer.step_count
    v2 = publisher.publish(reason="t2")
    reloaded = subscriber.maybe_reload()
    assert reloaded == v2

    expected_policy = {
        k: decay * first_shadow["policy"][k] + (1 - decay) * v
        for k, v in policy.state_dict().items()
    }
    for key, param in policy.state_dict().items():
        assert torch.allclose(publisher._shadow["policy"][key], expected_policy[key])
        # The published snapshot is the blend, distinct from the pure raw
        # weights whenever the two gradient steps actually moved them apart.
        if not torch.allclose(param, first_shadow["policy"][key]):
            assert not torch.allclose(publisher._shadow["policy"][key], param)


def test_ema_publisher_never_mutates_the_live_training_weights(tmp_path):
    """The swap-save-restore dance around ``publish`` must be transparent to
    training: the raw modules used for the next gradient step are exactly
    what they were right before publish, not the EMA-averaged copy."""
    path = str(tmp_path / "trainer.pt")
    policy, critic, optimizer, trainer_bundle = _trainer_side(path)
    publisher = EMAWeightPublisher(trainer_bundle, decay=0.5)

    optimizer.step(_random_batch())
    trainer_bundle.training_ticks = optimizer.step_count
    before = [p.clone() for p in policy.parameters()]
    publisher.publish(reason="t1")
    optimizer.step(_random_batch())  # a second publish forces a real EMA blend
    trainer_bundle.training_ticks = optimizer.step_count
    after_first_step = [p.clone() for p in policy.parameters()]
    publisher.publish(reason="t2")

    for pre, post in zip(after_first_step, policy.parameters()):
        assert torch.equal(pre, post)
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))


def test_ema_publisher_keeps_the_version_monotonic_across_publishes(tmp_path):
    path = str(tmp_path / "trainer.pt")
    _policy, _critic, optimizer, trainer_bundle = _trainer_side(path)
    publisher = EMAWeightPublisher(trainer_bundle, decay=0.95)

    versions = []
    for _ in range(4):
        optimizer.step(_random_batch())
        trainer_bundle.training_ticks = optimizer.step_count
        versions.append(publisher.publish(reason="tick"))

    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)


# ---------------------------------------------------- staleness (issue #100)


def test_staleness_is_none_before_anything_is_published(tmp_path):
    path = str(tmp_path / "trainer.pt")
    _actor_policy, _actor_critic, subscriber = _actor_side(path)
    assert subscriber.staleness() is None


def test_staleness_tracks_and_bounds_the_gap_between_publish_and_reload(tmp_path):
    path = str(tmp_path / "trainer.pt")
    _policy, _critic, optimizer, trainer_bundle = _trainer_side(path)
    publisher = WeightPublisher(trainer_bundle)
    _actor_policy, _actor_critic, subscriber = _actor_side(path)

    trainer_bundle.training_ticks = optimizer.step_count
    publisher.publish(reason="t1")
    assert subscriber.staleness() == 1  # published, not yet loaded
    subscriber.maybe_reload()
    assert subscriber.staleness() == 0  # caught up

    # Two more publishes land before the actor polls again.
    for _ in range(2):
        optimizer.step(_random_batch())
        trainer_bundle.training_ticks = optimizer.step_count
        publisher.publish(reason="tick")
    assert subscriber.staleness() == 2

    reloaded = subscriber.maybe_reload()
    assert reloaded is not None
    assert subscriber.staleness() == 0
    # The peak observed gap is retained even after catching back up.
    assert subscriber.stats()["max_staleness"] == 2
