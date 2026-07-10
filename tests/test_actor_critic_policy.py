"""Runtime-facing actor/critic policy + learner (issue #29): action emission,
NULL handling, one-tick delayed reward attribution, and eval-mode no-mutation
determinism. Mirrors tests/test_online_q_policy.py's shape.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cognitive_runtime.core.action import Action  # noqa: E402
from cognitive_runtime.core.memory import Memory  # noqa: E402
from cognitive_runtime.core.observation import Observation  # noqa: E402
from cognitive_runtime.core.perception import State  # noqa: E402
from cognitive_runtime.core.streams.events import StreamEvent  # noqa: E402
from cognitive_runtime.core.streams.fusion import LatentState  # noqa: E402
from cognitive_runtime.core.streams.synchronizer import TickWindow  # noqa: E402
from cognitive_runtime.core.world_model import Prediction  # noqa: E402
from cognitive_runtime.neural import ActorCriticOptimizer, MLPPolicyModel, MLPValueModel  # noqa: E402
from cognitive_runtime.policies.actor_critic import (  # noqa: E402
    ActorCriticLearner,
    ActorCriticPolicy,
    world_feature_width,
    world_features_vector,
)

ACTIONS = ["NULL", "MOVE_FORWARD", "ATTACK"]
FUSED_WIDTH = 2


def _stack(seed=0, lr=0.05):
    wf_width = world_feature_width(ACTIONS)
    policy_model = MLPPolicyModel(FUSED_WIDTH, wf_width, len(ACTIONS), hidden_dim=8)
    critic_model = MLPValueModel(FUSED_WIDTH, wf_width, hidden_dim=8)
    optimizer = ActorCriticOptimizer(policy_model, critic_model, lr=lr, seed=seed)
    return policy_model, critic_model, optimizer


def _memory(vector):
    memory = Memory()
    memory.set_fused_latent(LatentState(vector=list(vector), slices={}, layout_hash="layout-a"))
    return memory


def _state():
    return State(Observation(timestamp=0.0, tick=0, data={}))


def _reward_window(value, tick=0):
    event = StreamEvent(
        stream_id="reward.scalar",
        modality="reward",
        timestamp=float(tick),
        sequence_number=tick,
        payload={"value": value},
    )
    return TickWindow(
        tick_index=tick,
        started_at=float(tick),
        ended_at=float(tick + 1),
        events=[event],
        by_stream={"reward.scalar": [event]},
    )


def test_world_features_vector_degrades_gracefully_without_a_prediction():
    features = world_features_vector(None, [], ACTIONS)
    assert features == [0.0, 0.0, 0.0, 0.0] + [0.0] * len(ACTIONS)


def test_world_features_vector_fills_in_available_prediction_fields():
    prediction = Prediction(risk=0.4, p_death=0.1, predicted_reward=None, prediction_error=0.2)
    features = world_features_vector(prediction, ["MOVE_FORWARD"], ACTIONS)
    assert features[:4] == [0.4, 0.1, 0.0, 0.2]
    assert features[4:] == [0.0, 1.0, 0.0]  # one-hot over ACTIONS for MOVE_FORWARD


def test_policy_emits_a_valid_action_and_null_maps_to_empty_emission():
    policy_model, critic_model, _optimizer = _stack()
    action_space = [Action.from_key(key) for key in ACTIONS]
    policy = ActorCriticPolicy(policy_model, critic_model, ACTIONS, action_space=action_space, training=False)

    emissions = policy.emit(_state(), _memory([0.0, 0.0]), None)

    assert policy.latest_decision is not None
    assert policy.latest_decision.action_key in ACTIONS
    if policy.latest_decision.action_key == "NULL":
        assert emissions == []
    else:
        assert emissions == [Action.from_key(policy.latest_decision.action_key)]
        assert emissions[0] in action_space


def test_eval_mode_action_selection_is_argmax_and_deterministic_per_seed():
    policy_model, critic_model, _optimizer = _stack()
    policy = ActorCriticPolicy(policy_model, critic_model, ACTIONS, training=False)

    memory = _memory([0.3, -0.2])
    first = policy.emit(_state(), memory, None)
    second = policy.emit(_state(), memory, None)

    assert policy.latest_decision is not None
    assert first == second


def test_previous_decision_receives_reward_update_and_mutates_weights():
    policy_model, critic_model, optimizer = _stack()
    policy = ActorCriticPolicy(policy_model, critic_model, ACTIONS, training=True)
    learner = ActorCriticLearner(optimizer, policy, training=True)
    before = [p.clone() for p in policy_model.parameters()]

    policy.emit(_state(), _memory([1.0, 0.0]), None)
    learner.update(_reward_window(0.0, tick=0))
    assert learner.update_count == 0  # first decision has no previous decision yet

    policy.emit(_state(), _memory([0.0, 1.0]), None)
    learner.update(_reward_window(2.0, tick=1))

    assert learner.update_count == 1
    assert optimizer.step_count == 1
    assert any(not torch.equal(a, b) for a, b in zip(before, policy_model.parameters()))


def test_eval_mode_does_not_mutate_weights_or_optimizer_step_count():
    policy_model, critic_model, optimizer = _stack()
    policy = ActorCriticPolicy(policy_model, critic_model, ACTIONS, training=False)
    learner = ActorCriticLearner(optimizer, policy, training=False)
    before_policy = [p.clone() for p in policy_model.parameters()]
    before_critic = [p.clone() for p in critic_model.parameters()]

    policy.emit(_state(), _memory([1.0, 0.0]), None)
    learner.update(_reward_window(0.0, tick=0))
    policy.emit(_state(), _memory([0.0, 1.0]), None)
    learner.update(_reward_window(5.0, tick=1))

    assert optimizer.step_count == 0
    assert learner.skipped_updates == 1
    assert all(torch.equal(a, b) for a, b in zip(before_policy, policy_model.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(before_critic, critic_model.parameters()))


def test_learner_rejects_policy_with_different_modules():
    policy_model, critic_model, optimizer = _stack()
    other_policy_model, other_critic_model, _ = _stack()
    mismatched_policy = ActorCriticPolicy(other_policy_model, other_critic_model, ACTIONS, training=True)

    with pytest.raises(ValueError, match="same policy/critic modules"):
        ActorCriticLearner(optimizer, mismatched_policy, training=True)
