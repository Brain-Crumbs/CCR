import json

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.streams.events import StreamEvent
from cognitive_runtime.core.streams.fusion import LatentState
from cognitive_runtime.core.streams.synchronizer import TickWindow
from cognitive_runtime.models.online_q import OnlineQModel
from cognitive_runtime.policies.online_q import OnlineQLearner, OnlineQPolicy


ACTIONS = ["NULL", "MOVE_FORWARD", "ATTACK"]


def _model(training=True):
    return OnlineQModel.initialize(
        ACTIONS,
        latent_width=2,
        layout_hash="layout-a",
        lr=0.1,
        gamma=0.9,
        epsilon_start=1.0 if training else 0.0,
        epsilon_min=1.0 if training else 0.0,
        seed=7,
    )


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


def test_online_policy_emits_valid_actions():
    model = _model(training=False)
    model.bias[1] = 2.0
    action_space = [Action.from_key(key) for key in ACTIONS]
    policy = OnlineQPolicy(model, action_space=action_space, training=False)

    emissions = policy.emit(_state(), _memory([0.0, 0.0]), None)

    assert emissions == [Action("MOVE_FORWARD")]
    assert emissions[0] in action_space
    assert policy.latest_decision is not None
    assert policy.latest_decision.action_key == "MOVE_FORWARD"


def test_null_action_maps_to_empty_motor_emission():
    model = _model(training=False)
    model.bias[0] = 2.0
    policy = OnlineQPolicy(model, training=False)

    emissions = policy.emit(_state(), _memory([0.0, 0.0]), None)

    assert emissions == []
    assert policy.latest_decision is not None
    assert policy.latest_decision.action_key == "NULL"


def test_previous_action_not_current_action_receives_reward_update():
    model = _model(training=False)
    # State [1, 0] chooses MOVE_FORWARD; state [0, 1] chooses ATTACK.
    model.weights[1][0] = 1.0
    model.weights[2][1] = 1.0
    policy = OnlineQPolicy(model, training=False)
    learner = OnlineQLearner(model, policy, training=True)

    policy.emit(_state(), _memory([1.0, 0.0]), None)
    learner.update(_reward_window(0.0, tick=0))
    before_move = list(model.weights[1]), model.bias[1]
    before_attack = list(model.weights[2]), model.bias[2]

    policy.emit(_state(), _memory([0.0, 1.0]), None)
    assert policy.latest_decision.action_key == "ATTACK"
    learner.update(_reward_window(2.0, tick=1))

    assert (model.weights[1], model.bias[1]) != before_move
    assert (model.weights[2], model.bias[2]) == before_attack
    assert model.training_ticks == 1


def test_eval_mode_uses_zero_epsilon_and_does_not_mutate_weights():
    model = _model(training=True)
    model.weights[1][0] = 1.0
    model.weights[2][1] = 1.0
    policy = OnlineQPolicy(model, training=False)
    learner = OnlineQLearner(model, policy, training=False)
    before = model.to_dict()

    policy.emit(_state(), _memory([1.0, 0.0]), None)
    assert policy.latest_decision.epsilon == 0.0
    learner.update(_reward_window(0.0, tick=0))
    policy.emit(_state(), _memory([0.0, 1.0]), None)
    assert policy.latest_decision.epsilon == 0.0
    learner.update(_reward_window(5.0, tick=1))

    after = model.to_dict()
    assert after["weights"] == before["weights"]
    assert after["bias"] == before["bias"]
    assert after["training_ticks"] == before["training_ticks"]
    assert after["epsilon_state"] == before["epsilon_state"]


def test_learner_save_includes_stats(tmp_path):
    model = _model(training=False)
    model.weights[1][0] = 1.0
    model.weights[2][1] = 1.0
    policy = OnlineQPolicy(model, training=False)
    learner = OnlineQLearner(model, policy, training=True)

    policy.emit(_state(), _memory([1.0, 0.0]), None)
    learner.update(_reward_window(0.5, tick=0))
    policy.emit(_state(), _memory([0.0, 1.0]), None)
    learner.update(_reward_window(1.5, tick=1))
    path = tmp_path / "online-q.json"
    learner.save(str(path))

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    stats = raw["meta"]["learner_stats"]
    assert stats["training_ticks"] == 1
    assert stats["reward_total"] == 2.0
    assert stats["td_updates"] == 1
    assert "epsilon_state" in stats

