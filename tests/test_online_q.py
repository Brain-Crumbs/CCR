import json

import pytest

from cognitive_runtime.core.streams.fusion import LatentState
from cognitive_runtime.models.online_q import OnlineQModel


ACTIONS = ["NULL", "MOVE_FORWARD", "ATTACK"]


def _model(seed=0, epsilon_start=0.0, epsilon_min=0.0):
    return OnlineQModel.initialize(
        ACTIONS,
        latent_width=3,
        layout_hash="layout-a",
        lr=0.1,
        gamma=0.9,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay_ticks=10,
        seed=seed,
    )


def test_rewarded_action_q_value_increases():
    model = _model()
    state = [1.0, 0.0, 0.5]
    before = model.q_value("MOVE_FORWARD", state, [])
    metrics = model.td_update(state, [], "MOVE_FORWARD", 2.0, state, ["MOVE_FORWARD"], done=True)
    after = model.q_value("MOVE_FORWARD", state, [])

    assert metrics["td_error"] > 0.0
    assert after > before


def test_punished_action_q_value_decreases():
    model = _model()
    state = [0.0, 1.0, 0.5]
    before = model.q_value("ATTACK", state, [])
    metrics = model.td_update(state, [], "ATTACK", -2.0, state, ["ATTACK"], done=True)
    after = model.q_value("ATTACK", state, [])

    assert metrics["td_error"] < 0.0
    assert after < before


def test_save_load_round_trips_exactly(tmp_path):
    model = _model(seed=123, epsilon_start=1.0, epsilon_min=1.0)
    state = [0.25, -0.5, 1.0]
    model.td_update(state, [], "MOVE_FORWARD", 1.5, state, ["MOVE_FORWARD"], done=True)
    first_draw = model.select_action_key(state, [], epsilon=1.0)

    path = tmp_path / "online-q.json"
    model.save(str(path))
    loaded = OnlineQModel.load(
        str(path),
        expected_action_keys=ACTIONS,
        expected_layout_hash="layout-a",
        expected_latent_width=3,
    )

    assert loaded.to_dict() == model.to_dict()
    assert loaded.q_values(state, []) == model.q_values(state, [])
    assert loaded.select_action_key(state, [], epsilon=1.0) == model.select_action_key(
        state, [], epsilon=1.0
    )
    assert first_draw in ACTIONS

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    assert raw["format"] == "online-q-v1"
    assert raw["feature_names"][-1] == "bias"


def test_layout_and_action_mismatch_raise_clearly(tmp_path):
    model = _model()
    path = tmp_path / "online-q.json"
    model.save(str(path))

    with pytest.raises(ValueError, match="action-space mismatch"):
        OnlineQModel.load(str(path), expected_action_keys=["NULL", "USE"])
    with pytest.raises(ValueError, match="latent layout mismatch"):
        OnlineQModel.load(str(path), expected_layout_hash="layout-b")
    with pytest.raises(ValueError, match="latent width mismatch"):
        OnlineQModel.load(str(path), expected_latent_width=4)
    with pytest.raises(ValueError, match="latent width mismatch"):
        model.q_values([1.0, 2.0], [])


def test_latent_state_compatibility_checks():
    model = _model()
    latent = LatentState(vector=[1.0, 2.0, 3.0], slices={}, layout_hash="layout-a")
    assert model.q_values_from_latent(latent, []) == model.q_values(latent.vector, [])

    bad = LatentState(vector=[1.0, 2.0, 3.0], slices={}, layout_hash="layout-b")
    with pytest.raises(ValueError, match="latent layout mismatch"):
        model.q_values_from_latent(bad, [])


def test_epsilon_greedy_is_deterministic_with_fixed_seed():
    a = _model(seed=99, epsilon_start=1.0, epsilon_min=1.0)
    b = _model(seed=99, epsilon_start=1.0, epsilon_min=1.0)
    state = [0.0, 0.0, 0.0]

    seq_a = [a.select_action_key(state, [], epsilon=1.0) for _ in range(20)]
    seq_b = [b.select_action_key(state, [], epsilon=1.0) for _ in range(20)]

    assert seq_a == seq_b
    assert _model().select_action_key(state, [], epsilon=0.0) == ACTIONS[0]

