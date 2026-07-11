"""Reward profile schema/loader tests (issue #41)."""

import os

import pytest

from cognitive_runtime.programs.minecraft.reward_profile import (
    RewardProfileError,
    default_profile,
    load_reward_profile,
    reward_profile_from_dict,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _minimal(**overrides):
    data = {
        "name": "test",
        "tiers": {
            "survival": {
                "tick_alive": {"kind": "tick", "value": 0.01},
            },
        },
    }
    data.update(overrides)
    return data


def test_default_profile_has_expected_components():
    profile = default_profile()
    flat = profile.components()
    assert "tick_alive" in flat
    assert flat["tick_alive"][0] == "survival"
    assert flat["new_block_type"][0] == "capability"
    assert flat["spinning"][0] == "shaping"
    assert flat["tick_alive"][1].value == 0.01


def test_content_hash_stable_and_sensitive_to_changes():
    a = reward_profile_from_dict(_minimal())
    b = reward_profile_from_dict(_minimal())
    assert a.content_hash == b.content_hash
    c = reward_profile_from_dict(_minimal(name="different"))
    assert a.content_hash != c.content_hash


def test_missing_name_fails_at_load():
    with pytest.raises(RewardProfileError, match="name"):
        reward_profile_from_dict({"tiers": {}})


def test_unknown_kind_fails_at_load():
    data = _minimal()
    data["tiers"]["survival"]["bogus"] = {"kind": "not_a_real_kind", "value": 1.0}
    with pytest.raises(RewardProfileError, match="unknown kind"):
        reward_profile_from_dict(data)


def test_unknown_top_level_field_fails_at_load():
    data = _minimal()
    data["totally_unexpected"] = True
    with pytest.raises(RewardProfileError, match="unknown top-level field"):
        reward_profile_from_dict(data)


def test_capped_novelty_without_cap_fails_at_load():
    data = _minimal()
    data["tiers"]["capability"] = {
        "new_item": {"kind": "capped_novelty", "value": 0.5, "params": {"source": "event:new_item"}},
    }
    with pytest.raises(RewardProfileError, match="requires a 'cap'"):
        reward_profile_from_dict(data)


def test_missing_required_param_fails_at_load():
    data = _minimal()
    data["tiers"]["survival"]["damage"] = {"kind": "event_count", "value": -0.5, "params": {}}
    with pytest.raises(RewardProfileError, match="missing required key 'event_prefix'"):
        reward_profile_from_dict(data)


def test_invalid_scope_fails_at_load():
    data = _minimal()
    data["tiers"]["survival"]["tick_alive"]["scope"] = "galaxy"
    with pytest.raises(RewardProfileError, match="scope"):
        reward_profile_from_dict(data)


def test_duplicate_component_name_across_tiers_fails_at_load():
    data = _minimal()
    data["tiers"]["capability"] = {"tick_alive": {"kind": "death", "value": -1.0}}
    with pytest.raises(RewardProfileError, match="unique"):
        reward_profile_from_dict(data)


def test_intrinsic_missing_stream_fails_at_load():
    data = _minimal()
    data["intrinsic"] = {"learning_progress": {"weight": 1.0}}
    with pytest.raises(RewardProfileError, match="stream"):
        reward_profile_from_dict(data)


def test_intrinsic_component_loads_with_defaults():
    data = _minimal()
    data["intrinsic"] = {"learning_progress": {"stream": "internal.learning_progress", "weight": 2.0}}
    profile = reward_profile_from_dict(data)
    spec = profile.intrinsic["learning_progress"]
    assert spec.stream == "internal.learning_progress"
    assert spec.weight == 2.0


def test_bad_normalization_method_fails_at_load():
    data = _minimal(normalization={"method": "quantum"})
    with pytest.raises(RewardProfileError, match="normalization.method"):
        reward_profile_from_dict(data)


def test_load_missing_file_raises_profile_error():
    with pytest.raises(RewardProfileError):
        load_reward_profile("/nonexistent/path/profile.yaml")


def test_load_unsupported_extension_raises_profile_error():
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as fh:
        fh.write("name: test\n")
        path = fh.name
    try:
        with pytest.raises(RewardProfileError, match="unsupported extension"):
            load_reward_profile(path)
    finally:
        os.unlink(path)


def test_load_malformed_yaml_raises_profile_error():
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as fh:
        fh.write("name: [unterminated\n")
        path = fh.name
    try:
        with pytest.raises(RewardProfileError, match="invalid YAML"):
            load_reward_profile(path)
    finally:
        os.unlink(path)


def test_shipped_survival_profile_loads_and_matches_default():
    profile = load_reward_profile(os.path.join(REPO_ROOT, "goals", "survival.yaml"))
    default = default_profile()
    assert set(profile.components()) == set(default.components())


def test_shipped_ender_dragon_profile_loads_with_quest_and_intrinsic_tiers():
    profile = load_reward_profile(os.path.join(REPO_ROOT, "goals", "ender_dragon.yaml"))
    assert "quest" in profile.tiers
    assert profile.tiers["quest"]["defeated_dragon"].value == 1_000_000.0
    assert profile.tiers["quest"]["defeated_dragon"].scope == "brain"
    assert set(profile.intrinsic) == {
        "learning_progress", "safe_novelty", "predicted_risk_aversion",
    }
