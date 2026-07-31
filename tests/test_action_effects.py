"""Action-effect label derivation (issue #234, MF-E1)."""

from __future__ import annotations

import itertools

import pytest

from cognitive_runtime.programs.crafter.actions import ACTION_SPACE
from cognitive_runtime.training.action_effects import (
    ACTION_EFFECT_CLASSES,
    ACTION_EFFECT_LABEL_VERSION,
    INTERACTION_ACTION_NAMES,
    MOVEMENT_ACTION_NAMES,
    classify_action_effect,
    compute_action_effect_labels,
)


def test_movement_and_interaction_action_names_partition_the_action_space():
    """Every Crafter action is exactly one of: a move, an interaction, or
    ``NULL`` -- the classifier's precedence rule leans on this being a
    complete, non-overlapping partition."""
    names = {action.name for action in ACTION_SPACE}
    assert MOVEMENT_ACTION_NAMES & INTERACTION_ACTION_NAMES == set()
    assert MOVEMENT_ACTION_NAMES | INTERACTION_ACTION_NAMES | {"NULL"} == names
    assert MOVEMENT_ACTION_NAMES == {"MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN"}


def test_action_effect_label_version_is_set():
    assert ACTION_EFFECT_LABEL_VERSION


def test_moved_takes_precedence_when_position_changed():
    """Design constraint: document the precedence rule for ambiguous cases
    (the issue's own example -- moved *and* interacted)."""
    for blocked, interacted, facing_changed in itertools.product([False, True], repeat=3):
        if blocked and interacted:
            continue  # not a real combination compute_action_effect_labels ever forms
        assert classify_action_effect(
            position_changed=True, blocked=blocked, interacted=interacted,
            facing_changed=facing_changed,
        ) == "moved"


def test_blocked_forward_move_is_not_no_op():
    """Acceptance criterion: a blocked forward move is classified ``blocked``,
    not ``no_op`` -- the agent acted and the world refused."""
    assert classify_action_effect(
        position_changed=False, blocked=True, interacted=False, facing_changed=False,
    ) == "blocked"
    # Still blocked even when the attempt also changed facing (Crafter turns
    # *by* a blocked move -- blocked, not turned_only, takes the label).
    assert classify_action_effect(
        position_changed=False, blocked=True, interacted=False, facing_changed=True,
    ) == "blocked"


def test_facing_change_with_no_position_change_is_turned_only():
    assert classify_action_effect(
        position_changed=False, blocked=False, interacted=False, facing_changed=True,
    ) == "turned_only"


def test_interacted_when_a_non_movement_action_fires_without_displacement():
    assert classify_action_effect(
        position_changed=False, blocked=False, interacted=True, facing_changed=False,
    ) == "interacted"
    assert classify_action_effect(
        position_changed=False, blocked=False, interacted=True, facing_changed=True,
    ) == "interacted"


def test_no_op_when_nothing_changed():
    assert classify_action_effect(
        position_changed=False, blocked=False, interacted=False, facing_changed=False,
    ) == "no_op"


def test_classes_are_exhaustive_and_mutually_exclusive():
    """Every combination of the derived boolean signals resolves to exactly
    one of the five declared classes."""
    for position_changed, blocked, interacted, facing_changed in itertools.product(
        [False, True], repeat=4
    ):
        result = classify_action_effect(
            position_changed=position_changed, blocked=blocked,
            interacted=interacted, facing_changed=facing_changed,
        )
        assert result in ACTION_EFFECT_CLASSES


# --------------------------------------------------------------------------- recorded-episode integration

pytest.importorskip("crafter")
torch = pytest.importorskip("torch")

from cognitive_runtime.runtime.replay import list_episodes  # noqa: E402
from cognitive_runtime.training.nursery import (  # noqa: E402
    CRAFTER_SCENARIOS,
    NurseryConfig,
    _record_scenario_episode,
)


def _crafter_config(**overrides) -> NurseryConfig:
    base = dict(world="crafter", episode_ticks=40, train_seeds=(0, 1), holdout_seeds=(1000,))
    base.update(overrides)
    return NurseryConfig(**base)


def test_blocked_forward_scenario_yields_moved_and_blocked_labels(tmp_path):
    scenario = CRAFTER_SCENARIOS["blocked_forward"]
    cfg = _crafter_config()
    session_dir = _record_scenario_episode(str(tmp_path), "blocked-forward", 0, scenario, cfg)
    episode_id = list_episodes(session_dir)[0]

    labels = compute_action_effect_labels(session_dir, episode_id)

    assert len(labels) > 0
    assert all(label.action_effect_class in ACTION_EFFECT_CLASSES for label in labels)
    classes = {label.action_effect_class for label in labels}
    # The approach + recovery moves...
    assert "moved" in classes
    # ...and the intentional collision against the wall.
    assert "blocked" in classes
    assert "no_op" not in classes  # the blocked hold must not be swallowed into no_op
    for label in labels:
        if label.action_effect_class == "blocked":
            assert label.position_delta in (None, (0.0, 0.0))
            assert label.movement_magnitude == 0.0


def test_turn_scenario_boxed_in_is_blocked_not_turned_only(tmp_path):
    """The boxed-in ``turn`` scenario never displaces (issue #90); every
    directional attempt is refused, so every transition should be
    ``blocked`` -- not ``turned_only``, which is Crafter-unreachable because
    every facing change there comes from a movement attempt."""
    scenario = CRAFTER_SCENARIOS["turn"]
    cfg = _crafter_config()
    session_dir = _record_scenario_episode(str(tmp_path), "turn-boxed", 0, scenario, cfg)
    episode_id = list_episodes(session_dir)[0]

    labels = compute_action_effect_labels(session_dir, episode_id)

    assert len(labels) > 0
    assert all(label.action_effect_class == "blocked" for label in labels)
    assert any(label.facing_changed for label in labels)
    assert all(label.position_delta in (None, (0.0, 0.0)) for label in labels)


def test_walk_forward_short_scenario_yields_mostly_moved_labels(tmp_path):
    scenario = CRAFTER_SCENARIOS["walk_forward_short"]
    cfg = _crafter_config()
    session_dir = _record_scenario_episode(str(tmp_path), "walk-forward", 0, scenario, cfg)
    episode_id = list_episodes(session_dir)[0]

    labels = compute_action_effect_labels(session_dir, episode_id)

    assert len(labels) > 0
    moved_fraction = sum(1 for l in labels if l.action_effect_class == "moved") / len(labels)
    assert moved_fraction >= 0.5


def test_derivation_is_deterministic_across_runs(tmp_path):
    """Acceptance criterion: the same session yields identical labels across
    runs."""
    scenario = CRAFTER_SCENARIOS["blocked_forward"]
    cfg = _crafter_config()
    session_dir = _record_scenario_episode(str(tmp_path), "determinism", 0, scenario, cfg)
    episode_id = list_episodes(session_dir)[0]

    first = compute_action_effect_labels(session_dir, episode_id)
    second = compute_action_effect_labels(session_dir, episode_id)

    assert first == second
