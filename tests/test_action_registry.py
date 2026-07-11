"""Issue #60: action classification (world_changing vs
information_gathering) and the Minecraft registry's completeness (the issue
#32 completeness-test pattern, applied to the action space)."""

from __future__ import annotations

import pytest

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.action_registry import (
    DEFAULT_ACTION_REGISTRY,
    ActionDeclaration,
    ActionRegistry,
)
from cognitive_runtime.programs.minecraft.action_registry import MINECRAFT_ACTION_REGISTRY
from cognitive_runtime.programs.minecraft.actions import ACTION_SPACE


# ------------------------------------------------------------- declaration validation


def test_declaration_requires_at_least_one_classification():
    with pytest.raises(ValueError, match="neither world_changing nor information_gathering"):
        ActionDeclaration("FOO", world_changing=False, information_gathering=False)


def test_declaration_requires_a_name():
    with pytest.raises(ValueError, match="non-empty"):
        ActionDeclaration("", world_changing=True, information_gathering=False)


def test_declaration_allows_both():
    decl = ActionDeclaration("MOVE_FORWARD", world_changing=True, information_gathering=True)
    assert decl.world_changing and decl.information_gathering


# ------------------------------------------------------------- registry behavior


def test_undeclared_action_is_world_changing_by_default():
    """Conservative default: an undeclared action blocks a reflex-style
    precedence check rather than silently permitting it to be overridden."""
    registry = ActionRegistry()
    assert registry.is_world_changing(Action("ANYTHING")) is True
    assert registry.is_information_gathering(Action("ANYTHING")) is False


def test_default_registry_is_empty_and_conservative():
    assert DEFAULT_ACTION_REGISTRY.declarations == ()
    assert DEFAULT_ACTION_REGISTRY.is_world_changing(NULL_ACTION) is True


def test_missing_and_assert_complete():
    registry = ActionRegistry(
        [ActionDeclaration("MOVE_FORWARD", world_changing=True, information_gathering=True)]
    )
    action_space = [Action("MOVE_FORWARD"), Action("LOOK_LEFT")]
    assert registry.missing(action_space) == ["LOOK_LEFT"]
    with pytest.raises(ValueError, match="LOOK_LEFT"):
        registry.assert_complete(action_space)

    registry = registry.extend(
        [ActionDeclaration("LOOK_LEFT", world_changing=False, information_gathering=True)]
    )
    registry.assert_complete(action_space)  # does not raise


def test_extend_does_not_override_existing_declarations():
    base = ActionRegistry(
        [ActionDeclaration("USE", world_changing=True, information_gathering=False)]
    )
    extended = base.extend(
        [ActionDeclaration("USE", world_changing=False, information_gathering=True)]
    )
    assert extended.declaration_for(Action("USE")).world_changing is True


def test_parameterized_action_variants_share_one_declaration():
    """A parameterized action's variants (different hotbar slots, recipes,
    ...) all share the same base name, so one declaration classifies every
    variant -- unlike stream ids, an action's params never change its
    classification."""
    slot_0 = Action.make("SELECT_HOTBAR_SLOT", slot=0)
    slot_5 = Action.make("SELECT_HOTBAR_SLOT", slot=5)
    assert (
        MINECRAFT_ACTION_REGISTRY.is_world_changing(slot_0)
        == MINECRAFT_ACTION_REGISTRY.is_world_changing(slot_5)
    )


# ------------------------------------------------------------- completeness (acceptance criteria)


def test_minecraft_action_space_is_fully_classified():
    assert MINECRAFT_ACTION_REGISTRY.missing(ACTION_SPACE) == []
    MINECRAFT_ACTION_REGISTRY.assert_complete(ACTION_SPACE)  # does not raise
    for action in ACTION_SPACE:
        decl = MINECRAFT_ACTION_REGISTRY.declaration_for(action)
        assert decl is not None, action.name
        assert decl.world_changing or decl.information_gathering, action.name


def test_look_actions_are_pure_information_gathering():
    """The issue's own framing: camera/look actions are information-gathering,
    distinct from world-changing actions -- this is what lets the orienting
    reflex (issue #60) substitute a look for an otherwise-idle tick without
    ever being mistaken for a world-changing decision."""
    for name in ("LOOK_LEFT", "LOOK_RIGHT", "LOOK_UP", "LOOK_DOWN"):
        action = Action(name)
        assert MINECRAFT_ACTION_REGISTRY.is_information_gathering(action) is True
        assert MINECRAFT_ACTION_REGISTRY.is_world_changing(action) is False


def test_null_is_information_gathering_not_world_changing():
    assert MINECRAFT_ACTION_REGISTRY.is_information_gathering(NULL_ACTION) is True
    assert MINECRAFT_ACTION_REGISTRY.is_world_changing(NULL_ACTION) is False


def test_movement_is_both_world_changing_and_information_gathering():
    """The issue's own worked example: 'walking does both'."""
    for name in ("MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT"):
        action = Action(name)
        assert MINECRAFT_ACTION_REGISTRY.is_world_changing(action) is True
        assert MINECRAFT_ACTION_REGISTRY.is_information_gathering(action) is True


def test_mining_and_survival_actions_are_world_changing():
    """The issue's own worked example: 'mining changes the world'; fleeing
    and eating (survival-critical policy actions the reflex must never
    suppress) are movement/USE, both world-changing."""
    for name in ("ATTACK", "USE", "CRAFT"):
        assert MINECRAFT_ACTION_REGISTRY.is_world_changing(Action(name)) is True
