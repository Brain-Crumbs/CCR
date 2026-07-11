"""Action classification registry (issue #60): world-changing vs
information-gathering, the action-space analog of the stream classification
registry (issue #32's `core.streams.registry.StreamRegistry`).

Looking around, waiting and inspecting change what the agent can perceive
next tick without touching the world; mining, placing, attacking, crafting
and consuming change the world (and the agent's own state) directly. An
action can be both -- walking repositions the agent (world-changing) and
exposes a new view (information-gathering) in the same motor command. Every
declared action must be at least one of the two; declaring neither is a
mistake, not a legitimate "does nothing" classification (`NULL` is
information-gathering: waiting is how the agent lets more information
arrive without acting).

Generic, Program-agnostic core: no action *names* live here, only the
declaration/registry machinery a Program's concrete action space is checked
against -- mirrors `StreamRegistry`'s `missing()`/`assert_complete()`
completeness pattern for actions instead of streams. The scripted orienting
reflex (`core.orienting_reflex`) reads `is_world_changing` off a Program's
registry to decide whether it may substitute its own look/turn action for
the policy's this tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from cognitive_runtime.core.action import Action


@dataclass(frozen=True)
class ActionDeclaration:
    """One action name's classification. `name` matches `Action.name`
    exactly -- unlike stream ids, an action's parameters (hotbar slot,
    recipe id, ...) never change its classification, so one declaration
    covers every parameterized variant of a base verb."""

    name: str
    world_changing: bool
    information_gathering: bool
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ActionDeclaration.name must be non-empty")
        if not (self.world_changing or self.information_gathering):
            raise ValueError(
                f"{self.name!r} declares neither world_changing nor "
                "information_gathering; every action must be at least one"
            )


class ActionRegistry:
    """A set of `ActionDeclaration`s, one per action name."""

    def __init__(self, declarations: Optional[Iterable[ActionDeclaration]] = None) -> None:
        self._by_name: Dict[str, ActionDeclaration] = {}
        for decl in declarations or ():
            self._by_name[decl.name] = decl

    def extend(self, declarations: Iterable[ActionDeclaration]) -> "ActionRegistry":
        """A new registry with `declarations` layered under this one's --
        a name already declared here is kept (this registry takes priority),
        matching `StreamRegistry.extend`'s precedent of the extending
        registry never overriding what it extends."""
        merged: Dict[str, ActionDeclaration] = {}
        for decl in declarations:
            merged[decl.name] = decl
        merged.update(self._by_name)
        registry = ActionRegistry()
        registry._by_name = merged
        return registry

    @property
    def declarations(self) -> Tuple[ActionDeclaration, ...]:
        return tuple(self._by_name.values())

    def declaration_for(self, action: Action) -> Optional[ActionDeclaration]:
        return self._by_name.get(action.name)

    def is_world_changing(self, action: Action) -> bool:
        """Conservative default: an undeclared action counts as
        world-changing, so a precedence check keying off this (the
        orienting reflex) never mistakes a missing declaration for a
        safe-to-override one."""
        decl = self.declaration_for(action)
        return True if decl is None else decl.world_changing

    def is_information_gathering(self, action: Action) -> bool:
        decl = self.declaration_for(action)
        return False if decl is None else decl.information_gathering

    def missing(self, action_space: Iterable[Action]) -> List[str]:
        """Action names in `action_space` with no matching declaration."""
        return sorted({a.name for a in action_space if a.name not in self._by_name})

    def assert_complete(self, action_space: Iterable[Action]) -> None:
        missing = self.missing(action_space)
        if missing:
            raise ValueError(
                f"action(s) missing an ActionDeclaration: {missing}; every "
                "action needs a world_changing/information_gathering "
                "classification (issue #60)"
            )


#: No Program-agnostic action names exist (unlike streams' generic
#: "body.*"/"reward.*" modality conventions) -- every concrete action is a
#: Program's own verb, so the generic layer starts empty and each Program
#: supplies its own registry (see `programs.minecraft.action_registry`).
#: `ActionRegistry.is_world_changing` defaults an undeclared action to
#: `True`, so a runtime built without a Program-specific registry simply
#: never lets the orienting reflex override the policy -- inert, not wrong.
DEFAULT_ACTION_REGISTRY = ActionRegistry()
