"""Generic action representation.

Actions are opaque to the runtime: a name plus optional parameters.  The set
of valid actions is defined by the Program, not by the runtime.  NULL is a
real action -- the agent must learn when not to act.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True)
class Action:
    """An immutable, hashable action."""

    name: str
    params: Tuple[Tuple[str, Any], ...] = ()

    @staticmethod
    def make(name: str, **params: Any) -> "Action":
        return Action(name, tuple(sorted(params.items())))

    def param(self, key: str, default: Any = None) -> Any:
        return dict(self.params).get(key, default)

    @property
    def is_null(self) -> bool:
        return self.name == "NULL"

    def key(self) -> str:
        """Stable string identifier, used for recording and model classes."""
        if not self.params:
            return self.name
        parts = ",".join(f"{k}={v}" for k, v in self.params)
        return f"{self.name}:{parts}"

    @staticmethod
    def from_key(key: str) -> "Action":
        """Inverse of :meth:`key`.  Parameter values parse as int when possible."""
        if ":" not in key:
            return Action(key)
        name, _, rest = key.partition(":")
        params = {}
        for part in rest.split(","):
            k, _, v = part.partition("=")
            try:
                params[k] = int(v)
            except ValueError:
                params[k] = v
        return Action.make(name, **params)


NULL_ACTION = Action("NULL")
