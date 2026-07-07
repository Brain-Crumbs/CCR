"""Fake Minecraft bridge: the deterministic SimulatedWorld over the wire.

Speaks the exact line-delimited JSON protocol the real mineflayer bridge
speaks (see ``cognitive_runtime/programs/minecraft/remote.py``), but backs it
with :class:`SimulatedWorld` instead of a Minecraft server.  Two uses:

1. **Tests.** The whole remote path — subprocess management, JSON framing,
   event translation, error handling — is exercised with no Minecraft and no
   Node.  Because both this bridge and the in-process ``SimulatedBackend``
   wrap the same world, ``remote-via-fake-bridge`` reproduces the in-process
   backend byte-for-byte on the same seed (a cross-check test asserts it).
2. **A runnable protocol reference.** The Node bridge must match the
   behaviour here; this file is the shortest correct implementation.

Run manually:  ``python -m bridge.fake.sim_bridge``  (then type JSON lines),
or point the backend at it::

    CCR_MINECRAFT_BRIDGE_CMD="python -m bridge.fake.sim_bridge" \
        python -m cognitive_runtime run --backend remote --episode-ticks 50
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

from cognitive_runtime.core.action import Action
from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig
from cognitive_runtime.programs.minecraft.observations import build_observation
from cognitive_runtime.programs.minecraft.world import SimulatedWorld


class SimBridge:
    """Handles one protocol command at a time against a SimulatedWorld."""

    def __init__(self) -> None:
        self._world: Optional[SimulatedWorld] = None

    def _status(self) -> Dict[str, Any]:
        assert self._world is not None
        return {
            "tick": self._world.tick,
            "dead": self._world.dead,
            "death_reason": self._world.death_reason,
            "stats": dict(self._world.stats),
        }

    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        cmd = message.get("cmd")
        if cmd == "reset":
            config = SurvivalBoxConfig.from_dict(message.get("config") or {})
            self._world = SimulatedWorld(config, seed=int(message.get("seed", 0)))
            return {"ok": True, **self._status()}

        if cmd == "step":
            if self._world is None:
                return {"ok": False, "error": "step before reset"}
            spec = message.get("action") or {}
            action = Action.make(spec.get("name", "NULL"), **(spec.get("params") or {}))
            events = self._world.step(action)
            return {"ok": True, "events": events, **self._status()}

        if cmd == "observe":
            if self._world is None:
                return {"ok": False, "error": "observe before reset"}
            obs = build_observation(self._world, float(message.get("timestamp", 0.0)))
            return {
                "ok": True,
                "observation": {"tick": obs.tick, "data": obs.data, "frame": obs.frame},
            }

        if cmd == "close":
            return {"ok": True, "_close": True}

        return {"ok": False, "error": f"unknown command {cmd!r}"}


def main() -> None:
    bridge = SimBridge()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            response: Dict[str, Any] = {"ok": False, "error": f"bad json: {exc}"}
        else:
            response = bridge.handle(message)
        closing = response.pop("_close", False)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
        if closing:
            break


if __name__ == "__main__":
    main()
