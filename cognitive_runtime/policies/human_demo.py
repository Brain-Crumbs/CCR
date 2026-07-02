"""Human demonstration policy.

A human plays SurvivalBox through the terminal while the runtime records
observations and actions exactly as it does for any other policy -- the
recorded session becomes imitation-learning data.

Controls (type a command, then Enter; empty input = NULL):

    w/s/a/d  move        W        sprint       x    sneak
    q/e      look l/r    r/f      look up/down j    jump
    k        attack      u        use          1-9  hotbar slot
    .        null        quit     end episode
"""

from __future__ import annotations

from typing import Dict, Optional

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import Policy
from cognitive_runtime.core.world_model import Prediction

_KEYMAP: Dict[str, Action] = {
    "w": Action("MOVE_FORWARD"),
    "s": Action("MOVE_BACKWARD"),
    "a": Action("MOVE_LEFT"),
    "d": Action("MOVE_RIGHT"),
    "W": Action("SPRINT"),
    "x": Action("SNEAK"),
    "j": Action("JUMP"),
    "q": Action("LOOK_LEFT"),
    "e": Action("LOOK_RIGHT"),
    "r": Action("LOOK_UP"),
    "f": Action("LOOK_DOWN"),
    "k": Action("ATTACK"),
    "u": Action("USE"),
    ".": NULL_ACTION,
}
for _i in range(1, 10):
    _KEYMAP[str(_i)] = Action.make("SELECT_HOTBAR_SLOT", slot=_i - 1)

_FRAME_GLYPHS = {1: ".", 2: ",", 3: "~", 4: "≈", 5: "T", 6: "#", 7: "*", 8: "%", 9: "▒", 10: "█", 90: "Z", 99: "@"}


class HumanDemoPolicy(Policy):
    """Blocking terminal input each tick.  Use with realtime=False and a
    low tick budget; every tick waits for the human."""

    name = "human"

    def __init__(self, show_frame: bool = True):
        self.show_frame = show_frame
        self.stop_requested = False

    def reset(self) -> None:
        self.stop_requested = False
        print(__doc__)

    def decide(self, state: State, memory: Memory, prediction: Optional[Prediction]) -> Action:
        obs = state.observation.data
        if self.show_frame and state.observation.frame:
            for row in state.observation.frame:
                print("".join(_FRAME_GLYPHS.get(c, "?") for c in row))
        print(
            f"tick={state.observation.tick} hp={obs.get('health')} food={obs.get('hunger')} "
            f"night={obs.get('is_night')} front={obs.get('front_block')} "
            f"hotbar={obs.get('hotbar')} mobs={obs.get('mobs')}"
        )
        try:
            raw = input("action> ").strip()
        except EOFError:
            self.stop_requested = True
            return NULL_ACTION
        if raw == "quit":
            self.stop_requested = True
            return NULL_ACTION
        return _KEYMAP.get(raw, NULL_ACTION)
