"""Human demonstration policy.

A human plays SurvivalBox through the terminal while the runtime records
observations and actions exactly as it does for any other policy -- the
recorded session becomes imitation-learning data.

Keypresses do not go straight to an ``Action``: they are published onto an
``input.keypress`` **stream** and consumed back off it, so the human demo
dogfoods the same asynchronous stream path a real backend uses.  In realtime
mode a reader thread feeds the stream while the cognitive loop polls it (the
human types whenever they like, at their own irregular rate); in blocking
mode each tick reads one line synchronously through the very same stream.

Controls (type a command, then Enter; empty input = NULL):

    w/s/a/d  move        W        sprint       x    sneak
    q/e      look l/r    r/f      look up/down j    jump
    k        attack      u        use          1-9  hotbar slot
    .        null        quit     end episode
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional

from cognitive_runtime.core.action import NULL_ACTION, Action
from cognitive_runtime.core.memory import Memory
from cognitive_runtime.core.perception import State
from cognitive_runtime.core.policy import SingleActionPolicy
from cognitive_runtime.core.streams.bus import SensoryStreamBus
from cognitive_runtime.core.streams.events import StreamSpec
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

#: The stream raw human input flows on.  ``block`` overflow: keypresses are
#: precious demonstration data and must never be dropped.
INPUT_KEYPRESS_STREAM = "input.keypress"
INPUT_KEYPRESS_SPEC = StreamSpec(
    INPUT_KEYPRESS_STREAM,
    "input",
    description="Raw human keypresses from the terminal (demonstrations).",
    payload_schema='{"key": str}',
    overflow="block",
)


def _terminal_input() -> Optional[str]:
    """Read one line from the terminal; ``None`` on EOF."""
    try:
        return input("action> ").strip()
    except EOFError:
        return None


class KeypressInputStream:
    """A thread-safe ``input.keypress`` stream fed by a line source.

    In realtime mode a daemon reader thread pumps lines onto the stream as the
    human types; in blocking mode :meth:`read_blocking` pumps one line per
    call.  Either way the consumer :meth:`poll`\\ s the stream — the same async
    publish/consume contract a real backend uses.
    """

    def __init__(self, input_source: Callable[[], Optional[str]], realtime: bool):
        self._source = input_source
        self._realtime = realtime
        self.bus = SensoryStreamBus(
            thread_safe=realtime,
            wall_clock=time.monotonic if realtime else None,
        )
        self.bus.register(INPUT_KEYPRESS_SPEC)
        self._seq = 0
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.eof = False

    def start(self) -> None:
        if self._realtime and self._thread is None:
            self._running = True
            self._thread = threading.Thread(
                target=self._read_loop, name="human-input", daemon=True
            )
            self._thread.start()

    def _publish(self, line: str) -> None:
        # Timestamp here is a private monotonic counter for ordering only; this
        # stream is the policy's own channel and never enters the replay log.
        self.bus.publish(INPUT_KEYPRESS_STREAM, {"key": line}, timestamp=float(self._seq))
        self._seq += 1

    def _read_loop(self) -> None:
        while self._running:
            line = self._source()
            if line is None:
                self.eof = True
                break
            self._publish(line)

    def read_blocking(self) -> None:
        """Blocking mode: read one line synchronously and publish it."""
        line = self._source()
        if line is None:
            self.eof = True
            return
        self._publish(line)

    def poll(self) -> List[str]:
        """Drain and return keypresses that have arrived since the last poll."""
        return [event.payload["key"] for event in self.bus.drain()]

    def stop(self) -> None:
        self._running = False


class HumanDemoPolicy(SingleActionPolicy):
    """Terminal-driven human policy over an ``input.keypress`` stream.

    ``realtime=False`` (the default, used by ``demo``) blocks each tick on one
    line of input; ``realtime=True`` reads asynchronously on a thread and maps
    whatever keypress last arrived (NULL when the human is idle this tick).
    """

    name = "human"

    def __init__(
        self,
        show_frame: bool = True,
        realtime: bool = False,
        input_source: Optional[Callable[[], Optional[str]]] = None,
    ):
        self.show_frame = show_frame
        self.realtime = realtime
        self._input_source = input_source or _terminal_input
        self.stop_requested = False
        self.input_stream = KeypressInputStream(self._input_source, realtime)

    def reset(self) -> None:
        self.stop_requested = False
        self.input_stream.stop()
        self.input_stream = KeypressInputStream(self._input_source, self.realtime)
        self.input_stream.start()
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
        # Blocking mode feeds exactly one line onto the stream this tick;
        # realtime mode lets the reader thread feed it whenever the human types.
        if not self.realtime:
            self.input_stream.read_blocking()
        keys = self.input_stream.poll()
        if self.input_stream.eof:
            self.stop_requested = True
            return NULL_ACTION
        if not keys:
            return NULL_ACTION  # idle this tick — an explicit NULL decision
        key = keys[-1]  # latest keypress wins (one action per tick)
        if key == "quit":
            self.stop_requested = True
            return NULL_ACTION
        return _KEYMAP.get(key, NULL_ACTION)
