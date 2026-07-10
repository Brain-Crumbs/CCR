"""Real-Minecraft backend over a line-delimited JSON bridge.

The runtime never changes to inhabit a real Minecraft server; only this
``SurvivalBackend`` implementation is new.  It speaks a tiny, transport-
agnostic JSON protocol to a **bridge** subprocess that owns the actual world
connection:

- the shipped bridge (``bridge/mineflayer/index.js``) drives a headless
  mineflayer client against a Java server, and
- a Python **fake bridge** (``bridge/fake/sim_bridge.py``) speaks the exact
  same protocol backed by the deterministic :class:`SimulatedWorld`, so the
  whole remote path is testable with no Minecraft and no Node.

Wire protocol (one JSON object per line, request → response):

    → {"cmd": "reset",   "seed": int, "config": {...}, "connection": {...}}
    ← {"ok": true, "tick": 0, "dead": false, "death_reason": null, "stats": {}}

    → {"cmd": "step",    "action": {"name": str, "params": {...}}}
    ← {"ok": true, "events": [str, ...], "tick": int, "dead": bool,
       "death_reason": str|null, "stats": {...}}

    → {"cmd": "observe", "timestamp": float}
    ← {"ok": true, "observation": {"tick": int, "data": {...}, "frame": [[int]]}}

    → {"cmd": "close"}
    ← {"ok": true}

``events`` is the same semantic vocabulary the simulated world emits
(``damage:<reason>``, ``new_item:<item>``, ``broke_block:<block>``,
``placed_block``, ``ate_food``, ``entered_shelter``, ``survived_night``,
``died``) so the stream publisher and reward function are unchanged.  A
response with ``"ok": false`` carries an ``"error"`` string.

The backend is **non-deterministic** and does not support snapshots, so its
recordings are excluded from replay-by-re-simulation (see
``runtime/replay.py``) while staying fully usable for ``view`` and ``train``.
"""

from __future__ import annotations

import atexit
import collections
import dataclasses
import json
import os
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.observation import Observation
from cognitive_runtime.core.program import RecoverableEpisodeError
from cognitive_runtime.programs.minecraft.backend import SurvivalBackend
from cognitive_runtime.programs.minecraft.config import SurvivalBoxConfig
from cognitive_runtime.programs.minecraft.world import pixels_from_frame

#: Environment overrides.  The bridge command is shell-split; the rest are
#: forwarded to the bridge in every ``reset`` so a live client knows where to
#: connect.  The fake bridge ignores the connection block.
ENV_BRIDGE_CMD = "CCR_MINECRAFT_BRIDGE_CMD"
_CONNECTION_ENV = {
    "host": "CCR_MINECRAFT_HOST",
    "port": "CCR_MINECRAFT_PORT",
    "username": "CCR_MINECRAFT_USERNAME",
    "version": "CCR_MINECRAFT_VERSION",
    "auth": "CCR_MINECRAFT_AUTH",
}


def default_bridge_command() -> List[str]:
    """The bridge command: ``$CCR_MINECRAFT_BRIDGE_CMD`` or the bundled Node
    mineflayer bridge."""
    override = os.environ.get(ENV_BRIDGE_CMD)
    if override:
        parts = shlex.split(override, posix=(os.name != "nt"))
        if os.name == "nt":
            # POSIX shlex treats backslashes as escapes, corrupting Windows
            # paths such as C:\Python314\python.exe. Non-POSIX shlex keeps the
            # paths intact but preserves wrapping quotes, so trim those here.
            parts = [p[1:-1] if len(p) >= 2 and p[0] == p[-1] == '"' else p for p in parts]
        return parts
    repo_root = Path(__file__).resolve().parents[3]
    return ["node", str(repo_root / "bridge" / "mineflayer" / "index.js")]


def connection_from_env() -> Dict[str, Any]:
    """Read connection settings from the environment (``port`` as int)."""
    conn: Dict[str, Any] = {}
    for key, env in _CONNECTION_ENV.items():
        value = os.environ.get(env)
        if value is None:
            continue
        conn[key] = int(value) if key == "port" else value
    return conn


class BridgeError(RecoverableEpisodeError):
    """A bridge subprocess failed, crashed, or returned an error response.

    Recoverable at the episode level (issue #33): the runtime ends the
    current episode and checkpoints rather than crashing the whole run --
    ``RemoteBridge.start()`` respawns a dead subprocess on the next
    ``reset()``, so the following episode can reconnect."""


class RemoteBridge:
    """A subprocess speaking the line-delimited JSON bridge protocol.

    One request in, one response out.  ``stderr`` is drained on a daemon
    thread into a bounded ring so a chatty bridge (mineflayer logs a lot)
    never deadlocks on a full pipe, and its tail is available for diagnostics
    when a request fails.
    """

    def __init__(self, command: List[str], stderr_lines: int = 50):
        if not command:
            raise ValueError("bridge command must not be empty")
        self.command = list(command)
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_tail: Deque[str] = collections.deque(maxlen=stderr_lines)
        self._stderr_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Spawn the bridge subprocess, or a fresh one if the previous
        process has died (issue #33 crash-resume: a live run's next
        ``reset()`` reconnects instead of talking to a dead process)."""
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = None
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
            )
        except FileNotFoundError as exc:
            raise BridgeError(
                f"could not launch Minecraft bridge {self.command!r}: {exc}. "
                f"Set ${ENV_BRIDGE_CMD} or install the bridge "
                "(see bridge/mineflayer/README.md)."
            ) from exc
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="mc-bridge-stderr", daemon=True
        )
        self._stderr_thread.start()
        atexit.register(self.close)

    def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr_tail.append(line.rstrip("\n"))

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                try:
                    proc.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, ValueError, OSError):
                    pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

    # -- request/response ---------------------------------------------------

    def request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send one command and return its parsed response; raises on any
        protocol, crash, or ``ok: false`` error."""
        if self._proc is None:
            raise BridgeError("bridge not started; call start() first")
        if self._proc.poll() is not None:
            raise BridgeError(self._crash_message())
        assert self._proc.stdin is not None and self._proc.stdout is not None
        with self._lock:
            try:
                self._proc.stdin.write(json.dumps(message) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise BridgeError(self._crash_message()) from exc
            line = self._proc.stdout.readline()
        if line == "":
            raise BridgeError(self._crash_message())
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BridgeError(f"bridge returned non-JSON line: {line!r}") from exc
        if not response.get("ok", False):
            raise BridgeError(
                f"bridge error for {message.get('cmd')!r}: "
                f"{response.get('error', 'unknown error')}"
            )
        return response

    def _crash_message(self) -> str:
        code = self._proc.poll() if self._proc is not None else None
        tail = "\n".join(self._stderr_tail) or "(no stderr)"
        return (
            f"Minecraft bridge exited (code={code}) or is not responding.\n"
            f"command: {' '.join(self.command)}\n"
            f"stderr tail:\n{tail}"
        )


class RemoteMinecraftBackend(SurvivalBackend):
    """SurvivalBackend backed by a bridge subprocess (real Minecraft).

    Construction is cheap; the bridge is spawned lazily on the first
    :meth:`reset`, so an unavailable bridge fails at episode start with a
    clear message rather than at import time.  Per-tick status
    (``tick``/``dead``/``death_reason``/``stats``) is cached from each
    ``step`` response, so those accessors cost no extra round-trip.
    """

    deterministic = False       # a live server cannot be re-simulated
    supports_snapshots = False  # a live server cannot capture/restore state

    def __init__(
        self,
        config: SurvivalBoxConfig,
        bridge: Optional[RemoteBridge] = None,
        connection: Optional[Dict[str, Any]] = None,
    ):
        self._config = config
        self._bridge = bridge  # injected bridge (tests); else spawned on reset
        self._owns_bridge = bridge is None
        self._connection = connection if connection is not None else connection_from_env()
        self._tick = 0
        self._dead = False
        self._death_reason: Optional[str] = None
        self._stats: Dict[str, Any] = {}

    # -- lifecycle ----------------------------------------------------------

    def _ensure_bridge(self) -> RemoteBridge:
        if self._bridge is None:
            self._bridge = RemoteBridge(default_bridge_command())
        self._bridge.start()
        return self._bridge

    def reset(self, seed: int) -> None:
        bridge = self._ensure_bridge()
        response = bridge.request(
            {
                "cmd": "reset",
                "seed": seed,
                "config": dataclasses.asdict(self._config),
                "connection": self._connection,
            }
        )
        self._absorb_status(response)

    def close(self) -> None:
        if self._bridge is not None and self._owns_bridge:
            self._bridge.close()

    # -- stepping -----------------------------------------------------------

    def step(self, action: Action) -> List[str]:
        bridge = self._require_bridge()
        response = bridge.request(
            {
                "cmd": "step",
                "action": {"name": action.name, "params": dict(action.params)},
            }
        )
        self._absorb_status(response)
        events = response.get("events", [])
        return [str(e) for e in events]

    def observe(self, timestamp: float) -> Observation:
        bridge = self._require_bridge()
        response = bridge.request({"cmd": "observe", "timestamp": timestamp})
        obs = response.get("observation") or {}
        frame = obs.get("frame")
        # Prefer a bridge-supplied RGB frame (e.g. a future prismarine-viewer
        # screenshot); otherwise colorize the semantic grid the same way the
        # simulated backend does, so the neural pixel stream works either way.
        pixels = obs.get("pixels")
        if pixels is None and frame is not None:
            pixels = pixels_from_frame(frame)
        return Observation(
            timestamp=timestamp,
            tick=obs.get("tick", self._tick),
            data=obs.get("data", {}),
            frame=frame,
            pixels=pixels,
        )

    # -- cached status ------------------------------------------------------

    def tick(self) -> int:
        return self._tick

    def is_dead(self) -> bool:
        return self._dead

    def death_reason(self) -> Optional[str]:
        return self._death_reason

    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    # -- unsupported on a live world ---------------------------------------

    def snapshot(self) -> str:
        raise NotImplementedError("the remote Minecraft backend cannot snapshot a live world")

    def restore(self, snapshot_id: str) -> None:
        raise NotImplementedError("the remote Minecraft backend cannot restore a live world")

    # -- helpers ------------------------------------------------------------

    def _require_bridge(self) -> RemoteBridge:
        if self._bridge is None:
            raise BridgeError("reset() must precede step()/observe()")
        return self._bridge

    def _absorb_status(self, response: Dict[str, Any]) -> None:
        self._tick = int(response.get("tick", self._tick))
        self._dead = bool(response.get("dead", self._dead))
        self._death_reason = response.get("death_reason")
        self._stats = dict(response.get("stats") or {})
