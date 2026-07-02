"""Session recorder.

Every tick of every episode is recorded so sessions are replayable and
debuggable.  Layout on disk:

    <record_dir>/<session_id>/
        session.json                metadata for the whole session
        episode_00000.jsonl         one tick record per line
        episode_00000.summary.json  episode summary
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, IO, Optional


@dataclass
class TickRecord:
    session_id: str
    episode_id: str
    tick_id: int
    timestamp: float
    observation_hash: str
    selected_action: str
    action_ok: bool
    reward: float
    reward_components: Dict[str, float]
    events: list
    policy_name: str
    latency_ms: float
    observation: Optional[Dict[str, Any]] = None  # structured obs, if enabled


@dataclass
class EpisodeSummary:
    session_id: str
    episode_id: str
    seed: int
    policy_name: str
    duration_ticks: int
    total_reward: float
    success: bool
    termination_reason: str
    null_action_ticks: int = 0
    avg_latency_ms: float = 0.0
    ticks_per_second: float = 0.0
    missed_ticks: int = 0
    program_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Recorder:
    def __init__(
        self,
        record_dir: str,
        session_id: str,
        record_observations: bool = True,
        record_frames: bool = False,
    ):
        self.session_id = session_id
        self.session_dir = os.path.join(record_dir, session_id)
        self.record_observations = record_observations
        self.record_frames = record_frames
        os.makedirs(self.session_dir, exist_ok=True)
        self._episode_file: Optional[IO[str]] = None
        self._episode_index = 0

    def write_session_metadata(self, metadata: Dict[str, Any]) -> None:
        path = os.path.join(self.session_dir, "session.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)

    def start_episode(self, episode_index: int) -> str:
        self.end_episode_file()
        self._episode_index = episode_index
        episode_id = f"episode_{episode_index:05d}"
        path = os.path.join(self.session_dir, f"{episode_id}.jsonl")
        self._episode_file = open(path, "w", encoding="utf-8")
        return episode_id

    def write_tick(self, record: TickRecord) -> None:
        if self._episode_file is None:
            raise RuntimeError("start_episode() must be called before write_tick()")
        payload = asdict(record)
        if payload.get("observation") is not None and not self.record_frames:
            payload["observation"].pop("frame", None)
        if not self.record_observations:
            payload.pop("observation", None)
        self._episode_file.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")

    def write_summary(self, summary: EpisodeSummary) -> None:
        path = os.path.join(
            self.session_dir, f"episode_{self._episode_index:05d}.summary.json"
        )
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(summary.to_dict(), fh, indent=2, default=str)

    def end_episode_file(self) -> None:
        if self._episode_file is not None:
            self._episode_file.close()
            self._episode_file = None

    def close(self) -> None:
        self.end_episode_file()


class NullRecorder(Recorder):
    """Recorder that discards everything (for tests / throwaway runs)."""

    def __init__(self) -> None:  # noqa: super-init-not-called -- no disk I/O
        self.session_id = "null"
        self.session_dir = ""
        self.record_observations = False
        self.record_frames = False
        self._episode_file = None
        self._episode_index = 0

    def write_session_metadata(self, metadata: Dict[str, Any]) -> None:
        pass

    def start_episode(self, episode_index: int) -> str:
        self._episode_index = episode_index
        return f"episode_{episode_index:05d}"

    def write_tick(self, record: TickRecord) -> None:
        pass

    def write_summary(self, summary: EpisodeSummary) -> None:
        pass

    def end_episode_file(self) -> None:
        pass
