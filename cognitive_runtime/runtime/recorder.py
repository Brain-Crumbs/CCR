"""Session recorder (streams-v2).

The recorded artifact of a session is the **stream log**, not reconstructed
observations: every sensory and motor :class:`StreamEvent` is written in
bus-drain order, so replay can reset the program with the recorded seed,
re-inject the motor stream tick-aligned, and verify that every re-generated
sensory event hash matches.

Layout on disk (per session directory):

    <record_dir>/<session_id>/
        session.json                metadata + full stream catalog
        episode_00000.streams.jsonl    one StreamEvent per line, both directions
        episode_00000.decisions.jsonl  one cognitive tick per line
        episode_00000.summary.json     EpisodeSummary + per-stream counts/rates

`episode_XXXXX.streams.jsonl` lines::

    {"dir":"sensory"|"motor","stream_id":...,"modality":...,"timestamp":...,
     "seq":...,"payload":...,"confidence":...,"source":...,"hash":...}

Streams excluded from the log for size control keep a **hash-only** line
(``payload`` elided, ``"elided": true``) so replay verification stays complete.

`episode_XXXXX.decisions.jsonl` lines carry one cognitive tick each — this is
where NULL decisions are visible even though they emit no motor events.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, IO, List, Optional

from cognitive_runtime.core.streams.bus import stream_matches
from cognitive_runtime.core.streams.events import StreamEvent

RECORDING_FORMAT = "streams-v2"


@dataclass
class DecisionRecord:
    """One cognitive tick: the window it saw and the motor it emitted."""

    tick_index: int
    window_span: List[float]  # [started_at, ended_at] in simulated time
    n_events_by_stream: Dict[str, int]
    motor_emitted: List[str]  # StreamEvent hashes, or [] for a NULL decision
    policy_name: str
    latency_ms: float
    reward_window_total: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    program_ticks_per_cognitive_tick: int = 1
    # Runtime health: events/sec per stream_id, total counts, and streams that
    # fell silent.
    stream_event_rates: Dict[str, float] = field(default_factory=dict)
    stream_event_counts: Dict[str, int] = field(default_factory=dict)
    silent_streams: list = field(default_factory=list)
    program_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def stream_event_to_log(
    event: StreamEvent, direction: str, elide_payload: bool = False
) -> Dict[str, Any]:
    """Serialize a StreamEvent to one streams.jsonl record.

    ``elide_payload`` drops the (bulky) payload but keeps the hash, so replay
    can still verify the event even though its content is not stored.
    """
    record: Dict[str, Any] = {
        "dir": direction,
        "stream_id": event.stream_id,
        "modality": event.modality,
        "timestamp": event.timestamp,
        "seq": event.sequence_number,
        "confidence": event.confidence,
        "source": event.source,
        "hash": event.hash(),
    }
    if elide_payload:
        record["elided"] = True
    else:
        record["payload"] = event.payload
    return record


def stream_event_from_log(record: Dict[str, Any]) -> StreamEvent:
    """Rebuild a StreamEvent from a full streams.jsonl record.

    Raises ``KeyError`` on hash-only (elided) lines — they carry no payload and
    cannot round-trip; callers use their stored ``hash`` directly instead.
    """
    if record.get("elided"):
        raise KeyError("elided stream record has no payload to reconstruct")
    return StreamEvent(
        stream_id=record["stream_id"],
        modality=record["modality"],
        timestamp=record.get("timestamp", 0.0),
        sequence_number=record.get("seq", 0),
        payload=record["payload"],
        confidence=record.get("confidence", 1.0),
        source=record.get("source", ""),
    )


class Recorder:
    def __init__(
        self,
        record_dir: str,
        session_id: str,
        record_streams: Optional[List[str]] = None,
        exclude_streams: Optional[List[str]] = None,
    ):
        self.session_id = session_id
        self.session_dir = os.path.join(record_dir, session_id)
        self.record_streams = list(record_streams) if record_streams else ["*"]
        self.exclude_streams = list(exclude_streams or [])
        os.makedirs(self.session_dir, exist_ok=True)
        self._streams_file: Optional[IO[str]] = None
        self._decisions_file: Optional[IO[str]] = None
        self._episode_index = 0

    # -- payload filtering --------------------------------------------------

    def _elide(self, stream_id: str) -> bool:
        """True when a sensory stream's payload should be dropped (hash-only)."""
        if any(stream_matches(p, stream_id) for p in self.exclude_streams):
            return True
        return not any(stream_matches(p, stream_id) for p in self.record_streams)

    # -- session / episode lifecycle ---------------------------------------

    def write_session_metadata(self, metadata: Dict[str, Any]) -> None:
        payload = {"format": RECORDING_FORMAT, **metadata}
        path = os.path.join(self.session_dir, "session.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

    def start_episode(self, episode_index: int) -> str:
        self.end_episode_file()
        self._episode_index = episode_index
        episode_id = f"episode_{episode_index:05d}"
        base = os.path.join(self.session_dir, episode_id)
        self._streams_file = open(f"{base}.streams.jsonl", "w", encoding="utf-8")
        self._decisions_file = open(f"{base}.decisions.jsonl", "w", encoding="utf-8")
        return episode_id

    def write_cognitive_tick(
        self,
        sensory_events: List[StreamEvent],
        motor_events: List[StreamEvent],
        decision: DecisionRecord,
    ) -> None:
        """Log one cognitive tick: its sensory window, its motor emission, and
        the decision line — in bus-drain order (sensory first, then motor)."""
        if self._streams_file is None or self._decisions_file is None:
            raise RuntimeError("start_episode() must precede write_cognitive_tick()")
        for event in sensory_events:
            record = stream_event_to_log(
                event, "sensory", elide_payload=self._elide(event.stream_id)
            )
            self._streams_file.write(
                json.dumps(record, separators=(",", ":"), default=str) + "\n"
            )
        for event in motor_events:
            record = stream_event_to_log(event, "motor")
            self._streams_file.write(
                json.dumps(record, separators=(",", ":"), default=str) + "\n"
            )
        self._decisions_file.write(
            json.dumps(decision.to_dict(), separators=(",", ":"), default=str) + "\n"
        )

    def write_summary(self, summary: EpisodeSummary) -> None:
        path = os.path.join(
            self.session_dir, f"episode_{self._episode_index:05d}.summary.json"
        )
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(summary.to_dict(), fh, indent=2, default=str)

    def end_episode_file(self) -> None:
        for attr in ("_streams_file", "_decisions_file"):
            fh = getattr(self, attr)
            if fh is not None:
                fh.close()
                setattr(self, attr, None)

    def close(self) -> None:
        self.end_episode_file()


class NullRecorder(Recorder):
    """Recorder that discards everything (for tests / throwaway runs)."""

    def __init__(self) -> None:  # noqa: super-init-not-called -- no disk I/O
        self.session_id = "null"
        self.session_dir = ""
        self.record_streams = ["*"]
        self.exclude_streams = []
        self._streams_file = None
        self._decisions_file = None
        self._episode_index = 0

    def write_session_metadata(self, metadata: Dict[str, Any]) -> None:
        pass

    def start_episode(self, episode_index: int) -> str:
        self._episode_index = episode_index
        return f"episode_{episode_index:05d}"

    def write_cognitive_tick(
        self,
        sensory_events: List[StreamEvent],
        motor_events: List[StreamEvent],
        decision: DecisionRecord,
    ) -> None:
        pass

    def write_summary(self, summary: EpisodeSummary) -> None:
        pass

    def end_episode_file(self) -> None:
        pass
