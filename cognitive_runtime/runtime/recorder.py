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

A sensory event whose payload is an ndarray (a pixel frame) is a third case:
its bytes go to the binary :class:`~cognitive_runtime.runtime.frame_store.FrameStore`
under ``<session_dir>/frames/`` instead of either embedding or eliding, and
the line carries a small ``"frame_ref"`` (content hash) plus ``shape``/
``dtype`` instead of ``"payload"``.  This is purely additive to the streams-v2
schema -- a legacy line with an inline list payload, or an elided line, reads
back exactly as before.

`episode_XXXXX.decisions.jsonl` lines carry one cognitive tick each — this is
where NULL decisions are visible even though they emit no motor events.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, IO, List, Optional

import numpy as np

from cognitive_runtime.core.streams.bus import stream_matches
from cognitive_runtime.core.streams.events import StreamEvent
from cognitive_runtime.runtime.frame_store import (
    DEFAULT_DISK_BUDGET_BYTES,
    DEFAULT_SEGMENT_MAX_BYTES,
    DEFAULT_SEGMENT_MAX_SECONDS,
    FrameStore,
)

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
    # World-model prediction for this tick (issue #26): `risk` is always the
    # `TrendWorldModel` heuristic 0..1 unless a learned bridge (e.g.
    # `NeuralWorldModel`) supplied the additive Phase-D fields, in which case
    # they carry the learned p_death/prediction_error too.  `None` for every
    # session recorded before this field existed and for the heuristic's
    # unset additive fields.
    risk: float = 0.0
    p_death: Optional[float] = None
    prediction_error: Optional[float] = None
    # Deterministic attention controller (issue #59): the tick's full
    # `AttentionState.to_dict()` (mode, per-stream weights, selected streams,
    # focus stream, budget used/total, per-stream reason breakdown), or
    # `None` for every session recorded before this field existed. The
    # reason breakdown is what makes attention debuggable -- it is recorded
    # here rather than reconstructed after the fact.
    attention: Optional[Dict[str, Any]] = None
    # Scripted orienting reflex (issue #60): `OrientingDecision.to_dict()`
    # (`reason="orienting_reflex"`, the triggering stimulus stream, its
    # direction hint, and ticks remaining in the hold) when the reflex fired
    # this tick and substituted its look/turn action for the policy's;
    # `None` on every other tick and for every session recorded before this
    # field existed.
    reflex: Optional[Dict[str, Any]] = None
    #: The arbiter's three-mode switch (issue #95): `Arbiter.as_payload()`
    #: (mode, the (surprise, pain) reading that drove it, and the surprise
    #: calibrator's current calibration error), every tick; `None` for
    #: every session recorded before this field existed.
    arbiter_mode: Optional[Dict[str, Any]] = None
    #: Full motor-stack efference record (issue #168):
    #: ``MotorDecision.to_dict()`` carrying voluntary, reflex,
    #: caregiver_override, and actuated for every tick the organism-motor
    #: policy drives; ``None`` when the policy is not a
    #: ``MotorFreedomPolicy`` or for sessions recorded before this field.
    motor_decision: Optional[Dict[str, Any]] = None
    #: Decoded multi-horizon frames produced by the live cortex on this tick.
    #: The clinic aggregates these records into ``pixel-predictions-v1`` so a
    #: live run is inspectable without a separate offline export.
    live_prediction: Optional[Dict[str, Any]] = None

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
    realtime: bool = False
    #: Named curriculum preset (issue #30), if the run used one; `None` for a
    #: plain run and for episode summaries recorded before this field existed.
    curriculum: Optional[str] = None
    #: Ordered stage position within the curriculum runner's definition
    #: (issue #43), if any; `None` for a plain run, a bare `--curriculum`
    #: preset run, and every episode summary recorded before this field
    #: existed.
    curriculum_stage_index: Optional[int] = None
    # Runtime health: events/sec per stream_id, total counts, and streams that
    # fell silent.
    stream_event_rates: Dict[str, float] = field(default_factory=dict)
    stream_event_counts: Dict[str, int] = field(default_factory=dict)
    silent_streams: list = field(default_factory=list)
    # Realtime multi-rate health (Phase 5): missed-window accounting, stale
    # (stopped-publisher) streams, motor cadence, bounded-queue overflows, and
    # measured wall-clock rates (populated in realtime mode only).
    empty_windows: int = 0
    late_windows: int = 0
    stale_streams: list = field(default_factory=list)
    motor_emissions: int = 0
    motor_emission_rate: float = 0.0
    stream_overflow_counts: Dict[str, Any] = field(default_factory=dict)
    stream_wallclock_rates: Dict[str, float] = field(default_factory=dict)
    program_stats: Dict[str, Any] = field(default_factory=dict)
    # World-model prediction health (issue #26), averaged over the episode's
    # decisions; `avg_prediction_error` is `None` when no decision this
    # episode carried one (the heuristic `TrendWorldModel`, or episodes
    # recorded before this field existed).
    avg_risk: float = 0.0
    avg_prediction_error: Optional[float] = None
    # Combined novelty score (issue #27): world-model prediction error +
    # entity-persistence surprise, averaged over ticks where at least one was
    # available; `None` when neither ever fired this episode (no learned
    # world model and no occluded tracked entities), and for every episode
    # recorded before this field existed.
    avg_novelty: Optional[float] = None
    # Deterministic attention controller (issue #59), averaged/counted over
    # ticks where attention ran in "budgeted" mode; `None`/empty for
    # `attention="off"` runs and for episodes recorded before this field
    # existed.
    avg_attention_budget_used: Optional[float] = None
    #: stream_id -> number of ticks it held the hysteresis-protected focus
    #: this episode -- the attention timeline's per-episode summary.
    attention_focus_counts: Dict[str, int] = field(default_factory=dict)
    #: "off" or "budgeted" (issue #59); "off" for every episode recorded
    #: before this field existed.
    attention_mode: str = "off"
    #: Scripted orienting reflex (issue #60): "on"/"off"/"learned-only";
    #: "off" for every episode recorded before this field existed.
    reflex_mode: str = "off"
    #: Ticks the reflex substituted its look/turn action for the policy's
    #: this episode; 0 for `reflex_mode != "on"` and for every episode
    #: recorded before this field existed.
    reflex_activations: int = 0
    #: The arbiter's mode timeline this episode (issue #95): mode name ->
    #: ticks it was active. Empty for every episode recorded before this
    #: field existed.
    arbiter_mode_counts: Dict[str, int] = field(default_factory=dict)
    #: The surprise calibrator's Expected Calibration Error at the end of
    #: this episode (issue #95, task 4); `None` before enough observations
    #: accumulated for a fit, and for every episode recorded before this
    #: field existed.
    surprise_calibration_error: Optional[float] = None
    #: The hippocampus's episodic seed store size at the end of this episode
    #: (issue #96) -- persists across episodes within a run, so this is a
    #: running total, not a per-episode count. 0 for every episode recorded
    #: before this field existed.
    hippocampus_seeds: int = 0
    #: Organism identity (issue #88): `RuntimeConfig.resolve_name()`'s
    #: result for the run this episode belongs to; `None` for every episode
    #: recorded before this field existed (dashboards group those as
    #: "legacy").
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def stream_event_to_log(
    event: StreamEvent,
    direction: str,
    elide_payload: bool = False,
    frame_store: Optional[FrameStore] = None,
) -> Dict[str, Any]:
    """Serialize a StreamEvent to one streams.jsonl record.

    ``elide_payload`` drops the (bulky) payload but keeps the hash, so replay
    can still verify the event even though its content is not stored.  An
    ndarray payload that isn't elided goes to ``frame_store`` instead of being
    embedded inline: the line carries a small ``frame_ref`` (content hash)
    rather than the raw pixels.  If the store can't persist it (disk full even
    after evicting unpinned segments), the event degrades to hash-only rather
    than raising -- the hash was already computed either way.
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
    # Wall-clock arrival is metadata only (realtime mode) and never part of the
    # hash; it is omitted in fast-forward so those logs stay byte-identical.
    if event.arrived_at is not None:
        record["arrived_at"] = event.arrived_at
    if elide_payload:
        record["elided"] = True
    elif isinstance(event.payload, np.ndarray) and frame_store is not None:
        frame_ref = frame_store.write_frame(event.payload)
        if frame_ref is None:
            record["elided"] = True
        else:
            record["frame_ref"] = frame_ref
            record["shape"] = list(event.payload.shape)
            record["dtype"] = str(event.payload.dtype)
    else:
        record["payload"] = event.payload
    return record


def stream_event_from_log(
    record: Dict[str, Any], frame_store: Optional[FrameStore] = None
) -> StreamEvent:
    """Rebuild a StreamEvent from a full streams.jsonl record.

    Raises ``KeyError`` on hash-only (elided) lines, and on ``frame_ref``
    lines when no ``frame_store`` is given — neither carries enough to
    round-trip without one; callers use the stored ``hash`` directly instead.
    """
    if record.get("elided"):
        raise KeyError("elided stream record has no payload to reconstruct")
    if "frame_ref" in record:
        if frame_store is None:
            raise KeyError(
                "frame-referenced stream record requires a FrameStore to "
                "reconstruct; open one with frame_store.open_frame_store(session_dir)"
            )
        payload: Any = frame_store.read_frame(record["frame_ref"])
    else:
        payload = record["payload"]
    return StreamEvent(
        stream_id=record["stream_id"],
        modality=record["modality"],
        timestamp=record.get("timestamp", 0.0),
        sequence_number=record.get("seq", 0),
        payload=payload,
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
        pin_on_streams: Optional[List[str]] = None,
        frame_segment_max_bytes: int = DEFAULT_SEGMENT_MAX_BYTES,
        frame_segment_max_seconds: float = DEFAULT_SEGMENT_MAX_SECONDS,
        frame_disk_budget_bytes: Optional[int] = DEFAULT_DISK_BUDGET_BYTES,
    ):
        self.session_id = session_id
        self.session_dir = os.path.join(record_dir, session_id)
        self.record_streams = list(record_streams) if record_streams else ["*"]
        self.exclude_streams = list(exclude_streams or [])
        #: Glob patterns; a sensory event on a matching stream this tick pins
        #: the frame store's current rolling segment (deaths, damage, ...) so
        #: it survives rotation.
        self.pin_on_streams = list(pin_on_streams or [])
        os.makedirs(self.session_dir, exist_ok=True)
        self.frame_store = FrameStore(
            os.path.join(self.session_dir, "frames"),
            segment_max_bytes=frame_segment_max_bytes,
            segment_max_seconds=frame_segment_max_seconds,
            disk_budget_bytes=frame_disk_budget_bytes,
        )
        self._streams_file: Optional[IO[str]] = None
        self._decisions_file: Optional[IO[str]] = None
        self._episode_index = 0

    # -- payload filtering --------------------------------------------------

    def _elide(self, stream_id: str) -> bool:
        """True when a sensory stream's payload should be dropped (hash-only)."""
        if any(stream_matches(p, stream_id) for p in self.exclude_streams):
            return True
        return not any(stream_matches(p, stream_id) for p in self.record_streams)

    def _maybe_pin(self, sensory_events: List[StreamEvent]) -> None:
        """Pin the frame store's current segment when a high-value event
        (death, damage, ...) arrives this tick, so it survives rotation."""
        if not self.pin_on_streams:
            return
        for event in sensory_events:
            if any(stream_matches(p, event.stream_id) for p in self.pin_on_streams):
                self.frame_store.pin_current()
                return

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
                event, "sensory",
                elide_payload=self._elide(event.stream_id),
                frame_store=self.frame_store,
            )
            self._streams_file.write(
                json.dumps(record, separators=(",", ":"), default=str) + "\n"
            )
        self._maybe_pin(sensory_events)
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
        if self.frame_store is not None:
            self.frame_store.close()


class NullRecorder(Recorder):
    """Recorder that discards everything (for tests / throwaway runs)."""

    def __init__(self) -> None:  # noqa: super-init-not-called -- no disk I/O
        self.session_id = "null"
        self.session_dir = ""
        self.record_streams = ["*"]
        self.exclude_streams = []
        self.pin_on_streams = []
        self.frame_store = None
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
