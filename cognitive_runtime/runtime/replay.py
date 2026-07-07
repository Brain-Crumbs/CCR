"""Episode replay (streams-v2).

The recorded artifact is the stream log, so replay is stream-native:

1. Rebuild the Program from ``session.json`` (``program`` + ``program_config``)
   and ``reset(seed)`` with the recorded seed.
2. Per cognitive tick: inject the recorded **motor** events into the motor
   bus, ``step()`` the recorded number of program ticks, drain the **sensory**
   events and compare their hashes **in order** against the log.
3. Report the first divergence (stream_id + seq + tick) and compare the summed
   ``reward.scalar`` values.

Two tamper modes are both caught:

- A flipped **motor** payload makes the world step differently, so the
  regenerated sensory hashes diverge downstream.
- A flipped **sensory** payload no longer matches its own recorded hash
  (integrity check) — and, if the hash was flipped to match, it no longer
  matches the freshly regenerated event.

Sensory payloads elided for size (``exclude_streams``) are hash-only lines;
replay still verifies them against the regenerated event's hash.

Legacy sessions (no ``"format": "streams-v2"``) are rejected with a clear
message rather than silently misread.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

from cognitive_runtime.core.program import Program
from cognitive_runtime.core.streams import (
    MOTOR_COMMAND_STREAM,
    MotorStreamBus,
    SensoryStreamBus,
)
from cognitive_runtime.runtime.recorder import (
    RECORDING_FORMAT,
    stream_event_from_log,
)


class LegacyFormatError(RuntimeError):
    """Raised when a session predates the streams-v2 recording format."""


# --------------------------------------------------------------------- loading


def load_session_metadata(session_dir: str) -> Dict[str, Any]:
    with open(os.path.join(session_dir, "session.json"), encoding="utf-8") as fh:
        return json.load(fh)


def require_streams_v2(metadata: Dict[str, Any]) -> None:
    fmt = metadata.get("format")
    if fmt != RECORDING_FORMAT:
        raise LegacyFormatError(
            f"session is not {RECORDING_FORMAT!r} (format={fmt!r}); "
            "re-record it with the current runtime to replay it"
        )


def list_episodes(session_dir: str) -> List[str]:
    suffix = ".decisions.jsonl"
    return sorted(
        name[: -len(suffix)]
        for name in os.listdir(session_dir)
        if name.endswith(suffix)
    )


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_stream_log(session_dir: str, episode_id: str) -> List[Dict[str, Any]]:
    return _load_jsonl(os.path.join(session_dir, f"{episode_id}.streams.jsonl"))


def load_decisions(session_dir: str, episode_id: str) -> List[Dict[str, Any]]:
    return _load_jsonl(os.path.join(session_dir, f"{episode_id}.decisions.jsonl"))


def load_summary(session_dir: str, episode_id: str) -> Dict[str, Any]:
    path = os.path.join(session_dir, f"{episode_id}.summary.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def iter_cognitive_ticks(
    session_dir: str, episode_id: str
) -> Iterator[Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """Yield ``(decision, sensory_records, motor_records)`` per cognitive tick.

    The stream log is written in bus-drain order — a tick's sensory window
    first, then its motor emission — so each tick consumes exactly
    ``sum(n_events_by_stream)`` sensory lines followed by ``len(motor_emitted)``
    motor lines.
    """
    stream_log = load_stream_log(session_dir, episode_id)
    decisions = load_decisions(session_dir, episode_id)
    cursor = 0
    for decision in decisions:
        n_sensory = sum(decision.get("n_events_by_stream", {}).values())
        n_motor = len(decision.get("motor_emitted", []))
        sensory = stream_log[cursor : cursor + n_sensory]
        cursor += n_sensory
        motor = stream_log[cursor : cursor + n_motor]
        cursor += n_motor
        yield decision, sensory, motor


# ---------------------------------------------------------------------- replay


@dataclass
class ReplayResult:
    episode_id: str
    ticks_replayed: int
    matched: bool
    first_divergence_tick: Optional[int] = None
    first_divergence_stream: Optional[str] = None
    first_divergence_seq: Optional[int] = None
    reward_recorded: float = 0.0
    reward_replayed: float = 0.0
    notes: List[str] = field(default_factory=list)


def _window_reward(events: List[Any]) -> float:
    """Sum ``reward.scalar`` values over freshly regenerated stream events."""
    total = 0.0
    for event in events:
        if event.stream_id != "reward.scalar":
            continue
        payload = event.payload
        if isinstance(payload, dict) and isinstance(payload.get("value"), (int, float)):
            total += float(payload["value"])
    return total


def replay_episode(
    program: Program,
    session_dir: str,
    episode_id: str,
    verify: bool = True,
) -> ReplayResult:
    """Re-run recorded motor emissions through a fresh Program and verify that
    every regenerated sensory event hash matches the log, in order."""
    metadata = load_session_metadata(session_dir)
    require_streams_v2(metadata)
    summary = load_summary(session_dir, episode_id)
    seed = int(summary.get("seed", 0))
    ratio = int(metadata.get("program_ticks_per_cognitive_tick", 1))

    sensory_bus, motor_bus = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory_bus, motor_bus)
    # A realtime recording paced publication off simulated time; reproduce that
    # same pacing here (still fast-forward — no sleeping) so the paced sensory
    # subset regenerates bit-for-bit.  Fast-forward recordings leave it off.
    if metadata.get("realtime"):
        program.set_realtime(True)
    program.reset(seed=seed)

    matched = True
    result = ReplayResult(episode_id=episode_id, ticks_replayed=0, matched=True)
    reward_replayed = 0.0
    ticks = 0

    for decision, sensory_records, motor_records in iter_cognitive_ticks(
        session_dir, episode_id
    ):
        for _ in range(ratio):
            program.step()
        regenerated = sensory_bus.drain()
        tick_index = decision.get("tick_index", ticks)

        if verify:
            divergence = _verify_window(regenerated, sensory_records)
            if divergence is not None:
                stream_id, seq, note = divergence
                result.matched = False
                result.first_divergence_tick = tick_index
                result.first_divergence_stream = stream_id
                result.first_divergence_seq = seq
                result.notes.append(note)
                matched = False
                break

        reward_replayed += _window_reward(regenerated)
        ticks += 1

        # Re-inject this tick's recorded motor onto the bus; the next step()
        # applies it (one-tick actuation latency), exactly as during recording.
        for record in motor_records:
            motor_bus.publish(
                record.get("stream_id", MOTOR_COMMAND_STREAM),
                record.get("payload"),
                record.get("timestamp", 0.0),
                source=record.get("source", ""),
            )

    result.ticks_replayed = ticks
    reward_recorded = float(
        summary.get(
            "total_reward",
            sum(d.get("reward_window_total", 0.0) for d in load_decisions(session_dir, episode_id)),
        )
    )
    if verify and matched and abs(reward_recorded - reward_replayed) > 1e-3:
        result.matched = False
        result.notes.append(
            f"reward mismatch: recorded={reward_recorded:.4f} "
            f"replayed={reward_replayed:.4f}"
        )
    result.reward_recorded = round(reward_recorded, 4)
    result.reward_replayed = round(reward_replayed, 4)
    if not verify:
        result.matched = True
    return result


def _verify_window(
    regenerated: List[Any], recorded: List[Dict[str, Any]]
) -> Optional[Tuple[str, int, str]]:
    """Compare regenerated sensory events against the recorded lines in order.

    Returns ``(stream_id, seq, note)`` for the first divergence, or ``None``.
    """
    if len(regenerated) != len(recorded):
        return (
            "<count>",
            -1,
            f"event count mismatch: recorded={len(recorded)} replayed={len(regenerated)}",
        )
    for event, record in zip(regenerated, recorded):
        stored_hash = record.get("hash")
        # Integrity: a full (non-elided) line's payload must hash to its stored
        # hash — catches a tampered payload even if the world is unaffected.
        if not record.get("elided"):
            try:
                if stream_event_from_log(record).hash() != stored_hash:
                    return (
                        record.get("stream_id", "?"),
                        record.get("seq", -1),
                        "recorded payload does not match its stored hash",
                    )
            except KeyError:
                pass
        if event.hash() != stored_hash:
            return (
                event.stream_id,
                event.sequence_number,
                f"sensory hash diverged on {event.stream_id} seq={event.sequence_number}",
            )
    return None
