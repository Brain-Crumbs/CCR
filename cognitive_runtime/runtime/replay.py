"""Episode replay.

Two capabilities:

1. Load recorded traces for inspection (episode viewer, dataset building).
2. Re-simulate: feed the recorded motor emissions back into a fresh Program
   reset with the recorded seed and verify that every observation hash
   matches.  If hashes diverge, determinism is broken and the session
   cannot be trusted for debugging or training.

Replay mirrors the loop v2 exactly — driving the Program through
`step()` + the motor bus with the same one-tick actuation latency — so the
recorded trajectory reproduces byte-for-byte.  (Stream-native replay
verification against recorded stream hashes is Phase 3; this is the Phase-2
stopgap over the legacy tick records.)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.learner import window_reward
from cognitive_runtime.core.program import Program
from cognitive_runtime.core.streams import (
    MotorStreamBus,
    SensoryStreamBus,
    TickSynchronizer,
    publish_motor_command,
)


def load_session_metadata(session_dir: str) -> Dict[str, Any]:
    with open(os.path.join(session_dir, "session.json"), encoding="utf-8") as fh:
        return json.load(fh)


def list_episodes(session_dir: str) -> List[str]:
    episodes = []
    for name in sorted(os.listdir(session_dir)):
        if name.endswith(".jsonl"):
            episodes.append(name[: -len(".jsonl")])
    return episodes


def load_episode(session_dir: str, episode_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Returns (tick_records, summary)."""
    records = []
    with open(os.path.join(session_dir, f"{episode_id}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    summary_path = os.path.join(session_dir, f"{episode_id}.summary.json")
    summary: Dict[str, Any] = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
    return records, summary


@dataclass
class ReplayResult:
    episode_id: str
    ticks_replayed: int
    matched: bool
    first_divergence_tick: Optional[int] = None
    reward_recorded: float = 0.0
    reward_replayed: float = 0.0
    notes: List[str] = field(default_factory=list)


def replay_episode(
    program: Program,
    session_dir: str,
    episode_id: str,
    verify: bool = True,
    program_ticks_per_cognitive_tick: int = 1,
) -> ReplayResult:
    """Re-run the recorded motor emissions through a fresh Program instance.

    Mirrors loop v2: each cognitive tick steps the program `ratio` times
    (draining the motor bus, which carries the previous tick's recorded
    emission), rebuilds the compatibility observation, verifies its hash,
    then re-publishes this record's emission for the next step.
    """
    records, summary = load_episode(session_dir, episode_id)
    seed = int(summary.get("seed", 0))
    ratio = int(summary.get("program_ticks_per_cognitive_tick",
                            program_ticks_per_cognitive_tick))
    sensory_bus, motor_bus = SensoryStreamBus(), MotorStreamBus()
    program.attach_buses(sensory_bus, motor_bus)
    program.reset(seed=seed)
    synchronizer = TickSynchronizer(program_ticks_per_cognitive_tick=ratio)

    matched = True
    first_divergence: Optional[int] = None
    reward_replayed = 0.0
    ticks = 0

    for record in records:
        for _ in range(ratio):
            program.step()
        observation = program.observe()
        window = synchronizer.collect(sensory_bus, now=observation.timestamp)
        if verify and observation.hash() != record["observation_hash"]:
            matched = False
            first_divergence = record["tick_id"]
            break
        reward_replayed += window_reward(window)
        ticks += 1
        selected = record["selected_action"]
        if selected != "NULL":
            publish_motor_command(motor_bus, Action.from_key(selected), observation.timestamp)
        if program.is_complete() and ticks < len(records):
            # Recorded episode continued past completion: divergence.
            matched = False
            first_divergence = record["tick_id"]
            break

    reward_recorded = float(summary.get("total_reward", sum(r["reward"] for r in records)))
    notes = []
    if verify and matched and abs(reward_recorded - reward_replayed) > 1e-3:
        matched = False
        notes.append(
            f"reward mismatch: recorded={reward_recorded:.4f} replayed={reward_replayed:.4f}"
        )
    return ReplayResult(
        episode_id=episode_id,
        ticks_replayed=ticks,
        matched=matched if verify else True,
        first_divergence_tick=first_divergence,
        reward_recorded=round(reward_recorded, 4),
        reward_replayed=round(reward_replayed, 4),
        notes=notes,
    )
