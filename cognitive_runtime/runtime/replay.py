"""Episode replay.

Two capabilities:

1. Load recorded traces for inspection (episode viewer, dataset building).
2. Re-simulate: feed the recorded actions back into a fresh Program reset
   with the recorded seed and verify that every observation hash matches.
   If hashes diverge, determinism is broken and the session cannot be
   trusted for debugging or training.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cognitive_runtime.core.action import Action
from cognitive_runtime.core.program import Program


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
) -> ReplayResult:
    """Re-run the recorded actions through a fresh Program instance."""
    records, summary = load_episode(session_dir, episode_id)
    seed = int(summary.get("seed", 0))
    program.reset(seed=seed)

    matched = True
    first_divergence: Optional[int] = None
    reward_replayed = 0.0
    ticks = 0

    for record in records:
        observation = program.observe()
        if verify and observation.hash() != record["observation_hash"]:
            matched = False
            first_divergence = record["tick_id"]
            break
        action = Action.from_key(record["selected_action"])
        program.act(action)
        reward_replayed += program.reward().value
        ticks += 1
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
