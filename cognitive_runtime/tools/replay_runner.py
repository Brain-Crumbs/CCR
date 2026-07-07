"""Replay recorded episodes through a fresh Program and verify determinism."""

from __future__ import annotations

from typing import List

from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.runtime.replay import (
    ReplayResult,
    list_episodes,
    load_session_metadata,
    replay_episode,
    require_deterministic,
    require_streams_v2,
)


def replay_session(session_dir: str, episode_id: str | None = None, verify: bool = True) -> List[ReplayResult]:
    metadata = load_session_metadata(session_dir)
    require_streams_v2(metadata)
    require_deterministic(metadata)
    if metadata.get("program") != "MinecraftSurvivalBox":
        raise ValueError(f"unsupported program for replay: {metadata.get('program')}")
    episodes = [episode_id] if episode_id else list_episodes(session_dir)
    results = []
    for episode in episodes:
        program = MinecraftSurvivalBox(config=metadata.get("program_config") or None)
        results.append(replay_episode(program, session_dir, episode, verify=verify))
    return results


def format_results(results: List[ReplayResult]) -> str:
    lines = []
    for r in results:
        status = "OK " if r.matched else "DIVERGED"
        line = (
            f"[{status}] {r.episode_id}: ticks={r.ticks_replayed} "
            f"reward recorded={r.reward_recorded} replayed={r.reward_replayed}"
        )
        if r.first_divergence_tick is not None:
            line += f" first_divergence=tick:{r.first_divergence_tick}"
            if r.first_divergence_stream is not None:
                line += f" stream:{r.first_divergence_stream}"
            if r.first_divergence_seq is not None:
                line += f" seq:{r.first_divergence_seq}"
        for note in r.notes:
            line += f" ({note})"
        lines.append(line)
    return "\n".join(lines)
