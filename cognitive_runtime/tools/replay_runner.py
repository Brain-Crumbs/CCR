"""Replay recorded episodes through a fresh Program and verify determinism."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cognitive_runtime.core.program import Program
from cognitive_runtime.programs.minecraft.adapter import MinecraftSurvivalBox
from cognitive_runtime.programs.minecraft.reward_profile import RewardProfile
from cognitive_runtime.runtime.replay import (
    ReplayResult,
    list_episodes,
    load_session_metadata,
    replay_episode,
    require_deterministic,
    require_streams_v2,
)


def _build_replay_program(
    metadata: Dict[str, Any], reward_profile: Optional[RewardProfile]
) -> Program:
    """Rebuild the Program a session was recorded against, keyed off
    ``session.json``'s ``program`` field (``ProgramMetadata.name`` --
    ``cli.py``'s ``--world`` selector determines which one recorded a given
    session). Reward profiles (``--reward-profile``) are Minecraft-only;
    Crafter has no reward-profile system (issue #90)."""
    program_config = metadata.get("program_config") or None
    program_name = metadata.get("program")
    if program_name == "MinecraftSurvivalBox":
        return MinecraftSurvivalBox(config=program_config, reward_profile=reward_profile)
    if program_name == "CrafterWorld":
        if reward_profile is not None:
            raise ValueError(
                "--reward-profile only applies to MinecraftSurvivalBox sessions "
                f"(this session's program is {program_name!r})"
            )
        from cognitive_runtime.programs.crafter.adapter import CrafterWorld

        return CrafterWorld(config=program_config)
    raise ValueError(f"unsupported program for replay: {program_name!r}")


def replay_session(
    session_dir: str,
    episode_id: str | None = None,
    verify: bool = True,
    reward_profile: Optional[RewardProfile] = None,
) -> List[ReplayResult]:
    metadata = load_session_metadata(session_dir)
    require_streams_v2(metadata)
    require_deterministic(metadata)
    # A session recorded with `--reward-profile` scores `reward.scalar`
    # through `ProfileRewardEngine`, not the default hard-coded
    # `SurvivalReward` -- rebuilding the Program without the same profile
    # would silently replay-verify against the wrong reward function
    # instead of failing loudly (issue #61 found this while wiring
    # intrinsic components through replay).
    recorded_profile_meta = metadata.get("reward_profile")
    if recorded_profile_meta and reward_profile is None:
        raise ValueError(
            f"session {session_dir!r} was recorded with reward profile "
            f"{recorded_profile_meta.get('name')!r} "
            f"(content_hash={recorded_profile_meta.get('content_hash')!r}); pass the same "
            "profile via --reward-profile to replay it correctly -- without it, replay "
            "would score reward.scalar against the wrong (default) reward function"
        )
    if reward_profile is not None:
        if not recorded_profile_meta:
            raise ValueError(
                f"session {session_dir!r} was recorded without a reward profile (the "
                "default SurvivalReward); drop --reward-profile to replay it"
            )
        if reward_profile.content_hash != recorded_profile_meta.get("content_hash"):
            raise ValueError(
                f"--reward-profile content_hash {reward_profile.content_hash!r} does not "
                f"match the session's recorded {recorded_profile_meta.get('content_hash')!r} "
                f"({recorded_profile_meta.get('name')!r}); replay needs the exact profile "
                "content the session was recorded with"
            )
    episodes = [episode_id] if episode_id else list_episodes(session_dir)
    results = []
    for episode in episodes:
        program = _build_replay_program(metadata, reward_profile)
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
