"""Compare policies across identical episode configurations."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from cognitive_runtime.core.policy import Policy
from cognitive_runtime.core.program import Program
from cognitive_runtime.programs.minecraft.evaluation import summarize_episodes
from cognitive_runtime.runtime.config import RuntimeConfig
from cognitive_runtime.runtime.loop import CognitiveRuntime
from cognitive_runtime.runtime.recorder import EpisodeSummary, NullRecorder


def run_policy(
    program: Program,
    policy: Policy,
    episodes: int,
    seed: int,
    max_ticks: int,
    record_dir: Optional[str] = None,
    session_id: Optional[str] = None,
) -> List[EpisodeSummary]:
    """Run one policy for `episodes` episodes with seeds seed..seed+n-1."""
    config = RuntimeConfig(
        episodes=episodes,
        seed=seed,
        max_ticks_per_episode=max_ticks,
        realtime=False,
        record=record_dir is not None,
        record_dir=record_dir or "sessions",
        session_id=session_id,
    )
    recorder = None if record_dir is not None else NullRecorder()
    runtime = CognitiveRuntime(program=program, policy=policy, config=config, recorder=recorder)
    return runtime.run()


def compare_policies(
    program_factory: Callable[[], Program],
    policy_factories: Dict[str, Callable[[], Policy]],
    episodes: int = 3,
    seed: int = 0,
    max_ticks: int = 6000,
) -> List[Dict[str, Any]]:
    """Every policy sees the same episode seeds; returns one metrics row each."""
    rows = []
    for name, make_policy in policy_factories.items():
        summaries = run_policy(
            program=program_factory(),
            policy=make_policy(),
            episodes=episodes,
            seed=seed,
            max_ticks=max_ticks,
        )
        row = summarize_episodes(summaries)
        row["policy"] = name
        rows.append(row)
    return rows
