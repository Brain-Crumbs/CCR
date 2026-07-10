"""Post-run review report (issue #33, Phase F "live childhood run protocol"):
one command that, after a run, summarizes the run's episodes, compares them
against baseline sessions recorded on the same curriculum step, and shows
per-episode detail -- the loop-closing step between a curriculum run and
deciding whether to advance to the next step.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from cognitive_runtime.programs.minecraft.evaluation import comparison_table, summarize_episodes
from cognitive_runtime.runtime.recorder import EpisodeSummary
from cognitive_runtime.tools.episode_viewer import view_episode
from cognitive_runtime.tools.metrics_dashboard import load_summaries

#: Same shape as the dashboard's comparison columns, plus `session` so a run
#: and its baselines are distinguishable in one table.
_REVIEW_COLUMNS = [
    "session", "episodes", "avg_survival_ticks", "death_rate", "success_rate",
    "avg_total_reward", "reward_per_minute", "avg_risk", "avg_prediction_error",
    "avg_novelty",
]


def _baseline_rows(
    record_dir: str, exclude_session_id: str, curriculum: Optional[str]
) -> List[Dict]:
    """One row per policy, aggregated over every *other* session directory
    under `record_dir` recorded on the same curriculum -- the baselines a
    live run should be judged against before increasing difficulty."""
    by_policy: Dict[str, List[EpisodeSummary]] = {}
    if not os.path.isdir(record_dir):
        return []
    for name in sorted(os.listdir(record_dir)):
        if name == exclude_session_id:
            continue
        session_dir = os.path.join(record_dir, name)
        if not os.path.isdir(session_dir):
            continue
        for summary in load_summaries(session_dir):
            if summary.curriculum != curriculum:
                continue
            by_policy.setdefault(summary.policy_name, []).append(summary)
    rows = []
    for policy_name in sorted(by_policy):
        row = summarize_episodes(by_policy[policy_name])
        row["session"] = f"baseline:{policy_name}"
        rows.append(row)
    return rows


def review_run(
    session_dir: str,
    record_dir: str = "sessions",
    episode: Optional[str] = None,
    tail: int = 3,
) -> str:
    """Summarize `session_dir`, compare it against baseline sessions on the
    same curriculum under `record_dir`, and show per-episode detail for
    `episode` (or the run's last `tail` episodes when not given)."""
    session_id = os.path.basename(os.path.normpath(session_dir))
    summaries = load_summaries(session_dir)
    if not summaries:
        return f"(no recorded episodes under {session_dir})"

    curriculum = summaries[0].curriculum
    run_row = summarize_episodes(summaries)
    run_row["session"] = session_id
    lines = [
        f"=== review: {session_id} (curriculum={curriculum or '-'}) ===",
        comparison_table([run_row], columns=_REVIEW_COLUMNS),
    ]

    baseline_rows = _baseline_rows(record_dir, session_id, curriculum)
    lines.append("")
    if baseline_rows:
        lines.append(f"baseline sessions on curriculum={curriculum or '-'!r}:")
        lines.append(comparison_table(baseline_rows, columns=_REVIEW_COLUMNS))
    else:
        lines.append(
            f"(no baseline sessions found for curriculum={curriculum or '-'!r} "
            f"under {record_dir})"
        )

    episode_ids = sorted(s.episode_id for s in summaries)
    chosen = [episode] if episode else episode_ids[-max(tail, 0):] if tail else []
    for episode_id in chosen:
        lines.append("")
        lines.append(view_episode(session_dir, episode_id, tail=10))
    return "\n".join(lines)
