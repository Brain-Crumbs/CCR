"""Aggregate metrics across recorded sessions, grouped by policy."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from cognitive_runtime.programs.minecraft.evaluation import comparison_table, summarize_episodes
from cognitive_runtime.runtime.recorder import EpisodeSummary


def _load_summaries(session_dir: str) -> List[EpisodeSummary]:
    summaries = []
    for name in sorted(os.listdir(session_dir)):
        if not name.endswith(".summary.json"):
            continue
        with open(os.path.join(session_dir, name), encoding="utf-8") as fh:
            raw = json.load(fh)
        known = {f for f in EpisodeSummary.__dataclass_fields__}  # type: ignore[attr-defined]
        summaries.append(EpisodeSummary(**{k: v for k, v in raw.items() if k in known}))
    return summaries


def dashboard(record_dir: str) -> str:
    """One row per policy, aggregated over every session under record_dir."""
    if not os.path.isdir(record_dir):
        return f"(no sessions directory at {record_dir})"
    by_policy: Dict[str, List[EpisodeSummary]] = {}
    for session_id in sorted(os.listdir(record_dir)):
        session_dir = os.path.join(record_dir, session_id)
        if not os.path.isdir(session_dir):
            continue
        for summary in _load_summaries(session_dir):
            by_policy.setdefault(summary.policy_name, []).append(summary)
    if not by_policy:
        return f"(no recorded episodes under {record_dir})"
    rows: List[Dict[str, Any]] = []
    for policy_name in sorted(by_policy):
        row = summarize_episodes(by_policy[policy_name])
        row["policy"] = policy_name
        rows.append(row)
    return comparison_table(rows)
