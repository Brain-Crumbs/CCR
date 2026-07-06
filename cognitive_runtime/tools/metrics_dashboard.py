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


def _per_stream_rate_table(summaries: List[EpisodeSummary]) -> str:
    """Average events/sec per stream_id across every episode."""
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for summary in summaries:
        for stream_id, rate in (summary.stream_event_rates or {}).items():
            totals[stream_id] = totals.get(stream_id, 0.0) + float(rate)
            counts[stream_id] = counts.get(stream_id, 0) + 1
    if not totals:
        return ""
    lines = ["", "per-stream average events/sec:"]
    for stream_id in sorted(totals, key=lambda s: -totals[s] / counts[s]):
        lines.append(f"  {stream_id}: {round(totals[stream_id] / counts[stream_id], 3)}")
    return "\n".join(lines)


def dashboard(record_dir: str) -> str:
    """One row per policy, aggregated over every session under record_dir."""
    if not os.path.isdir(record_dir):
        return f"(no sessions directory at {record_dir})"
    by_policy: Dict[str, List[EpisodeSummary]] = {}
    all_summaries: List[EpisodeSummary] = []
    for session_id in sorted(os.listdir(record_dir)):
        session_dir = os.path.join(record_dir, session_id)
        if not os.path.isdir(session_dir):
            continue
        for summary in _load_summaries(session_dir):
            by_policy.setdefault(summary.policy_name, []).append(summary)
            all_summaries.append(summary)
    if not by_policy:
        return f"(no recorded episodes under {record_dir})"
    rows: List[Dict[str, Any]] = []
    for policy_name in sorted(by_policy):
        row = summarize_episodes(by_policy[policy_name])
        row["policy"] = policy_name
        rows.append(row)
    return comparison_table(rows) + "\n" + _per_stream_rate_table(all_summaries)
