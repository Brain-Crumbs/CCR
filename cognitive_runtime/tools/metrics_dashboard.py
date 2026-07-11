"""Aggregate metrics across recorded sessions, grouped by policy."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from cognitive_runtime.programs.minecraft.evaluation import comparison_table, summarize_episodes
from cognitive_runtime.runtime.recorder import EpisodeSummary


def load_summaries(session_dir: str) -> List[EpisodeSummary]:
    """Every recorded `EpisodeSummary` under one session directory (shared
    with `tools.review`, issue #33's post-run review command)."""
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


def _realtime_health(summaries: List[EpisodeSummary]) -> str:
    """Realtime multi-rate health, aggregated over realtime episodes only."""
    realtime = [s for s in summaries if getattr(s, "realtime", False)]
    if not realtime:
        return ""
    n = len(realtime)
    empty = sum(int(s.empty_windows) for s in realtime)
    late = sum(int(s.late_windows) for s in realtime)
    overflows = sum(
        sum(sum(policy.values()) for policy in (s.stream_overflow_counts or {}).values())
        for s in realtime
    )
    motor_rate = sum(float(s.motor_emission_rate) for s in realtime) / n
    stale = sorted({sid for s in realtime for sid in (s.stale_streams or [])})

    # Average measured wall-clock rate per stream across realtime episodes.
    wall_totals: Dict[str, float] = {}
    wall_counts: Dict[str, int] = {}
    for s in realtime:
        for sid, rate in (s.stream_wallclock_rates or {}).items():
            wall_totals[sid] = wall_totals.get(sid, 0.0) + float(rate)
            wall_counts[sid] = wall_counts.get(sid, 0) + 1

    lines = [
        "",
        f"realtime health ({n} realtime episode(s)):",
        f"  empty windows: {empty}   late windows: {late}   "
        f"queue overflows: {overflows}",
        f"  avg motor emission rate: {round(motor_rate, 3)}/s",
    ]
    if stale:
        lines.append(f"  stale streams: {', '.join(stale)}")
    if wall_totals:
        lines.append("  measured wall-clock rates (events/sec):")
        for sid in sorted(wall_totals, key=lambda s: -wall_totals[s] / wall_counts[s]):
            lines.append(f"    {sid}: {round(wall_totals[sid] / wall_counts[sid], 3)}")
    return "\n".join(lines)


def _attention_focus_table(summaries: List[EpisodeSummary]) -> str:
    """Total ticks each stream held the hysteresis-protected focus, across
    every `attention="budgeted"` episode (issue #59)."""
    totals: Dict[str, int] = {}
    for summary in summaries:
        if summary.attention_mode != "budgeted":
            continue
        for stream_id, count in (summary.attention_focus_counts or {}).items():
            totals[stream_id] = totals.get(stream_id, 0) + int(count)
    if not totals:
        return ""
    lines = ["", "attention focus totals (issue #59, budgeted episodes):"]
    for stream_id in sorted(totals, key=lambda s: -totals[s]):
        lines.append(f"  {stream_id}: {totals[stream_id]}")
    return "\n".join(lines)


#: Dashboard comparison columns: curriculum + policy identify the group,
#: everything else is `evaluation.comparison_table`'s default set.
_DASHBOARD_COLUMNS = [
    "curriculum", "policy", "episodes", "avg_survival_ticks", "death_rate",
    "success_rate", "avg_total_reward", "reward_per_minute", "avg_food_consumed",
    "avg_unique_items", "avg_blocks_placed", "null_action_rate",
    "avg_max_distance", "avg_decision_latency_ms", "stream_events_per_sec",
    "avg_risk", "avg_prediction_error", "avg_novelty",
    "attention_mode", "avg_attention_budget_used",
    "reflex_mode", "reflex_activations",
]


def _statistical_section(by_group: Dict[tuple, List[EpisodeSummary]]) -> str:
    """Mean +/- CI per group (issue #44), alongside the dashboard's plain-mean
    table -- the statistical evaluation harness's report, rendered here so
    recorded sessions get both views from one command."""
    from cognitive_runtime.training.statistical_evaluation import (
        compute_statistics, format_statistics_report,
    )

    stats = [
        compute_statistics(f"{policy} [{curriculum}]" if curriculum != "-" else policy, group)
        for (curriculum, policy), group in sorted(by_group.items())
    ]
    return "\nstatistical summary (mean +/- 95% CI, issue #44):\n" + format_statistics_report(stats)


def dashboard(record_dir: str, statistical: bool = False) -> str:
    """One row per (curriculum, policy) group, aggregated over every session
    under record_dir -- so a curriculum run (issue #30) is comparable across
    steps, and plain runs (curriculum=None) still group by policy alone.

    ``statistical=True`` appends the statistical evaluation harness's mean
    +/- confidence-interval report for the same groups."""
    if not os.path.isdir(record_dir):
        return f"(no sessions directory at {record_dir})"
    by_group: Dict[tuple, List[EpisodeSummary]] = {}
    all_summaries: List[EpisodeSummary] = []
    for session_id in sorted(os.listdir(record_dir)):
        session_dir = os.path.join(record_dir, session_id)
        if not os.path.isdir(session_dir):
            continue
        for summary in load_summaries(session_dir):
            key = (summary.curriculum or "-", summary.policy_name)
            by_group.setdefault(key, []).append(summary)
            all_summaries.append(summary)
    if not by_group:
        return f"(no recorded episodes under {record_dir})"
    rows: List[Dict[str, Any]] = []
    for curriculum, policy_name in sorted(by_group):
        row = summarize_episodes(by_group[(curriculum, policy_name)])
        row["policy"] = policy_name
        row["curriculum"] = curriculum
        rows.append(row)
    out = (
        comparison_table(rows, columns=_DASHBOARD_COLUMNS)
        + "\n"
        + _per_stream_rate_table(all_summaries)
        + _realtime_health(all_summaries)
        + _attention_focus_table(all_summaries)
    )
    if statistical:
        out += "\n" + _statistical_section(by_group)
    return out
