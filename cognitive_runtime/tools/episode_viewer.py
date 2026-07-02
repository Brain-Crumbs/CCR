"""Inspect a recorded episode: summary, reward breakdown, key moments."""

from __future__ import annotations

from typing import Any, Dict, List

from cognitive_runtime.runtime.replay import load_episode


def view_episode(session_dir: str, episode_id: str, tail: int = 10) -> str:
    records, summary = load_episode(session_dir, episode_id)
    lines: List[str] = [f"=== {episode_id} ({session_dir}) ==="]

    if summary:
        for key in (
            "policy_name", "seed", "duration_ticks", "total_reward", "success",
            "termination_reason", "null_action_ticks", "avg_latency_ms",
            "ticks_per_second",
        ):
            lines.append(f"  {key}: {summary.get(key)}")
        program_stats = summary.get("program_stats") or {}
        if program_stats:
            lines.append("  program_stats:")
            for key, value in program_stats.items():
                lines.append(f"    {key}: {value}")

    # Reward component totals.
    component_totals: Dict[str, float] = {}
    action_counts: Dict[str, int] = {}
    event_ticks: List[str] = []
    for record in records:
        for name, value in (record.get("reward_components") or {}).items():
            component_totals[name] = round(component_totals.get(name, 0.0) + value, 4)
        action = record.get("selected_action", "?")
        action_counts[action] = action_counts.get(action, 0) + 1
        for event in record.get("events") or []:
            if not event.startswith("damage:") or len(event_ticks) < 50:
                event_ticks.append(f"tick {record['tick_id']}: {event}")

    lines.append("  reward components (episode totals):")
    for name, value in sorted(component_totals.items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"    {name}: {value}")
    lines.append("  action distribution:")
    for name, count in sorted(action_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {name}: {count}")
    if event_ticks:
        lines.append(f"  events ({min(len(event_ticks), 40)} shown of {len(event_ticks)}):")
        lines.extend(f"    {e}" for e in event_ticks[:40])
    if records and tail > 0:
        lines.append(f"  last {min(tail, len(records))} ticks:")
        for record in records[-tail:]:
            obs = (record.get("observation") or {}).get("data", {})
            lines.append(
                f"    tick {record['tick_id']}: action={record['selected_action']} "
                f"reward={record['reward']} hp={obs.get('health')} food={obs.get('hunger')}"
            )
    return "\n".join(lines)
