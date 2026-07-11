"""Inspect a recorded streams-v2 episode: stream throughput, reward
breakdown, event timeline, decision/action distribution, recent decisions."""

from __future__ import annotations

from typing import Any, Dict, List

from cognitive_runtime.core.modulation import INTERNAL_MODULATION_STREAM_IDS
from cognitive_runtime.core.streams.motor import MOTOR_COMMAND_STREAM
from cognitive_runtime.runtime.replay import (
    iter_cognitive_ticks,
    load_stream_log,
    load_summary,
)


def _motor_label(motor_records: List[Dict[str, Any]]) -> str:
    for record in motor_records:
        if record.get("stream_id") != MOTOR_COMMAND_STREAM:
            continue
        payload = record.get("payload")
        if isinstance(payload, dict) and isinstance(payload.get("action"), str):
            return payload["action"]
    return "NULL"


def view_episode(session_dir: str, episode_id: str, tail: int = 10) -> str:
    summary = load_summary(session_dir, episode_id)
    stream_log = load_stream_log(session_dir, episode_id)
    lines: List[str] = [f"=== {episode_id} ({session_dir}) ==="]

    if summary:
        for key in (
            "policy_name", "seed", "duration_ticks", "total_reward", "success",
            "termination_reason", "null_action_ticks", "avg_latency_ms",
            "ticks_per_second", "program_ticks_per_cognitive_tick",
            "avg_risk", "avg_prediction_error", "avg_novelty",
            "attention_mode", "avg_attention_budget_used",
        ):
            lines.append(f"  {key}: {summary.get(key)}")
        program_stats = summary.get("program_stats") or {}
        if program_stats:
            lines.append("  program_stats:")
            for key, value in program_stats.items():
                lines.append(f"    {key}: {value}")
        focus_counts = summary.get("attention_focus_counts") or {}
        if focus_counts:
            lines.append("  attention focus (ticks held, issue #59):")
            for stream_id, count in sorted(focus_counts.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {stream_id}: {count}")

    # Per-stream event counts and rates.
    counts = summary.get("stream_event_counts") or {}
    rates = summary.get("stream_event_rates") or {}
    if counts or rates:
        lines.append("  streams (count, events/sec):")
        for stream_id in sorted(set(counts) | set(rates)):
            lines.append(
                f"    {stream_id}: {counts.get(stream_id, 0)} "
                f"({rates.get(stream_id, 0.0)}/s)"
            )
    silent = summary.get("silent_streams") or []
    if silent:
        lines.append(f"  silent streams: {', '.join(silent)}")

    # Reward component totals (from reward.scalar payloads).
    component_totals: Dict[str, float] = {}
    event_timeline: List[str] = []
    for record in stream_log:
        stream_id = record.get("stream_id", "")
        if stream_id == "reward.scalar" and isinstance(record.get("payload"), dict):
            for name, value in (record["payload"].get("components") or {}).items():
                if isinstance(value, (int, float)):
                    component_totals[name] = round(
                        component_totals.get(name, 0.0) + float(value), 4
                    )
        elif stream_id.startswith("event.") and len(event_timeline) < 40:
            detail = record.get("payload") if not record.get("elided") else None
            event_timeline.append(
                f"t={record.get('timestamp')}: {stream_id}"
                + (f" {detail}" if detail else "")
            )

    lines.append("  reward components (episode totals):")
    for name, value in sorted(component_totals.items(), key=lambda kv: -abs(kv[1])):
        lines.append(f"    {name}: {value}")

    # Decision/action distribution (incl. NULL windows) and recent decisions.
    action_counts: Dict[str, int] = {}
    recent: List[str] = []
    health = hunger = None
    novelty = None
    value_estimate = None
    internal_values: Dict[str, float] = {}
    attention_timeline: List[str] = []
    last_focus_stream = object()  # sentinel: always logs the first tick's focus
    for decision, sensory, motor in iter_cognitive_ticks(session_dir, episode_id):
        for record in sensory:
            if record.get("elided"):
                continue
            stream_id = record.get("stream_id")
            if stream_id == "body.health":
                health = record.get("payload")
            elif stream_id == "body.hunger":
                hunger = record.get("payload")
            elif stream_id == "model.novelty":
                payload = record.get("payload")
                if isinstance(payload, dict):
                    novelty = payload.get("novelty")
            elif stream_id == "model.value_estimate":
                payload = record.get("payload")
                if isinstance(payload, dict):
                    value_estimate = payload.get("value_estimate")
            elif stream_id in INTERNAL_MODULATION_STREAM_IDS:
                payload = record.get("payload")
                if isinstance(payload, dict) and isinstance(payload.get("value"), (int, float)):
                    internal_values[stream_id] = payload["value"]
        action = _motor_label(motor)
        action_counts[action] = action_counts.get(action, 0) + 1
        line = (
            f"    tick {decision.get('tick_index')}: action={action} "
            f"reward={decision.get('reward_window_total')} hp={health} food={hunger} "
            f"risk={decision.get('risk', 0.0)}"
        )
        if decision.get("p_death") is not None:
            line += f" p_death={decision.get('p_death')}"
        if decision.get("prediction_error") is not None:
            line += f" pred_error={decision.get('prediction_error')}"
        if novelty is not None:
            line += f" novelty={novelty}"
        if value_estimate is not None:
            line += f" value_estimate={value_estimate}"
        for stream_id in INTERNAL_MODULATION_STREAM_IDS:
            if stream_id in internal_values:
                line += f" {stream_id}={internal_values[stream_id]}"
        attention = decision.get("attention")
        if isinstance(attention, dict) and attention.get("mode") == "budgeted":
            focus_stream = attention.get("focus_stream")
            line += f" attention_focus={focus_stream} budget={attention.get('budget_used')}"
            if focus_stream != last_focus_stream:
                reason = (attention.get("reasons") or {}).get(focus_stream, {})
                components = reason.get("components", {})
                top = sorted(components.items(), key=lambda kv: -abs(kv[1]))[:3]
                why = ", ".join(f"{name}={round(value, 3)}" for name, value in top)
                attention_timeline.append(
                    f"t={decision.get('tick_index')}: focus -> {focus_stream} ({why})"
                )
                last_focus_stream = focus_stream
        recent.append(line)

    lines.append("  action distribution:")
    for name, count in sorted(action_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {name}: {count}")

    if event_timeline:
        lines.append(f"  events ({len(event_timeline)} shown):")
        lines.extend(f"    {e}" for e in event_timeline)

    if attention_timeline:
        lines.append(f"  attention timeline (issue #59, {len(attention_timeline)} focus changes):")
        lines.extend(f"    {e}" for e in attention_timeline)

    if recent and tail > 0:
        lines.append(f"  last {min(tail, len(recent))} decisions:")
        lines.extend(recent[-tail:])
    return "\n".join(lines)
