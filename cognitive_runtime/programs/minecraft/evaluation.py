"""Survival evaluation metrics.

Computes the metric families from the MVP plan -- survival, competence,
behavior quality, runtime health -- from episode summaries and (optionally)
tick traces.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from cognitive_runtime.runtime.recorder import EpisodeSummary

TICK_RATE_ASSUMED = 20.0  # sim ticks per second, for per-minute rates


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_episodes(summaries: List[EpisodeSummary]) -> Dict[str, Any]:
    """Aggregate metrics for one policy across episodes."""
    if not summaries:
        return {}
    ticks = [s.duration_ticks for s in summaries]
    minutes = [t / TICK_RATE_ASSUMED / 60.0 for t in ticks]
    deaths = [s for s in summaries if s.termination_reason.startswith("death")]
    stats = [s.program_stats for s in summaries]

    def stat(name: str, default: float = 0.0) -> List[float]:
        return [float(p.get(name, default) or 0.0) for p in stats]

    return {
        "policy": summaries[0].policy_name,
        "episodes": len(summaries),
        # Survival.
        "avg_survival_ticks": round(_mean(ticks), 1),
        "avg_survival_minutes": round(_mean(minutes), 2),
        "death_rate": round(len(deaths) / len(summaries), 3),
        "damage_per_minute": round(
            _mean([d / m if m > 0 else 0.0 for d, m in zip(stat("damage_taken"), minutes)]), 2
        ),
        # Competence.
        "avg_unique_items": round(_mean(stat("unique_items_collected")), 2),
        "avg_blocks_broken": round(_mean(stat("blocks_broken")), 2),
        "avg_blocks_placed": round(_mean(stat("blocks_placed")), 2),
        "avg_food_consumed": round(_mean(stat("food_consumed")), 2),
        "nights_survived_rate": round(
            _mean([1.0 if p.get("survived_night") else 0.0 for p in stats]), 2
        ),
        # Behavior quality.
        "null_action_rate": round(
            _mean(
                [s.null_action_ticks / s.duration_ticks for s in summaries if s.duration_ticks]
            ),
            3,
        ),
        "avg_max_distance": round(_mean(stat("max_distance_from_spawn")), 2),
        "avg_total_reward": round(_mean([s.total_reward for s in summaries]), 3),
        "reward_per_minute": round(
            _mean(
                [s.total_reward / m if m > 0 else 0.0 for s, m in zip(summaries, minutes)]
            ),
            3,
        ),
        "success_rate": round(_mean([1.0 if s.success else 0.0 for s in summaries]), 3),
        # World-model prediction health (issue #26).
        "avg_risk": round(_mean([s.avg_risk for s in summaries]), 4),
        "avg_prediction_error": (
            round(_mean([s.avg_prediction_error for s in summaries if s.avg_prediction_error is not None]), 4)
            if any(s.avg_prediction_error is not None for s in summaries)
            else None
        ),
        # Runtime health.
        "avg_decision_latency_ms": round(_mean([s.avg_latency_ms for s in summaries]), 3),
        "avg_ticks_per_second": round(_mean([s.ticks_per_second for s in summaries]), 1),
        "missed_ticks": sum(s.missed_ticks for s in summaries),
        "cognitive_tick_ratio": summaries[0].program_ticks_per_cognitive_tick,
        # Stream throughput: total events/sec across all streams, averaged.
        "stream_events_per_sec": round(
            _mean([sum(s.stream_event_rates.values()) for s in summaries]), 1
        ),
    }


def comparison_table(rows: List[Dict[str, Any]], columns: List[str] | None = None) -> str:
    """Plain-text comparison table across policies."""
    if not rows:
        return "(no results)"
    columns = columns or [
        "policy", "episodes", "avg_survival_ticks", "death_rate", "success_rate",
        "avg_total_reward", "reward_per_minute", "avg_food_consumed",
        "avg_unique_items", "avg_blocks_placed", "null_action_rate",
        "avg_max_distance", "avg_decision_latency_ms", "stream_events_per_sec",
    ]
    widths = {
        c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns
    }
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))
    return "\n".join(lines)
