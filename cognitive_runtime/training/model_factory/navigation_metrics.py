"""Held-out goal-navigation trajectory metrics (epic #212 Sec 12.1).

The evaluator consumes only frozen recorder artifacts.  Oracle labels remain
training/evaluation evidence and never enter the deployed sensory workspace.
All per-episode arrays follow the manifest order so paired Model Factory
comparisons retain their immutable episode identities.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


NAVIGATION_METRICS_FORMAT = "goal-navigation-metrics-v1"
CARDINAL_ACTIONS = frozenset(("MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT"))


def _decision_rows(session_path: Path) -> list[Mapping[str, Any]]:
    paths = sorted(session_path.glob("episode_*.decisions.jsonl"))
    if not paths:
        raise ValueError(f"navigation session has no decision log: {session_path}")
    rows: list[Mapping[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _episode_metrics(entry: Mapping[str, Any]) -> Dict[str, Any]:
    session_id = str(entry["session_id"])
    rows = _decision_rows(Path(str(entry["session_path"])))
    labelled = [row for row in rows if isinstance(row.get("oracle_labels"), Mapping)]
    if not labelled:
        raise ValueError(f"navigation session {session_id!r} has no oracle-labelled decisions")

    active = []
    success = False
    route_invalidated = False
    for row in labelled:
        labels = row["oracle_labels"]
        behavior = labels.get("behavior_mixture")
        if not isinstance(behavior, Mapping):
            raise ValueError(
                f"navigation session {session_id!r} lacks recorded behavior-mixture evidence"
            )
        if behavior.get("route_invalidated"):
            route_invalidated = True
        distance = labels.get("geodesic_distance")
        if distance is not None and math.isclose(float(distance), 0.0, abs_tol=1e-12):
            success = True
        if behavior.get("expert_action") is not None:
            if (
                behavior.get("selected_action") in CARDINAL_ACTIONS
                and not isinstance(behavior.get("selected_action_legal"), bool)
            ):
                raise ValueError(
                    f"navigation session {session_id!r} lacks pre-action legality evidence"
                )
            active.append((labels, behavior))
        if success:
            break

    initial_distance = None
    if active:
        initial_distance = active[0][1].get("pre_action_geodesic_distance")
    if initial_distance is None:
        raise ValueError(
            f"navigation session {session_id!r} lacks its pre-action initial geodesic distance"
        )
    step_cost = float(labelled[0]["oracle_labels"].get("map_cost_assumptions", {}).get("step_cost", 1.0))
    action_steps = sum(
        1 for _, behavior in active if behavior.get("selected_action") in CARDINAL_ACTIONS
    )
    optimal_steps = float(initial_distance) / step_cost if step_cost > 0 else 0.0
    geodesic_efficiency = (
        min(1.0, optimal_steps / action_steps)
        if success and action_steps > 0 else 0.0
    )
    collisions = sum(
        1
        for _, behavior in active
        if behavior.get("selected_action") in CARDINAL_ACTIONS
        and behavior.get("selected_action_legal") is False
    )
    collision_rate = collisions / action_steps if action_steps else 0.0
    return {
        "session_id": session_id,
        "scenario": str(entry.get("scenario", "unknown")),
        "success": float(success),
        "geodesic_efficiency": geodesic_efficiency,
        "collision_rate": collision_rate,
        "collision_count": collisions,
        "action_steps": action_steps,
        "replan_applicable": route_invalidated,
        "replan_recovery": float(success) if route_invalidated else None,
    }


def evaluate_navigation_sessions(
    session_entries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate success, efficiency, collisions, and recovery on held-out sessions."""
    if not session_entries:
        raise ValueError("goal-navigation evaluation requires held-out sessions")
    episodes = [_episode_metrics(entry) for entry in session_entries]
    recovery_values = [
        float(episode["replan_recovery"])
        for episode in episodes
        if episode["replan_applicable"]
    ]
    return {
        "format": NAVIGATION_METRICS_FORMAT,
        "evaluator": "frozen-heldout-trajectory-v1",
        "success_rate": statistics.fmean(episode["success"] for episode in episodes),
        "geodesic_efficiency": statistics.fmean(
            episode["geodesic_efficiency"] for episode in episodes
        ),
        "collision_rate": statistics.fmean(episode["collision_rate"] for episode in episodes),
        "replan_recovery": statistics.fmean(recovery_values) if recovery_values else None,
        "episode_ids": [episode["session_id"] for episode in episodes],
        "per_episode": {
            "success_rate": [episode["success"] for episode in episodes],
            "geodesic_efficiency": [episode["geodesic_efficiency"] for episode in episodes],
            "collision_rate": [episode["collision_rate"] for episode in episodes],
            "replan_recovery": [episode["replan_recovery"] for episode in episodes],
            "replan_applicable": [episode["replan_applicable"] for episode in episodes],
        },
        "episodes": episodes,
    }


__all__ = ["NAVIGATION_METRICS_FORMAT", "evaluate_navigation_sessions"]
