"""Semantic event labels and event-aware prediction metrics.

This module deliberately consumes semantic grids and semantic-decoder
probabilities.  Pixel colours are neither stable entity identifiers nor a
useful basis for evaluating a hallucinated entity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot
from typing import Any, Iterable, Mapping, Optional, Sequence

try:  # Keep this small evaluation utility importable without Crafter.
    from cognitive_runtime.programs.crafter.streams import SEMANTIC_LEGEND_NAMES
    COW_ID = next(code for code, name in SEMANTIC_LEGEND_NAMES.items() if name == "cow")
except ImportError:  # pragma: no cover - only for minimal installs
    COW_ID = 14

MOVEMENT_ACTIONS = frozenset({"MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_FORWARD", "MOVE_BACKWARD", "FORWARD", "BACKWARD"})


@dataclass(frozen=True)
class FrameEventLabels:
    cow_visible: bool
    cow_cells: tuple[tuple[int, int], ...]
    entity_entered: bool
    entity_left: bool
    position_changed: bool
    blocked_forward: bool
    semantic_scene_changed: bool


def _cow_cells(grid: Optional[Sequence[Sequence[int]]], cow_id: int) -> tuple[tuple[int, int], ...]:
    if grid is None:
        return ()
    return tuple((row, col) for row, values in enumerate(grid) for col, value in enumerate(values) if value == cow_id)


def extract_frame_event_labels(
    semantic_grids: Sequence[Optional[Sequence[Sequence[int]]]],
    *,
    positions: Optional[Sequence[Optional[tuple[float, float]]]] = None,
    actions: Optional[Sequence[Optional[str]]] = None,
    cow_id: int = COW_ID,
) -> list[FrameEventLabels]:
    """Label every observed frame; transition fields describe ``t-1 -> t``.

    A blocked transition requires an attempted movement, no positional change,
    and an unchanged semantic scene.  That last condition prevents idle/no
    action frames from being mislabeled as collisions.
    """
    labels: list[FrameEventLabels] = []
    positions = positions or [None] * len(semantic_grids)
    actions = actions or [None] * max(0, len(semantic_grids) - 1)
    for index, grid in enumerate(semantic_grids):
        cells = _cow_cells(grid, cow_id)
        previous = semantic_grids[index - 1] if index else None
        previous_cells = _cow_cells(previous, cow_id)
        scene_changed = bool(index and grid is not None and previous is not None and grid != previous)
        position_changed = bool(index and positions[index] is not None and positions[index - 1] is not None and positions[index] != positions[index - 1])
        attempted = bool(index and index - 1 < len(actions) and actions[index - 1] in MOVEMENT_ACTIONS)
        blocked = bool(index and attempted and not position_changed and not scene_changed)
        labels.append(FrameEventLabels(
            cow_visible=bool(cells), cow_cells=cells,
            entity_entered=bool(cells and not previous_cells),
            entity_left=bool(previous_cells and not cells),
            position_changed=position_changed, blocked_forward=blocked,
            semantic_scene_changed=scene_changed,
        ))
    return labels


def rollout_health(prediction_dispersion: float, target_dispersion: float, *, epsilon: float = 1e-6) -> dict[str, Any]:
    """Classify rollout dynamics without treating underdynamic as healthy."""
    if target_dispersion <= epsilon:
        return {"prediction_dispersion": prediction_dispersion, "target_dispersion": target_dispersion,
                "dispersion_ratio": None, "state": "not_evaluable", "frozen_rollout": False}
    ratio = prediction_dispersion / target_dispersion
    state = "healthy" if ratio >= .25 else "underdynamic" if ratio >= .05 else "frozen"
    return {"prediction_dispersion": prediction_dispersion, "target_dispersion": target_dispersion,
            "dispersion_ratio": ratio, "state": state, "frozen_rollout": state == "frozen"}


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return sum(values) / len(values) if values else None


def evaluate_entity_predictions(
    target_labels: Sequence[FrameEventLabels],
    cow_probabilities: Sequence[float],
    predicted_cells: Sequence[Sequence[tuple[int, int]]],
    *, threshold: float = .5,
) -> dict[str, Any]:
    """Return cow metrics for aligned target frames and semantic predictions."""
    n = min(len(target_labels), len(cow_probabilities), len(predicted_cells))
    tp = fp = fn = tn = 0
    localization: list[float] = []
    for target, probability, cells in zip(target_labels[:n], cow_probabilities[:n], predicted_cells[:n]):
        predicted = probability >= threshold
        if predicted and target.cow_visible: tp += 1
        elif predicted: fp += 1
        elif target.cow_visible: fn += 1
        else: tn += 1
        if cells and target.cow_cells:
            localization.append(min(hypot(a - c, b - d) for a, b in cells for c, d in target.cow_cells))
    entries = [i for i, label in enumerate(target_labels[:n]) if label.entity_entered]
    entry_latencies = []
    for i in entries:
        later = next((j for j in range(i, n) if cow_probabilities[j] >= threshold), None)
        if later is not None: entry_latencies.append(later - i)
    persistence = []
    for i, label in enumerate(target_labels[:n]):
        if not label.entity_left: continue
        duration = 0
        while i + duration < n and cow_probabilities[i + duration] >= threshold:
            duration += 1
        persistence.append(duration)
    return {
        "n_samples": n, "threshold": threshold, "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "cow_precision": tp / (tp + fp) if tp + fp else None,
        "cow_recall": tp / (tp + fn) if tp + fn else None,
        "cow_false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "cow_false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "entity_cell_localization_error": _mean(localization),
        "entity_entry_incorporation_latency": _mean(entry_latencies),
        "entity_disappearance_persistence_duration": _mean(persistence),
        "entity_entries": len(entries), "entity_exits": sum(label.entity_left for label in target_labels[:n]),
    }


def motion_stratified_metrics(samples: Mapping[str, Sequence[Mapping[str, float]]]) -> dict[str, dict[str, Any]]:
    """Aggregate model/baseline errors by event stratum.

    Each sample contains ``model_mse``, ``copy_last_mse``, optional
    ``change_mask_mean``, and optional motion magnitudes.
    """
    result: dict[str, dict[str, Any]] = {}
    for name, rows in samples.items():
        model = _mean(float(row["model_mse"]) for row in rows)
        copy = _mean(float(row["copy_last_mse"]) for row in rows)
        changes = _mean(float(row["change_mask_mean"]) for row in rows if "change_mask_mean" in row)
        motion_error = _mean(abs(float(row["predicted_motion"]) - float(row["target_motion"])) for row in rows if "predicted_motion" in row)
        result[name] = {
            "n_samples": len(rows), "model_mse": model, "copy_last_mse": copy,
            "model_over_copy_last": (model / copy if model is not None and copy and copy > 0 else None),
            "change_mask_mean": changes, "predicted_motion_vs_target_motion": motion_error,
        }
    return result


def flattened_motion_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compatibility-friendly named metrics for dashboards and promotion gates."""
    out: dict[str, Any] = {}
    for stratum in ("static", "moving"):
        values = metrics.get(stratum, {})
        for key in ("model_mse", "copy_last_mse", "model_over_copy_last", "change_mask_mean"):
            out[f"{stratum}_{key}"] = values.get(key)
        if stratum == "moving":
            out["predicted_motion_vs_target_motion"] = values.get("predicted_motion_vs_target_motion")
    return out


def labels_to_json(labels: Sequence[FrameEventLabels]) -> list[dict[str, Any]]:
    return [asdict(label) for label in labels]
