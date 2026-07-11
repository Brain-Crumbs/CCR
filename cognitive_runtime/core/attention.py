"""Deterministic attention controller (issue #59): a budgeted resource-
allocation layer between memory and fusion.

Every cognitive tick, every ``agent_input``-classified stream (issue #32's
:class:`~cognitive_runtime.core.streams.registry.StreamRegistry`
classification -- the same set the "raw input" fusion profile draws from)
gets scored for salience from nothing but its own recent history in the
:class:`~cognitive_runtime.core.streams.temporal_buffer.TemporalBuffer` plus
its :class:`~cognitive_runtime.core.streams.registry.AttentionMetadata`
(relative compute cost). This is deliberately generic: no Program, no
Minecraft-specific stream id, and no hidden per-environment logic -- streams,
modality and registry metadata only, exactly like ``core.streams.fusion``.

A stream's salience score blends:

- ``novelty``    -- did this tick's payload change from the last one?
- ``prediction_error`` -- how sharply is a numeric stream's value moving
  (the trend magnitude over a short window; a stand-in for "how wrong would
  a naive predictor be about this stream right now").
- ``reward_relevance`` -- correlation of the stream's recent numeric values
  with the recent values of the catalog's reward stream(s), so streams that
  move with reward automatically pull focus.
- ``risk``       -- the current ``internal.risk`` stream value (issue #58),
  shared across every stream's signal this tick: a global "state of alert"
  term, not a per-stream one.
- ``recency``    -- a half-life decayed freshness term, same formula
  ``TemporalFusion`` uses for event-modality streams.
- ``boredom``    -- fraction of the recent window that repeats the same
  value; a penalty so a constant signal fades from focus.
- ``compute_cost`` -- the registry's coarse low/medium/high cost, a penalty
  so attention isn't "free" to spend on an expensive stream without payoff.

:class:`AttentionBudget` then forces a choice: only the top streams (by
score) receive nonzero weight, capped at a fixed total.  A
:class:`AttentionController` in ``"off"`` mode instead gives every stream
weight ``1.0`` -- the uniform, "every stream contributes" baseline this
issue's ablation compares against -- so a fusion path that multiplies each
stream's slice by its attention weight reproduces the pre-#59 fused output
exactly whenever attention is off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from cognitive_runtime.core.hashing import hash_payload
from cognitive_runtime.core.streams.events import StreamSpec
from cognitive_runtime.core.streams.registry import StreamRegistry
from cognitive_runtime.core.streams.temporal_buffer import TemporalBuffer

#: The internal.* stream (issue #58) whose value feeds the shared `risk` term.
RISK_STREAM_ID = "internal.risk"

ATTENTION_MODES = frozenset({"off", "budgeted"})

#: Coarse relative-compute-cost -> numeric penalty scale (registry.py's
#: `ATTENTION_COMPUTE_COSTS`).
DEFAULT_COMPUTE_COST_SCALE: Dict[str, float] = {"low": 0.1, "medium": 0.4, "high": 1.0}


def _scalar(payload: Any) -> Optional[float]:
    """A stream payload's numeric reading, generic across the two shapes
    every stream in this codebase uses: a bare number (`body.health`), or a
    dict with a `"value"` key (`internal.*`, `reward.scalar`)."""
    if isinstance(payload, bool):
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, dict):
        value = payload.get("value")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


@dataclass(frozen=True)
class AttentionSignal:
    """One stream's raw salience inputs for one tick."""

    stream_id: str
    novelty: float
    prediction_error: float
    uncertainty: Optional[float]
    reward_relevance: float
    risk: float
    recency: float
    boredom: float
    compute_cost: float


@dataclass(frozen=True)
class AttentionCoefficients:
    """Config-driven weights the controller's score is a linear blend of."""

    novelty: float = 1.0
    prediction_error: float = 1.0
    uncertainty: float = 0.5
    reward_relevance: float = 1.0
    risk: float = 0.5
    recency: float = 0.5
    boredom: float = -0.5
    compute_cost: float = -0.5


@dataclass(frozen=True)
class AttentionBudget:
    """Hard cap on this tick's attention spend: at most `max_streams`
    streams selected, their weights summing to at most `max_total_weight`."""

    max_total_weight: float = 4.0
    max_streams: Optional[int] = 6

    def __post_init__(self) -> None:
        if self.max_total_weight <= 0:
            raise ValueError(f"max_total_weight must be positive, got {self.max_total_weight!r}")
        if self.max_streams is not None and self.max_streams <= 0:
            raise ValueError(f"max_streams must be positive, got {self.max_streams!r}")


@dataclass(frozen=True)
class AttentionConfig:
    coefficients: AttentionCoefficients = field(default_factory=AttentionCoefficients)
    budget: AttentionBudget = field(default_factory=AttentionBudget)
    #: Minimum ticks a captured focus persists before a merely-equal-or-lower
    #: salience challenger can take over (hysteresis/decay against thrash).
    dwell_ticks: int = 5
    #: A challenger displaces the current focus immediately (bypassing
    #: dwell) only when its score exceeds the focus's captured score by more
    #: than this margin -- the "spike captures focus" bottom-up path.
    displacement_margin: float = 0.25
    #: Trailing window (in events) used for trend/novelty/correlation.
    window: int = 16
    half_life_seconds: float = 1.0
    reward_stream_prefix: str = "reward."
    compute_cost_scale: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_COMPUTE_COST_SCALE)
    )


@dataclass(frozen=True)
class AttentionReason:
    """A stream's score and the named contribution of every term -- the
    "reason breakdown" the episode viewer renders."""

    signal: AttentionSignal
    score: float
    components: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 6),
            "components": {k: round(v, 6) for k, v in self.components.items()},
        }


@dataclass(frozen=True)
class AttentionState:
    """One tick's attention allocation: selected streams, their weights, the
    budget spent, and why -- recorded like any other stream/decision data."""

    tick_index: int
    mode: str
    weights: Dict[str, float]
    selected_streams: Tuple[str, ...]
    focus_stream: Optional[str]
    budget_used: float
    budget_total: float
    reasons: Dict[str, AttentionReason]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick_index": self.tick_index,
            "mode": self.mode,
            "weights": {k: round(v, 6) for k, v in self.weights.items()},
            "selected_streams": list(self.selected_streams),
            "focus_stream": self.focus_stream,
            "budget_used": round(self.budget_used, 6),
            "budget_total": round(self.budget_total, 6),
            "reasons": {sid: reason.to_dict() for sid, reason in self.reasons.items()},
        }


def _numeric_series(events, scalar=_scalar) -> List[float]:
    values: List[float] = []
    for event in events:
        value = scalar(event.payload)
        if value is not None:
            values.append(value)
    return values


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    xs = xs[-n:]
    ys = ys[-n:]
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(corr) else float(corr)


class AttentionController:
    """Deterministic weighted-salience scoring under a hard budget.

    Generic: built from a stream catalog + a
    :class:`~cognitive_runtime.core.streams.registry.StreamRegistry` (issue
    #32 metadata) and read out of a plain
    :class:`~cognitive_runtime.core.streams.temporal_buffer.TemporalBuffer`
    -- no Program, no Minecraft import, anywhere in this module.
    """

    def __init__(
        self,
        catalog: Iterable[StreamSpec],
        stream_registry: StreamRegistry,
        mode: str = "budgeted",
        config: Optional[AttentionConfig] = None,
    ) -> None:
        if mode not in ATTENTION_MODES:
            raise ValueError(f"unknown attention mode {mode!r}; expected one of {sorted(ATTENTION_MODES)}")
        self.mode = mode
        self.config = config or AttentionConfig()
        catalog = list(catalog)
        self.stream_ids: Tuple[str, ...] = tuple(
            stream_registry.ids_by_classification(catalog, "agent_input")
        )
        self._compute_cost: Dict[str, float] = {}
        for stream_id in self.stream_ids:
            decl = stream_registry.declaration_for(stream_id)
            metadata = decl.attention if decl is not None else None
            cost_label = metadata.relative_compute_cost if metadata is not None else "low"
            self._compute_cost[stream_id] = self.config.compute_cost_scale.get(cost_label, 0.1)
        self._focus: Optional[str] = None
        self._captured_score: float = 0.0
        self._dwell_remaining: int = 0

    def reset(self) -> None:
        self._focus = None
        self._captured_score = 0.0
        self._dwell_remaining = 0

    # ------------------------------------------------------------- scoring

    def _reference_time(self, buffer: TemporalBuffer) -> float:
        ref = 0.0
        for stream_id in self.stream_ids:
            latest = buffer.latest(stream_id)
            if latest is not None:
                ref = max(ref, latest.timestamp)
        return ref

    def _recency(self, last_ts: float, reference_time: float) -> float:
        dt = max(reference_time - last_ts, 0.0)
        if self.config.half_life_seconds <= 0:
            return 1.0
        return 0.5 ** (dt / self.config.half_life_seconds)

    def _reward_series(self, buffer: TemporalBuffer) -> List[float]:
        reward_stream = next(
            (sid for sid in self.stream_ids if sid.startswith(self.config.reward_stream_prefix)),
            None,
        )
        if reward_stream is None:
            return []
        return _numeric_series(buffer.window(reward_stream, self.config.window))

    def _risk(self, buffer: TemporalBuffer) -> float:
        latest = buffer.latest(RISK_STREAM_ID)
        if latest is None:
            return 0.0
        value = _scalar(latest.payload)
        return value if value is not None else 0.0

    def _signal_for(
        self, stream_id: str, buffer: TemporalBuffer, reference_time: float,
        reward_series: List[float], risk: float,
    ) -> AttentionSignal:
        events = buffer.window(stream_id, self.config.window)
        if not events:
            return AttentionSignal(
                stream_id=stream_id, novelty=0.0, prediction_error=0.0, uncertainty=None,
                reward_relevance=0.0, risk=risk, recency=0.0, boredom=1.0, compute_cost=0.0,
            )
        latest = events[-1]
        recency = self._recency(latest.timestamp, reference_time)
        hashes = [hash_payload(e.payload) for e in events]
        novelty = 1.0 if len(hashes) < 2 or hashes[-1] != hashes[-2] else 0.0
        distinct = len(set(hashes))
        boredom = 1.0 - (distinct / len(hashes))
        values = _numeric_series(events)
        if len(values) >= 2:
            slope = (values[-1] - values[0]) / (len(values) - 1)
            prediction_error = min(abs(slope), 1.0)
        else:
            prediction_error = 0.0
        reward_relevance = abs(_pearson(values, reward_series)) if values and reward_series else 0.0
        return AttentionSignal(
            stream_id=stream_id,
            novelty=novelty,
            prediction_error=prediction_error,
            uncertainty=None,
            reward_relevance=reward_relevance,
            risk=risk,
            recency=recency,
            boredom=boredom,
            compute_cost=self._compute_cost.get(stream_id, 0.1),
        )

    def _score(self, signal: AttentionSignal) -> Tuple[float, Dict[str, float]]:
        c = self.config.coefficients
        components = {
            "novelty": c.novelty * signal.novelty,
            "prediction_error": c.prediction_error * signal.prediction_error,
            "uncertainty": c.uncertainty * (signal.uncertainty or 0.0),
            "reward_relevance": c.reward_relevance * signal.reward_relevance,
            "risk": c.risk * signal.risk,
            "recency": c.recency * signal.recency,
            "boredom": c.boredom * signal.boredom,
            "compute_cost": c.compute_cost * signal.compute_cost,
        }
        return sum(components.values()), components

    # -------------------------------------------------------------- focus

    def _update_focus(self, reasons: Dict[str, AttentionReason]) -> Optional[str]:
        if not reasons:
            self._focus = None
            return None
        best_id = max(reasons, key=lambda sid: reasons[sid].score)
        best_score = reasons[best_id].score
        if self._focus is None or self._focus not in reasons:
            self._capture(best_id, best_score)
            return self._focus
        if best_id == self._focus:
            self._captured_score = best_score
            self._dwell_remaining = self.config.dwell_ticks
            return self._focus
        if best_score > self._captured_score + self.config.displacement_margin:
            self._capture(best_id, best_score)  # bottom-up capture: a real spike wins now
        elif self._dwell_remaining <= 0:
            self._capture(best_id, best_score)  # dwell expired: hand off to the current best
        else:
            self._dwell_remaining -= 1  # hysteresis: hold focus against a marginal challenger
        return self._focus

    def _capture(self, stream_id: str, score: float) -> None:
        self._focus = stream_id
        self._captured_score = score
        self._dwell_remaining = self.config.dwell_ticks

    # ------------------------------------------------------------- budget

    def _apply_budget(
        self, reasons: Dict[str, AttentionReason], focus: Optional[str]
    ) -> Tuple[List[str], Dict[str, float]]:
        budget = self.config.budget
        ranked = sorted(reasons, key=lambda sid: -reasons[sid].score)
        candidates = ranked[: budget.max_streams] if budget.max_streams is not None else list(ranked)
        if focus is not None and focus not in candidates:
            if candidates:
                candidates[-1] = focus
            else:
                candidates = [focus]
        weights = {sid: 0.0 for sid in self.stream_ids}
        scores = {sid: max(reasons[sid].score, 0.0) for sid in candidates}
        total = sum(scores.values())
        if total <= 0.0:
            even = budget.max_total_weight / len(candidates) if candidates else 0.0
            for sid in candidates:
                weights[sid] = even
        else:
            for sid in candidates:
                weights[sid] = (scores[sid] / total) * budget.max_total_weight
        selected = sorted(sid for sid in candidates if weights.get(sid, 0.0) > 0.0)
        return selected, weights

    # ------------------------------------------------------------- public

    def compute(self, tick_index: int, buffer: TemporalBuffer) -> AttentionState:
        """Score every agent-input stream against `buffer` and allocate this
        tick's attention. `"off"` gives every stream uniform weight `1.0` --
        gating multiplied by it is a no-op, byte-identical to no attention."""
        if self.mode == "off":
            return AttentionState(
                tick_index=tick_index,
                mode="off",
                weights={sid: 1.0 for sid in self.stream_ids},
                selected_streams=tuple(self.stream_ids),
                focus_stream=None,
                budget_used=0.0,
                budget_total=0.0,
                reasons={},
            )
        reference_time = self._reference_time(buffer)
        reward_series = self._reward_series(buffer)
        risk = self._risk(buffer)
        reasons: Dict[str, AttentionReason] = {}
        for stream_id in self.stream_ids:
            signal = self._signal_for(stream_id, buffer, reference_time, reward_series, risk)
            score, components = self._score(signal)
            reasons[stream_id] = AttentionReason(signal=signal, score=score, components=components)
        focus = self._update_focus(reasons)
        selected, weights = self._apply_budget(reasons, focus)
        return AttentionState(
            tick_index=tick_index,
            mode="budgeted",
            weights=weights,
            selected_streams=tuple(selected),
            focus_stream=focus,
            budget_used=sum(weights.values()),
            budget_total=self.config.budget.max_total_weight,
            reasons=reasons,
        )
